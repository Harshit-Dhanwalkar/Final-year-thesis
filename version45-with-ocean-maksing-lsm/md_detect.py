# FIX:
# 0.1. Why seperate augmented csv for speed plots? Just merge/append in main detection csv.
# 0.2. Need to replace 'run_all_statistical_analyses', instaed just call individual functions
# 0.3. Exporting 2 seperate CSVs (MD-track-id-coordinates.csv and output_2024/track_id/track_id_detections.csv) which have difference of date format.
# 1. Relative humidity physically out of range (-11, 198)% due to ERA5 artifact from spectral-to-grid interpolation. Do not use raw r values in any threshold logic (e.g. "convection active if r > 70%") without clipping.
# 2. Ocean-land shading to incorporate landfall line
# 3. Inconsistent Parameter Names
#    Location: Multiple functions use buffer_km vs buffer_deg
#    antecedent_analysis.py uses buffer_km=100
#    plot_sm_tp_map() uses buffer_deg=5.0
# 4. Rephase CE 0.5/0.3 vs 8.0/3.0" -> Standardize (use 8-15)

import os
import sys
import gc
import threading
import time
import traceback
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from scipy.optimize import linear_sum_assignment
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from utils.progress_utils import ProgressSpinner, ProgressBar
from utils import (
    std_print,
    is_point_land,
    calculate_track_curvature,
    get_default_buffer_km,
    extract_and_average_variable,
    haversine_distance,
    get_time_coord,
    common_bbox,
    legend_style
)
from speed_analysis import compute_track_speeds

def parse_args():
    parser = argparse.ArgumentParser(description="Detect Monsoon Depressions for given year(s).")
    parser.add_argument("--year", type=int, nargs='+', help="Year(s) to process (e.g., 2010 2011 2012). If not provided, auto‑detect from data.")
    return parser.parse_args()

def detect_year_from_data():
    """Return the most recent year that has data in ../Data/single-levels/ncfiles/"""
    data_root = os.path.join("..", "Data", "single-levels", "ncfiles")
    if not os.path.exists(data_root):
        return None
    years = []
    for item in os.listdir(data_root):
        if item.isdigit() and os.path.isdir(os.path.join(data_root, item)):
            years.append(int(item))
    if years:
        return max(years)
    return None

# Map domain
FULL_EXTENT = [70.0, 95.0, 7.0, 30.0]  # lon_min, lon_max, lat_min, lat_max
# JJAS_MONTHS = [6, 7, 8, 9]

# Detection thresholds
# for wind creterion : https://mausam.imd.gov.in/imd_latest/monsoonreport2024.pdf page: 284
WIND_SPEED_LOWER_BOUND_MS = 5  # m/s, == 17 knots
WIND_SPEED_UPPER_BOUND_MS = 19  # m/s, == 27 knots
MSL_LOWER_BOUND_PA = 100000  # Pa
PRECIPITATION_THRESHOLD_MM = 0.1  # mm/hr
# SWVL1 Range: $0.02-0.55$ m^3/m^3
VORTICITY_THRESHOLD = 3e-6  # 5e-6  # > 0 /s, # Positive cyclonic, Anti-clockwise in North Hemisphere
MAX_SPEED_MS = 12  # m/s,  (36 km/hr Max MD centre translational-speed)
MAX_GAP_HOURS = 24  # hr, Bridge small gaps in Detection
TRACKING_SEARCH_RADIUS_KM = 50  # km, Hourly tracking
LINKING_SEARCH_RADIUS_KM = 100  # km, Multi-hour gaps
MIN_TRACK_DURATION_HOURS = 24
STITCH_DISTANCE_KM = 450  # km, For gap bridging (MD spped 15-25 km/h, 24 hours, the system can easily travel 23x15 = 360 km)
MERGE_DISTANCE_KM = 250  # km, For simultaneous systems, distance to consider two detections as same MD
BUFFER = 10.0  # degrees

# Land-depression mode (env: MD_LAND_MODE=1)
LAND_MODE = os.environ.get("MD_LAND_MODE", "0") == "1"
if LAND_MODE:
    MIN_NET_DISPLACEMENT_KM   = 100   # 250 -> 100
    MAX_GAP_HOURS             = 72    # 24 -> 72 (bridge fragmented lows)
    MAX_CURVATURE             = 4.5   # 3.0 -> 4.5 (allow meandering)
    REQUIRE_OCEAN_FRACTION    = False
    ALLOWED_BEARINGS          = None
else:
    MIN_NET_DISPLACEMENT_KM   = 250
    MAX_GAP_HOURS             = 24
    MAX_CURVATURE             = 3.0
    REQUIRE_OCEAN_FRACTION    = True
    ALLOWED_BEARINGS          = (240, 360, 0, 30)

# Constants
EARTH_R = 6371.0  # km

# Font sizes
FS_TITLE = 14
FS_LABEL = 12
FS_TICK = 10
FS_LEGEND = 10
FS_ANNOT = 9

month_colors = {"June": "#FF6B6B", "July": "#FF4500", "August": "#1E90FF", "September": "#FF00FF",}
month_order = ["June", "July", "August", "September"]

# Data loading
def load_dataset_async(month_dirs, data_dir, result_container):
    file_list = []
    for month_dir in month_dirs:
        month_path = os.path.join(data_dir, month_dir)
        if os.path.isdir(month_path):
            file_list.extend(
                os.path.join(month_path, f)
                for f in os.listdir(month_path)
                if f.endswith(".nc")
            )
    if not file_list:
        result_container["error"] = "No netCDF files found"
        result_container["success"] = False
        return
    try:
        ds = xr.open_mfdataset(
            file_list,
            chunks={},
            combine="by_coords",
            parallel=True,
            coords="minimal",
            data_vars="minimal",
            compat="override",
            engine="h5netcdf",
        )
        # Clip monsoon domain (reduces RAM)
        ds = ds.sel(
            latitude=slice(FULL_EXTENT[3], FULL_EXTENT[2]),
            longitude=slice(FULL_EXTENT[0], FULL_EXTENT[1])
        )
        if "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        ds, time_coord = get_time_coord(ds)
        ds = ds.sortby("time")
        result_container["ds"] = ds
        result_container["success"] = True
    except Exception as e:
        result_container["error"] = str(e)
        result_container["success"] = False


# Temporal helpers
def diagnose_temporal_issues(data_slice):
    """
    Identifies:
    1. Gaps in time series
    2. Duplicate timestamps
    3. Overlapping data from monthly files
    4. Irregular sampling
    """
    # if 'valid_time' in ds.coords:
    #     ds = ds.rename({'valid_time': 'time'})
    data_slice, time_coord = get_time_coord(data_slice)
    times = pd.to_datetime(data_slice[time_coord].values)
    start_time = times[0].strftime("%d/%m/%Y %H:%M")
    end_time = times[-1].strftime("%d/%m/%Y %H:%M")
    total_time = str(times[-1] - times[0])
    total_time_hours = str(int((times[-1] - times[0]).total_seconds() / 3600))

    std_print(None, None, "Total points", len(times))
    std_print(None, None, "Start, End", f'{start_time} to {end_time}')
    std_print(None, None, "Total duration", f"{total_time}  ({total_time_hours} hrs)")

    std_print("INFO", "md", "--- Time step statistics ---")
    time_diffs = (np.diff(times).astype("timedelta64[m]").astype(int) / 60)  # Convert to hours
    mode_diff = np.argmax(np.bincount(time_diffs.astype(int)))
    std_print(None, None, "Mean time interval", f"{np.mean(time_diffs):.2f} hours")
    std_print(None, None, "Median time interval", f"{np.median(time_diffs):.2f} hours")
    std_print(None, None, "Mode time interval", f"{mode_diff} hours")
    std_print(None, None, "Std Dev", f"{np.std(time_diffs):.2f} hours")
    std_print(None, None, "Min / Max time interval", f"{np.min(time_diffs):.2f} / {np.max(time_diffs):.2f} hours")

    # Find irregular gaps
    unique_diffs, counts = np.unique(time_diffs.astype(int), return_counts=True)
    std_print("INFO", "md", "--- Time step distribution ---")
    for diff, count in sorted(zip(unique_diffs, counts), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / len(time_diffs)
        std_print(None, None, "Time intervals", f"{diff:02d} hour for {count} times ({percentage:.1f}%)")

    # # Detect month boundaries
    # std_print("INFO", "md", "--- Month boundary analysis ---")
    # month_changes = []
    # for i in range(1, len(times)):
    #     if times[i].month != times[i - 1].month:
    #         month_changes.append((i, times[i - 1], times[i], time_diffs[i - 1]))
    # if month_changes:
    #     std_print("INFO", "md", f"Found {len(month_changes)} month transitions")
    #     for idx, prev_time, curr_time, gap in month_changes:
    #         prev_time = prev_time.strftime('%d/%m/%Y %H:%M')
    #         curr_time = curr_time.strftime('%d/%m/%Y %H:%M')
    #         std_print(None, None, f"Index {idx}: {prev_time} -> {curr_time} (gap: {gap:.1f} hours)")
    # else:
    #     std_print("INFO", "md", "No month transitions detected", "single continuous month")

    # Detect duplicates
    std_print("INFO", "md", "--- Duplicate/overlap detection ---")
    unique_times, indices = np.unique(times, return_index=True)
    n_duplicates = len(times) - len(unique_times)
    if n_duplicates > 0:
        std_print(None, None, "Found duplicate timestamps", {n_duplicates})
        duplicate_mask = np.zeros(len(times), dtype=bool)
        duplicate_mask[indices] = True
        duplicate_indices = np.where(~duplicate_mask)[0]
        for idx in duplicate_indices[:5]:  # Show first 5
            std_print(None, None, f"Duplicate at index {idx}", times[idx])
    else:
        std_print("INFO", "md", "No duplicates timestamps")

    # Detect negative gaps (time goes backwards)
    std_print("INFO", "md", "--- Time continuity check ---")
    negative_gaps = np.where(time_diffs < 0)[0]
    if len(negative_gaps) > 0:
        std_print(None, None, "Found Time reversals/backward jumps", len(negative_gaps))
        for idx in negative_gaps[:5]:
            std_print(None, None, f"Jump backward at index {idx}", f"{times[idx]} -> {times[idx + 1]} (difference: {time_diffs[idx]:.1f} hours)")
    else:
        std_print("INFO", "md", "Time is monotonically increasing")

    # Detect zero-gaps (identical consecutive times)
    zero_gaps = np.where(time_diffs == 0)[0]
    if len(zero_gaps) > 0:
        std_print("INFO", "md", "Found identical consecutive timestamps", len(zero_gaps))
    if n_duplicates > 0:
        std_print("INFO", "md", "Duplicates/repeated timestamps", n_duplicates)
    if len(negative_gaps) > 0:
        std_print("INFO", "md", "Time reversals/backward jumps", len(negative_gaps))
    # if len(month_changes) > 0 and len(unique_diffs) > 1:
    #     std_print("WARN", "md", "Files likely concatenated",
    #               "Different resolutions at month boundaries"
    #               )
    return {
        "times": times,
        "time_diffs": time_diffs,
        "unique_intervals": unique_diffs,
        "interval_counts": counts,
        "n_duplicates": n_duplicates,
        # "month_changes": month_changes,
        "negative_gaps": negative_gaps,
        "zero_gaps": zero_gaps,
    }


def fix_temporal_resolution(ds, target_resolution_hours=1):
    ds, time_coord = get_time_coord(ds)
    times = pd.to_datetime(ds[time_coord].values)
    std_print("INFO", "md", "Dataset temporal resolution (time points)", len(times))

    # Regular time grid
    new_times = pd.date_range(start=times[0], end=times[-1], freq=f"{target_resolution_hours}h")

    # Reindex dataset to regular grid with interpolation
    for var in ["u10", "v10"]:
        if var in ds.data_vars:
            ds[var] = ds[var].interp({time_coord: new_times}, method="nearest")
    for var in ["tp", "swvl1", "msl", "t2m", "d2m"]:
        if var in ds.data_vars:
            ds[var] = ds[var].interp(
                {time_coord: new_times}, method="linear", kwargs={"fill_value": np.nan}
            )

    # Remove duplicate times if they exist
    if len(times) != len(np.unique(times)):
        std_print("WARN", "md", "Removing duplicate timestamps")
        # Keep first occurrence of each unique time
        unique_times, unique_indices = np.unique(times, return_index=True)
        ds = ds.isel({time_coord: sorted(unique_indices)})
        times = pd.to_datetime(ds[time_coord].values)

    # Add small epsilon to times if too close
    time_diffs = np.diff(times).astype('timedelta64[s]').astype(float)
    if np.any(time_diffs < 1):  # Less than 1 second difference
        std_print("WARN", "md", "Adding small epsilon to near-duplicate times")
        # Add microsecond increments to near-duplicate times
        for i in range(1, len(times)):
            if (times[i] - times[i-1]).total_seconds() < 1:
                times[i] = times[i-1] + pd.Timedelta(microseconds=1)

    ds_fixed = ds.reindex({time_coord: new_times}, method="nearest")
    std_print("INFO", "md", "Resampled to 1-h grid", len(new_times))
    # original_bounds = (ds[time_coord].min(), ds[time_coord].max())
    # out_of_bounds = (new_times < original_bounds[0]) | (new_times > original_bounds[1])
    # std_print('DEBUG', 'md', 'Interpolation points outside original time bounds', f'{out_of_bounds.sum()}')
    return ds_fixed


def validate_temporal_consistency(ds, expected_hours=1, tolerance=0.1):
    ds, time_coord = get_time_coord(ds)
    times = pd.to_datetime(ds[time_coord].values)
    time_diffs = np.diff(times).astype("timedelta64[m]").astype(float) / 60

    min_expected = expected_hours * (1 - tolerance)
    max_expected = expected_hours * (1 + tolerance)
    within_tolerance = (time_diffs >= min_expected) & (time_diffs <= max_expected)
    n_valid = np.sum(within_tolerance)
    n_total = len(time_diffs)
    validity_pct = 100 * n_valid / n_total
    is_valid = validity_pct >= 95  # At least 95% compliance

    report = {
        "is_valid": is_valid,
        "expected_hours": expected_hours,
        "tolerance": tolerance,
        "valid_intervals": n_valid,
        "total_intervals": n_total,
        "validity_percentage": validity_pct,
        "min_interval": np.min(time_diffs),
        "max_interval": np.max(time_diffs),
        "mean_interval": np.mean(time_diffs),
        "violations": np.where(~within_tolerance)[0].tolist(),
    }

    std_print("INFO", "md", "--- Temporal validation ---")
    std_print(None, None, "Expected", f"{expected_hours}±{tolerance * 100:.0f}% hour(s)")
    std_print(None, None, "Compliance", f"{n_valid}/{n_total} intervals ({validity_pct:.1f}%)")
    # std_print(None, None, "Valid", f"{'YES' if is_valid else 'NO'}")
    if not is_valid and len(report["violations"]) > 0:
        std_print(None, None, "Violations at indices", f"{report['violations'][:10]}")
    return is_valid, report


# def resample_dataset_to_resolution(ds, target_hours=1):
#     ds, time_coord = get_time_coord(ds)
#     times = pd.to_datetime(ds[time_coord].values)
#     if target_hours == 1:
#         std_print("INFO", "md", "Dataset at 1-hourly resolution", f"{len(times)} steps")
#         return ds
#     new_times = pd.date_range(start=times[0], end=times[-1], freq=f"{target_hours}h")
#     original_count, new_count = len(times), len(new_times)
#     std_print("INFO", "md", "--- Temporal resampling ---")
#     std_print(None, None, "Input", f"{original_count} time steps (hourly)")
#     std_print(None, None, "Output", f"{new_count} time steps ({target_hours}-hourly)")
#     std_print(None, None, "Reduction", f"{100 * (1 - new_count / original_count):.1f}%")
#
#     # Interpolate all variables to new time grid
#     ds_resampled = ds.interp(
#         {time_coord: new_times},
#         method="linear",
#         kwargs={"fill_value": np.nan},
#         # kwargs={"fill_value": "extrapolate"}
#     )
#     return ds_resampled


def ensure_flux_results_complete(all_flux_results, all_classifications):
    if not all_flux_results or not all_classifications:
        return all_flux_results
    # Lookup dict from classifications
    class_lookup = {c["track_id"]: c for c in all_classifications}
    # Augment flux results with classification data
    for flux_res in all_flux_results:
        track_id = flux_res.get("track_id")
        if track_id and track_id in class_lookup:
            classification = class_lookup[track_id]
            # Only add if not already present
            if "is_long_lived" not in flux_res:
                flux_res["is_long_lived"] = classification.get("is_long_lived", False)
            if "duration_hours" not in flux_res:
                flux_res["duration_hours"] = classification.get("duration_hours", np.nan)
            if "category" not in flux_res:
                flux_res["category"] = classification.get("category", "Unknown")
    return all_flux_results


# Detection criteria helpers
def has_closed_contour(data_array, lat, lon):
    lat_min, lat_max = lat - BUFFER, lat + BUFFER
    lon_min, lon_max = lon - BUFFER, lon + BUFFER
    # Check bounds
    if (
        lat_min < data_array.latitude.min()
        or lat_max > data_array.latitude.max()
        or lon_min < data_array.longitude.min()
        or lon_max > data_array.longitude.max()
    ):
        return False
    # Ensure BUFFER region is within the dataset bounds
    if not (
        data_array.latitude.min() <= lat_min
        and data_array.latitude.max() >= lat_max
        and data_array.longitude.min() <= lon_min
        and data_array.longitude.max() >= lon_max
    ):
        return False
    # Extract subset of MSL data
    subset = data_array.sel(
        latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
    )
    if subset.size == 0:
        return False

    min_val = subset.min().item()
    boundary_lats = [subset.latitude.min().item(), subset.latitude.max().item()]
    boundary_lons = [subset.longitude.min().item(), subset.longitude.max().item()]
    left_boundary = subset.sel(longitude=boundary_lons[0], method="nearest")
    right_boundary = subset.sel(longitude=boundary_lons[1], method="nearest")
    top_boundary = subset.sel(latitude=boundary_lats[0], method="nearest")
    bottom_boundary = subset.sel(latitude=boundary_lats[1], method="nearest")

    if (
        np.all(left_boundary > min_val)
        and np.all(right_boundary > min_val)
        and np.all(top_boundary > min_val)
        and np.all(bottom_boundary > min_val)
    ):
        return True

    # Find the nearest grid point to the detected (lat, lon)
    nearest_subset = data_array.sel(latitude=lat, longitude=lon, method="nearest")
    central_pressure = nearest_subset.item()
    min_val_subset = subset.min().item()

    if np.isclose(central_pressure, min_val_subset, rtol=0.01):
        # Check if the boundary pressure is higher than the center pressure
        if subset.latitude.size > 1 and subset.longitude.size > 1:
            boundary_pressure = np.nanmean(
                [
                    subset.isel(latitude=0).values.mean(),  # North edge
                    subset.isel(latitude=-1).values.mean(),  # South edge
                    subset.isel(longitude=0).values.mean(),  # West edge
                    subset.isel(longitude=-1).values.mean(),  # East edge
                ]
            )
            if boundary_pressure > central_pressure + 20:
                return True
    return False


def check_wind_criteria(u_data, v_data, lat, lon):
    lat_min, lat_max = lat - BUFFER, lat + BUFFER
    lon_min, lon_max = lon - BUFFER, lon + BUFFER

    if not (
        u_data.latitude.min() <= lat_min
        and u_data.latitude.max() >= lat_max
        and u_data.longitude.min() <= lon_min
        and u_data.longitude.max() >= lon_max
    ):
        return False
    u_subset = u_data.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    v_subset = v_data.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    wind_speed = np.sqrt(u_subset**2 + v_subset**2)
    max_wind_speed = wind_speed.max().item()

    # Check if max wind speed is within the MD speed range
    if (
        max_wind_speed >= WIND_SPEED_LOWER_BOUND_MS
        and max_wind_speed <= WIND_SPEED_UPPER_BOUND_MS
    ):
        return True
    # Secondary check: exclude weak systems
    if wind_speed.mean().item() < 5:
        return False
    return True


def calculate_vorticity(ds, latime_index, lon_idx):
    """
    Compute relative vorticity using finite differences.
    This uses local grid points (not spatial averaging) because vorticity
    calculation requires spatial gradients (dv/dx - du/dy).
    The local wind field structure is essential for vorticity calculation.
    """
    # Get grid dimensions
    n_lat, n_lon = len(ds.latitude), len(ds.longitude)
    # Get u, v at the point
    # u = float(ds["u10"].isel(latitude=latime_index, longitude=lon_idx).values)
    # v = float(ds["v10"].isel(latitude=latime_index, longitude=lon_idx).values)

    # Get grid spacing
    lat = float(ds.latitude.values[latime_index])
    dlat= np.abs(ds.latitude.values[1] - ds.latitude.values[0])
    dlon = np.abs(ds.longitude.values[1] - ds.longitude.values[0])

    # Convert to meters
    dy = EARTH_R * np.deg2rad(dlat)
    dx = EARTH_R * np.cos(np.deg2rad(lat)) * np.deg2rad(dlon)

    # Get neighboring points with boundary checks
    # For latitude (north-south)
    if latime_index == 0:  # Southern boundary ( Using forward difference)
        u_north = float(ds["u10"].isel(latitude=latime_index + 1, longitude=lon_idx).values)
        u_south = float(ds["u10"].isel(latitude=latime_index, longitude=lon_idx).values)
        du_dy = (u_north - u_south) / dy
    elif latime_index == n_lat - 1:  # Northern boundary
        u_north = float(ds["u10"].isel(latitude=latime_index, longitude=lon_idx).values)
        u_south = float(ds["u10"].isel(latitude=latime_index - 1, longitude=lon_idx).values)
        du_dy = (u_north - u_south) / dy
    else:  # Interior points
        u_north = float(ds["u10"].isel(latitude=latime_index + 1, longitude=lon_idx).values)
        u_south = float(ds["u10"].isel(latitude=latime_index - 1, longitude=lon_idx).values)
        du_dy = (u_north - u_south) / (2 * dy)
    # For longitude (east-west)
    if lon_idx == 0:  # Western boundary
        v_east = float(ds["v10"].isel(latitude=latime_index, longitude=lon_idx + 1).values)
        v_west = float(ds["v10"].isel(latitude=latime_index, longitude=lon_idx).values)
        dv_dx = (v_east - v_west) / dx
    elif lon_idx == n_lon - 1:  # Eastern boundary
        v_east = float(ds["v10"].isel(latitude=latime_index, longitude=lon_idx).values)
        v_west = float(ds["v10"].isel(latitude=latime_index, longitude=lon_idx - 1).values)
        dv_dx = (v_east - v_west) / dx
    else:  # Interior points
        v_east = float(ds["v10"].isel(latitude=latime_index, longitude=lon_idx + 1).values)
        v_west = float(ds["v10"].isel(latitude=latime_index, longitude=lon_idx - 1).values)
        dv_dx = (v_east - v_west) / (2 * dx)
    # Relative vorticity (ζ = ∂v/∂x - ∂u/∂y)
    vorticity = dv_dx - du_dy
    return vorticity


def check_vorticity_criteria(data_slice, lat, lon):
    try:
        data_slice, time_coord = get_time_coord(data_slice)
        time_step = data_slice["time"].values if "time" in data_slice.coords else None
        # Get u, v at center point
        u_center = extract_and_average_variable(
            data_slice,
            lat,
            lon,
            time_step,
            time_coord,
            var_name = "u10",
            buffer_km = get_default_buffer_km("u10"),
            aggregation = "mean",
        )
        v_center = extract_and_average_variable(
            data_slice,
            lat,
            lon,
            time_step,
            time_coord,
            var_name = "v10",
            buffer_km = get_default_buffer_km("v10"), 
            aggregation = "mean",
        )
        if u_center is None or v_center is None or np.isnan(u_center) or np.isnan(v_center):
            return False, np.nan
        # if np.isnan(u_center) or np.isnan(v_center):
        #     u_center, v_center = np.nan, np.nan
        #     return False, np.nan

        # Get latitude and longitude arrays
        lats = data_slice.latitude.values
        lons = data_slice.longitude.values
        # Find indices nearest to the center
        latime_index = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        # Check at the boundaries
        if (
            latime_index == 0
            or latime_index == len(lats) - 1
            or lon_idx == 0
            or lon_idx == len(lons) - 1
        ):
            return False, np.nan

        # Get grid spacing in radians
        dlat = float(lats[1] - lats[0])
        dlon = float(lons[1] - lons[0])
        dlat_rad = np.deg2rad(dlat)
        dlon_rad = np.deg2rad(dlon)

        # Adjust dlon for latitude
        lat_rad = np.deg2rad(float(lat))
        dlon_rad_adjusted = dlon_rad * np.cos(lat_rad)

        # Get neighboring u, v values
        try:
            u_north = float(data_slice["u10"].isel(latitude=latime_index + 1, longitude=lon_idx).values)
            u_south = float(data_slice["u10"].isel(latitude=latime_index - 1, longitude=lon_idx).values)
            v_east = float(data_slice["v10"].isel(latitude=latime_index, longitude=lon_idx + 1).values)
            v_west = float(data_slice["v10"].isel(latitude=latime_index, longitude=lon_idx - 1).values)
        except Exception as e:
            std_print("ERROR", "md", "Error extracting neighboring values", str(e))
            return False, np.nan

        # Compute finite differences ( m/s per meter)
        du_dy = (u_north - u_south) / (2 * EARTH_R * dlat_rad)
        dv_dx = (v_east - v_west) / (2 * EARTH_R * dlon_rad_adjusted)

        # Compute relative vorticity (dv/dx - du/dy)
        vorticity = dv_dx - du_dy  # s^-1

        # Check if vorticity is positive (cyclonic)
        if vorticity > VORTICITY_THRESHOLD:
            return True, vorticity
        else:
            return False, vorticity
    except Exception as e:
        std_print("ERROR", "md", "Computing vorticity", str(e))
        traceback.print_exc()
        return False, np.nan


def check_moisture_criteria(data_slice, lat, lon):
    try:
        # Get time coordinate
        data_slice, time_coord = get_time_coord(data_slice)
        # Handle time coordinate
        if hasattr(data_slice, 'coords') and "time" in data_slice.coords:
            time = data_slice["time"].values
        else:
            time = None
        tpearson_p = extract_and_average_variable(
            data_slice,
            lat,
            lon,
            time,
            time_coord,
            var_name="tp",
            buffer_km=get_default_buffer_km("tp"),
            aggregation="mean",  # or 'sum' if cumulative
        )
        if not np.isnan(tpearson_p):
            precipitation_ok = (tpearson_p * 1000) >= PRECIPITATION_THRESHOLD_MM  # m/hr -> mm/hr
            return precipitation_ok, precipitation_ok # bool, value
        else:
            return False, np.nan
    except Exception as e:
        std_print("ERROR", "md", "Failed moisture criteria", str(e))
        return False, np.nan


def compute_surface_heat_fluxes(data_slice, lat, lon):
    try:
        # Get time coordinate
        data_slice, time_coord = get_time_coord(data_slice)
        # Handle time coordinate
        if hasattr(data_slice, 'coords') and "time" in data_slice.coords:
            time_step = data_slice["time"].values
        else:
            time_step = None
        slhtf = extract_and_average_variable(
            data_slice,
            lat,
            lon,
            time_step,
            time_coord,
            var_name="avg_slhtf",
            buffer_km=get_default_buffer_km("lhf"),
            aggregation="mean",
        )
        ishf = extract_and_average_variable(
            data_slice,
            lat,
            lon,
            time_step,
            time_coord,
            var_name="avg_ishf",
            buffer_km=get_default_buffer_km("lhf"),
            aggregation="mean",
        )
        total_hf = slhtf + ishf if not (np.isnan(slhtf) or np.isnan(ishf)) else np.nan
        return {"latent_flux": slhtf, "sensible_flux": ishf, "total_flux": total_hf}
    except Exception as e:
        std_print("ERROR", "md", "NaN values for all fluxes", str(e))
        return {"latent_flux": np.nan, "sensible_flux": np.nan, "total_flux": np.nan}


def compute_asm_for_track(data_slice, track_df, hours_before=24):
    try:
        # Get time coordinate
        data_slice, time_coord = get_time_coord(data_slice)
        # if hasattr(data_slice, 'coords') and "time" in data_slice.coords:
        #     time = data_slice["time"].values
        # else:
        #     time = None
        start_time = pd.to_datetime(track_df["time"].iloc[0])
        end_time = start_time
        start_window = start_time - pd.Timedelta(hours=hours_before)

        # Extract soil moisture data for current location before track start
        avg_lat = track_df["latitude"].iloc[:5].mean()
        avg_lon = track_df["longitude"].iloc[:5].mean()

        # Extract data
        spatial_point = ds.sel(latitude=avg_lat, longitude=avg_lon, method="nearest")
        std_print("DEBUG", "md", "Memory usage",  f"{spatial_point.nbytes / 1e6:.2f} MB")
        # FIX: use extract_and_average_variable or extract_track_variable for swvl1
        subset = spatial_point.sel({time_coord: slice(start_window, end_time)})
        if "swvl1" in subset.data_vars:
            asm_value = float(subset["swvl1"].mean().values)
            return asm_value
        else:
            return np.nan
    except Exception as e:
        std_print("ERROR", "md", "Computing ASM", str(e))
        return np.nan


# Centre detection
def find_low_pressure_center(
    data_slice, time_step, prev_center,
    search_radius_km=TRACKING_SEARCH_RADIUS_KM,
    smoothing_radius_km=100,
):
    required_vars = ["msl", "u10", "v10", "tp"]
    if not all(var in data_slice.data_vars for var in required_vars):
        return None
    try:
        ds_local, time_coord_local = get_time_coord(data_slice)
        ds_local = ds_local.sel(**{time_coord_local: time_step}, method="nearest")
        msl_data = ds_local["msl"]
        u_data = ds_local["u10"]
        v_data = ds_local["v10"]
        # Find all low pressure candidates
        msl_values = msl_data.values.flatten()
        msl_shape = msl_data.shape
        # std_print("DEBUG", "md", "MSL data shape", msl_shape)

        # Get top N lowest pressure points
        n_candidates = 6 # N
        if len(msl_values) < n_candidates:
            n_candidates = len(msl_values)
        if n_candidates == 0:
            return None

        flat_indices = np.argsort(msl_values)[:n_candidates]
        candidates = []
        for idx in flat_indices:
            latime_index, lon_idx = np.unravel_index(idx, msl_shape)
            lat = float(msl_data.latitude.values[latime_index])
            lon = float(msl_data.longitude.values[lon_idx])
            pressure = float(msl_values[idx])
            # Check if pressure is reasonable candidate
            if pressure > MSL_LOWER_BOUND_PA:
                continue

            # Calculate distance from previous center if available
            distance = 0
            if prev_center is not None and isinstance(prev_center, (tuple, list)) and len(prev_center) == 2:
                prev_lat, prev_lon = prev_center
                try:
                    distance = haversine_distance(prev_lon, prev_lat, lon, lat)
                    if distance > search_radius_km:
                        continue
                except Exception as e :
                    std_print("ERROR", "md", "Error", str(e))
                    pass
            candidates.append({
                "lat": lat,
                "lon": lon,
                "pressure": pressure,
                "distance": distance
            })

        # Sort candidates by pressure (lowest first)
        candidates.sort(key=lambda x: x["pressure"])
        for candidate in candidates:
            lat_center = candidate["lat"]
            lon_center = candidate["lon"]
            min_msl_val = candidate["pressure"]

            # Relaxed pressure criteria check
            pressure_ok = min_msl_val <= MSL_LOWER_BOUND_PA

            # Closed contour check
            if msl_data is not None:
                closed_contour = has_closed_contour(msl_data, lat_center, lon_center)
            else:
                closed_contour = False

            # Wind criteria check
            if u_data is not None and v_data is not None:
                wind_ok = check_wind_criteria(u_data, v_data, lat_center, lon_center)
            else:
                wind_ok = False

            # Moisture criteria check
            if ds_local is not None:
                moisture_ok, moisture_value = check_moisture_criteria(ds_local, lat_center, lon_center)
            else:
                moisture_ok, moisture_value = False, np.nan

            # Vorticity criteria check
            if ds_local is not None:
                vorticity_ok, vorticity_value = check_vorticity_criteria(ds_local, lat_center, lon_center)
            else:
                vorticity_ok , vorticity_value = False, np.nan

            score = (
                int(pressure_ok)
                + int(closed_contour)
                + int(wind_ok)
                + int(moisture_ok)
                + int(vorticity_ok)
            )

            if score >= 2:  # Accept if at least 2 criteria pass
                return (lat_center, lon_center)
        return None # No candidate met the criteria
    except Exception as e:
        time_step = pd.to_datetime(time_step).strftime("%d/%m/%Y %H:%M")
        std_print("ERROR", "md", f"Finding low pressure center at {time_step}", str(e))
        return None


# Tracking, stitching, consolidation, split mixed,  smoothing, filtering, 
def track_monsoon_depressions(detections, search_radius_km, max_speed_ms, max_time_gap_hours):
    if detections.empty:
        return []
    detections = detections.sort_values("time").reset_index(drop=True)
    detections["time"] = pd.to_datetime(detections["time"])
    detections["hour"] = detections["time"].dt.floor("h")

    # Group by time
    time_groups = detections.groupby("hour")
    time_steps = sorted(time_groups.groups.keys())
    std_print("DEBUG", "md", "Hours with detections", len(time_steps))

    if len(time_steps) > 1:
        time_diffs = []
        for i in range(1, len(time_steps)):
            diff = (time_steps[i] - time_steps[i - 1]).total_seconds() / 3600
            time_diffs.append(diff)
        time_diffs_mean = np.mean(time_diffs)
        time_diffs_min = np.min(time_diffs)
        time_diffs_max = np.max(time_diffs)
        std_print("INFO", "md", "--- Time gaps between detections ---")
        std_print(None, None, "Mean time time gaps", f"{time_diffs_mean:.2f} hour(s)")
        std_print(None, None, "Min / Max time gaps", f"{time_diffs_min:.2f} / {time_diffs_max:.2f} hour(s)")
        std_print(None, None, "Std Dev", f"{np.std(time_diffs):.2f} hour(s)")
        # Show distribution of gaps
        unique_gaps, gap_counts = np.unique(np.round(time_diffs), return_counts=True)
        for gap, count in sorted(zip(unique_gaps, gap_counts)):
            std_print(None, None, "Hour gaps of length", f"{gap:02.0f} occur {count} times")

    tracks = []
    if len(time_steps) < 2:
        std_print("WARN", "md", "Not enough time steps for tracking")
        return tracks

    active_tracks = {}
    first_detections = time_groups.get_group(time_steps[0])
    for i, (_, row) in enumerate(first_detections.iterrows()):
        active_tracks[i] = [row]

    std_print("INFO", "md", "Initial active tracks", f"{len(active_tracks)}")
    for time_index in range(1, len(time_steps)):
        curr_time_step = time_steps[time_index]
        prev_time_step = time_steps[time_index - 1]
        time_diff_step = (curr_time_step - prev_time_step).total_seconds() / 3600
        curr_time_str = pd.to_datetime(curr_time_step).strftime("%d/%m/%Y %H:%M")
        prev_time_str = pd.to_datetime(prev_time_step).strftime("%d/%m/%Y %H:%M")
        time_diff_str = pd.to_datetime(time_diff_step).strftime("%d/%m/%Y %H:%M")
        current_detections = time_groups.get_group(curr_time_step)

        # Only proceed if time gap is reasonable
        if time_diff_step> max_time_gap_hours:
            std_print(None, None, "Large time gap (hours)", f"{time_diff_str}  (from {prev_time_str} to {curr_time_str})")
            # End all current tracks
            for track_id, points in list(active_tracks.items()):
                if len(points) >= 3:
                    tracks.append(pd.DataFrame(points))
            active_tracks = {}
            # Start new tracks from current detections
            for i, (_, row) in enumerate(current_detections.iterrows()):
                active_tracks[i] = [row]
            continue

        # Cost matrix for assignment
        n_tracks = len(active_tracks)
        n_detections = len(current_detections)
        if n_detections == 0:
            continue

        cost_matrix = np.full((n_tracks, n_detections), np.inf)
        # Calculate costs for each track-detection pair
        for i, (track_id, points) in enumerate(active_tracks.items()):
            last_point = points[-1]
            for j, (_, det) in enumerate(current_detections.iterrows()):
                # Calculate distance
                distance = haversine_distance(
                    last_point["longitude"],
                    last_point["latitude"],
                    det["longitude"],
                    det["latitude"],
                )
                # Check if within search radius
                if distance > search_radius_km:
                    continue
                # Calculate MD centre speed
                speed_ms = (distance * 1000) / (time_diff_step * 3600)
                if speed_ms > max_speed_ms:
                    continue
                # Cost: distance (prioritize closer matches)
                cost_matrix[i, j] = distance

        # Solve assignment problem
        if np.isfinite(cost_matrix).any():
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
            # Update active tracks with assignments
            new_active_tracks = {}
            assigned_detections = set()
            for time_index_m, direction_index_m in zip(track_indices, det_indices):
                cost = cost_matrix[time_index_m, direction_index_m]
                if cost < np.inf:
                    track_id = list(active_tracks.keys())[time_index_m]
                    det_row = current_detections.iloc[direction_index_m]
                    active_tracks[track_id].append(det_row)
                    new_active_tracks[track_id] = active_tracks[track_id]
                    assigned_detections.add(direction_index_m)

            # End unassigned tracks (minimum length check)
            for track_id, points in active_tracks.items():
                if track_id not in new_active_tracks:
                    if len(points) >= 3:
                        tracks.append(pd.DataFrame(points))

            # Start new tracks from unassigned detections
            for j in range(n_detections):
                if j not in assigned_detections:
                    new_id = (max(new_active_tracks.keys()) + 1 if new_active_tracks else 0)
                    new_active_tracks[new_id] = [current_detections.iloc[j]]
            active_tracks = new_active_tracks
        else:
            # No valid assignments
            for track_id, points in active_tracks.items():
                if len(points) >= 3:
                    tracks.append(pd.DataFrame(points))
            # Start new tracks from current detections
            active_tracks = {}
            for i, (_, row) in enumerate(current_detections.iterrows()):
                active_tracks[i] = [row]

    # End remaining tracks
    for points in active_tracks.values():
        if len(points) >= 3:
            tracks.append(pd.DataFrame(points))
    std_print("INFO", "md", "Tracks found", len(tracks))
    return tracks


def stitch_sequential_tracks(tracks, max_time_gap_hours, dist_km):
    if not tracks:
        return []
    # Sort tracks by start time
    tracks = sorted(tracks, key=lambda x: x["time"].iloc[0])
    stitched = []
    used = set()

    for i in range(len(tracks)):
        if i in used:
            continue
        current_track = tracks[i].copy()
        used.add(i)
        any_joined = True
        while any_joined:
            any_joined = False
            last_time = pd.to_datetime(current_track["time"].iloc[-1])
            last_lon = current_track["longitude"].iloc[-1]
            last_lat = current_track["latitude"].iloc[-1]
            for j in range(len(tracks)):
                if j in used:
                    continue
                next_track = tracks[j]
                start_time = pd.to_datetime(next_track["time"].iloc[0])
                # Check time gap
                hour_gap = (start_time - last_time).total_seconds() / 3600
                if 0 < hour_gap <= max_time_gap_hours:
                    # Check distance gap
                    gap_distance = haversine_distance(
                        last_lon,
                        last_lat,
                        next_track["longitude"].iloc[0],
                        next_track["latitude"].iloc[0],
                    )
                    if gap_distance < dist_km:
                        current_track = (
                            pd.concat([current_track, next_track])
                            .drop_duplicates()
                            .sort_values("time")
                        )
                        used.add(j)
                        any_joined = True
                        break  # Start looking again from the new end of current_track
        stitched.append(current_track)
    return stitched


def consolidate_simultaneous_tracks(tracks, dist_km=350, time_tolerance_hours=1):
    if not tracks:
        return []
    final_tracks = []
    used_indices = set()
    for i in range(len(tracks)):
        if i in used_indices:
            continue
        base = tracks[i].copy()
        base["time"] = pd.to_datetime(base["time"]).dt.tz_localize(None)

        used_indices.add(i)
        for j in range(i + 1, len(tracks)):
            if j in used_indices:
                continue
            other = tracks[j]

            # Find time overlap (with tolerance)
            base_times = sorted(list(set(pd.to_datetime(base["time"]).dt.floor("h"))))
            other_times = sorted(list(set(pd.to_datetime(other["time"]).dt.floor("h"))))

            # Check if time windows overlap
            base_time_range = (base_times[0], base_times[-1])
            other_time_range = (other_times[0], other_times[-1])
            time_overlap = (base_time_range[0] <= other_time_range[1]) and (
                other_time_range[0] <= base_time_range[1]
            )
            if time_overlap:
                # For overlapping times, check spatial distance
                overlap_times = set(base_times).intersection(set(other_times))
                if overlap_times:
                    overlap_dists = []
                    for t in list(overlap_times)[:10]:  # Sample check
                        base_subset = base[
                            pd.to_datetime(base["time"]).dt.floor("h") == t
                        ]
                        other_subset = other[
                            pd.to_datetime(other["time"]).dt.floor("h") == t
                        ]
                        if not base_subset.empty and not other_subset.empty:
                            p1, p2 = base_subset.iloc[0], other_subset.iloc[0]
                            overlap_distance = haversine_distance(
                                p1["longitude"],
                                p1["latitude"],
                                p2["longitude"],
                                p2["latitude"],
                            )
                            overlap_dists.append(overlap_distance)
                    if overlap_dists and np.mean(overlap_dists) < dist_km:
                        # Merge: keep lower pressure at each time
                        combined = pd.concat([base, other])
                        base = (
                            combined.sort_values(["time", "msl"])
                            .drop_duplicates("time", keep="first")
                            .sort_values("time")
                        )
                        used_indices.add(j)
        final_tracks.append(base)
    return final_tracks


def split_mixed_tracks(tracks):
    """Split tracks that contain both ocean-originated MDs and inland disturbances"""
    split_tracks = []
    for track in tracks:
        if len(track) < 10:  # Too short to split
            split_tracks.append(track)
            continue

        # Check if track has both ocean and land phases
        if "is_land" not in track.columns:
            track["is_land"] = track.apply(
                lambda row: is_point_land(row["latitude"], row["longitude"]), axis=1
            )

        # Find longest continuous ocean segment
        ocean_segments = []
        current_segment = []
        in_ocean = False
        for i, row in track.iterrows():
            if not row["is_land"]:  # Ocean point
                if not in_ocean:
                    # Start new ocean segment
                    current_segment = [row]
                    in_ocean = True
                else:
                    # Continue ocean segment
                    current_segment.append(row)
            else:  # Land point
                if in_ocean and len(current_segment) >= 5:
                    # End ocean segment
                    ocean_segments.append(pd.DataFrame(current_segment))
                in_ocean = False
                current_segment = []

        # Check last segment
        if in_ocean and len(current_segment) >= 5:
            ocean_segments.append(pd.DataFrame(current_segment))

        # If multiple distinct ocean segments, keep the earliest/longest
        if len(ocean_segments) >= 2:
            # Find longest ocean segment
            longest_segment = max(ocean_segments, key=lambda df: len(df))
            # Check if it's a proper MD (starts over ocean, moves WNW)
            if len(longest_segment) >= 12:  # At least 12-hour duration
                bearing = calculate_track_direction(longest_segment)
                if bearing and (240 <= bearing <= 360 or 0 <= bearing <= 30):
                    split_tracks.append(longest_segment)
                    std_print("INFO", "md", "Split mixed track, Kept ocean segment", f"{len(longest_segment)} points")
                else:
                    split_tracks.append(track)  # Keep original
            else:
                split_tracks.append(track)  # Keep original
        else:
            split_tracks.append(track)  # Keep original
    return split_tracks


def smooth_track_coordinates(track_df, window_size=5, iterations=2):
    if len(track_df) < window_size:
        std_print("ERROR", "md", f"Insufficient Track points (<{window_size})", len(track_df))
        return track_df
    track_df = track_df.copy()

    # Check if required columns exist
    if "latitude" not in track_df.columns or "longitude" not in track_df.columns:
        std_print("ERROR", "md", "track_df missing 'latitude' and/or 'longitude' columns")
        return track_df

    for _ in range(iterations):
        # Moving average
        lat_smooth, lon_smooth = (
            track_df["latitude"].rolling(window=window_size, center=True, min_periods=1).mean(),
            track_df["longitude"].rolling(window=window_size, center=True, min_periods=1).mean(),
        )
        # Apply smoothed values
        track_df["latitude"], track_df["longitude"] = lat_smooth, lon_smooth
    return track_df


def calculate_track_direction(track_data):
    if len(track_data) < 2:
        return None
    start_lat = track_data.iloc[0]["latitude"]
    start_lon = track_data.iloc[0]["longitude"]
    end_lat = track_data.iloc[-1]["latitude"]
    end_lon = track_data.iloc[-1]["longitude"]

    lat1 = np.radians(start_lat)
    lon1 = np.radians(start_lon)
    lat2 = np.radians(end_lat)
    lon2 = np.radians(end_lon)
    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360  # Normalize to 0-360
    return bearing


def filter_valid_tracks(tracks, min_duration, min_distance):
    """Filter tracks by duration, displacement, and direction"""
    valid_tracks = []
    fail_reasons = {"duration": 0, "distance": 0, "direction": 0, "intensity": 0, "curvature": 0}

    tracks_to_process = tracks.values() if isinstance(tracks, dict) else tracks
    for idx, track in enumerate(tracks_to_process):
        track_num = idx + 1
        duration = (track["time"].max() - track["time"].min()).total_seconds() / 3600
        total_dist = haversine_distance(
            track.iloc[0]["longitude"], track.iloc[0]["latitude"],
            track.iloc[-1]["longitude"], track.iloc[-1]["latitude"],
        )
        bearing = calculate_track_direction(track)
        t_start = pd.to_datetime(track["time"].min()).strftime("%d/%m %H:%M")
        t_end   = pd.to_datetime(track["time"].max()).strftime("%d/%m %H:%M")
        std_print(
            "DEBUG", "md",
            f"Track {track_num:02d} [{t_start}->{t_end}], pts={len(track):3d}, dur={duration:5.1f}h,"
            f" dist={total_dist:6.1f}km, bear={f'{bearing:.0f}°' if bearing is not None else 'N/A':>6s},"
            f" ({track.iloc[0]['latitude']:.1f}°N,{track.iloc[0]['longitude']:.1f}°E)"
            f"->({track.iloc[-1]['latitude']:.1f}°N,{track.iloc[-1]['longitude']:.1f}°E),"
        )

        # 1. Duration check
        if duration < min_duration:
            std_print("DEBUG", "md", f"  FAIL duration : {duration:.1f}h < {min_duration}h")
            fail_reasons["duration"] += 1
            continue

        # 2. Distance check (net displacement)
        if total_dist < min_distance:
            std_print("DEBUG", "md", f"  FAIL distance : {total_dist:.1f}km < {min_distance}km")
            fail_reasons["distance"] += 1
            continue

        # 3. Direction check (WNW preferred for MDs)
        if ALLOWED_BEARINGS is not None and bearing is not None:
            lo1, hi1, lo2, hi2 = ALLOWED_BEARINGS
            is_valid_md_direction = (lo1 <= bearing <= hi1) or (lo2 <= bearing <= hi2)
            if not is_valid_md_direction:
                std_print("DEBUG", "md", f"  FAIL direction : {bearing:.0f}° not in WNW range")
                fail_reasons["direction"] += 1
                continue

        # 4. Intensity check - minimum pressure variation
        if "msl" in track.columns:
            pressure_drop = track["msl"].max() - track["msl"].min()
            if pressure_drop < 200:  # < 2 hPa
                std_print("DEBUG", "md", f"  FAIL intensity : pressure range {pressure_drop:.1f} Pa < 200 Pa")
                fail_reasons["intensity"] += 1
                continue

        # 5. Require significant ocean portion
        if REQUIRE_OCEAN_FRACTION and "is_land" in track.columns:
            ocean_fraction = 1 - (track["is_land"].sum() / len(track))
            if ocean_fraction < 0.3:
                std_print("DEBUG", "md", f"  FAIL ocean fraction : {ocean_fraction:.2f} < 0.30")
                fail_reasons["origin"] = fail_reasons.get("origin", 0) + 1
                continue

        # 6. Curvature check (avoid stationary/recurving systems)
        curvature = calculate_track_curvature(track)
        if curvature > MAX_CURVATURE:
            std_print("DEBUG", "md", f"  FAIL curvature : {curvature:.2f} > 3.0")
            fail_reasons["curvature"] += 1
            continue

        valid_tracks.append(track)

    # Summary
    std_print("INFO", "md", "Track filtering statistics")
    for reason, count in fail_reasons.items():
        if count > 0:
            std_print(None, None, f"No. of Tracks failed {reason} threshold", f"{count}")
    return valid_tracks



# Main detection process
def process_files(ds):
    try:
        ds, time_coord = get_time_coord(ds)
        times = pd.to_datetime(ds[time_coord].values)
        time_diffs = np.diff(times).astype("timedelta64[s]").astype(float) / 3600

        std_print("INFO", "md", "--- Temporal check ---")
        # Verify dataset has regular spacing
        if np.std(time_diffs) > 0.1:
            std_print("WARN", "md", "Irregular intervals detected")
            ds = fix_temporal_resolution(ds, target_resolution_hours=1)
            times = pd.to_datetime(ds[time_coord].values)
            time_diffs = np.diff(times).astype("timedelta64[s]").astype(float) / 3600
            std_print(None, None, "After fix Mean interval (hours)", f"{np.mean(time_diffs):.2f}")

        required_vars = ["msl", "u10", "v10", "tp"]
        if not all(var in ds.data_vars for var in required_vars):
            std_print("ERROR", "md", "Missing required variables", required_vars)
            return None

        times = ds[time_coord].values
        start_time = pd.to_datetime(times[0]).strftime("%d/%m/%Y %H:%M")
        end_time = pd.to_datetime(times[-1]).strftime("%d/%m/%Y %H:%M")
        std_print(None, None, "Time range", f"{start_time} to {end_time}")
        # std_print("DEBUG", "md", "Number of time steps after verification", len(times))

        std_print("INFO", "md", "--- Detection progress ---")
        # Container for detections
        md_detections = []
        total_steps = len(times)
        progress_bar = ProgressBar(total_steps, "Detection Progress")
        prev_center = None
        detection_count = 0
        for i, time_step in enumerate(times):
            if i % 5 == 0:  # Update progress every 5 steps
                progress_bar.update(current=i)
            low_press_center = find_low_pressure_center(
                ds, time_step, prev_center,
                search_radius_km=TRACKING_SEARCH_RADIUS_KM,
                smoothing_radius_km=100
            )
            if low_press_center is None:
                prev_center = None
                continue

            lat_center, lon_center = low_press_center
            msl_value = extract_and_average_variable(
                ds, lat_center, lon_center, time_step, time_coord,
                var_name="msl", buffer_km=100, aggregation="mean"
            )
            md_detections.append({
                "time": time_step,
                "latitude": lat_center,
                "longitude": lon_center,
                "msl": msl_value
            })
            prev_center = (
                lat_center,
                lon_center
            )
            detection_count += 1
        progress_bar.finish()

        std_print("INFO", "md", "Low-pressure systems detected centers", f"{detection_count} / {total_steps}")
        if not md_detections:
            std_print("WARN", "md", "No MD centers detected", "Insufficient low-pressure signatures in dataset")
            sys.exit(1)

        detections_df = pd.DataFrame(md_detections)
        detections_df["month"] = pd.to_datetime(detections_df["time"]).dt.strftime("%B")

        # Verification
        detection_hours = [int(h) for h in detections_df["time"].dt.hour.unique()]
        std_print(None, None, "Detections occur at hours", sorted(detection_hours))

        # Initial Tracking
        initial_tracks = track_monsoon_depressions(
            detections_df, search_radius_km=LINKING_SEARCH_RADIUS_KM,
            max_speed_ms=MAX_SPEED_MS, max_time_gap_hours=MAX_GAP_HOURS
        )
        std_print("INFO", "md", "Initial raw tracks found", len(initial_tracks))

        # Filter out very short tracks immediately (<5 points)
        initial_tracks = [t for t in initial_tracks if len(t) >= 5]
        std_print("INFO", "md", "After removing very short tracks", len(initial_tracks))

        # Stitch sequential fragments (gaps)
        stitched_tracks = stitch_sequential_tracks(
            initial_tracks, max_time_gap_hours=MAX_GAP_HOURS, dist_km=STITCH_DISTANCE_KM
        )
        std_print("INFO", "md", "Tracks after stitching gaps", len(stitched_tracks))

        # Consolidate simultaneous/overlapping tracks (Spatial Merge)
        consolidated_tracks = consolidate_simultaneous_tracks(
            stitched_tracks, dist_km=MERGE_DISTANCE_KM
        )
        std_print("INFO", "md", "Tracks after consolidating centers", len(consolidated_tracks))

        # # Split mixed tracks (ocean MD + inland disturbance)
        # split_mixed_tracks = split_mixed_tracks(consolidated_tracks)
        # std_print('INFO', 'md', 'Tracks after splitting mixed systems', len(split_tracks))

        # Filter by environment
        # ocean_tracks = filter_tracks_by_environment(split_mixed_tracks, require_ocean_start=True)
        # std_print('INFO', 'md', 'Tracks over ocean', len(ocean_tracks)")

        # Quality Filter tracks (Noise Removal)
        filtered_valid_tracks = filter_valid_tracks(
            # ocean_tracks,
            consolidated_tracks, min_duration=MIN_TRACK_DURATION_HOURS,
            min_distance=MIN_NET_DISPLACEMENT_KM
        )
        std_print("INFO", "md", "Valid tracks after QC", len(filtered_valid_tracks))
        if not filtered_valid_tracks:
            std_print(None, None, "No tracks survived filtering criteria", f"0 / {len(filtered_valid_tracks)}")
            return pd.DataFrame()  # Return empty dataframe

        # Smooth each track
        smoothed_tracks = []
        for track in filtered_valid_tracks:
            if len(track) > 5:
                smoothed_track = smooth_track_coordinates(track, window_size=5, iterations=4)
                smoothed_tracks.append(smoothed_track)
            else:
                smoothed_tracks.append(track)

        # Assign track IDs
        track_df_with_ids = pd.concat(
            [track.assign(track_id=i + 1) for i, track in enumerate(smoothed_tracks)],
            ignore_index=True
        )

        # Restore month column if lost
        if "month" not in track_df_with_ids.columns:
            track_df_with_ids["month"] = pd.to_datetime(track_df_with_ids["time"]).dt.strftime("%B")
        std_print("INFO", "md", "Final no. of tracks ready for analysis", len(smoothed_tracks))
        return track_df_with_ids
    except Exception as e:
        std_print("ERROR", "md", "process_files", str(e))
        traceback.print_exc()
        return None
    finally:
        # FIX: Memory Leak Risk: ds is passed as a parameter, so closing it here might cause issues elsewhere
        if "ds" in locals():
            ds.close()


# Pre-genesis tracing
def trace_progenitor_path(ds, genesis_lat, genesis_lon, genesis_time,
                          pre_hours=24, step_h=6, search_radius_km=200):
    """
    Walk backward from the first detected MD point (T_0) through ERA5,
    following the MSL minimum within a rolling search box.

    This finds the sub-threshold LPS that existed before MD detection
    criteria were met, the system criteria label as 'not yet an MD'.
    """
    genesis_time = pd.to_datetime(genesis_time)
    ds, time_coord = get_time_coord(ds)
    search_deg = search_radius_km / 111.0  # 1° ≈ 111 km

    progen_points = []
    current_lat = genesis_lat
    current_lon = genesis_lon

    for h in range(step_h, pre_hours + step_h, step_h):  # 6, 12, ..., pre_hours
        t = genesis_time - pd.Timedelta(hours=h)
        try:
            ds_t = ds.sel({time_coord: t}, method="nearest")
            if "msl" not in ds_t:
                break
            msl_t = ds_t["msl"]

            # Search box centred on previous (closer-to-T_0) position
            sub = msl_t.sel(
                latitude=slice(current_lat + search_deg, current_lat - search_deg),
                longitude=slice(current_lon - search_deg, current_lon + search_deg),
            )
            if sub.size == 0 or sub.isnull().all():
                break

            vals = sub.values
            idx = np.unravel_index(np.nanargmin(vals), vals.shape)
            found_lat = float(sub.latitude.values[idx[0]])
            found_lon = float(sub.longitude.values[idx[1]])
            msl_val = float(vals[idx])

            # Reject if the minimum jumped too far from current position
            dist = haversine_distance(current_lon, current_lat, found_lon, found_lat)
            if dist > search_radius_km * 1.5:
                std_print("DEBUG", "md",
                          f"Progenitor T-{h}h: MSL min too far ({dist:.0f} km) - stopping")
                break

            progen_points.append({
                "time":      t,
                "latitude":  found_lat,
                "longitude": found_lon,
                "msl":       msl_val / 100.0 if msl_val > 10000 else msl_val,
                "rel_hours": -h,
            })
            # Walk the search box back one more step from found position
            current_lat = found_lat
            current_lon = found_lon

        except Exception as e:
            std_print("DEBUG", "md", f"Progenitor trace stopped at T-{h}h", str(e))
            break

    if not progen_points:
        return pd.DataFrame()

    # Reverse: oldest -> newest (forward in time, but all before T_0)
    return pd.DataFrame(progen_points[::-1]).reset_index(drop=True)


# Plotting
def plot_and_export_tracks(track_df, output_dir, year, progenitor_dict=None):
    std_print("INFO", "md", "--- MD track plots ---")
    track_data = track_df.sort_values("time")

    # Figure
    fig = plt.figure(figsize=(14, 11))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]
    ax.set_extent(FULL_EXTENT, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, linewidth=0.8, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    progenitor_legend_added = False
    legend_handles = []
    for _, (track_id, track_data) in enumerate(track_df.groupby("track_id")):
        track_data = track_data.sort_values("time")
        month = track_data["month"].iloc[0]
        track_color = month_colors.get(month, "black")
        bbox_style = common_bbox()

        # Start/End (for annotatation)
        start_point = track_data.iloc[0]
        end_point = track_data.iloc[-1]
        start_time_str = pd.to_datetime(start_point["time"]).strftime("%d/%m %H:%M")
        end_time_str = pd.to_datetime(end_point["time"]).strftime("%d/%m %H:%M")
        start_lon = start_point["longitude"]
        start_lat = start_point["latitude"]
        end_lon= end_point["longitude"]
        end_lat = end_point["latitude"]

        # Pre-genesis track
        progen = (progenitor_dict or {}).get(int(track_id))
        if progen is not None and not progen.empty:
            progen_line = ax.plot(
                progen["longitude"].values,
                progen["latitude"].values,
                "--",
                color=track_color,
                linewidth=1.0,
                alpha=0.45,
                transform=ccrs.PlateCarree(),
                zorder=2,
                label="Pre-genesis (sub-threshold)" if not progenitor_legend_added else None,
            )
            ax.plot(
                progen["longitude"].values,
                progen["latitude"].values,
                "o",
                color=track_color,
                markersize=1.5,
                alpha=0.40,
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
            # Bridge connector: last progenitor point -> first MD point
            ax.plot(
                [progen["longitude"].iloc[-1], start_lon],
                [progen["latitude"].iloc[-1],  start_lat],
                "--",
                color=track_color,
                linewidth=0.8,
                alpha=0.30,
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
            if not progenitor_legend_added:
                legend_handles.append(progen_line[0])
                progenitor_legend_added = True

        # Plot all detection points
        ax.plot(
            track_data["longitude"],
            track_data["latitude"],
            "o",
            color=track_color,
            markersize=1.2,
            transform=ccrs.PlateCarree(),
            zorder=3
        )

        # Plot connecting lines
        line = plt.plot(
            track_data["longitude"],
            track_data["latitude"],
            "-",
            color=track_color,
            alpha=0.8,
            linewidth=1.5,
            solid_capstyle="round",
            label=f"Low-press. sys. Track ({month})",
            transform=ccrs.PlateCarree(),
            zorder=2
        )

        # Legend
        month_already_in_legend = False
        for handle in legend_handles:
            if (hasattr(handle, "get_label") and handle.get_label() == f"Low-press. sys. Track ({month})"):
                month_already_in_legend = True
                break
        if not month_already_in_legend:
            legend_handles.append(line)

        # Annotate Start point
        ax.plot(
            start_lon,
            start_lat,
            marker="o",
            color="green",
            markersize=7,
            markeredgewidth=0,
            transform=ccrs.PlateCarree(),
            zorder=8
        )
        ax.text(
            start_lon - 0.4,
            start_lat,
            f"T{track_id}",
            transform=ccrs.PlateCarree(),
            fontsize=FS_ANNOT,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox=bbox_style,
            zorder=9
        )
        ax.text(
            start_lon + 0.3,
            start_lat,
            start_time_str,
            transform=ccrs.PlateCarree(),
            fontsize=FS_ANNOT,
            fontweight="normal",
            color="black",
            ha="left",
            va="center",
            bbox=bbox_style
        )

        # Annotate End point
        ax.plot(
            end_lon,
            end_lat,
            marker="o",
            color="orange",
            markersize=7,
            markeredgewidth=0,
            transform=ccrs.PlateCarree(),
            zorder=8
        )
        ax.text(
            end_lon - 0.4,
            end_lat,
            f"T{track_id}",
            transform=ccrs.PlateCarree(),
            fontsize=FS_ANNOT,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox=bbox_style,
            zorder=9
        )
        ax.text(
            end_lon + 0.3,
            end_lat,
            end_time_str,
            transform=ccrs.PlateCarree(),
            fontsize=FS_ANNOT,
            fontweight="normal",
            color="black",
            ha="left",
            va="center",
            bbox=bbox_style
        )

    # Legend
    legend_handles, labels = ax.get_legend_handles_labels()
    bbox_style = common_bbox()
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", **legend_style())

    title =  f"Detected MD Tracks ({month_order[0]}-{month_order[-1]}, {year})"
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude (°E)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=FS_LABEL, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save
    md_track_path = os.path.join(output_dir, f"MD_tracks_{year}.png")
    plt.savefig(md_track_path)
    std_print("SAVED", "md", f"All Tracks for {year} plot", md_track_path)
    plt.close(fig)

    # Export
    if track_df.empty:
        std_print("ERROR", "md", "No track coordinates to export")
        return

    # std_print("DEBUG", "md", "No. of Track Coordinates to export", len(track_df))
    export_df = track_df[["time", "latitude", "longitude", "track_id"]].copy()
    export_df["time"] = pd.to_datetime(export_df["time"]).dt.strftime("%d/%m/%Y %H:%M")
    output_filename = os.path.join(output_dir, f"MD_coordinates_{year}_ALL.csv")
    export_df.to_csv(output_filename, index=False)
    std_print("EXPORT", "md", "Track Coordinates detections CSV", output_filename)


def plot_track_full_map(track_df_single, track_id, output_dir, progenitor_df=None):
    # Figure
    fig = plt.figure(figsize=(13, 10), dpi=100)
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]
    ax.set_extent(FULL_EXTENT, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle="--")
    ax.add_feature(cfeature.RIVERS)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    track_data = track_df_single.sort_values("time")
    track_times = pd.to_datetime(track_df_single["time"])
    month = track_data["month"].iloc[0]
    track_color = month_colors.get(month, "black")
    bbox_style = common_bbox()

    # Start/End (for annotatation)
    start_point = track_data.iloc[0]
    end_point = track_data.iloc[-1]
    start_datetime =  pd.to_datetime(start_point["time"])
    end_datetime = pd.to_datetime(end_point["time"])
    start_time_str = start_datetime.strftime("%d/%m %H:%M")
    end_time_str = end_datetime.strftime("%d/%m %H:%M")
    start_lon = start_point["longitude"]
    end_lon = end_point["longitude"]
    start_lat = start_point["latitude"]
    end_lat = end_point["latitude"]

    # Pre-genesis dashed track (sub-threshold precursor before T_0)
    if progenitor_df is not None and not progenitor_df.empty:
        ax.plot(
            progenitor_df["longitude"].values,
            progenitor_df["latitude"].values,
            "--",
            color=track_color,
            linewidth=1.2,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            zorder=3,
            label="Pre-genesis (sub-threshold)",
        )
        ax.plot(
            progenitor_df["longitude"].values,
            progenitor_df["latitude"].values,
            "o",
            color=track_color,
            markersize=2,
            alpha=0.45,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        # Bridge connector: last progenitor point -> first MD point
        ax.plot(
            [progenitor_df["longitude"].iloc[-1], start_lon],
            [progenitor_df["latitude"].iloc[-1],  start_lat],
            "--",
            color=track_color,
            linewidth=1.0,
            alpha=0.35,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        # Label oldest progenitor point with its relative time
        oldest = progenitor_df.iloc[0]
        rel_h  = int(oldest["rel_hours"])   # negative, e.g. -24
        ax.text(
            oldest["longitude"] - 0.3,
            oldest["latitude"]  + 0.2,
            f"T{rel_h}h",
            transform=ccrs.PlateCarree(),
            fontsize=FS_ANNOT - 1,
            color="dimgray",
            ha="right",
            va="bottom",
            zorder=6,
        )

    # Connecting lines
    ax.plot(
        track_data["longitude"].values,
        track_data["latitude"].values,
        "-o",
        color=track_color,
        transform=ccrs.PlateCarree(),
        linewidth=1.0,
        solid_capstyle="round",
        markersize=2,
        zorder=5
    )

    # 12-hour interval markers
    hours_elapsed = (track_times - start_datetime).dt.total_seconds() / 3600
    for elapsed_hours in np.arange(0, hours_elapsed.max() + 12, 12):
        idx = np.argmin(np.abs(hours_elapsed - elapsed_hours))
        if idx < len(track_df_single):
            point = track_df_single.iloc[idx]
            ax.plot(
                point["longitude"],
                point["latitude"],
                "o",
                color=track_color,
                markersize=4,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            time_label = f"{pd.to_datetime(point['time']).strftime('%d/%m %H:%M')}"
            ax.text(
                point["longitude"] - 0.1,
                point["latitude"] + 0.1,
                time_label,
                transform=ccrs.PlateCarree(),
                fontsize=FS_ANNOT - 1,
                fontweight="bold",
                color="black",
                ha="left",
                va="bottom",
                bbox=bbox_style,
                zorder=7
            )

    # Annotate Start point
    ax.plot(
        start_lon,
        start_lat,
        marker="o",
        color="green",
        markersize=3,
        markeredgewidth=5,
        transform=ccrs.PlateCarree(),
        zorder=8
    )
    # Annotate End point
    ax.plot(
        end_lon,
        end_lat,
        marker="o",
        color="orange",
        markersize=3,
        markeredgewidth=5,
        transform=ccrs.PlateCarree(),
        zorder=8
    )

    # Landfall marker
    if "is_land" in track_df_single.columns:
        land_status = track_df_single["is_land"].values
        transitions = np.diff(land_status.astype(int))
        ocean_to_land = np.where(transitions == 1)[0]
        if len(ocean_to_land) > 0:
            landfall_idx = ocean_to_land[0] + 1
            if landfall_idx < len(track_df_single):
                landfall_point = track_df_single.iloc[landfall_idx]
                lf_lon = landfall_point["longitude"]
                lf_lat = landfall_point["latitude"]
                lf_time = pd.to_datetime(landfall_point["time"]).strftime("%d/%m %H:%M")
                ax.plot(
                    lf_lon,
                    lf_lat,
                    marker="s",
                    color="yellow",
                    markersize=5,
                    transform=ccrs.PlateCarree(),
                    label="Landfall",
                    zorder=6
                )
                ax.text(
                    lf_lon + 0.2,
                    lf_lat + 0.2,
                    f"Landfall: {lf_time}",
                    transform=ccrs.PlateCarree(),
                    fontsize=FS_ANNOT,
                    fontweight="bold",
                    color="black",
                    ha="left",
                    va="top",
                    bbox=bbox_style,
                    zorder=8
                )

    # Legend
    legend_handles, labels = ax.get_legend_handles_labels()
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", **legend_style())

    title = f"Monsoon Depression Track {track_id}\n{start_time_str} to {end_time_str}, {years_to_process}"
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=10)
    plt.tight_layout()

    # Save
    full_map = f"MD_track_{track_id}_full_map.png"
    full_map_path = os.path.join(output_dir, full_map)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(full_map_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    std_print("SAVED", "md", "Full-map track plot", full_map_path)


def plot_detailed_track(track_df, track_id, output_dir, progenitor_df=None):
    track_id = int(track_id)
    track_data = track_df.sort_values("time")
    track_times = pd.to_datetime(track_data["time"])
    month = track_data["month"].iloc[0]
    track_color = month_colors.get(month, "teal")
    bbox_style = common_bbox()

    start_datetime = pd.to_datetime(track_data.iloc[0]["time"])
    end_datetime = pd.to_datetime(track_data.iloc[-1]["time"])
    start_time_str = start_datetime.strftime("%d/%m %H:%M")
    end_time_str = end_datetime.strftime("%d/%m %H:%M")

    # Figure
    fig = plt.figure(figsize=(13, 10))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]

    # Calculate extent based on track with padding, but clamp to clipped domain
    # Expand extent to include progenitor points if present
    all_lons = track_df["longitude"].values
    all_lats = track_df["latitude"].values
    if progenitor_df is not None and not progenitor_df.empty:
        all_lons = np.concatenate([progenitor_df["longitude"].values, all_lons])
        all_lats = np.concatenate([progenitor_df["latitude"].values, all_lats])

    min_lon = max(all_lons.min() - 4, FULL_EXTENT[0])
    min_lat = max(all_lats.min() - 4, FULL_EXTENT[3])
    max_lon = min(all_lons.max() + 4, FULL_EXTENT[1])
    max_lat = min(all_lats.max() + 4, FULL_EXTENT[2])

    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=1.0)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle="--")
    ax.add_feature(cfeature.RIVERS)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.5,linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    start_lon = track_df["longitude"].iloc[0]
    end_lon = track_df["longitude"].iloc[-1]
    start_lat = track_df["latitude"].iloc[0]
    end_lat = track_df["latitude"].iloc[-1]

    # Pre-genesis dashed track
    if progenitor_df is not None and not progenitor_df.empty:
        ax.plot(
            progenitor_df["longitude"].values,
            progenitor_df["latitude"].values,
            "--",
            color=track_color,
            linewidth=1.4,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
            zorder=3,
            label="Pre-genesis (sub-threshold)",
        )
        ax.plot(
            progenitor_df["longitude"].values,
            progenitor_df["latitude"].values,
            "o",
            color=track_color,
            markersize=2.5,
            alpha=0.45,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        # Bridge connector: last progenitor point -> first MD point
        ax.plot(
            [progenitor_df["longitude"].iloc[-1], start_lon],
            [progenitor_df["latitude"].iloc[-1],  start_lat],
            "--",
            color=track_color,
            linewidth=1.0,
            alpha=0.35,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        # Time labels on progenitor points at every 12 h step
        for _, prow in progenitor_df.iterrows():
            rel_h = int(prow["rel_hours"])
            if rel_h % 12 == 0:
                ax.text(
                    prow["longitude"] - 0.15,
                    prow["latitude"]  + 0.15,
                    f"T{rel_h}h",
                    transform=ccrs.PlateCarree(),
                    fontsize=FS_ANNOT - 1,
                    color="dimgray",
                    ha="right",
                    va="bottom",
                    bbox=common_bbox(),
                    zorder=6,
                )

    # # Connecting lines
    ax.plot(
        track_df["longitude"],
        track_df["latitude"],
        "-o",
        color=track_color,
        transform=ccrs.PlateCarree(),
        linewidth=1.5,
        solid_capstyle="round",
        markersize=2,
        # alpha=0.8,
        zorder=2,
    )

    # 12-hour interval markers
    hours_elapsed = (track_times - start_datetime).dt.total_seconds() / 3600
    # std_print('DEBUG', 'md', f"Track {track_id} time gaps: {track_df[time_coord].diff().value_counts()}")
    # There time gaps present 5hr, 3 hr and 1 hour gap for track 1 due to which there won't be exaclty 11 points between 2 12 hours marker points
    for elapsed_hours in np.arange(0, hours_elapsed.max() + 12, 12):
        idx = np.argmin(np.abs(hours_elapsed - elapsed_hours))
        if idx < len(track_df):
            point = track_df.iloc[idx]
            time_label = f"{int(elapsed_hours)}h {pd.to_datetime(point['time']).strftime('%d/%m %H:%M')}"
            ax.plot(
                point["longitude"],
                point["latitude"],
                "o",
                color=track_color,
                markersize=4,
                transform=ccrs.PlateCarree(),
                zorder=5
            )
            ax.text(
                point["longitude"] - 0.1,
                point["latitude"] + 0.1,
                time_label,
                transform=ccrs.PlateCarree(),
                fontsize=FS_ANNOT -1,
                fontweight="bold",
                color="black",
                ha="left",
                va="bottom",
                bbox=bbox_style,
                zorder=7
            )

    # Annotate Start point
    ax.plot(
        start_lon,
        start_lat,
        marker="o",
        color="green",
        markersize=5,
        markeredgewidth=5,
        transform=ccrs.PlateCarree(),
        zorder=8
    )
    # Annotate End point
    ax.plot(
        end_lon,
        end_lat,
        marker="o",
        color="orange",
        markersize=5,
        markeredgewidth=5,
        transform=ccrs.PlateCarree(),
        zorder=8
    )

    # Landfall marker
    if "is_land" in track_df.columns:
        land_status = track_df["is_land"].values
        transitions = np.diff(land_status.astype(int))
        ocean_to_land = np.where(transitions == 1)[0]
        if len(ocean_to_land) > 0:
            landfall_idx = ocean_to_land[0] + 1
            if landfall_idx < len(track_df):
                landfall_point = track_df.iloc[landfall_idx]
                lf_lon = landfall_point["longitude"]
                lf_lat = landfall_point["latitude"]
                lf_time = pd.to_datetime(landfall_point["time"]).strftime("%d/%m %H:%M")
                ax.plot(
                    lf_lon,
                    lf_lat,
                    marker="s",
                    color="yellow",
                    markersize=5,
                    transform=ccrs.PlateCarree(),
                    label="Landfall",
                    zorder=6
                )
                ax.text(
                    lf_lon + 0.4,
                    lf_lat + 0.1,
                    f"Landfall: {lf_time}",
                    transform=ccrs.PlateCarree(),
                    fontsize=FS_ANNOT,
                    fontweight="bold",
                    color="black",
                    ha="left",
                    va="top",
                    bbox=bbox_style,
                    zorder=8
                )

    # Legend
    legend_handles, labels = ax.get_legend_handles_labels()
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", **legend_style())

    title = f"Monsoon Depression Track {track_id}\n{start_time_str} to {end_time_str}, {years_to_process}"
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=10)
    plt.tight_layout()

    # Save
    file_name = f"MD_track_{track_id}_detailed.png"
    outpath = os.path.join(output_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    std_print("SAVED", "md", "Detailed track plot", outpath)


if __name__ == "__main__":
    t0 = time.time()
    args = parse_args()
    if args.year is not None:
        years_to_process = args.year
    else:
        detected = detect_year_from_data()
        years_to_process = [detected] if detected else [datetime.now().year]

    # Process each year
    for MONSOON_YEAR in years_to_process:
        print(f"\nProcessing year: {MONSOON_YEAR}")
        # Define months
        MONTH_DIRS = [
            f"June-{MONSOON_YEAR}",
            f"July-{MONSOON_YEAR}",
            f"August-{MONSOON_YEAR}",
            f"September-{MONSOON_YEAR}"
        ]

        output_dir = f"output_{MONSOON_YEAR}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            std_print("INFO", "md", f"Created output directory for year {MONSOON_YEAR}", output_dir)
        else:
            std_print("WARN", "md", f"Using existing output directory for year {MONSOON_YEAR}", output_dir)

        DATA_DIR = os.path.join("..", "Data", "single-levels", "ncfiles", f"{MONSOON_YEAR}")
        # PL_DIR = os.path.join("..", "Data", "pressure-levels", "nc-files", f"{MONSOON_YEAR}")
        # HGT_DIR = os.path.join("..", "Data", "elevation", "hgtfiles")
        ## lsm values: 1.0 = land, 0.0 = ocean. Threshold >= 0.5 gives boolean mask
        # LSM_SOILTYPE_Z_PATH = os.path.join("..", "Data", "single-levels", "ncfiles", "lsm-soiltype-z.nc")

        output_dir = f"output_{MONSOON_YEAR}"
        if not os.path.exists(output_dir):
            std_print("WARN", "md", f"NO output directory for year {MONSOON_YEAR} found - skipping")
            continue
        DETECTION_CSV = os.path.join(output_dir, f"MD_tracks_{MONSOON_YEAR}.csv")

        # Thresholds
        std_print("INFO", "md", "--- Detection thresholds ---")
        std_print(None, None, "Wind (10m)", f"{WIND_SPEED_LOWER_BOUND_MS}–{WIND_SPEED_UPPER_BOUND_MS} m/s")
        std_print(None, None, "MSLP", f"< {MSL_LOWER_BOUND_PA/100:.0f} hPa")
        std_print(None, None, "Precipitation", f"> {PRECIPITATION_THRESHOLD_MM} mm/hr")
        std_print(None, None, "Vorticity", f"> {VORTICITY_THRESHOLD} /s")
        std_print(None, None, "Min duration", f"{MIN_TRACK_DURATION_HOURS} h")
        std_print(None, None, "Min displacement", f"{MIN_NET_DISPLACEMENT_KM} km")
        std_print(None, None, "Max speed", f"{MAX_SPEED_MS} m/s")
        std_print(None, None, "Max gap", f"{MAX_GAP_HOURS} h")

        # Load dataset Info
        container = {"ds": None, "success": False, "error": None}
        t = threading.Thread(target=load_dataset_async, args=(MONTH_DIRS, DATA_DIR, container))
        t.daemon = True
        t.start()
        spinner = ProgressSpinner(len(MONTH_DIRS) * 31, f"Loading ERA5 {MONSOON_YEAR} ({len(MONTH_DIRS)} months)")
        step = 0
        while t.is_alive():
            spinner.update(current=step)
            time.sleep(0.2)
            step += 1
        spinner.finish()

        if not container["success"]:
            std_print("ERROR", "md", f"Load failed for {MONSOON_YEAR}", container["error"])
            continue

        ds = container["ds"]
        std_print("INFO", "md", f"Loaded {MONSOON_YEAR}", f"{len(MONTH_DIRS)} months")

        std_print("INFO", "md", "--- Detection ---")
        diagnose_temporal_issues(ds)
        std_print("INFO", "md", "Fixing temporal resolution")
        ds = fix_temporal_resolution(ds, target_resolution_hours=1)
        validate_temporal_consistency(ds, expected_hours=1, tolerance=0.05)

        # Compute detection vars into RAM (clipped domain)
        detect_vars = ["msl", "u10", "v10", "tp"]
        available = [v for v in detect_vars if v in ds.data_vars]
        std_print("DEBUG", "md", "Computing detection vars into RAM", available)
        ds_det = ds[available].compute()
        ram_mb = sum(v.nbytes for v in ds_det.data_vars.values()) / 1024**2
        std_print("INFO", "md", "Detection dataset in RAM", f"{ram_mb:.0f} MB")

        full_track_df = process_files(ds_det)

        # Pre-genesis tracing
        progenitor_dict = {}   # {track_id: DataFrame}
        if full_track_df is not None and not full_track_df.empty:
            std_print("INFO", "md", "--- Pre-genesis tracing ---")
            for tid in sorted(full_track_df["track_id"].unique()):
                try:
                    t_df_genesis = full_track_df[full_track_df["track_id"] == tid].sort_values("time")
                    genesis_row = t_df_genesis.iloc[0]
                    progen = trace_progenitor_path(
                        ds_det,
                        genesis_lat=float(genesis_row["latitude"]),
                        genesis_lon=float(genesis_row["longitude"]),
                        genesis_time=pd.to_datetime(genesis_row["time"]),
                        pre_hours=24,   # look 24 h before T_0
                        step_h=6,
                        search_radius_km=200,
                    )
                    progenitor_dict[int(tid)] = progen
                    if not progen.empty:
                        std_print("INFO", "md",
                                  f"Track {tid} progenitor",
                                  f"{len(progen)} points traced back to "
                                  f"T{int(progen['rel_hours'].iloc[0])}h")
                    else:
                        std_print("WARN", "md", f"Track {tid}", "no progenitor path found")
                except Exception as e:
                    std_print("ERROR", "md", f"Track {tid} progenitor trace failed", str(e))
        del ds_det
        gc.collect()

        if full_track_df is None or full_track_df.empty:
            std_print("ERROR", "md", f"No tracks for {MONSOON_YEAR} - skipping")
            continue

        if "track_id" not in full_track_df.columns:
            std_print("ERROR", "md", "track_id column missing")
            continue

        # Land/sea classification
        std_print("INFO", "md", "Classifying land/ocean points")
        full_track_df["is_land"] = full_track_df.apply(
            lambda row: is_point_land(row["latitude"], row["longitude"]), axis=1
        )

        # Per-track smoothing, speeds, per-track plots
        std_print("INFO", "md", "--- Per-track Process ---")
        for track_id in sorted(full_track_df["track_id"].unique()):
            track_id = int(track_id)
            mask = full_track_df["track_id"] == track_id
            t_df = full_track_df.loc[mask].copy()

            # Second smooth pass with small window
            t_df = smooth_track_coordinates(t_df, window_size=3, iterations=2)
            full_track_df.loc[mask, ["latitude", "longitude"]] = t_df[["latitude", "longitude"]].values

            # Compute speeds
            t_df = compute_track_speeds(t_df)
            for col in ["speed_kmh", "speed_ms", "distance_km"]:
                full_track_df.loc[mask, col] = t_df[col].values

            # Per-track map
            track_dir = os.path.join(output_dir, f"track_{track_id}")
            os.makedirs(track_dir, exist_ok=True)
            plot_track_full_map(t_df, track_id, track_dir, progenitor_df=progenitor_dict.get(track_id))

            spd = t_df["speed_kmh"].dropna()
            std_print("INFO", "md", f"Track {track_id} speed",
                      f"mean {spd.mean():.1f} km/h  |  "
                      f"dur {(t_df['time'].max()-t_df['time'].min()).total_seconds()/3600:.0f} h")

        # Combined plot
        plot_and_export_tracks(full_track_df, output_dir, year=MONSOON_YEAR, progenitor_dict=progenitor_dict)

        # Export CSV
        export_cols = [
            "time",
            "latitude",
            "longitude",
            "track_id",
            "month",
            "is_land",
            "msl",
            "speed_kmh",
            "speed_ms",
            "distance_km"
        ]
        export_cols = [c for c in export_cols if c in full_track_df.columns]

        master_df = full_track_df[export_cols].copy()
        master_df["time"] = pd.to_datetime(master_df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        master_df.to_csv(DETECTION_CSV, index=False)
        std_print("EXPORT", "md", "Detection track CSV", DETECTION_CSV)
        std_print("INFO", "md", f"Year {MONSOON_YEAR} tracks exported", full_track_df["track_id"].nunique())
        std_print("INFO", "md", f"Year {MONSOON_YEAR} total rows", len(master_df))

    elapsed = time.time() - t0
    print(f"\nAll years detection finished in {elapsed/60:.1f} min")
