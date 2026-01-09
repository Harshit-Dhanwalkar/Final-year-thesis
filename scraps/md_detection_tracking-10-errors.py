import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict
import math

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from azimuthal_hovmoller import (
    compute_azimuthal_hovmoller,
    plot_hovmoller
)
from speed_analysis import (
    is_point_land,
    compute_track_speeds,
    plot_speed_analysis,
)
from correlation_analysis import analyze_single_track

# import traceback
# import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f" >  [INFO   ] Created output directory: {output_dir}")
else:
    print(f" >  [INFO   ] {output_dir} directory already present.")
print("-------------------------------------------------------------")

DATA_DIR = os.path.join("Data", "single-levels", "ncfiles")
MONTH_DIRS = [
    "June-2024",
    "July-2024",
    "August-2024",
    "September-2024",
]

DISTANCE_THRESHOLD_KM = 51  # 300      # km
# TIME_THRESHOLD_HOURS = 1               # hr

# for wind creterion : https://mausam.imd.gov.in/imd_latest/monsoonreport2024.pdf page: 284
WIND_SPEED_LOWER_BOUND_MS = 5        # == 17knots  # 8     # m/s
WIND_SPEED_UPPER_BOUND_MS = 19       # == 27knots  # 16    # m/s
MSL_LOWER_BOUND_PA = 100000  # 99800   # Pa
PRECIPITATION_THRESHOLD_MM = 0.3       # mm/hr

MAX_GAP_HOURS = 12
MAX_GAP_ALLOWED_HOURS = 6              # Bridge small gaps In Detection
MERGE_DISTANCE_KM = 100                # Distance to consider two detections as same MD
STITCH_DISTANCE_KM = 300
MIN_TRACK_DURATION_HOURS = 24
MIN_NET_DISPLACEMENT_KM = 250          # MD travel distance
BUFFER = 5.0                           # deg

EARTH_R = 6371.0 # km

month_colors = {
    "June": "orange",
    "July": "red",
    "August": "blue",
    "September": "purple",
}
month_order = [
    "June",
    "July",
    "August",
    "September",
]

# Hovmoller
radii = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
variables = [
    "wind10",      # wind10 = sqrt(u10^2 + v10^2)
    'vr10',        # Radial Wind
    'vt10',        # Tangential Wind
    "tp",          # total precipitation
    "t2m",         # 2m temperature
    "msl",         # mean sea level pressure
    # "sp",          # surface pressure
    "stl1",        # soil temperature level 1
    "swvl1",       # soil water volume level 1
    "avg_slhtf",   # surface latent heat flux
    "avg_ishf"     # surface sensible heat flux
]

# desired_tracks = [1, 3, 4, 5]

# CONFIGURATION
RESOLUTION_CONFIGS = {
    '1h': {
        'hours': 1,
        'max_time_gap_allowed': 12,
        'search_radius_km': 150,
        'max_speed_ms': 15,
        'description': '1-Hourly (ERA5 native resolution)',
        'output_dir_suffix': '_1h'
    },
    '3h': {
        'hours': 3,
        'max_time_gap_allowed': 12,
        'search_radius_km': 150,
        'max_speed_ms': 15,
        'description': '3-Hourly (Satellite era resolution)',
        'output_dir_suffix': '_3h'
    },
    '6h': {
        'hours': 6,
        'max_time_gap_allowed': 24,
        'search_radius_km': 200,
        'max_speed_ms': 15,
        'description': '6-Hourly (IMD operational standard)',
        'output_dir_suffix': '_6h'
    }
}


def has_closed_contour(data_array, lat, lon):
    lat_min, lat_max = lat - BUFFER, lat + BUFFER
    lon_min, lon_max = lon - BUFFER, lon + BUFFER

    # Check bounds
    if (
        lat_min < data_array.latitude.min() or
        lat_max > data_array.latitude.max() or
        lon_min < data_array.longitude.min() or
        lon_max > data_array.longitude.max()
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
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )

    if subset.size == 0:
        return False

    min_val = subset.min().item()

    boundary_lats = [subset.latitude.min().item(),
                     subset.latitude.max().item()]
    boundary_lons = [subset.longitude.min().item(),
                     subset.longitude.max().item()]

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
    nearest_subset = data_array.sel(latitude=lat,
                                    longitude=lon,
                                    method="nearest")
    central_pressure = nearest_subset.item()
    min_val_subset = subset.min().item()

    if np.isclose(central_pressure, min_val_subset, rtol=0.01):
        # Check if the boundary pressure is higher than the center pressure
        if subset.latitude.size > 1 and subset.longitude.size > 1:
            boundary_pressure = np.nanmean([
                subset.isel(latitude=0).values.mean(),      # North edge
                subset.isel(latitude=-1).values.mean(),     # South edge
                subset.isel(longitude=0).values.mean(),     # West edge
                subset.isel(longitude=-1).values.mean(),    # East edge
            ])
            if boundary_pressure > central_pressure + 50:
                return True

    return False


def check_wind_criteria(u_data, v_data, lat, lon):
    BUFFER = 5.0
    lat_min, lat_max = lat - BUFFER, lat + BUFFER
    lon_min, lon_max = lon - BUFFER, lon + BUFFER

    if not (
        u_data.latitude.min() <= lat_min
        and u_data.latitude.max() >= lat_max
        and u_data.longitude.min() <= lon_min
        and u_data.longitude.max() >= lon_max
    ):
        return False

    u_subset = u_data.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )
    v_subset = v_data.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )

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

def calculate_vorticity(ds, lat_idx, lon_idx):
    # Get u, v at the point
    u = ds['u10'].isel(latitude=lat_idx, longitude=lon_idx).values
    v = ds['v10'].isel(latitude=lat_idx, longitude=lon_idx).values

    # Get neighboring points
    if (lat_idx == 0 or lat_idx == len(ds.latitude) - 1 or 
        lon_idx == 0 or lon_idx == len(ds.longitude) - 1):
        return np.nan

    # Get grid spacing
    lat = ds.latitude.values[lat_idx]
    dlat = np.abs(ds.latitude.values[1] - ds.latitude.values[0])
    dlon = np.abs(ds.longitude.values[1] - ds.longitude.values[0])

    # Convert to meters
    dy = EARTH_R * np.deg2rad(dlat)
    dx = EARTH_R * np.cos(np.deg2rad(lat)) * np.deg2rad(dlon)

    # Get u and v at neighboring points
    u_north = float(ds['u10'].isel(latitude=lat_idx+1, longitude=lon_idx).values)
    u_south = float(ds['u10'].isel(latitude=lat_idx-1, longitude=lon_idx).values)
    v_east = float(ds['v10'].isel(latitude=lat_idx, longitude=lon_idx+1).values)
    v_west = float(ds['v10'].isel(latitude=lat_idx, longitude=lon_idx-1).values)

    # Calculate gradients
    du_dy = (u_north - u_south) / (2 * dy)
    dv_dx = (v_east - v_west) / (2 * dx)

    # Relative vorticity (ζ = ∂v/∂x - ∂u/∂y)
    vorticity = dv_dx - du_dy

    return vorticity

def check_vorticity_criteria(data_slice, lat, lon):
    try:
        if 'u10' not in data_slice.data_vars or 'v10' not in data_slice.data_vars:
            return False, np.nan

        # Get latitude and longitude arrays
        lats = data_slice.latitude.values
        lons = data_slice.longitude.values

        # Find indices nearest to the center
        lat_idx = np.argmin(np.abs(lats - lat))
        lon_idx = np.argmin(np.abs(lons - lon))
        # Check if we're not at the boundaries
        if lat_idx == 0 or lat_idx == len(lats) - 1 or lon_idx == 0 or lon_idx == len(lons) - 1:
            return False, np.nan

        # Get grid spacing in radians
        dlat_rad = np.deg2rad(lats[1] - lats[0])
        dlon_rad = np.deg2rad(lons[1] - lons[0])

        # Adjust dlon for latitude
        dlon_rad_adjusted = dlon_rad * np.cos(np.deg2rad(lat))

        try:
            # Get u, v values at center and neighbors
            u_center = float(data_slice['u10'].isel(latitude=lat_idx, longitude=lon_idx).values)
            v_center = float(data_slice['v10'].isel(latitude=lat_idx, longitude=lon_idx).values)

            u_north = float(data_slice['u10'].isel(latitude=lat_idx+1, longitude=lon_idx).values)
            u_south = float(data_slice['u10'].isel(latitude=lat_idx-1, longitude=lon_idx).values)
            v_east = float(data_slice['v10'].isel(latitude=lat_idx, longitude=lon_idx+1).values)
            v_west = float(data_slice['v10'].isel(latitude=lat_idx, longitude=lon_idx-1).values)

            # Check for NaN values
            if any(np.isnan(v) for v in [u_north, u_south, v_east, v_west]):
                return False, np.nan

            # Compute finite differences
            du_dy = (u_north - u_south) / (2 * EARTH_R * dlat_rad)  # m/s per meter
            dv_dx = (v_east - v_west) / (2 * EARTH_R * dlon_rad_adjusted)  # m/s per meter

            # Compute relative vorticity (dv/dx - du/dy)
            vorticity = dv_dx - du_dy  # s^-1

            # Check if vorticity is positive (cyclonic)
            return vorticity > 1e-5, vorticity

        except (IndexError, ValueError, TypeError) as e:
            return False, np.nan

    except Exception as e:
        # logger.warning(f" >  [Error] Computing vorticity: {e}")
        print(" >  [ERROR] Computing vorticity: {e}")
        return False, np.nan

def check_moisture_criteria(data_slice, lat, lon):
    lat_min, lat_max = lat - BUFFER, lat + BUFFER
    lon_min, lon_max = lon - BUFFER, lon + BUFFER

    try:
        # Check bounds
        if not (data_slice.latitude.min() <= lat_min and 
                data_slice.latitude.max() >= lat_max and
                data_slice.longitude.min() <= lon_min and 
                data_slice.longitude.max() >= lon_max):
            return False
        # Check total precipitation (tp m/hr)
        if "tp" not in data_slice.data_vars:
            print(" >  [Warning] Missing 'tp'. Skipping this check.")
            return False

        tp_data = data_slice["tp"]

        lat_slice = slice(
            max(lat_max, data_slice.latitude.min().item()),
            min(lat_min, data_slice.latitude.max().item())
        )
        lon_slice = slice(
            max(lon_min, data_slice.longitude.min().item()),
            min(lon_max, data_slice.longitude.max().item())
        )
        tp_subset = tp_data.sel(
            latitude=lat_slice,
            longitude=lon_slice,
        )
        if tp_subset.size == 0:
            return False

        max_precipitation = float(tp_subset.max().values)
        precipitation_check = (max_precipitation * 1000) >= PRECIPITATION_THRESHOLD_MM  # m/hr -> mm/hr

        return precipitation_check

    except Exception as e:
        print(f" >  [Warning] Moisture check error: {str(e)}")
        return False

# Helper function
def diagnose_temporal_issues(ds):
    """
    Identifies:
    1. Gaps in time series
    2. Duplicate timestamps
    3. Overlapping data from monthly files
    4. Irregular sampling
    """
    time_coord = "time" if "time" in ds.coords else "valid_time"
    times = pd.to_datetime(ds[time_coord].values)

    # Basic stats
    print(f"\n [TEMPORAL RANGE]")
    print(f"  Start: {times[0]}")
    print(f"  End:   {times[-1]}")
    print(f"  Total duration: {(times[-1] - times[0]).total_seconds() / 3600:.1f} hours")
    print(f"  Total points: {len(times)}")

    # Calculate time differences
    time_diffs = np.diff(times).astype('timedelta64[m]').astype(float) / 60  # Convert to hours

    print(f"\n [TIME STEP STATISTICS]")
    print(f"  Mean interval:   {np.mean(time_diffs):.2f} hours")
    print(f"  Median interval: {np.median(time_diffs):.2f} hours")
    print(f"  Std Dev:         {np.std(time_diffs):.2f} hours")
    print(f"  Min interval:    {np.min(time_diffs):.2f} hours")
    print(f"  Max interval:    {np.max(time_diffs):.2f} hours")

    # Identify anomalies
    mode_diff = np.argmax(np.bincount(time_diffs.astype(int)))
    print(f" >  Most common interval: {mode_diff} hours")

    # Find irregular gaps
    unique_diffs, counts = np.unique(time_diffs.astype(int), return_counts=True)
    print(f"\n [TIME STEP DISTRIBUTION]")
    for diff, count in sorted(zip(unique_diffs, counts), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / len(time_diffs)
        print(f"  {diff:2d} hour intervals: {count:4d} times ({percentage:5.1f}%)")

    # Detect month boundaries
    print(f"\n [MONTH BOUNDARY ANALYSIS]")
    month_changes = []
    for i in range(1, len(times)):
        if times[i].month != times[i-1].month:
            month_changes.append((i, times[i-1], times[i], time_diffs[i-1]))

    if month_changes:
        print(f"  Found {len(month_changes)} month transitions:")
        for idx, prev_time, curr_time, gap in month_changes:
            print(f"   Index {idx}: {prev_time.strftime('%Y-%m-%d %H:%M')} → "
                  f"{curr_time.strftime('%Y-%m-%d %H:%M')} (gap: {gap:.1f} hours)")
    else:
        print(" >  No month transitions detected (single continuous month)")

    # Detect duplicates
    print(f"\n [DUPLICATE/OVERLAP DETECTION]")
    unique_times, indices = np.unique(times, return_index=True)
    n_duplicates = len(times) - len(unique_times)

    if n_duplicates > 0:
        print(f" >  Found {n_duplicates} duplicate timestamps!")
        duplicate_mask = np.zeros(len(times), dtype=bool)
        duplicate_mask[indices] = True
        duplicate_indices = np.where(~duplicate_mask)[0]
        for idx in duplicate_indices[:5]:  # Show first 5
            print(f"  Duplicate at index {idx}: {times[idx]}")
    else:
        print(" >  No duplicate timestamps")

    # Detect negative gaps (time goes backwards)
    print(f"\n [TIME CONTINUITY CHECK]")
    negative_gaps = np.where(time_diffs < 0)[0]
    if len(negative_gaps) > 0:
        print(f" >  Found {len(negative_gaps)} backward time jumps!")
        for idx in negative_gaps[:5]:
            print(f"  Jump backward at index {idx}: {times[idx]} → {times[idx+1]} "
                  f"(difference: {time_diffs[idx]:.1f} hours)")
    else:
        print(" >  Time is monotonically increasing")

    # Detect zero-gaps (identical consecutive times)
    zero_gaps = np.where(time_diffs == 0)[0]
    if len(zero_gaps) > 0:
        print(f" >  Found {len(zero_gaps)} identical consecutive timestamps!")

    # Summary diagnosis
    print(f"\n [DIAGNOSIS SUMMARY]")
    if len(unique_diffs) == 1:
        print(f"  REGULAR SPACING: All intervals are {unique_diffs[0]} hours")
    else:
        print(f" >  IRREGULAR SPACING: {len(unique_diffs)} different interval types")
        print(f"     Likely cause: Multiple monthly files with different resolutions")
        print(f"     or overlapping time periods from file combination")

    if n_duplicates > 0:
        print(f" >  DUPLICATES PRESENT: {n_duplicates} repeated timestamps")

    if len(negative_gaps) > 0:
        print(f" >  TIME REVERSALS: {len(negative_gaps)} backward jumps")

    if len(month_changes) > 0 and len(unique_diffs) > 1:
        print(f" >  FILES LIKELY CONCATENATED: Different resolutions at month boundaries")

    return {
        'times': times,
        'time_diffs': time_diffs,
        'unique_intervals': unique_diffs,
        'interval_counts': counts,
        'n_duplicates': n_duplicates,
        'month_changes': month_changes,
        'negative_gaps': negative_gaps,
        'zero_gaps': zero_gaps
    }


def fix_temporal_resolution(ds, target_resolution_hours=1, method='interpolate'):
    time_coord = "time" if "time" in ds.coords else "valid_time"
    times = pd.to_datetime(ds[time_coord].values)

    print(f"\n [TEMPORAL RESAMPLING]")
    print(f"  Original: {len(times)} time points")

    if method == 'interpolate':
        # Create regular time grid
        time_start = times[0]
        time_end = times[-1]
        new_times = pd.date_range(
            start=time_start,
            end=time_end,
            freq=f'{target_resolution_hours}h'
        )

        # Reindex dataset to regular grid with interpolation
        ds_fixed = ds.reindex({time_coord: new_times}, method='nearest')

        print(f"  Resampled: {len(new_times)} time points (every {target_resolution_hours}h)")
        print(f"  Interpolation method applied")

    elif method == 'drop_duplicates':
        # Remove exact duplicate timestamps, keep first occurrence
        _, unique_indices = np.unique(times, return_index=True)
        ds_fixed = ds.isel({time_coord: np.sort(unique_indices)})

        print(f"  Cleaned: {len(unique_indices)} unique time points")
        print(f"  Duplicates removed")

    elif method == 'forward_fill':
        # Forward-fill missing values, keep original time grid
        ds_fixed = ds.fillna(method='ffill', dim=time_coord)

        print(f"  Filled: Missing values forward-filled")
        print(f"  Forward-fill method applied")
    else:
        raise ValueError(f"Unknown method: {method}")

    return ds_fixed


def validate_temporal_consistency(ds, expected_hours=1, tolerance=0.1):
    time_coord = "time" if "time" in ds.coords else "valid_time"
    times = pd.to_datetime(ds[time_coord].values)
    time_diffs = np.diff(times).astype('timedelta64[m]').astype(float) / 60

    min_expected = expected_hours * (1 - tolerance)
    max_expected = expected_hours * (1 + tolerance)

    within_tolerance = (time_diffs >= min_expected) & (time_diffs <= max_expected)
    n_valid = np.sum(within_tolerance)
    n_total = len(time_diffs)
    validity_pct = 100 * n_valid / n_total

    is_valid = validity_pct >= 95  # At least 95% compliance

    report = {
        'is_valid': is_valid,
        'expected_hours': expected_hours,
        'tolerance': tolerance,
        'valid_intervals': n_valid,
        'total_intervals': n_total,
        'validity_percentage': validity_pct,
        'min_interval': np.min(time_diffs),
        'max_interval': np.max(time_diffs),
        'mean_interval': np.mean(time_diffs),
        'violations': np.where(~within_tolerance)[0].tolist()
    }

    print(f"\n [TEMPORAL VALIDATION]")
    print(f"  Expected: {expected_hours}±{tolerance*100:.0f}% hours")
    print(f"  Compliance: {validity_pct:.1f}% ({n_valid}/{n_total} intervals)")
    print(f"  Valid: {' YES' if is_valid else '✗ NO'}")

    if not is_valid and len(report['violations']) > 0:
        print(f"  Violations at indices: {report['violations'][:10]}")

    return is_valid, report


def resample_dataset_to_resolution(ds, target_hours=1):
    time_coord = "time" if "time" in ds.coords else "valid_time"
    times = pd.to_datetime(ds[time_coord].values)

    if target_hours == 1:
        print(f" > Dataset already at 1-hourly resolution ({len(times)} steps)")
        return ds

    new_times = pd.date_range(
        start=times[0],
        end=times[-1],
        freq=f'{target_hours}h'
    )

    original_count = len(times)
    new_count = len(new_times)

    print(f"\n [TEMPORAL RESAMPLING]")
    print(f"  Input:  {original_count} time steps (hourly)")
    print(f"  Output: {new_count} time steps ({target_hours}-hourly)")
    print(f"  Reduction: {100*(1-new_count/original_count):.1f}%")

    # Interpolate all variables to new time grid
    ds_resampled = ds.interp(
        {time_coord: new_times},
        method='linear',
        kwargs={"fill_value": "extrapolate"}
    )

    return ds_resampled

# Helper function to calculate haversine distance
def haversine(lon1, lat1, lon2, lat2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_R * c

def find_low_pressure_center(
    data_slice: xr.Dataset,
    time_step: np.datetime64,
    prev_center: None,
    search_radius_km: float = DISTANCE_THRESHOLD_KM,
    smoothing_radius_km: int = 100,
) -> Optional[Tuple[float, float]]:
    required_vars = ["msl", "u10", "v10", "tp"]
    if not all(var in data_slice.data_vars for var in required_vars):
        return None

    try:
        msl_data = data_slice["msl"]
        u_data = data_slice["u10"]
        v_data = data_slice["v10"]

        # Find all low pressure candidates
        msl_values = msl_data.values.flatten()

        # Get top N lowest pressure points
        n_candidates = 6
        flat_indices = np.argsort(msl_values)[:n_candidates]

        # Convert flat indices to 2D indices
        shape = msl_data.shape
        candidates = []
        for idx in flat_indices:
            lat_idx, lon_idx = np.unravel_index(idx, shape)
            lat = float(msl_data.latitude.values[lat_idx])
            lon = float(msl_data.longitude.values[lon_idx])
            pressure = float(msl_values[idx])

            # Check if this is a reasonable candidate
            if pressure > MSL_LOWER_BOUND_PA + 500:
                continue

            # Calculate distance from previous center if available
            distance = 0
            if prev_center is not None:
                prev_lat, prev_lon = prev_center
                distance = haversine(prev_lon, prev_lat, lon, lat)
                if distance > DISTANCE_THRESHOLD_KM:
                    continue

            candidates.append({
                'lat': lat,
                'lon': lon,
                'pressure': pressure,
                'distance': distance
            })

        if not candidates:
            return None

        # Sort candidates by pressure (lowest first)
        candidates.sort(key=lambda x: x['pressure'])

        for candidate in candidates:
            lat_center = candidate['lat']
            lon_center = candidate['lon']
            min_msl_val = candidate['pressure']

            # Relaxed criteria check
            pressure_ok = min_msl_val <= MSL_LOWER_BOUND_PA + 1000
            try:
                closed_contour = has_closed_contour(msl_data, lat_center, lon_center)
            except:
                closed_contour = False
            try:
                wind_ok = check_wind_criteria(u_data, v_data, lat_center, lon_center)
            except:
                wind_ok = False
            try:
                moisture_ok = check_moisture_criteria(data_slice, lat_center, lon_center)
            except:
                moisture_ok = False
            try:
                vorticity_ok, _ = check_vorticity_criteria(data_slice, lat_center, lon_center)
            except:
                vorticity_ok = False

            score = int(pressure_ok) + int(closed_contour) + int(wind_ok) + int(moisture_ok) + int(vorticity_ok)

            # if (pressure_ok and closed_contour and wind_ok and moisture_ok and vorticity_ok):
            if score >= 3:
                return lat_center, lon_center

        return None

    except Exception as e:
        # logger.warning(f" >  [Error] Finding low pressure center at {time_step}: {e}")
        print(f" >  [ERROR] Finding low pressure center at {time_step}: {e}")
        return None

def track_monsoon_depressions(
    detections,
    max_speed_ms=10,
    search_radius_km=100,
    max_time_gap_hours=12
):
    if detections.empty:
        return []

    detections = detections.sort_values("time").reset_index(drop=True)
    detections['time'] = pd.to_datetime(detections['time'])
    detections['hour'] = detections['time'].dt.floor('H')

    print(f"\n >  [TRACKING DEBUG]")
    print(f"    Total detections: {len(detections)}")
    print(f"    Time range: {detections['time'].min()} to {detections['time'].max()}")

    # Group by time
    time_groups = detections.groupby('hour')
    time_steps = sorted(time_groups.groups.keys())

    print(f"    Hours with detections: {len(time_steps)}")
    print(f"    Dataset has {len(time_steps)} hourly steps with detections")

    if len(time_steps) > 1:
        time_diffs = []
        for i in range(1, len(time_steps)):
            diff = (time_steps[i] - time_steps[i-1]).total_seconds() / 3600
            time_diffs.append(diff)

        print(f"    Time gaps between detections:")
        print(f"      Mean: {np.mean(time_diffs):.1f} hours")
        print(f"      Min/Max: {np.min(time_diffs):.1f}/{np.max(time_diffs):.1f} hours")
        print(f"      Std Dev: {np.std(time_diffs):.1f} hours")

        # Show distribution of gaps
        unique_gaps, gap_counts = np.unique(np.round(time_diffs), return_counts=True)
        for gap, count in sorted(zip(unique_gaps, gap_counts)):
            print(f"      {int(gap)}-hour gaps: {count} times")

    tracks = []
    if len(time_steps) < 2:
        print(" >  [WARNING] Not enough time steps for tracking")
        return tracks

    active_tracks = {}
    first_detections = time_groups.get_group(time_steps[0])
    for i, (_, row) in enumerate(first_detections.iterrows()):
        active_tracks[i] = [row]

    print(f" >  [INFO  ] Initial active tracks: {len(active_tracks)}")

    for t_idx in range(1, len(time_steps)):
        current_hour = time_steps[t_idx]
        current_detections = time_groups.get_group(current_hour)
        prev_hour = time_steps[t_idx-1]
        time_diff_hours = (current_hour - prev_hour).total_seconds() / 3600

        # Only proceed if time gap is reasonable
        if time_diff_hours > max_time_gap_hours:
            print(f" > [INFO  ] Large time gap: {time_diff_hours:.1f} hours from {prev_hour} to {current_hour}")
            # End all current tracks
            for track_id, points in list(active_tracks.items()):
                if len(points) >= 3:
                    tracks.append(pd.DataFrame(points))
            active_tracks = {}
            # Start new tracks from current detections
            for i, (_, row) in enumerate(current_detections.iterrows()):
                active_tracks[i] = [row]
            continue

        # Prepare cost matrix for assignment
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
                distance = haversine(
                    last_point['longitude'],
                    last_point['latitude'],
                    det['longitude'],
                    det['latitude']
                )

                # Check if within search radius
                if distance > search_radius_km:
                    continue

                # Calculate speed
                speed_ms = (distance * 1000) / (time_diff_hours * 3600)

                # Check speed constraint
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

            for t_idx_m, d_idx_m in zip(track_indices, det_indices):
                cost = cost_matrix[t_idx_m, d_idx_m]

                if cost < np.inf:  # Valid assignment
                    track_id = list(active_tracks.keys())[t_idx_m]
                    det_row = current_detections.iloc[d_idx_m]

                    # Append to track
                    active_tracks[track_id].append(det_row)
                    new_active_tracks[track_id] = active_tracks[track_id]
                    assigned_detections.add(d_idx_m)

            # End unassigned tracks (minimum length check)
            for track_id, points in active_tracks.items():
                if track_id not in new_active_tracks:
                    if len(points) >= 3:
                        tracks.append(pd.DataFrame(points))

            # Start new tracks from unassigned detections
            for j in range(n_detections):
                if j not in assigned_detections:
                    new_id = max(new_active_tracks.keys()) + 1 if new_active_tracks else 0
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

    print(f" >  [INFO  ] Final tracks found: {len(tracks)}")

    return tracks

def stitch_sequential_tracks(tracks: List[pd.DataFrame], gap_hours=6, dist_km=400) -> List[pd.DataFrame]:
    if not tracks:
       return []

    # Sort tracks by start time
    tracks = sorted(tracks, key=lambda x: x['time'].iloc[0])
    stitched = []
    used = set()

    for i in range(len(tracks)):
        if i in used: continue
        current_track = tracks[i].copy()
        used.add(i)

        any_joined = True
        while any_joined:
            any_joined = False
            last_time = pd.to_datetime(current_track['time'].iloc[-1])
            last_lon = current_track['longitude'].iloc[-1]
            last_lat = current_track['latitude'].iloc[-1]

            for j in range(len(tracks)):
                if j in used: continue
                next_track = tracks[j]
                start_time = pd.to_datetime(next_track['time'].iloc[0])

                # Check time gap
                hour_gap = (start_time - last_time).total_seconds() / 3600
                if 0 < hour_gap <= gap_hours:
                    # Check distance gap
                    d = haversine(last_lon, last_lat, next_track['longitude'].iloc[0], next_track['latitude'].iloc[0])
                    if d < dist_km:
                        current_track = pd.concat([current_track, next_track]).drop_duplicates().sort_values('time')
                        used.add(j)
                        any_joined = True
                        break # Start looking again from the new end of current_track
        stitched.append(current_track)
    return stitched

# def consolidate_simultaneous_tracks(tracks: List[pd.DataFrame], dist_km=350) -> List[pd.DataFrame]:
#     if not tracks:
#        return []
#
#     final_tracks = []
#     used_indices = set()
#
#     for i in range(len(tracks)):
#         if i in used_indices: continue
#         base = tracks[i].copy()
#         used_indices.add(i)
#
#         for j in range(i + 1, len(tracks)):
#             if j in used_indices: continue
#             other = tracks[j]
#
#             # Check overlap in time
#             common_times = set(base['time']).intersection(set(other['time']))
#             if common_times:
#                 # Check distance during overlap
#                 overlap_dists = []
#                 for t in list(common_times)[:5]:
#                     p1 = base[base['time'] == t].iloc[0]
#                     p2 = other[other['time'] == t].iloc[0]
#                     overlap_dists.append(haversine(p1['longitude'], p1['latitude'], p2['longitude'], p2['latitude']))
#
#                 if np.mean(overlap_dists) < dist_km:
#                     # Merge: keep the point with lower pressure at each time step
#                     combined = pd.concat([base, other])
#                     base = combined.sort_values('msl').drop_duplicates('time').sort_values('time')
#                     used_indices.add(j)
#
#         final_tracks.append(base)
#     return final_tracks

def consolidate_simultaneous_tracks(tracks, dist_km=350, time_tolerance_hours=1):
    if not tracks: return []

    final_tracks = []
    used_indices = set()

    for i in range(len(tracks)):
        if i in used_indices: continue
        base = tracks[i].copy()
        used_indices.add(i)
        for j in range(i + 1, len(tracks)):
            if j in used_indices: continue
            other = tracks[j]
            # Find time overlap (with tolerance)
            base_times = sorted(list(set(pd.to_datetime(base['time']).dt.floor('h'))))
            other_times = sorted(list(set(pd.to_datetime(other['time']).dt.floor('h'))))
            # Check if time windows overlap
            base_time_range = (base_times[0], base_times[-1])
            other_time_range = (other_times[0], other_times[-1])
            time_overlap = (
                (base_time_range[0] <= other_time_range[1]) and 
                (other_time_range[0] <= base_time_range[1])
            )
            if time_overlap:
                # For overlapping times, check spatial distance
                overlap_times = set(base_times).intersection(set(other_times))
                if overlap_times:
                    overlap_dists = []
                    for t in list(overlap_times)[:10]:  # Sample check
                        base_subset = base[pd.to_datetime(base['time']).dt.floor('h') == t]
                        other_subset = other[pd.to_datetime(other['time']).dt.floor('h') == t]
                        if not base_subset.empty and not other_subset.empty:
                            p1 = base_subset.iloc[0]
                            p2 = other_subset.iloc[0]
                            d = haversine(
                                p1['longitude'],
                                p1['latitude'],
                                p2['longitude'],
                                p2['latitude']
                            )
                            overlap_dists.append(d)
                    if overlap_dists and np.mean(overlap_dists) < dist_km:
                        # Merge: keep lower pressure at each time
                        combined = pd.concat([base, other])
                        base = (combined
                                .sort_values(['time', 'msl'])
                                .drop_duplicates('time', keep='first')
                                .sort_values('time'))
                        used_indices.add(j)
        final_tracks.append(base)

    return final_tracks

def smooth_track_coordinates(
    track_df: pd.DataFrame,
    window_size: int = 5,
    iterations: int = 2
) -> pd.DataFrame:
    if len(track_df) < window_size:
        return track_df

    track_df = track_df.copy()

    # Check if required columns exist
    if 'latitude' not in track_df.columns or 'longitude' not in track_df.columns:
        # logger.error(" >  [ERROR] Track DataFrame is missing 'latitude' and/or 'longitude' columns")
        print(" >  [ERROR] Track DataFrame is missing 'latitude' and/or 'longitude' columns")
        return track_df

    # logger.info(f"Applying {window_size}-point rolling mean smoothing on track coordinates...")

    for _ in range(iterations):
        # Moving average
        lat_smooth = track_df['latitude'].rolling(
            window=window_size, 
            center=True, 
            min_periods=1
        ).mean()
        lon_smooth = track_df['longitude'].rolling(
            window=window_size, 
            center=True, 
            min_periods=1
        ).mean()

        # Apply smoothed values
        track_df['latitude'] = lat_smooth
        track_df['longitude'] = lon_smooth

    return track_df

def calculate_track_direction(track_data):
    if len(track_data) < 2:
        return None

    start_lat = track_data.iloc[0]["latitude"]
    start_lon = track_data.iloc[0]["longitude"]
    end_lat = track_data.iloc[-1]["latitude"]
    end_lon = track_data.iloc[-1]["longitude"]

    # Calculate bearing
    lat1, lon1 = np.radians(start_lat), np.radians(start_lon)
    lat2, lon2 = np.radians(end_lat), np.radians(end_lon)

    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360  # Normalize to 0-360

    return bearing

# def merge_nearby_tracks(tracks: List[pd.DataFrame], max_dist_km: float = 400) -> List[pd.DataFrame]:
#     if not tracks: return []
#
#     merged_list = []
#     used_indices = set()
#
#     for i in range(len(tracks)):
#         if i in used_indices: continue
#         base_track = tracks[i].copy()
#         used_indices.add(i)
#
#         for j in range(i + 1, len(tracks)):
#             if j in used_indices: continue
#             other_track = tracks[j]
#
#             # Find common timestamps
#             common_times = set(base_track['time']).intersection(set(other_track['time']))
#             if common_times:
#                 # Check spatial distance during overlap
#                 distances = []
#                 for t in common_times:
#                     p1 = base_track[base_track['time'] == t].iloc[0]
#                     p2 = other_track[other_track['time'] == t].iloc[0]
#                     distances.append(haversine(p1['longitude'], p1['latitude'], p2['longitude'], p2['latitude']))
#
#                 if np.mean(distances) < max_dist_km:
#                     # Combine and resolve overlaps by keeping the deeper low (min MSL)
#                     combined = pd.concat([base_track, other_track])
#                     base_track = combined.sort_values('msl').drop_duplicates('time').sort_values('time')
#                     used_indices.add(j)
#
#         merged_list.append(base_track)
#     return merged_list

# def remove_isolated_tracks(tracks, min_points=5, min_duration_hours=24):
#     filtered_tracks = []
#
#     for track in tracks:
#         # Basic length check
#         if len(track) < min_points:
#             continue
#
#         # Duration check
#         start_time = pd.to_datetime(track["time"].iloc[0])
#         end_time = pd.to_datetime(track["time"].iloc[-1])
#         duration_hours = (end_time - start_time).total_seconds() / 3600
#
#         if duration_hours < min_duration_hours:
#             continue
#
#         # Check if track has reasonable point density
#         time_diffs = []
#         for i in range(1, len(track)):
#             prev_time = pd.to_datetime(track.iloc[i-1]["time"])
#             curr_time = pd.to_datetime(track.iloc[i]["time"])
#             time_diff = (curr_time - prev_time).total_seconds() / 3600
#             time_diffs.append(time_diff)
#
#         # At least 60% of points should have reasonable time gaps (1-6 hours)
#         reasonable_gaps = sum(1 for td in time_diffs if 1 <= td <= 6)
#         gap_ratio = reasonable_gaps / len(time_diffs) if time_diffs else 0
#
#         if gap_ratio < 0.6:
#             continue
#
#         filtered_tracks.append(track)
#
#     return filtered_tracks

# def filter_by_spatial_clustering(tracks, min_cluster_size=3, max_distance_km=500):
#     if len(tracks) < 2:
#         return tracks
#
#     # Calculate mean position for each track
#     track_positions = []
#     for track in tracks:
#         mean_lat = track["latitude"].mean()
#         mean_lon = track["longitude"].mean()
#         track_positions.append((mean_lat, mean_lon))
#
#     # Find clusters using simple distance-based clustering
#     clusters = []
#     visited = [False] * len(track_positions)
#
#     for i in range(len(track_positions)):
#         if not visited[i]:
#             cluster = [i]
#             visited[i] = True
#
#             # Find all nearby tracks
#             for j in range(i+1, len(track_positions)):
#                 if not visited[j]:
#                     distance = haversine(
#                         track_positions[i][1], track_positions[i][0],
#                         track_positions[j][1], track_positions[j][0]
#                     )
#                     if distance < max_distance_km:
#                         cluster.append(j)
#                         visited[j] = True
#
#             clusters.append(cluster)
#
#     # Keep only tracks in clusters with minimum size
#     filtered_tracks = []
#     for cluster in clusters:
#         if len(cluster) >= min_cluster_size:
#             for idx in cluster:
#                 filtered_tracks.append(tracks[idx])
#
#     print(f" > Spatial clustering: {len(tracks)} -> {len(filtered_tracks)} tracks")
#     return filtered_tracks

# def filter_tracks_by_environment(
#     tracks,
#     require_ocean_start=True,
# ):
#     """Remove tracks that originate over land or never enter ocean"""
#     filtered = []
#     for track in tracks:
#         start_point = track.iloc[0]
#         is_land_start = is_point_land(start_point['latitude'], start_point['longitude'])
#         if require_ocean_start and is_land_start:
#             continue
#         # Check if track has any ocean points
#         has_ocean = False
#         for _, row in track.iterrows():
#             if not is_point_land(row['latitude'], row['longitude']):
#                 has_ocean = True
#                 break
#         if not has_ocean:
#             continue
#         filtered.append(track)
#
#     return filtered

def filter_valid_tracks(
    tracks,
    min_duration=MIN_TRACK_DURATION_HOURS,
    min_distance=MIN_NET_DISPLACEMENT_KM,
):
    """Filter tracks by duration, displacement, and direction"""
    valid_tracks = []
    if isinstance(tracks, dict):
        tracks_to_process = tracks.values()
    else:
        tracks_to_process = tracks

    for track in tracks_to_process:
        # Check minimum track duration
        duration = (track['time'].max() - track['time'].min()).total_seconds() / 3600
        if duration < min_duration:
            continue

        # Check minimum displacement
        total_dist = haversine(
            track.iloc[0]['longitude'],
            track.iloc[0]['latitude'],
            track.iloc[-1]['longitude'],
            track.iloc[-1]['latitude']
        )
        if total_dist < min_distance:
            continue

        # Check direction (WNW preferred)
        bearing = calculate_track_direction(track)
        if bearing and not (240 <= bearing <= 360 or 0 <= bearing <= 30):
            continue

        valid_tracks.append(track)
    return valid_tracks

def process_files(ds):
    try:
        time_coord = "time" if "time" in ds.coords else "valid_time"
        times = pd.to_datetime(ds[time_coord].values)
        time_diffs = np.diff(times).astype('timedelta64[s]').astype(float) / 3600

        print(f"\n >  [TEMPORAL CHECK] in process_files()")
        print(f"    Dataset time steps: {len(times)}")
        print(f"    Mean interval: {np.mean(time_diffs):.2f} hours")
        print(f"    Min interval: {np.min(time_diffs):.2f} hours")
        print(f"    Max interval: {np.max(time_diffs):.2f} hours")

        # Verify dataset has regular spacing
        if np.std(time_diffs) > 0.1:
            print(f" >  [WARNING] Irregular intervals detected")
            print(f"    Applying temporal fix...")
            ds = fix_temporal_resolution(ds, target_resolution_hours=1, method='interpolate')
            times = pd.to_datetime(ds[time_coord].values)
            time_diffs = np.diff(times).astype('timedelta64[s]').astype(float) / 3600
            print(f"    After fix: Mean interval = {np.mean(time_diffs):.2f} hours")

        required_vars = ["msl", "u10", "v10", "tp"]
        if not all(var in ds.data_vars for var in required_vars):
            print(f" >  [Error  ] Missing required variables")
            return None

        times = ds[time_coord].values
        print(f" >  [INFO   ] Time range: {times[0]} to {times[-1]}")
        print(f" >  [INFO   ] Number of time steps: {len(times)}")

        md_detections = []
        prev_center = None
        detection_count = 0

        print(f"\n >  [DETECTION PROGRESS]")
        total_steps = len(times)

        for i, time_step in enumerate(times):
            # Progress indicator
            if i % 100 == 0:
                print(f"    Processing time step {i+1}/{total_steps}...")

            data_slice = ds.sel(**{time_coord: time_step})
            # Find center
            center = find_low_pressure_center(
                data_slice,
                time_step,
                prev_center,
                search_radius_km=DISTANCE_THRESHOLD_KM
            )

            if center is not None:
                lat, lon = center
                md_detections.append({
                    "time": time_step,
                    "latitude": lat,
                    "longitude": lon,
                })
                prev_center = (lat, lon)  # Update for next time step
                detection_count += 1
            else:
                prev_center = None  # Reset if no detection

        print(f"\n >  [INFO   ] Total detections found: {detection_count} / {total_steps} time steps")
        print(f"    Detection rate: {100*detection_count/total_steps:.1f}%")

        if not md_detections:
            print(" >  [INFO   ] No MD centers detected")
            return None

        detections_df = pd.DataFrame(md_detections)
        detections_df["month"] = pd.to_datetime(detections_df["time"]).dt.strftime("%B")

        # Verify detection times are on the hour
        detection_hours = detections_df['time'].dt.hour.unique()
        print(f"    Detections occur at hours: {sorted(detection_hours)}")

        # Initial Tracking
        initial_tracks = track_monsoon_depressions(
            detections_df,
            max_speed_ms=15,  # Increased for MDs
            search_radius_km=DISTANCE_THRESHOLD_KM,  # for better linking
            max_time_gap_hours=MAX_GAP_ALLOWED_HOURS # gaps
        )
        print(f" >  [INFO    ] Initial raw tracks found: {len(initial_tracks)}")

        # Filter out very short tracks immediately
        initial_tracks = [t for t in initial_tracks if len(t) >= 5]
        print(f"    After removing very short tracks (<5 points): {len(initial_tracks)}")

        # Stitch sequential fragments (gaps)
        stitched_tracks = stitch_sequential_tracks(
            initial_tracks,
            gap_hours=MAX_GAP_HOURS,
            dist_km=STITCH_DISTANCE_KM
        )
        print(f" >  [INFO    ] Tracks after stitching gaps: {len(stitched_tracks)}")

        # Consolidate simultaneous/overlapping tracks (Spatial Merge)
        consolidated_tracks = consolidate_simultaneous_tracks(
            stitched_tracks,
            dist_km=MERGE_DISTANCE_KM
        )
        print(f" >  [INFO    ] Tracks after consolidating centers: {len(consolidated_tracks)}")

        # # Filter by environment
        # ocean_tracks = filter_tracks_by_environment(
        #     consolidated_tracks,
        #     require_ocean_start=True
        # )
        # print(f" >  [INFO    ] Tracks over ocean: {len(ocean_tracks)}")

        # Quality Filter tracks (Noise Removal)
        filtered_valid_tracks = filter_valid_tracks(
            consolidated_tracks,
            # ocean_tracks,
            min_duration=MIN_TRACK_DURATION_HOURS,
            min_distance=MIN_NET_DISPLACEMENT_KM
        )
        print(f" >  [INFO    ] Valid tracks after QC: {len(filtered_valid_tracks)}")
        if not filtered_valid_tracks:
            print(" >  [INFO    ] No tracks survived filtering criteria")
            return None

        # Smooth each track
        smoothed_tracks = []
        for track in filtered_valid_tracks:
            if len(track) > 5:
                smoothed_track = smooth_track_coordinates(
                    track,
                    window_size=5,
                    iterations=4
                )
                smoothed_tracks.append(smoothed_track)
            else:
                smoothed_tracks.append(track)

        # Assign track IDs
        track_df_with_ids = pd.concat(
            [track.assign(track_id=i+1) for i, track in enumerate(smoothed_tracks)],
            ignore_index=True
        )

        print(f"\n >  [FINAL   ] {len(smoothed_tracks)} tracks ready for analysis")
        return track_df_with_ids

    except Exception as e:
        print(f" > Error in process_files: {e}")
        # traceback.print_exc()
        return None

    finally:
        if 'ds' in locals():
            ds.close()


def plot_tracks(track_df, output_dir="output"):
    fig = plt.figure(figsize=(14, 9))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # type: ignore[assignment]
    ax.set_extent([70, 100, 5, 30], crs=ccrs.PlateCarree())

    # Map features
    # ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=6)
    # ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, linewidth=0.8, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle='--', alpha=0.7)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

    # Grid lines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    for track_plot_index, (track_id_str, track_data) in enumerate(track_df.groupby("track_id")):
        track_data = track_data.sort_values("time")
        start_point = track_data.iloc[0]
        end_point = track_data.iloc[-1]
        start_time = pd.to_datetime(start_point["time"]).strftime('%d/%m %H:%M')
        end_time = pd.to_datetime(end_point["time"]).strftime('%d/%m %H:%M')

        month = track_data["month"].iloc[0]
        color = month_colors.get(month, "black")

        # Plot all detection points
        plt.plot(
            track_data["longitude"],
            track_data["latitude"],
            "o",
            color=color,
            markersize=1,
            transform=ccrs.PlateCarree(),
            zorder=3
        )

        # Plot connecting lines
        plt.plot(
            track_data["longitude"],
            track_data["latitude"],
            "-",
            color=color,
            alpha=0.8,
            linewidth=1,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

        # Annotate Start point
        plt.annotate(
            start_time,
            xy=(start_point["longitude"], start_point["latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            fontweight="normal",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.7,
                edgecolor='none',
                linewidth=0.3
            ),
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

        # Annotate End point
        plt.annotate(
            end_time,
            xy=(end_point["longitude"], end_point["latitude"]),
            xytext=(-4, -4),
            textcoords="offset points",
            fontsize=6,
            fontweight="normal",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.05",
                facecolor="white",
                alpha=0.7,
                edgecolor='none',
                linewidth=0.3
            ),
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    legend_handles = []

    for month, color in month_colors.items():
        if month in track_df["month"].unique():
            (legend_line,) = plt.plot(
                [],
                [],
                "-",
                color=color,
                linewidth=1,
                label=f"Low-press. sys. Track ({month})",
            )
            legend_handles.append(legend_line)

    ax.set_title(
        f"Detected MD Tracks ({month_order[0]}-{month_order[-1]})",
        fontsize=13,
        fontweight='bold',
        pad=12
    )
    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
        handles=legend_handles,
        loc='lower left',
        fontsize=9,
        framealpha=0.95,
        edgecolor='black',
        fancybox=True,
        shadow=True
    )
    plt.tight_layout()

    output_filename = os.path.join(output_dir, "MD-track.png")
    print("\n----------------------- MD track plots ----------------------")
    plt.savefig(output_filename)
    print(f" >> [Saved   ] combined track plot as {output_filename}")
    # plt.show()
    plt.close()


def plot_track_full_map(
    track_df_single,
    track_id,
    full_extent,
    output_dir,
):
    fig = plt.figure(figsize=(12, 8))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # type: ignore[assignment]
    ax.set_extent(full_extent, crs=ccrs.PlateCarree())

    # Background features
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle='--')
    ax.add_feature(cfeature.RIVERS)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # only plot this track
    track_data = track_df_single.sort_values("time")
    month = track_data["month"].iloc[0]
    color = month_colors.get(month, "black")

    ax.plot(
        track_data["longitude"],
        track_data["latitude"],
        "-o",
        color=color,
        transform=ccrs.PlateCarree(),
        linewidth=1,
        markersize=1,
        zorder=5
    )

    # annotate start/end
    start = track_data.iloc[0]
    end = track_data.iloc[-1]
    start_label = pd.to_datetime(start["time"]).strftime('%d/%m %H:%M')
    end_label = pd.to_datetime(end["time"]).strftime('%d/%m %H:%M')
    ax.annotate(start_label,
                    xy=(start["longitude"],
                        start["latitude"]),
                    xytext=(5,5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7),
                    transform=ccrs.PlateCarree(),
                    zorder=6
                )
    ax.annotate(end_label,
                    xy=(end["longitude"],
                        end["latitude"]),
                    xytext=(5,-10),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7),
                    transform=ccrs.PlateCarree(),
                    zorder=6
                )
    ax.set_title(f"Track {track_id} on full dataset extent")
    outpath = os.path.join(
        output_dir,
        f"track_{int(track_id)}_full_map.png"
    )
    fig.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f" >> [Saved   ] Full-map for track {track_id}: {outpath}")

def plot_single_track(track_df, track_id, output_dir):
    track_data = track_df.sort_values("time")
    time_coord = 'time' if 'time' in track_df.columns else 'valid_time'
    if time_coord not in track_df.columns:
        time_coord = [col for col in track_df.columns if 'time' in col.lower()][0]

    start_time_str = pd.to_datetime(track_df[time_coord].iloc[0]).strftime('%d/%m/%Y %H:%M')
    end_time_str = pd.to_datetime(track_df[time_coord].iloc[-1]).strftime('%d/%m/%Y %H:%M')

    fig = plt.figure(figsize=(13, 10))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # type: ignore[assignment]

    min_lon, max_lon = track_df["longitude"].min() - 4, track_df["longitude"].max() + 4
    min_lat, max_lat = track_df["latitude"].min() - 4, track_df["latitude"].max() + 4

    month = track_data["month"].iloc[0]
    color = month_colors.get(month, "black")

    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Add map features
    # ax.coastlines(resolution='50m')
    ax.coastlines(resolution='50m', linewidth=1.0)
    # ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    # ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, linewidth=0.6, zorder=6)
    # ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle='--')
    ax.add_feature(cfeature.STATES, linewidth=0.4, linestyle='--', alpha=0.6)
    # ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.RIVERS, linewidth=0.4, alpha=0.5)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    for track_plot_index, (track_id_str, track_data) in enumerate(track_df.groupby("track_id")):
        track_data = track_data.sort_values("time")
        start_point = track_data.iloc[0]
        end_point = track_data.iloc[-1]
        start_time = pd.to_datetime(start_point["time"])
        end_time = pd.to_datetime(end_point["time"])
        start_label = start_time.strftime("%d/%m %H:%M")
        end_label = end_time.strftime("%d/%m %H:%M")

        month = track_data["month"].iloc[0]
        color = month_colors.get(month, "black")

        # Plot all detection points
        ax.plot(
            track_data["longitude"],
            track_data["latitude"],
            "o",
            color=color,
            markersize=1,
            transform=ccrs.PlateCarree(),
            zorder=3
        )

        # Plot connecting lines
        ax.plot(
            track_data["longitude"],
            track_data["latitude"],
            "-",
            color=color,
            alpha=0.8,
            linewidth=1,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

        # Annotate Start point
        plt.annotate(
            start_label,
            xy=(start_point["longitude"], start_point["latitude"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="normal",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.05",
                facecolor="white",
                alpha=0.7,
                edgecolor='none'
            ),
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

        # Annotate End point
        plt.annotate(
            end_label,
            xy=(end_point["longitude"], end_point["latitude"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="normal",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.05",
                facecolor="white",
                alpha=0.7,
                edgecolor='none'
            ),
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    ax.plot(
        track_df["longitude"],
        track_df["latitude"],
        color=color,
        linewidth=2,
        marker='o',
        markersize=1,
        transform=ccrs.PlateCarree(),
        label=f'Track {track_id}',
        zorder=3,
        alpha=0.85
    )

    # Start point (green circle)
    ax.plot(
        track_df["longitude"].iloc[0],
        track_df["latitude"].iloc[0],
        'o',
        color='green',
        markersize=1,
        transform=ccrs.PlateCarree(),
        label='Start',
        zorder=4
    )
    # End point (red X)
    ax.plot(
        track_df["longitude"].iloc[-1],
        track_df["latitude"].iloc[-1],
        'x',
        color='red',
        markersize=1,
        markeredgewidth=2,
        transform=ccrs.PlateCarree(),
        label='End',
        zorder=4
    )

    # Add landfall indicator if applicable
    if 'is_land' in track_df.columns:
        land_status = track_df['is_land'].values
        transitions = np.diff(land_status.astype(int))
        ocean_to_land = np.where(transitions == 1)[0]
        if len(ocean_to_land) > 0:
            landfall_idx = ocean_to_land[0] + 1
            if landfall_idx < len(track_df):
                landfall_point = track_df.iloc[landfall_idx]
                ax.plot(
                    landfall_point["longitude"],
                    landfall_point["latitude"],
                    's',
                    color='purple',
                    markersize=8,
                    markeredgecolor='indigo',
                    markeredgewidth=1.5,
                    transform=ccrs.PlateCarree(),
                    label='Landfall',
                    zorder=5
                )

    ax.set_title(
        f"Monsoon Depression Track {track_id}\n{start_time_str} to {end_time_str}",
        fontsize=12,
        fontweight='bold',
        pad=10
    )
    ax.legend(
        loc='upper left',
        fontsize=9,
        framealpha=0.95,
        edgecolor='black',
        fancybox=True
    )
    plt.tight_layout()

    track_id_str = str(track_id).replace('/', '_').replace(' ', '_')
    file_name = f"MD-track-{track_id_str}-detailed.png"
    fullpath = os.path.join(output_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(fullpath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f" >> [Saved   ] Detailed track {track_id_str} plot: {fullpath}")

def export_track_coordinates(track_df, output_dir):
    print("\n------------------------------------------------------------")
    if track_df.empty:
        print(" >  [Warning ] No track coordinates to export.")
        return

    export_df = track_df[['time', 'latitude', 'longitude', 'track_id']].copy()
    export_df['time'] = pd.to_datetime(export_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    output_filename = os.path.join(
        output_dir, f"MD-coordinates-ALL.csv"
    )
    export_df.to_csv(output_filename, index=False)
    print(f" >> [Exported] Coordinates for {len(track_df)} detections exported to {output_filename}")


def export_single_track_coordinates(track_df, track_id, output_dir):
    export_df = track_df[['time', 'latitude', 'longitude', 'track_id']].copy()
    export_df['time'] = pd.to_datetime(export_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    track_id_str = str(track_id).replace('/', '_').replace(' ', '_')
    file_name = f"MD-track-{track_id_str}-coordinates.csv"
    export_df.to_csv(os.path.join(output_dir, file_name), index=False)
    print(f" >> [Exported] Track {track_id} to: {file_name}")


def load_dataset_async(month_dirs, data_dir, result_container):
    file_list = []
    monthly_stats = {}

    for month_dir in month_dirs:
        month_path = os.path.join(data_dir, month_dir)
        if os.path.isdir(month_path):
            # file_list.extend([
            #     os.path.join(month_path, f)
            #     for f in os.listdir(month_path) if f.endswith(".nc")
            # ])
            month_files = [os.path.join(month_path, f) 
                          for f in os.listdir(month_path) if f.endswith(".nc")]
            file_list.extend(month_files)
            monthly_stats[month_dir] = {
                'n_files': len(month_files),
                'files': month_files[:3]  # First 3 files for checking
            }

    if not file_list:
        result_container["error"] = "No netCDF files found in the specified directories."
        result_container["success"] = False
        return

    try:
        ds = xr.open_mfdataset(
            file_list,
            combine="by_coords",
            data_vars='all',
            coords='all',  # type: ignore[arg-type]
            compat='override',
            join='outer'
        ).compute()

        if 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})

        ds = ds.sortby('time')

        result_container["ds"] = ds
        result_container["success"] = True
    except Exception as e:
        result_container["error"] = str(e)
        result_container["success"] = False


def load_dataset_with_resolution(month_dirs, data_dir, resolution_hours, result_container):
    file_list = []
    monthly_stats = {}

    for month_dir in month_dirs:
        month_path = os.path.join(data_dir, month_dir)
        if os.path.isdir(month_path):
            month_files = [os.path.join(month_path, f) 
                          for f in os.listdir(month_path) if f.endswith(".nc")]
            file_list.extend(month_files)
            monthly_stats[month_dir] = len(month_files)

    if not file_list:
        result_container["error"] = "No netCDF files found in the specified directories."
        result_container["success"] = False
        return

    try:
        print(f"\n > Loading {len(file_list)} files from {len(month_dirs)} months...")

        ds = xr.open_mfdataset(
            file_list,
            combine="by_coords",
            data_vars='all',
            coords='all',  # type: ignore[arg-type]
            compat='override',
            join='outer'
        ).compute()

        # Standardize time coordinate
        if 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})

        ds = ds.sortby('time')

        # Resample to target resolution
        if resolution_hours != 1:
            ds = resample_dataset_to_resolution(ds, resolution_hours)

        result_container["ds"] = ds
        result_container["resolution_hours"] = resolution_hours
        result_container["success"] = True
        result_container["monthly_stats"] = monthly_stats

    except Exception as e:
        result_container["error"] = str(e)
        result_container["success"] = False

def load_and_process_all_resolutions(
    month_dirs,
    data_dir,
    output_base_dir="output",
):
    all_results = {}
    for res_key, config in RESOLUTION_CONFIGS.items():
        resolution_hours = config['hours']
        output_dir = f"{output_base_dir}{config['output_dir_suffix']}"

        print(f" > [PROCESSING] {config['description']}")
        os.makedirs(output_dir, exist_ok=True)
        result_container = {"ds": None, "success": False, "error": None}

        load_thread = threading.Thread(
            target=load_dataset_with_resolution,
            args=(month_dirs, data_dir, resolution_hours, result_container)
        )
        load_thread.daemon = False
        load_thread.start()

        # Spinner while loading
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0

        while load_thread.is_alive():
            sys.stdout.write(f"\r {spinner[spinner_idx]} Loading {resolution_hours}-hourly data...")
            sys.stdout.flush()
            spinner_idx = (spinner_idx + 1) % len(spinner)
            time.sleep(0.1)

        load_thread.join()

        if result_container["success"]:
            print(f"\r >  Successfully loaded {resolution_hours}-hourly dataset")
            all_results[res_key] = {
                'ds': result_container['ds'],
                'output_dir': output_dir,
                'config': config,
                'resolution_hours': resolution_hours,
                'success': True
            }
        else:
            print(f"\r >  Failed to load {resolution_hours}-hourly data: {result_container['error']}")
            all_results[res_key] = {'success': False, 'error': result_container['error']}


def diagnose_detection_issues(ds, track_df):
    """
    Checks for:
    - Missing detections (gaps in continuous pressure)
    - False detections (weak systems)
    - Tracking jumps
    """
    print("--- Detection Quality Diagnostics ---")
    # 1. Check for large gaps in time
    times = pd.to_datetime(track_df['time'])
    time_diffs = times.diff().dt.total_seconds().dropna() / 3600

    large_gaps = time_diffs[time_diffs > 6]
    if len(large_gaps) > 0:
        print(f" > Found {len(large_gaps)} large time gaps (>6 hours)")
        print(f" > Max gap: {time_diffs.max():.1f} hours")
    else:
        print(" > No large time gaps detected")

    # 2. Check for sudden position jumps
    if 'distance_km' in track_df.columns:
        large_jumps = track_df[track_df['distance_km'] > 150]
        if len(large_jumps) > 0:
            print(f" >  Found {len(large_jumps)} large position jumps (>150 km)")
            print(f" >  Max jump: {track_df['distance_km'].max():.1f} km")
        else:
            print(" >  No suspicious position jumps")

    # 3. Check speed consistency
    if 'speed_kmh' in track_df.columns:
        speeds = track_df['speed_kmh'].dropna()
        if len(speeds) > 0:
            unrealistic = speeds[(speeds < 3) | (speeds > 30)]
            if len(unrealistic) > 0:
                print(f" >  Found {len(unrealistic)} points with unrealistic speeds")
                print(f" >  Range: {speeds.min():.1f} - {speeds.max():.1f} km/h")
            else:
                print(" >  All speeds within reasonable range")

    # 4. Check track direction consistency
    if len(track_df) >= 3:
        bearings = []
        for i in range(len(track_df) - 1):
            lat1, lon1 = track_df.iloc[i][['latitude', 'longitude']]
            lat2, lon2 = track_df.iloc[i+1][['latitude', 'longitude']]

            # Calculate bearing
            dlon = np.deg2rad(lon2 - lon1)
            lat1_rad = np.deg2rad(lat1)
            lat2_rad = np.deg2rad(lat2)

            x = np.sin(dlon) * np.cos(lat2_rad)
            y = (np.cos(lat1_rad) * np.sin(lat2_rad) - 
                 np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon))

            bearing = np.degrees(np.arctan2(x, y))
            bearing = (bearing + 360) % 360
            bearings.append(bearing)

        bearings = np.array(bearings)

        # MDs typically move WNW (240-330 degrees)
        wnd_bearings = bearings[(bearings >= 240) & (bearings <= 360) |
                                (bearings >= 0) & (bearings <= 30)]

        if len(wnd_bearings) / len(bearings) < 0.7:
            print(f" >  Only {100*len(wnd_bearings)/len(bearings):.1f}% of "
                  f"movement is WNW (expected >70%)")
        else:
            print(f" >  Track direction consistent with MD climatology "
                  f"({100*len(wnd_bearings)/len(bearings):.1f}% WNW)")


if __name__ == "__main__":
    start_time = time.time()
    print("\n--------------------- Applied Thresholds --------------------")
    print(f"  Distance          : {DISTANCE_THRESHOLD_KM} km")
    print(
        f"  Wind Speed (10m)  : {WIND_SPEED_LOWER_BOUND_MS} to {WIND_SPEED_UPPER_BOUND_MS} m/s"
    )
    print(f"  MSLP              : < {MSL_LOWER_BOUND_PA} Pa (or {MSL_LOWER_BOUND_PA / 100:.0f} hPa)")
    print(f"  Precipitation     : > {PRECIPITATION_THRESHOLD_MM} mm/hr")
    print(f"  Min Track Duration: {MIN_TRACK_DURATION_HOURS} hours")
    print(f"  Min Displacement  : {MIN_NET_DISPLACEMENT_KM} km")

    print("\n------------- Combining monthly netCDF files ----------------")
    print(" >  [INFO    ] Processing all temporal resolutions...")

    for res_key, config in RESOLUTION_CONFIGS.items():
        resolution_hours = config['hours']
        output_dir = f"output{config['output_dir_suffix']}"

        print(f" > [PROCESSING] {config['description']}")

        os.makedirs(output_dir, exist_ok=True)

        # Load dataset for this resolution
        result_container = {"ds": None, "success": False, "error": None}

        load_thread = threading.Thread(
            target=load_dataset_with_resolution,
            args=(MONTH_DIRS, DATA_DIR, resolution_hours, result_container)
        )
        load_thread.daemon = False
        load_thread.start()

        # Spinner while loading
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0

        while load_thread.is_alive():
            sys.stdout.write(f"\r {spinner[spinner_idx]} Loading {resolution_hours}-hourly data...")
            sys.stdout.flush()
            spinner_idx = (spinner_idx + 1) % len(spinner)
            time.sleep(0.1)
        load_thread.join()

        if not result_container["success"]:
            print(f"\r >  [Error    ] Failed to load {resolution_hours}-hourly data: {result_container['error']}")
            continue

        ds = result_container["ds"]
        print(f"\r >  [INFO    ] Successfully loaded {resolution_hours}-hourly dataset")

        # Diagnose temporal issues
        print("\n [TEMPORAL DIAGNOSTICS]")
        diag_report = diagnose_temporal_issues(ds)

        # Apply temporal fix if needed
        time_coord = "time" if "time" in ds.coords else "valid_time"
        times = pd.to_datetime(ds[time_coord].values)
        time_diffs = np.diff(times).astype('timedelta64[s]').astype(float) / 3600

        if np.std(time_diffs) > 0.1:
            print("\n [APPLYING TEMPORAL FIX]")
            print(" >  Original dataset has variable time spacing")
            print(f" >  Resampling to regular {resolution_hours}-hour grid...")

            ds = fix_temporal_resolution(ds, target_resolution_hours=resolution_hours, method='interpolate')

            is_valid, val_report = validate_temporal_consistency(ds, expected_hours=resolution_hours, tolerance=0.05)

            if not is_valid:
                print("\n >  [Warning] Some intervals still outside tolerance")
            else:
                print(f"\n > [INFO   ] Temporal resolution successfully regularized to {resolution_hours}h")

        # Dataset info
        print("\n--------------------- Dataset Info --------------------------")
        years = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%Y").unique()
        months = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%b").unique()
        ds_lat_min = ds.latitude.min().item()
        ds_lat_max = ds.latitude.max().item()
        ds_lon_min = ds.longitude.min().item()
        ds_lon_max = ds.longitude.max().item()

        time_values = ds[time_coord].values
        time_diff = time_values[1] - time_values[0]
        time_diff = pd.to_timedelta(time_diff)
        actual_resolution_hours = time_diff.total_seconds() / 3600.0

        print("  Year      :", years)
        print("  Months    :", months)
        print(f"  Dataset bounds: lat [{ds_lat_min:.1f}, {ds_lat_max:.1f}] N, lon [{ds_lon_min:.1f}, {ds_lon_max:.1f}] E")
        print(f"  Actual Time Resolution: {actual_resolution_hours:.1f} hours ({time_diff})")

        # Process files
        print("\n---------------------- Processing ---------------------------")
        full_track_df = process_files(ds)

        if full_track_df is not None and not full_track_df.empty:
            if 'track_id' not in full_track_df.columns:
                print(" >> [Error   ] 'track_id' column not found")
                continue

            # Plot tracks
            plot_tracks(full_track_df, output_dir=output_dir)

            # Process each track
            pad = 0.5
            try:
                full_extent = [
                    ds_lon_min - pad,
                    ds_lon_max + pad,
                    ds_lat_min - pad,
                    ds_lat_max + pad,
                ]
            except Exception:
                full_extent = [70, 100, 5, 30]

            unique_track_ids = sorted(full_track_df['track_id'].unique())
            print(f"\n >  [INFO    ] Found {len(unique_track_ids)} MD tracks for {resolution_hours}-hourly resolution.")

            # Process each track
            for track_id in unique_track_ids:
                track_dir = os.path.join(output_dir, f"track_{int(track_id)}")
                os.makedirs(track_dir, exist_ok=True)

                track_df = full_track_df[full_track_df['track_id'] == track_id].copy().reset_index(drop=True)

                # Land/Ocean classification
                track_df['is_land'] = track_df.apply(
                    lambda row: is_point_land(row['latitude'], row['longitude']),
                    axis=1
                )

                # Smooth coordinates
                track_df = smooth_track_coordinates(track_df, window_size=3)

                print(f"\n------------------------ MD Track {track_id} -------------------------")
                # print(f" >> [Diagnostics] for MD track {track_id}")
                # diagnose_detection_issues(ds, track_df)

                # Plot maps
                plot_track_full_map(track_df, track_id, full_extent, track_dir)
                plot_single_track(track_df, track_id, track_dir)

            #     # Export coordinates
            #     export_single_track_coordinates(track_df, track_id, track_dir)
            #
            #     # Export detections csv
            #     track_csv = os.path.join(track_dir, f"track_{int(track_id)}_detections.csv")
            #     track_df.to_csv(track_csv, index=False)
            #     print(f" >> [Exported] Track detections CSV: {track_csv}")
            #
            #     # Speed analysis
            #     print(f" >  [INFO    ] Computing translation speed for Track {track_id}")
            #     speed_dir = os.path.join(track_dir, "translational-speed")
            #     os.makedirs(speed_dir, exist_ok=True)
            #     track_df = compute_track_speeds(track_df, smoothing_sigma=2.0)
            #     track_csv = os.path.join(speed_dir, f"track_{int(track_id)}_detections_augmented.csv")
            #     track_df.to_csv(track_csv, index=False)
            #     print(f" >> [Saved   ] Augmented track data: {track_csv}")
            #     plot_speed_analysis(track_df, track_id, speed_dir)
            #     print(f" >  Speed Statistics for Track {track_id}:")
            #     print(f"      Mean speed: {track_df['speed_kmh'].mean():.1f} km/h")
            #     print(f"      Median speed: {track_df['speed_kmh'].median():.1f} km/h")
            #
            #     # Hovmoller plots
            #     print("\n------------------ Azimuthal Hovmoller Plots ----------------")
            #     print(f" >  [INFO    ] Computing Hovmoller for Track {track_id}")
            #     hov_dir = os.path.join(track_dir, "hovmoller")
            #     os.makedirs(hov_dir, exist_ok=True)
            #     hov_data, track_times, landfall_index = compute_azimuthal_hovmoller(
            #         ds=ds,
            #         track_df=track_df,
            #         radii_km=radii,
            #         variables=variables,
            #         n_theta=72
            #     )
            #     for var in variables:
            #         print(f" >  [Plotting] Hovmoller for {var}")
            #         plot_hovmoller(
            #             hov_arr=hov_data[var],
            #             track_times=track_times,
            #             radii_km=radii,
            #             var_name=var,
            #             track_df=track_df,
            #             outdir=hov_dir,
            #             landfall_index=landfall_index
            #         )
            #
            #     # Correlation analysis
            #     print("\n------------------- Correlation analysis --------------------")
            #     print(f" >  [INFO    ] Correlation analysis for Track {track_id}")
            #     correlation_plots_dir = os.path.join(track_dir, "correlation_plots")
            #     os.makedirs(correlation_plots_dir, exist_ok=True)
            #     results = analyze_single_track(
            #         ds, track_df, track_id,
            #         buffer_km=100,
            #         max_lag_hours=24,
            #         output_dir=correlation_plots_dir
            #     )
            #     if results:
            #         print(f" >  Correlation Results for Track {track_id}:")
            #         print(f"      Pearson r: {results['correlation']:.3f}")
            #         print(f"      p-value: {results['p_value']:.4f}")
            #         print(f"      Samples: {results['n_samples']}")
            #         print(f"      Mean TP: {results['mean_tp']:.2f} mm")
            #
            # # Export all coordinates
            # export_track_coordinates(full_track_df, output_dir)

        else:
            print(f" >  [INFO   ] No tracks found for {resolution_hours}-hourly resolution")

        # Clean up dataset
        ds.close()
        print(f"\n >> Completed processing for {resolution_hours}-hourly resolution")

    end_time = time.time()
    print(f"\n > All resolutions executed in {end_time - start_time:.2f} seconds.\n")
