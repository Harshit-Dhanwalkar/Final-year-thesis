# FIX:
# 0.1. add label for duration in detailed plots
# 0.2. need to replace 'run_all_statistical_analyses', instaed just call individual functions
# 1. No Memory Management for Large Datasets
#    Location: md_detection_tracking.py, load_dataset_async()
#    Issue   : .compute() loads entire dataset into memory
#              .compute() # Could cause OOM for very large datasets
# 2. Ocean-land shading to incorporate landfall line
# 3. Inconsistent Landfall Detection
#    Location: Multiple files (antecedent_analysis.py, correlation_analysis.py)
#    Issue   : Different logic for detecting landfall in different functions
#    antecedent_analysis.py: is_point_land()
#    correlation_analysis.py: row.get('is_land', False)
#    Fix     : Standardize to use pre-computed 'is_land' column
# 4. Inconsistent Parameter Names
#    Location: Multiple functions use buffer_km vs buffer_deg
#    antecedent_analysis.py uses buffer_km=100
#    plot_sm_tp_map() uses buffer_deg=5.0
# 5. Rephase CE 0.5/0.3 vs 8.0/3.0" → Standardize (use 8-15)
# TODO:
# 1. Add mosac of all tiles in Data/HGT-files
# 2. H4 GAM: pip install pygam; SM ~ s(TP) → Plateau fig
# 3. H5 HWSD:
#    ds = xr.open_dataset('SoilGrids250m_CLAY.nc')
#    tracks['CLAY_%'] = ds.CLAY.isel(depth=0).interp(lon=tracks.lon,lat=tracks.lat)
#    # FSI ~ CLAY_% (r>0.6 expected)
# 4. H7 SRTM:
#    !pip install richdem rioxarray
#    dem = rioxarray.open_rasterio('SRTM_India.tif')
#    tracks['MEAN_ELEV'] = dem.interp(...)  # r(dur,elev)<-0.7
# 5. Add duirnal HF (LHF, SHF and THF) amplitude time series plot -> \Delta(HF) = (HF_max -HF_min)_24h -> see veriation in this \Delta(HF)
# 6. Add land-surface-mask (Data/nc-files/lsm.nc)
# 7. from utils.debug_utils.py
#      # At the start of your main section
#      debug_logger = DebugLogger(output_dir)
# 8. Before each major analysis section
#    debug_logger.log_function_call("process_files", args=(ds))
#    debug_logger.log_data_shape("combined_dataset", ds)
#    # After processing tracks
#    if full_track_df is not None:
#       debug_logger.log_data_shape("full_track_df", full_track_df)
#       debug_logger.log_warning(f"Number of tracks: {len(full_track_df['track_id'].unique())}")
# 9. In each track loop
#    for track_id in unique_track_ids:
#       debug_logger.log_function_call(f"process_track_{track_id}")
#       # ... code ...
#       # After each analysis
#       debug_logger.log_data_shape(f"track_{track_id}_correlation", pd.DataFrame([correlation_result]) if correlation_result else None)
#    # At the end, generate diagnostic report
#    all_data = {
#        'full_track_df': full_track_df,
#        'all_correlation_results': all_correlation_results,
#        'all_flux_results': all_flux_results,
#        'all_landfall_results': all_landfall_results,
#        'all_classifications': all_classifications,
#        'all_asm_results': all_asm_results,
#        'all_memory_results': all_memory_results,
#        'fsi_df': fsi_df,
#        'amplitude_df': amplitude_df,
#        'coupling_efficiency_df': coupling_efficiency_df
#    }
#    diagnostic_report = generate_diagnostic_report(all_data, output_dir)


# INFO: For 2020, 2021, 2022, 2023, 2024, 2025 expect ->  (FSI > 0.4 long; clay > 30%, ACF > 0.35).

import os
import sys
import threading
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

import traceback

from utils.progress_utils import ProgressSpinner, ProgressBar
from utils import (
    std_print,
    is_point_land,
    find_landfall_time,
    setup_output_dir,
    export_results_to_csv,
    find_origin_type,
    calculate_track_curvature,
    classify_track_status,
    extract_spatial_subset,
    get_default_buffer_km,
    calculate_buffer_deg,
    extract_and_average_variable,
    clamp_to_domain,
    haversine_distance,
    get_time_coord,
    setup_map_extent,
    add_land_ocean_shading,
    common_bbox,
    legend_style,
)
from azimuthal_hovmoller import (
    compute_azimuthal_hovmoller_sectors,
    average_azimuthally,
    plot_hovmoller_azimuthal_average,
)
from speed_analysis import (
    compute_track_speeds,
    plot_speed_analysis_kmh,
    plot_speed_analysis_ms,
)
from antecedent_analysis import (
    plot_antecedent_spatial_map,
    analyze_antecedent_correlations,
    analyze_antecedent_trend,
    analyze_antecedent_for_all_tracks,
)
from correlation_analysis import (
    analyze_single_track,
    compute_single_track_correlation,
    analyze_all_tracks,
)
from elevation_track_vis import create_elevation_vrt, plot_tracks_on_elevation
from elevation_analysis import run_elevation_analysis
from statistics_analysis import (
    analyze_asm_fsi_relationship,
    analyze_gam_nonlinearity,
    test_granger_causality,
    analyze_tsi,
    load_elevation_metrics,
    run_all_statistical_analyses,
)
from soilmoiture_analysis import (
    plot_spatial_tp_swvl1,
    compute_soil_moisture_memory,
    analyze_soil_memory_by_track,
)
from heatflux_analysis import (
    classify_md_duration,
    analyze_heat_flux_evolution,
    analyze_flux_transition,
    export_heat_flux_results,
    export_landfall_flux_results,
    print_md_classification_summary,
)
from coupling_metrics import (
    compute_fsi_all_tracks,
    compute_diurnal_amplitude_all_tracks,
    compute_coupling_efficiency,
    export_coupling_metrics,
    generate_coupling_metrics_report,
)

from timeseries import tp_swvl1_timeseries, extract_heat_flux_timeseries, hf_timeseries

DATA_DIR = os.path.join("..", "Data", "single-levels", "ncfiles")
MONSOON_SEASONS = {
    2019: [
        "June-2019",
        "July-2019",
        "August-2019",
        "September-2019",
    ],
    2020: [
        "June-2020",
        "July-2020",
        "August-2020",
        "September-2020",
    ],
    2021: [
        "June-2021",
        "July-2021",
        "August-2021",
        "September-2021",
    ],
    2022: [
        "June-2022",
        "July-2022",
        "August-2022",
        "September-2022",
    ],
    2023: [
        "June-2023",
        "July-2023",
        "August-2023",
        "September-2023",
    ],
    2024: [
        "June-2024",
        "July-2024",
        "August-2024",
        "September-2024",
    ],
    2025: [
        # "June-2025",
        "July-2025",
        "August-2025",
        "September-2025",
    ],
}

MONSOON_YEAR = 2024  # Available : [2019, 2020, 2021, 2022, 2023, 2024, 2025]
MONTH_DIRS = MONSOON_SEASONS.get(MONSOON_YEAR, [])
output_dir = f"output_{MONSOON_YEAR}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    std_print("INFO", "md", f"Created output directory for year {MONSOON_YEAR}", output_dir)
else:
    std_print("INFO", "md", f"Using existing `{output_dir}` as output directory for year {MONSOON_YEAR}")

# Use clipped domain instead of dataset bounds (10N 70E - 30N 95E)
clipped_min_lat = 7.0  # N
clipped_max_lat = 30.0  # N
clipped_min_lon = 70.0  # E
clipped_max_lon = 95.0  # E

month_colors = {
    "June": "#FF6B6B",
    "July": "#FF4500",
    "August": "#1E90FF",
    "September": "#9932CC",
}
month_order = [
    "June",
    "July",
    "August",
    "September",
]

# TEST:
# MIN_TRACK_DURATION_HOURS = 18  # Reduce from 24
# MIN_NET_DISPLACEMENT_KM = 200  # Reduce from 250
# WIND_SPEED_LOWER_BOUND_MS = 4  # Reduce from 5
# For more realistic MD rainfall (moderate to heavy rain):
# PRECIPITATION_THRESHOLD_MM = 1.0  # 1 mm/hr = 24 mm/day (moderate rain)
# # Or for stricter criteria:
# PRECIPITATION_THRESHOLD_MM = 2.0  # 2 mm/hr = 48 mm/day (heavy rain)
# # If using daily data, convert directly:
# PRECIPITATION_THRESHOLD_DAILY_MM = 15.0  # 15 mm/day
# PRECIPITATION_THRESHOLD_MM = PRECIPITATION_THRESHOLD_DAILY_MM / 24  # Convert to hourly

# for wind creterion : https://mausam.imd.gov.in/imd_latest/monsoonreport2024.pdf page: 284
WIND_SPEED_LOWER_BOUND_MS = 5  # m/s == 17 knots
WIND_SPEED_UPPER_BOUND_MS = 19  # m/s == 27 knots
MSL_LOWER_BOUND_PA = 100000  # Pa
PRECIPITATION_THRESHOLD_MM = 0.1  # mm/hr
# SWVL1 Range: $0.02-0.55$ m^3/m^3
VORTICITY_THRESHOLD = 3e-6  # 5e-6    # > 0 /s, # Positive cyclonic
MAX_SPEED_MS = 12  # m/s  (36 km/hr Max MD centre translational-speed)
MAX_GAP_HOURS = 24  # hr Bridge small gaps in Detection
TRACKING_SEARCH_RADIUS_KM = 50  # km Hourly tracking
LINKING_SEARCH_RADIUS_KM = 100  # km Multi-hour gaps
MIN_TRACK_DURATION_HOURS = 24
MIN_NET_DISPLACEMENT_KM = 250  # km MD travel distance
STITCH_DISTANCE_KM = 450  # km For gap bridging (MD spped 15–25 km/h, 24 hours, the system can easily travel 23 x 15 = 360 km.)
MERGE_DISTANCE_KM = (
    250  # km For simultaneous systems, distance to consider two detections as same MD
)
BUFFER = 10.0  # degrees
# Constants
EARTH_R = 6371.0  # km

# Hovmoller
radii = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]
variables = [
    "wind10",  # wind10 = sqrt(u10^2 + v10^2)
    "vr10",  # Radial Wind
    "vt10",  # Tangential Wind
    "vor_rel",  # Relative Vorticity
    "vor_abs",  # Absolute Vorticity
    # 'speed_kmh',
    "speed_ms",  # MD center transitional speed in m/s
    "msl",  # mean sea level pressure
    "sp",  # surface pressure
    "tp",  # total precipitation
    "t2m",  # 2m temperature
    "stl1",  # soil temperature level 1
    "swvl1",  # soil water volume level 1
    "avg_slhtf",  # surface latent heat flux
    "avg_ishf",  # surface sensible heat flux
    "total_hf",  # Total surface heat flux
]
VARIABLE_BUFFERS = {
    "u10": 150,  # Wind field (MD circulation)
    "v10": 150,  # Wind field (MD circulation)
    "msl": 100,  # Mean sea level pressure (pressure gradient)
    "tp": 150,  # Total precipitation (MD extent)
    "t2m": 100,  # 2m temperature
    "swvl1": 100,  # Soil water volume layer 1
    "avg_slhtf": 150,  # Surface latent heat flux
    "avg_ishf": 150,  # Integrated sensible heat flux
}


# Helper function
def check_statistical_requirements(coupling_df, asm_df):
    """Check if data meets minimum requirements for statistical tests"""
    checks = {
        "asm_fsi": len(asm_df) >= 5 and "FSI" in coupling_df.columns,  # for H3
        "gam": len(coupling_df) >= 10 and "FSI" in coupling_df.columns,  # for H4
        "granger": len(coupling_df) >= 20
        and ("TP" in coupling_df.columns or "mean_tp" in coupling_df.columns),  # for H5
        "tsi": len(coupling_df) >= 5,  # for H7
    }
    for test, passed in checks.items():
        if not passed:
            std_print("WARN", "stats", f"{test} requirements not met", "Skipping test")
    return checks


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
    u_subset = u_data.sel(
        latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
    )
    v_subset = v_data.sel(
        latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
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
    """
    Compute relative vorticity using finite differences. This uses local grid points (not spatial averaging) because vorticity calculation requires spatial gradients (dv/dx - du/dy). The local wind field structure is essential for vorticity calculation.
    """
    # Get grid dimensions
    n_lat, n_lon = len(ds.latitude), len(ds.longitude)
    # Get u, v at the point
    u, v = (
        float(ds["u10"].isel(latitude=lat_idx, longitude=lon_idx).values),
        float(ds["v10"].isel(latitude=lat_idx, longitude=lon_idx).values),
    )

    # Get grid spacing
    lat = float(ds.latitude.values[lat_idx])
    dlat, dlon = (
        np.abs(ds.latitude.values[1] - ds.latitude.values[0]),
        np.abs(ds.longitude.values[1] - ds.longitude.values[0]),
    )

    # Convert to meters
    dy, dx = (
        EARTH_R * np.deg2rad(dlat),
        EARTH_R * np.cos(np.deg2rad(lat)) * np.deg2rad(dlon),
    )

    # Get neighboring points with boundary checks
    # For latitude (north-south)
    if lat_idx == 0:  # Southern boundary ( Using forward difference)
        u_north, u_south = (
            float(ds["u10"].isel(latitude=lat_idx + 1, longitude=lon_idx).values),
            float(ds["u10"].isel(latitude=lat_idx, longitude=lon_idx).values),
        )
        du_dy = (u_north - u_south) / dy
    elif lat_idx == n_lat - 1:  # Northern boundary
        u_north, u_south = (
            float(ds["u10"].isel(latitude=lat_idx, longitude=lon_idx).values),
            float(ds["u10"].isel(latitude=lat_idx - 1, longitude=lon_idx).values),
        )
        du_dy = (u_north - u_south) / dy
    else:  # Interior points
        u_north, u_south = (
            float(ds["u10"].isel(latitude=lat_idx + 1, longitude=lon_idx).values),
            float(ds["u10"].isel(latitude=lat_idx - 1, longitude=lon_idx).values),
        )
        du_dy = (u_north - u_south) / (2 * dy)
    # For longitude (east-west)
    if lon_idx == 0:  # Western boundary
        v_east, v_west = (
            float(ds["v10"].isel(latitude=lat_idx, longitude=lon_idx + 1).values),
            float(ds["v10"].isel(latitude=lat_idx, longitude=lon_idx).values),
        )
        dv_dx = (v_east - v_west) / dx
    elif lon_idx == n_lon - 1:  # Eastern boundary
        v_east, v_west = (
            float(ds["v10"].isel(latitude=lat_idx, longitude=lon_idx).values),
            float(ds["v10"].isel(latitude=lat_idx, longitude=lon_idx - 1).values),
        )
        dv_dx = (v_east - v_west) / dx
    else:  # Interior points
        v_east, v_west = (
            float(ds["v10"].isel(latitude=lat_idx, longitude=lon_idx + 1).values),
            float(ds["v10"].isel(latitude=lat_idx, longitude=lon_idx - 1).values),
        )
        dv_dx = (v_east - v_west) / (2 * dx)
    # Relative vorticity (ζ = ∂v/∂x - ∂u/∂y)
    vorticity = dv_dx - du_dy
    return vorticity


def check_vorticity_criteria(data_slice, lat, lon):
    try:
        # Find the time step index
        time = data_slice["time"].values if "time" in data_slice.coords else None
        # Get u, v at center point
        u_center = extract_and_average_variable(
            data_slice,
            lat,
            lon,
            time,
            "u10",
            buffer_km=get_default_buffer_km("u10"),  # 150 km
            aggregation="mean",
        )
        v_center = extract_and_average_variable(
            data_slice,
            lat,
            lon,
            time,
            "v10",
            buffer_km=get_default_buffer_km("u10"),  # 150 kmm
            aggregation="mean",
        )
        if np.isnan(u_center) or np.isnan(v_center):
            u_center, v_center = np.nan, np.nan

        # Get latitude and longitude arrays
        lats, lons = data_slice.latitude.values, data_slice.longitude.values
        # Find indices nearest to the center
        lat_idx, lon_idx = np.argmin(np.abs(lats - lat)), np.argmin(np.abs(lons - lon))
        # Check at the boundaries
        if (
            lat_idx == 0
            or lat_idx == len(lats) - 1
            or lon_idx == 0
            or lon_idx == len(lons) - 1
        ):
            return False, np.nan

        # Get grid spacing in radians
        dlat_rad, dlon_rad = (
            np.deg2rad(lats[1] - lats[0]),
            np.deg2rad(lons[1] - lons[0]),
        )
        # Adjust dlon for latitude
        dlon_rad_adjusted = dlon_rad * np.cos(np.deg2rad(lat))
        # Get neighboring u, v values
        u_north = float(
            data_slice["u10"].isel(latitude=lat_idx + 1, longitude=lon_idx).values
        )
        u_south = float(
            data_slice["u10"].isel(latitude=lat_idx - 1, longitude=lon_idx).values
        )
        v_east = float(
            data_slice["v10"].isel(latitude=lat_idx, longitude=lon_idx + 1).values
        )
        v_west = float(
            data_slice["v10"].isel(latitude=lat_idx, longitude=lon_idx - 1).values
        )

        # Compute finite differences ( m/s per meter)
        du_dy, dv_dx = (
            (u_north - u_south) / (2 * EARTH_R * dlat_rad),
            (v_east - v_west) / (2 * EARTH_R * dlon_rad_adjusted),
        )

        # Compute relative vorticity (dv/dx - du/dy)
        vorticity = dv_dx - du_dy  # s^-1
        # Check if vorticity is positive (cyclonic)
        return vorticity > VORTICITY_THRESHOLD, vorticity
    except Exception as e:
        std_print("ERROR", "md", "Computing vorticity", str(e))
        return False, np.nan


def check_moisture_criteria(data_slice, lat, lon):
    try:
        # Check total precipitation (tp m/hr)
        time = data_slice["time"].values if "time" in data_slice.coords else None
        tp_value = extract_and_average_variable(
            data_slice,
            lat,
            lon,
            time,
            "tp",
            buffer_km=get_default_buffer_km("tp"),  # 150 km
            aggregation="mean",  # or 'sum' if cumulative
        )
        if not np.isnan(tp_value):
            precipitation_check = (
                tp_value * 1000
            ) >= PRECIPITATION_THRESHOLD_MM  # m/hr -> mm/hr
        else:
            precipitation_check = False
        return precipitation_check
    except KeyError as e:
        std_print("ERROR", "md", "Missing variable for moisture check:", str(e))
        return False


def compute_surface_heat_fluxes(ds, lat, lon):
    try:
        time = None
        slhtf = extract_and_average_variable(
            ds,
            lat,
            lon,
            time,
            "avg_slhtf",
            buffer_km=get_default_buffer_km("lhf"),  # 150 km
            aggregation="mean",
        )
        ishf = extract_and_average_variable(
            ds,
            lat,
            lon,
            time,
            "avg_ishf",
            buffer_km=get_default_buffer_km("lhf"),  # 150 km
            aggregation="mean",
        )
        total_hf = slhtf + ishf if not (np.isnan(slhtf) or np.isnan(ishf)) else np.nan
        return {"latent_flux": slhtf, "sensible_flux": ishf, "total_flux": total_hf}
    except:
        return {"latent_flux": np.nan, "sensible_flux": np.nan, "total_flux": np.nan}


def compute_asm_for_track(ds, track_df, hours_before=24):
    try:
        start_time = pd.to_datetime(track_df["time"].iloc[0])
        end_time = start_time
        start_window = start_time - pd.Timedelta(hours=hours_before)
        # Extract soil moisture data for this location before track start
        avg_lat, avg_lon = (
            track_df["latitude"].iloc[:5].mean(),
            track_df["longitude"].iloc[:5].mean(),
        )
        # Extract data
        ds, time_coord = get_time_coord(ds)
        spatial_point = ds.sel(latitude=avg_lat, longitude=avg_lon, method="nearest")
        # FIX: use extract_and_average_variable for swvl1
        subset = spatial_point.sel({time_coord: slice(start_window, end_time)})
        if "swvl1" in subset.data_vars:
            asm_value = float(subset["swvl1"].mean().values)
            return asm_value
        else:
            return np.nan
    except Exception as e:
        std_print("ERROR", "md", "Computing ASM", str(e))
        return np.nan


def diagnose_temporal_issues(ds):
    """
    Identifies:
    1. Gaps in time series
    2. Duplicate timestamps
    3. Overlapping data from monthly files
    4. Irregular sampling
    """
    # if 'valid_time' in ds.coords:
    #     ds = ds.rename({'valid_time': 'time'})
    ds, time_coord = get_time_coord(ds)
    times = pd.to_datetime(ds[time_coord].values)

    start_time = times[0].strftime('%d/%m/%Y %H:%M')
    end_time = times[-1].strftime('%d/%m/%Y %H:%M')
    std_print(None, None, "Start", start_time)
    std_print(None, None, "End", end_time)
    std_print(None, None, "Total duration (hours)", f"{(times[-1] - times[0]).total_seconds() / 3600:.1f}")
    std_print(None, None, "Total points", len(times))
    std_print(None, None, "Total points", len(times))

    # Calculate time differences
    time_diffs = (
        np.diff(times).astype("timedelta64[m]").astype(float) / 60
    )  # Convert to hours

    std_print("INFO", "md", " --- Time step statistics ---")
    std_print(None, None, "Mean interval", f"{np.mean(time_diffs):.2f} hours")
    std_print(None, None, "Median interval", f"{np.median(time_diffs):.2f} hours")
    std_print(None, None, "Std Dev", f"{np.std(time_diffs):.2f} hours")
    std_print(None, None, "Min interval", f"{np.min(time_diffs):.2f} hours")
    std_print(None, None, "Max interval", f"{np.max(time_diffs):.2f} hours")
    # Identify anomalies
    mode_diff = np.argmax(np.bincount(time_diffs.astype(int)))
    std_print(None, None, "Most common interval (hours)", mode_diff)

    # Find irregular gaps
    unique_diffs, counts = np.unique(time_diffs.astype(int), return_counts=True)
    std_print("INFO", "md", " --- Time step distribution ---")
    for diff, count in sorted(
        zip(unique_diffs, counts), key=lambda x: x[1], reverse=True
    ):
        percentage = 100 * count / len(time_diffs)
        std_print(None, None, "Hour intervals", f"{diff:2d} for {count:4d} times ({percentage:5.1f}%)")

    # Detect month boundaries
    std_print("INFO", "md", " --- Month boundary analysis ---")
    month_changes = []
    for i in range(1, len(times)):
        if times[i].month != times[i - 1].month:
            month_changes.append((i, times[i - 1], times[i], time_diffs[i - 1]))
    if month_changes:
        std_print("INFO", "md", f"Found {len(month_changes)} month transitions :")
        for idx, prev_time, curr_time, gap in month_changes:
            std_print(None, None, f"Index {idx}: {prev_time.strftime('%d/%m/%Y %H:%M')} -> {curr_time.strftime('%d/%m/%Y %H:%M')}", f"(gap: {gap:.1f} hours)")
    else:
        std_print("INFO", "md", "No month transitions detected (single continuous month)")

    # Detect duplicates
    std_print("INFO", "md", " --- Duplicate/overlap detection ---")
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
    std_print("INFO", "md", " --- Time continuity check ---")
    negative_gaps = np.where(time_diffs < 0)[0]
    if len(negative_gaps) > 0:
        std_print(None, None, "Found Time reversals", f"{len(negative_gaps)} backward jumps")
        for idx in negative_gaps[:5]:
            std_print(None, None, f"Jump backward at index {idx}", f"{times[idx]} -> {times[idx + 1]} (difference: {time_diffs[idx]:.1f} hours)")
    else:
        std_print("INFO", "md", "Time is monotonically increasing")

    # Detect zero-gaps (identical consecutive times)
    zero_gaps = np.where(time_diffs == 0)[0]
    if len(zero_gaps) > 0:
        std_print("INFO", "md", "Found identical consecutive timestamps", f"{len(zero_gaps)}")
    if n_duplicates > 0:
        std_print("INFO", "md", "Duplicates present", f"{n_duplicates} repeated timestamps")
    if len(negative_gaps) > 0:
        std_print("INFO", "md", "Time reversals", f"{len(negative_gaps)} backward jumps")
    if len(month_changes) > 0 and len(unique_diffs) > 1:
        std_print("INFO", "md", "Files likely concatenated", "Different resolutions at month boundaries")
    return {
        "times": times,
        "time_diffs": time_diffs,
        "unique_intervals": unique_diffs,
        "interval_counts": counts,
        "n_duplicates": n_duplicates,
        "month_changes": month_changes,
        "negative_gaps": negative_gaps,
        "zero_gaps": zero_gaps,
    }


def fix_temporal_resolution(ds, target_resolution_hours=1):
    ds, time_coord = get_time_coord(ds)
    times = pd.to_datetime(ds[time_coord].values)
    std_print("INFO", "md", "Dataset temporal resolution", f"{len(times)} time points")
    # Regular time grid
    time_start, time_end = times[0], times[-1]
    new_times = pd.date_range(
        start=time_start, end=time_end, freq=f"{target_resolution_hours}h"
    )
    # Reindex dataset to regular grid with interpolation
    for var in ["u10", "v10"]:
        if var in ds.data_vars:
            ds[var] = ds[var].interp({time_coord: new_times}, method="nearest")
    for var in ["tp", "swvl1", "msl", "t2m"]:
        if var in ds.data_vars:
            ds[var] = ds[var].interp(
                {time_coord: new_times}, method="linear", kwargs={"fill_value": np.nan}
            )
    ds_fixed = ds.reindex({time_coord: new_times}, method="nearest")
    std_print("INFO", "md", "Resampled time points (hours)", f"{len(new_times)}/{target_resolution_hours}")
    # original_bounds = (ds[time_coord].min(), ds[time_coord].max())
    # out_of_bounds = (new_times < original_bounds[0]) | (new_times > original_bounds[1])
    # std_print('INFO', 'md', 'Interpolation points outside original time bounds', f'{out_of_bounds.sum()}')
    return ds_fixed


def validate_temporal_consistency(ds, expected_hours=1, tolerance=0.1):
    ds, time_coord = get_time_coord(ds)
    times = pd.to_datetime(ds[time_coord].values)
    time_diffs = np.diff(times).astype("timedelta64[m]").astype(float) / 60

    min_expected, max_expected = (
        (expected_hours * (1 - tolerance)),
        (expected_hours * (1 + tolerance)),
    )
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
    std_print("INFO", "md", " --- Temporal validation ---")
    std_print(None, None, "Expected", f"{expected_hours}±{tolerance * 100:.0f}% hour(s)")
    std_print(None, None, "Compliance", f"{n_valid}/{n_total} intervals ({validity_pct:.1f}%)")
    std_print(None, None, "Valid", f"{'YES' if is_valid else 'NO'}")
    if not is_valid and len(report["violations"]) > 0:
        std_print(None, None, "Violations at indices", f"{report['violations'][:10]}")
    return is_valid, report


def resample_dataset_to_resolution(ds, target_hours=1):
    ds, time_coord = get_time_coord(ds)
    times = pd.to_datetime(ds[time_coord].values)
    if target_hours == 1:
        std_print("INFO", "md", "Dataset at 1-hourly resolution", f"{len(times)} steps")
        return ds
    new_times = pd.date_range(start=times[0], end=times[-1], freq=f"{target_hours}h")
    original_count, new_count = len(times), len(new_times)
    std_print("INFO", "md", " --- Temporal resampling ---")
    std_print(None, None, "Input", f"{original_count} time steps (hourly)")
    std_print(None, None, "Output", f"{new_count} time steps ({target_hours}-hourly)")
    std_print(None, None, "Reduction", f"{100 * (1 - new_count / original_count):.1f}%")

    # Interpolate all variables to new time grid
    ds_resampled = ds.interp(
        {time_coord: new_times},
        method="linear",
        kwargs={"fill_value": np.nan},
        # kwargs={"fill_value": "extrapolate"}
    )
    return ds_resampled


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
                flux_res["duration_hours"] = classification.get(
                    "duration_hours", np.nan
                )
            if "category" not in flux_res:
                flux_res["category"] = classification.get("category", "Unknown")
    return all_flux_results


def find_low_pressure_center(
    data_slice,
    time_step,
    prev_center,
    search_radius_km=TRACKING_SEARCH_RADIUS_KM,
    smoothing_radius_km=100,
):
    required_vars = ["msl", "u10", "v10", "tp"]
    if not all(var in data_slice.data_vars for var in required_vars):
        return None
    try:
        msl_data = data_slice["msl"]
        u_data = data_slice["u10"]
        v_data = data_slice["v10"]
        # Find all low pressure candidates
        msl_values = msl_data.values.flatten()
        shape = msl_data.shape
        # Get top N lowest pressure points
        n_candidates = 6
        flat_indices = np.argsort(msl_values)[:n_candidates]
        candidates = []
        # Convert flat indices to 2D indices
        for idx in flat_indices:
            lat_idx, lon_idx = np.unravel_index(idx, shape)
            lat, lon = (
                float(msl_data.latitude.values[lat_idx]),
                float(msl_data.longitude.values[lon_idx]),
            )
            pressure = float(msl_values[idx])
            # Check if this is a reasonable candidate
            # if pressure > MSL_LOWER_BOUND_PA + 500:
            if pressure > MSL_LOWER_BOUND_PA:
                continue
            # Calculate distance from previous center if available
            distance = 0
            if prev_center is not None:
                prev_lat, prev_lon = prev_center
                distance = haversine_distance(prev_lon, prev_lat, lon, lat)
                if distance > search_radius_km:
                    continue
            candidates.append(
                {"lat": lat, "lon": lon, "pressure": pressure, "distance": distance}
            )
        if not candidates:
            return None
        # Sort candidates by pressure (lowest first)
        candidates.sort(key=lambda x: x["pressure"])
        for candidate in candidates:
            lat_center, lon_center = candidate["lat"], candidate["lon"]
            min_msl_val = candidate["pressure"]
            # Relaxed criteria check
            pressure_ok = min_msl_val <= MSL_LOWER_BOUND_PA  # + 1000
            if msl_data is not None:
                closed_contour = has_closed_contour(msl_data, lat_center, lon_center)
            else:
                closed_contour = False
            if u_data is not None and v_data is not None:
                wind_ok = check_wind_criteria(u_data, v_data, lat_center, lon_center)
            else:
                wind_ok = False
            if data_slice is not None:
                moisture_ok = check_moisture_criteria(
                    data_slice, lat_center, lon_center
                )
            else:
                moisture_ok = False
            if data_slice is not None:
                vorticity_ok, _ = check_vorticity_criteria(
                    data_slice, lat_center, lon_center
                )
            else:
                vorticity_ok = False
            score = (
                int(pressure_ok)
                + int(closed_contour)
                + int(wind_ok)
                + int(moisture_ok)
                + int(vorticity_ok)
            )
            if score >= 2:  # Accept if at least no. of criteria pass
                return lat_center, lon_center

            # debug_info = {
            #     'pressure_ok': pressure_ok,
            #     'closed_contour': closed_contour,
            #     'wind_ok': wind_ok,
            #     'moisture_ok': moisture_ok,
            #     'vorticity_ok': vorticity_ok,
            #     'pressure_value': min_msl_val,
            #     'score': score
            # }
            # i = 0
            # if i % 100000 == 0:  # Print every nth time step
            #     std_print('DEBUG', 'md', f't={time_step}: {debug_info}')
            #     return None
    except Exception as e:
        std_print("ERROR", "md", f"Finding low pressure center at {time_step}", str(e))
        return None


def track_monsoon_depressions(
    detections, max_speed_ms, search_radius_km, max_time_gap_hours
):
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
        std_print("INFO", "md", " --- Time gaps between detections ---")
        std_print(None, None, "Mean", f"{np.mean(time_diffs):.1f} hour(s)")
        std_print(None, None, "Min/Max", f"{np.min(time_diffs):.1f}/{np.max(time_diffs):.1f} hour(s)")
        std_print(None, None, "Std Dev", f"{np.std(time_diffs):.1f} hour(s)")
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
    for t_idx in range(1, len(time_steps)):
        current_hour = time_steps[t_idx]
        current_detections = time_groups.get_group(current_hour)
        prev_hour = time_steps[t_idx - 1]
        time_diff_hours = (current_hour - prev_hour).total_seconds() / 3600

        # Only proceed if time gap is reasonable
        if time_diff_hours > max_time_gap_hours:
            std_print(None, None, "Large time gap (hours)", f"{time_diff_hours:.1f}  (from {prev_hour} to {current_hour})")
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
                speed_ms = (distance * 1000) / (time_diff_hours * 3600)
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
                if cost < np.inf:
                    track_id = list(active_tracks.keys())[t_idx_m]
                    det_row = current_detections.iloc[d_idx_m]
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
                    new_id = (
                        max(new_active_tracks.keys()) + 1 if new_active_tracks else 0
                    )
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
        base["time"] = pd.to_datetime(base["time"]).dt.tz_localize(
            None
        )  # Ensure UTC naive
        # other['time'] = pd.to_datetime(other['time']).dt.tz_localize(None)
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
                    std_print("INFO", "md", "Split mixed track", f"Kept ocean segment of {len(longest_segment)} points")
                else:
                    split_tracks.append(track)  # Keep original
            else:
                split_tracks.append(track)  # Keep original
        else:
            split_tracks.append(track)  # Keep original
    return split_tracks


def smooth_track_coordinates(track_df, window_size=5, iterations=2):
    if len(track_df) < window_size:
        std_print("ERROR", "md", "Track have less points than", window_size)
        return track_df
    track_df = track_df.copy()
    # Check if required columns exist
    if "latitude" not in track_df.columns or "longitude" not in track_df.columns:
        std_print("ERROR", "md", "Track DataFrame is missing 'latitude' and/or 'longitude' columns")
        return track_df

    for _ in range(iterations):
        # Moving average
        lat_smooth, lon_smooth = (
            track_df["latitude"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean(),
            track_df["longitude"]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean(),
        )
        # Apply smoothed values
        track_df["latitude"], track_df["longitude"] = lat_smooth, lon_smooth
    return track_df


def calculate_track_direction(track_data):
    if len(track_data) < 2:
        return None
    start_lat, start_lon = (
        track_data.iloc[0]["latitude"],
        track_data.iloc[0]["longitude"],
    )
    end_lat, end_lon = track_data.iloc[-1]["latitude"], track_data.iloc[-1]["longitude"]

    lat1, lon1 = np.radians(start_lat), np.radians(start_lon)
    lat2, lon2 = np.radians(end_lat), np.radians(end_lon)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360  # Normalize to 0-360
    return bearing


def filter_valid_tracks(tracks, min_duration, min_distance):
    """Filter tracks by duration, displacement, and direction"""
    valid_tracks = []
    # Debug counters
    fail_reasons = {
        "duration": 0,
        "distance": 0,
        "direction": 0,
        "intensity": 0,
        "curvature": 0,
    }

    tracks_to_process = tracks.values() if isinstance(tracks, dict) else tracks
    for track in tracks_to_process:
        # 1. Duration check
        duration = (track["time"].max() - track["time"].min()).total_seconds() / 3600
        if duration < min_duration:
            fail_reasons["duration"] += 1
            continue
        # 2. Distance check (Minimum displacement)
        total_dist = haversine_distance(
            track.iloc[0]["longitude"],
            track.iloc[0]["latitude"],
            track.iloc[-1]["longitude"],
            track.iloc[-1]["latitude"],
        )
        if total_dist < min_distance:  # Short duration must move
            fail_reasons["distance"] += 1
            continue
        # Check direction (WNW preferred)
        bearing = calculate_track_direction(track)
        # 3. Direction check (WNW preferred for MDs)
        bearing = calculate_track_direction(track)
        if bearing is not None:
            # MDs typically move WNW (240-360) or NW (300-360)
            is_valid_md_direction = (240 <= bearing <= 360) or (0 <= bearing <= 30)
            if not is_valid_md_direction:
                fail_reasons["direction"] += 1
                continue
        # 4. Intensity check - Ensure minimum pressure drop
        if "msl" in track.columns:
            pressure_drop = track["msl"].max() - track["msl"].min()
            if pressure_drop < 200:  # Less than 2 hPa drop (weak system)
                fail_reasons["intensity"] += 1
                continue
        # 5. Require significant ocean portion
        if "is_land" in track.columns:
            ocean_fraction = 1 - (track["is_land"].sum() / len(track))
            if ocean_fraction < 0.3:  # Less than 30% over ocean
                fail_reasons["origin"] += 1
                continue
        # 6. Curvature check (avoid stationary/recurving systems)
        curvature = calculate_track_curvature(track)
        if curvature > 3.0:  # Too curved/stationary
            fail_reasons["curvature"] += 1
            continue
        valid_tracks.append(track)
    # Debug output
    std_print("INFO", "md", "Track filtering statistics:")
    for reason, count in fail_reasons.items():
        if count > 0:
            # std_print('DEBUG', 'md', f"No. of Tracks failed {reason} threshold", f"{count}")
            std_print(None, None, f"No. of Tracks failed {reason} threshold", f"{count}")
    return valid_tracks


def process_files(ds):
    try:
        ds, time_coord = get_time_coord(ds)
        times = pd.to_datetime(ds[time_coord].values)
        time_diffs = np.diff(times).astype("timedelta64[s]").astype(float) / 3600
        std_print("INFO", "md", " --- Temporal check ---")
        std_print(None, None, "Dataset time steps", len(times))
        std_print(None, None, "Mean interval", f"{np.mean(time_diffs):.2f} hours")
        std_print(None, None, "Min interval", f"{np.min(time_diffs):.2f} hours")
        std_print(None, None, "Max interval", f"{np.max(time_diffs):.2f} hours")

        # Verify dataset has regular spacing
        if np.std(time_diffs) > 0.1:
            std_print("WARN", "md", "Irregular intervals detected")
            ds = fix_temporal_resolution(ds, target_resolution_hours=1)
            times = pd.to_datetime(ds[time_coord].values)
            time_diffs = np.diff(times).astype("timedelta64[s]").astype(float) / 3600
            std_print("INFO", "md", "After fix Mean interval (hours)", f"{np.mean(time_diffs):.2f}")

        required_vars = ["msl", "u10", "v10", "tp"]
        if not all(var in ds.data_vars for var in required_vars):
            std_print("ERROR", "md", "Missing required variables", required_vars)
            return None

        times = ds[time_coord].values
        start = pd.Timestamp(times[0]).strftime("%d/%m/%Y %H:%M")
        end = pd.Timestamp(times[-1]).strftime("%d/%m/%Y %H:%M")
        std_print("INFO", "md", "Time range", f"{start} to {end}")
        std_print("DEBUG", "md", "Number of time steps after verification", len(times))

        std_print("INFO", "md", " --- Detection progress ---")
        # Container for detections
        md_detections = []
        total_steps = len(times)
        progress_bar = ProgressBar(total_steps, "Detection Progress")
        prev_center = None
        detection_count = 0
        for i, time_step in enumerate(times):
            if i % 5 == 0:  # Update progress every 5 steps
                progress_bar.update(current=i)
            data_slice = ds.sel(**{time_coord: time_step})
            center = find_low_pressure_center(
                data_slice,
                time_step,
                prev_center,
                search_radius_km=TRACKING_SEARCH_RADIUS_KM,
                smoothing_radius_km=100,
            )
            if center is not None:
                lat, lon = center
                data_slice = ds.sel(time=time_step)
                msl_value = extract_and_average_variable(
                    data_slice,
                    lat,
                    lon,
                    time_step,
                    "msl",
                    buffer_km=100,
                    aggregation="mean",
                )
                md_detections.append(
                    {
                        "time": time_step,
                        "latitude": lat,
                        "longitude": lon,
                        "msl": msl_value,
                    }
                )
                prev_center = (lat, lon)
                detection_count += 1
            else:
                prev_center = None
        progress_bar.finish()
        std_print("INFO", "md", "Low-pressure systems detected", f"{detection_count} / {total_steps} centers found")
        if not md_detections:
            std_print("INFO", "md", "No MD centers detected", "Insufficient low-pressure signatures in dataset")
            sys.exit(1)

        detections_df = pd.DataFrame(md_detections)
        detections_df["month"] = pd.to_datetime(detections_df["time"]).dt.strftime("%B")
        # Verify detection times are on the hour
        detection_hours = [int(h) for h in detections_df["time"].dt.hour.unique()]
        std_print(None, None, "Detections occur at hours", sorted(detection_hours))
        # Initial Tracking
        initial_tracks = track_monsoon_depressions(
            detections_df,
            max_speed_ms=MAX_SPEED_MS,
            search_radius_km=LINKING_SEARCH_RADIUS_KM,
            max_time_gap_hours=MAX_GAP_HOURS,
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
        # split_tracks = split_mixed_tracks(consolidated_tracks)
        # std_print('INFO', 'md', 'Tracks after splitting mixed systems', len(split_tracks))

        # Filter by environment
        # ocean_tracks = filter_tracks_by_environment(consolidated_tracks, require_ocean_start=True)
        # std_print('INFO', 'md', 'Tracks over ocean', len(ocean_tracks)")

        # Quality Filter tracks (Noise Removal)
        filtered_valid_tracks = filter_valid_tracks(
            consolidated_tracks,
            min_duration=MIN_TRACK_DURATION_HOURS,
            min_distance=MIN_NET_DISPLACEMENT_KM,
        )
        std_print("INFO", "md", "Valid tracks after QC", len(filtered_valid_tracks))
        if not filtered_valid_tracks:
            std_print(None, None, "No tracks survived filtering criteria")
            return pd.DataFrame()  # Return empty dataframe

        # Smooth each track
        smoothed_tracks = []
        for track in filtered_valid_tracks:
            if len(track) > 5:
                smoothed_track = smooth_track_coordinates(
                    track, window_size=5, iterations=4
                )
                smoothed_tracks.append(smoothed_track)
            else:
                smoothed_tracks.append(track)

        # Assign track IDs
        track_df_with_ids = pd.concat(
            [track.assign(track_id=i + 1) for i, track in enumerate(smoothed_tracks)],
            ignore_index=True,
        )
        # Restore month column if lost
        if "month" not in track_df_with_ids.columns:
            track_df_with_ids["month"] = pd.to_datetime(
                track_df_with_ids["time"]
            ).dt.strftime("%B")
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


def plot_tracks(track_df, output_dir="output"):
    std_print("INFO", "md", " --- MD track plots ---")
    track_data = track_df.sort_values("time")

    fig = plt.figure(figsize=(14, 11))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]
    # ax.set_extent([70, 100, 5, 30], crs=ccrs.PlateCarree())
    ax.set_extent([clipped_min_lon, clipped_max_lon, clipped_min_lat, clipped_max_lat], crs=ccrs.PlateCarree(),)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, linewidth=0.8, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

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

    legend_handles = []
    months_in_plot = set()
    for track_plot_index, (track_id_str, track_data) in enumerate(
        track_df.groupby("track_id")
    ):
        track_data = track_data.sort_values("time")
        start_point, end_point = track_data.iloc[0], track_data.iloc[-1]
        start_time = pd.to_datetime(start_point["time"]).strftime("%d/%m %H:%M")
        end_time = pd.to_datetime(end_point["time"]).strftime("%d/%m %H:%M")
        start_lon, start_lat = start_point["longitude"], start_point["latitude"]
        end_lon, end_lat = end_point["longitude"], end_point["latitude"]

        month = track_data["month"].iloc[0]
        color = month_colors.get(month, "black")
        bbox_style = common_bbox()

        # Plot all detection points
        ax.plot(
            track_data["longitude"],
            track_data["latitude"],
            "o",
            color=color,
            markersize=1.2,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        # Plot connecting lines
        # line, = ax.plot(
        #     track_data["longitude"],
        #     track_data["latitude"],
        #     "-",
        #     color=color,
        #     alpha=0.8,
        #     linewidth=1.2,
        #     transform=ccrs.PlateCarree(),
        #     zorder=2,
        # )
        # # Legend
        # if month not in months_in_plot:
        #     proxy_line = plt.Line2D([0], [0], color=color, linewidth=1.2, label=f"Low-press. sys. Track ({month})")
        #     legend_handles.append(proxy_line)
        #     months_in_plot.add(month)
        line, = plt.plot(
            track_data["longitude"],
            track_data["latitude"],
            "-",
            color=color,
            alpha=0.8,
            linewidth=1.2,
            label=f"Low-press. sys. Track ({month})",
            transform=ccrs.PlateCarree(),
            zorder=2
        )

        # Legend
        month_already_in_legend = False
        for handle in legend_handles:
            if hasattr(handle, 'get_label') and handle.get_label() == f"Low-press. sys. Track ({month})":
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
            zorder=8,
        )
        ax.text(
            start_lon - 0.4,
            start_lat,
            f"T{int(track_id_str)}",
            transform=ccrs.PlateCarree(),
            fontsize=8,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox=bbox_style,
            zorder=9,
        )
        ax.text(
            start_lon + 0.3,
            start_lat,
            start_time,
            transform=ccrs.PlateCarree(),
            fontsize=8,
            fontweight="normal",
            color="black",
            ha="left",
            va="center",
            bbox=bbox_style,
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
            zorder=8,
        )
        ax.text(
            end_lon - 0.4,
            end_lat,
            f"T{int(track_id_str)}",
            transform=ccrs.PlateCarree(),
            fontsize=8,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            bbox=bbox_style,
            zorder=9,
        )
        ax.text(
            end_lon + 0.3,
            end_lat,
            end_time,
            transform=ccrs.PlateCarree(),
            fontsize=8,
            fontweight="normal",
            color="black",
            ha="left",
            va="center",
            bbox=bbox_style,
        )

    # Legend
    legend_handles, labels = ax.get_legend_handles_labels()
    bbox_style = common_bbox()
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            **legend_style()
        )

    ax.set_title(
        f"Detected MD Tracks ({month_order[0]}-{month_order[-1]}, {MONSOON_YEAR})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_filename = os.path.join(output_dir, f"MD_track_{MONSOON_YEAR}.png")
    plt.savefig(output_filename)
    std_print("SAVED", "md", "Combined track plot", output_filename)
    plt.close(fig)


def plot_track_full_map(track_df_single, track_id, full_extent, output_dir):
    fig = plt.figure(figsize=(13, 10))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]

    ax.set_extent(
        [clipped_min_lon, clipped_max_lon, clipped_min_lat, clipped_max_lat],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle="--")
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

    track_times = pd.to_datetime(track_df["time"])
    track_data = track_df_single.sort_values("time")
    month = track_data["month"].iloc[0]
    track_color = month_colors.get(month, "black")
    bbox_style = common_bbox()

    # start/end (for annotatation)
    start_datetime, end_datetime = (
        pd.to_datetime(track_data.iloc[0]["time"]),
        pd.to_datetime(track_data.iloc[-1]["time"]),
    )
    start_time_str, end_time_str = (
        start_datetime.strftime("%d/%m %H:%M"),
        end_datetime.strftime("%d/%m %H:%M"),
    )
    start_lon, end_lon = track_df["longitude"].iloc[0], track_df["longitude"].iloc[-1]
    start_lat, end_lat = track_df["latitude"].iloc[0], track_df["latitude"].iloc[-1]
    ax.plot(
        track_data["longitude"],
        track_data["latitude"],
        "-o",
        color=track_color,
        transform=ccrs.PlateCarree(),
        linewidth=1,
        markersize=2,
        zorder=5,
    )
    # 12-hour interval markers
    hours_elapsed = (track_times - start_datetime).dt.total_seconds() / 3600
    for elapsed_hours in np.arange(0, hours_elapsed.max() + 12, 12):
        idx = np.argmin(np.abs(hours_elapsed - elapsed_hours))
        if idx < len(track_df):
            point = track_df.iloc[idx]
            point_time = pd.to_datetime(point[time_coord])
            ax.plot(
                point["longitude"],
                point["latitude"],
                "o",
                color=track_color,
                markersize=4,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            time_label = f"{int(elapsed_hours)}h {point_time.strftime('%d/%m %H:%M')}"
            ax.text(
                point["longitude"] - 0.1,
                point["latitude"] + 0.1,
                time_label,
                transform=ccrs.PlateCarree(),
                fontsize=5,
                fontweight="bold",
                color="black",
                ha="left",
                va="bottom",
                bbox=bbox_style,
                zorder=7,
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
        zorder=8,
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
        zorder=8,
    )
    # landfall marker (yellow square)
    if "is_land" in track_df.columns:
        land_status = track_df["is_land"].values
        transitions = np.diff(land_status.astype(int))
        ocean_to_land = np.where(transitions == 1)[0]
        if len(ocean_to_land) > 0:
            landfall_idx = ocean_to_land[0] + 1
            if landfall_idx < len(track_df):
                landfall_point = track_df.iloc[landfall_idx]
                lf_lon, lf_lat = landfall_point["longitude"], landfall_point["latitude"]
                lf_time_str = pd.to_datetime(landfall_point["time"]).strftime("%d/%m %H:%M")
                ax.plot(
                    lf_lon,
                    lf_lat,
                    marker="s",
                    color="yellow",
                    markersize=5,
                    transform=ccrs.PlateCarree(),
                    label="Landfall",
                    zorder=6,
                )
                ax.text(
                    lf_lon + 0.2,
                    lf_lat + 0.2,
                    f"Landfall: {lf_time_str}",
                    transform=ccrs.PlateCarree(),
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                    ha="left",
                    va="top",
                    bbox=bbox_style,
                    zorder=8,
                )

    # Legend
    legend_handles, labels = ax.get_legend_handles_labels()
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            **legend_style()
        )

    ax.set_title(
        f"Monsoon Depression Track {track_id}\n{start_time_str} to {end_time_str}, {MONSOON_YEAR} ",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    plt.tight_layout()
    outpath = os.path.join(output_dir, f"track_{int(track_id)}_full_map.png")
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    std_print("SAVED", "md", f"Full-map for track {track_id}", outpath)


def plot_single_track(track_df, track_id, output_dir):
    track_id = int(track_id)
    track_data = track_df.sort_values("time")
    track_times = pd.to_datetime(track_data["time"])
    month = track_data["month"].iloc[0]
    track_color = month_colors.get(month, "teal")
    bbox_style = common_bbox()

    start_datetime, end_datetime = (
        pd.to_datetime(track_data.iloc[0]["time"]),
        pd.to_datetime(track_data.iloc[-1]["time"]),
    )
    start_time_str, end_time_str = (
        start_datetime.strftime("%d/%m %H:%M"),
        end_datetime.strftime("%d/%m %H:%M"),
    )

    fig = plt.figure(figsize=(13, 10))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]

    # FIX: for track which goes till around 70 (track 3) minus 4 lat will show no results there
    # So check if out of bound of dataset too
    # min_lon, max_lon = track_df["longitude"].min() - 4, track_df["longitude"].max() + 4
    # min_lat, max_lat = track_df["latitude"].min() - 4, track_df["latitude"].max() + 4

    # Calculate extent based on track with padding, but clamp to clipped domain
    min_lon, min_lat = (
        max(track_df["longitude"].min() - 4, clipped_min_lon),
        max(track_df["latitude"].min() - 4, clipped_min_lat),
    )
    max_lon, max_lat = (
        min(track_df["longitude"].max() + 4, clipped_max_lon),
        min(track_df["latitude"].max() + 4, clipped_max_lat),
    )

    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=1.0)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle="--")
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

    ax.plot(
        track_df["longitude"],
        track_df["latitude"],
        "-",
        color=track_color,
        alpha=0.8,
        linewidth=1.5,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )  # connecting lines
    ax.plot(
        track_df["longitude"],
        track_df["latitude"],
        "o",
        color=track_color,
        markersize=2,
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )  # markers for all points (hourly detections)
    # 12-hour interval markers
    hours_elapsed = (track_times - start_datetime).dt.total_seconds() / 3600
    # INFO:
    # std_print('DEBUG', 'md', f"Track {track_id} time gaps: {track_df[time_coord].diff().value_counts()}")
    # There time gaps present 5hr, 3 hr and 1 hour gap for track 1 due to which there won't be exaclty 11 points between 2 12 hours marker points
    for elapsed_hours in np.arange(0, hours_elapsed.max() + 12, 12):
        idx = np.argmin(np.abs(hours_elapsed - elapsed_hours))
        if idx < len(track_df):
            point = track_df.iloc[idx]
            point_time = pd.to_datetime(point[time_coord])
            ax.plot(
                point["longitude"],
                point["latitude"],
                "o",
                color=track_color,
                markersize=4,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            time_label = f"{int(elapsed_hours)}h {point_time.strftime('%d/%m %H:%M')}"
            ax.text(
                point["longitude"] - 0.1,
                point["latitude"] + 0.1,
                time_label,
                transform=ccrs.PlateCarree(),
                fontsize=5,
                fontweight="bold",
                color="black",
                ha="left",
                va="bottom",
                bbox=bbox_style,
                zorder=7,
            )

    # Annotate Start point
    ax.plot(
        track_df["longitude"].iloc[0],
        track_df["latitude"].iloc[0],
        marker="o",
        color="green",
        markersize=5,
        markeredgewidth=5,
        transform=ccrs.PlateCarree(),
        zorder=8,
    )
    # Annotate End point
    ax.plot(
        track_df["longitude"].iloc[-1],
        track_df["latitude"].iloc[-1],
        marker="o",
        color="orange",
        markersize=5,
        markeredgewidth=5,
        transform=ccrs.PlateCarree(),
        zorder=8,
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
                lf_time_str = pd.to_datetime(landfall_point["time"]).strftime(
                    "%d/%m %H:%M"
                )
                ax.plot(
                    lf_lon,
                    lf_lat,
                    marker="s",
                    color="yellow",
                    markersize=5,
                    transform=ccrs.PlateCarree(),
                    label="Landfall",
                    zorder=6,
                )
                ax.text(
                    lf_lon + 0.4,
                    lf_lat + 0.1,
                    f"Landfall: {lf_time_str}",
                    transform=ccrs.PlateCarree(),
                    fontsize=8,
                    fontweight="bold",
                    color="black",
                    ha="left",
                    va="top",
                    bbox=bbox_style,
                    zorder=8,
                )

    # Legend
    legend_handles, labels = ax.get_legend_handles_labels()
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            **legend_style()
        )

    ax.set_title(
        f"Monsoon Depression Track {track_id}\n{start_time_str} to {end_time_str}, {MONSOON_YEAR}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    plt.tight_layout()
    file_name = f"MD_track_{track_id}_detailed.png"
    fullpath = os.path.join(output_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(fullpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    std_print("SAVED", "md", f"Detailed track {track_id} plot", fullpath)


def export_track_coordinates(track_df, output_dir):
    if track_df.empty:
        std_print("ERROR", "md", "No track coordinates to export")
        return
    export_df = track_df[["time", "latitude", "longitude", "track_id"]].copy()
    export_df["time"] = pd.to_datetime(export_df["time"]).dt.strftime("%d/%m/%Y %H:%M")
    output_filename = os.path.join(output_dir, "MD_coordinates_ALL.csv")
    export_df.to_csv(output_filename, index=False)
    std_print("INFO", "md", "No. of Track Coordinates to export", len(track_df))
    std_print("EXPORT", "md", "Track Coordinates detections to CSV", output_filename)


def export_single_track_coordinates(track_df, track_id, output_dir):
    track_id = int(track_id)
    export_df = track_df[["time", "latitude", "longitude", "track_id"]].copy()
    export_df["time"] = pd.to_datetime(export_df["time"]).dt.strftime("%d/%m/%Y %H:%M")
    track_id_str = str(track_id).replace("/", "_").replace(" ", "_")
    file_name = f"MD-track-{track_id_str}-coordinates.csv"
    export_df.to_csv(os.path.join(output_dir, file_name), index=False)
    std_print("EXPORT", "md", f"Track {track_id} to CSV", file_name)


def load_dataset_async(month_dirs, data_dir, result_container):
    file_list = []
    monthly_stats = {}
    for month_dir in month_dirs:
        month_path = os.path.join(data_dir, month_dir)
        if os.path.isdir(month_path):
            month_files = [
                os.path.join(month_path, f)
                for f in os.listdir(month_path)
                if f.endswith(".nc")
            ]
            file_list.extend(month_files)
            monthly_stats[month_dir] = {
                "n_files": len(month_files),
                "files": month_files[:3],
            }
    if not file_list:
        result_container["error"] = (
            "No netCDF files found in the specified directories."
        )
        result_container["success"] = False
        return
    try:
        # FIX: Issue: .compute() loads entire dataset into memory, Could cause OOM for very large datasets
        ds = xr.open_mfdataset(
            file_list,
            combine="by_coords",
            data_vars="all",
            coords="all",  # type: ignore[arg-type]
            compat="override",
            join="outer",
        ).compute()
        # ds = xr.open_mfdataset(
        #     file_list,
        #     chunks={'time': 24},
        #     combine='by_coords',
        #     parallel=True,
        #     coords='minimal',   # type: ignore[arg-type]
        #     data_vars='minimal',
        #     compat='override'
        # )
        if "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        ds, time_coord = get_time_coord(ds)
        ds = ds.sortby("time")
        result_container["ds"] = ds
        result_container["success"] = True
    except Exception as e:
        result_container["error"] = str(e)
        result_container["success"] = False


def diagnose_detection_issues(ds, track_df):
    """
    Checks for:
    - Missing detections (gaps in continuous pressure)
    - False detections (weak systems)
    - Tracking jumps
    """
    std_print("INFO", "md", " --- Detection Quality Diagnostics --")
    # 1. Check for large gaps in time
    times = pd.to_datetime(track_df["time"])
    time_diffs = times.diff().dt.total_seconds().dropna() / 3600
    large_gaps = time_diffs[time_diffs > 6]
    if len(large_gaps) > 0:
        std_print("INFO", "md", "Tracks with time gaps >6 hours", len(large_gaps))
        std_print("INFO", "md", "Max gap", f"{time_diffs.max():.1f} hours")
    else:
        std_print("INFO", "md", "No large time gaps detected")

    # 2. Check for sudden position jumps
    if "distance_km" in track_df.columns:
        large_jumps = track_df[track_df["distance_km"] > 150]
        if len(large_jumps) > 0:
            std_print("INFO", "md", "Tracks position jumps (>150 km)", len(large_jumps))
            std_print("INFO", "md", "Max jump", f"{track_df['distance_km'].max():.1f} km")
        else:
            std_print("INFO", "md", "No suspicious position jumps")

    # 3. Check speed consistency
    if "speed_kmh" in track_df.columns:
        speeds = track_df["speed_kmh"].dropna()
        if len(speeds) > 0:
            unrealistic = speeds[(speeds < 3) | (speeds > 30)]
            if len(unrealistic) > 0:
                std_print("INFO", "md", "Tracks with unrealistic speeds", len(unrealistic))
                std_print("INFO", "md", "Range", f"{speeds.min():.1f} - {speeds.max():.1f} km/h")
            else:
                std_print("INFO", "md", "All speeds within reasonable range")

    # 4. Check track direction consistency
    if len(track_df) >= 3:
        bearings = []
        for i in range(len(track_df) - 1):
            lat1, lon1 = track_df.iloc[i][["latitude", "longitude"]]
            lat2, lon2 = track_df.iloc[i + 1][["latitude", "longitude"]]
            # Calculate bearing
            dlon = np.deg2rad(lon2 - lon1)
            lat1_rad, lat2_rad = np.deg2rad(lat1), np.deg2rad(lat2)
            x = np.sin(dlon) * np.cos(lat2_rad)
            y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
                lat2_rad
            ) * np.cos(dlon)
            bearing = np.degrees(np.arctan2(x, y))
            bearing = (bearing + 360) % 360
            bearings.append(bearing)
        bearings = np.array(bearings)

        # MDs typically move WNW (240-330 degrees)
        wnd_bearings = bearings[
            (bearings >= 240) & (bearings <= 360) | (bearings >= 0) & (bearings <= 30)
        ]
        if len(wnd_bearings) / len(bearings) < 0.7:
            std_print("INFO", "md", "Movement in WNW (expected >70%)", f"{100 * len(wnd_bearings) / len(bearings):.1f}%")
        else:
            std_print("INFO", "md", "Track direction consistent", f"{100 * len(wnd_bearings) / len(bearings):.1f}% WNW")


def generate_summary_report(
    all_correlation_results,
    all_flux_results,
    all_landfall_results,
    all_classifications,
    all_sm_results,
    all_asm_results,
    all_memory_results,
    output_dir="output",
):
    report_path = os.path.join(output_dir, "Analysis_summary_report.txt")
    with open(report_path, "w") as f:
        f.write("Monsoon depression analysis - summary\n\n")
        # Track Summary
        f.write("1. Track statistics\n")
        if all_classifications and len(all_classifications) > 0:
            df_class = pd.DataFrame(all_classifications)
            short_mds = df_class[~df_class["is_long_lived"]]
            long_mds = df_class[df_class["is_long_lived"]]
            f.write(f"   Total MDs detected:                              {len(df_class)}\n")
            f.write(f"   Short-lived MDs (< 96h):                         {len(short_mds)} ({100 * len(short_mds) / len(df_class):.1f}%)\n")
            f.write(f"   Long-lived MDs (≥ 96h):                          {len(long_mds)} ({100 * len(long_mds) / len(df_class):.1f}%)\n")
            f.write(f"   Mean duration (all):                             {df_class['duration_hours'].mean():.1f} ± {df_class['duration_hours'].std():.1f} hours\n\n")
        else:
            f.write("   No track classification data available\n\n")
        # Precipitation-Soil Moisture Correlation
        f.write("2. Precipitation - soil moisture correlation\n")
        if all_correlation_results and len(all_correlation_results) > 0:
            df_corr = pd.DataFrame(all_correlation_results)
            f.write(f"   Tracks analyzed:                                 {len(df_corr)}\n")
            f.write(f"   Mean correlation (TP vs SM):                     {df_corr['correlation'].mean():.3f}\n")
            f.write(f"   Std Dev:                                         {df_corr['correlation'].std():.3f}\n")
            f.write(f"   Positive correlations:                           {(df_corr['correlation'] > 0).sum()}/{len(df_corr)}\n")
            f.write(f"   Significant (p < 0.05):                          {(df_corr['p_value'] < 0.05).sum()}/{len(df_corr)}\n\n")
        else:
            f.write("   No correlation data available\n\n")
        # Heat Flux Analysis
        f.write("3. Surface heat flux analysis\n")
        if all_flux_results and len(all_flux_results) > 0:
            df_flux = pd.DataFrame(all_flux_results)
            f.write(f"   Tracks with flux data:                           {len(df_flux)}\n")
            f.write(f"   Mean total heat flux:                            {df_flux['mean_total_hf'].mean():.1f} ± {df_flux['mean_total_hf'].std():.1f} W/m²\n")
            f.write(f"   Mean latent heat flux:                           {df_flux['mean_latent_hf'].mean():.1f} W/m²\n")
            f.write(f"   Mean sensible heat flux:                         {df_flux['mean_sensible_hf'].mean():.1f} W/m²\n\n")
        else:
            f.write("   No heat flux data available\n\n")
        # Landfall Flux Transitions
        f.write("4. Landfall flux transitions\n")
        if all_landfall_results and len(all_landfall_results) > 0:
            f.write(f"   Tracks with landfall data:                       {len(all_landfall_results)}\n")
            before_fluxes = [r["before"]["total"] for r in all_landfall_results if r]
            after_fluxes = [r["after"]["total"] for r in all_landfall_results if r]
            if before_fluxes and after_fluxes:
                f.write(f"   Mean before landfall:                            {np.nanmean(before_fluxes):.1f} W/m²\n")
                f.write(f"   Mean after landfall:                             {np.nanmean(after_fluxes):.1f} W/m²\n")
                f.write(f"   Mean change (ocean→land):                        {np.nanmean(after_fluxes) - np.nanmean(before_fluxes):+.1f} W/m²\n\n")
        else:
            f.write("   No landfall flux data available\n\n")
        # Soil Moisture Summary
        f.write("5. Soil moisture analysis\n")
        if (
            all_sm_results is not None
            and not (isinstance(all_sm_results, pd.DataFrame) and all_sm_results.empty)
            and len(all_sm_results) > 0
        ):
            # Check if it's a DataFrame or list
            if isinstance(all_sm_results, pd.DataFrame):
                df_sm = all_sm_results
            else:
                df_sm = pd.DataFrame(all_sm_results)
            f.write(f"   Tracks analyzed:                                 {len(df_sm)}\n")
            f.write(f"   Mean soil moisture:                              {df_sm['mean_swvl1'].mean():.3f} m³/m³\n")
            f.write(f"   Mean precipitation:                              {df_sm['mean_tp'].mean():.2f} mm/hr\n\n")
        else:
            f.write("   No soil moisture data available\n\n")
        f.write("6. Soil Moisture Memory Analysis\n")
        if (
            all_memory_results is not None
            and not (
                isinstance(all_memory_results, pd.DataFrame)
                and all_memory_results.empty
            )
            and len(all_memory_results) > 0
        ):
            if isinstance(all_memory_results, pd.DataFrame):
                df_memory = all_memory_results
            else:
                df_memory = pd.DataFrame(all_memory_results)
            f.write(f"   Tracks analyzed:                                 {len(df_memory)}\n")
            f.write(f"   Mean memory index (24h lag):                     {df_memory['memory_index_24h'].mean():.3f}\n")
            f.write(f"   Std Dev:                                         {df_memory['memory_index_24h'].std():.3f}\n")
            f.write(f"   Range:                                           {df_memory['memory_index_24h'].min():.3f} - {df_memory['memory_index_24h'].max():.3f}\n")
            # High memory (>0.5) indicates strong persistence
            high_memory = (df_memory["memory_index_24h"] > 0.5).sum()
            f.write(f"   Tracks with strong memory (>0.5):                {high_memory}/{len(df_memory)} ({100 * high_memory / len(df_memory):.1f}%)\n\n")
        else:
            f.write("   No soil moisture memory data available\n\n")
        # ASM Summary
        f.write("7. ASM (Antecedent Soil Moisture) analysis\n")
        if all_asm_results is not None:
            if isinstance(all_asm_results, pd.DataFrame):
                if not all_asm_results.empty:
                    df_asm = all_asm_results
                    f.write(f"   Tracks analyzed:                                 {len(df_asm)}\n")
                    # Find ASM column
                    asm_col = None
                    for col in ["ASM", "SWVL1_pre24h_mean", "antecedent_sm"]:
                        if col in df_asm.columns:
                            asm_col = col
                            break
                    if asm_col:
                        f.write(f"   Mean antecedent soil moisture:                   {df_asm[asm_col].mean():.3f} m³/m³\n")
                    else:
                        f.write("   Could not find ASM data column\n")
                else:
                    f.write("   No ASM data available (empty DataFrame)\n")
            elif isinstance(all_asm_results, list) and len(all_asm_results) > 0:
                f.write(f"   Tracks analyzed:                                 {len(all_asm_results)}\n")
                f.write("   ASM data available as list\n")
            else:
                f.write("   No ASM data available\n")
        else:
            f.write("   No ASM data available\n")
    std_print("SAVED", "md", "Summary report", report_path)
    return report_path


if __name__ == "__main__":
    start_time = time.time()

    def ms_to_kmh(speed_ms):
        """Convert m/s to km/h"""
        return speed_ms * 3.6

    std_print("INFO", "md", " --- Applied Thresholds ---")
    # Radius used to associate a detection in time t with time t+1
    std_print(None, None, "Search Radius (Tracking)", f"{TRACKING_SEARCH_RADIUS_KM} km")
    # Radius used to bridge gaps if a system disappears for a few hours
    std_print(None, None, "Search Radius (Linking)", f"{LINKING_SEARCH_RADIUS_KM} km")
    # Wind speed limits to identify Monsoon Depressions (MDs) - based on IMD standards
    wind_lower_kmh = ms_to_kmh(WIND_SPEED_LOWER_BOUND_MS)
    wind_upper_kmh = ms_to_kmh(WIND_SPEED_UPPER_BOUND_MS)
    std_print(None, None, "Wind Speed (10m)", f"{WIND_SPEED_LOWER_BOUND_MS}-{WIND_SPEED_UPPER_BOUND_MS} m/s ({wind_lower_kmh:.1f}-{wind_upper_kmh:.1f} km/h)")
    # Pressure threshold for MDs (low-pressure systems)
    std_print(None, None, "MSLP threshold", f"< {MSL_LOWER_BOUND_PA} Pa ({MSL_LOWER_BOUND_PA / 100:.0f} hPa)")
    # Minimum rain rate to ensure the system is meteorologically active (convective)
    daily_rain_mm = PRECIPITATION_THRESHOLD_MM * 24
    std_print(None, None, "Precipitation (hourly)", f"> {PRECIPITATION_THRESHOLD_MM} mm/hr (≈ {daily_rain_mm:.1f} mm/day)")
    # Relative vorticity has to be positive value to ensure cyclonic motion
    std_print(None, None, "Vorticity", f"> {VORTICITY_THRESHOLD} /s")
    # Minimum duration to filter out short-lived noise or transient weather features
    std_print(None, None, "Min Track Duration", f"{MIN_TRACK_DURATION_HOURS} hours")
    # Minimum displacement to ensures the system is moving/traveling rather than being stationary noise
    std_print(None, None, "Min Displacement", f"{MIN_NET_DISPLACEMENT_KM} km")
    # Physical limit on how fast a depression can reasonably travel
    max_speed_kmh = ms_to_kmh(MAX_SPEED_MS)
    std_print(None, None, "Max Translational Speed", f"{MAX_SPEED_MS} m/s ({max_speed_kmh:.1f} km/h)")
    # Maximum time a system can "disappear" before it's considered dead
    std_print(None, None, "Max Gap allowed", f"{MAX_GAP_HOURS} hours")
    # Stitch bridge large gaps in time
    std_print(None, None, "Stitch Distance", f"{STITCH_DISTANCE_KM} km")
    # Merge very close but similiar magnitude low pressure centres under the radius
    std_print(None, None, "Merge Distance", f"{MERGE_DISTANCE_KM} km")

    print("\n----------------------------------------------- Combining monthly netCDF files --------------------------------------")
    std_print("INFO", "md", " --- Combining all monthly files for track detection ---")
    start_processing_time = time.time()
    num_files = len(MONTH_DIRS) * 3

    result_container = {"ds": None, "success": False, "error": None}
    load_thread = threading.Thread(
        target=load_dataset_async,
        args=(MONTH_DIRS, DATA_DIR, result_container),
    )
    load_thread.daemon = True
    load_thread.start()
    spinner = ProgressSpinner(
        total_files=len(MONTH_DIRS) * 31,
        description=f"Loading ERA5 data for {MONSOON_YEAR} ({len(MONTH_DIRS)} months)",
    )
    step = 0
    while load_thread.is_alive():
        spinner.update(current=step)
        time.sleep(0.2)
        step += 1
    elapsed_time, _ = spinner.finish()
    start_processing_time = time.time() - elapsed_time
    if result_container["success"]:
        ds = result_container["ds"]
        std_print("INFO", "md", "Successfully loaded all monthly files", f"{len(MONTH_DIRS)} months combined")
        std_print("INFO", "md", "Time for load and combine files", f"{elapsed_time:.2f} seconds")
    else:
        std_print("ERROR", "md", "Loading files", f"{result_container['error']}")
        sys.exit(1)
    if result_container["success"]:
        ds = result_container["ds"]
        end_processing_time = time.time()
        std_print("INFO", "md", "Time for load and combine files", f"{end_processing_time - start_processing_time:.2f} seconds")
        # Diagnose temporal issues
        diag_report = diagnose_temporal_issues(ds)
        std_print("INFO", "md", " --- Applying temporal fix ---")
        std_print(None, None, "Original dataset has variable time spacing")
        std_print(None, None, "Resampling to regular 1-hour grid...")
        ds = fix_temporal_resolution(ds, target_resolution_hours=1)
        is_valid, val_report = validate_temporal_consistency(
            ds, expected_hours=1, tolerance=0.05
        )
        if not is_valid:
            std_print("WARN", "md", "Some intervals still outside tolerance, This may affect tracking accuracy")
        else:
            std_print("INFO", "md", "Temporal resolution successfully regularized")
    else:
        std_print("ERROR", "md", "Loading files", f"{result_container['error']}")
        sys.exit(1)

    print("\n------------------------------------------------ Dataset Info ------------------------------------------------------")
    # Temporal info
    ds, time_coord = get_time_coord(ds)
    years = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%Y").unique()
    months = (
        pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%b").unique()
    )
    hours = (
        pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%H:%M").unique()
    )
    time_values = ds[time_coord].values
    time_diff = time_values[1] - time_values[0]
    time_diff = pd.to_timedelta(time_diff)
    total_seconds = time_diff.total_seconds()
    resolution_hours = total_seconds / 3600.0
    # Spacial info
    ds_lat_min = ds.latitude.min().item()
    ds_lat_max = ds.latitude.max().item()
    ds_lon_min = ds.longitude.min().item()
    ds_lon_max = ds.longitude.max().item()

    std_print(None, None, "Year", f"{MONSOON_YEAR}")
    std_print(None, None, "Months", f"{', '.join(MONTH_DIRS)}")
    std_print(None, None, "Hours", hours)
    std_print(None, None, "Variables", variables)
    std_print(None, None, "Dataset bounds", f"lat [{ds_lat_min:.1f}, {ds_lat_max:.1f}] N, lon [{ds_lon_min:.1f}, {ds_lon_max:.1f}] E")
    std_print(None, None, "    using clipped bounds", f"lat [{clipped_min_lat}, {clipped_max_lat}] N, lon [{clipped_min_lon}, {clipped_max_lon}] E")
    std_print(None, None, "Data Time Resolution", f"{resolution_hours:.1f} hours ({time_diff})")

    print(
        "\n-------------------------------------------------- Processing ----------------------------------------"
    )
    full_track_df = process_files(ds)
    if full_track_df is not None and not full_track_df.empty:
        if "track_id" not in full_track_df.columns:
            std_print("ERROR", "md", "`track_id` column not found")
            sys.exit(1)
        plot_tracks(full_track_df, output_dir)
        export_track_coordinates(full_track_df, output_dir)

        # FIX: even after clipping it is using full extend (Detected)
        pad = 0.5
        try:
            full_extent = [
                ds_lon_min - pad,
                ds_lon_max + pad,
                ds_lat_min - pad,
                ds_lat_max + pad,
            ]
            std_print("DEBUG", "md", "Full extent (Detected)", full_extent)
        except Exception:
            # full_extent = [70, 100, 5, 30]
            full_extent = [70, 95, 10, 30]
            std_print("DEBUG", "md", "Full extent (Mannual)", full_extent)

        full_track_df["is_land"] = full_track_df.apply(lambda row: is_point_land(row["latitude"], row["longitude"]), axis=1)
        unique_track_ids = sorted([int(tid) for tid in full_track_df["track_id"].unique()])
        std_print("INFO", "md", "MD tracks for individual export", len(unique_track_ids))
        # Results containers
        all_correlation_results = []
        tp_series = {}
        swvl1_series = {}
        valid_times = {}
        all_flux_results = []
        all_landfall_results = []
        all_classifications = []
        asm_results_list = []
        track_dict = {}
        flux_dict = {}
        all_sm_results = []
        all_asm_results = []
        all_memory_results = []
        fsi_df = pd.DataFrame()
        amplitude_df = pd.DataFrame()
        coupling_efficiency_df = pd.DataFrame()
        # Process each track
        for track_id in unique_track_ids:
            track_id = int(track_id)
            track_df = full_track_df[full_track_df["track_id"] == track_id].copy()
            track_dict[track_id] = track_df.copy()  # for coupling metrics
            track_dir = os.path.join(output_dir, f"track_{int(track_id)}")
            os.makedirs(track_dir, exist_ok=True)
            track_df = (full_track_df[full_track_df["track_id"] == track_id].reset_index(drop=True).copy())

            print(f"\n----------------------------------------------- MD Track {track_id} --------------------------------------------------------")
            # 1) Land/Ocean classification
            landfall_time = find_landfall_time(track_df)  # Ensure is_land column exists
            std_print("INFO", "md", "Land/ocean classification computed", f"{track_df['is_land'].sum()} land points")

            # 2) Smooth coordinates
            track_df = smooth_track_coordinates(track_df, window_size=3, iterations=4)

            # 3) Diagnose detections
            diagnose_detection_issues(ds, track_df)

            # 4) Plot maps
            plot_track_full_map(track_df, track_id, full_extent, track_dir)
            # std_print('DEBUG', 'md', f'Skipping Plotting single Track {track_id}')
            plot_single_track(track_df, track_id, track_dir)

            # 5) Export coordinates
            export_single_track_coordinates(track_df, track_id, track_dir)

            # 6) Export detections csv
            track_csv = os.path.join(track_dir, f"track_{int(track_id)}_detections.csv")
            track_df.to_csv(track_csv, index=False)
            std_print("EXPORT", "md", f"Track {track_id} Detections to CSV", track_csv)

            # 7) Plot speed analysis
            print("\n------------------------------------------------ Translational Speed Plots ----------------------------------------")
            std_print("INFO", "md", f"Translational Speedf Analysis for Track {track_id}")
            speed_dir = os.path.join(track_dir, "translational-speed")
            os.makedirs(speed_dir, exist_ok=True)
            track_df = compute_track_speeds(track_df)
            track_csv = os.path.join(
                speed_dir, f"track_{int(track_id)}_detections_augmented.csv"
            )
            track_df.to_csv(track_csv, index=False)
            std_print("EXPORT", "md", f"Augmented track data for Track {track_id}", track_csv)
            plot_speed_analysis_kmh(track_df, track_id, speed_dir)
            plot_speed_analysis_ms(track_df, track_id, speed_dir)
            # track_df_augmented = analyze_translational_speed(track_df, output_dir=speed_dir)
            speed_kmh = track_df["speed_kmh"].dropna()
            speed_ms = track_df["speed_ms"].dropna()
            distance_km = track_df["distance_km"].dropna()
            std_print("INFO", "md", f"Speed Statistics for Track {track_id}")
            std_print(None, None, "mean", f"{speed_kmh.mean():.2f} km/h ({speed_ms.mean():.2f} m/s)")
            std_print(None, None, "median", f"{speed_kmh.median():.2f} km/h ({speed_ms.median():.2f} m/s)")
            std_print(None, None, "std deviation", f"{speed_kmh.std():.2f} km/h ({speed_ms.std():.2f} m/s)")
            std_print(None, None, "min/max", f"{speed_kmh.min():.2f} / {speed_kmh.max():.2f} km/h")
            std_print(None, None, "total distance", f"{distance_km.sum():.1f} km")
            std_print(None, None, "duration", f"{(track_df['time'].max() - track_df['time'].min()).total_seconds() / 3600:.1f} hours")

            # 8) Generate Hovmöller Plots
            print("\n------------------------------------------------ Azimuthal Hovmoller Plots ----------------------------------------")
            # std_print('DEBUG', 'md', f'Skipping Hovmöller for Track {track_id}')
            std_print("INFO", "md", f"Computing Hovmöller for Track {track_id}")
            hov_dir = os.path.join(track_dir, "hovmoller")
            os.makedirs(hov_dir, exist_ok=True)
            try:
                radii_km = [0] + radii
                hov_dir, times, landfall_idx = compute_azimuthal_hovmoller_sectors(ds, track_df, radii_km, variables, n_theta=16)
                hov_avg = average_azimuthally(hov_dir, variables)
                for var in variables:
                    plot_hovmoller_azimuthal_average(hov_avg[var], times, radii_km, var, track_df, track_id, output_dir, landfall_idx,)
            except Exception as e:
                std_print("ERROR", "md", f"Hovmöller computation failed for Track {track_id}", str(e))
                traceback.print_exc()

            # 9) Antecedent Soil Moisture Analysis
            print("\n---------------------------------------------- Antecedent Soil Moisture Analysis ----------------------------------")
            # std_print("DEBUG", "md", f"Skipping antecedent SM for Track {track_id}")
            # Ensure is_land column exists for antecedent analysis
            # track_df['is_land'] = track_df.apply(lambda row: is_point_land(row['latitude'], row['longitude']), axis=1)
            # std_print('DEBUG', 'md', 'Land/ocean classification computed', f'{track_df["is_land"].sum()} land points')
            try:
                asm_dir = os.path.join(track_dir, "antecedent_sm")
                os.makedirs(asm_dir, exist_ok=True)
                analyze_antecedent_trend(ds, track_df, track_id, output_dir=asm_dir)
                plot_antecedent_spatial_map(ds, track_df, track_id, hours_before=24, output_dir=asm_dir)
            except Exception as e:
                std_print("ERROR", "md", f"Track {track_id} ASM analysis failed", str(e))
                traceback.print_exc()

            # 10) Correlation analysis (swvl1 vs tp vs heat flux)
            print("\n------------------------------------------------ Correlation analysis ---------------------------------------------")
            # std_print("DEBUG", "md", f"Skipping Correlation Analysis for Track {track_id}")
            std_print("INFO", "md", f"Track {track_id}: Correlation analysis")
            correlation_dir = os.path.join(track_dir, "correlation_analysis")
            os.makedirs(correlation_dir, exist_ok=True)
            correlation_result = analyze_single_track(ds, track_df, track_id, buffer_km=100, max_lag_hours=24, output_dir=correlation_dir)
            try:
                # FIX: If analyze_single_track returns None, line 'landfall_time': correlation_result.get('landfall_time') will fail
                if correlation_result:
                    all_correlation_results.append(
                        {
                            "track_id": track_id,
                            "correlation": correlation_result["correlation"],
                            "p_value": correlation_result["p_value"],
                            "mean_tp": correlation_result["mean_tp"],
                            "mean_swvl1": correlation_result["mean_swvl1"],
                            "n_samples": correlation_result["n_samples"],
                            "landfall_time": correlation_result.get("landfall_time"),
                            "landfall_index": correlation_result.get("landfall_index"),
                            "tp_series": tp_series,
                            "swvl1_series": swvl1_series,
                            "valid_times": valid_times,
                        }
                    )
            except Exception as e:
                std_print("ERROR", "md", "Correlation analysis failed", str(e))

            # 11) Heat Flux Analysis and MD Duration Classification
            print("\n---------------------------------------------- Heat Flux & Duration Analysis --------------------------------------")
            # std_print("DEBUG", "md", f"Skipping Heat Flux and Duration Analysis for Track {track_id}")
            std_print("INFO", "md", f"Track {track_id}: Heat Flux and Duration Analysis")
            classification = classify_md_duration(track_df)
            classification["track_id"] = track_id
            all_classifications.append(classification)
            std_print("DEBUG", "md", f"Classification Track {track_id} is classified as {classification['category']}")
            std_print(None, None, "Duration", f"{classification['duration_hours']:.1f} hours ({classification['num_points']} points)")
            heatflux_dir = os.path.join(track_dir, "heat_flux_analysis")
            os.makedirs(heatflux_dir, exist_ok=True)
            flux_df = extract_heat_flux_timeseries(ds, track_df, track_id, buffer_km=100)
            try:
                if flux_df is not None:
                    flux_dict[track_id] = flux_df.copy()  # store for coupling metrics
                    # std_print('DEBUG', 'md', f'Track {track_id} flux columns', list(flux_df.columns))
                    std_print("INFO", "md", "Heat flux data extracted", f"{len(flux_df)} points")
                    flux_analysis = analyze_heat_flux_evolution(flux_df, track_df, track_id)
                    if flux_analysis:
                        all_flux_results.append(
                            {
                                "track_id": track_id,
                                "is_long_lived": classification["is_long_lived"],
                                "duration_hours": classification["duration_hours"],
                                "category": classification["category"],
                                **flux_analysis,
                            }
                        )
                        std_print("INFO", "md", f"Heat flux analysis for track {track_id} :")
                        std_print(None, None, "Mean Total HF", f"{flux_analysis['mean_total_hf']:.1f} W/m²")
                        std_print(None, None, "Mean Latent HF", f"{flux_analysis['mean_latent_hf']:.1f} W/m²")
                        std_print(None, None, "Mean Sensible HF", f"{flux_analysis['mean_sensible_hf']:.1f} W/m²")
                        if ("ocean_mean_total_hf" in flux_analysis and "land_mean_total_hf" in flux_analysis):
                            std_print(None, None, "Ocean phase Total HF", f"{flux_analysis['ocean_mean_total_hf']:.1f} W/m²")
                            std_print(None, None, "Land phase Total HF ", f"{flux_analysis['land_mean_total_hf']:.1f} W/m²")
                            std_print(None, None, "Change (Ocean->Land)", f"{flux_analysis['total_hf_change']:+.1f} W/m²")
                else:
                    std_print("WARN", "md", f"Track {track_id}", "No heat flux data")
            except Exception as e:
                std_print("ERROR", "md", f"Track {track_id} Heat Flux analysis failed", str(e))
                traceback.print_exc()
            try:
                std_print("INFO", "md", f" --- Track {track_id}: Landfall transition ---")
                landfall_result = analyze_flux_transition(
                    ds, track_df, track_id, hours_before=24, hours_after=24
                )
                if landfall_result:
                    std_print("INFO", "md", "Before landfall :")
                    std_print(None, None, "Latent", f"{landfall_result['before']['latent']:.1f} W/m²")
                    std_print(None, None, "Sensible", f"{landfall_result['before']['sensible']:.1f} W/m²")
                    std_print("INFO", "md", "After landfall  :")
                    std_print(None, None, "Latent", f"{landfall_result['after']['latent']:.1f} W/m²")
                    std_print(None, None, "Sensible", f"{landfall_result['after']['sensible']:.1f} W/m²")
                    std_print("INFO", "md", "Change          :")
                    std_print(None, None, "Total", f"{landfall_result['change']['total']:+.1f} W/m²")
                    all_landfall_results.append(landfall_result)
                    landfall_csv = os.path.join(
                        heatflux_dir, f"track_{track_id}_landfall_flux.csv"
                    )
                    pd.DataFrame([landfall_result]).to_csv(landfall_csv, index=False)
                    std_print("EXPORT", "md", "Landfall Flux transition analysis", landfall_csv)
                else:
                    std_print("WARN", "md", f"Track {track_id}", "Could not analyze Landfall flux transition")
            except Exception as e:
                std_print("ERROR", "md", f"Track {track_id} Landfall Flux analysis failed", str(e))
                traceback.print_exc()

            # 12) Time series
            print("\n------------------------------------------------------- Time Series -----------------------------------------------")
            std_print("INFO", "md", f"Track {track_id}: Time Series")
            timeseries_dir = os.path.join(track_dir, "time_series")
            os.makedirs(timeseries_dir, exist_ok=True)
            try:
                std_print("INFO", "md", f" --- TP-SWVL1 time series analysis for track {track_id} ---")
                if correlation_result:
                    # Check if the required keys exist in correlation_result
                    if all(key in correlation_result for key in ["tp_series", "swvl1_series", "valid_times"]):
                        tp_swvl1_timeseries(
                            tp=correlation_result["tp_series"],
                            swvl1=correlation_result["swvl1_series"],
                            valid_times=correlation_result["valid_times"],
                            landfall_index=correlation_result.get("landfall_index", -1),
                            track_id=track_id,
                            track_df=track_df,
                            output_dir=timeseries_dir,
                            originates_over_land=correlation_result.get("originates_over_land", True)
                        )
                    else:
                        std_print("WARN", "md", f"Track {track_id}", "Time series data not available in correlation_result")
                std_print("INFO", "md", f" --- Heat flux time series analysis for track {track_id} ---")
                hf_timeseries(
                    flux_df=flux_df,
                    track_id=track_id,
                    output_dir=timeseries_dir
                )
            except Exception as e:
                std_print("ERROR", "md", f"Track {track_id} Time series failed", str(e))
                traceback.print_exc()

            # 13) Soil moisture analysis
            print("\n------------------------------------------------ Soil Moisture Analysis -------------------------------------------")
            # std_print("DEBUG", "md", f"Skipping Soil Moisture Analysis for Track {track_id}")
            std_print("INFO", "md", f"Track {track_id}: Soil moisture Analysis")
            try:
                sm_result = compute_single_track_correlation(ds=ds, track_df=track_df, track_id=track_id, buffer_km=100)
                if sm_result:
                    all_sm_results.append(sm_result)
                    sm_dir = os.path.join(track_dir, "soilmoiture_analysis")
                    os.makedirs(sm_dir, exist_ok=True)
                    plot_spatial_tp_swvl1(ds, track_df, track_id, buffer_km=150, output_dir=sm_dir)
                    std_print("SAVED", "md", f"Track {track_id} Soil moisture analysis plots", sm_dir)
                    try:
                        swvl1_vals = []
                        ds, time_coord = get_time_coord(ds)
                        for _, row in track_df.iterrows():
                            try:
                                # FIX: use extract_and_average_variable()
                                subset = extract_spatial_subset(ds, row["latitude"], row["longitude"], pd.to_datetime(row["time"]), buffer_km=100, time_coord=time_coord,)
                                if subset is not None and "swvl1" in subset.data_vars:
                                    sm_val = float(subset["swvl1"].mean().values)
                                    if not np.isnan(sm_val):
                                        swvl1_vals.append(sm_val)
                            except Exception as e:
                                std_print("ERROR", "md", "Continuing...", str(e))
                                continue
                        if len(swvl1_vals) > 48:  # Need enough data for 24h lag
                            std_print("DEBUG", "md", f"Computing Soil moisture Memory for Track {track_id}")
                            memory_24h = compute_soil_moisture_memory(swvl1_vals, lag_hours=24)
                            memory_result = {
                                "track_id": track_id,
                                "memory_index_24h": memory_24h,
                                "n_samples": len(swvl1_vals),
                                "mean_swvl1": np.mean(swvl1_vals),
                                "std_swvl1": np.std(swvl1_vals),
                            }
                            all_memory_results.append(memory_result)
                            std_print("INFO", "md", f"Soil moisture memory for track {track_id}")
                            std_print(None, None, "Memory index (24h lag)", f"{memory_24h:.3f}")
                            std_print(None, None, "Mean SWVL1", f"{np.mean(swvl1_vals):.3f} m³/m³")
                            std_print(None, None, "Samples", len(swvl1_vals))
                            # Save individual track memory results
                            memory_csv = os.path.join(sm_dir, f"track_{track_id}_soil_memory.csv")
                            pd.DataFrame([memory_result]).to_csv(memory_csv, index=False)
                            std_print("EXPORT", "md", f"Track {track_id} soil memory analysis", memory_csv)
                        else:
                            std_print("WARN", "md", f"Track {track_id}", "Insufficient data for soil memory analysis")
                    except Exception as e:
                        std_print("ERROR", "md", f"Soil moisture memory analysis failed for Track {track_id}", str(e))
                        traceback.print_exc()
            except Exception as e:
                std_print("ERROR", "md", f"Track {track_id} Soil moisture analysis failed", str(e))

            # 14) Elevation analysis
            # FIX:
            # print("\n------------------------------------------------- Elevation Analyses ----------------------------------------------")
            # try:
            #     plot_tracks_elvation_mosaic(track_df=track_df, output_dir='output')
            # except Exception as e:
            #     std_print('ERROR', 'md', f'Track {track_id} Soil moisture analysis failed', str(e))

        # 1. Global Coupling metrics
        std_print("INFO", "md", " --- Coupling Metrics Analysis ---")
        # std_print("DEBUG", "md", "Skipping Coupling Metrics Analysis")
        correlation_df = (pd.DataFrame(all_correlation_results) if all_correlation_results else pd.DataFrame())
        duration_df = (pd.DataFrame(all_classifications) if all_classifications else pd.DataFrame())
        coupling_metrics_dir = os.path.join(output_dir, "coupling_metrics")
        os.makedirs(coupling_metrics_dir, exist_ok=True)
        if all_flux_results:
            flux_results_df = pd.DataFrame(all_flux_results)
        else:
            flux_results_df = pd.DataFrame()
        try:
            coupling_metrics_dir = os.path.join(output_dir, "coupling_metrics")
            os.makedirs(coupling_metrics_dir, exist_ok=True)
            # Compute FSI
            if not correlation_df.empty and track_dict:
                std_print("INFO", "md", "Computing Feedback Strength Index (FSI)")
                fsi_df = compute_fsi_all_tracks(correlation_df, track_dict, ds)
                if not fsi_df.empty:
                    fsi_csv = os.path.join(coupling_metrics_dir, "coupling_metrics_FSI.csv")
                    fsi_df.to_csv(fsi_csv, index=False)
                    std_print("EXPORT", "md", "FSI to CSV", fsi_csv)
            else:
                std_print("WARN", "md", "Insufficient data for FSI computation")
                fsi_df = pd.DataFrame()
            # Compute Diurnal Amplitude
            if flux_dict:
                std_print("INFO", "md", "Computing Diurnal SHF Amplitude")
                amplitude_df = compute_diurnal_amplitude_all_tracks(
                    flux_dict, track_dict
                )
            else:
                std_print("WARN", "md", "No flux data available for diurnal amplitude")
                amplitude_df = pd.DataFrame()
            # Compute Coupling Efficiency
            if not fsi_df.empty and not amplitude_df.empty and not duration_df.empty:
                std_print("INFO", "md", "Computing Coupling Efficiency")
                coupling_efficiency_df = compute_coupling_efficiency(fsi_df, correlation_df, amplitude_df, duration_df)
                export_coupling_metrics(fsi_df, amplitude_df, coupling_efficiency_df, coupling_metrics_dir)
                generate_coupling_metrics_report(fsi_df, amplitude_df, coupling_efficiency_df, coupling_metrics_dir)
            else:
                std_print("WARN", "md", "Insufficient data for coupling efficiency computation")
            if not correlation_df.empty and not fsi_df.empty:
                fsi_df = fsi_df.merge(
                    correlation_df[["track_id", "correlation", "p_value", "mean_tp", "mean_swvl1"]],
                    on="track_id",
                    how="left",
                    suffixes=("", "_corr"),
                )
                fsi_path = os.path.join(coupling_metrics_dir, "coupling_metrics_FSI.csv")
                fsi_df.to_csv(fsi_path, index=False)
                std_print("EXPORT", "md", "FSI metrics with TP/SWVL1", fsi_path)
                # FIX: verify  FSI data
                # if os.path.exists(path):
                #     df = pd.read_csv(path)
                #     std_print('DEBUG', 'md', ' FSI shape', f'{df.shape}')
                #     std_print('DEBUG', 'md', ' FSI columns', str(list(df.columns)))
                #     # Verify it has the right columns
                #     required_cols = ['mean_tp', 'mean_swvl1']
                #     missing = [col for col in required_cols if col not in df.columns]
                #     if missing:
                #         std_print('ERROR', 'md', ' FSI missing columns', str(missing))
                #         # Fix by merging again
                #         df = fsi_df.merge(
                #             correlation_df[['track_id', 'mean_tp', 'mean_swvl1']],
                #             on='track_id',
                #             how='left'
                #         )
                #         df.to_csv(path, index=False)
            else:
                std_print("WARN", "md", "Cannot enhance FSI data", "Missing correlation or FSI data")
        except Exception as e:
            std_print("ERROR", "md", "Coupling metrics analysis failed", str(e))
            traceback.print_exc()
        # 2. Global Antecedent soil moisture Analysis
        std_print("info", "md", " --- Antecedent soil moisture analysis ---")
        # std_print("DEBUG", "md", "Skipping Global Antecedent soil moisture Analysis")
        try:
            asm_dir = os.path.join(output_dir, "global_antecedent_analysis")
            os.makedirs(asm_dir, exist_ok=True)
            asm_results_df = analyze_antecedent_for_all_tracks(ds=ds, track_df=full_track_df, hours_before=24, output_dir=asm_dir)
            all_asm_results = asm_results_df
            # Export the combined results for all tracks
            if asm_results_df is not None:
                csv_path = os.path.join(asm_dir, "all_tracks_antecedent_sm.csv")
                asm_results_df.to_csv(csv_path, index=False)
                std_print("EXPORT", "md", "Global Antecedent SM Results", csv_path)
            # ASM results reuse
            if (
                all_asm_results is not None
                and isinstance(all_asm_results, pd.DataFrame)
                and not all_asm_results.empty
            ):
                asm_df = all_asm_results.copy()
                std_print("INFO", "md", "Using ASM data from antecedent analysis", f"{len(asm_df)} tracks")
        except Exception as e:
            std_print("ERROR", "md", "Global antecedent analysis failed", str(e))
            traceback.print_exc()
        # 3. Global Soil Moisture Memory Analysis
        std_print("INFO", "md", " --- Global Soil Moisture Memory Analysis ---")
        # std_print("DEBUG", "md", "Skipping Global Soil Moisture Memory Analysis")
        try:
            memory_dir = os.path.join(output_dir, "global_soil_memory")
            os.makedirs(memory_dir, exist_ok=True)
            if "all_memory_results" in locals() and all_memory_results:
                memory_df = pd.DataFrame(all_memory_results)
                memory_csv_path = os.path.join(memory_dir, "all_tracks_soil_memory.csv")
                memory_df.to_csv(memory_csv_path, index=False)
                std_print("EXPORT", "md", "Global soil memory results", memory_csv_path)
                if not memory_df.empty:
                    std_print("INFO", "md", "Soil Moisture Memory Summary :")
                    std_print(None, None, "Mean memory index (24h)", f"{memory_df['memory_index_24h'].mean():.3f}")
                    std_print(None, None, "Std dev memory index", f"{memory_df['memory_index_24h'].std():.3f}")
                    std_print(None, None, "Min memory index", f"{memory_df['memory_index_24h'].min():.3f}")
                    std_print(None, None, "Max memory index", f"{memory_df['memory_index_24h'].max():.3f}")
            else:
                std_print("WARN", "md", "No soil memory results available for global analysis")
        except Exception as e:
            std_print("ERROR", "md", "Global soil memory analysis failed", str(e))
            traceback.print_exc()
        # 4. Global Elevation Analysis
        print("\n------------------------------------------------ Elevation Analyses ----------------------------------------------")
        std_print("DEBUG", "md", "Skipping Global Elevation Analysis")
        # try:
        #     fsi_df = compute_fsi_all_tracks(correlation_df, track_dict, ds)
        #     hdt_dir = os.path.abspath("../Data/Elevation/HGT-files/")
        #     elevation_metrics = run_elevation_analysis(tracks_df=full_track_df, coupling_df=fsi_df, output_dir=output_dir, hgt_dir=hdt_dir,)
        #     if not elevation_metrics.empty:
        #         std_print("INFO", "md", "Elevation analysis complete", f"{len(elevation_metrics)} tracks with TSI computed")
        #         std_print(None, None, f"  Mean TSI: {elevation_metrics['TSI'].mean():.3f}")
        #         std_print(None, None, f"  TSI range: [{elevation_metrics['TSI'].min():.3f}, {elevation_metrics['TSI'].max():.3f}]")
        #     else:
        #         std_print("ERROR", "md", "Elevation analysis failed", "Check HGT data availability and paths")
        # except Exception as e:
        #     std_print("ERROR", "md", "Statistical analyses failed", str(e))
        #     traceback.print_exc()
        # 5. Global Statistical Analysis
        print("\n------------------------------------------------ Statistical Analyses ----------------------------------------------")
        # std_print("DEBUG", "md", "Skipping Global Statistical Analysis")
        try:
            if (
                all_asm_results is not None
                and isinstance(all_asm_results, pd.DataFrame)
                and not all_asm_results.empty
            ):
                asm_df = all_asm_results.copy()
                std_print("INFO", "md", "Using ASM data from antecedent analysis", f"{len(asm_df)} tracks")
            else:
                std_print("WARN", "md", "No ASM data available", "Skipping statistical analyses")
                asm_df = pd.DataFrame()
            # clay_data_path for H2 analysis
            clay_data_path = os.path.join("..", "Data", "soil", "clay_percentage.csv")
            # Only run statistical analyses if we have both coupling and ASM data
            if not asm_df.empty and os.path.exists(
                os.path.join(coupling_metrics_dir, "coupling_metrics_FSI.csv")
            ):
                stats_results = run_all_statistical_analyses(
                    coupling_metrics_dir=coupling_metrics_dir,
                    asm_results=asm_df,
                    correlation_df=correlation_df
                    if "correlation_df" in locals()
                    else None,
                    output_dir=output_dir,
                )
                if stats_results and len(stats_results) > 0:
                    std_print("INFO", "md", "Statistical analyses completed successfully")
                    summary_report_path = os.path.join(
                        output_dir, "Analysis_summary_report.txt"
                    )
                    elevation_df = load_elevation_metrics(output_dir="output")
                    h7_results = analyze_tsi(
                        elevation_df,
                        fsi_df,
                        # coupling_df,
                        output_dir="output",
                    )
                    if os.path.exists(summary_report_path):
                        with open(summary_report_path, "a") as f:
                            f.write("\n7. Statistical Analysis Results\n\n")
                            if stats_results.get("asm_fsi"):
                                f.write(
                                    f"(ASM-FSI): r = {stats_results['asm_fsi'].get('correlation', 'N/A'):.3f}\n"
                                )
                            if stats_results.get("gam"):
                                gam_r2 = stats_results["gam"].get("r_squared", "N/A")
                                r2_str = (
                                    f"{gam_r2:.3f}"
                                    if isinstance(gam_r2, (int, float))
                                    else "N/A"
                                )
                                f.write(f"(GAM plateau): r-squared = {r2_str}\n")
                            if stats_results.get("granger"):
                                f.write(
                                    f"(Granger causality): {stats_results['granger'].get('n_significant_tracks', 'N/A')} tracks significant\n"
                                )
                            tsi_dur_corr = stats_results.get("tsi", {}).get(
                                "tsi_duration_correlation"
                            )
                            if tsi_dur_corr is not None and isinstance(
                                tsi_dur_corr, (int, float, np.number)
                            ):
                                f.write(f"(TSI-duration): r = {tsi_dur_corr:.3f}\n")
                            else:
                                f.write(f"(TSI-duration): r = N/A\n")
                else:
                    std_print("WARN", "md", "Statistical analyses returned no results")
            else:
                std_print("WARN", "md", "Insufficient data for statistical analyses", "Check coupling metrics and ASM files")
        except Exception as e:
            std_print("ERROR", "md", "Statistical analyses failed", str(e))
            traceback.print_exc()
        print("\nDEBUG:")
        std_print(None, None, f"Total tracks: {len(full_track_df['track_id'].unique()) if full_track_df is not None else 0}")
        std_print(None, None, f"Correlation results: {len(all_correlation_results)}")
        std_print(None, None, f"FSI data shape: {fsi_df.shape if fsi_df is not None else 'None'}")
        std_print(None, None, f"FSI columns: {list(fsi_df.columns) if fsi_df is not None else 'None'}")
        if fsi_df is not None and not fsi_df.empty:
            missing_for_stats = []
            for col in ["TP", "SWVL1", "correlation", "p_value"]:
                if col not in fsi_df.columns:
                    missing_for_stats.append(col)
            if missing_for_stats:
                std_print("WARN", "Missing columns for stats", str(missing_for_stats))
                std_print("DEBUG", f"Available columns : {list(fsi_df.columns)}")
        print("\n")
        # 6. Global Flux Analysis
        print("\n------------------------------------------------ Global Flux Analysis ----------------------------------------------")
        # std_print("DEBUG", "md", "Skipping Heat FLux Analysis")
        std_print("INFO", "md", " --- Global Heat Flux Analysis ---")
        if all_flux_results:
            all_flux_results = ensure_flux_results_complete(
                all_flux_results, all_classifications
            )
            export_heat_flux_results(all_flux_results, output_dir)
        if all_landfall_results:
            export_landfall_flux_results(all_landfall_results, output_dir)
        else:
            std_print("WARN", "md", "No Global heat flux transition results to export")
        # 7. Final Summaries and Report Generation
        print("\n-------------------------------------------------- Global Summaries -------------------------------------------------")
        # std_print("DEBUG", "md", "Skipping Global Summaries")
        std_print("INFO", "md", " --- Overall Classification Summary ---")
        if all_classifications:
            print_md_classification_summary(all_classifications, output_dir)
        std_print("INFO", "md", " --- Generating Overall Summary Report ---")
        try:
            generate_summary_report(
                all_correlation_results if all_correlation_results else [],
                all_flux_results if all_flux_results else [],
                all_landfall_results if all_landfall_results else [],
                all_classifications if all_classifications else [],
                all_sm_results
                if "all_sm_results" in locals() and all_sm_results
                else [],
                all_asm_results if all_asm_results is not None else [],
                all_memory_results
                if "all_memory_results" in locals() and all_memory_results
                else [],
                output_dir,
            )
        except Exception as e:
            std_print("ERROR", "md", "Overall summary report generation failed", str(e))
    else:
        std_print("INFO", "md", "No tracks found after processing; skipping global analysis")
    print()
    end_time = time.time()
    total_seconds = end_time - start_time
    if total_seconds < 60:
        std_print(None, None, f"Executed in {total_seconds:.2f} seconds.")
    elif total_seconds < 3600:
        mins = int(total_seconds // 60)
        secs = int(total_seconds % 60)
        std_print(None, None, f"Executed in {mins} mins {secs} secs.")
    else:
        hrs = int(total_seconds // 3600)
        mins = int((total_seconds % 3600) // 60)
        secs = int(total_seconds % 60)
        std_print(None, None, f"Executed in {hrs} hrs {mins} mins {secs} secs.")
