'''
1. No Memory Management for Large Datasets
Location: md_detection_tracking-13.py, load_dataset_async()
Issue: .compute() loads entire dataset into memory
.compute()  # Could cause OOM for very large datasets

2. Inconsistent Landfall Detection
Location: Multiple files (antecedent_analysis.py, correlation_analysis.py)
Issue: Different logic for detecting landfall in different functions
antecedent_analysis.py: is_point_land()
correlation_analysis.py: row.get('is_land', False)
Fix: Standardize to use pre-computed 'is_land' column

3. Incomplete Error Handling in plot_flux_analysis()
Location: correlation_analysis.py
Issue: Tries to plot with insufficient data without proper fallback

4. Inconsistent Parameter Names
Location: Multiple functions use buffer_km vs buffer_deg
# antecedent_analysis.py uses buffer_km=100
# plot_sm_tp_map() uses buffer_deg=5.0
'''

import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from numba import none
from numba.core.serialize import custom_rebuild
from numba.core import analysis
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom

from utils import (
    std_print,
    is_point_land,
    find_landfall_time,
    setup_output_dir,
    export_results_to_csv,
    find_origin_type,
    classify_track_status,
    extract_spatial_subset,
    calculate_buffer_deg,
    clamp_to_domain,
    haversine_distance,
    standardize_time_coord,
    setup_map_extent,
    add_land_ocean_shading
)

from azimuthal_hovmoller import (
    compute_azimuthal_hovmoller,
    plot_hovmoller
)

from speed_analysis import (
    compute_track_speeds,
    plot_speed_analysis_kmh,
    plot_speed_analysis_ms
)

from antecedent_analysis import (
    compute_antecedent_sm,
    compute_antecedent_sm_multiwindow,
    compute_antecedent_sm_post_landfall,
    export_multiwindow_results,
    compute_antecedent_sm_all_tracks,
    compute_asm_tp_correlation,
    compute_asm_duration_correlation,
    create_asm_summary_report,
    export_antecedent_results,
    plot_antecedent_sm_timeseries,
    plot_sm_tp_map,
)

from correlation_analysis import (
    analyze_single_track,
    analyze_flux_correlations,
    plot_flux_analysis
)
from soilmoiture_analysis import (
    compute_single_track_correlation,
    analyze_all_tracks,
    plot_track_timeseries,
    plot_spatial_tp_sm,
    export_results,
)

from heatflux_analysis import (
    classify_md_duration,
    extract_heat_flux_timeseries,
    analyze_heat_flux_evolution,
    analyze_flux_transition,
    plot_heat_flux_timeseries,
    export_heat_flux_results,
    export_landfall_flux_results,
    print_md_classification_summary
)

from coupling_metrics import (
    compute_fsi_all_tracks,
    compute_diurnal_amplitude_all_tracks,
    compute_coupling_efficiency,
    export_coupling_metrics,
    generate_coupling_metrics_report,
)

import traceback

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    std_print('INFO', 'md', 'Created output directory', output_dir)
else:
    std_print('INFO', 'md', 'Using existing output directory', output_dir, '\n')

DATA_DIR = os.path.join("..", "Data", "single-levels", "ncfiles")
MONTH_DIRS = [
    "June-2024",
    "July-2024",
    "August-2024",
    "September-2024",
]

TRACKING_SEARCH_RADIUS_KM = 50    # For hourly tracking
LINKING_SEARCH_RADIUS_KM = 100    # For multi-hour gaps
# for wind creterion : https://mausam.imd.gov.in/imd_latest/monsoonreport2024.pdf page: 284
WIND_SPEED_LOWER_BOUND_MS = 5        # == 17knots  # 8     # m/s
WIND_SPEED_UPPER_BOUND_MS = 19       # == 27knots  # 16    # m/s
MSL_LOWER_BOUND_PA = 100000  # 99800                       # Pa
PRECIPITATION_THRESHOLD_MM = 0.3                           # mm/hr
VORTICITY_THRESHOLD = 5e-6           # > 0                 # /s

MAX_SPEED_MS = 15
MAX_GAP_HOURS = 12              # Bridge small gaps In Detection

STITCH_DISTANCE_KM = 300   # 400 # For gap bridging
MERGE_DISTANCE_KM = 350    # 100 # For simultaneous systems, distance to consider two detections as same MD

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
radii = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]
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
    "avg_ishf",    # surface sensible heat flux
    'total_hf'     # Total surface heat flux
]

# Use clipped domain instead of dataset bounds (10N 70E - 30N 95E) in plot_single_track func
clipped_min_lat = 10.0  # N
clipped_max_lat = 30.0  # N
clipped_min_lon = 70.0  # E
clipped_max_lon = 95.0  # E

# desired_tracks = [1, 3, 4, 5]

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
            boundary_pressure = np.nanmean([
                subset.isel(latitude=0).values.mean(),      # North edge
                subset.isel(latitude=-1).values.mean(),     # South edge
                subset.isel(longitude=0).values.mean(),     # West edge
                subset.isel(longitude=-1).values.mean(),    # East edge
            ])
            # if boundary_pressure > central_pressure + 50:
            if boundary_pressure > central_pressure + 20:
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
    # Get grid dimensions
    n_lat = len(ds.latitude)
    n_lon = len(ds.longitude)

    # Get u, v at the point
    u = float(ds['u10'].isel(latitude=lat_idx, longitude=lon_idx).values)
    v = float(ds['v10'].isel(latitude=lat_idx, longitude=lon_idx).values)

    # Get grid spacing
    lat = float(ds.latitude.values[lat_idx])
    dlat = np.abs(ds.latitude.values[1] - ds.latitude.values[0])
    dlon = np.abs(ds.longitude.values[1] - ds.longitude.values[0])

    # Convert to meters
    dy = EARTH_R * np.deg2rad(dlat)
    dx = EARTH_R * np.cos(np.deg2rad(lat)) * np.deg2rad(dlon)

    # # Get neighboring points
    # if (lat_idx == 0 or lat_idx == len(ds.latitude) - 1 or 
    #     lon_idx == 0 or lon_idx == len(ds.longitude) - 1):
    #     return np.nan
    # # Get u and v at neighboring points
    # u_north = float(ds['u10'].isel(latitude=lat_idx+1, longitude=lon_idx).values)
    # u_south = float(ds['u10'].isel(latitude=lat_idx-1, longitude=lon_idx).values)
    # v_east = float(ds['v10'].isel(latitude=lat_idx, longitude=lon_idx+1).values)
    # v_west = float(ds['v10'].isel(latitude=lat_idx, longitude=lon_idx-1).values)
    # # Calculate gradients
    # du_dy = (u_north - u_south) / (2 * dy)
    # dv_dx = (v_east - v_west) / (2 * dx)

    # Get neighboring points with boundary checks
    # For latitude (north-south)
    if lat_idx == 0:  # Southern boundary
        u_north = float(ds['u10'].isel(latitude=lat_idx+1, longitude=lon_idx).values)
        u_south = float(ds['u10'].isel(latitude=lat_idx, longitude=lon_idx).values)  # Use forward difference
        du_dy = (u_north - u_south) / dy
    elif lat_idx == n_lat - 1:  # Northern boundary
        u_north = float(ds['u10'].isel(latitude=lat_idx, longitude=lon_idx).values)
        u_south = float(ds['u10'].isel(latitude=lat_idx-1, longitude=lon_idx).values)
        du_dy = (u_north - u_south) / dy
    else:  # Interior points
        u_north = float(ds['u10'].isel(latitude=lat_idx+1, longitude=lon_idx).values)
        u_south = float(ds['u10'].isel(latitude=lat_idx-1, longitude=lon_idx).values)
        du_dy = (u_north - u_south) / (2 * dy)
    # For longitude (east-west)
    if lon_idx == 0:  # Western boundary
        v_east = float(ds['v10'].isel(latitude=lat_idx, longitude=lon_idx+1).values)
        v_west = float(ds['v10'].isel(latitude=lat_idx, longitude=lon_idx).values)
        dv_dx = (v_east - v_west) / dx
    elif lon_idx == n_lon - 1:  # Eastern boundary
        v_east = float(ds['v10'].isel(latitude=lat_idx, longitude=lon_idx).values)
        v_west = float(ds['v10'].isel(latitude=lat_idx, longitude=lon_idx-1).values)
        dv_dx = (v_east - v_west) / dx
    else:  # Interior points
        v_east = float(ds['v10'].isel(latitude=lat_idx, longitude=lon_idx+1).values)
        v_west = float(ds['v10'].isel(latitude=lat_idx, longitude=lon_idx-1).values)
        dv_dx = (v_east - v_west) / (2 * dx)

    # Relative vorticity (ζ = ∂v/∂x - ∂u/∂y)
    vorticity = dv_dx - du_dy
    return vorticity

def check_vorticity_criteria(data_slice, lat, lon):
    try:
        # Find the time step index
        if 'time' in data_slice.coords:
            time = data_slice['time'].values
        else:
            # If no time coordinate, we can't compute vorticity
            return False, np.nan

        # Get u, v at center point
        u_center = float(data_slice['u10'].sel(
            latitude=lat, longitude=lon, method='nearest'
        ).values)
        v_center = float(data_slice['v10'].sel(
            latitude=lat, longitude=lon, method='nearest'
        ).values)

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

        # Get neighboring u, v values
        u_north = float(data_slice['u10'].isel(latitude=lat_idx+1, longitude=lon_idx).values)
        u_south = float(data_slice['u10'].isel(latitude=lat_idx-1, longitude=lon_idx).values)
        v_east = float(data_slice['v10'].isel(latitude=lat_idx, longitude=lon_idx+1).values)
        v_west = float(data_slice['v10'].isel(latitude=lat_idx, longitude=lon_idx-1).values)

        # Compute finite differences
        du_dy = (u_north - u_south) / (2 * EARTH_R * dlat_rad)  # m/s per meter
        dv_dx = (v_east - v_west) / (2 * EARTH_R * dlon_rad_adjusted)  # m/s per meter

        # Compute relative vorticity (dv/dx - du/dy)
        vorticity = dv_dx - du_dy  # s^-1
        # Check if vorticity is positive (cyclonic)
        return vorticity > VORTICITY_THRESHOLD, vorticity

    except Exception as e:
        std_print('ERROR', 'md', 'Computing vorticity:', e)
        return False, np.nan

def check_moisture_criteria(data_slice, lat, lon):
    lat_min, lat_max = lat - BUFFER, lat + BUFFER
    lon_min, lon_max = lon - BUFFER, lon + BUFFER

    try:
        # Check total precipitation (tp m/hr)
        if "tp" in data_slice.data_vars:
            tp_data = data_slice["tp"]
            tp_subset = tp_data.sel(
                latitude=slice(lat_max, lat_min),
                longitude=slice(lon_min, lon_max)
            )
            max_precipitation = tp_subset.max().item()
            precipitation_check = (max_precipitation * 1000) >= PRECIPITATION_THRESHOLD_MM # m/hr -> mm/hr
            return precipitation_check
        else:
            std_print('WARN', 'md', f'Missing `tp`. Skipping check')
            return False
    except KeyError as e:
        std_print('ERROR', 'md', f'Missing variable for moisture check:', e, 'Skipping this check.')
        return False

def compute_surface_heat_fluxes(ds, lat, lon, buffer_km=100):
    lat_buffer, lon_buffer = calculate_buffer_deg(lat, buffer_km)
    try:
        slhtf = ds['avg_slhtf'].sel(
            latitude=slice(lat + lat_buffer, lat - lat_buffer),
            longitude=slice(lon - lon_buffer, lon + lon_buffer)
        ).mean().values
        ishf = ds['avg_ishf'].sel(
            latitude=slice(lat + lat_buffer, lat - lat_buffer),
            longitude=slice(lon - lon_buffer, lon + lon_buffer)
        ).mean().values
        total_hf = slhtf + ishf if not (np.isnan(slhtf) or np.isnan(ishf)) else np.nan
        return {
            'latent_flux': slhtf,
            'sensible_flux': ishf,
            'total_flux': total_hf
        }
    except:
        return {'latent_flux': np.nan, 'sensible_flux': np.nan, 'total_flux': np.nan}

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
    time_coord = "time" if "time" in ds.coords else "valid_time"
    times = pd.to_datetime(ds[time_coord].values)
    std_print(None, None, 'Start', times[0])
    std_print(None, None, 'End', times[-1])
    std_print(None, None, 'Total duration (hours)', f'{(times[-1] - times[0]).total_seconds() / 3600:.1f}')
    std_print(None, None, 'Total points', len(times))
    std_print(None, None, 'Total points', len(times))

    # Calculate time differences
    time_diffs = np.diff(times).astype('timedelta64[m]').astype(float) / 60  # Convert to hours

    std_print('INFO', 'md', 'Time step statistics')
    std_print(None, None, f'Mean interval   {np.mean(time_diffs):.2f} hours')
    std_print(None, None, f'Median interval {np.median(time_diffs):.2f} hours')
    std_print(None, None, f'Std Dev         {np.std(time_diffs):.2f} hours')
    std_print(None, None, f'Min interval    {np.min(time_diffs):.2f} hours')
    std_print(None, None, f'Max interval    {np.max(time_diffs):.2f} hours')
    # Identify anomalies
    mode_diff = np.argmax(np.bincount(time_diffs.astype(int)))
    std_print(None, None ,'Most common interval (hours)', mode_diff)

    # Find irregular gaps
    unique_diffs, counts = np.unique(time_diffs.astype(int), return_counts=True)
    std_print('INFO', 'md', 'Time step distribution')
    for diff, count in sorted(zip(unique_diffs, counts), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / len(time_diffs)
        std_print(None, None, 'Hour intervals', f'{diff:2d} for {count:4d} times ({percentage:5.1f}%)')

    # Detect month boundaries
    std_print('INFO', 'md', 'Month boundary analysis')
    month_changes = []
    for i in range(1, len(times)):
        if times[i].month != times[i-1].month:
            month_changes.append((i, times[i-1], times[i], time_diffs[i-1]))
    if month_changes:
        std_print('INFO', 'md', f'Found {len(month_changes)} month transitions:')
        for idx, prev_time, curr_time, gap in month_changes:
            std_print(None, None, f"Index {idx}: {prev_time.strftime('%Y-%m-%d %H:%M')} -> {curr_time.strftime('%Y-%m-%d %H:%M')}  (gap: {gap:.1f} hours)")
    else:
        std_print('INFO', 'md','No month transitions detected (single continuous month)')

    # Detect duplicates
    std_print('INFO', 'md', 'Duplicate/overlap detection')
    unique_times, indices = np.unique(times, return_index=True)
    n_duplicates = len(times) - len(unique_times)
    if n_duplicates > 0:
        std_print(None, None, 'Found duplicate timestamps', {n_duplicates})
        duplicate_mask = np.zeros(len(times), dtype=bool)
        duplicate_mask[indices] = True
        duplicate_indices = np.where(~duplicate_mask)[0]
        for idx in duplicate_indices[:5]:  # Show first 5
            std_print(None, None, f'Duplicate at index {idx} :', times[idx])
    else:
        std_print('INFO', 'md', 'No duplicates timestamps')

    # Detect negative gaps (time goes backwards)
    std_print('INFO', 'md', 'Time continuity check')
    negative_gaps = np.where(time_diffs < 0)[0]
    if len(negative_gaps) > 0:
        std_print(None, None, 'Found Time reversals', len(negative_gaps), 'backward jumps')
        for idx in negative_gaps[:5]:
            std_print(None, None, f'Jump backward at index {idx}', f'{times[idx]} -> {times[idx+1]} (difference: {time_diffs[idx]:.1f} hours)')
    else:
        std_print('INFO', 'md', 'Time is monotonically increasing')

    # Detect zero-gaps (identical consecutive times)
    zero_gaps = np.where(time_diffs == 0)[0]
    if len(zero_gaps) > 0:
        std_print('INFO', 'md', 'Found identical consecutive timestamps', len(zero_gaps))
    if n_duplicates > 0:
        std_print('INFO', 'md', 'Duplicates present', n_duplicates, 'repeated timestamps')
    if len(negative_gaps) > 0:
        std_print('INFO', 'md', 'Time reversals', len(negative_gaps), 'backward jumps')
    if len(month_changes) > 0 and len(unique_diffs) > 1:
        std_print('INFO', 'md', 'Files likely concatenated', 'Different resolutions at month boundaries')

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

# def fix_temporal_resolution(ds, target_resolution_hours=1, method='interpolate'):
def fix_temporal_resolution(ds, target_resolution_hours=1):
    time_coord = "time" if "time" in ds.coords else "valid_time"
    times = pd.to_datetime(ds[time_coord].values)

    std_print('INFO', 'md', 'Dataset temporal resolution', len(times), 'time points')
    # Create regular time grid
    time_start = times[0]
    time_end = times[-1]
    new_times = pd.date_range(
        start=time_start,
        end=time_end,
        freq=f'{target_resolution_hours}h'
    )
    # Reindex dataset to regular grid with interpolation
    ####
    for var in ['u10', 'v10']:
        if var in ds.data_vars:
            ds[var] = ds[var].interp(
                {time_coord: new_times}, 
                method='nearest'
            )
    for var in ['tp', 'swvl1', 'msl', 't2m']:
        if var in ds.data_vars:
            ds[var] = ds[var].interp(
                {time_coord: new_times}, 
                method='linear',
                kwargs={'fill_value': np.nan}
            )
    ####
    ds_fixed = ds.reindex({time_coord: new_times}, method='nearest')
    std_print('INFO', 'md', 'Resampled time points (hours):',  f'{len(new_times)}/{target_resolution_hours}')
    # original_bounds = (ds[time_coord].min(), ds[time_coord].max())
    # out_of_bounds = (new_times < original_bounds[0]) | (new_times > original_bounds[1])
    # print(f" >  [INFO md   ] Interpolation: {out_of_bounds.sum()} points outside original time bounds")
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
    std_print('INFO', 'md', 'Temporal validation')
    std_print('Expected :', f'{expected_hours}±{tolerance*100:.0f}% hours')
    std_print('Compliance :', f'{n_valid}/{n_total} intervals ({validity_pct:.1f}%)')
    std_print('Valid :', f"{'YES' if is_valid else 'NO'}")
    if not is_valid and len(report['violations']) > 0:
        std_print(None, None, 'Violations at indices :', f"{report['violations'][:10]}")
    return is_valid, report

def resample_dataset_to_resolution(ds, target_hours=1):
    time_coord = "time" if "time" in ds.coords else "valid_time"
    times = pd.to_datetime(ds[time_coord].values)
    if target_hours == 1:
        std_print('INFO', 'md', 'Dataset at 1-hourly resolution :', f'{len(times)} steps')
        return ds
    new_times = pd.date_range(start=times[0], end=times[-1], freq=f'{target_hours}h')
    original_count = len(times)
    new_count = len(new_times)
    std_print('INFO', 'md', 'Temporal resampling')
    std_print(None, None, 'Input :', f'{original_count} time steps (hourly)')
    std_print(None, None, 'Output :', f'{new_count} time steps ({target_hours}-hourly)')
    std_print(None, None, 'Reduction :', f'{100*(1-new_count/original_count):.1f}%')

    # Interpolate all variables to new time grid
    ds_resampled = ds.interp(
        {time_coord: new_times},
        method='linear',
        # kwargs={"fill_value": "extrapolate"}
        kwargs={"fill_value": np.nan}
    )
    return ds_resampled

def ensure_flux_results_complete(all_flux_results, all_classifications):
    """Ensure all_flux_results has required fields from classifications"""
    if not all_flux_results or not all_classifications:
        return all_flux_results
    # Create lookup dict from classifications
    class_lookup = {c['track_id']: c for c in all_classifications}
    # Augment flux results with classification data
    for flux_res in all_flux_results:
        track_id = flux_res.get('track_id')
        if track_id and track_id in class_lookup:
            classification = class_lookup[track_id]
            # Only add if not already present
            if 'is_long_lived' not in flux_res:
                flux_res['is_long_lived'] = classification.get('is_long_lived', False)
            if 'duration_hours' not in flux_res:
                flux_res['duration_hours'] = classification.get('duration_hours', np.nan)
            if 'category' not in flux_res:
                flux_res['category'] = classification.get('category', 'Unknown')
    return all_flux_results

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

def find_low_pressure_center(data_slice, time_step, prev_center, search_radius_km = TRACKING_SEARCH_RADIUS_KM, smoothing_radius_km = 100) :
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
                if distance > search_radius_km:
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
            if score >= 3:  # Accept if at least no. of criteria pass
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
        std_print('ERROR', 'md', f'Finding low pressure center at {time_step} :', e)
        return None

def track_monsoon_depressions(detections, max_speed_ms=15, search_radius_km = TRACKING_SEARCH_RADIUS_KM, max_time_gap_hours = MAX_GAP_HOURS):
    if detections.empty:
        return []
    detections = detections.sort_values("time").reset_index(drop=True)
    detections['time'] = pd.to_datetime(detections['time'])
    detections['hour'] = detections['time'].dt.floor('h')

    # print(f" >  -- Tracking DEBUG --")
    # print(f" >  [DEBUG   ] Total detections:                            {len(detections)}")
    # print(f" >  [DEBUG   ] Time range:                                  {detections['time'].min()} to {detections['time'].max()}")
    # # Group by time
    time_groups = detections.groupby('hour')
    time_steps = sorted(time_groups.groups.keys())
    # print(f" >  [DEBUG   ] Hours with detections:                        {len(time_steps)}")

    if len(time_steps) > 1:
        time_diffs = []
        for i in range(1, len(time_steps)):
            diff = (time_steps[i] - time_steps[i-1]).total_seconds() / 3600
            time_diffs.append(diff)
        std_print('DEBUG', 'md', 'Time gaps between detections :')
        std_print(None, None, 'Mean (hours)', f'{np.mean(time_diffs):.1f}')
        std_print(None, None, 'Min/Max (hours)', f'{np.min(time_diffs):.1f}/{np.max(time_diffs):.1f}')
        std_print(None, None, 'Std Dev (hours)', f'{np.std(time_diffs):.1f} hours')
        # Show distribution of gaps
        unique_gaps, gap_counts = np.unique(np.round(time_diffs), return_counts=True)
        for gap, count in sorted(zip(unique_gaps, gap_counts)):
            std_print(None, None, 'Hour gaps of length', f'{int(gap)} occur {count} times')

    tracks = []
    if len(time_steps) < 2:
        std_print('WARN', 'Not enough time steps for tracking')
        return tracks

    active_tracks = {}
    first_detections = time_groups.get_group(time_steps[0])
    for i, (_, row) in enumerate(first_detections.iterrows()):
        active_tracks[i] = [row]
    std_print('INFO', 'md', 'Initial active tracks ', f'{len(active_tracks)}')
    for t_idx in range(1, len(time_steps)):
        current_hour = time_steps[t_idx]
        current_detections = time_groups.get_group(current_hour)
        prev_hour = time_steps[t_idx-1]
        time_diff_hours = (current_hour - prev_hour).total_seconds() / 3600

        # Only proceed if time gap is reasonable
        if time_diff_hours > max_time_gap_hours:
            std_print(None, None, 'Large time gap ', f'{time_diff_hours:.1f} hours from {prev_hour} to {current_hour}')
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
                if cost < np.inf:
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
    std_print('INFO', 'md', 'Tracks found :', f'{len(tracks)}')
    return tracks

def stitch_sequential_tracks(tracks, max_time_gap_hours=6, dist_km=400):
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
                if 0 < hour_gap <= max_time_gap_hours:
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
        base['time'] = pd.to_datetime(base['time']).dt.tz_localize(None)  # Ensure UTC naive
        # other['time'] = pd.to_datetime(other['time']).dt.tz_localize(None)
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
                            d = haversine(p1['longitude'], p1['latitude'], p2['longitude'], p2['latitude'])
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

def smooth_track_coordinates(track_df, window_size = 5, iterations = 2):
    if len(track_df) < window_size:
        std_print('ERROR', 'md', 'Track have less points than', window_size)
        return track_df

    track_df = track_df.copy()
    # Check if required columns exist
    if 'latitude' not in track_df.columns or 'longitude' not in track_df.columns:
        std_print('ERROR', 'md', "Track DataFrame is missing 'latitude' and/or 'longitude' columns")
        return track_df

    for _ in range(iterations):
        # Moving average
        lat_smooth = track_df['latitude'].rolling(window=window_size, center=True, min_periods=1).mean()
        lon_smooth = track_df['longitude'].rolling(window=window_size, center=True, min_periods=1).mean()
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

    lat1, lon1 = np.radians(start_lat), np.radians(start_lon)
    lat2, lon2 = np.radians(end_lat), np.radians(end_lon)

    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360  # Normalize to 0-360

    return bearing

def filter_valid_tracks(tracks, min_duration=MIN_TRACK_DURATION_HOURS, min_distance=MIN_NET_DISPLACEMENT_KM):
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

        std_print('INFO', 'md', 'Temporal check')
        std_print(None, None, 'Dataset time steps :', len(times))
        std_print(None, None, 'Mean interval (hours)', f'{np.mean(time_diffs):.2f}')
        std_print(None, None, 'Min interval  (hours)', f'{np.min(time_diffs):.2f}')
        std_print(None, None, 'Max interval  (hours)', f'{np.max(time_diffs):.2f}')
        # Verify dataset has regular spacing
        if np.std(time_diffs) > 0.1:
            print(f" >  [WARN ] Irregular intervals detected")
            ds = fix_temporal_resolution(ds, target_resolution_hours=1)
            times = pd.to_datetime(ds[time_coord].values)
            time_diffs = np.diff(times).astype('timedelta64[s]').astype(float) / 3600
            std_print('INFO', 'md', 'After fix Mean interval (hours)', f'{np.mean(time_diffs):.2f}')

        required_vars = ["msl", "u10", "v10", "tp"]
        if not all(var in ds.data_vars for var in required_vars):
            std_print('ERROR', 'md', 'Missing required variables', required_vars)
            return None

        times = ds[time_coord].values
        std_print('INFO', 'md', 'Time range :', f'{times[0]} to {times[-1]}')
        std_print('INFO', 'md', 'Number of time steps :', len(times))

        md_detections = []
        prev_center = None
        detection_count = 0
        std_print('INFO', 'md', 'Detection progress')
        total_steps = len(times)
        # spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner = ["◜", "◝", "◞", "◟"]
        dots_sequence = ["   ", ".  ", ".. ", "..."]
        spinner_index = 0
        for i, time_step in enumerate(times):
            if i % 5 == 0:
                # current_spinner = spinner[spinner_index]
                # spinner_index = (spinner_index + 1) % len(spinner)
                # sys.stdout.write(f"\r{current_spinner} Processing time step {i+1}/{total_steps}")
                current_spinner = spinner[spinner_index % len(spinner)]
                current_dots = dots_sequence[spinner_index % len(dots_sequence)]
                sys.stdout.write(f"\r{current_spinner} Processing time step {i+1}/{total_steps}{current_dots}")
                sys.stdout.flush()
                time.sleep(0.1)
                spinner_index += 1

            data_slice = ds.sel(**{time_coord: time_step})
            # Find center
            center = find_low_pressure_center(
                data_slice,
                time_step,
                prev_center,
                search_radius_km = TRACKING_SEARCH_RADIUS_KM,
                smoothing_radius_km = 100
            )
            if center is not None:
                lat, lon = center
                data_slice = ds.sel(time=time_step)
                msl_value = float(data_slice['msl'].sel(
                    latitude=lat,
                    longitude=lon,
                    method='nearest'
                ).values)
                md_detections.append({
                    "time": time_step,
                    "latitude": lat,
                    "longitude": lon,
                    "msl": msl_value,
                })
                prev_center = (lat, lon)  # Update for next time step
                detection_count += 1
            else:
                prev_center = None  # Reset if no detection

        std_print('\rINFO', 'md', 'Total detections found :', f'{detection_count} / {total_steps} time steps')
        std_print(None, None, 'Detection rate', f'{100*detection_count/total_steps:.1f}%')
        if not md_detections:
            std_print('INFO', 'md', 'No MD centers detected')
            return None

        detections_df = pd.DataFrame(md_detections)
        detections_df["month"] = pd.to_datetime(detections_df["time"]).dt.strftime("%B")
        # Verify detection times are on the hour
        detection_hours = [int(h) for h in detections_df['time'].dt.hour.unique()]
        std_print(None, None, 'Detections occur at hours', sorted(detection_hours))
        # Initial Tracking
        initial_tracks = track_monsoon_depressions(
            detections_df,
            max_speed_ms = MAX_SPEED_MS,
            search_radius_km = LINKING_SEARCH_RADIUS_KM,  # larger for better linking
            max_time_gap_hours = MAX_GAP_HOURS
        )
        std_print('INFO', 'md', 'Initial raw tracks found :', len(initial_tracks))

        # Filter out very short tracks immediately (<5 points)
        initial_tracks = [t for t in initial_tracks if len(t) >= 5]
        std_print('INFO', 'md', 'After removing very short tracks :', len(initial_tracks))

        # Stitch sequential fragments (gaps)
        stitched_tracks = stitch_sequential_tracks(
            initial_tracks,
            max_time_gap_hours = MAX_GAP_HOURS,
            dist_km = STITCH_DISTANCE_KM
        )
        std_print('INFO', 'md', 'Tracks after stitching gaps :', len(stitched_tracks))

        # Consolidate simultaneous/overlapping tracks (Spatial Merge)
        consolidated_tracks = consolidate_simultaneous_tracks(
            stitched_tracks,
            dist_km = MERGE_DISTANCE_KM
        )
        std_print('INFO', 'md', 'Tracks after consolidating centers :', len(consolidated_tracks))

        # # Filter by environment
        # ocean_tracks = filter_tracks_by_environment(
        #     consolidated_tracks,
        #     require_ocean_start=True
        # )
        # print(f" >  [INFO md   ] Tracks over ocean                 :          {len(ocean_tracks)}")
        # Quality Filter tracks (Noise Removal)
        filtered_valid_tracks = filter_valid_tracks(
            consolidated_tracks,
            # ocean_tracks,
            min_duration=MIN_TRACK_DURATION_HOURS,
            min_distance=MIN_NET_DISPLACEMENT_KM
        )
        std_print('INFO', 'md', 'Valid tracks after QC :', len(filtered_valid_tracks))
        if not filtered_valid_tracks:
            std_print(None, None, 'No tracks survived filtering criteria')
            return pd.DataFrame()  # Return empty dataframe

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
        # Restore month column if lost
        if 'month' not in track_df_with_ids.columns:
            track_df_with_ids['month'] = pd.to_datetime(
                track_df_with_ids['time']
            ).dt.strftime("%B")
        std_print('FINAL', 'md', 'Tracks ready for analysis :', len(smoothed_tracks))
        return track_df_with_ids
    except Exception as e:
        std_print('ERROR', 'md', 'process_files', e)
        traceback.print_exc()
        return None
    finally:
        if 'ds' in locals():
            ds.close()


def plot_tracks(track_df, output_dir="output"):
    fig = plt.figure(figsize=(14, 9))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # type: ignore[assignment]
    # ax.set_extent([70, 100, 5, 30], crs=ccrs.PlateCarree())
    ax.set_extent([clipped_min_lon, clipped_max_lon, clipped_min_lat, clipped_max_lat], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, linewidth=0.8, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle='--', alpha=0.7)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color="gray", alpha=0.6, linestyle="--")
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
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor='none', linewidth=0.3),
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
            bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor='none', linewidth=0.3),
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
    print("\n------------------------------------ MD track plots ------------------------------------")
    plt.savefig(output_filename)
    print(f" >> [SAVED   ] Combined track plot               :          {output_filename}")
    # plt.show()
    plt.close()

def plot_track_full_map(track_df_single, track_id, full_extent, output_dir):
    fig = plt.figure(figsize=(12, 8))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # type: ignore[assignment]

    ax.set_extent([clipped_min_lon, clipped_max_lon, clipped_min_lat, clipped_max_lat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle='--')
    ax.add_feature(cfeature.RIVERS)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

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
    ax.annotate(
        start_label,
        xy=(start["longitude"], start["latitude"]),
        xytext=(5,5),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7),
        transform=ccrs.PlateCarree(),
        zorder=6
    )
    ax.annotate(
        end_label,
        xy=(end["longitude"], end["latitude"]),
        xytext=(5,-10),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7),
        transform=ccrs.PlateCarree(),
        zorder=6
    )
    ax.set_title(f"Track {track_id}")
    outpath = os.path.join(output_dir, f"track_{int(track_id)}_full_map.png")
    fig.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f" >> [SAVED   ] Full-map for track {track_id}              :          {outpath}")

def plot_single_track(track_df, track_id, output_dir):
    track_data = track_df.sort_values("time")
    time_coord = 'time' if 'time' in track_df.columns else 'valid_time'
    if time_coord not in track_df.columns:
        time_coord = [col for col in track_df.columns if 'time' in col.lower()][0]

    start_time_str = pd.to_datetime(track_df[time_coord].iloc[0]).strftime('%d/%m/%Y %H:%M')
    end_time_str = pd.to_datetime(track_df[time_coord].iloc[-1]).strftime('%d/%m/%Y %H:%M')

    fig = plt.figure(figsize=(13, 10))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # type: ignore[assignment]

    # FIX: for track which goes till around 70 (track 3) minux 4 lat will show no results there
    # So check if out of bound of dataset too
    # min_lon, max_lon = track_df["longitude"].min() - 4, track_df["longitude"].max() + 4
    # min_lat, max_lat = track_df["latitude"].min() - 4, track_df["latitude"].max() + 4

    # Calculate extent based on track with padding, but clamp to clipped domain
    min_lon = max(track_df["longitude"].min() - 4, clipped_min_lon)
    max_lon = min(track_df["longitude"].max() + 4, clipped_max_lon)
    min_lat = max(track_df["latitude"].min() - 4, clipped_min_lat)
    max_lat = min(track_df["latitude"].max() + 4, clipped_max_lat)

    month = track_data["month"].iloc[0]
    color = month_colors.get(month, "black")

    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=1.0)
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, linewidth=0.6, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.4, linestyle='--', alpha=0.6)
    ax.add_feature(cfeature.RIVERS, linewidth=0.4, alpha=0.5)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Helper: common bbox style used for start, end and landfall annotattion
    common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)
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
            bbox=common_bbox,
            transform=ccrs.PlateCarree(),
            zorder=3,
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
            bbox=common_bbox,
            transform=ccrs.PlateCarree(),
            zorder=3,
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
        marker='o',
        color='green',
        markersize=4,
        transform=ccrs.PlateCarree(),
        label='Start',
        zorder=4
    )
    # End point (red X)
    ax.plot(
        track_df["longitude"].iloc[-1],
        track_df["latitude"].iloc[-1],
        marker='x',
        color='red',
        markersize=4,
        markeredgewidth=2,
        transform=ccrs.PlateCarree(),
        label='End',
        zorder=4
    )
    # Annotate Landfall (if applicable) with date/time label
    if 'is_land' in track_df.columns:
        land_status = track_df['is_land'].values
        transitions = np.diff(land_status.astype(int))
        ocean_to_land = np.where(transitions == 1)[0]
        if len(ocean_to_land) > 0:
            landfall_idx = ocean_to_land[0] + 1
            if landfall_idx < len(track_df):
                landfall_point = track_df.iloc[landfall_idx]
                lf_lon = landfall_point["longitude"]
                lf_lat = landfall_point["latitude"]
                lf_time_str = pd.to_datetime(landfall_point["time"]).strftime("%d/%m %H:%M")
                ax.plot(
                    lf_lon, lf_lat,
                    marker='s',
                    color='yellow',
                    markersize=4,
                    transform=ccrs.PlateCarree(),
                    zorder=6
                )
                ax.text(
                    lf_lon + 0.2, # right to point
                    lf_lat + 0.2, # up to point
                    f"Landfall\n{lf_time_str}",
                    transform=ccrs.PlateCarree(),
                    fontsize=8,
                    fontweight='normal',
                    color='black',
                    ha='left',
                    va='bottom',
                    bbox=common_bbox,
                    zorder=7
                )
    ax.set_title(f"Monsoon Depression Track {track_id}\n{start_time_str} to {end_time_str}", fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95, edgecolor='black', fancybox=True)
    plt.tight_layout()

    track_id_str = str(track_id).replace('/', '_').replace(' ', '_')
    file_name = f"MD-track-{track_id_str}-detailed.png"
    fullpath = os.path.join(output_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(fullpath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f" >> [SAVED   ] Detailed track {track_id_str} plot             :          {fullpath}")

def export_track_coordinates(track_df, output_dir):
    if track_df.empty:
        std_print('ERROR', 'md', 'No track coordinates to export', e)
        return

    export_df = track_df[['time', 'latitude', 'longitude', 'track_id']].copy()
    export_df['time'] = pd.to_datetime(export_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    output_filename = os.path.join(
        output_dir, f"MD-coordinates-ALL.csv"
    )
    export_df.to_csv(output_filename, index=False)
    print(f" >> [INFO md   ] No. of Track Coordinates to export:         {len(track_df)}")
    print(f" >> [EXPORTED] Track Coordinates detections CSV  :         {output_filename}")

def export_single_track_coordinates(track_df, track_id, output_dir):
    export_df = track_df[['time', 'latitude', 'longitude', 'track_id']].copy()
    export_df['time'] = pd.to_datetime(export_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    track_id_str = str(track_id).replace('/', '_').replace(' ', '_')
    file_name = f"MD-track-{track_id_str}-coordinates.csv"
    export_df.to_csv(os.path.join(output_dir, file_name), index=False)
    print(f" >> [EXPORTED] Track {track_id} CSV                       :          {file_name}")

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
        # FIX: Issue: .compute() loads entire dataset into memory, Could cause OOM for very large datasets
        ds = xr.open_mfdataset(
            file_list,
            combine="by_coords",
            data_vars='all',
            coords='all',  # type: ignore[arg-type]
            compat='override',
            join='outer'
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
        if 'valid_time' in ds.coords and 'time' not in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        ds = standardize_time_coord(ds)
        ds = ds.sortby('time')
        # ds = ds.sel(
        #     latitude=slice(clipped_max_lat, clipped_min_lat),
        #     longitude=slice(clipped_min_lon, clipped_max_lon)
        # )
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
    print(f" >  -- Detection Quality Diagnostics --")
    # 1. Check for large gaps in time
    times = pd.to_datetime(track_df['time'])
    time_diffs = times.diff().dt.total_seconds().dropna() / 3600

    large_gaps = time_diffs[time_diffs > 6]
    if len(large_gaps) > 0:
        print(f" >  [INFO md   ] Tracks with time gaps >6 hours    :          {len(large_gaps)} ")
        print(f" >  [INFO md   ] Max gap                           :           {time_diffs.max():.1f} hours")
    else:
        print(f" >  [INFO md   ] No large time gaps detected")

    # 2. Check for sudden position jumps
    if 'distance_km' in track_df.columns:
        large_jumps = track_df[track_df['distance_km'] > 150]
        if len(large_jumps) > 0:
            print(f" >  [INFO md   ] Tracks position jumps (>150 km)   :          {len(large_jumps)}")
            print(f" >  [INFO md   ] Max jump                          :          {track_df['distance_km'].max():.1f} km")
        else:
            print(f" >  [INFO md   ] No suspicious position jumps")

    # 3. Check speed consistency
    if 'speed_kmh' in track_df.columns:
        speeds = track_df['speed_kmh'].dropna()
        if len(speeds) > 0:
            unrealistic = speeds[(speeds < 3) | (speeds > 30)]
            if len(unrealistic) > 0:
                print(f" >  [INFO md   ] Tracks with unrealistic speeds    :          {len(unrealistic)}")
                print(f" >  [INFO md   ] Range                             :          {speeds.min():.1f} - {speeds.max():.1f} km/h")
            else:
                print(f" >  [INFO md   ] All speeds within reasonable range")

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
            y = (np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon))

            bearing = np.degrees(np.arctan2(x, y))
            bearing = (bearing + 360) % 360
            bearings.append(bearing)
        bearings = np.array(bearings)

        # MDs typically move WNW (240-330 degrees)
        wnd_bearings = bearings[
            (bearings >= 240) & (bearings <= 360) |
            (bearings >= 0) & (bearings <= 30)
        ]
        if len(wnd_bearings) / len(bearings) < 0.7:
            print(f" >  [INFO md   ] Movement in WNW (expected >70%)   :          {100*len(wnd_bearings)/len(bearings):.1f}%")
        else:
            print(f" >  [INFO md   ] Track direction consistent        :          {100*len(wnd_bearings)/len(bearings):.1f}% WNW")

def generate_summary_report(all_correlation_results, all_flux_results, all_landfall_results, all_classifications, all_sm_results, output_dir="output"):
    report_path = os.path.join(output_dir, "Analysis_summary_report.txt")
    with open(report_path, 'w') as f:
        f.write("Monsoon depression analysis - summary\n")
        # Track Summary
        f.write("1. Track statistics\n")
        if all_classifications:
            df_class = pd.DataFrame(all_classifications)
            short_mds = df_class[~df_class['is_long_lived']]
            long_mds = df_class[df_class['is_long_lived']]
            f.write(f"   Total MDs detected:                              {len(df_class)}\n")
            f.write(f"   Short-lived MDs (< 96h):                         {len(short_mds)} ({100*len(short_mds)/len(df_class):.1f}%)\n")
            f.write(f"   Long-lived MDs (≥ 96h):                          {len(long_mds)} ({100*len(long_mds)/len(df_class):.1f}%)\n")
            f.write(f"   Mean duration (all):                             {df_class['duration_hours'].mean():.1f} ± {df_class['duration_hours'].std():.1f} hours\n\n")
        # Precipitation-Soil Moisture Correlation
        f.write("2. Precipitation - soil moisture correlation\n")
        if all_correlation_results:
            df_corr = pd.DataFrame(all_correlation_results)
            f.write(f"   Tracks analyzed:                                 {len(df_corr)}\n")
            f.write(f"   Mean correlation (TP vs SM):                     {df_corr['correlation'].mean():.3f}\n")
            f.write(f"   Std Dev:                                         {df_corr['correlation'].std():.3f}\n")
            f.write(f"   Positive correlations:                           {(df_corr['correlation'] > 0).sum()}/{len(df_corr)}\n")
            f.write(f"   Significant (p < 0.05):                          {(df_corr['p_value'] < 0.05).sum()}/{len(df_corr)}\n\n")
        # Heat Flux Analysis
        f.write("3. Surface heat flux analysis\n")
        if all_flux_results:
            df_flux = pd.DataFrame(all_flux_results)
            f.write(f"   Tracks with flux data:                           {len(df_flux)}\n")
            f.write(f"   Mean total heat flux:                            {df_flux['mean_total_hf'].mean():.1f} ± {df_flux['mean_total_hf'].std():.1f} W/m²\n")
            f.write(f"   Mean latent heat flux:                           {df_flux['mean_latent_hf'].mean():.1f} W/m²\n")
            f.write(f"   Mean sensible heat flux:                         {df_flux['mean_sensible_hf'].mean():.1f} W/m²\n\n")
        # Landfall Flux Transitions
        f.write("4. Landfall flux transitions\n")
        if all_landfall_results:
            f.write(f"   Tracks with landfall data:                       {len(all_landfall_results)}\n")
            before_fluxes = [r['before']['total'] for r in all_landfall_results if r]
            after_fluxes = [r['after']['total'] for r in all_landfall_results if r]
            if before_fluxes and after_fluxes:
                f.write(f"   Mean before landfall:                            {np.nanmean(before_fluxes):.1f} W/m²\n")
                f.write(f"   Mean after landfall:                             {np.nanmean(after_fluxes):.1f} W/m²\n")
                f.write(f"   Mean change (ocean→land):                        {np.nanmean(after_fluxes) - np.nanmean(before_fluxes):.1f} W/m²\n\n")
        # Soil Moisture Summary
        f.write("5. Soil moisture analysis\n")
        if all_sm_results:
            df_sm = pd.DataFrame(all_sm_results)
            f.write(f"   Tracks analyzed:                                 {len(df_sm)}\n")
            f.write(f"   Mean soil moisture:                              {df_sm['mean_swvl1'].mean():.3f} m³/m³\n")
            f.write(f"   Mean precipitation:                              {df_sm['mean_tp'].mean():.2f} mm/hr\n\n")
    std_print("SAVED", 'md', f'Summary report : {report_path}')
    return report_path

if __name__ == "__main__":
    start_time = time.time()
    std_print('INFO', 'md', 'Applied Thresholds')
    std_print('INFO', 'md', f'Distance threshold: {TRACKING_SEARCH_RADIUS_KM} km')
    std_print('INFO', 'md', f'Wind Speed (10m): {WIND_SPEED_LOWER_BOUND_MS}-{WIND_SPEED_UPPER_BOUND_MS} m/s')
    std_print('INFO', 'md', f'MSLP threshold: < {MSL_LOWER_BOUND_PA} Pa ({MSL_LOWER_BOUND_PA/100:.0f} hPa)')
    std_print('INFO', 'md', f'Precipitation: > {PRECIPITATION_THRESHOLD_MM} mm/hr')
    std_print('INFO', 'md', f'Vorticity: > 0 /s')
    std_print('INFO', 'md', f'Min Track Duration: {MIN_TRACK_DURATION_HOURS} hours')
    std_print('INFO', 'md', f'Min Displacement: {MIN_NET_DISPLACEMENT_KM} km')

    print("\n--------------------------------- Combining monthly netCDF files ------------------------")
    print(" >  [INFO md   ] Combining all monthly files for track detection")
    start_processing_time = time.time()
    num_files = len(MONTH_DIRS) * 3
    result_container = {"ds": None, "success": False, "error": None}
    load_thread = threading.Thread(target=load_dataset_async, args=(MONTH_DIRS, DATA_DIR, result_container,))
    load_thread.daemon = True
    load_thread.start()

    spinner_index = 0
    # spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner = ["◜", "◝", "◞", "◟"]
    dots_sequence = ["   ", ".  ", ".. ", "..."]
    while load_thread.is_alive():
        current_spinner = spinner[spinner_index % len(spinner)]
        current_dots = dots_sequence[spinner_index % len(dots_sequence)]
        sys.stdout.write(f"\r{current_spinner} Loading ERA5 files from {len(MONTH_DIRS)} months{current_dots}")
        # sys.stdout.write(f"\r{spinner[spinner_index % len(spinner)]} Loading ERA5 files from {len(MONTH_DIRS)} months...")
        sys.stdout.flush()
        time.sleep(0.1)
        spinner_index += 1
    print()

    if result_container["success"]:
        ds = result_container["ds"]
        end_processing_time = time.time()
        std_print('INFO', 'md', 'Time for load and combine files', f'{end_processing_time - start_processing_time:.2f}', 'seconds')

        # Diagnose temporal issues
        diag_report = diagnose_temporal_issues(ds)
        print(" >  [APPLYING TEMPORAL FIX]")
        print("      Original dataset has variable time spacing")
        print("      Resampling to regular 1-hour grid...")

        ds = fix_temporal_resolution(ds, target_resolution_hours=1)
        is_valid, val_report = validate_temporal_consistency(ds, expected_hours=1, tolerance=0.05)
        if not is_valid:
            std_print('WARN', 'md', 'Some intervals still outside tolerance')
            std_print('WARN', 'md', 'This may affect tracking accuracy')
        else:
            std_print('INFO', 'md', 'Temporal resolution successfully regularized')
    else:
        print(f"\r >  [ERROR md  ] Loading files: {result_container['error']}")
        sys.exit(1)

    print("\n---------------------------------- Dataset Info ----------------------------------------")
    # Temporal info
    time_coord = "time" if "time" in ds.coords else "valid_time"
    years = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%Y").unique()
    months = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%b").unique()
    hours = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%H:%M").unique()
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

    print(f" >  Year                      :     {years}")
    print(f" >  Months                    :     {months}")
    print(f" >  Hours                     :     {hours}")
    print(f" >  Variables                 :     {variables}") #[var for var in ds.data_vars if var in ["msl", "u10", "v10", "tp", "t2m"]])
    print(f" >  Dataset bounds            :     lat [{ds_lat_min:.1f}, {ds_lat_max:.1f}] N, lon [{ds_lon_min:.1f}, {ds_lon_max:.1f}] E")
    print(f" >       using clipped bounds :     lat [{clipped_min_lat}, {clipped_max_lat}] N, lon [{clipped_min_lon}, {clipped_max_lon}] E")
    print(f" >  Data Time Resolution      :     {resolution_hours:.1f} hours ({time_diff})")

    print("\n------------------------------------ Processing ----------------------------------------")
    full_track_df = process_files(ds)
    if full_track_df is not None and not full_track_df.empty:
        if 'track_id' not in full_track_df.columns:
            print(" >  [ERROR md  ] 'track_id' column not found")
            sys.exit(1)
        # # For plotting and processing only `desired_tracks` selectively
        # if 'track_id' in full_track_df.columns:
        #     initial_track_count = len(full_track_df['track_id'].unique())
        #     full_track_df = full_track_df[full_track_df['track_id'].isin(desired_tracks)].copy()
        #     filtered_track_count = len(full_track_df['track_id'].unique())
        #     print(f"\n >   [INFO md   ] Filtered tracks for Azimuthal-Hovmöller analysis.")
        #     print(f" >   [INFO md   ] Retaining {filtered_track_count} tracks ({desired_tracks}) out of {initial_track_count} total.")
        # else:
        #     print(" >  [WARN ]: 'track_id' column not found. Skipping track filter.")
        plot_tracks(full_track_df, output_dir)

        # FIX: even after clipping it is using full extend (Detected)
        pad = 0.5
        try:
            full_extent = [ds_lon_min - pad, ds_lon_max + pad, ds_lat_min - pad, ds_lat_max + pad]
            print(f" [DEBUG   ] Full extent (Detected)                  :          {full_extent}")
        except Exception:
            # full_extent = [70, 100, 5, 30]
            full_extent = [70, 95, 10, 30]
            print(f" [DEBUG   ] Full extent (Mannual)                   :          {full_extent}")

        full_track_df['is_land'] = full_track_df.apply(lambda row: is_point_land(row['latitude'], row['longitude']), axis=1)
        unique_track_ids = sorted([int(tid) for tid in full_track_df['track_id'].unique()])
        std_print('INFO', 'md', 'MD tracks for individual export', len(unique_track_ids))
        # Results containers
        all_correlation_results = []
        all_sm_results = []
        all_flux_results = []
        all_landfall_results = []
        all_classifications = []
        asm_results_list = []
        multiwindow_data = []
        track_dict = {}
        flux_dict = {} 
        # Process each track
        for track_id in unique_track_ids:
            track_id = int(track_id)
            track_df = full_track_df[full_track_df['track_id'] == track_id].copy()
            track_dict[track_id] = track_df.copy() # for coupling metrics
            track_dir = os.path.join(output_dir, f"track_{int(track_id)}")
            os.makedirs(track_dir, exist_ok=True)
            track_df = full_track_df[full_track_df['track_id'] == track_id].reset_index(drop=True).copy()

            # 1) Land/Ocean classification
            # Ensure is_land column exists
            landfall_time = find_landfall_time(track_df)
            std_print('INFO', 'md', 'Land/ocean classification computed', f'{track_df["is_land"].sum()} land points')
            # 2) Smooth coordinates
            track_df = smooth_track_coordinates(track_df, window_size=3, iterations=4)
            print(f"\n--------------------------------- MD Track {track_id} ---------------------------------")
            diagnose_detection_issues(ds, track_df)
            # 3) Plot maps
            plot_track_full_map(track_df, track_id, full_extent, track_dir)
            plot_single_track(track_df, track_id, track_dir)
            # 4) Export coordinates
            export_single_track_coordinates(track_df, track_id, track_dir)
            # 5) Export detections csv
            track_csv = os.path.join(track_dir, f"track_{int(track_id)}_detections.csv")
            track_df.to_csv(track_csv, index=False)
            print(f" >> [EXPORTED] Track detections CSV of {track_id}         :          {track_csv}")
            # 6) Plot speed analysis
            print("\n---------------------------------- Translational Speed Plots --------------------------")
            speed_dir = os.path.join(track_dir, "translational-speed")
            os.makedirs(speed_dir, exist_ok=True)
            track_df = compute_track_speeds(track_df)
            track_csv = os.path.join(speed_dir, f"track_{int(track_id)}_detections_augmented.csv")
            track_df.to_csv(track_csv, index=False)
            print(f" >> [SAVED   ] Augmented track data for Track {track_id}  :          {track_csv}")
            plot_speed_analysis_kmh(track_df, track_id, speed_dir)
            plot_speed_analysis_ms(track_df, track_id, speed_dir)
            print(f" >  [INFO md   ] Speed Statistics for Track {track_id}      : \n"
                  f"                mean   = {track_df['speed_kmh'].mean():.2f} km/h ({track_df['speed_ms'].mean():.2f} m/s), \n"
                  f"                median = {track_df['speed_kmh'].median():.2f} km/h ({track_df['speed_ms'].median():.2f} m/s)"
            )
            # print(f" > [DEBUG  ] Mean speed: {metrics['mean_speed_kmh']:.1f} km/h ({metrics['mean_speed_ms']:.1f} m/s)")
            # print(f" > [DEBUG  ] Median speed: {metrics['median_speed_kmh']:.1f} km/h")
            # print(f" > [DEBUG  ] Speed range: {metrics['min_speed_kmh']:.1f}-{metrics['max_speed_kmh']:.1f} km/h")
            # print(f" > [DEBUG  ] Track duration: {metrics.get('duration_hours', 0):.1f} hours")
            # print(f" > [DEBUG  ] Total distance: {metrics.get('total_distance_km', 0):.1f} km")
            # 7) Generate Hovmöller Plots
            print("\n---------------------------------- Azimuthal Hovmoller Plots --------------------------")
            std_print('DEBUG', 'md', f'Skipping Hovmöller for Track {track_id}')
            # std_print('INFO', 'md', f'Computing Hovmöller for Track {track_id}')
            # hov_dir = os.path.join(track_dir, "hovmoller")
            # os.makedirs(hov_dir, exist_ok=True)
            # hov_data, track_times, landfall_index = compute_azimuthal_hovmoller(
            #     ds=ds,
            #     track_df=track_df,
            #     radii_km=radii,
            #     variables=variables,
            #     n_theta=72
            # )
            # for var in variables:
            #     print(f" >  [PLOTTING] Hovmöller for variable            :          {var}")
            #     plot_hovmoller(
            #         hov_arr=hov_data[var],
            #         track_times=track_times,
            #         radii_km=radii,
            #         var_name=var,
            #         track_df=track_df,
            #         outdir=hov_dir
            #     )
            # 8) Antecedent Soil Moisture Analysis
            print("\n-------------------------------- Antecedent Soil Moisture Analysis ---------------------")
            std_print('INFO', 'md', f'Computing antecedent SM for Track {track_id}')
           # Ensure is_land column exists for antecedent analysis
            track_df['is_land'] = track_df.apply(lambda row: is_point_land(row['latitude'], row['longitude']), axis=1)
            std_print('INFO', 'md', 'Land/ocean classification computed', f'{track_df["is_land"].sum()} land points')
            try:
                asm_dir = os.path.join(track_dir, "antecedent_sm")
                os.makedirs(asm_dir, exist_ok=True)
                # Compute multi-window ASM (OCEAN -> LAND TRANSITION FOCUSED)
                multiwindow_results = compute_antecedent_sm_multiwindow(ds=ds, track_df=track_df, track_id=track_id, buffer_km=100, land_only=False)
                if multiwindow_results:
                    export_multiwindow_results(track_id, multiwindow_results, asm_dir)
                    # Compute ASM for single track (track_id)
                    asm_24h_value = multiwindow_results['antecedent_24h']['value']
                    if isinstance(asm_24h_value, dict):
                        asm_24h_value = asm_24h_value.get('value', np.nan)
                        asm_results_list.append({'track_id': track_id, 'ASM_24h': asm_24h_value, 'has_landfall': True})
                        print(f"DEBUG  md Track {track_id} Multi-window ASM computed")
                    # No need of this since 'land_only=true'
                    # # Plot timeseries if landfall detected
                    # landfall_detected = any(
                    #     not np.isnan(multiwindow_results[window]['value']) 
                    #     for window in multiwindow_results
                    # )
                    # if landfall_detected:
                    plot_antecedent_sm_timeseries(ds, track_df, track_id, buffer_km=100, output_dir=asm_dir)
                else:
                    std_print('WARN', 'md', f'Track {track_id} No multi-window ASM computed')
            except Exception as e:
                std_print('ERROR', 'md', f'Track {track_id} ASM analysis failed', str(e))
                traceback.print_exc()
            try:
                post_asm_value = compute_antecedent_sm_post_landfall(ds=ds, track_df=track_df, track_id=track_id, hours_after_landfall=12, buffer_km=100)
                if post_asm_value is not None and isinstance(post_asm_value, (int, float)) and not np.isnan(post_asm_value):
                    for res in asm_results_list:
                        if res['track_id'] == track_id:
                            res['Post_ASM_12h'] = post_asm_value
            except Exception as e:
                std_print('ERROR', 'md', f'Track {track_id} Post-landfall failed', str(e))
                traceback.print_exc()
            # 9) Correlation analysis (swvl1 vs tp vs heat flux)
            print("\n---------------------------------- Correlation analysis --------------------------------")
            std_print('INFO', 'md', f'Track {track_id}: Correlation analysis')
            correlation_dir = os.path.join(track_dir, "correlation_analysis")
            os.makedirs(correlation_dir, exist_ok=True)
            try:
                correlation_result = analyze_single_track(ds=ds, track_df=track_df, track_id=track_id, buffer_km=100, max_lag_hours=24, output_dir=correlation_dir)
                if correlation_result:
                    all_correlation_results.append({
                        'track_id': track_id,
                        'correlation': correlation_result['correlation'],
                        'p_value': correlation_result['p_value'],
                        'mean_tp': correlation_result['mean_tp'],
                        'mean_swvl1': correlation_result['mean_swvl1'],
                        'n_samples': correlation_result['n_samples'],
                        'landfall_time': correlation_result.get('landfall_time'),
                        'landfall_index': correlation_result.get('landfall_index')
                    })
            except Exception as e:
                std_print('ERROR', 'md', 'Correlation analysis failed', e)
            flux_dir = os.path.join(track_dir, "flux_analysis")
            os.makedirs(flux_dir, exist_ok=True)
            try:
                flux_results = analyze_flux_correlations(ds, track_df, track_id, buffer_km=100)
                if flux_results:
                    plot_flux_analysis(flux_results, flux_dir, track_id)
                    flux_csv_path = os.path.join(flux_dir, f'track_{track_id}_flux_summary.csv')
                    pd.DataFrame([flux_results]).to_csv(flux_csv_path, index=False)
                    print(f" >> [EXPORTED] Flux Summary CSV                  :          {os.path.basename(flux_csv_path)}")
            except Exception as e:
                std_print('ERROR', 'md', f'Flux analysis failed for Track {track_id}', str(e))
            # 10) Landfall flux analysis
            print("\n----------------------------------- Landfall Flux Analysis -----------------------------")
            try:
                landfall_result = analyze_flux_transition(ds, track_df, track_id, hours_before=24, hours_after=24)
                if landfall_result:
                    print(f" >  [INFO   ] Landfall flux transition Track {track_id}   :")
                    print(f"                Before landfall                  : \n"
                          f"                 Latent                          :          {landfall_result['before']['latent']:.1f} W/m²,\n"
                          f"                 Sensible                        :          {landfall_result['before']['sensible']:.1f} W/m²"
                    )
                    print(f"                After landfall                   : \n"
                          f"                 Latent                          :          {landfall_result['after']['latent']:.1f} W/m², \n"
                          f"                 Sensible                        :          {landfall_result['after']['sensible']:.1f} W/m²"
                    )
                    print(f"                Change                           : \n"
                          f"                 Total                           :          {landfall_result['change']['total']:.1f} W/m²"
                    )
                    all_landfall_results.append(landfall_result)
                    landfall_csv = os.path.join(flux_dir, f'track_{track_id}_landfall_flux.csv')
                    pd.DataFrame([landfall_result]).to_csv(landfall_csv, index=False)
                    print(f" >> [SAVED   ] Landfall flux transition          :          {landfall_csv}")
            except Exception as e:
                print(f" >  [WARN ] Landfall flux analysis failed for track {track_id}: {e}")
            # 11) Soil moisture analysis
            print("\n---------------------------------- Soil moisture analysis ------------------------------")
            try:
                sm_result = compute_single_track_correlation(ds=ds, track_df=track_df, track_id=track_id, buffer_km=100)
                if sm_result:
                    all_sm_results.append(sm_result)
                    sm_dir = os.path.join(track_dir, "sm_analysis")
                    os.makedirs(sm_dir, exist_ok=True)
                    plot_track_timeseries(ds, track_df, track_id, buffer_km=100, output_dir=sm_dir)
                    plot_spatial_tp_sm(ds, track_df, track_id, buffer_deg=5.0, output_dir=sm_dir)
                    print(f" >> [SAVED   ] SM analysis plots for Track {track_id}")
            except Exception as e:
                 std_print('ERROR', 'md', f"SM analysis failed for Track {track_id}")
            # 12) Heat Flux Analysis and MD Duration Classification
            print("\n------------------------------------ Heat Flux & Duration Analysis ---------------------")
            classification = classify_md_duration(track_df)
            classification['track_id'] = track_id
            all_classifications.append(classification)
            print(f"Classification Track {track_id} which is {classification['category']}:")
            print(f"                  Duration                       :          {classification['duration_hours']:.1f} hours ({classification['num_points']} points)")
            hf_dir = os.path.join(track_dir, "heat_flux_analysis")
            os.makedirs(hf_dir, exist_ok=True)
            flux_df = extract_heat_flux_timeseries(ds, track_df, track_id, buffer_km=100)
            if flux_df is not None:
                flux_dict[track_id] = flux_df.copy() # store for coupling metrics
                print(f" >  [INFO md   ] Heat flux data extracted          :          {len(flux_df)} points")
                flux_analysis = analyze_heat_flux_evolution(flux_df, track_df, track_id)
                if flux_analysis:
                    all_flux_results.append({
                        'track_id': track_id,
                        'is_long_lived': classification['is_long_lived'],
                        'duration_hours': classification['duration_hours'],
                        'category': classification['category'],
                        **flux_analysis
                    })
                    print(f" >  [INFO md   ] Heat flux analysis for Track {track_id}    :")
                    print(f"                Mean Total HF                    :          {flux_analysis['mean_total_hf']:.1f} W/m²")
                    print(f"                Mean Latent HF                   :          {flux_analysis['mean_latent_hf']:.1f} W/m²")
                    print(f"                Mean Sensible HF                 :          {flux_analysis['mean_sensible_hf']:.1f} W/m²")
                    if 'ocean_mean_total_hf' in flux_analysis and 'land_mean_total_hf' in flux_analysis:
                        print(f"                Ocean phase Total HF             :          {flux_analysis['ocean_mean_total_hf']:.1f} W/m²")
                        print(f"                Land phase Total HF              :          {flux_analysis['land_mean_total_hf']:.1f} W/m²")
                        print(f"                Change (Ocean->Land)             :          {flux_analysis['total_hf_change']:.1f} W/m²")
                plot_heat_flux_timeseries(flux_df, track_id, hf_dir)
            else:
                 std_print('WARN', 'md', f"No heat flux data for Track {track_id}")
            print(f"DEBUG Heat flux transition analysis for Track {track_id}")
            try:
                landfall_result = analyze_flux_transition(ds=ds, track_df=track_df, track_id=track_id, hours_before=24, hours_after=24)
                if landfall_result:
                    all_landfall_results.append(landfall_result)
                    landfall_csv = os.path.join(hf_dir, f'track_{track_id}_flux_transition.csv')
                    pd.DataFrame([landfall_result]).to_csv(landfall_csv, index=False)
                    print(f" >> [SAVED   ] Flux transition analysis          :          {landfall_csv}")
                else:
                    std_print('WARN', 'md', f"Could not analyze flux transition for Track {track_id}")
            except Exception as e:
                std_print('ERROR', 'md', f"Flux transition analysis failed for Track {track_id}: {e}")
                traceback.print_exc()

        # Coupling metrics
        std_print('INFO', 'md', "Starting Coupling Metrics Analysis")
        correlation_df = pd.DataFrame(all_correlation_results) if all_correlation_results else pd.DataFrame()
        duration_df = pd.DataFrame(all_classifications) if all_classifications else pd.DataFrame()
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
                std_print('INFO', 'md', "Computing Feedback Strength Index (FSI)")
                fsi_df = compute_fsi_all_tracks(correlation_df, track_dict, ds)
            else:
                std_print('WARN', 'md', "Insufficient data for FSI computation")
                fsi_df = pd.DataFrame()
            # Compute Diurnal Amplitude
            if flux_dict:
                std_print('INFO', 'md', "Computing Diurnal SHF Amplitude")
                amplitude_df = compute_diurnal_amplitude_all_tracks(flux_dict, track_dict)
            else:
                std_print('WARN', 'md', "No flux data available for diurnal amplitude")
                amplitude_df = pd.DataFrame()
            # Compute Coupling Efficiency
            if not fsi_df.empty and not amplitude_df.empty and not duration_df.empty:
                std_print('INFO', 'md', "Computing Coupling Efficiency")
                coupling_efficiency_df = compute_coupling_efficiency(fsi_df, correlation_df, amplitude_df, duration_df)
                export_coupling_metrics(fsi_df, amplitude_df, coupling_efficiency_df, coupling_metrics_dir)
                generate_coupling_metrics_report(fsi_df, amplitude_df, coupling_efficiency_df, coupling_metrics_dir)
            else:
                std_print('WARN', 'md', "Insufficient data for coupling efficiency computation")
        except Exception as e:
            std_print('ERROR', 'md', f"Coupling metrics analysis failed: {e}")
            traceback.print_exc()

        # Post-loop processing: Global Analysis
        if not full_track_df.empty:
            std_print('INFO', 'md', "Starting Global Heat Flux & Duration Analysis")

            # 1. Prepare Antecedent Soil Moisture (ASM) Data
            all_asm_df = pd.DataFrame(asm_results_list) if asm_results_list else pd.DataFrame()
            asm_analysis_dir = os.path.join(output_dir, "antecedent_analysis")
            os.makedirs(asm_analysis_dir, exist_ok=True)
            if not all_asm_df.empty:
                export_antecedent_results(all_asm_df, asm_analysis_dir)
                std_print('INFO', 'md', f"Saved combined ASM results for {len(all_asm_df)} tracks")
            else:
                std_print('WARN', 'md', "No ASM results collected")

            # 2. Multi-Window ASM Aggregation (Loading from saved CSVs)
            std_print('INFO', 'md', "Aggregating Multi-Window ASM data from track folders")
            for track_id in unique_track_ids:
                asm_track_csv = os.path.join(output_dir, f"track_{int(track_id)}", "antecedent_sm", 'asm_multiwindow_results.csv')
                if os.path.exists(asm_track_csv):
                    try:
                        track_multi_df = pd.read_csv(asm_track_csv)
                        if 'track_id' not in track_multi_df.columns:
                            track_multi_df['track_id'] = int(track_id)
                        multiwindow_data.append(track_multi_df)
                        std_print('INFO', 'md', f"Loaded multi-window data for Track {track_id}")
                    except Exception as e:
                        std_print('WARN', 'md', f"Could not load multi-window CSV for Track {track_id}: {e}")

            # 3. Process Multi-Window Results and Generate Spatial Maps
            if multiwindow_data:
                all_multiwindow_df = pd.concat(multiwindow_data, ignore_index=True)
                multi_out_path = os.path.join(asm_analysis_dir, 'asm_multiwindow_all_tracks.csv')
                all_multiwindow_df.to_csv(multi_out_path, index=False)
                std_print('INFO', 'md', f"Aggregated multi-window ASM results saved to {multi_out_path}")
                # Plot Spatial Maps for tracks with valid data
                for _, multi_row in all_multiwindow_df.iterrows():
                    t_id = multi_row['track_id']
                    # Check for any valid ASM window
                    if any(not np.isnan(multi_row.get(f'{w}_asm', np.nan)) for w in ['antecedent_24h', 'approach_12h', 'landfall_6h', 'post_landfall_6h']):
                        track_subset = full_track_df[full_track_df['track_id'] == t_id]
                        if len(track_subset) >= 3:
                            std_print('INFO', 'md', f"Generating Before/After Spatial Map for Track {t_id}")
                            plot_sm_tp_map(ds=ds, track_df=track_subset, track_id=t_id, hours_before=24, hours_after=24, buffer_km=300, min_extent_deg=10.0, output_dir=asm_analysis_dir)

            # 4. Global ASM Correlation Analysis
            # We re-compute the clean 24h ASM for all tracks to ensure consistency for correlation
            std_print('INFO', 'md', "Computing Global ASM Correlation Analysis")
            all_asm_df_clean = compute_antecedent_sm_all_tracks(ds=ds, full_track_df=full_track_df, hours_before_landfall=24, exclude_last_hours=6, buffer_km=100)
            if not all_asm_df_clean.empty:
                # A) ASM vs. Precipitation
                corr_tp_df = compute_asm_tp_correlation(ds=ds, full_track_df=full_track_df, asm_results_df=all_asm_df_clean, output_dir=asm_analysis_dir)
                # B) ASM vs. Duration
                duration_df = pd.DataFrame(all_classifications)
                corr_duration_df = compute_asm_duration_correlation(asm_results_df=all_asm_df_clean, duration_df=duration_df, output_dir=asm_analysis_dir)
                # C) ASM Summary Report
                create_asm_summary_report(asm_results_df=all_asm_df_clean, corr_tp_df=corr_tp_df, corr_duration_df=corr_duration_df, output_dir=asm_analysis_dir)

            # 5. Global Flux Analysis
            std_print('INFO', 'md', "Finalizing Global Flux Analysis")
            if all_flux_results:
                all_flux_results = ensure_flux_results_complete(all_flux_results, all_classifications)
                export_heat_flux_results(all_flux_results, output_dir)
            if all_landfall_results:
                export_landfall_flux_results(all_landfall_results, output_dir)
            else:
                std_print('WARN', 'md', "No heat flux transition results to export")

            # 6. Final Summaries and Report Generation
            if all_classifications:
                print_md_classification_summary(all_classifications, output_dir)
            std_print('INFO', 'md', "Generating Master Summary Report")
            try:
                generate_summary_report(
                    all_correlation_results=all_correlation_results,
                    all_flux_results=all_flux_results,
                    all_landfall_results=all_landfall_results,
                    all_classifications=all_classifications,
                    all_sm_results=all_sm_results,
                    output_dir=output_dir
                )
            except Exception as e:
                std_print('ERROR', 'md', f"Master summary report generation failed: {e}")

            export_track_coordinates(full_track_df, output_dir)
            std_print('INFO', 'md', "Global Post-Processing Finished Successfully")
        else:
            std_print('INFO', 'md', "No tracks found after processing; skipping global analysis")
    print("----------------------------------------------------------------------------------------")
    end_time = time.time()
    total_seconds = end_time - start_time
    if total_seconds < 60:
        print(f"\n > Executed in {total_seconds:.2f} seconds.\n")
    elif total_seconds < 3600:
        mins = int(total_seconds // 60)
        secs = int(total_seconds % 60)
        print(f"\n > Executed in {mins} mins {secs} secs.\n")
    else:
        hrs = int(total_seconds // 3600)
        mins = int((total_seconds % 3600) // 60)
        secs = int(total_seconds % 60)
        print(f"\n > Executed in {hrs} hrs {mins} mins {secs} secs.\n")
