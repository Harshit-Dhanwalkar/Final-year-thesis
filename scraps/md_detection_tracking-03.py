import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
import math

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from azimuthal_hovmoller import compute_azimuthal_hovmoller,plot_hovmoller
from speed_analysis import is_point_land

from typing import Optional, Tuple, List, Dict

import traceback
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f" >> Created output directory: {output_dir}")
else:
    print(f" >> {output_dir} directory already present.")
print("------------------------------------------------------------")

DATA_DIR = os.path.join("Data", "single-levels", "ncfiles")
MONTH_DIRS = [
    "July-2024",
    "August-2024",
    "September-2024",
]

DISTANCE_THRESHOLD_KM = 150  # 300     # km
TIME_THRESHOLD_HOURS = 1.5   # 3       # hr
WIND_SPEED_LOWER_BOUND_MS = 8.5  # m/s
WIND_SPEED_UPPER_BOUND_MS = 17       # m/s
MSL_LOWER_BOUND_PA = 100000  # 99800   # Pa
PRECIPITATION_THRESHOLD_MM = 0.5       # mm/hr
BUFFER = 5.0                           # deg

EARTH_R = 6371.0 # km

month_colors = {
    "July": "red",
    "August": "blue",
    "September": "purple",
}
month_order = [
    "July",
    "August",
    "September",
]

# hovmoller
radii = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350]
variables = [
    "wind10",      # wind10 = sqrt(u10^2 + v10^2)
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

    # Ensure the BUFFER region is within the dataset bounds
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

def calculate_vorticity(ds):
    # Extract lat/lon in radians for proper metric calculations
    lat_rad = np.deg2rad(ds.latitude.values)
    lon_rad = np.deg2rad(ds.longitude.values)

    # Calculate grid spacing in meters
    dlon = np.gradient(lon_rad, axis=-1)  # dλ in radians
    dlat = np.gradient(lat_rad, axis=0)   # dφ in radians

    # Convert to meters at each latitude
    dx = EARTH_R * np.cos(lat_rad[:, np.newaxis]) * dlon  # dx = R cos(φ) dλ
    dy = EARTH_R * dlat  # dy = R dφ

    dv_dx = np.gradient(ds['v10'].values, axis=2) / dx[:, :, np.newaxis]
    du_dy = np.gradient(ds['u10'].values, axis=1) / dy[:, :, np.newaxis]
    vorticity = dv_dx - du_dy

    vorticity_da = xr.DataArray(
        vorticity,
        dims=ds['u10'].dims,
        coords=ds['u10'].coords,
        name='vorticity_10m'
    )

    return vorticity_da

def check_vorticity_criteria(data_slice, lat, lon, current_time=None, ds=None):
    # if 'vo' in data_slice.data_vars:
    #     vo_data = data_slice["vo"]
    #     # Extract region around center
    #     subset = vo_data.sel(
    #         latitude=slice(lat+radius_deg, lat-radius_deg),
    #         longitude=slice(lon-radius_deg, lon+radius_deg)
    #     )
    #     if subset.mean().item() > 0:
    #         return True
    try:
        # Find the time step index
        if 'time' in data_slice.coords:
            current_time = data_slice['time'].values
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
        return vorticity > 1e-5, vorticity

    except Exception as e:
        logger.warning(f"Error computing vorticity: {e}")
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
            # print("Warning: Missing 'tp'. Skipping this check.")
            return False
    except KeyError as e:
        print(
            f"Warning: Missing variable for moisture check: {e}. Skipping this check."
        )
        return False

def calculate_distances(lat_grid, lon_grid, lat, lon):
    lat_rad_grid = np.radians(lat_grid)
    lon_rad_grid = np.radians(lon_grid)
    center_lat_rad = np.radians(lat)
    center_lon_rad = np.radians(lon)

    dlat = lat_rad_grid - center_lat_rad
    dlon = lon_rad_grid - center_lon_rad

    a = np.sin(dlat/2)**2 + np.cos(center_lat_rad) * np.cos(lat_rad_grid) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return EARTH_R * c

def find_low_pressure_center(
    data_slice: xr.Dataset, 
    time_step: np.datetime64,
    prev_center: Optional[Tuple[float, float]] = None,
    search_radius_km: float = 300
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
        n_candidates = 5
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
            if pressure > MSL_LOWER_BOUND_PA + 500:  # Allow slightly higher pressure
                continue

            # Calculate distance from previous center if available
            if prev_center is not None:
                prev_lat, prev_lon = prev_center
                distance = haversine(prev_lon, prev_lat, lon, lat)
                if distance > search_radius_km:
                    continue

            candidates.append({
                'lat': lat,
                'lon': lon,
                'pressure': pressure,
                'distance': distance if prev_center else 0
            })

        if not candidates:
            return None

        # Sort candidates by pressure (lowest first), then by distance from prev center
        if prev_center:
            candidates.sort(key=lambda x: (x['pressure'], x['distance']))
        else:
            candidates.sort(key=lambda x: x['pressure'])

        # Try each candidate until we find one that meets criteria
        for candidate in candidates:
            lat = candidate['lat']
            lon = candidate['lon']
            min_msl = candidate['pressure']

            # Relaxed criteria check
            pressure_ok = min_msl <= MSL_LOWER_BOUND_PA + 1000  # More flexible
            closed_contour = has_closed_contour(msl_data, lat, lon)
            wind_ok = check_wind_criteria(u_data, v_data, lat, lon)
            moisture_ok = check_moisture_criteria(data_slice, lat, lon)

            # Vorticity check - make optional for now
            try:
                vorticity_ok, _ = check_vorticity_criteria(data_slice, lat, lon)
            except:
                vorticity_ok = True  # Skip if error

            if (pressure_ok and closed_contour and wind_ok and moisture_ok and vorticity_ok):
                return lat, lon

        return None

    except Exception as e:
        logger.warning(f"Error finding low pressure center at {time_step}: {e}")
        return None

# helper function to calculate haversine distance
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

# def group_detections_into_tracks(
#     detections, time_threshold_hours=TIME_THRESHOLD_HOURS,
#     distance_threshold_km=DISTANCE_THRESHOLD_KM
# ):
#     if detections.empty:
#         return []
#
#     detections = detections.sort_values("time").reset_index(drop=True)
#     tracks = []
#     current_track = []
#
#     for i in range(len(detections)):
#         if not current_track:
#             current_track.append(i)
#             continue
#
#         prev_idx = current_track[-1]
#         current_idx = i
#
#         prev_time = pd.to_datetime(detections.iloc[prev_idx]["time"])
#         current_time = pd.to_datetime(detections.iloc[current_idx]["time"])
#         time_diff_hours = (current_time - prev_time).total_seconds() / 3600
#
#         prev_lat = detections.iloc[prev_idx]["latitude"]
#         prev_lon = detections.iloc[prev_idx]["longitude"]
#         current_lat = detections.iloc[current_idx]["latitude"]
#         current_lon = detections.iloc[current_idx]["longitude"]
#
#         distance = haversine(prev_lon, prev_lat, current_lon, current_lat)
#
#         # Check detection be part of the current track
#         if (
#             time_diff_hours <= time_threshold_hours
#             and distance <= distance_threshold_km
#         ):
#             current_track.append(i)
#         else:
#             tracks.append(detections.iloc[current_track].copy())
#             current_track = [i]
#
#     # Add the last track
#     if current_track:
#         tracks.append(detections.iloc[current_track].copy())
#
#     return tracks
def track_monsoon_depressions(detections, max_speed_ms=10, search_radius_km=300, max_gap_hours=6):
    if detections.empty:
        return []

    detections = detections.sort_values("time").reset_index(drop=True)
    detections_list = detections.to_dict('records')

    tracks = []
    assigned = set()

    for i in range(len(detections_list)):
        if i in assigned:
            continue

        # Start new track
        current_track = [i]
        assigned.add(i)

        current_idx = i
        current_point = detections_list[i]

        keep_searching = True
        while keep_searching:
            keep_searching = False
            current_time = pd.to_datetime(current_point["time"])

            # Look ahead for potential next points
            best_next_idx = None
            best_score = float('inf')

            for j in range(current_idx + 1, len(detections_list)):
                if j in assigned:
                    continue

                next_point = detections_list[j]
                next_time = pd.to_datetime(next_point["time"])

                # Time difference in hours
                time_diff = (next_time - current_time).total_seconds() / 3600

                # Only consider points within reasonable time
                if time_diff < 1 or time_diff > max_gap_hours:
                    continue

                # Calculate distance
                distance = haversine(
                    current_point["longitude"], current_point["latitude"],
                    next_point["longitude"], next_point["latitude"]
                )

                if distance > search_radius_km:
                    continue

                # Calculate speed
                speed_ms = (distance * 1000) / (time_diff * 3600)  # m/s
                if speed_ms > max_speed_ms:
                    continue

                # Score based on combination of distance and time
                score = distance + (time_diff * 5)  # Weight time difference

                if score < best_score:
                    best_score = score
                    best_next_idx = j

            # If we found a suitable next point
            if best_next_idx is not None:
                current_track.append(best_next_idx)
                assigned.add(best_next_idx)
                current_idx = best_next_idx
                current_point = detections_list[best_next_idx]
                keep_searching = True

        # Only keep tracks with reasonable length
        if len(current_track) >= 3:  # 3-point tracks are useful initially
            track_df = detections.iloc[current_track].copy()
            tracks.append(track_df)

    return tracks

def merge_close_tracks(tracks, time_gap_hours=6, distance_threshold_km=150):
    if len(tracks) < 2:
        return tracks

    merged = True
    while merged:
        merged = False
        new_tracks = []
        used = [False] * len(tracks)

        for i in range(len(tracks)):
            if used[i]:
                continue

            track_i = tracks[i].copy()
            track_i_end = pd.to_datetime(track_i["time"].iloc[-1])
            track_i_lat = track_i["latitude"].iloc[-1]
            track_i_lon = track_i["longitude"].iloc[-1]

            # Try to merge with subsequent tracks
            for j in range(i+1, len(tracks)):
                if used[j]:
                    continue

                track_j = tracks[j].copy()
                track_j_start = pd.to_datetime(track_j["time"].iloc[0])
                track_j_lat = track_j["latitude"].iloc[0]
                track_j_lon = track_j["longitude"].iloc[0]

                # Check time gap and distance
                time_gap = (track_j_start - track_i_end).total_seconds() / 3600
                distance = haversine(track_i_lon, track_i_lat, track_j_lon, track_j_lat)

                if 0 < time_gap <= time_gap_hours and distance <= distance_threshold_km:
                    # Merge tracks
                    merged_track = pd.concat([track_i, track_j], ignore_index=True)
                    merged_track = merged_track.sort_values("time").reset_index(drop=True)
                    track_i = merged_track
                    used[j] = True
                    merged = True
                    print(f"Merged tracks {i} and {j} (gap: {time_gap:.1f}h, distance: {distance:.1f}km)")

            new_tracks.append(track_i)
            used[i] = True

        tracks = new_tracks

    return tracks

def smooth_track_coordinates(track_df: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
    if len(track_df) < window_size:
        return track_df

    track_df = track_df.copy()

    # Check if required columns exist
    if 'latitude' not in track_df.columns or 'longitude' not in track_df.columns:
        logger.error("Track DataFrame is missing 'latitude' and/or 'longitude' columns")
        return track_df

    logger.info(f"Applying {window_size}-point rolling mean smoothing on track coordinates...")

    # Apply moving average
    lat_smooth = track_df['latitude'].rolling(window=window_size, center=True, min_periods=1).mean()
    lon_smooth = track_df['longitude'].rolling(window=window_size, center=True, min_periods=1).mean()

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

def filter_valid_tracks(
    tracks,
    min_length=3,
    min_duration_hours=24,
    min_distance_km=50,
    max_distance_km=1000,
):
    valid_tracks = []

    for track_idx, track in enumerate(tracks):
        print(f"\n[DEBUG] Checking Track {track_idx} with {len(track)} points")
        # Check minimum track length
        if len(track) < min_length:
            # logger.debug(f"Track {track_idx}: Skipped - too short ({len(track)} points)")
            print(f"  FAIL: Too short ({len(track)} points < {min_length})")
            continue
        else:
            print(f"  PASS: Length OK ({len(track)} points)")

        # Check minimum track duration (24 hours)
        start_time = pd.to_datetime(track["time"].iloc[0])
        end_time = pd.to_datetime(track["time"].iloc[-1])
        duration_hours = (end_time - start_time).total_seconds() / 3600

        if duration_hours < min_duration_hours:
            # logger.debug(f"Track {track_idx}: Skipped - duration too short ({duration_hours:.1f}h)")
            print(f"  FAIL: Duration too short ({duration_hours:.1f}h < {min_duration_hours}h)")
            continue
        else:
            print(f"  PASS: Duration OK ({duration_hours:.1f}h)")

        # Check if track has reasonable movement
        start_lat = track.iloc[0]["latitude"]
        start_lon = track.iloc[0]["longitude"]
        end_lat = track.iloc[-1]["latitude"]
        end_lon = track.iloc[-1]["longitude"]
        total_distance = haversine(start_lon, start_lat, end_lon, end_lat)

        if total_distance < min_distance_km:
            # logger.debug(f"Track {track_idx}: Skipped - distance too small ({total_distance:.1f}km)")
            print(f"  FAIL: Distance too small ({total_distance:.1f}km < {min_distance_km}km)")
            continue
        else:
            print(f"  PASS: Min distance OK ({total_distance:.1f}km)")
        # if total_distance > max_distance_km:
        #     logger.debug(f"Track {track_idx}: Skipped - distance too large ({total_distance:.1f}km)")
        #     continue

        # Check if track has reasonable continuity
        time_diffs = []
        distances = []

        for i in range(1, len(track)):
            prev_time = pd.to_datetime(track.iloc[i-1]["time"])
            curr_time = pd.to_datetime(track.iloc[i]["time"])
            time_diff = (curr_time - prev_time).total_seconds() / 3600

            prev_lat = track.iloc[i-1]["latitude"]
            prev_lon = track.iloc[i-1]["longitude"]
            curr_lat = track.iloc[i]["latitude"]
            curr_lon = track.iloc[i]["longitude"]
            distance = haversine(prev_lon, prev_lat, curr_lon, curr_lat)

            time_diffs.append(time_diff)
            distances.append(distance)

        # Check for reasonable time gaps (most should be 0.5-8 hours)
        if time_diffs:
            avg_time_gap = np.mean(time_diffs)
            if avg_time_gap > 8 or avg_time_gap < 0.5:
                logger.debug(f"Track {track_idx}: Skipped - unreasonable time gap ({avg_time_gap:.1f}h)")
                print(f"  FAIL: Unreasonable time gap (avg: {avg_time_gap:.1f}h)")
                continue
            else:
                print(f"  PASS: Time gap OK (avg: {avg_time_gap:.1f}h)")
        else:
            print(f"  WARNING: No time diffs to check")

        # Check for reasonable speeds
        if distances and time_diffs:
            speeds = [d/t if t > 0 else 0 for d, t in zip(distances, time_diffs)]
            avg_speed_kmh = np.mean(speeds)
            avg_speed_ms = avg_speed_kmh / 3.6

            # Monsoon depressions typically move at 3-6 m/s (10.8-21.6 km/h)
            if not (1.5 <= avg_speed_ms <= 10):
                # logger.debug(f"Track {track_idx}: Skipped - unreasonable speed ({avg_speed_ms:.1f}m/s)")
                print(f"  FAIL: Unreasonable speed ({avg_speed_ms:.1f}m/s)")
                continue
            else:
                print(f"  PASS: Speed OK ({avg_speed_ms:.1f}m/s)")
        else:
            print(f"  WARNING: No speeds to check")

        # Check track direction - typically moves NW or W
        bearing = calculate_track_direction(track)
        if bearing is not None:
            # Monsoon depressions in BOB typically move NW (270-360°) or W (240-300°)
            # if not (240 <= bearing <= 360 or 0 <= bearing <= 30):
            if not (200 <= bearing <= 360 or 0 <= bearing <= 50):
                # logger.debug(f"Track {track_idx}: Skipped - unlikely direction ({bearing:.1f}°)")
                print(f"  FAIL: Unlikely direction ({bearing:.1f}°)")
                continue
            else:
                print(f"  PASS: Direction OK ({bearing:.1f}°)")
        else:
            print(f"  WARNING: Could not calculate bearing")

        valid_tracks.append(track)
        # logger.info(f"Track {track_idx}: Accepted - {len(track)} points, {duration_hours:.1f}h, {total_distance:.1f}km, speed={avg_speed_ms:.1f}m/s, bearing={bearing:.1f}°")
        print(f"[DEBUG] Track {track_idx} ACCEPTED")

    print(f"[DEBUG] Total tracks accepted: {len(valid_tracks)} out of {len(tracks)}")
    return valid_tracks

def filter_tracks_by_duration(full_track_df, min_steps=24) -> pd.DataFrame:
    initial_track_count = len(full_track_df['track_id'].unique())
    track_counts = full_track_df.groupby('track_id').size()

    # Identify track IDs that meet the minimum length requirement
    long_tracks = track_counts[track_counts >= min_steps].index.tolist()

    filtered_df = full_track_df[full_track_df['track_id'].isin(long_tracks)].copy()
    final_count = len(filtered_df['track_id'].unique())

    logger.info(f"Duration filter applied (Min {min_steps} steps): Retained {final_count} tracks out of {initial_track_count}.")

    return filtered_df

def remove_isolated_tracks(tracks, min_points=3, min_duration_hours=24):
    filtered_tracks = []

    for track in tracks:
        # Basic length check
        if len(track) < min_points:
            continue

        # Duration check
        start_time = pd.to_datetime(track["time"].iloc[0])
        end_time = pd.to_datetime(track["time"].iloc[-1])
        duration_hours = (end_time - start_time).total_seconds() / 3600

        if duration_hours < min_duration_hours:
            continue

        # Check if track has reasonable point density
        time_diffs = []
        for i in range(1, len(track)):
            prev_time = pd.to_datetime(track.iloc[i-1]["time"])
            curr_time = pd.to_datetime(track.iloc[i]["time"])
            time_diff = (curr_time - prev_time).total_seconds() / 3600
            time_diffs.append(time_diff)

        # At least 60% of points should have reasonable time gaps (1-6 hours)
        # reasonable_gaps = sum(1 for td in time_diffs if 1 <= td <= 6)
        reasonable_gaps = sum(1 for td in time_diffs if 0.5 <= td <= 8)
        gap_ratio = reasonable_gaps / len(time_diffs) if time_diffs else 0

        # if gap_ratio < 0.6:
        if gap_ratio < 0.4:
            continue

        filtered_tracks.append(track)

    return filtered_tracks

def filter_by_spatial_clustering(tracks, min_cluster_size=3, max_distance_km=500):
    if len(tracks) < 2:
        return tracks

    # Calculate mean position for each track
    track_positions = []
    for track in tracks:
        mean_lat = track["latitude"].mean()
        mean_lon = track["longitude"].mean()
        track_positions.append((mean_lat, mean_lon))

    # Find clusters using simple distance-based clustering
    clusters = []
    visited = [False] * len(track_positions)

    for i in range(len(track_positions)):
        if not visited[i]:
            cluster = [i]
            visited[i] = True

            # Find all nearby tracks
            for j in range(i+1, len(track_positions)):
                if not visited[j]:
                    distance = haversine(
                        track_positions[i][1], track_positions[i][0],
                        track_positions[j][1], track_positions[j][0]
                    )
                    if distance < max_distance_km:
                        cluster.append(j)
                        visited[j] = True

            clusters.append(cluster)

    # Keep only tracks in clusters with minimum size
    filtered_tracks = []
    for cluster in clusters:
        if len(cluster) >= min_cluster_size:
            for idx in cluster:
                filtered_tracks.append(tracks[idx])

    print(f"Spatial clustering: {len(tracks)} -> {len(filtered_tracks)} tracks")
    return filtered_tracks

def filter_tracks_by_genesis_location(full_track_df: pd.DataFrame, land_check_func, coastal_buffer_km=100) -> pd.DataFrame:
    valid_track_ids = []
    initial_track_count = len(full_track_df['track_id'].unique())

    for track_id in full_track_df['track_id'].unique():
        track_df = full_track_df[full_track_df['track_id'] == track_id].sort_values(
            by='time', ascending=True
        ).reset_index(drop=True)

        # Get the first few points for better determination
        genesis_points = track_df.head(3)  # Check first 3 points

        # Check if majority of genesis points are over ocean
        ocean_count = 0
        for _, point in genesis_points.iterrows():
            if not land_check_func(point['latitude'], point['longitude']):
                ocean_count += 1

        # If at least 2 out of 3 genesis points are over ocean, keep track
        if ocean_count >= 2:
            valid_track_ids.append(track_id)
        else:
            logger.info(f"Filtered Track {track_id}: {ocean_count}/3 genesis points over ocean")

    filtered_df = full_track_df[full_track_df['track_id'].isin(valid_track_ids)].copy()

    logger.info(f"Genesis filter applied: Retained {len(valid_track_ids)} tracks out of {initial_track_count}.")

    return filtered_df

def process_files(ds):
    try:
        required_vars = ["msl", "u10", "v10", "tp"]
        if not all(var in ds.data_vars for var in required_vars):
            print(f"Error: Missing required variables")
            return None

        time_coord = "time" if "time" in ds.coords else "valid_time"
        total_steps = len(ds[time_coord].values)

        print(f"Total time steps: {total_steps}")
        print(f"Time range: {ds[time_coord].values[0]} to {ds[time_coord].values[-1]}")

        md_detections = []
        prev_center = None

        for i, time_step in enumerate(ds[time_coord].values):
            data_slice = ds.sel(**{time_coord: time_step})

            # Find center with continuity constraint
            center = find_low_pressure_center(
                data_slice, time_step, prev_center, search_radius_km=300
            )

            if center is not None:
                lat, lon = center
                md_detections.append({
                    "time": time_step,
                    "latitude": lat,
                    "longitude": lon,
                })
                prev_center = (lat, lon)  # Update for next time step
            else:
                prev_center = None  # Reset if no detection

        if not md_detections:
            print("No MD centers detected")
            return None

        detections_df = pd.DataFrame(md_detections)
        detections_df["month"] = pd.to_datetime(detections_df["time"]).dt.strftime("%B")

        print(f"Total detections: {len(detections_df)}")

        # Initial tracking
        tracks = track_monsoon_depressions(
            detections_df,
            max_speed_ms=8,  # MDs typically 3-6 m/s
            search_radius_km=300
        )

        print(f"Found {len(tracks)} initial tracks")

        # Debug: Print initial track lengths
        for i, track in enumerate(tracks):
            print(f"[DEBUG]  Track {i}: {len(track)} points, duration: {(pd.to_datetime(track['time'].iloc[-1]) - pd.to_datetime(track['time'].iloc[0])).total_seconds()/3600:.1f}h")

        # Merge split tracks
        tracks = merge_close_tracks(tracks, time_gap_hours=6, distance_threshold_km=150)
        print(f"[DEBUG]  After merging: {len(tracks)} tracks")

        # Debug: Print merged track lengths
        for i, track in enumerate(tracks):
            print(f"[DEBUG]  Track {i} after merge: {len(track)} points")

        if not tracks:
            return None

        # quality filtering
        filtered_tracks = filter_valid_tracks(
            tracks,
            min_length=3,                # At least 3 points
            min_duration_hours=24,       # At least 1 day
            min_distance_km=50,          # At least 50 km displacement
            max_distance_km=1000         # Not too far
        )
        print(f"After quality filtering: {len(filtered_tracks)} tracks")

        # Debug: Show what was filtered out
        if len(filtered_tracks) < len(tracks):
            print("\n[Debug] Tracks filtered out:")
            filtered_indices = set()
            for ft in filtered_tracks:
                # Find index in original tracks
                for idx, orig_track in enumerate(tracks):
                    if ft.equals(orig_track):
                        filtered_indices.add(idx)
                        break

            for i, track in enumerate(tracks):
                if i not in filtered_indices:
                    duration = (pd.to_datetime(track["time"].iloc[-1]) - pd.to_datetime(track["time"].iloc[0])).total_seconds() / 3600
                    print(f"  Track {i}: {len(track)} points, {duration:.1f}h")

        if not filtered_tracks:
            return None

        # Remove isolated tracks
        clean_tracks = remove_isolated_tracks(
            filtered_tracks,
            min_points=3,
            min_duration_hours=24
        )
        print(f"After removing isolated tracks: {len(clean_tracks)} tracks")

        if not clean_tracks:
            return None

        # Apply spatial clustering to remove distant outliers
        # clustered_tracks = filter_by_spatial_clustering(
        #     clean_tracks,
        #     min_cluster_size=1,      # At least 1 tracks in a region
        #     max_distance_km=300      # Tracks within 300km of each other
        # )
        #
        # print(f"After spatial clustering: {len(clustered_tracks)} tracks")
        #
        # if not clustered_tracks:
        #     return None

        # Smooth each track
        smoothed_tracks = []
        for track in filtered_tracks:
            smoothed_track = smooth_track_coordinates(track, window_size=3)
            smoothed_tracks.append(smoothed_track)

        # Assign track IDs
        track_df_with_ids = pd.concat(
            [track.assign(track_id=i+1) for i, track in enumerate(smoothed_tracks)],
            ignore_index=True
        )

        return track_df_with_ids

    except Exception as e:
        print(f"Error in process_files: {e}")
        traceback.print_exc()
        return None

    finally:
        if 'ds' in locals():
            ds.close()

def plot_tracks(track_df):
    fig = plt.figure(figsize=(12, 8))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # type: ignore[assignment]
    ax.set_extent([70, 100, 5, 30], crs=ccrs.PlateCarree())

    # Map features
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle='--')
    ax.add_feature(cfeature.RIVERS)

    # Grid lines
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
            markersize=2,
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
            linewidth=2,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

        # Annotate Start point
        plt.annotate(
            start_time,
            xy=(start_point["longitude"], start_point["latitude"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
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
            end_time,
            xy=(end_point["longitude"], end_point["latitude"]),
            xytext=(-3, -3),
            textcoords="offset points",
            fontsize=7,
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

    legend_handles = []
    for month, color in month_colors.items():
        if month in track_df["month"].unique():
            (legend_line,) = plt.plot(
                [],
                [],
                "-",
                color=color,
                linewidth=2,
                label=f"Low-press. sys. Track ({month})",
            )
            legend_handles.append(legend_line)

    plt.title(
        f"Detected MD Tracks ({month_order[0]}-{month_order[-1]})"
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(handles=legend_handles)

    output_filename = os.path.join(output_dir, "MD-track.png")
    print("\n---------------------- MD track plots ----------------------")
    plt.savefig(output_filename)
    print(f" > Combined Track plot saved as {output_filename}")
    plt.show()
    plt.close()

def plot_track_full_map(track_df_single, track_id, full_extent, output_dir):
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
        linewidth=2,
        markersize=4,
        zorder=5
    )

    # annotate start/end
    start = track_data.iloc[0]
    end = track_data.iloc[-1]
    start_label = pd.to_datetime(start["time"]).strftime('%d/%m %H:%M')
    end_label = pd.to_datetime(end["time"]).strftime('%d/%m %H:%M')
    ax.annotate(start_label, xy=(start["longitude"], start["latitude"]),
                xytext=(5,5), textcoords="offset points", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7),
                transform=ccrs.PlateCarree(), zorder=6)
    ax.annotate(end_label, xy=(end["longitude"], end["latitude"]),
                xytext=(5,-10), textcoords="offset points", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7),
                transform=ccrs.PlateCarree(), zorder=6)

    ax.set_title(f"Track of MD track {track_id}")
    outpath = os.path.join(output_dir, f"track_{int(track_id)}_full_map.png")
    # plt.show()
    fig.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f" >> Saved full-map for track {track_id}: {outpath}")

def plot_single_track(track_df, track_id, output_dir):
    track_data = track_df.sort_values("time")
    time_coord = 'time' if 'time' in track_df.columns else 'valid_time'
    if time_coord not in track_df.columns:
        time_coord = [col for col in track_df.columns if 'time' in col.lower()][0]

    start_time_str = pd.to_datetime(track_df[time_coord].iloc[0]).strftime('%d/%m/%Y %H:%M')
    end_time_str = pd.to_datetime(track_df[time_coord].iloc[-1]).strftime('%d/%m/%Y %H:%M')

    fig = plt.figure(figsize=(12, 10))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # type: ignore[assignment]

    min_lon, max_lon = track_df["longitude"].min() - 5, track_df["longitude"].max() + 5
    min_lat, max_lat = track_df["latitude"].min() - 5, track_df["latitude"].max() + 5

    month = track_data["month"].iloc[0]
    color = month_colors.get(month, "black")

    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Add map features
    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
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
        plt.plot(
            track_data["longitude"],
            track_data["latitude"],
            "o",
            color=color,
            markersize=2,
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
            linewidth=2,
            transform=ccrs.PlateCarree(),
            zorder=2
        )

        # Annotate Start point
        plt.annotate(
            start_label,
            xy=(start_point["longitude"], start_point["latitude"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
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
            fontsize=9,
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
        markersize=4,
        transform=ccrs.PlateCarree(),
        label=f'Track {track_id}',
        zorder=3
    )

    ax.plot(
        track_df["longitude"].iloc[0],
        track_df["latitude"].iloc[0],
        'go',
        markersize=8,
        transform=ccrs.PlateCarree(),
        label='Start',
        zorder=4
    )
    ax.plot(
        track_df["longitude"].iloc[-1],
        track_df["latitude"].iloc[-1],
        'rx',
        markersize=10,
        markeredgewidth=2,
        transform=ccrs.PlateCarree(),
        label='End',
        zorder=4
    )

    ax.set_title(f"MD Track : {start_time_str} to {end_time_str}")

    track_id_str = str(track_id).replace('/', '_').replace(' ', '_')
    file_name = f"MD-track-{track_id_str}-cropped.png"
    fullpath = os.path.join(output_dir, file_name)
    plt.savefig(fullpath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f" >> Saved cropped track {track_id_str} plot: {fullpath}")


# def print_track_details(track_df):
#     if track_df.empty:
#         print("No track details to display.")
#         return
#
#     print("\n--- Monsoon Depression Track Details ---")
#     for track_id, track_data in track_df.groupby("track_id"):
#         start_time = pd.to_datetime(track_data["time"].iloc[0])
#         end_time = pd.to_datetime(track_data["time"].iloc[-1])
#         duration_hours = (end_time - start_time).total_seconds() / 3600
#         start_month = track_data["month"].iloc[0]
#
#         print(f" Track ID: {track_id + 1} ({start_month})")
#         print(f"  Start: {start_time.strftime('%Y/%m/%d %H:%M')}")
#         print(f"  End:   {end_time.strftime('%Y/%m/%d %H:%M')}")
#         print(f"  Duration: {duration_hours:.1f} hours ({len(track_data)} steps)")
#         print("  Detections:")
#         display_df = track_data[["time", "latitude", "longitude"]].copy()
#         display_df["time"] = pd.to_datetime(display_df["time"]).dt.strftime('%Y/%m/%d %H:%M')
#         display_df["latitude"] = display_df["latitude"].apply(lambda x: f"{x:.2f}°N")
#         display_df["longitude"] = display_df["longitude"].apply(lambda x: f"{x:.2f}°E")
#         print(display_df.to_string(index=False, header=True))


def export_track_coordinates(track_df, output_dir):
    if track_df.empty:
        print("  Warning: No track coordinates to export.")
        return

    export_df = track_df[['time', 'latitude', 'longitude', 'track_id']].copy()
    export_df['time'] = pd.to_datetime(export_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    output_filename = os.path.join(
        output_dir, f"MD-coordinates-ALL.csv"
    )
    export_df.to_csv(output_filename, index=False)
    print(f" > Coordinates for {len(track_df)} detections exported to {output_filename}")


def export_single_track_coordinates(track_df, track_id, output_dir):
    export_df = track_df[['time', 'latitude', 'longitude', 'track_id']].copy()
    export_df['time'] = pd.to_datetime(export_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    track_id_str = str(track_id).replace('/', '_').replace(' ', '_')
    file_name = f"MD-track-{track_id_str}-coordinates.csv"
    export_df.to_csv(os.path.join(output_dir, file_name), index=False)
    print(f" >> Exported individual track {track_id} to: {file_name}")


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
            compat='override'
        ).compute()

        if 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})

        ds = ds.sortby('time')

        result_container["ds"] = ds
        result_container["success"] = True
    except Exception as e:
        result_container["error"] = str(e)
        result_container["success"] = False


if __name__ == "__main__":
    start_time = time.time()
    print("\n-------------------- Applied Thresholds --------------------")
    print(f"  Distance          : {DISTANCE_THRESHOLD_KM} km")
    print(f"  Time              : {TIME_THRESHOLD_HOURS} hr(s)")
    print(
        f"  Wind Speed (10m)  : {WIND_SPEED_LOWER_BOUND_MS} to {WIND_SPEED_UPPER_BOUND_MS} m/s"
    )
    print(f"  MSLP              : < {MSL_LOWER_BOUND_PA} Pa (or {MSL_LOWER_BOUND_PA / 100:.0f} hPa)")
    print(f"  Precipitation     : > {PRECIPITATION_THRESHOLD_MM} mm/hr")
    print()

    print(" > Combining all monthly files for track detection")

    start_processing_time = time.time()
    num_files = len(MONTH_DIRS) * 3

    result_container = {"ds": None, "success": False, "error": None}

    load_thread = threading.Thread(
        target=load_dataset_async, args=(MONTH_DIRS, DATA_DIR, result_container)
    )
    load_thread.daemon = True
    load_thread.start()

    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner_index = 0

    while load_thread.is_alive():
        current_spinner = spinner[spinner_index]
        spinner_index = (spinner_index + 1) % len(spinner)

        sys.stdout.write(f"\r{current_spinner} Loading {num_files} files from {len(MONTH_DIRS)} months...")
        sys.stdout.flush()
        time.sleep(0.1)

    if result_container["success"]:
        ds = result_container["ds"]
        end_processing_time = time.time()
        print(
            f"\r > Successfully loaded and combined files in {end_processing_time - start_processing_time:.2f} seconds.  "
        )
    else:
        print(f"\r > Error loading files: {result_container['error']}")
        sys.exit(1)

    print("\n-------------------- Dataset Info --------------------------")
    time_coord = "time" if "time" in ds.coords else "valid_time"

    years = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%Y").unique()
    months = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%b").unique()
    hours = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%H:%M").unique()
    ds_lat_min = ds.latitude.min().item()
    ds_lat_max = ds.latitude.max().item()
    ds_lon_min = ds.longitude.min().item()
    ds_lon_max = ds.longitude.max().item()

    time_values = ds[time_coord].values
    time_diff = time_values[1] - time_values[0]
    time_diff = pd.to_timedelta(time_diff)
    total_seconds = time_diff.total_seconds()
    resolution_hours = total_seconds / 3600.0

    print("  Year                  :", years)
    print("  Months                :", months)
    print("  Hours                 :", hours)
    print("  Variables             :", [var for var in ds.data_vars if var in ["msl", "u10", "v10", "tp", "t2m"]])
    print(f"  Dataset bounds      : lat [{ds_lat_min:.1f}, {ds_lat_max:.1f}] N, lon [{ds_lon_min:.1f}, {ds_lon_max:.1f}] E")
    print(f"  Data Time Resolution: {resolution_hours:.1f} hours ({time_diff})")

    print("\n-------------------- Processing ----------------------------")
    full_track_df = process_files(ds)

    if full_track_df is not None and not full_track_df.empty:
        if 'track_id' not in full_track_df.columns:
            print("Error: 'track_id' column not found in full_track_df. Cannot plot/export individual tracks.")
            sys.exit(1)

        # 1. Eliminate tracks that originate over land
        full_track_df = filter_tracks_by_genesis_location(
            full_track_df,
            land_check_func=is_point_land
        )

        # 2. Eliminate short-duration (noisy) tracks
        full_track_df = filter_tracks_by_duration(
            full_track_df,
            min_steps=24
        )
        # 3. Re-index tracks after filtering
        if not full_track_df.empty:
            unique_tracks = full_track_df['track_id'].unique()
            id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_tracks, 1)}
            full_track_df['track_id'] = full_track_df['track_id'].map(id_mapping)
            logger.info("Track IDs re-indexed after filtering.")

        # if 'track_id' in full_track_df.columns:
        #     initial_track_count = len(full_track_df['track_id'].unique())
        #     full_track_df = full_track_df[full_track_df['track_id'].isin(desired_tracks)].copy()
        #     filtered_track_count = len(full_track_df['track_id'].unique())
        #     print(f"\nINFO: Filtered tracks for Azimuthal-Hovmöller analysis.")
        #     print(f"INFO: Retaining {filtered_track_count} tracks ({desired_tracks}) out of {initial_track_count} total.")
        # else:
        #     print("WARNING: 'track_id' column not found. Skipping track filter.")

        print(f"Found {len(full_track_df['track_id'].unique())} tracks after physical constraints")
        plot_tracks(full_track_df)

        pad = 0.5
        try:
            ds_lon_min = ds.longitude.min().item()
            ds_lon_max = ds.longitude.max().item()
            ds_lat_min = ds.latitude.min().item()
            ds_lat_max = ds.latitude.max().item()
            full_extent = [ds_lon_min - pad, ds_lon_max + pad, ds_lat_min - pad, ds_lat_max + pad]
            print("[DEBUG]")
            print("Applied extent :", full_extent)
        except Exception:
            full_extent = [70, 100, 5, 30]
            print("[DEBUG]")
            print("Default extent :", full_extent)

        unique_track_ids = sorted(full_track_df['track_id'].unique())
        print(f"\n >> Found {len(unique_track_ids)} MD tracks for individual export.")

        for track_id in unique_track_ids:
            track_dir = os.path.join(output_dir, f"track_{int(track_id)}")
            os.makedirs(track_dir, exist_ok=True)

            # track_df = full_track_df[full_track_df['track_id'] == track_id].copy().reset_index(drop=True)
            track_df = full_track_df[full_track_df['track_id'] == track_id].copy()
            track_df['is_land'] = track_df.apply(
                lambda row: is_point_land(row['latitude'], row['longitude']), 
                axis=1
            )

            track_df = smooth_track_coordinates(track_df, window_size=3)
            logger.info(f" >> Smoothed coordinates for Track {track_id} (Window size: 3)")

            # 1) Full-map (full dataset extent, only this track shown)
            plot_track_full_map(track_df, track_id, full_extent, track_dir)
            # 2) Cropped map (zoom around track)
            plot_single_track(track_df, track_id, track_dir)
            # 3) Export coordinates for this single track into its dir
            export_single_track_coordinates(track_df, track_id, track_dir)
            # 4) Export the augmented combined CSV
            track_csv = os.path.join(track_dir, f"track_{int(track_id)}_detections.csv")
            track_df.to_csv(track_csv, index=False)
            print(f" >> Exported track detections CSV: {track_csv}")
            # 5) Generate Hovmöller Plots
            print(f" >> Computing Hovmöller for Track {track_id}")
            hov_dir = os.path.join(track_dir, "hovmoller")
            os.makedirs(hov_dir, exist_ok=True)
            hov_data, track_times = compute_azimuthal_hovmoller(
                ds=ds,
                track_df=track_df,
                radii_km=radii,
                variables=variables,
                max_workers=4
            )
            for var in variables:
                print(f" >> Plotting Hovmöller for {var}")
                plot_hovmoller(
                    hov_data[var],
                    track_times,
                    radii,
                    var,
                    track_df=track_df,
                    outdir=hov_dir
                )

        # print_track_details(full_track_df)
        export_track_coordinates(full_track_df, output_dir)
    else:
        print(" > No tracks found")

    end_time = time.time()

    print("------------------------------------------------------------")
    print(f"\nSuccessfully executed in {end_time - start_time:.2f} seconds.\n")
