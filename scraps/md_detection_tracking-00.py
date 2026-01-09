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

DISTANCE_THRESHOLD_KM = 300            # km
TIME_THRESHOLD_HOURS = 1               # hr
WIND_SPEED_LOWER_BOUND_MS = 8  # 7.5   # m/s
WIND_SPEED_UPPER_BOUND_MS = 16.0       # m/s
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

desired_tracks = [1, 3, 4, 5]


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


def find_low_pressure_center(data_slice, time_step):
    required_vars = ["msl", "u10", "v10", "tp"]
    if not all(var in data_slice.data_vars for var in required_vars):
        # print(f"Missing required single-level variables: {required_vars}")
        return None

    try:
        msl_data = data_slice["msl"]
        u_data = data_slice["u10"]
        v_data = data_slice["v10"]
        # tp_data = data_slice["tp"]

        min_msl_da = msl_data.where(
            msl_data == msl_data.min(), drop=True
        )
        if min_msl_da.size == 0:
            return None

        # Center of the low-pressure system is the location of minimum MSLP
        lat_center = min_msl_da.latitude.values[0]
        lon_center = min_msl_da.longitude.values[0]
        min_msl_val = min_msl_da.min().item()

        if (
            # 1. Low MSLP threshold
            min_msl_val <= MSL_LOWER_BOUND_PA
            # 2. Closed isobar check
            and has_closed_contour(msl_data, lat_center, lon_center)
            # 3. Wind criteria (using 10m wind)
            and check_wind_criteria(u_data, v_data, lat_center, lon_center)
            # 4. Moisture criteria (using total precipitation, tp)
            and check_moisture_criteria(data_slice, lat_center, lon_center)
        ):
            return lat_center, lon_center

    except Exception:
        print(f"Error processing time step {time_step}")
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

def group_detections_into_tracks(
    detections, time_threshold_hours=TIME_THRESHOLD_HOURS,
    distance_threshold_km=DISTANCE_THRESHOLD_KM
):
    if detections.empty:
        return []

    detections = detections.sort_values("time").reset_index(drop=True)
    tracks = []
    current_track = []

    for i in range(len(detections)):
        if not current_track:
            current_track.append(i)
            continue

        prev_idx = current_track[-1]
        current_idx = i

        prev_time = pd.to_datetime(detections.iloc[prev_idx]["time"])
        current_time = pd.to_datetime(detections.iloc[current_idx]["time"])
        time_diff_hours = (current_time - prev_time).total_seconds() / 3600

        prev_lat = detections.iloc[prev_idx]["latitude"]
        prev_lon = detections.iloc[prev_idx]["longitude"]
        current_lat = detections.iloc[current_idx]["latitude"]
        current_lon = detections.iloc[current_idx]["longitude"]

        distance = haversine(prev_lon, prev_lat, current_lon, current_lat)

        # Check detection be part of the current track
        if (
            time_diff_hours <= time_threshold_hours
            and distance <= distance_threshold_km
        ):
            current_track.append(i)
        else:
            tracks.append(detections.iloc[current_track].copy())
            current_track = [i]

    # Add the last track
    if current_track:
        tracks.append(detections.iloc[current_track].copy())

    return tracks


def filter_valid_tracks(
    tracks, min_track_length=5,
    min_track_duration_hours=24,
    min_total_distance=20
):
    valid_tracks = []

    for track in tracks:
        # Check minimum track length
        if len(track) < min_track_length:
            continue

        # Check minimum track duration (24 hours)
        start_time = pd.to_datetime(track["time"].iloc[0])
        end_time = pd.to_datetime(track["time"].iloc[-1])
        duration_hours = (end_time - start_time).total_seconds() / 3600

        if duration_hours < min_track_duration_hours:
            continue

        # Check if track has reasonable movement
        start_lat = track.iloc[0]["latitude"]
        start_lon = track.iloc[0]["longitude"]
        end_lat = track.iloc[-1]["latitude"]
        end_lon = track.iloc[-1]["longitude"]
        total_distance = haversine(start_lon, start_lat, end_lon, end_lat)

        if total_distance < min_total_distance:
            continue
        valid_tracks.append(track)

    return valid_tracks


def process_files(ds):
    try:
        required_vars = ["msl", "u10", "v10", "tp"]
        if not all(var in ds.data_vars for var in required_vars):
            print(
                f"Error: One or more required variables   \
                ({', '.join(required_vars)}) not found in \
                the combined dataset. Exiting..."
            )
            return None

        time_coord = "time" if "time" in ds.coords else "valid_time"
        total_steps = len(ds[time_coord].values)

        # DEBUG:
        print(f"Total time steps in dataset: {total_steps}")
        print(f"Time range: {ds[time_coord].values[0]} to {ds[time_coord].values[-1]}")

        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_index = 0
        md_track = []
        for i, time_step in enumerate(ds[time_coord].values):
            progress = (i + 1) / total_steps

            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            current_spinner = spinner[spinner_index]
            spinner_index = (spinner_index + 1) % len(spinner)
            sys.stdout.write(
                f"\r{current_spinner} month(s) processed: |{bar}| {progress:.2%}"
            )
            sys.stdout.flush()

            data_slice = ds.sel(**{time_coord: time_step})
            result = find_low_pressure_center(data_slice, time_step)

            if result is not None:
                lat, lon = result
                md_track.append(
                    {
                        "time": time_step,
                        "latitude": lat,
                        "longitude": lon,
                    }
                )
        print()

        if md_track:
            track_df = pd.DataFrame(md_track)
            track_df["month"] = pd.to_datetime(track_df["time"]).dt.strftime("%B")

            individual_tracks = group_detections_into_tracks(track_df)
            valid_tracks = filter_valid_tracks(individual_tracks)

            print(f" > Found {len(individual_tracks)} potential tracks")
            print(f" > After filtering: {len(valid_tracks)} valid tracks")

            if not valid_tracks:
                print(" > No valid monsoon depression tracks \
                found after filtering")
                return None

            # track IDs
            for track_id, track in enumerate(valid_tracks):
                track["track_id"] = track_id + 1
            track_df_with_ids = pd.concat(valid_tracks, ignore_index=True)

            return track_df_with_ids

    except FileNotFoundError:
        print(
            f" > Error: File not found in {DATA_DIR}. \
                Please check the directory and file names."
        )
    except KeyError as e:
        print(f"Error: Missing expected variable or coordinate: {e}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f" > No monsoon depression centers were detected.")
    return None


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
    # plt.show()
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

    ax.set_title(f"Track {track_id} on full dataset extent")
    outpath = os.path.join(output_dir, f"track_{int(track_id)}_full_map.png")
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

    print("  Year      :", years)
    print("  Months    :", months)
    print("  Hours     :", hours)
    print("  Variables :", [var for var in ds.data_vars if var in ["msl", "u10", "v10", "tp", "t2m"]])
    print(f"  Dataset bounds: lat [{ds_lat_min:.1f}, {ds_lat_max:.1f}] N, lon [{ds_lon_min:.1f}, {ds_lon_max:.1f}] E")
    print(f"  Data Time Resolution: {resolution_hours:.1f} hours ({time_diff})")

    print("\n-------------------- Processing ----------------------------")
    full_track_df = process_files(ds)

    if full_track_df is not None and not full_track_df.empty:
        if 'track_id' not in full_track_df.columns:
            print("Error: 'track_id' column not found in full_track_df. Cannot plot/export individual tracks.")
            sys.exit(1)

        if 'track_id' in full_track_df.columns:
            initial_track_count = len(full_track_df['track_id'].unique())
            full_track_df = full_track_df[full_track_df['track_id'].isin(desired_tracks)].copy()
            filtered_track_count = len(full_track_df['track_id'].unique())
            print(f"\nINFO: Filtered tracks for Azimuthal-Hovmöller analysis.")
            print(f"INFO: Retaining {filtered_track_count} tracks ({desired_tracks}) out of {initial_track_count} total.")
        else:
            print("WARNING: 'track_id' column not found. Skipping track filter.")

        plot_tracks(full_track_df)

        pad = 0.5
        try:
            ds_lon_min = ds.longitude.min().item()
            ds_lon_max = ds.longitude.max().item()
            ds_lat_min = ds.latitude.min().item()
            ds_lat_max = ds.latitude.max().item()
            full_extent = [ds_lon_min - pad, ds_lon_max + pad, ds_lat_min - pad, ds_lat_max + pad]
        except Exception:
            full_extent = [60, 100, 0, 35]

        unique_track_ids = sorted(full_track_df['track_id'].unique())
        print(f"\n >> Found {len(unique_track_ids)} MD tracks for individual export.")

        for track_id in unique_track_ids:
            track_dir = os.path.join(output_dir, f"track_{int(track_id)}")
            os.makedirs(track_dir, exist_ok=True)

            track_df = full_track_df[full_track_df['track_id'] == track_id].copy().reset_index(drop=True)
            track_df['is_land'] = track_df.apply(
                lambda row: is_point_land(row['latitude'], row['longitude']), 
                axis=1
            )

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
