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
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from geopy.distance import geodesic

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f" >> Created output directory: {output_dir}")
else:
    print(f" >> {output_dir} directory already present.")
print("------------------------------------------------------------")

DATA_DIR = os.path.join("Data", "single-levels", "ncfiles")
MONTH_DIRS = [
    "August-2024",
    "September-2024",
]

DISTANCE_THRESHOLD_KM = 300       # km
TIME_THRESHOLD_HOURS = 1          # hr
WIND_SPEED_LOWER_BOUND_MS = 7.5   # m/s
WIND_SPEED_UPPER_BOUND_MS = 16.0  # m/s
MSL_LOWER_BOUND_PA = 100000 # 99800  # Pa
PRECIPITATION_THRESHOLD_MM = 0.5  # mm/hr
BUFFER = 5.0                      # deg

month_colors = {
    "August": "blue",
    "September": "purple",
}

month_order = ["August", "September"]

def has_closed_contour(data_array, lat, lon):
    lat_min, lat_max = lat - BUFFER, lat + BUFFER
    lon_min, lon_max = lon - BUFFER, lon + BUFFER

    ####
    # Check bounds
    if (lat_min < data_array.latitude.min() or lat_max > data_array.latitude.max() or
        lon_min < data_array.longitude.min() or lon_max > data_array.longitude.max()):
        return False
    ####

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

    ####
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
            if boundary_pressure > central_pressure + 50: 
                 return True
    ####

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


def check_moisture_criteria(data_slice, lat, lon):
    lat_min, lat_max = lat - BUFFER, lat + BUFFER
    lon_min, lon_max = lon - BUFFER, lon + BUFFER

    try:
        # Check total precipitation (tp)
        if "tp" in data_slice.data_vars:
            tp_data = data_slice["tp"]
            tp_subset = tp_data.sel(
                latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
            )
            max_precipitation = tp_subset.max().item()

            precipitation_check = (max_precipitation * 1000) >= PRECIPITATION_THRESHOLD_MM

            return precipitation_check
        else:
             # print("Warning: Missing 'tp' for convection check. Skipping this check.")
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

        # Apply MD detection criteria
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


def plot_initial_isobars(ax, ds, track_data, color):
    start_time = pd.to_datetime(track_data.iloc[0]['time'])
    start_lat = track_data.iloc[0]['latitude']
    start_lon = track_data.iloc[0]['longitude']

    time_coord = "time" if "time" in ds.coords else "valid_time"
    ds_time = ds.sel({time_coord: start_time}, method="nearest")

    contour_buffer = 7.0
    lat_min, lat_max = start_lat - contour_buffer, start_lat + contour_buffer
    lon_min, lon_max = start_lon - contour_buffer, start_lon + contour_buffer

    try:
        msl_subset = ds_time['msl'].sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )

        if msl_subset.ndim > 2:
            msl_subset = msl_subset.squeeze()
        if msl_subset.ndim != 2:
            raise ValueError(f"MSL subset is {msl_subset.ndim}D, expected 2D.")

    except Exception as e:
        print(f"Could not subset MSL data for contour plot: {e}")
        return

    msl_hpa = msl_subset / 100.0

    min_hpa = msl_hpa.min().item()
    # max_hpa = msl_hpa.max().item()
    start_level = np.floor(min_hpa).item()
    levels = np.arange(start_level, start_level + 5.1, 1)

    # Create contour levels, using 2 hPa intervals if range is wide enough, else 1 hPa
    # if max_hpa - min_hpa > 4:
    #     levels = np.arange(np.floor(min_hpa / 2) * 2, np.ceil(max_hpa / 2) * 2 + 2, 2)
    # else:
    #     levels = np.arange(np.floor(min_hpa), np.ceil(max_hpa) + 1, 1)
    if len(levels) < 2:
        print(f" > Warning: Not enough pressure range for clear isobars around {min_hpa:.1f} hPa.")
        return

    # Plot the contours (isobars)
    contour_plot = ax.contour(
        msl_hpa.longitude,
        msl_hpa.latitude,
        msl_hpa.values,
        levels=levels,
        colors=color,
        linewidths=0.8,
        linestyles='--',
        transform=ccrs.PlateCarree(),
        zorder=4
    )

    ax.clabel(contour_plot, inline=True, fontsize=8, fmt='%1.0f')

    # print(f" > Plotted initial isobars ({min_hpa:.1f}-{max_hpa:.1f} hPa) for track {track_data['track_id'].iloc[0]}")
    print(f" > Plotted initial isobars (levels: {levels[0]:.0f}-{levels[-1]:.0f} hPa) for track {track_data['track_id'].iloc[0]}")


def group_detections_into_tracks(
    detections, time_threshold_hours=TIME_THRESHOLD_HOURS, distance_threshold_km=DISTANCE_THRESHOLD_KM
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

        distance = geodesic((prev_lat, prev_lon), (current_lat, current_lon)).km

        # Check detection be part of the current track
        if (
            time_diff_hours <= time_threshold_hours
            and distance <= distance_threshold_km
        ):
            current_track.append(i)
        else:
            # Finish current track and start a new one
            tracks.append(detections.iloc[current_track].copy())
            current_track = [i]

    # Add the last track
    if current_track:
        tracks.append(detections.iloc[current_track].copy())

    return tracks


def filter_valid_tracks(
        tracks, min_track_length=5, min_track_duration_hours=24, min_total_distance=20
):
    valid_tracks = []

    for track in tracks:
        # Check minimum track length
        if len(track) < min_track_length:
            continue

        # Check minimum track duration (e.g., 24 hours)
        start_time = pd.to_datetime(track["time"].iloc[0])
        end_time = pd.to_datetime(track["time"].iloc[-1])
        duration_hours = (end_time - start_time).total_seconds() / 3600

        if duration_hours < min_track_duration_hours:
            continue

        # Check if track has reasonable movement
        start_lat, start_lon = track.iloc[0]["latitude"], \
            track.iloc[0]["longitude"]
        end_lat, end_lon = track.iloc[-1]["latitude"], \
            track.iloc[-1]["longitude"]
        total_distance = geodesic(
            (start_lat, start_lon), (end_lat, end_lon)
        ).km

        if total_distance < min_total_distance:
            continue

        valid_tracks.append(track)

    return valid_tracks


def process_files(level_label):
    try:
        required_vars = ["msl", "u10", "v10", "tp"]
        if not all(var in ds.data_vars for var in required_vars):
            print(
                f"Error: One or more required variables ({', '.join(required_vars)}) not found in the combined dataset. Exiting..."
            )
            return None

        # Determine time coordinate name
        time_coord = "time" if "time" in ds.coords else "valid_time"
        total_steps = len(ds[time_coord].values)

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
                # f"\r{current_spinner} {month_order[0]}-{month_order[-1]} month(s) at {level_label}: |{bar}| {progress:.2%}"
                f"\r{current_spinner} month(s) at {level_label}: |{bar}| {progress:.2%}"
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
                print(" > No valid monsoon depression tracks found after filtering")
                return None

            # track IDs
            for track_id, track in enumerate(valid_tracks):
                track["track_id"] = track_id
            track_df_with_ids = pd.concat(valid_tracks, ignore_index=True)

            return track_df_with_ids, ds

    except FileNotFoundError:
        print(
            f" > Error: File not found in {DATA_DIR}. Please check the directory and file names."
        )
    except KeyError as e:
        print(f"Error: Missing expected variable or coordinate: {e}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f" > No monsoon depression centers were detected at {level_label}.")
    return None, None


def plot_tracks(track_df, ds_combined, level_label):
    if track_df.empty:
        print(f"  Warning: No data to plot for {level_label}.")
        return

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([70, 100, 5, 30])

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

    for track_id, track_data in track_df.groupby("track_id"):
        start_date = pd.to_datetime(track_data["time"].iloc[0]).strftime(
            "%d/%m"
        )
        end_date = pd.to_datetime(track_data["time"].iloc[-1]).strftime(
            "%d/%m"
        )

        month = track_data["month"].iloc[0]
        color = month_colors.get(month, "black")
        track_data = track_data.sort_values("time")

        start_lon, start_lat = (
            track_data.iloc[0]["longitude"],
            track_data.iloc[0]["latitude"],
        )
        end_lon, end_lat = (
            track_data.iloc[-1]["longitude"],
            track_data.iloc[-1]["latitude"],
        )

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

        # Plot arrows for direction
        # for i in range(len(track_data) - 1):
        #     lon1, lat1 = track_data.iloc[i]["longitude"], track_data.iloc[i]["latitude"]
        #     lon2, lat2 = (
        #         track_data.iloc[i + 1]["longitude"],
        #         track_data.iloc[i + 1]["latitude"],
        #     )
        #
        #     # Draw arrows between points
        #     plt.arrow(
        #         lon1,
        #         lat1,
        #         lon2 - lon1,
        #         lat2 - lat1,
        #         length_includes_head=True,
        #         head_width=0.2,
        #         head_length=0.3,
        #         fc=color,
        #         ec=color,
        #         alpha=0.5,
        #         transform=ccrs.PlateCarree(),
        #         zorder=2,
        #     )

        # Annotate Start/End points
        plt.annotate(
            f"{track_id + 1}, S:{start_date}",
            xy=(start_lon, start_lat),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="normal",
            color="gray",
            bbox=dict(
                boxstyle="round,pad=0.1",
                facecolor="white",
                alpha=0.5,
                edgecolor=color
            ),
            arrowprops=dict(arrowstyle="->", color=color, connectionstyle="arc3,rad=0"),
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

        plt.annotate(
            f"{track_id + 1}, E:{end_date}",
            xy=(end_lon, end_lat),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="normal",
            color="gray",
            bbox=dict(
                boxstyle="round,pad=0.1",
                facecolor="white",
                alpha=0.5,
                edgecolor=color
            ),
            arrowprops=dict(arrowstyle="->", color=color, connectionstyle="arc3,rad=0"),
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

        plot_initial_isobars(ax, ds_combined, track_data, color)

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
        f"Detected MD Tracks ({month_order[0]}-{month_order[-1]}) at {level_label}"
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(handles=legend_handles)

    output_filename = os.path.join(output_dir, f"MD-track-{level_label}.png")
    print("\n---------------------- MD track plots ----------------------")
    plt.savefig(output_filename)
    print(f" > Track plot saved as {output_filename}")
    plt.show()
    plt.close()


def print_track_details(track_df, level_label):
    if track_df.empty:
        print(f"No track details to display for {level_label}.")
        return

    print(f"\n--- Monsoon Depression Track Details at {level_label} ---")
    for track_id, track_data in track_df.groupby("track_id"):
        start_time = pd.to_datetime(track_data["time"].iloc[0])
        end_time = pd.to_datetime(track_data["time"].iloc[-1])
        duration_hours = (end_time - start_time).total_seconds() / 3600
        start_month = track_data["month"].iloc[0]
        # month = track_data['month'].iloc[0]

        print(f" Track ID: {track_id + 1} ({start_month})")
        # print(f"  Month: {month}")
        print(f"  Start: {start_time.strftime('%Y-%m-%d %H:%M')}Z")
        print(f"  End:   {end_time.strftime('%Y-%m-%d %H:%M')}Z")
        print(f"  Duration: {duration_hours:.1f} hours ({len(track_data)} steps)")
        print("  Detections:")
        display_df = track_data[["time", "latitude", "longitude"]].copy()
        display_df["time"] = pd.to_datetime(display_df["time"]).dt.strftime('%Y-%m-%d %H:%M')
        display_df["latitude"] = display_df["latitude"].apply(lambda x: f"{x:.2f}°N")
        display_df["longitude"] = display_df["longitude"].apply(lambda x: f"{x:.2f}°E")
        print(display_df.to_string(index=False, header=True))

def export_track_coordinates(track_df, level_label, output_dir):
    if track_df.empty:
        print(f"  Warning: No track coordinates to export for {level_label}.")
        return

    export_df = track_df[['time', 'latitude', 'longitude', 'track_id']].copy()
    export_df['time'] = pd.to_datetime(export_df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    output_filename = os.path.join(
        output_dir, f"MD-coordinates-{level_label}.csv"
    )
    export_df.to_csv(output_filename, index=False)
    print(f" > Coordinates for {len(track_df)} detections exported to {output_filename}")


def load_dataset_async(month_dirs, data_dir, result_container):
    file_list = []
    for month_dir in month_dirs:
        month_path = os.path.join(data_dir, month_dir)
        if os.path.isdir(month_path):
            file_list.extend([
                os.path.join(month_path, f)
                for f in os.listdir(month_path) if f.endswith(".nc")
            ])

    if not file_list:
        result_container["error"] = "No netCDF files found in the specified directories."
        result_container["success"] = False
        return

    try:
        ds = xr.open_mfdataset(
            file_list,
            combine="by_coords",
            # concat_dim='valid_time',
            data_vars='all',
            coords='all',
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
    SURFACE_LEVEL_LABEL = "Surface_Level"
    print("\n-------------------- Applied Thresholds --------------------")
    print(f"  Level             : {SURFACE_LEVEL_LABEL}")
    print(f"  Distance          : {DISTANCE_THRESHOLD_KM} km")
    print(f"  Time              : {TIME_THRESHOLD_HOURS} hrs")
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
            f"\r > Successfully loaded and combined {num_files} files in {end_processing_time - start_processing_time:.2f} seconds."
        )
    else:
        print(f"\r > Error loading files: {result_container['error']}")
        sys.exit(1)

    print("\n-------------------- Dataset Info --------------------------")
    time_coord = "time" if "time" in ds.coords else "valid_time"

    years = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%Y").unique()
    months_avail = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%b").unique()
    hours_avail = pd.to_datetime(ds[time_coord].values).to_series().dt.strftime("%H:%M").unique()
    ds_lat_min = ds.latitude.min().item()
    ds_lat_max = ds.latitude.max().item()
    ds_lon_min = ds.longitude.min().item()
    ds_lon_max = ds.longitude.max().item()

    print("  Year      :", years)
    print("  Months    :", months_avail)
    print("  Hours     :", hours_avail)
    print("  Variables :", [var for var in ds.data_vars if var in ["msl", "u10", "v10", "tp", "t2m"]])
    print(f"  Dataset bounds: lat [{ds_lat_min:.1f}, {ds_lat_max:.1f}] N, lon [{ds_lon_min:.1f}, {ds_lon_max:.1f}] E")

    print("\n-------------------- Processing ----------------------------")

    full_track_df, ds_combined = process_files(SURFACE_LEVEL_LABEL)

    if full_track_df is not None and ds_combined is not None:
        plot_tracks(full_track_df, ds_combined, SURFACE_LEVEL_LABEL)
        print_track_details(full_track_df, SURFACE_LEVEL_LABEL)
        export_track_coordinates(full_track_df, SURFACE_LEVEL_LABEL, output_dir)
    else:
        print(f" > No tracks found for {SURFACE_LEVEL_LABEL}")

    end_time = time.time()

    print("------------------------------------------------------------")
    print(f"\nSuccessfully executed in {end_time - start_time:.2f} seconds.\n")
