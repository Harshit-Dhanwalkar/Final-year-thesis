import os
import sys
import threading
import time
from datetime import datetime

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

# --- USER-DEFINED CONSTANTS (Adjusted for better tracking) ---
DATA_DIR = os.path.join("Data", "single-levels", "ncfiles")
MONTH_DIRS = [
    "August-2024",
    "September-2024",
]

# --- IMPROVED TRACKING THRESHOLDS (Based on our discussion) ---
DISTANCE_THRESHOLD_KM = 300  # km  # Increased for better track continuity
TIME_THRESHOLD_HOURS = 1     # hr  # Increased to bridge detection gaps
WIND_SPEED_LOWER_BOUND_MS = 7.5  # m/s
WIND_SPEED_UPPER_BOUND_MS = 16.0  # m/s
MSL_LOWER_BOUND_PA = 100000      # 1000 hPa
PRECIPITATION_THRESHOLD_MM = 0.5 # mm/hr # Decreased for inland tracking
BUFFER = 5.0 # deg (Area around the minimum pressure point to check for closed contour)
SURFACE_LEVEL_LABEL = "Surface" # Label for the single-level data

# --- COLOR MAPPING ---
month_colors = {
    "August": "blue",
    "September": "purple",
}

month_order = ["August", "September"]

# --- CORE FUNCTIONS ---

def has_closed_contour(data_array, lat, lon):
    lat_min, lat_max = lat - BUFFER, lat + BUFFER
    lon_min, lon_max = lon - BUFFER, lon + BUFFER

    # Check bounds
    if (lat_min < data_array.latitude.min() or lat_max > data_array.latitude.max() or
        lon_min < data_array.longitude.min() or lon_max > data_array.longitude.max()):
        return False

    # Extract subset of MSL data
    subset = data_array.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )

    if subset.size == 0:
        return False

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
            # A difference of 50 Pa (0.5 hPa) is a decent indicator of a closed contour.
            if boundary_pressure > central_pressure + 50: 
                 return True

    return False

def find_md_centers(ds_month, level_label):
    """
    Identifies potential Monsoon Depression centers (min MSL, strong wind, rain)
    and returns a DataFrame of detections.
    """
    detection_points = []
    
    # Data variables check
    if level_label == SURFACE_LEVEL_LABEL:
        msl_var = 'msl'
        u_var = 'u10'
        v_var = 'v10'
        tp_var = 'tp'
    else:
        print(f"Error: Variables not defined for level {level_label}")
        return pd.DataFrame()

    time_coord = "time" if "time" in ds_month.coords else "valid_time"
    
    # Calculate wind speed
    wind_speed = np.sqrt(ds_month[u_var]**2 + ds_month[v_var]**2)
    
    for time_step in ds_month[time_coord]:
        msl_data = ds_month[msl_var].sel(**{time_coord: time_step})
        wind_data = wind_speed.sel(**{time_coord: time_step})
        tp_data = ds_month[tp_var].sel(**{time_coord: time_step})
        
        # Apply initial pressure and wind filters
        msl_filtered = msl_data.where((msl_data < MSL_LOWER_BOUND_PA) & 
                                      (wind_data > WIND_SPEED_LOWER_BOUND_MS) &
                                      (wind_data < WIND_SPEED_UPPER_BOUND_MS), drop=True)
        
        if msl_filtered.size == 0:
            continue
            
        # Find local minimum pressure point
        min_msl = msl_filtered.min().item()
        
        # Find coordinates of the minimum pressure point
        min_coords = msl_filtered.where(msl_filtered == min_msl, drop=True)
        
        # Use the first minimum point found
        lat = min_coords.latitude.values[0]
        lon = min_coords.longitude.values[0]

        # --- APPLY FINAL CRITERIA ---
        
        # 1. Closed Contour Check
        if not has_closed_contour(msl_data, lat, lon):
            continue
            
        # 2. Rainfall Check
        nearest_tp = tp_data.sel(latitude=lat, longitude=lon, method="nearest")
        
        # Convert total precipitation (m) to mm/hr
        # Assuming tp is accumulated over the time step (e.g., 1 hour for hourly data)
        tp_mm_hr = nearest_tp.item() * 1000 # m -> mm
        
        if tp_mm_hr < PRECIPITATION_THRESHOLD_MM:
            continue
            
        # All checks passed
        detection_points.append({
            'time': time_step.item(),
            'latitude': lat,
            'longitude': lon,
            'msl': min_msl,
            'wind_speed': wind_data.sel(latitude=lat, longitude=lon, method="nearest").item(),
            'precipitation': tp_mm_hr,
        })
        
    return pd.DataFrame(detection_points)

def group_detections_into_tracks(df):
    """
    Groups individual detections into continuous tracks based on distance and time thresholds.
    """
    if df.empty:
        return []

    df = df.sort_values(by='time').reset_index(drop=True)
    tracks = []
    
    # Convert time to datetime objects for calculation
    df['dt_time'] = pd.to_datetime(df['time'])

    for i in range(len(df)):
        current_point = df.iloc[i]
        
        # Try to extend an existing track
        extended = False
        for track in tracks:
            last_point = track.iloc[-1]
            
            # Check time gap (must be less than TIME_THRESHOLD_HOURS)
            time_diff = (current_point['dt_time'] - last_point['dt_time']).total_seconds() / 3600
            
            if 0 < time_diff <= TIME_THRESHOLD_HOURS:
                # Check distance (must be less than DISTANCE_THRESHOLD_KM)
                distance = geodesic(
                    (last_point['latitude'], last_point['longitude']),
                    (current_point['latitude'], current_point['longitude'])
                ).km
                
                if distance <= DISTANCE_THRESHOLD_KM:
                    # Point belongs to this track
                    track = pd.concat([track, pd.DataFrame([current_point])], ignore_index=True)
                    tracks[-1] = track # Update the track in the list
                    extended = True
                    break
        
        if not extended:
            # Start a new track
            tracks.append(pd.DataFrame([current_point]))

    # Filter out tracks that are too short (e.g., less than 12 hours)
    min_duration_hours = 12 
    final_tracks = []
    for track in tracks:
        start_time = track['dt_time'].min()
        end_time = track['dt_time'].max()
        duration = (end_time - start_time).total_seconds() / 3600
        
        if duration >= min_duration_hours:
            track['track_id'] = f"MD_{start_time.strftime('%m%d%H')}"
            final_tracks.append(track.drop(columns=['dt_time']))
            
    return final_tracks

def process_files(level_label):
    """
    Loads and merges instantaneous and accumulated data files 
    before finding centers. Uses xr.combine_nested for robust concatenation 
    of monthly files.
    """
    all_detections = pd.DataFrame()
    ds_list = [] # Store all merged monthly datasets

    for month_dir in MONTH_DIRS:
        full_path = os.path.join(DATA_DIR, month_dir)
        
        # --- File Path Construction ---
        instant_file = os.path.join(full_path, "data_stream-oper_stepType-instant.nc")
        accum_file = os.path.join(full_path, "data_stream-oper_stepType-accum.nc")
        
        print(f"\n > Attempting to load and merge data for {month_dir}...")
        
        try:
            # 1. Load Instantaneous Data (msl, u10, v10)
            if not os.path.exists(instant_file):
                print(f"   Warning: File not found: {instant_file}. Skipping month.")
                continue
            ds_instant = xr.open_dataset(instant_file)
            
            # 2. Load Accumulated Data (tp)
            if not os.path.exists(accum_file):
                print(f"   Warning: File not found: {accum_file}. Skipping month.")
                continue
            ds_accum = xr.open_dataset(accum_file)
            
            # 3. Merge variables within the month (msl, u10, v10, tp)
            ds_merged = xr.merge([
                ds_instant[['msl', 'u10', 'v10']],
                ds_accum[['tp']]
            ], compat='override')
            
            ds_list.append(ds_merged)
            
            # 4. Find MD Centers using the merged dataset (before final concatenation)
            detections = find_md_centers(ds_merged, level_label)
            if not detections.empty:
                detections['month'] = month_dir.split('-')[0]
                all_detections = pd.concat([all_detections, detections], ignore_index=True)
            else:
                print(f"   Info: No MD centers found in merged data for {month_dir}.")

        except Exception as e:
            print(f"   Error reading or merging files for {month_dir}: {e}")

    if all_detections.empty or not ds_list:
        return None, None

    time_coord = "time" if "time" in ds_list[0].dims else "valid_time"
    ds_combined = xr.combine_nested(ds_list, concat_dim=time_coord, combine_attrs='drop')
    # Group tracks across all months
    all_tracks = group_detections_into_tracks(all_detections)
    if not all_tracks:
        return None, None

    full_track_df = pd.concat(all_tracks, ignore_index=True)
    return full_track_df, ds_combined


def plot_initial_isobars(ax, ds, track_data, color):
    start_time = pd.to_datetime(track_data.iloc[0]['time'])
    start_lat = track_data.iloc[0]['latitude']
    start_lon = track_data.iloc[0]['longitude']

    time_coord = "time" if "time" in ds.coords else "valid_time"

    # Selects the data array for the single nearest time, ensuring a 2D array
    ds_time = ds.sel({time_coord: start_time}, method="nearest")
    
    # Define a smaller area for the contour plot (e.g., 14x14 degrees around the start point)
    contour_buffer = 7.0 
    lat_min, lat_max = start_lat - contour_buffer, start_lat + contour_buffer
    lon_min, lon_max = start_lon - contour_buffer, start_lon + contour_buffer
    
    # Select MSL pressure data for the subset region
    try:
        # Note: Xarray slice for latitude often goes from max to min
        msl_subset = ds_time['msl'].sel(
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max)
        )
        
        # Ensures the data array is 2D by dropping any remaining singleton dimensions
        if msl_subset.ndim > 2:
             msl_subset = msl_subset.squeeze()
        
        # FINAL CHECK: Ensure the data array is 2D
        if msl_subset.ndim != 2:
            raise ValueError(f"MSL subset is {msl_subset.ndim}D, expected 2D.")
            
    except Exception as e:
        print(f"Could not subset MSL data for contour plot: {e}")
        return

    # Convert Pascals (Pa) to Hectopascals (hPa or mb) and round to integer
    msl_hpa = msl_subset / 100.0 
    
    # Determine the contour levels (typically 2 hPa intervals)
    min_hpa = msl_hpa.min().item()
    max_hpa = msl_hpa.max().item()
    
    # Create contour levels, using 2 hPa intervals if range is wide enough, else 1 hPa
    if max_hpa - min_hpa > 4:
        levels = np.arange(np.floor(min_hpa / 2) * 2, np.ceil(max_hpa / 2) * 2 + 2, 2)
    else:
        levels = np.arange(np.floor(min_hpa), np.ceil(max_hpa) + 1, 1)

    # Plot the contours (isobars)
    contour_plot = ax.contour(
        msl_hpa.longitude,
        msl_hpa.latitude,
        msl_hpa.values,
        levels=levels,
        colors=color,
        linewidths=0.7,
        linestyles='--',
        transform=ccrs.PlateCarree(),
        zorder=4
    )
    
    # Label the contours
    ax.clabel(contour_plot, inline=True, fontsize=8, fmt='%1.0f')
    
    print(f" > Plotted initial isobars ({min_hpa:.1f}-{max_hpa:.1f} hPa) for track {track_data['track_id'].iloc[0]}")


def plot_tracks(full_track_df, ds_combined, level_label):
    """Plots all detected tracks, including the initial MSL contours."""
    
    # Create the plot figure and GeoAxes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) 
    
    # Set map extent (30N 70E to 5N 100E)
    ax.set_extent([70, 100, 5, 30], crs=ccrs.PlateCarree())

    # Add features
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=6)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.LAKES, facecolor='lightblue')
    ax.add_feature(cfeature.RIVERS)
    
    # Gridlines and Labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    # Loop through each unique track
    unique_tracks = full_track_df['track_id'].unique()
    
    for track_id in unique_tracks:
        track_data = full_track_df[full_track_df['track_id'] == track_id].copy()
        month = track_data['month'].iloc[0]
        color = month_colors.get(month, 'black')
        
        # Plot the track line
        ax.plot(
            track_data["longitude"],
            track_data["latitude"],
            "-",
            color=color,
            alpha=0.8,
            linewidth=2,
            transform=ccrs.PlateCarree(),
            zorder=2
        )
        
        # Plot the start point (formation)
        start_time_str = pd.to_datetime(track_data.iloc[0]['time']).strftime('%d/%HZ')
        ax.plot(
            track_data.iloc[0]["longitude"],
            track_data.iloc[0]["latitude"],
            'o',
            color=color,
            markersize=5,
            markeredgecolor='black',
            label=f'Start: {start_time_str}',
            transform=ccrs.PlateCarree(),
            zorder=5
        )

        # Plot the end point (dissipation)
        end_time_str = pd.to_datetime(track_data.iloc[-1]['time']).strftime('%d/%HZ')
        ax.plot(
            track_data.iloc[-1]["longitude"],
            track_data.iloc[-1]["latitude"],
            's',
            color=color,
            markersize=5,
            markeredgecolor='black',
            label=f'End: {end_time_str}',
            transform=ccrs.PlateCarree(),
            zorder=5
        )

        # Plot initial isobars for the first detection point
        plot_initial_isobars(ax, ds_combined, track_data, color)
        
    ax.set_title(f'Monsoon Depression Tracks ({level_label} Level) with Initial Isobars')
    
    # Create a simple legend for the months/colors
    legend_handles = []
    for month in month_order:
        # Check if the month has any tracks found
        if month in full_track_df['month'].unique():
            legend_handles.append(plt.Line2D([0], [0], color=month_colors[month], linewidth=2, label=f'{month} Track'))
    
    ax.legend(handles=legend_handles, loc='lower left', title="Tracks")
    
    # Save the figure
    file_name = f"MD-track-Initial_Isobars-{level_label}.png"
    plt.savefig(os.path.join(output_dir, file_name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n >> Saved track plot with initial isobars: {file_name}")

def print_track_details(full_track_df, level_label):
    """Prints a summary of all detected tracks."""
    print(f"\n-------------------- Track Summary ({level_label}) ----------------------")
    unique_tracks = full_track_df['track_id'].unique()
    
    for track_id in unique_tracks:
        track_data = full_track_df[full_track_df['track_id'] == track_id]
        
        start_time = pd.to_datetime(track_data['time'].min())
        end_time = pd.to_datetime(track_data['time'].max())
        duration = (end_time - start_time).total_seconds() / 3600
        
        start_loc = f"({track_data.iloc[0]['latitude']:.1f}N, {track_data.iloc[0]['longitude']:.1f}E)"
        end_loc = f"({track_data.iloc[-1]['latitude']:.1f}N, {track_data.iloc[-1]['longitude']:.1f}E)"
        
        month = track_data['month'].iloc[0]
        
        print(f"Track ID: {track_id}")
        print(f"  Month: {month}")
        print(f"  Start: {start_time.strftime('%Y-%m-%d %H:%M')} at {start_loc}")
        print(f"  End:   {end_time.strftime('%Y-%m-%d %H:%M')} at {end_loc}")
        print(f"  Duration: {duration:.1f} hours ({len(track_data)} points)")
        print("-" * 20)

def export_track_coordinates(full_track_df, level_label, output_dir):
    """Exports all track coordinates to a single CSV file."""
    file_name = f"MD-tracks-coordinates-{level_label}.csv"
    full_track_df.to_csv(os.path.join(output_dir, file_name), index=False)
    print(f"\n >> Exported track coordinates: {file_name}")


# --- MAIN EXECUTION ---
def main():
    """Main function to run the MD detection and tracking process."""
    print("-------------------- Initialization ------------------------")
    
    # Print loaded configuration
    print(f"Configuration:")
    print(f"  Data Dir  : {DATA_DIR}")
    print(f"  Months    : {MONTH_DIRS}")
    print(f"  Thresholds:")
    print(f"    Time Gap : {TIME_THRESHOLD_HOURS} hr")
    print(f"    Distance : {DISTANCE_THRESHOLD_KM} km")
    print(f"    Min Wind : {WIND_SPEED_LOWER_BOUND_MS} m/s")
    print(f"    Min MSL  : {MSL_LOWER_BOUND_PA / 100} hPa")
    print(f"    Min Prec : {PRECIPITATION_THRESHOLD_MM} mm/hr")

    print("\n-------------------- Processing ----------------------------")

    # Process the data using the surface level label
    full_track_df, ds_combined = process_files(SURFACE_LEVEL_LABEL)

    if full_track_df is not None and ds_combined is not None:
        plot_tracks(full_track_df, ds_combined, SURFACE_LEVEL_LABEL) 
        print_track_details(full_track_df, SURFACE_LEVEL_LABEL)
        export_track_coordinates(full_track_df, SURFACE_LEVEL_LABEL, output_dir)
    else:
        print(f" > No valid tracks or data found for {SURFACE_LEVEL_LABEL} level")

if __name__ == "__main__":
    main()
