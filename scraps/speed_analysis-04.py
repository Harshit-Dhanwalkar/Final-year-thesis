"""
The average translational velocity of a monsoon depression is approx 2-6 m/s, to roughly 7-22 km/h.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader

from utils.land_utils import is_point_land

input_dir = "output"  # track CSVs path
if not os.path.exists(input_dir):
    os.makedirs(input_dir)
    print(f" >  [INFO    ] Created directory: {input_dir}")

EARTH_R = 6371.0

def compute_track_speeds(track_df):
    track_df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(track_df['time']).values
    lats = track_df['latitude'].values
    lons = track_df['longitude'].values

    n = len(track_df)
    speeds_kmh = np.full(n, np.nan)
    distances_km = np.full(n, np.nan)

    # Calculate speed between consecutive points
    for i in range(n - 1):
        time_gap = (times[i+1] - times[i]).astype('timedelta64[s]').astype(float) / 3600.0

        # Accept gaps between 0.5 and 12 hours
        if 0.5 <= time_gap <= 12:
            dist = geodesic((lats[i], lons[i]), (lats[i+1], lons[i+1])).km
            if time_gap <= 0:
                speeds_kmh[i] = np.nan
                continue
            if np.isnan(lats[i]) or np.isnan(lons[i]) or np.isnan(lats[i+1]) or np.isnan(lons[i+1]):
                speeds_kmh[i] = np.nan
                distances_km[i] = np.nan
                continue
            speed = dist / time_gap
            speeds_kmh[i] = speed
            distances_km[i] = dist

    # Forward fill last point
    if n > 1 and not np.isnan(speeds_kmh[-2]):
        speeds_kmh[-1] = speeds_kmh[-2]
        distances_km[-1] = distances_km[-2]

    # Smooth with 3-point rolling mean
    speeds_series = pd.Series(speeds_kmh)
    speeds_smoothed = speeds_series.rolling(window=3, center=True, min_periods=1).mean().values

    # track_df['raw_speed_kmh'] = speeds_kmh
    track_df['speed_kmh'] = speeds_smoothed
    track_df['speed_ms'] = speeds_smoothed / 3.6
    track_df['distance_km'] = distances_km
    return track_df

def plot_speed_analysis_kmh(track_df, track_id, output_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    times = pd.to_datetime(track_df['time'])
    # Find landfall index
    landfall_idx = -1
    if 'is_land' in track_df.columns:
        land_status = track_df['is_land'].values
        landfall_indices = np.where(np.diff(land_status.astype(int)) == 1)[0] + 1
        if len(landfall_indices) > 0:
            landfall_idx = landfall_indices[0]

    # # Plot raw speed points
    # if 'raw_speed_kmh' in track_df.columns:
    #     ax.plot(
    #         times,
    #         track_df['raw_speed_kmh'],
    #         'o',
    #         color='gray',
    #         alpha=0.4,
    #         markersize=5,
    #         label='Raw Speed',
    #         zorder=1
    #     )
    # Plot smoothed speed line
    ax.plot(
        times,
        track_df['speed_kmh'],
        '-',
        color='blue',
        linewidth=2.5,
        label='Speed',
        zorder=2
    )
    # Add landfall line if found
    if landfall_idx >= 0 and landfall_idx < len(times):
        landfall_time = times.iloc[landfall_idx]
        landfall_speed = track_df['speed_kmh'].iloc[landfall_idx]
        ax.axvline(
            x=landfall_time,
            color='black',
            linestyle='--',
            linewidth=2.5,
            alpha=0.8,
            label='Landfall',
            zorder=4
        )
        # ax.plot(
        #     landfall_time,
        #     landfall_speed,
        #     's',
        #     color='red',
        #     markersize=10,
        #     markerfacecolor='none',
        #     markeredgewidth=2,
        #     zorder=5
        # )
    ax.set_xlabel("Time", fontsize=14, fontweight='bold')
    ax.set_ylabel("Translation Speed (km/h)", fontsize=14, fontweight='bold')
    if landfall_idx >= 0:
        landfall_time_str = pd.to_datetime(times.iloc[landfall_idx]).strftime("%Y-%m-%d %H:%M UTC")
        title = f"Track {track_id}: Translation Speed (km/h)\nLandfall at {landfall_time_str}"
    else:
        title = f"Track {track_id}: Translation Speed (km/h)"

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    outpath = os.path.join(output_dir, f"track_{int(track_id)}_speed_kmh.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" >> [SAVED   ] Speed plot (km/h)    : {outpath}")


def plot_speed_analysis_ms(track_df, track_id, output_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    times = pd.to_datetime(track_df['time'])
    # Find landfall index
    landfall_idx = -1
    if 'is_land' in track_df.columns:
        land_status = track_df['is_land'].values
        landfall_indices = np.where(np.diff(land_status.astype(int)) == 1)[0] + 1
        if len(landfall_indices) > 0:
            landfall_idx = landfall_indices[0]

    # # Plot raw speed points (convert to m/s)
    # if 'raw_speed_kmh' in track_df.columns:
    #     ax.plot(
    #         times,
    #         track_df['raw_speed_kmh']/3.6,
    #         'o',
    #         color='gray',
    #         alpha=0.4,
    #         markersize=5,
    #         label='Raw Speed',
    #         zorder=1
    #     )

    # Plot smoothed speed line
    ax.plot(
        times,
        track_df['speed_ms'],
        '-',
        color='blue',
        linewidth=2.5,
        label='Speed',
        zorder=2
    )
    # Add landfall line if found
    if landfall_idx >= 0 and landfall_idx < len(times):
        landfall_time = times.iloc[landfall_idx]
        landfall_speed = track_df['speed_ms'].iloc[landfall_idx]
        ax.axvline(
            x=landfall_time,
            color='black',
            linestyle='--',
            linewidth=2.5,
            alpha=0.8,
            label='Landfall',
            zorder=4
        )
        # ax.plot(
        #     landfall_time,
        #     landfall_speed,
        #     's',
        #     color='red',
        #     markersize=10,
        #     markerfacecolor='none',
        #     markeredgewidth=2,
        #     zorder=5
        # )
    ax.set_xlabel("Time", fontsize=14, fontweight='bold')
    ax.set_ylabel("Translation Speed (m/s)", fontsize=14, fontweight='bold')
    if landfall_idx >= 0:
        landfall_time_str = pd.to_datetime(times.iloc[landfall_idx]).strftime("%Y-%m-%d %H:%M UTC")
        title = f"Track {track_id}: Translation Speed (m/s)\nLandfall at {landfall_time_str}"
    else:
        title = f"Track {track_id}: Translation Speed (m/s)"
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    outpath = os.path.join(output_dir, f"track_{int(track_id)}_speed_ms.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" >> [Saved   ] Speed plot (m/s)     : {outpath}")

def main():
    track_files = []
    for file in os.listdir(input_dir):
        if file.startswith('track_') and file.endswith('.csv') and 'speed' not in file and 'detections' not in file:
            track_files.append(os.path.join(input_dir, file))

    if not track_files:
        print(f" >  [ERROR   ] No track CSV files found in {input_dir}")
        return
    print(f" >  [INFO    ] {len(track_files)} track files")

    all_tracks_data = {}
    # Process each track
    for fpath in sorted(track_files):
        try:
            df = pd.read_csv(fpath)
            # Extract track ID from filename
            fname = os.path.basename(fpath)
            track_id = int(fname.split('_')[1].split('.')[0])

            # land/ocean classification
            df['is_land'] = [is_point_land(lat, lon) for lat, lon in zip(df['latitude'], df['longitude'])]

            # Compute speeds
            df = compute_track_speeds(df)
            all_tracks_data[track_id] = {
                'df': df,
                'speed_kmh': df['speed_kmh'].values,
                'speed_ms': df['speed_ms'].values,
            }

            # Plot track speeds (both km/h and m/s)
            plot_speed_analysis_kmh(df, track_id, input_dir)
            plot_speed_analysis_ms(df, track_id, input_dir)

            mean_spd = df['speed_kmh'].mean()
            std_spd = df['speed_kmh'].std()
            print(f"[Track {track_id}] Mean: {mean_spd:.2f}±{std_spd:.2f} km/h ({mean_spd/3.6:.2f}±{std_spd/3.6:.2f} m/s)")
        except Exception as e:
            print(f" >  [ERROR   ] Processing {fpath}: {e}")
            continue
    if not all_tracks_data:
        print(" >  [ERROR   ] No tracks processed successfully")
        return
    print(f" >> [SAVED   ] Speed plots saved to : {input_dir}/")


if __name__ == "__main__":
    main()
