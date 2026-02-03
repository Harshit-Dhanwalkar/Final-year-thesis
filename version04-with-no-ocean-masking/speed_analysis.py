"""
The average translational velocity of a monsoon depression is approx 2-6 m/s, to roughly 7-22 km/h.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from geopy.distance import geodesic
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader

from utils import (
    std_print,
    is_point_land,
    add_land_ocean_shading
)

input_dir = "output"

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
    if 'is_land' in track_df.columns:
        add_land_ocean_shading(ax, times, track_df['is_land'].values, land_color='bisque', ocean_color='azure')
    land_status = track_df['is_land'].values

    # Plot speed line
    ax.plot(
        times,
        track_df['speed_kmh'],
        '-',
        color='blue',
        linewidth=2.5,
        label='Translation Speed',
        zorder=2
    )
    # Shade land regions
    if len(times) > 0:
        land_regions = []
        start_idx = None
        for i in range(len(land_status)):
            if land_status[i] and start_idx is None:
                # Start of land region
                start_idx = i
            elif not land_status[i] and start_idx is not None:
                # End of land region
                if i - 1 > start_idx:  # At least 2 points in land
                    land_regions.append((start_idx, i-1))
                start_idx = None
        # Check if last region extends to end
        if start_idx is not None and start_idx < len(land_status) - 1:
            land_regions.append((start_idx, len(land_status)-1))
        # Shade each land region as a single span (no overlapping)
        for start_idx, end_idx in land_regions:
            if end_idx > start_idx:  # Need at least 2 points
                ax.axvspan(
                    times.iloc[start_idx],
                    times.iloc[end_idx],
                    alpha=0.6,
                    facecolor='bisque',
                    zorder=0,
                    edgecolor='none'
                )
    ax.set_xlabel("Time", fontsize=12, fontweight='bold')
    ax.set_ylabel("Translation Speed (km/h)", fontsize=12, fontweight='bold')

    # Calculate land/ocean stats for title
    land_percentage = np.mean(land_status) * 100 if len(land_status) > 0 else 0
    land_points = np.sum(land_status)
    ocean_points = len(land_status) - land_points
    title = f"Track {track_id}: Translation Speed (km/h)\n"
    title += f"Land Points: {land_points} ({land_percentage:.1f}%), Ocean Points: {ocean_points} ({100-land_percentage:.1f}%)"
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Only add legend if there are labeled artists
    # handles, labels = ax.get_legend_handles_labels()
    # if handles and labels:
    #     ax.legend(loc='upper right', fontsize=8, framealpha=0.9, edgecolor='black')
    legend_elements = [
        ax.get_lines()[0],  # Speed line
        Patch(facecolor='bisque', alpha=0.15, label='Land Phase', edgecolor='none')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9, edgecolor='black')

    ax.grid(True, alpha=0.5, linestyle='--')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    outpath = os.path.join(output_dir, f"track_{int(track_id)}_speed_kmh.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'sa', 'Speed plot (km/h)', outpath)


def plot_speed_analysis_ms(track_df, track_id, output_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    times = pd.to_datetime(track_df['time'])
    # Ensure 'is_land' column exists
    if 'is_land' not in track_df.columns:
        track_df['is_land'] = [is_point_land(lat, lon) for lat, lon in zip(track_df['latitude'], track_df['longitude'])]
    land_status = track_df['is_land'].values

    # Plot speed line
    ax.plot(
        times,
        track_df['speed_ms'],
        '-',
        color='blue',
        linewidth=2.5,
        label='Translation Speed',
        zorder=2
    )
    # Shade land regions
    if len(times) > 0:
        land_regions = []
        start_idx = None
        for i in range(len(land_status)):
            if land_status[i] and start_idx is None:
                # Start of land region
                start_idx = i
            elif not land_status[i] and start_idx is not None:
                # End of land region
                if i - 1 > start_idx:  # At least 2 points in land
                    land_regions.append((start_idx, i-1))
                start_idx = None
        # Check if last region extends to end
        if start_idx is not None and start_idx < len(land_status) - 1:
            land_regions.append((start_idx, len(land_status)-1))
        # Shade each land region as a single span (no overlapping)
        for start_idx, end_idx in land_regions:
            if end_idx > start_idx:  # Need at least 2 points
                ax.axvspan(
                    times.iloc[start_idx],
                    times.iloc[end_idx],
                    alpha=0.6,
                    facecolor='bisque',
                    zorder=0,
                    edgecolor='none'
                )
    ax.set_xlabel("Time", fontsize=12, fontweight='bold')
    ax.set_ylabel("Translation Speed (m/s)", fontsize=12, fontweight='bold')

    # Calculate stats for title
    land_percentage = np.mean(land_status) * 100 if len(land_status) > 0 else 0
    land_points = np.sum(land_status)
    ocean_points = len(land_status) - land_points
    title = f"Track {track_id}: Translation Speed (m/s)\n"
    title += f"Land Points: {land_points} ({land_percentage:.1f}%), Ocean Points: {ocean_points} ({100-land_percentage:.1f}%)"
    ax.set_title(title, fontsize=13, fontweight='bold')

    # # Only add legend if there are labeled artists
    legend_elements = [
        ax.get_lines()[0],  # Speed line
        Patch(facecolor='bisque', alpha=0.15, label='Land Phase', edgecolor='none')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9, edgecolor='black')

    ax.grid(True, alpha=0.5, linestyle='--')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    outpath = os.path.join(output_dir, f"track_{int(track_id)}_speed_ms.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'sa', 'Speed plot (m/s)', outpath)

def main():
    track_files = []
    for file in os.listdir(input_dir):
        if file.startswith('track_') and file.endswith('.csv') and 'speed' not in file and 'detections' not in file:
            track_files.append(os.path.join(input_dir, file))

    if not track_files:
        std_print('ERROR', 'sa', 'No Track CSV files found in', input_dir)
        return
    std_print('INFO', 'sa', f'Track files found for speed analysis: ', len(track_files))

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
            std_print('ERROR', 'sa', f'Processing {fpath}', e)
            continue
    if not all_tracks_data:
        std_print('ERROR', 'sa', f'No tracks processed successfully')
        return
    std_print('SAVED', 'sa', 'Translation speed plot', input_dir, '/')

if __name__ == "__main__":
    main()
