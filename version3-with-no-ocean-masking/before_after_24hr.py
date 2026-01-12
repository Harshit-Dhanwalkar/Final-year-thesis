import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils import (
    std_print,
    find_landfall_time,
    setup_map_extent
)

def plot_md_24h_before_after_landfall(track_df, track_id, output_dir='output'):
    """
    Plot MD track points:
    - 24 points (24 hours) BEFORE landfall
    - 24 points (24 hours) AFTER landfall
    If no landfall: show nothing
    """
    track_id = int(track_id)
    track = track_df[track_df['track_id'] == track_id].sort_values('time').copy()

    if track.empty:
        std_print('ERROR', 'ba', f'Track {track_id} not found')
        return False

    # Find landfall
    landfall_time = None
 
    if 'is_land' in track.columns:
        landfall_rows = track[track['is_land'] == True]
        if not landfall_rows.empty:
            landfall_time = pd.to_datetime(landfall_rows.iloc[0]['time'])

    landfall_time = find_landfall_time(track)
    # If no landfall, return
    if landfall_time is None:
        std_print('WARN', 'ba', f'Track {track_id}: No landfall detected. Skipping.')
        return False

    # Convert time column to datetime
    track['time_dt'] = pd.to_datetime(track['time'])
 
    # Define 24h windows
    before_24h_start = landfall_time - pd.Timedelta(hours=24)
    after_24h_end = landfall_time + pd.Timedelta(hours=24)

    # Extract track segments
    track_before = track[(track['time_dt'] >= before_24h_start) & (track['time_dt'] < landfall_time)].copy()
    track_after = track[(track['time_dt'] > landfall_time) & (track['time_dt'] <= after_24h_end)].copy()

    # Check if we have data in both windows
    if len(track_before) == 0 or len(track_after) == 0:
        std_print('WARN', 'ba', f'Track {track_id}: Incomplete data in 24h window.', f"\n        Before: {len(track_before)}, After: {len(track_after)}")
        return False

    std_print('INFO', 'ba', f'Track {track_id} 24h_before_after_landfall:')
    print(f"                Landfall time:                    {landfall_time}")
    print(f"                Points 24h before:                {len(track_before)}")
    print(f"                Points 24h after:                 {len(track_after)}")

    # Get spatial extent
    all_lats = np.concatenate([track_before['latitude'].values, track_after['latitude'].values])
    all_lons = np.concatenate([track_before['longitude'].values, track_after['longitude'].values])

    centroid_lat = np.mean(all_lats)
    centroid_lon = np.mean(all_lons)

    lat_range = max(5.0, (all_lats.max() - all_lats.min())) * 1.5
    lon_range = max(5.0, (all_lons.max() - all_lons.min())) * 1.5

    min_lat = centroid_lat - lat_range / 2
    max_lat = centroid_lat + lat_range / 2
    min_lon = centroid_lon - lon_range / 2
    max_lon = centroid_lon + lon_range / 2

    # Create figure
    fig, ax = plt.subplots(
        figsize=(12, 10),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # Plot base map
    ax.coastlines(resolution='50m', linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.1)

    # Plot BEFORE track (blue line + points)
    ax.plot(
        track_before['longitude'],
        track_before['latitude'],
        'b-',
        linewidth=2.5,
        alpha=0.7,
        zorder=5,
        transform=ccrs.PlateCarree()
    )
    ax.scatter(
        track_before['longitude'],
        track_before['latitude'],
        s=80,
        c='blue',
        edgecolors='darkblue',
        linewidths=1,
        label=f'24h BEFORE landfall ({len(track_before)} points)',
        zorder=6,
        alpha=0.8,
        transform=ccrs.PlateCarree()
    )

    # Plot AFTER track (red line + points)
    ax.plot(
        track_after['longitude'],
        track_after['latitude'],
        'r-',
        linewidth=2.5,
        alpha=0.7,
        zorder=5,
        transform=ccrs.PlateCarree()
    )
    ax.scatter(
        track_after['longitude'],
        track_after['latitude'],
        s=80,
        c='red',
        edgecolors='darkred',
        linewidths=1,
        label=f'24h AFTER landfall ({len(track_after)} points)',
        zorder=6,
        alpha=0.8,
        transform=ccrs.PlateCarree()
    )

    # Mark landfall point
    ax.scatter(
        track_before['longitude'].iloc[-1],
        track_before['latitude'].iloc[-1],
        s=400,
        marker='o',
        color='yellow',
        edgecolors='darkorange',
        linewidths=2,
        label='LANDFALL',
        zorder=10,
        transform=ccrs.PlateCarree()
    )

    ax.coastlines(resolution='50m', linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.1)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(
        f'Track {track_id}: MD Position 24h Before and After Landfall',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f'track_{track_id}_24h_before_after.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ba', '24h before/after plot', outpath)
    return True


def plot_all_tracks_24h(track_df, output_dir='output'):
    """Create plots for all tracks with landfall"""
    unique_tracks = sorted(track_df['track_id'].unique())
    success_count = 0
    for track_id in unique_tracks:
        try:
            success = plot_md_24h_before_after_landfall(track_df, track_id, output_dir)
            if success:
                success_count += 1
        except Exception as e:
            std_print('ERROR', 'ba', f'Failed to plot track {track_id}:', e)
            continue

    std_print('INFO', 'an', 'Generated plots', f'{success_count}/{len(unique_tracks)}')
    return success_count
