"""
The average translational velocity of a monsoon depression is approx 2-6 m/s, to roughly 7-22 km/h.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from geopy.distance import geodesic
import regionmask
import functools
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom

# Land/Ocean classification
@functools.lru_cache(maxsize=128)
def get_land_mask():
    return regionmask.defined_regions.natural_earth_v5_1_2.countries_50

def is_point_land(lat, lon):
    try:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False
        point = sgeom.Point(lon, lat)
        land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
        reader = shpreader.Reader(land_shp)
        for geometry in reader.geometries():
            if geometry.contains(point):
                return True
        return False
    except:
        return False

def gaussian_filter1d_weighted(data, weights, sigma):
    valid = ~np.isnan(data)
    if np.sum(valid) < 3:
        return data

    # Create weighted data
    weighted_data = data.copy()
    weighted_data[valid] *= weights[valid]

    # Apply Gaussian filter to both data and weights
    filtered_data = gaussian_filter1d(
        np.nan_to_num(weighted_data, nan=0), sigma, mode='nearest'
    )
    filtered_weights = gaussian_filter1d(
        np.nan_to_num(weights * valid.astype(float), nan=0), 
        sigma, mode='nearest'
    )

    # Normalize by filtered weights
    result = np.where(filtered_weights > 0, 
                      filtered_data / filtered_weights, 
                      np.nan)

    # Restore original NaNs
    result[~valid] = np.nan

    return result


def interpolate_small_gaps(speeds, max_gap=3):
    """Interpolate over small gaps in speed data."""
    result = speeds.copy()
    n = len(speeds)

    # Find NaN gaps
    is_nan = np.isnan(speeds)
    if not np.any(is_nan):
        return result

    # Find gap boundaries
    gap_starts = np.where(np.diff(np.concatenate([[False], is_nan, [False]])) == 1)[0]
    gap_ends = np.where(np.diff(np.concatenate([[False], is_nan, [False]])) == -1)[0]
    for start, end in zip(gap_starts, gap_ends):
        gap_length = end - start
        if gap_length <= max_gap and start > 0 and end < n:
            # Linear interpolation for small gaps
            result[start:end] = np.linspace(
                speeds[start-1], speeds[end], gap_length + 2
            )[1:-1]
    return result


def apply_adaptive_smoothing(track_df, sigma=2.0):
    raw_speeds = track_df['raw_speed_kmh'].values
    time_gaps = track_df['time_gap_hours'].values

    # Stage 1: Remove extreme outliers using MAD
    valid_mask = ~np.isnan(raw_speeds)
    if np.sum(valid_mask) < 3:
        track_df['speed_kmh'] = raw_speeds
        track_df['speed_ms'] = raw_speeds / 3.6
        return track_df

    median_speed = np.nanmedian(raw_speeds[valid_mask])
    mad = np.nanmedian(np.abs(raw_speeds[valid_mask] - median_speed))

    # Use 5*MAD as outlier threshold (more conservative than 3*MAD)
    threshold = 5 * mad if mad > 0 else 10  # fallback to 10 km/h
    outlier_mask = np.abs(raw_speeds - median_speed) > threshold

    cleaned_speeds = raw_speeds.copy()
    cleaned_speeds[outlier_mask] = np.nan

    # Stage 2: Time-weighted Gaussian smoothing
    # Adjust sigma based on time gaps
    if len(cleaned_speeds) > 3:
        # Weight by inverse of time gaps (more weight to regular intervals)
        weights = np.where(
            ~np.isnan(time_gaps) & (time_gaps > 0),
            1.0 / time_gaps,
            1.0
        )
        weights = weights / np.nanmean(weights)  # normalize

        # Apply weighted Gaussian filter
        smoothed = gaussian_filter1d_weighted(cleaned_speeds, weights, sigma)
    else:
        smoothed = cleaned_speeds.copy()

    # Stage 3: Spline interpolation for small gaps (≤3 points)
    smoothed = interpolate_small_gaps(smoothed, max_gap=3)

    # Stage 4: Clamp to reasonable MD speed range
    # Based on literature: 2-6 m/s = 7.2-21.6 km/h
    # Allow slightly wider range: 5-25 km/h
    smoothed = np.clip(smoothed, 5, 25)

    track_df['speed_kmh'] = smoothed
    track_df['speed_ms'] = smoothed / 3.6

    return track_df


def compute_track_speeds(track_df):
    print("\n------------------- Speed analysis --------------------")
    track_df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(track_df['time']).values
    lats = track_df['latitude'].values
    lons = track_df['longitude'].values

    n_points = len(track_df)
    speeds_kmh = np.full(n_points, np.nan)
    distances_km = np.full(n_points, np.nan)

    # Calculate speeds between consecutive points
    for i in range(n_points - 1):
        time_gap_hours = (times[i+1] - times[i]).astype('timedelta64[s]').astype(float) / 3600.0

        if 0.5 <= time_gap_hours <= 12:
            dist_km = geodesic((lats[i], lons[i]), (lats[i+1], lons[i+1])).km
            speed_kmh = dist_km / time_gap_hours

            speeds_kmh[i] = speed_kmh
            distances_km[i] = dist_km

    # Forward fill last point
    if n_points > 1 and not np.isnan(speeds_kmh[-2]):
        speeds_kmh[-1] = speeds_kmh[-2]
        distances_km[-1] = distances_km[-2]

    # Smooth speeds using rolling mean
    speeds_series = pd.Series(speeds_kmh)
    speeds_smoothed = speeds_series.rolling(window=3, center=True, min_periods=1).mean().values

    # Clamp to reasonable MD speed range (5-25 km/h or ~1.4-7 m/s)
    speeds_smoothed = np.clip(speeds_smoothed, 5, 25)

    track_df['speed_kmh'] = speeds_smoothed
    track_df['speed_ms'] = speeds_smoothed / 3.6
    track_df['distance_km'] = distances_km

    return track_df


def plot_speed_analysis(track_df, track_id, output_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    times = pd.to_datetime(track_df['time'])

    # Find landfall index
    landfall_index = -1
    if 'is_land' in track_df.columns:
        land_status = track_df['is_land'].values
        landfall_indices = np.where(np.diff(land_status.astype(int)) == 1)[0] + 1
        if len(landfall_indices) > 0:
            landfall_index = landfall_indices[0]
        elif land_status[0] and len(land_status) > 1 and not land_status[1]:
            landfall_index = 0
        elif land_status[0] and all(land_status):
            landfall_index = 0

    # Plot raw speed points
    if 'raw_speed_kmh' in track_df.columns:
        ax.plot(
            times,
            track_df['raw_speed_kmh'],
            'o',
            color='gray',
            alpha=0.4,
            markersize=5,
            label='Raw Speed',
            zorder=1
        )

    # Plot smoothed speed line
    ax.plot(
        times,
        track_df['speed_kmh'],
        '-',
        color='blue',
        linewidth=2.5,
        label='Smoothed Speed',
        zorder=2
    )

    # Plot spline speed if available
    if 'spline_speed_kmh' in track_df.columns:
        ax.plot(
            times,
            track_df['spline_speed_kmh'],
            '--',
            color='red',
            linewidth=1.8,
            label='Spline Fit',
            alpha=0.8,
            zorder=3
        )

    # Add landfall line if found
    if landfall_index != -1 and landfall_index < len(times):
        landfall_time = times[landfall_index]
        landfall_speed = track_df['speed_kmh'].iloc[landfall_index] if landfall_index < len(track_df) else None

        # Vertical line at landfall time
        ax.axvline(
            x=landfall_time,
            color='red',
            linestyle='--',
            linewidth=2.5,
            alpha=0.8,
            label='Landfall',
            zorder=4
        )

        # Marker at landfall point
        if landfall_speed is not None:
            ax.plot(
                landfall_time,
                landfall_speed,
                's',
                color='red',
                markersize=10,
                markerfacecolor='none',
                markeredgewidth=2,
                zorder=5,
                label='Landfall Point'
            )

    ax.axhspan(
        7.2,
        21.6,
        alpha=0.15,
        color='green',
        label='Typical MD Range (2-6 m/s = 7.2-21.6 km/h)',
        zorder=0
    )

    # Set labels and title
    ax.set_xlabel("Time", fontsize=14, fontweight='bold')
    ax.set_ylabel("Translation Speed (km/h)", fontsize=14, fontweight='bold')

    # Add landfall information to title
    if landfall_index != -1:
        landfall_time_str = pd.to_datetime(times[landfall_index]).strftime("%Y-%m-%d %H:%M UTC")
        title = f"Track {track_id}: Translation Speed Analysis\nLandfall at {landfall_time_str}"
    else:
        title = f"Track {track_id}: Translation Speed Analysis (No Landfall Detected)"

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.legend(
        loc='upper right',
        fontsize=10,
        framealpha=0.9,
        edgecolor='black'
    )
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.grid(True, alpha=0.2, linestyle='-', which='major')

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=8)

    # stats_text = (
    #     f"Statistics:\n"
    #     f"• Mean: {track_df['speed_kmh'].mean():.1f} km/h\n"
    #     f"• Median: {track_df['speed_kmh'].median():.1f} km/h\n"
    #     f"• Std Dev: {track_df['speed_kmh'].std():.1f} km/h\n"
    #     f"• Min: {track_df['speed_kmh'].min():.1f} km/h\n"
    #     f"• Max: {track_df['speed_kmh'].max():.1f} km/h"
    # )
    # ax.text(
    #     0.02, 0.98, stats_text,
    #     transform=ax.transAxes,
    #     verticalalignment='top',
    #     fontsize=10,
    #     bbox=dict(
    #         boxstyle='round',
    #         facecolor='white',
    #         alpha=0.9,
    #         edgecolor='black'
    #     ),
    #     linespacing=1.5
    # )
    plt.tight_layout()
    outpath = f"{output_dir}/track_{int(track_id)}_speed_single.png"
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" >> [Saved   ] Single speed analysis plot: {outpath}")
