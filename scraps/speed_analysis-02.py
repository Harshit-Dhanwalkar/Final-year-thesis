"""
The average translational velocity of a monsoon depression is approx 2-6 m/s, to roughly 7-22 km/h.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.ndimage import gaussian_filter1d
from geopy.distance import geodesic
import matplotlib.pyplot as plt
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
        # Get land geometries from Natural Earth
        land_shp = shpreader.natural_earth(resolution='110m',
                                          category='physical',
                                          name='land')
        # Check if point is in any land polygon
        reader = shpreader.Reader(land_shp)
        for geometry in reader.geometries():
            if geometry.contains(point):
                return True
        return False

    except Exception as e:
        try:
            land_mask = get_land_mask()
            mask = land_mask.mask([lon], [lat])
            return not np.isnan(mask[0,0])
        except:
            return False

def gaussian_filter1d_weighted(data, weights, sigma):
    """Apply weighted Gaussian filter."""
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
        weights = np.where(~np.isnan(time_gaps) & (time_gaps > 0), 
                          1.0 / time_gaps, 1.0)
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


def compute_track_speeds(track_df, smoothing_sigma=2.0):
    track_df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(track_df['time']).values
    lats = track_df['latitude'].values
    lons = track_df['longitude'].values
    
    n_points = len(track_df)
    speeds_kmh = np.full(n_points, np.nan)
    distances_km = np.full(n_points, np.nan)
    time_gaps_hours = np.full(n_points, np.nan)

    # Calculate raw speeds between consecutive points
    for i in range(n_points - 1):
        time_gap = (times[i+1] - times[i]).astype('timedelta64[s]').astype(float) / 3600.0
        
        if 0.1 <= time_gap <= 12:  # Accept wider range initially
            dist_km = geodesic(
                (lats[i], lons[i]),
                (lats[i+1], lons[i+1])
            ).km
            
            speed = dist_km / time_gap
            
            # Store values
            distances_km[i] = dist_km
            time_gaps_hours[i] = time_gap
            speeds_kmh[i] = speed
    
    # Forward-fill last point
    if n_points > 1 and not np.isnan(speeds_kmh[-2]):
        speeds_kmh[-1] = speeds_kmh[-2]
        time_gaps_hours[-1] = time_gaps_hours[-2]
    
    track_df['raw_speed_kmh'] = speeds_kmh
    track_df['distance_km'] = distances_km
    track_df['time_gap_hours'] = time_gaps_hours
    
    # Apply multi-stage smoothing
    track_df = apply_adaptive_smoothing(track_df, smoothing_sigma)
    
    return track_df


def compute_spline_speeds(track_df, smoothing_factor=None):
    times = pd.to_datetime(track_df['time'])
    time_seconds = (times - times.iloc[0]).dt.total_seconds().values
    
    lats = track_df['latitude'].values
    lons = track_df['longitude'].values
    
    # Fit splines to lat/lon
    try:
        if smoothing_factor is None:
            # Adaptive smoothing based on number of points
            smoothing_factor = len(track_df) * 0.01
        
        lat_spline = UnivariateSpline(time_seconds, lats, s=smoothing_factor, k=3)
        lon_spline = UnivariateSpline(time_seconds, lons, s=smoothing_factor, k=3)
        
        # Compute derivatives (velocities)
        dlat_dt = lat_spline.derivative()(time_seconds)  # deg/s
        dlon_dt = lon_spline.derivative()(time_seconds)  # deg/s
        
        # Convert to speed in km/h
        # Approximate: 1 deg lat ≈ 111 km
        mean_lat = np.mean(lats)
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(mean_lat))
        
        speeds_kmh = np.sqrt(
            (dlat_dt * km_per_deg_lat)**2 +
            (dlon_dt * km_per_deg_lon)**2
        ) * 3600  # Convert from km/s to km/h
        
        # Clamp to reasonable range
        speeds_kmh = np.clip(speeds_kmh, 2, 30)
        
        track_df['spline_speed_kmh'] = speeds_kmh
        track_df['spline_speed_ms'] = speeds_kmh / 3.6
        
    except Exception as e:
        print(f"Warning: Spline fitting failed: {e}")
        track_df['spline_speed_kmh'] = track_df['speed_kmh']
        track_df['spline_speed_ms'] = track_df['speed_ms']
    
    return track_df


def plot_speed_comparison(track_df, track_id, output_dir):
    """
    Enhanced visualization comparing different smoothing methods.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    times = pd.to_datetime(track_df['time'])

    # Plot 1: Raw vs Smoothed speeds
    ax = axes[0]
    if 'raw_speed_kmh' in track_df.columns:
        ax.plot(times, track_df['raw_speed_kmh'], 'o',
               color='gray', alpha=0.3, markersize=3, label='Raw')
    
    ax.plot(times, track_df['speed_kmh'], '-',
           color='blue', linewidth=2, label='Smoothed')
    
    if 'spline_speed_kmh' in track_df.columns:
        ax.plot(times, track_df['spline_speed_kmh'], '--',
               color='red', linewidth=1.5, label='Spline')
    
    # Add typical MD speed range
    ax.axhspan(7.2, 21.6, alpha=0.15, color='green',
              label='Typical MD Range (2-6 m/s)')
    
    ax.set_ylabel("Speed (km/h)", fontsize=11)
    ax.set_title(f"Track {track_id}: Translation Speed Comparison",
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Time gaps
    ax = axes[1]
    if 'time_gap_hours' in track_df.columns:
        ax.bar(times[:-1], track_df['time_gap_hours'][:-1],
              width=pd.Timedelta(hours=0.5),
              color='orange', alpha=0.6, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=1, color='green', linestyle='--', 
              linewidth=1, label='1 hour (ERA5 resolution)')
    ax.set_ylabel("Time Gap (hours)", fontsize=11)
    ax.set_title("Time Intervals Between Detections", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: By stage
    ax = axes[2]
    if 'stage' in track_df.columns:
        stage_colors = {'ocean': 'blue', 'landfall': 'orange', 'inland': 'red'}
        
        for stage, color in stage_colors.items():
            subset = track_df[track_df['stage'] == stage]
            if not subset.empty:
                ax.plot(pd.to_datetime(subset['time']), 
                       subset['speed_kmh'], 'o-', 
                       color=color, linewidth=2, markersize=4, 
                       label=stage.capitalize(), alpha=0.8)
    else:
        ax.plot(times, track_df['speed_kmh'], 'o-', 
               color='blue', linewidth=2, markersize=4)
    
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Speed (km/h)", fontsize=11)
    ax.set_title("Speed by Track Stage", fontsize=12)
    if 'stage' in track_df.columns:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = (
        f"Mean: {track_df['speed_kmh'].mean():.1f} km/h\n"
        f"Median: {track_df['speed_kmh'].median():.1f} km/h\n"
        f"Std Dev: {track_df['speed_kmh'].std():.1f} km/h\n"
        f"Range: {track_df['speed_kmh'].min():.1f}-{track_df['speed_kmh'].max():.1f} km/h\n"
        f"Expected: 7.2-21.6 km/h (MD typical)"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    outpath = f"{output_dir}/track_{int(track_id)}_speed_analysis.png"
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" >> [Saved   ] Speed analysis: {outpath}")
