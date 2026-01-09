"""
The average translational velocity of a monsoon depression is approx 2-6 m/s, to roughly 7-22 km/h.
"""

import warnings
import sys
import os
import threading
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regionmask
from geopy.distance import geodesic
from scipy.stats import mannwhitneyu
import xarray as xr
import dask
import functools
from dask.distributed import Client, LocalCluster
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numba

warnings.filterwarnings(
    "ignore",
    message="No gridpoint belongs to any region.*",
    category=UserWarning
)

output_dir = "output"
DATA_DIR = os.path.join("Data", "single-levels", "ncfiles")
MONTH_DIRS = [
    "July-2024",
    "August-2024",
    "September-2024",
]

# Constants
M_PER_DEG_LAT = 111320.0
EARTH_R = 6371.0 # km

MAX_GAP_HOURS = 12

VARIABLE_UNITS = {
    'u10': 'm/s',
    'v10': 'm/s',
    'wind10': 'm/s',  # wind10 = sqrt(u10^2 + v10^2)
    'speed_kmh': 'km/h',
    'speed_ms': 'm/s',
    'msl': 'hPa',
    'tp': 'mm/hr',
    't2m': '°C',
    'sp': 'hPa',
    'avg_slhtf': 'W/m²',
    'avg_ishf': 'W/m²'
}

VARIABLE_NAMES = {
    'u10': 'Zonal Wind (10m)',
    'v10': 'Meridional Wind (10m)',
    'wind10': 'Wind Speed (10m)',
    'speed_kmh': 'Translation Speed',
    'msl': 'Mean Sea Level Pressure',
    'tp': 'Total Precipitation',
    't2m': '2m Temperature',
    'sp': 'Surface Pressure',
    'avg_slhtf': 'Surface Latent Heat Flux',
    'avg_ishf': 'Surface Sensible Heat Flux'
}

# desired_tracks = [1, 3, 4, 5]


def load_dataset_blocking(month_dirs, data_dir, use_dask=True):
    file_list = []
    for month_dir in month_dirs:
        month_path = os.path.join(data_dir, month_dir)
        if os.path.isdir(month_path):
            file_list.extend([
                os.path.join(month_path, f)
                for f in os.listdir(month_path) if f.endswith(".nc")
            ])
    if not file_list:
        raise FileNotFoundError("No netCDF files found for loading wind data.")

    print("Loading ERA5 dataset for wind field...")

    if use_dask:
        # Start local cluster
        cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='4GB')
        client = Client(cluster)
        print(f"Dask dashboard: {client.dashboard_link}")

    ds = xr.open_mfdataset(
        file_list,
        combine="by_coords",
        parallel=True,  # parallel reading
        chunks={'time': 24, 'latitude': 100, 'longitude': 100},
        data_vars='all',
        coords='all',  # type: ignore[arg-type]
        compat='override'
    ).compute()
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    ds = ds.sortby('time')
    return ds

# ---------- wind sampling ----------
def sample_wind(ds, lat, lon, time):
    try:
        u = float(ds['u10'].interp(latitude=float(lat), longitude=float(lon), time=np.datetime64(time)).values)
        v = float(ds['v10'].interp(latitude=float(lat), longitude=float(lon), time=np.datetime64(time)).values)
        return u, v
    except Exception:
        # fallback to nearest
        try:
            u = float(ds['u10'].sel(latitude=lat, longitude=lon, time=time, method='nearest').values)
            v = float(ds['v10'].sel(latitude=lat, longitude=lon, time=time, method='nearest').values)
            return u, v
        except Exception:
            return np.nan, np.nan

def sample_wind_batch(ds, points):
    # points: list of tuples (lat, lon, time)
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    times = [p[2] for p in points]

    try:
        # vectorized interpolation if possible
        u_values = ds['u10'].interp(
            latitude=xr.DataArray(lats, dims='points'),
            longitude=xr.DataArray(lons, dims='points'),
            time=xr.DataArray(times, dims='points'),
            method='linear'
        ).values
        v_values = ds['v10'].interp(
            latitude=xr.DataArray(lats, dims='points'),
            longitude=xr.DataArray(lons, dims='points'),
            time=xr.DataArray(times, dims='points'),
            method='linear'
        ).values
        return list(zip(u_values, v_values))
    except Exception:
        # Fall back to individual sampling
        results = []
        for lat, lon, time in points:
            try:
                u = float(ds['u10'].sel(latitude=lat, longitude=lon, time=time, method='nearest').values)
                v = float(ds['v10'].sel(latitude=lat, longitude=lon, time=time, method='nearest').values)
                results.append((u, v))
            except Exception:
                results.append((np.nan, np.nan))
        return results

# ---------- Degree conversions ----------
def meters_to_deg_lat(m):
    return m / M_PER_DEG_LAT

def meters_to_deg_lon(m, lat_deg):
    # 1 deg lon in meters depends on latitude
    return m / (M_PER_DEG_LAT * np.cos(np.deg2rad(lat_deg)))

# Time Series Plot
def plot_speed_time_series(track_df, track_id, output_dir, method_name="adaptive"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    df = track_df.dropna(subset=['speed_kmh']).reset_index(drop=True)

    if df.empty:
        print(f"Warning: No speed data for track {track_id}")
        plt.close(fig)
        return

    stage_colors = {'ocean': 'blue', 'landfall': 'orange', 'inland': 'red'}

    # Plot the speed line
    ax.plot(df['time'], df['speed_kmh'], color='gray', linestyle='-', linewidth=1.5, zorder=0, alpha=0.7)

    # Plot points by stage
    for stage, color in stage_colors.items():
        subset = df[df['stage'] == stage]
        if not subset.empty:
            ax.plot(
                subset['time'],
                subset['speed_kmh'],
                'o',
                color=color,
                label=stage.capitalize(),
                markersize=6,
                markeredgewidth=1,
                markeredgecolor='black',
                zorder=3
            )

    ax.set_xlabel("Time")
    ax.set_ylabel(f"MD Speed ({VARIABLE_UNITS['speed_kmh']})")
    ax.set_title(f"Track {track_id} Speed Over Time")
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.autofmt_xdate()

    fig.tight_layout()
    legend_x = 1.02
    legend_y = 0.98
    legend_anchor = (legend_x, legend_y)

    leg = ax.legend(title="Stage", loc='upper left', bbox_to_anchor=legend_anchor, fontsize=9, frameon=True, framealpha=0.9)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.5)

    stats_text = f"Mean: {df['speed_kmh'].mean():.1f} km/h\n"
    stats_text += f"Median: {df['speed_kmh'].median():.1f} km/h\n"
    stats_text += f"N: {len(df)}"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='gray', linewidth=0.5))


    outpath = os.path.join(output_dir, f"track_{int(track_id)}_speed_timeseries.png")
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved speed time series plot:", outpath)


# ---------- Land/Ocean classification ----------
@functools.lru_cache(maxsize=128)
def get_land_mask():
    return regionmask.defined_regions.natural_earth_v5_1_2.countries_50

def is_point_land(lat, lon):
    land_mask = get_land_mask()
    mask = land_mask.mask([lon], [lat])
    return not np.isnan(mask[0,0])

# _land_mask = None
# def is_point_land(lat, lon):
#     global _land_mask
#     if _land_mask is None:
#         _land_mask = regionmask.defined_regions.natural_earth_v5_1_2.countries_50
#     mask = _land_mask.mask([lon], [lat])
#     return not np.isnan(mask[0,0])

# ---------- Compute speeds along detection tracks ----------
@numba.jit(nopython=True, parallel=True)
def apply_speed_qc(speeds):
    n = len(speeds)
    mask = np.zeros(n, dtype=np.bool_)
    for i in numba.prange(n):
        mask[i] = (speeds[i] >= 0) and (speeds[i] <= 25) # km/hr
        # mask[i] = (speeds[i] >= 0) and (speeds[i] <= 70) # km/hr

    qc_speeds = np.where(mask, speeds, np.nan)
    smoothed = np.copy(qc_speeds)

    # for i in range(1, len(smoothed)-1):
    #     window = qc_speeds[i-1:i+2]
    #     valid = window[~np.isnan(window)]
    #     if len(valid) >= 2:
    #         smoothed[i] = np.median(valid)
    # if len(smoothed) > 2:
    #     s = pd.Series(smoothed)
    #     interpolated = s.interpolate(method='linear', limit=2, limit_direction='both')
    #     smoothed = interpolated.values
    # return smoothed
    for i in range(1, n-1):
        if not np.isnan(qc_speeds[i]):
            continue

        # Get window
        window = qc_speeds[i-1:i+2]
        valid_count = 0
        window_sum = 0.0

        for val in window:
            if not np.isnan(val):
                valid_count += 1
                window_sum += val

        if valid_count >= 2:
            smoothed[i] = window_sum / valid_count

    return smoothed

def compute_track_speeds_adaptive(track_df, min_time_gap_hours=MAX_GAP_HOURS):
    """centered finite difference method"""
    track_df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(track_df['time']).values
    lats = track_df['latitude'].values
    lons = track_df['longitude'].values

    speeds = np.full(len(track_df), np.nan)

    # Calculate speeds for middle points using centered difference
    for i in range(1, len(track_df)-1):
        prev_idx = i - 1
        next_idx = i + 1

        # Calculate total time difference between prev and next
        total_time_hours = (times[next_idx] - times[prev_idx]).astype('timedelta64[s]').astype(float) / 3600.0

        # Skip if time gap is too large (indicates data gap)
        if total_time_hours > min_time_gap_hours:
            continue

        if total_time_hours > 0:
            # Calculate distance between prev and next
            dist_km = geodesic(
                (lats[prev_idx], lons[prev_idx]),
                (lats[next_idx], lons[next_idx])
            ).km

            # Speed = distance / time
            speeds[i] = dist_km / total_time_hours

    if len(track_df) > 1:
        # forward difference for first point
        time_gap = (times[1] - times[0]).astype('timedelta64[s]').astype(float) / 3600.0
        if 0 < time_gap <= min_time_gap_hours:
            dist_km = geodesic((lats[0], lons[0]), (lats[1], lons[1])).km
            speeds[0] = dist_km / time_gap

        # backward difference for last point
        time_gap = (times[-1] - times[-2]).astype('timedelta64[s]').astype(float) / 3600.0
        if 0 < time_gap <= min_time_gap_hours:
            dist_km = geodesic((lats[-2], lons[-2]), (lats[-1], lons[-1])).km
            speeds[-1] = dist_km / time_gap

    # Apply quality control
    speeds = apply_speed_qc(speeds)
    track_df['speed_kmh'] = speeds
    track_df['speed_ms'] = speeds / 3.6  # Convert to m/s for analysis

    return track_df

def compute_md_metrics(track_df, track_id):
    """
    Compute comprehensive MD metrics for research.
    """
    metrics = {'track_id': track_id}

    # Basic speed statistics
    valid_speeds = track_df['speed_kmh'].dropna()
    if len(valid_speeds) > 0:
        metrics['mean_speed_kmh'] = float(valid_speeds.mean())
        metrics['median_speed_kmh'] = float(valid_speeds.median())
        metrics['std_speed_kmh'] = float(valid_speeds.std())
        metrics['min_speed_kmh'] = float(valid_speeds.min())
        metrics['max_speed_kmh'] = float(valid_speeds.max())

        # Calculate quartiles
        q1 = float(valid_speeds.quantile(0.25))
        q2 = float(valid_speeds.quantile(0.50))
        q3 = float(valid_speeds.quantile(0.75))
        metrics['q1_kmh'] = q1
        metrics['q2_kmh'] = q2
        metrics['q3_kmh'] = q3
        metrics['iqr_kmh'] = q3 - q1

        # Convert to m/s for meteorological analysis
        metrics['mean_speed_ms'] = metrics['mean_speed_kmh'] / 3.6
        metrics['median_speed_ms'] = metrics['median_speed_kmh'] / 3.6
        metrics['q1_ms'] = q1 / 3.6
        metrics['q2_ms'] = q2 / 3.6
        metrics['q3_ms'] = q3 / 3.6
    else:
        metrics['mean_speed_kmh'] = np.nan
        metrics['median_speed_kmh'] = np.nan
        metrics['std_speed_kmh'] = np.nan

    # Track duration
    if len(track_df) >= 2:
        start_time = pd.to_datetime(track_df['time'].iloc[0])
        end_time = pd.to_datetime(track_df['time'].iloc[-1])
        duration_hours = (end_time - start_time).total_seconds() / 3600.0
        metrics['duration_hours'] = duration_hours
        metrics['n_points'] = len(track_df)

        # Total distance traveled
        total_distance = 0
        for i in range(len(track_df)-1):
            dist_km = geodesic(
                (track_df['latitude'].iloc[i], track_df['longitude'].iloc[i]),
                (track_df['latitude'].iloc[i+1], track_df['longitude'].iloc[i+1])
            ).km
            total_distance += dist_km
        metrics['total_distance_km'] = total_distance

        # Net displacement
        start_lat = track_df['latitude'].iloc[0]
        start_lon = track_df['longitude'].iloc[0]
        end_lat = track_df['latitude'].iloc[-1]
        end_lon = track_df['longitude'].iloc[-1]
        net_displacement_km = geodesic(
            (start_lat, start_lon),
            (end_lat, end_lon)
        ).km
        metrics['net_displacement_km'] = net_displacement_km

        # Average translation speed (over entire track)
        if duration_hours > 0:
            metrics['translation_speed_kmh'] = net_displacement_km / duration_hours

    # Speed by stage (ocean/landfall/inland)
    if 'stage' in track_df.columns:
        for stage in ['ocean', 'landfall', 'inland']:
            stage_speeds = track_df[track_df['stage'] == stage]['speed_kmh'].dropna()
            if len(stage_speeds) > 0:
                metrics[f'{stage}_mean_speed_kmh'] = float(stage_speeds.mean())
                metrics[f'{stage}_median_speed_kmh'] = float(stage_speeds.median())
                metrics[f'{stage}_n_points'] = len(stage_speeds)
                # Calculate quartiles for each stage
                metrics[f'{stage}_q1_kmh'] = float(stage_speeds.quantile(0.25))
                metrics[f'{stage}_q2_kmh'] = float(stage_speeds.quantile(0.50))
                metrics[f'{stage}_q3_kmh'] = float(stage_speeds.quantile(0.75))
            else:
                metrics[f'{stage}_mean_speed_kmh'] = np.nan
                metrics[f'{stage}_median_speed_kmh'] = np.nan
                metrics[f'{stage}_n_points'] = 0

    return metrics

# ---------- Split track into ocean / land / landfall / inland ----------
def classify_track_landfall(track_df):
    df = track_df.copy().reset_index(drop=True)
    if 'is_land' not in track_df.columns:
        track_df['is_land'] = track_df.apply(
            lambda row: is_point_land(row['latitude'], row['longitude']), 
            axis=1
    )
    df['is_land'] = df.apply(lambda r: is_point_land(r['latitude'], r['longitude']), axis=1)
    # find first land index
    land_idxs = df.index[df['is_land'] == True].tolist()
    if not land_idxs:
        df['stage'] = 'ocean'
        return df
    first_land_idx = land_idxs[0]
    stages = []
    for idx, row in df.iterrows():
        if idx < first_land_idx:
            stages.append('ocean')
        elif idx == first_land_idx:
            stages.append('landfall')
        else:
            stages.append('inland')
    df['stage'] = stages
    return df

# ---------- Statistical comparisons ----------
def compare_speeds_by_stage(df, track_id, output_dir_path):
    out = {}
    stats_data = {}

    for st in ['ocean','landfall','inland']:
        s = df[df['stage']==st]['speed_kmh'].dropna()
        out[st] = {
            'n': len(s),
            'median': float(s.median()) if len(s)>0 else np.nan,
            'mean': float(s.mean()) if len(s)>0 else np.nan,
            'std': float(s.std()) if len(s)>0 else np.nan,
            'min': float(s.min()) if len(s)>0 else np.nan,
            'max': float(s.max()) if len(s)>0 else np.nan
        }

        # Store for CSV output
        if len(s) > 0:
            # Calculate quartiles
            q1 = float(s.quantile(0.25))
            q2 = float(s.quantile(0.50))
            q3 = float(s.quantile(0.75))

            stats_data[st] = {
                'track_id': track_id,
                'stage': st,
                'n_points': len(s),
                'mean_kmh': float(s.mean()),
                'median_kmh': float(s.median()),
                'std_kmh': float(s.std()),
                'min_kmh': float(s.min()),
                'max_kmh': float(s.max()),
                'q1_kmh': q1,
                'q2_kmh': q2,
                'q3_kmh': q3,
                'iqr_kmh': q3 - q1,
                'mean_ms': float(s.mean() / 3.6),
                'median_ms': float(s.median() / 3.6),
                'q1_ms': q1 / 3.6,
                'q2_ms': q2 / 3.6,
                'q3_ms': q3 / 3.6
            }

    # Mann-Whitney tests
    ocean_s = df[df['stage']=='ocean']['speed_kmh'].dropna()
    inland_s = df[df['stage']=='inland']['speed_kmh'].dropna()
    landfall_s = df[df['stage']=='landfall']['speed_kmh'].dropna()

    stats = {}
    if len(ocean_s) > 0 and len(inland_s) > 0:
        stat, p = mannwhitneyu(ocean_s, inland_s, alternative='two-sided')
        stats['ocean_vs_inland'] = {'statistic': stat, 'p_value': p}
    else:
        stats['ocean_vs_inland'] = {'statistic': None, 'p_value': None}

    if len(ocean_s) > 0 and len(landfall_s) > 0:
        stat2, p2 = mannwhitneyu(ocean_s, landfall_s, alternative='two-sided')
        stats['ocean_vs_landfall'] = {'statistic': stat2, 'p_value': p2}
    else:
        stats['ocean_vs_landfall'] = {'statistic': None, 'p_value': None}

    # Save stage statistics to CSV
    if stats_data:
        stats_df = pd.DataFrame(stats_data.values())
        # csv_path = os.path.join(output_dir_path, f"track_{int(track_id)}_stage_statistics.csv")
        # stats_df.to_csv(csv_path, index=False)
        # print(f"Saved stage statistics CSV: {csv_path}")

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        data_to_plot = [df[df['stage']==s]['speed_kmh'].dropna() for s in ['ocean','landfall','inland']]

        # Create boxplot
        bp = ax.boxplot(data_to_plot, tick_labels=['Ocean','Landfall','Inland'], 
                       showfliers=False, patch_artist=True)

        # Customize box colors
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Customize median line
        for median in bp['medians']:
            median.set_color('red')
            median.set_linewidth(2)

        ax.set_ylabel(f"Speed ({VARIABLE_UNITS['speed_kmh']})")
        ax.set_title(f"Track {track_id}: Speeds by Stage")
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # sample size annotations
        for i, data in enumerate(data_to_plot, 1):
            if len(data) > 0:
                ax.text(i, ax.get_ylim()[-1], f'n={len(data)}',
                       horizontalalignment='center', verticalalignment='top',
                       fontsize=9, fontweight='bold', color='black',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        png_path = os.path.join(output_dir_path, f"track_{int(track_id)}_speeds_boxplot.png")
        fig.tight_layout()
        fig.savefig(png_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved speed boxplot: {png_path}")

    return out, stats, stats_data

def analyze_single_track_wrapper(args):
    """Wrapper for parallel processing"""
    track_df, track_id, track_output_dir = args
    return analyze_single_track(track_df, track_id, track_output_dir)

def analyze_all_tracks_parallel(track_df_with_ids, max_workers=None):
    """Analyze all tracks in parallel"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count() - 1, 4)

    results = {}
    all_metrics = []
    all_stage_stats = []

    # Get unique track IDs
    track_ids = sorted(track_df_with_ids['track_id'].unique())
    print(f"\nFound {len(track_ids)} tracks to analyze")
    print(f"Using {max_workers} parallel workers")

    # Prepare arguments for parallel processing
    args_list = []
    for track_id in track_ids:
        track_output_dir = os.path.join(output_dir, f"track_{int(track_id)}")
        if not os.path.exists(track_output_dir):
            os.makedirs(track_output_dir)

        this_track = track_df_with_ids[track_df_with_ids['track_id'] == track_id].copy()
        this_track['time'] = pd.to_datetime(this_track['time'])

        args_list.append((this_track, track_id, track_output_dir))

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_track = {
            executor.submit(analyze_single_track_wrapper, args): args[1] 
            for args in args_list
        }
        for future in as_completed(future_to_track):
            track_id = future_to_track[future]
            try:
                track_df, metrics, summary_stats, stat_tests = future.result()
                results[track_id] = {
                    'track_df': track_df,
                    'metrics': metrics,
                    'summary_stats': summary_stats,
                    'stat_tests': stat_tests
                }
                all_metrics.append(metrics)
                print(f" Completed track {track_id}")
            except Exception as e:
                print(f" Error processing track {track_id}: {e}")
    return results, all_metrics, all_stage_stats

def analyze_single_track(track_df, track_id, track_output_dir):
    """Analyze a single track with adaptive method"""
    track_df = track_df.sort_values("time").reset_index(drop=True)
    track_df = compute_track_speeds_adaptive(track_df)
    track_df = classify_track_landfall(track_df)

    windspeed_dir = os.path.join(track_output_dir, "windspeed")
    os.makedirs(windspeed_dir, exist_ok=True)

    plot_speed_time_series(track_df, track_id, windspeed_dir)

    metrics = compute_md_metrics(track_df, track_id)
    summary_stats, stat_tests, stats_data = compare_speeds_by_stage(
        track_df, track_id, windspeed_dir
    )
    csv_path = os.path.join(track_output_dir, f"track_{int(track_id)}_augmented.csv")
    track_df.to_csv(csv_path, index=False)
    print(f"Saved augmented track data: {csv_path}")

    return track_df, metrics, summary_stats, stat_tests

def analyze_all_tracks(track_df_with_ids):
    """Analyze all tracks with adaptive method"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = {}
    all_metrics = []
    all_stage_stats = []

    # Get unique track IDs
    track_ids = sorted(track_df_with_ids['track_id'].unique())
    print(f"\nFound {len(track_ids)} tracks to analyze")

    for track_id in track_ids:
        print(f"\n{'='*60}")
        print(f"Processing track {track_id}")
        print(f"{'='*60}")
        track_output_dir = os.path.join(output_dir, f"track_{int(track_id)}")
        if not os.path.exists(track_output_dir):
            os.makedirs(track_output_dir)

        # Extract this track
        this_track = track_df_with_ids[track_df_with_ids['track_id'] == track_id].copy()
        print(f"  Number of points: {len(this_track)}")

        # Ensure time is datetime
        this_track['time'] = pd.to_datetime(this_track['time'])

        # Analyze track
        track_df, metrics, summary_stats, stat_tests = analyze_single_track(this_track, track_id, track_output_dir)

        # Store results
        results[track_id] = {
            'track_df': track_df,
            'metrics': metrics,
            'summary_stats': summary_stats,
            'stat_tests': stat_tests
        }

        # Collect metrics for summary
        all_metrics.append(metrics)

        # Collect stage statistics if available
        for stage in ['ocean', 'landfall', 'inland']:
            stage_key = f'{stage}_mean_speed_kmh'
            if stage_key in metrics and not np.isnan(metrics[stage_key]):
                all_stage_stats.append({
                    'track_id': track_id,
                    'stage': stage,
                    'mean_speed_kmh': metrics[stage_key],
                    'median_speed_kmh': metrics.get(f'{stage}_median_speed_kmh', np.nan),
                    'q1_kmh': metrics.get(f'{stage}_q1_kmh', np.nan),
                    'q2_kmh': metrics.get(f'{stage}_q2_kmh', np.nan),
                    'q3_kmh': metrics.get(f'{stage}_q3_kmh', np.nan),
                    'n_points': metrics.get(f'{stage}_n_points', 0)
                })
    return results, all_metrics, all_stage_stats

if __name__ == "__main__":
    # Load track data
    track_csv_path = os.path.join(output_dir, "MD-coordinates-ALL.csv")

    try:
        full_track_df = pd.read_csv(track_csv_path)
        print(f"Loaded track data from {track_csv_path}")

        if full_track_df.empty or 'track_id' not in full_track_df.columns:
            print("Error: The loaded track file is empty or missing 'track_id'.")
            sys.exit(1)

        # if 'track_id' in full_track_df.columns:
        #     initial_track_count = len(full_track_df['track_id'].unique())
        #     full_track_df = full_track_df[full_track_df['track_id'].isin(desired_tracks)].copy()
        #     filtered_track_count = len(full_track_df['track_id'].unique())
        #     print(f"\nINFO: Filtered tracks for Azimuthal/Hovmöller analysis.")
        #     print(f"INFO: Retaining {filtered_track_count} tracks ({desired_tracks}) out of {initial_track_count} total.")
        # else:
        #     print("WARNING: 'track_id' column not found. Skipping track filter.")

        full_track_df['time'] = pd.to_datetime(full_track_df['time'])

        print(f"\nDataset Information:")
        print(f"  Total points: {len(full_track_df)}")
        print(f"  Unique tracks: {len(full_track_df['track_id'].unique())}")
        print(f"  Time range: {full_track_df['time'].min()} to {full_track_df['time'].max()}")

        # Check time intervals
        time_diffs = []
        for track_id in full_track_df['track_id'].unique():
            track_times = full_track_df[full_track_df['track_id'] == track_id]['time'].sort_values()
            diffs = track_times.diff().dt.total_seconds().dropna() / 3600.0
            time_diffs.extend(diffs.tolist())

        if time_diffs:
            print(f"  Typical time intervals: min={min(time_diffs):.1f}h, "
                  f"max={max(time_diffs):.1f}h, mean={np.mean(time_diffs):.1f}h")

        print("\nStarting speed analysis (adaptive method)")
        if len(full_track_df['track_id'].unique()) > 1:
            print("Using parallel processing for multiple tracks...")
            results, all_metrics, all_stage_stats = analyze_all_tracks_parallel(full_track_df)
        else:
            results, all_metrics, all_stage_stats = analyze_all_tracks(full_track_df)

        # Create summary DataFrame
        # if all_metrics:
        #     summary_df = pd.DataFrame(all_metrics)
        #     summary_csv = os.path.join(output_dir, "all_tracks_summary.csv")
        #     summary_df.to_csv(summary_csv, index=False)
        #     print(f"\nSaved summary of all tracks: {summary_csv}")

        print(f"Results stored in '{output_dir}' directory")

    except FileNotFoundError:
        print(f"Error: Could not find required track file at {track_csv_path}.")
        print("Run md-detection-tracking.py first to generate tracks.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
