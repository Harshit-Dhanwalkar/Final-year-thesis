"""
1. Surface Heat Flux Analysis (Latent, Sensible, Total)
2. Landfall Flux Transitions
3. Flux-Precipitation-Soil Moisture Correlations
4. Short vs Long-lived MD Classification
5. Duration-dependent analysis of MD behavior
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils import (
    std_print,
    is_point_land,
    find_landfall_time,
    calculate_buffer_deg,
    export_results_to_csv
)


# Helper functions
def deaccumulate(ds, var, t_now, t_prev):
    """Convert accumulated J m-2 to W m-2."""
    dt = (pd.to_datetime(t_now) - pd.to_datetime(t_prev)).total_seconds()
    if dt <= 0:
        return np.nan
    # ERA5 sign convention: negative downward -> positive upward
    return -(float(ds[var].sel(time=t_now).values) - float(ds[var].sel(time=t_prev).values)) / dt

def mean(lst):
    return np.nanmean(lst) if lst else np.nan

def get_environment(is_land_list):
    """Determine dominant environment (ocean or land)"""
    if not is_land_list:
        return "Unknown"
    land_count = sum(is_land_list)
    ocean_count = len(is_land_list) - land_count
    if land_count == 0 and ocean_count == 0:
        return "Unknown"
    if land_count > ocean_count:
        return "Land"
    elif ocean_count > land_count:
        return "Ocean"
    else:
        return "Mixed"  # Equal land and ocean points

def extract_heat_flux_timeseries(ds, track_df, track_id, buffer_km=100):
    track_id = int(track_id)
    std_print('INFO', 'hf', 'Dataset variables available', list(ds.data_vars.keys()))
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    ds_times = pd.to_datetime(ds.time.values)

    track = track_df.sort_values('time').reset_index(drop=True)
    if 'is_land' not in track.columns:
        track['is_land'] = track.apply(lambda r: is_point_land(r['latitude'], r['longitude']), axis=1)

    # Container
    flux_records = []

    for i in range(1, len(track)):
        row_now = track.iloc[i]
        row_prev = track.iloc[i - 1]

        t_now = pd.to_datetime(row_now['time'])
        t_prev = pd.to_datetime(row_prev['time'])

        dt_seconds = (t_now - t_prev).total_seconds()
        if dt_seconds == 0:
            std_print('ERROR', 'hf', f'Time dt_seconds = 0. Continuing...', f'Track {track_id}')
            continue
        elif dt_seconds < 0:
            std_print('ERROR', 'hf', f'Time dt_seconds < 0. Continuing...', f'Track {track_id}')
            continue

        lat_c = row_now["latitude"]
        lon_c = row_now["longitude"]
        is_land = row_now['is_land']
        lat_buf, lon_buf = calculate_buffer_deg(row_now['latitude'], buffer_km)

        # Find nearest time indices
        idx_now  = np.argmin(np.abs(ds_times - t_now))
        idx_prev = np.argmin(np.abs(ds_times - t_prev))
        try:
            # Create bounds for spatial selection
            lat_min = lat_c - lat_buf
            lat_max = lat_c + lat_buf
            lon_min = lon_c - lon_buf
            lon_max = lon_c + lon_buf
            ds_now = ds.isel({time_coord: idx_now})
            # Create spatial mask
            lat_mask = (ds_now.latitude >= lat_min) & (ds_now.latitude <= lat_max)
            lon_mask = (ds_now.longitude >= lon_min) & (ds_now.longitude <= lon_max)

            # Apply masks
            if lat_mask.any() and lon_mask.any():
                sub_now = ds_now.where(lat_mask & lon_mask, drop=True)
                # Check for heat flux variables
                if 'avg_slhtf' in sub_now.data_vars and 'avg_ishf' in sub_now.data_vars:
                    slhf_w = float(sub_now["avg_slhtf"].mean().values)
                    sshf_w = float(sub_now["avg_ishf"].mean().values)
                elif 'slhf' in sub_now.data_vars and 'sshf' in sub_now.data_vars:
                    # Select previous time data for deaccumulation
                    ds_prev = ds.isel({time_coord: idx_prev})
                    sub_prev = ds_prev.where(lat_mask & lon_mask, drop=True)
                    # De-accumulate accumulated fluxes J m-2 -> W m-2
                    dt_seconds = 3600.0  # Assuming 1-hour difference
                    if 'slhf' in sub_now.data_vars and 'slhf' in sub_prev.data_vars:
                        slhf_w = -(float(sub_now["slhf"].mean().values) -
                                   float(sub_prev["slhf"].mean().values)) / dt_seconds
                    else:
                        slhf_w = np.nan
                    if 'sshf' in sub_now.data_vars and 'sshf' in sub_prev.data_vars:
                        sshf_w = -(float(sub_now["sshf"].mean().values) -
                                   float(sub_prev["sshf"].mean().values)) / dt_seconds
                    else:
                        sshf_w = np.nan
                else:
                    std_print('WARN', 'hf', f'No valid heat flux variables found at {t_now}')
                    continue
                if np.isnan(slhf_w) or np.isnan(sshf_w):
                    std_print('WARN', 'hf', f'Track {track_id}: NaN values encountered at {t_now}',
                             f'SLHF={slhf_w:.1f}, SSHF={sshf_w:.1f}')
                    continue

                flux_records.append({
                    "time": t_now,
                    "latent_heat_flux": slhf_w,
                    "sensible_heat_flux": sshf_w,
                    "total_heat_flux": slhf_w + sshf_w,
                    "latitude": lat_c,
                    "longitude": lon_c,
                    "is_land": is_land,
                })
            else:
                std_print('WARN', 'hf', f'No data within buffer at {t_now}')
                
        except Exception as e:
            std_print('ERROR', 'hf', f'Error processing time step at {t_now}', str(e))
            continue
    
    if not flux_records:
        std_print('WARN', 'hf', f'Track {track_id}: No heat-flux data extracted')
        return None
    
    std_print('INFO', 'hf', f'Extracted heat flux records for Track {track_id}', len(flux_records), 'records')
    return pd.DataFrame(flux_records)

def analyze_flux_transition(
    ds,
    track_df,
    track_id : int,
    hours_before=24,
    hours_after=24,
    buffer_km=100,
):
    track_id = int(track_id)
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    track_df = track_df.sort_values('time')
    if 'is_land' not in track_df.columns:
        track_df['is_land'] = track_df.apply(lambda r: is_point_land(r['latitude'], r['longitude']), axis=1)

    # Find first landfall
    # Ensure is_land column exists
    if 'is_land' not in track_df.columns:
        track_df['is_land'] = track_df.apply(
            lambda row: is_point_land(row['latitude'], row['longitude']),
            axis=1
        )
    landfall_time = find_landfall_time(track_df)

    if landfall_time is None:
        std_print('INFO', 'hf', f'Track {track_id}: No landfall detected', 'Using track midpoint as reference')
        midpoint_idx = len(track_df) // 2
        landfall_time = pd.to_datetime(track_df.iloc[midpoint_idx]['time'])

    # Time windows
    t_start = landfall_time - pd.Timedelta(hours=hours_before)
    t_end = landfall_time + pd.Timedelta(hours=hours_after)
    ds_times = pd.to_datetime(ds[time_coord].values)

    # Containers
    before = {"latent": [], "sensible": [], "total": []}
    after  = {"latent": [], "sensible": [], "total": []}
    before_is_land = []
    after_is_land = []

    for _, row in track_df.iterrows():
        t = pd.to_datetime(row['time'])
        if t < t_start or t > t_end:
            continue

        lat_c = row['latitude']
        lon_c = row['longitude']
        is_land = row.get('is_land', False)
        lat_buf, lon_buf = calculate_buffer_deg(row['latitude'], buffer_km=100)

        # Nearest ERA5 time
        idx = np.argmin(np.abs(ds_times - t))
        t_nearest = ds_times[idx]
        t_prev = t_nearest - pd.Timedelta(hours=1)
        idx_prev = np.argmin(np.abs(ds_times - t_prev))
        try:
            subset_now = ds.isel({time_coord: idx}).sel(
                latitude=slice(lat_c + lat_buf, lat_c - lat_buf),
                longitude=slice(lon_c - lon_buf, lon_c + lon_buf),
            )
            subset_prev = ds.isel({time_coord: idx_prev}).sel(
                latitude=slice(lat_c + lat_buf, lat_c - lat_buf),
                longitude=slice(lon_c - lon_buf, lon_c + lon_buf),
            )
            # Check which flux variables are available
            if 'avg_slhtf' in ds.data_vars and 'avg_ishf' in ds.data_vars:
                # Use average fluxes directly (already in W/m²)
                slhf_w = float(subset_now["avg_slhtf"].mean().values)
                sshf_w = float(subset_now["avg_ishf"].mean().values)
                std_print('INFO', 'hf', 'Using avg_slhtf and avg_ishf (Accumulated)')
            elif 'slhf' in ds.data_vars and 'sshf' in ds.data_vars:
                # De-accumulate accumulated fluxes  J m-2  ->  W m-2  (ERA5 negative downward)
                dt_seconds = 3600.0
                slhf_w = -(float(subset_now["slhf"].mean().values) -
                           float(subset_prev["slhf"].mean().values)) / dt_seconds
                sshf_w = -(float(subset_now["sshf"].mean().values) -
                           float(subset_prev["sshf"].mean().values)) / dt_seconds
                std_print('INFO', 'hf', 'Using avg_slhtf and avg_ishf (De-accumulated)')
            else:
                std_print('WARN', 'hf', f'No valid heat flux variables at {t}')
                continue
            if np.isnan(slhf_w) or np.isnan(sshf_w):
                continue

            total = slhf_w + sshf_w
            bucket = after if t >= landfall_time else before
            bucket["latent"].append(slhf_w)
            bucket["sensible"].append(sshf_w)
            bucket["total"].append(total)

            # Track land/ocean status
            status_bucket = after_is_land if t >= landfall_time else before_is_land
            status_bucket.append(is_land)
        except Exception as e:
            std_print('ERROR', 'hf', f'Error at {t}', str(e))
            continue

    # Determine if this was an actual landfall transition
    has_actual_landfall = False
    if len(before["latent"]) > 0 and len(after["latent"]) > 0:
        # Check if we transitioned from ocean to land
        has_actual_landfall = (get_environment(before_is_land) == "Ocean" and 
                              get_environment(after_is_land) == "Land")

    results = {
        "track_id": track_id,
        "landfall_time": landfall_time,
        "has_actual_landfall": has_actual_landfall,
        "before_environment": get_environment(before_is_land),
        "after_environment": get_environment(after_is_land),
        "before": {
            "latent":   mean(before["latent"]),
            "sensible": mean(before["sensible"]),
            "total":    mean(before["total"]),
            "n_samples": len(before["latent"]),
        },
        "after": {
            "latent":   mean(after["latent"]),
            "sensible": mean(after["sensible"]),
            "total":    mean(after["total"]),
            "n_samples": len(after["latent"]),
        },
        "change": {
            "latent":   mean(after["latent"]) - mean(before["latent"]),
            "sensible": mean(after["sensible"]) - mean(before["sensible"]),
            "total":    mean(after["total"]) - mean(before["total"]),
        },
    }
    if has_actual_landfall:
        std_print('INFO', 'hf', f'Track {track_id}: Landfall flux transition')
    else:
        std_print('INFO', 'hf', f'Track {track_id}: Heat flux transition (no landfall)')
    std_print('INFO', 'hf', 'Before Total HF', f"{results['before']['total']:.1f}", 'W/m²')
    std_print('INFO', 'hf', 'After Total HF', f"{results['after']['total']:.1f}", 'W/m²')
    std_print('INFO', 'hf', 'Total HF Change', f"{results['change']['total']:.1f}", 'W/m²')
    return results


def analyze_heat_flux_evolution(flux_df, track_df, track_id):
    """Analyze heat flux statistics and ocean/land phase differences"""
    track_id = int(track_id)
    if flux_df is None or len(flux_df) < 5:
        std_print('WARN', 'hf', f'Track {track_id}: Insufficient flux data', len(flux_df) if flux_df is not None else 0)
        return None
    results = {
        'track_id': track_id,
        'n_samples': len(flux_df),
        'mean_latent_hf': flux_df['latent_heat_flux'].mean(),
        'mean_sensible_hf': flux_df['sensible_heat_flux'].mean(),
        'mean_total_hf': flux_df['total_heat_flux'].mean(),
        'std_latent_hf': flux_df['latent_heat_flux'].std(),
        'std_sensible_hf': flux_df['sensible_heat_flux'].std(),
        'std_total_hf': flux_df['total_heat_flux'].std(),
    }

    # Ocean vs Land phase analysis
    if 'is_land' in flux_df.columns:
        ocean_phase = flux_df[flux_df['is_land'] == False]
        land_phase = flux_df[flux_df['is_land'] == True]
        if len(ocean_phase) > 0:
            results['ocean_mean_latent_hf'] = ocean_phase['latent_heat_flux'].mean()
            results['ocean_mean_sensible_hf'] = ocean_phase['sensible_heat_flux'].mean()
            results['ocean_mean_total_hf'] = ocean_phase['total_heat_flux'].mean()
            results['ocean_n_samples'] = len(ocean_phase)
        if len(land_phase) > 0:
            results['land_mean_latent_hf'] = land_phase['latent_heat_flux'].mean()
            results['land_mean_sensible_hf'] = land_phase['sensible_heat_flux'].mean()
            results['land_mean_total_hf'] = land_phase['total_heat_flux'].mean()
            results['land_n_samples'] = len(land_phase)
            # Calculate changes only if both phases exist
            if len(ocean_phase) > 0:
                results['latent_hf_change'] = (results['land_mean_latent_hf'] - results['ocean_mean_latent_hf'])
                results['sensible_hf_change'] = (results['land_mean_sensible_hf'] - results['ocean_mean_sensible_hf'])
                results['total_hf_change'] = (results['land_mean_total_hf'] - results['ocean_mean_total_hf'])
    return results

def compute_flux_tp_correlation(flux_df, track_df, track_id):
    if flux_df is None or len(flux_df) < 5:
        std_print('WARN', 'hf', f'Track {track_id}: Insufficient flux data for correlation', len(flux_df) if flux_df is not None else 0)
        return None

    flux_times = pd.to_datetime(flux_df['time'])
    track_times = pd.to_datetime(track_df['time'])

    common_times = np.intersect1d(flux_times, track_times)
    if len(common_times) < 3:
        std_print('WARN', 'hf', f'Track {track_id}: Insufficient common times for flux-tp correlation', len(common_times))
        return None

    flux_aligned = flux_df[flux_df['time'].isin(common_times)]['total_heat_flux'].values
    track_aligned = track_df[track_df['time'].isin(common_times)]

    if 'tp' in track_aligned.columns:
        tp_vals = track_aligned['tp'].values * 1000  # Convert to mm/hr
        if len(flux_aligned) == len(tp_vals):
            try:
                r, p = pearsonr(flux_aligned, tp_vals)
                return {'r': r, 'p': p, 'n': len(flux_aligned)}
            except:
                return None
    return None


def classify_md_duration(
    track_df,
    short_threshold_hours=96, # 96 hours = 4 days
): 
    track_times = pd.to_datetime(track_df['time'])
    duration_hours = (track_times.max() - track_times.min()).total_seconds() / 3600
    classification = {
        'duration_hours': duration_hours,
        'is_long_lived': duration_hours >= short_threshold_hours,
        'category': 'Long-lived MD' if duration_hours >= short_threshold_hours else 'Short MD',
        'start_time': track_times.min(),
        'end_time': track_times.max(),
        'num_points': len(track_df)
    }
    return classification

def plot_heat_flux_timeseries(flux_df, track_df, track_id, output_dir):
    if flux_df is None or len(flux_df) < 3:
        std_print('WARN', 'hf', f'Track {track_id}: Insufficient data points for heat flux plot', len(flux_df) if flux_df is not None else 0)
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    times = pd.to_datetime(flux_df['time'])

    # Plot 1: Latent Heat Flux
    ax1 = axes[0]
    ax1.plot(
        times,
        flux_df['latent_heat_flux'],
        'g-',
        linewidth=2.5,
        marker='o',
        markersize=6,
        label='Latent Heat Flux',
        alpha=0.8
    )
    ax1.set_ylabel('Latent Heat Flux (W/m²)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Track {track_id}: Surface Heat Flux Evolution', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.5, linestyle='--')
    ax1.legend(fontsize=10, loc='upper left')

    # Plot 2: Sensible Heat Flux
    ax2 = axes[1]
    ax2.plot(
        times,
        flux_df['sensible_heat_flux'],
        'orange',
        linewidth=2.5,
        marker='s',
        markersize=6,
        label='Sensible Heat Flux',
        alpha=0.8
    )
    ax2.set_ylabel('Sensible Heat Flux (W/m²)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.5, linestyle='--')
    ax2.legend(fontsize=10, loc='upper left')

    # Plot 3: Total Heat Flux
    ax3 = axes[2]
    ax3.plot(
        times,
        flux_df['total_heat_flux'],
        'purple',
        linewidth=2.5,
        marker='D',
        markersize=6,
        label='Total Heat Flux',
        alpha=0.8
    )

    # Shade land/ocean regions
    if 'is_land' in flux_df.columns:
        for i in range(len(flux_df) - 1):
            if flux_df.iloc[i]['is_land']:
                ax3.axvspan(times.iloc[i], times.iloc[i+1], alpha=0.2, color='brown')
            else:
                ax3.axvspan(times.iloc[i], times.iloc[i+1], alpha=0.2, color='blue')

    ax3.set_ylabel('Total Heat Flux (W/m²)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.5, linestyle='--')

    # Add legend for ocean/land shading
    legend_elements = [
        Patch(facecolor='blue', alpha=0.1, label='Ocean'),
        Patch(facecolor='red', alpha=0.1, label='Land'),
    ]
    ax3.legend(
        handles=legend_elements + [ax3.get_lines()[0]],
        loc='upper left',
        fontsize=10
    )

    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'track_{int(track_id)}_heat_flux_timeseries.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'hf', 'Heat flux timeseries plot', plot_path)


def export_heat_flux_results(
    all_flux_results,
    output_dir='output',
):
    if not all_flux_results:
        std_print('WARN', 'hf', 'No flux results to export', )
        return

    results_df = pd.DataFrame(all_flux_results)
    export_results_to_csv(results_df, output_dir, 'heat_flux_analysis_summary', 'hf')


def export_landfall_flux_results(all_landfall_results, output_dir='output'):
    """Export landfall flux transition results to CSV"""
    if not all_landfall_results:
        std_print('WARN', 'hf', 'No landfall flux results to export', )
        return

    results_list = []
    seen_tracks = set()
    for result in all_landfall_results:
        if result is not None:
            track_id = result['track_id']
            # Skip duplicate results for same track
            if track_id in seen_tracks:
                continue
            seen_tracks.add(track_id)
            results_list.append({
                'track_id': result['track_id'],
                'reference_time': result['landfall_time'],
                'has_actual_landfall': result['has_actual_landfall'],
                'before_environment': result.get('before_environment', 'Unknown'),
                'after_environment': result.get('after_environment', 'Unknown'),
                'before_latent_hf': result['before']['latent'],
                'before_sensible_hf': result['before']['sensible'],
                'before_total_hf': result['before']['total'],
                'before_n_samples': result['before']['n_samples'],
                'after_latent_hf': result['after']['latent'],
                'after_sensible_hf': result['after']['sensible'],
                'after_total_hf': result['after']['total'],
                'after_n_samples': result['after']['n_samples'],
                'change_latent_hf': result['change']['latent'],
                'change_sensible_hf': result['change']['sensible'],
                'change_total_hf': result['change']['total'],
            })
    if results_list:
        results_df = pd.DataFrame(results_list)
        os.makedirs(output_dir, exist_ok=True)
        heat_flux_transitions_csv = os.path.join(output_dir, 'heat_flux_transitions.csv')
        results_df.to_csv(heat_flux_transitions_csv, index=False)
        std_print('SAVED', 'hf', 'Landfall HF transition analyses', heat_flux_transitions_csv)
        std_print('INFO', 'hf', f'Tracks with actual landfall', results_df['has_actual_landfall'].sum())
        std_print('INFO', 'hf', f'Tracks using track midpoint', (~results_df['has_actual_landfall']).sum())


def print_md_classification_summary(classifications, output_dir='output'):
    df = pd.DataFrame(classifications)
    short_mds = df[~df['is_long_lived']]
    long_mds = df[df['is_long_lived']]
    std_print('INFO', 'hf', 'Total MDs detected', len(df), 'tracks')
    std_print('INFO', 'hf', 'Short-lived MDs (< 96h)', len(short_mds), f"({100*len(short_mds)/len(df):.1f}%)")
    std_print('INFO', 'hf', 'Long-lived MDs (≥ 96h)', len(long_mds), f"({100*len(long_mds)/len(df):.1f}%)")
    if len(short_mds) > 0:
        std_print('INFO', 'hf', 'Short-lived MD mean duration', f"{short_mds['duration_hours'].mean():.1f} ± {short_mds['duration_hours'].std():.1f}", 'hours')
        std_print('INFO', 'hf', 'Short-lived MD range', f"{short_mds['duration_hours'].min():.1f} - {short_mds['duration_hours'].max():.1f}", 'hours')
    if len(long_mds) > 0:
        std_print('INFO', 'hf', 'Long-lived MD mean duration', f"{long_mds['duration_hours'].mean():.1f} ± {long_mds['duration_hours'].std():.1f}", 'hours')
        std_print('INFO', 'hf', 'Long-lived MD range', f"{long_mds['duration_hours'].min():.1f} - {long_mds['duration_hours'].max():.1f}", 'hours')

    classification_summary_path = os.path.join(output_dir, f'classification_summary.txt')
    std_print('SAVED', 'hf', 'Classification Summary', classification_summary_path)
