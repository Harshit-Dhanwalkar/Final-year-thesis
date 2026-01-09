"""
Enhanced MD Heat Flux Analysis Module
Implements:
1. Surface Heat Flux Analysis (Latent, Sensible, Total)
2. Landfall Flux Transitions
3. Flux-Precipitation-Soil Moisture Correlations
4. Short vs Long-lived MD Classification
5. Duration-dependent analysis of MD behavior

Following the research paper findings that surface heat fluxes matter more than
soil moisture alone for MD inland penetration.
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


# Helper functions
def deaccumulate(ds, var, t_now, t_prev):
    """Convert accumulated J m-2 to W m-2."""
    dt = (pd.to_datetime(t_now) - pd.to_datetime(t_prev)).total_seconds()
    if dt <= 0:
        return np.nan
    # ERA5 sign convention: negative downward -> positive upward
    return -(float(ds[var].sel(time=t_now).values) - float(ds[var].sel(time=t_prev).values)) / dt

def extract_heat_flux_timeseries(
    ds,
    track_df,
    track_id,
    buffer_km=100,
):
    print(f" [DEBUG]  Dataset variables available: {list(ds.data_vars.keys())}")

    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    track = track_df.sort_values('time').reset_index(drop=True)
    ds_times = pd.to_datetime(ds.time.values)

    # Container
    flux_records = []

    lat_buffer = buffer_km / 111.0

    for i in range(1, len(track)):
        row_now = track.iloc[i]
        row_prev = track.iloc[i - 1]

        t_now = pd.to_datetime(row_now['time'])
        t_prev = pd.to_datetime(row_prev['time'])

        dt_seconds = (t_now - t_prev).total_seconds()
        if dt_seconds == 0:
            print(" >  [ERROR   ] `dt_seconds` time is zero. Continuing...")
            continue
        elif dt_seconds < 0:
            print(" >  [ERROR   ] `dt_seconds` time is negative. Continuing...")
            continue

        lat_c = row_now["latitude"]
        lon_c = row_now["longitude"]
        lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat_c)))

        # nearest ERA5 times
        idx_now  = np.argmin(np.abs(ds_times - t_now))
        idx_prev = np.argmin(np.abs(ds_times - t_prev))
        try:
            sub_now = ds.isel({time_coord: idx_now}).sel(
                latitude=slice(lat_c + lat_buffer, lat_c - lat_buffer),
                longitude=slice(lon_c - lon_buffer, lon_c + lon_buffer),
            )
            sub_prev = ds.isel({time_coord: idx_prev}).sel(
                latitude=slice(lat_c + lat_buffer, lat_c - lat_buffer),
                longitude=slice(lon_c - lon_buffer, lon_c + lon_buffer),
            )
            # Check which flux variables are available
            if 'avg_slhtf' in ds.data_vars and 'avg_ishf' in ds.data_vars:
                slhf_w = float(sub_now["avg_slhtf"].mean().values)
                sshf_w = float(sub_now["avg_ishf"].mean().values)
                # print(f" >> [DEBUG] Using avg_slhtf and avg_ishf at {t_now}")
            else:
                print(f" >  [WARNING ] No valid heat flux variables found at {t_now}")
                print(f"      Available vars:                                       {list(ds.data_vars.keys())}")
                continue
            if np.isnan(slhf_w) or np.isnan(sshf_w):
                print(f" >  [WARNING ] NaN values encountered at {t_now}:           SLHF={slhf_w}, SSHF={sshf_w}")
                continue

            flux_records.append({
                    "time": t_now,
                    "latent_heat_flux": slhf_w,
                    "sensible_heat_flux": sshf_w,
                    "total_heat_flux": slhf_w + sshf_w,
                    "latitude": lat_c,
                    "longitude": lon_c,
            })
        except Exception as e:
            print(f" >  [WARNING ] Error processing time step {t_now}: {str(e)}")
            continue
    if not flux_records:
        print(f" >  [WARNING ] No heat-flux data extracted for Track {track_id}")
        return None
    print(f" >  [INFO    ] Extracted {len(flux_records)} heat flux records for Track {track_id}")
    return pd.DataFrame(flux_records)

def analyze_landfall_flux_transition(
    ds,
    track_df,
    track_id,
    hours_before=24,
    hours_after=24,
    buffer_km=100,
):
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    track_df = track_df.sort_values('time')

    # Find first landfall
    landfall_idx = -1
    landfall_time = None
    for i, row in track_df.iterrows():
        if row.get('is_land', False):
            landfall_idx = i
            landfall_time = pd.to_datetime(row['time'])
            break

    if landfall_time is None:
        print(f" >  [INFO    ] Track {track_id}: No landfall detected for flux analysis")
        return None

    # Time windows
    t_start = landfall_time - pd.Timedelta(hours=hours_before)
    t_end = landfall_time + pd.Timedelta(hours=hours_after)
    ds_times = pd.to_datetime(ds[time_coord].values)

    # Containers
    before = {"latent": [], "sensible": [], "total": []}
    after  = {"latent": [], "sensible": [], "total": []}

    lat_buffer = buffer_km / 111.0

    for _, row in track_df.iterrows():
        t = pd.to_datetime(row['time'])
        if t < t_start or t > t_end:
            continue

        lat_c = row['latitude']
        lon_c = row['longitude']
        lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat_c)))

        # Nearest ERA5 time
        idx = np.argmin(np.abs(ds_times - t))
        t_nearest = ds_times[idx]
        t_prev = t_nearest - pd.Timedelta(hours=1)
        idx_prev = np.argmin(np.abs(ds_times - t_prev))

        try:
            subset_now = ds.isel({time_coord: idx}).sel(
                latitude=slice(lat_c + lat_buffer, lat_c - lat_buffer),
                longitude=slice(lon_c - lon_buffer, lon_c + lon_buffer),
            )
            subset_prev = ds.isel({time_coord: idx_prev}).sel(
                latitude=slice(lat_c + lat_buffer, lat_c - lat_buffer),
                longitude=slice(lon_c - lon_buffer, lon_c + lon_buffer),
            )
            # Check which flux variables are available
            if 'avg_slhtf' in ds.data_vars and 'avg_ishf' in ds.data_vars:
                # Use average fluxes directly (already in W/m²)
                slhf_w = float(subset_now["avg_slhtf"].mean().values)
                sshf_w = float(subset_now["avg_ishf"].mean().values)
                print(f" >> [DEBUG] Using avg_slhtf and avg_ishf (accumulated)")
            elif 'slhf' in ds.data_vars and 'sshf' in ds.data_vars:
                # De-accumulate accumulated fluxes  J m-2  ->  W m-2  (ERA5 negative downward)
                dt_seconds = 3600.0
                slhf_w = -(float(subset_now["slhf"].mean().values) -
                           float(subset_prev["slhf"].mean().values)) / dt_seconds
                sshf_w = -(float(subset_now["sshf"].mean().values) -
                           float(subset_prev["sshf"].mean().values)) / dt_seconds
                print(f" >> [DEBUG] Using avg_slhtf and avg_ishf (De-accumulated)")
            else:
                print(f" >  [WARNING] No valid heat flux variables at {t}")
                continue

            if np.isnan(slhf_w) or np.isnan(sshf_w):
                continue

            total = slhf_w + sshf_w
            bucket = after if t >= landfall_time else before
            bucket["latent"].append(slhf_w)
            bucket["sensible"].append(sshf_w)
            bucket["total"].append(total)
        except Exception as e:
            print(f" >> [WARNING] Error at {t}: {str(e)}")
            continue

    # statistics
    def _mean(lst):
        return np.nanmean(lst) if lst else np.nan

    results = {
        "track_id": track_id,
        "landfall_time": landfall_time,
        "before": {
            "latent":   _mean(before["latent"]),
            "sensible": _mean(before["sensible"]),
            "total":    _mean(before["total"]),
            "n_samples": len(before["latent"]),
        },
        "after": {
            "latent":   _mean(after["latent"]),
            "sensible": _mean(after["sensible"]),
            "total":    _mean(after["total"]),
            "n_samples": len(after["latent"]),
        },
        "change": {
            "latent":   _mean(after["latent"]) - _mean(before["latent"]),
            "sensible": _mean(after["sensible"]) - _mean(before["sensible"]),
            "total":    _mean(after["total"]) - _mean(before["total"]),
        },
    }
    print(f" >  [INFO    ] Track {track_id}: Landfall flux transition")
    print(f"      Before – Total HF                     : {results['before']['total']:.1f} W m-2")
    print(f"      After  – Total HF                     : {results['after']['total']:.1f} W m-2")
    print(f"      Change                                : {results['change']['total']:.1f} W m-2")
    return results


def analyze_heat_flux_evolution(flux_df, track_df, track_id):
    """Analyze heat flux statistics and ocean/land phase differences"""
    if flux_df is None or len(flux_df) < 5:
        print(f" >  [WARNING ] Track {track_id}: Insufficient flux data for evolution analysis")
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
            # Calculate changes
            if len(ocean_phase) > 0:
                results['latent_hf_change'] = (results['land_mean_latent_hf'] - results['ocean_mean_latent_hf'])
                results['sensible_hf_change'] = (results['land_mean_sensible_hf'] - results['ocean_mean_sensible_hf'])
                results['total_hf_change'] = (results['land_mean_total_hf'] - results['ocean_mean_total_hf'])
    return results


def compute_flux_tp_correlation(flux_df, track_df):
    if flux_df is None or len(flux_df) < 5:
        return None

    flux_times = pd.to_datetime(flux_df['time'])
    track_times = pd.to_datetime(track_df['time'])

    common_times = np.intersect1d(flux_times, track_times)
    if len(common_times) < 3:
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


def classify_md_duration(track_df, short_threshold_hours=72):
    """Classify MD as short-lived or long-lived based on duration"""
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
        print(f" >  [WARNING ] Track {track_id}: Insufficient data for heat flux plot")
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
    ax1.grid(True, alpha=0.3, linestyle='--')
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
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='upper left')

    # Plot 3: Total Heat Flux with land/ocean shading
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
                ax3.axvspan(times.iloc[i], times.iloc[i+1], alpha=0.1, color='brown')
            else:
                ax3.axvspan(times.iloc[i], times.iloc[i+1], alpha=0.1, color='blue')

    ax3.set_ylabel('Total Heat Flux (W/m²)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

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
    outpath = os.path.join(output_dir, f'track_{int(track_id)}_heat_flux_timeseries.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" >> [SAVED   ] Heat flux timeseries: {outpath}")


def plot_flux_comparison_short_vs_long(all_flux_results, output_dir):
    """Compare heat flux characteristics between short-lived and long-lived MDs"""
    if not all_flux_results or len(all_flux_results) < 2:
        print(" >  [WARNING ] Insufficient tracks for short vs long comparison")
        return

    results_df = pd.DataFrame(all_flux_results)

    # Categorize
    short_md = results_df[~results_df['is_long_lived']]
    long_md = results_df[results_df['is_long_lived']]

    if len(short_md) == 0 or len(long_md) == 0:
        print(" >  [WARNING ] No tracks in one of the categories")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Mean Total Heat Flux
    ax1 = axes[0, 0]
    categories = ['Short MD\n(< 72h)', 'Long-lived MD\n(≥ 72h)']
    means = [short_md['mean_total_hf'].mean(), long_md['mean_total_hf'].mean()]
    stds = [short_md['mean_total_hf'].std(), long_md['mean_total_hf'].std()]
    ax1.bar(
        categories,
        means,
        yerr=stds,
        capsize=10,
        color=['lightcoral', 'lightblue'],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    ax1.set_ylabel('Mean Total Heat Flux (W/m²)', fontsize=8, fontweight='bold')
    ax1.set_title('Total Heat Flux Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Latent vs Sensible
    ax2 = axes[0, 1]
    x = np.arange(2)
    width = 0.35
    latent_means = [short_md['mean_latent_hf'].mean(), long_md['mean_latent_hf'].mean()]
    sensible_means = [short_md['mean_sensible_hf'].mean(), long_md['mean_sensible_hf'].mean()]
    ax2.bar(
        x - width/2,
        latent_means,
        width,
        label='Latent',
        color='green',
        alpha=0.7,
        edgecolor='black'
    )
    ax2.bar(
        x + width/2,
        sensible_means,
        width,
        label='Sensible',
        color='orange',
        alpha=0.7,
        edgecolor='black'
    )
    ax2.set_ylabel('Heat Flux (W/m²)', fontsize=8, fontweight='bold')
    ax2.set_title('Latent vs Sensible Heat Flux', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Sample counts
    ax3 = axes[1, 0]
    ax3.bar(
        categories,
        [len(short_md), len(long_md)],
        color=['lightcoral', 'lightblue'],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    ax3.set_ylabel('Number of MDs', fontsize=8, fontweight='bold')
    ax3.set_title('MD Sample Size', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_data = [
        ['Metric', 'Short MD', 'Long-lived MD'],
        ['N tracks', f'{len(short_md)}', f'{len(long_md)}'],
        ['Mean Total HF (W/m²)', f'{short_md["mean_total_hf"].mean():.1f}',
         f'{long_md["mean_total_hf"].mean():.1f}'],
        ['Mean Latent HF (W/m²)', f'{short_md["mean_latent_hf"].mean():.1f}',
         f'{long_md["mean_latent_hf"].mean():.1f}'],
        ['Mean Sensible HF (W/m²)', f'{short_md["mean_sensible_hf"].mean():.1f}',
         f'{long_md["mean_sensible_hf"].mean():.1f}'],
        ['Avg Duration (h)', f'{short_md["duration_hours"].mean():.1f}',
         f'{long_md["duration_hours"].mean():.1f}'],
    ]
    table = ax4.table(
        cellText=summary_data,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.0)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    fig.suptitle(
        'Short vs Long-lived Monsoon Depressions: Heat Flux Analysis',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, 'short_vs_long_hf_comparison.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" >> [SAVED   ] Short vs Long comparison: {outpath}")


def export_heat_flux_results(all_flux_results, output_dir='output'):
    if not all_flux_results:
        print(" >  [WARNING ] No flux results to export")
        return

    results_df = pd.DataFrame(all_flux_results)
    os.makedirs(output_dir, exist_ok=True)

    outpath = os.path.join(output_dir, 'heat_flux_analysis_summary.csv')
    results_df.to_csv(outpath, index=False)
    print(f" >> [SAVED   ] Heat flux summary: {outpath}")


def export_landfall_flux_results(
    all_landfall_results,
    output_dir='output',
):
    """Export landfall flux transition results to CSV"""
    if not all_landfall_results:
        print(" >  [WARNING ] No landfall flux results to export")
        return

    results_list = []
    for result in all_landfall_results:
        if result is not None:
            results_list.append({
                'track_id': result['track_id'],
                'landfall_time': result['landfall_time'],
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
        outpath = os.path.join(output_dir, 'landfall_flux_transitions.csv')
        results_df.to_csv(outpath, index=False)
        print(f" >> [SAVED   ] Landfall flux transitions: {outpath}")


def print_md_classification_summary(classifications, output_dir='output'):
    df = pd.DataFrame(classifications)
    short_mds = df[~df['is_long_lived']]
    long_mds = df[df['is_long_lived']]
    print("\n -- Monsoon depression duration classification summary --")
    print(f" >  [INFO    ] Total MDs detected:                          {len(df)}")
    print(f"      Short-lived MDs (< 72h):                              {len(short_mds)} ({100*len(short_mds)/len(df):.1f}%)")
    print(f"      Long-lived MDs  (≥ 72h):                              {len(long_mds)} ({100*len(long_mds)/len(df):.1f}%)")

    if len(short_mds) > 0:
        print(f" >  [INFO    ] Short-lived MD Statistics:")
        print(f"      Mean duration:                                        {short_mds['duration_hours'].mean():.1f} ± {short_mds['duration_hours'].std():.1f} hours")
        print(f"      Range:                                                {short_mds['duration_hours'].min():.1f} - {short_mds['duration_hours'].max():.1f} hours")

    if len(long_mds) > 0:
        print(f" >  [INFO    ] Long-lived MD Statistics:")
        print(f"      Mean duration:                                        {long_mds['duration_hours'].mean():.1f} ± {long_mds['duration_hours'].std():.1f} hours")
        print(f"      Range:                                                {long_mds['duration_hours'].min():.1f} - {long_mds['duration_hours'].max():.1f} hours")
