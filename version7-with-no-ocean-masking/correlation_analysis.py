"""
Correlation Analysis: Precipitation vs Soil Moisture
Analyzes the temporal relationship between TP and SWVL1 during track passage

- Direct correlation (overall, pre-landfall, post-landfall)
- Lagged correlation analysis
- Visual comparisons with landfall markers
- Separate time series and scatter plots
- Heat flux correlations with soil moisture

1. TP vs SWVL1 (primary feedback loop)
2. TP vs Surface Heat Fluxes
3. Heat Fluxes vs Track Characteristics
4. SWVL1 vs Total/Latent/Sensible Heat Flux
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import pearsonr, spearmanr, linregress
from statsmodels.tsa.stattools import acf

import traceback

from utils import (
    std_print,
    is_point_land,
    find_landfall_time,
    find_origin_type,
    add_land_ocean_shading,
    extract_spatial_subset,
    extract_regional_data,
    setup_output_dir,
    get_time_coord,
    export_results_to_csv
)

# Helper functions
def lag_correlation(x, y, max_lag=48): # max_lag=72, # hrs
    x = np.asarray(x)
    y = np.asarray(y)
    # Remove NaN values while keeping alignment
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 10:
        std_print('WARN', 'ca', f'Insufficient non-NaN data for lag correlation', f'{len(x_clean)} points')
        return [], []

    lags = range(-max_lag, max_lag+1)
    correlations = []
    for lag in lags:
        if lag < 0:
            # x lags y (y leads x)
            if -lag >= len(x_clean):
                correlations.append(np.nan)
                continue
            x_slice = x_clean[:-(-lag)] if -lag > 0 else x_clean
            y_slice = y_clean[(-lag):] if -lag > 0 else y_clean
        elif lag > 0:
            # y lags x (x leads y)
            if lag >= len(x_clean):
                correlations.append(np.nan)
                continue
            x_slice = x_clean[lag:]
            y_slice = y_clean[:-lag] if lag > 0 else y_clean
        else:
            # Zero lag
            x_slice = x_clean
            y_slice = y_clean
        # Check for sufficient data and non-zero variance
        if len(x_slice) < 3:
            correlations.append(np.nan)
            continue
        std_x = np.nanstd(x_slice)
        std_y = np.nanstd(y_slice)
        if std_x == 0 or std_y == 0:
            correlations.append(np.nan)
            continue
        # Calculate correlation
        try:
            r = np.corrcoef(x_slice, y_slice)[0, 1]
            correlations.append(r)
        except:
            correlations.append(np.nan)
    return list(lags), correlations

def extract_spatial_mean(ds, var_name, t, lat, lon, buffer_km):
    ds, time_coord = get_time_coord(ds)
    subset = extract_spatial_subset(ds, lat, lon, t, buffer_km=buffer_km, time_coord=time_coord)
    if subset is None or subset.sizes.get('latitude', 0) == 0:
        return np.nan
    if var_name in subset:
        try:
            val = float(subset[var_name].mean().values)
            return val if not np.isnan(val) else np.nan
        except:
            return np.nan
    else:
        return np.nan

def effective_sample_size(x, y, max_lag=10):
    """Calculate effective sample size accounting for autocorrelation"""
    # Remove NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 20:
        return len(x_clean)  # Too short for reliable autocorrelation
    try:
        # Calculate autocorrelation at lag-1
        acf_x = acf(x_clean, nlags=max_lag, fft=True)
        acf_y = acf(y_clean, nlags=max_lag, fft=True)
        rho1 = acf_x[1] if len(acf_x) > 1 else 0
        rho2 = acf_y[1] if len(acf_y) > 1 else 0
        # Thiebaux and Zwiers (1984) formula
        n = len(x_clean)
        n_eff = n * (1 - rho1 * rho2) / (1 + rho1 * rho2)
        # Ensure reasonable bounds
        n_eff = max(min(n_eff, n), n/10)
        return n_eff
    except:
        return len(x_clean)  # Fallback to original sample size

def analyze_single_track(ds, track_df, track_id, buffer_km=51, max_lag_hours=50, output_dir="output"):
    std_print('INFO', 'ca', f'Processing Correlation Analysis for Track {track_id}')
    # # Split into phases
    # ocean_phase = track_df[track_df['is_land'] == False].copy()
    # land_phase = track_df[track_df['is_land'] == True].copy()
    if len(track_df['track_id'].unique()) == 1:
        # Already filtered to single track
        current_track = track_df.sort_values('time').copy()
    else:
        # Full DataFrame, needs filtering
        current_track = track_df[track_df['track_id'] == track_id].sort_values('time').copy()
    if current_track.empty:
        std_print('ERROR', 'ca', f'track_df is None for track {track_id}')
        traceback.print_exc()
        return None

    landfall_index = -1
    originates_over_land = True # if this is False tp_sm_time_series plot will not generate for land originated MDs

    if 'is_land' in current_track.columns:
        # Find first transition from ocean to land
        is_land_values = current_track['is_land'].values
        for i in range(1, len(is_land_values)):
            if not is_land_values[i-1] and is_land_values[i]:
                landfall_index = i
                break
        # Check if track originates over land
        originates_over_land = is_land_values[0] if len(is_land_values) > 0 else False

    required_vars = ['tp', 'swvl1']
    missing_vars = [var for var in required_vars if var not in ds.data_vars]
    if missing_vars:
        std_print('ERROR', 'ca', 'Missing variables in dataset', missing_vars)
        return None

    ds, time_coord = get_time_coord(ds)
    tp_series = []
    swvl1_series = []
    latent_hf_series = []
    sensible_hf_series = []
    valid_times = []

    for i, row in current_track.iterrows():
        t = pd.to_datetime(row['time'])
        lat, lon = row['latitude'], row['longitude']
        # Use pre-computed is_land if available, otherwise calculate
        subset = extract_spatial_subset(ds, lat, lon, t, buffer_km, time_coord)
        if subset is None:
            std_print('WARN', 'ca', f'Track {track_id}', f'No data at time {t}')
            continue

        if 'tp' in subset.data_vars:
            tp = float(subset['tp'].mean().values)
        else:
            tp = np.nan
        if 'swvl1' in subset.data_vars:
            swvl1 = float(subset['swvl1'].mean().values)
        else:
            swvl1 = np.nan
        if 'avg_slhtf' in subset.data_vars:
            latent_hf = float(subset['avg_slhtf'].mean().values)
        else:
            latent_hf = np.nan
        if 'avg_ishf' in subset.data_vars:
            sensible_hf = float(subset['avg_ishf'].mean().values)
        else:
            sensible_hf = np.nan

        if np.isfinite(tp) and np.isfinite(swvl1):
            tp_series.append(tp)
            swvl1_series.append(swvl1)
            latent_hf_series.append(latent_hf)
            sensible_hf_series.append(sensible_hf)
            valid_times.append(t)

    if len(valid_times) < 4:
        std_print('WARN', 'ca', f'Track {track_id} Insufficient data' , f'only {len(valid_times)} valid points')
        return None

    # std_print('DEBUG', 'ca', f'Track {track_id} :', f'tp_series shape={len(tp_series)}, swvl1_series shape={len(swvl1_series)}')
    # std_print('DEBUG', 'ca', f'Track {track_id} :', f'Valid times={len(valid_times)}')
    # std_print('DEBUG', 'ca', f'Track {track_id} :', f'tp range={np.nanmin(tp_series):.3f} to {np.nanmax(tp_series):.3f}')
    # std_print('DEBUG', 'ca', f'Track {track_id} :', f'sm range={np.nanmin(swvl1_series):.3f} to {np.nanmax(swvl1_series):.3f}')

    # Units conversion
    tp_series = np.array(tp_series) * 1000 # m -> mm
    swvl1_series = np.array(swvl1_series)
    latent_hf_series = np.array(latent_hf_series)
    sensible_hf_series = np.array(sensible_hf_series)

    # Generate plots
    os.makedirs(output_dir, exist_ok=True)
    std_print('INFO', 'ca', f'Generating plots for Track {track_id}')
    try:
        plot_tp_swvl1_time_series(tp_series, swvl1_series, valid_times, landfall_index, track_id, output_dir, originates_over_land)
    except Exception as e:
        std_print('ERROR', 'ca','Failed to create TP-SWVL1 time series plot', str(e))
    try:
        plot_scatter_analysis(ds, track_df, track_id, buffer_km = 100, output_dir = output_dir)
    except Exception as e:
        std_print('ERROR', 'ca','Failed to create TP-SWL1 correlation scatter plot', str(e))
    try:
        plot_lagged_correlation(tp_series, swvl1_series, max_lag_hours, track_id, output_dir)
    except Exception as e:
        std_print('ERROR', 'ca','Failed to TP-SWVL1 lag correlation plot', str(e))
    try:
        if not np.all(np.isnan(latent_hf_series)) or not np.all(np.isnan(sensible_hf_series)):
            plot_swvl1_heat_flux_correlations(swvl1_series, latent_hf_series, sensible_hf_series, valid_times, landfall_index, track_id, output_dir, originates_over_land)
    except Exception as e:
        std_print('ERROR', 'ca', f'Track {track_id} SWVL1-HF plot failed', str(e))

    # Global Correlation
    corr, p_val = pearsonr(tp_series, swvl1_series) if np.std(tp_series) > 0 else (np.nan, np.nan)
    # Phase-based Correlations
    corr_pre, corr_post = np.nan, np.nan
    if landfall_index != -1:
        tp_pre, sm_pre = tp_series[:landfall_index], swvl1_series[:landfall_index]
        tp_post, sm_post = tp_series[landfall_index:], swvl1_series[landfall_index:]
        if len(tp_pre) >= 3 and np.std(tp_pre) > 0 and np.std(sm_pre) > 0:
            corr_pre, _ = pearsonr(tp_pre, sm_pre)
        if len(tp_post) >= 3 and np.std(tp_post) > 0 and np.std(sm_post) > 0:
            corr_post, _ = pearsonr(tp_post, sm_post)

    # Calculate correlations
    correlation_results = {
        'track_id': track_id,
        'correlation': np.nan,
        'p_value': np.nan,
        'corr_pre': np.nan,
        'corr_post': np.nan,
        'mean_tp': np.mean(tp_series),
        'mean_swvl1': np.mean(swvl1_series),
        'n_samples': len(valid_times),
        'landfall_time': valid_times[landfall_index] if landfall_index != -1 else None,
        'landfall_index': landfall_index,
        'originates_over_land': originates_over_land
    }
    if len(tp_series) >= 3 and np.std(tp_series) > 0 and np.std(swvl1_series) > 0:
        try:
            corr, p_val_uncorrected = pearsonr(tp_series, swvl1_series)
            # Calculate effective sample size
            n_eff = effective_sample_size(tp_series, swvl1_series)
            dof_eff = max(1, int(n_eff) - 2)
            # Recalculate p-value with effective DOF
            t_stat = corr * np.sqrt(dof_eff / (1 - corr**2 + 1e-10))
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof_eff))
            p_val_adj = 2 * (1 - stats.t.cdf(abs(t_stat), dof_eff))

            std_print('INFO', 'ca', f"p_original={p_val:.4f}, p_adjusted={p_val_adj:.4f}, n_eff={n_eff:.0f}")
            correlation_results['correlation'] = corr
            correlation_results['p_value'] = p_val
            correlation_results['p_value_uncorrected'] = p_val_uncorrected
            correlation_results['n_effective'] = n_eff
            correlation_results['dof'] = dof_eff
        except Exception as e:
            std_print('ERROR', 'ca', 'Correlation calculation failed', str(e))

    # Global correlation
    if len(tp_series) >= 3 and np.std(tp_series) > 0 and np.std(swvl1_series) > 0:
        try:
            corr, p_val = pearsonr(tp_series, swvl1_series)
            correlation_results['correlation'] = corr
            correlation_results['p_value'] = p_val
        except:
            pass

    # Phase-based correlations
    if landfall_index != -1 and landfall_index > 0:
        # Pre-landfall
        if landfall_index >= 3:
            tp_pre = tp_series[:landfall_index]
            sm_pre = swvl1_series[:landfall_index]
            if np.std(tp_pre) > 0 and np.std(sm_pre) > 0:
                try:
                    corr_pre, _ = pearsonr(tp_pre, sm_pre)
                    correlation_results['corr_pre'] = corr_pre
                except:
                    pass
        # Post-landfall
        if len(tp_series) - landfall_index >= 3:
            tp_post = tp_series[landfall_index:]
            sm_post = swvl1_series[landfall_index:]
            if np.std(tp_post) > 0 and np.std(sm_post) > 0:
                try:
                    corr_post, _ = pearsonr(tp_post, sm_post)
                    correlation_results['corr_post'] = corr_post
                except:
                    pass

    std_print('INFO', 'ca', f'Correlation for Track {track_id} :')
    std_print(None, None, f"r={correlation_results['correlation']:.3f},")
    std_print(None, None, f"p={correlation_results['p_value']:.4f},")
    std_print(None, None, f"n={len(valid_times)}")
    return correlation_results

def plot_tp_swvl1_time_series(tp, sm, valid_times, landfall_index, track_id, output_dir, originates_over_land=True, track_df=None):
    track_id = int(track_id)
    try:
        # Validate inputs
        if not isinstance(tp, np.ndarray):
            tp = np.array(tp)
        if not isinstance(sm, np.ndarray):
            sm = np.array(sm)
        if not isinstance(valid_times, (list, np.ndarray, pd.DatetimeIndex)):
            raise ValueError("valid_times must be datetime-like")

        # Check for sufficient data
        valid_data_mask = ~(np.isnan(tp) | np.isnan(sm))
        valid_count = np.sum(valid_data_mask)

        if valid_count < 2:
            std_print('WARN', 'ca', f'Track {track_id} Insufficient valid data', f'only {valid_count} points')
            return False

        if len(tp) != len(sm) or len(tp) != len(valid_times):
            std_print('WARN', 'ca', f'Track {track_id}', 'Data length mismatch between tp, sm, and times')
            return False

        # Create figure with dual axes
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Determine landfall status text
        if originates_over_land:
            status_text = "Originates over land"
            status_color = 'blue'
        elif landfall_index == -1:
            status_text = "No Landfall"
            status_color = 'gray'
        else:
            status_text = "Landfall"
            status_color = 'red'

        if track_df is not None and 'is_land' in track_df.columns:
            # Use actual track data for most accurate shading
            is_land_array = track_df['is_land'].values
            std_print('DEBUG', 'ca', f'Track {track_id}', 'Using actual track is_land data for shading')
        else:
            # Infer from landfall info
            is_land_array = np.zeros(len(valid_times), dtype=bool)
            if originates_over_land:
                is_land_array[:] = True
                std_print('DEBUG', 'ca', f'Track {track_id}', 'Land-originated track - entire track marked as land')
            elif landfall_index > 0:
                is_land_array[landfall_index:] = True
                std_print('DEBUG', 'ca', f'Track {track_id}', f'Ocean-originated with landfall at index {landfall_index}')
            else:
                std_print('DEBUG', 'ca', f'Track {track_id}', 'Ocean-originated, no landfall - entire track marked as ocean')

        # Plot precipitation on primary axis
        color_tp = 'tab:blue'
        ax1.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Total Precipitation (tp) (mm/hr)', color=color_tp, fontsize=11, fontweight='bold')
        line1 = ax1.plot(
            valid_times,
            tp,
            '-o',
            color=color_tp,
            linewidth=2.5,
            markersize=5,
            label='Precipitation',
            alpha=0.8,
            zorder=3
        )
        ax1.tick_params(axis='y', labelcolor=color_tp)
        ax1.grid(True, alpha=0.4, linestyle='--', zorder=0)

        # land/ocean shading
        add_land_ocean_shading(ax1, pd.to_datetime(valid_times), is_land_array, land_color='wheat', ocean_color='lightblue')

        # Landfall vertical line (only if actual landfall occurred)
        if landfall_index != -1 and landfall_index < len(valid_times) and not originates_over_land:
            landfall_time = valid_times[landfall_index]
            ax1.axvline(
                x=landfall_time,
                color=status_color,
                linestyle='--',
                linewidth=2.5,
                alpha=0.8,
                label=status_text,
                zorder=4
            )

        # Soil moisture on secondary axis
        ax2 = ax1.twinx()
        color_swvl1 = 'tab:brown'
        ax2.set_ylabel('Soil Water Volumatric Layer 1 (swvl1) (m³/m³)', color=color_swvl1, fontsize=11, fontweight='bold')
        line2 = ax2.plot(
            valid_times,
            sm,
            's-',
            color=color_swvl1,
            linewidth=2.5,
            markersize=5,
            label='Soil Moisture',
            alpha=0.8,
            zorder=3
        )
        ax2.tick_params(axis='y', labelcolor=color_swvl1)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]

        # landfall line to legend if present
        if landfall_index != -1 and landfall_index < len(valid_times) and not originates_over_land:
            lines.append(plt.Line2D([0], [0], color=status_color, linestyle='--', linewidth=2.5))
            labels.append(status_text)

        # land/ocean patches to legend
        lines.append(Patch(facecolor='lightblue', alpha=0.5, label='Ocean Phase'))
        lines.append(Patch(facecolor='wheat', alpha=0.5, label='Land Phase'))
        labels += ['Ocean Phase', 'Land Phase']

        ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.95, edgecolor='black')

        # Title based on track characteristics
        title = f'Track {track_id}: Precipitation & Soil Moisture Evolution'
        if originates_over_land:
            title += '\n(Land-Originated MD)'
        elif landfall_index != -1:
            title += '\n(Ocean-Originated, Landfall Detected)'
        else:
            title += '\n(Ocean-Originated, No Landfall)'
        ax1.set_title(title, fontsize=13, fontweight='bold', pad=12)

        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()

        # Save
        track_dir = os.path.join(output_dir, "time_series")
        os.makedirs(track_dir, exist_ok=True)
        plot_path = os.path.join(track_dir, f"track_{track_id}_TP-SWVL1_time_series.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        std_print('SAVED', 'ca', 'TP-SWVL1 time series', plot_path)
        return True
    except Exception as e:
        std_print('ERROR', 'ca', f'Track {track_id} TP-SWVL1 plot failed', str(e))
        traceback.print_exc()
        return False

# NOTE: now extract_regional_data function included in spatial_util
# def extract_regional_data(ds, track_df, track_id, variable, buffer_km = 100):
#     track_id = int(track_id)
#     track = track_df[track_df['track_id'] == track_id].sort_values('time')
#     values = []
#     valid_times = []
#     for _, row in track.iterrows():
#         time_coord = "time" if "time" in ds.coords else "valid_time"
#         lat_c = row['latitude']
#         lon_c = row['longitude']
#         t = pd.to_datetime(row['time'])
#         subset = extract_spatial_subset(ds, lat_c, lon_c, t, buffer_km=buffer_km, time_coord=time_coord)
#         if variable in subset.data_vars:
#             val = float(subset[variable].mean().values)
#             if not np.isnan(val):
#                 values.append(val)
#                 valid_times.append(t)
#     return np.array(valid_times), np.array(values)

# Compute comprehensive correlation between TP and SWVL1 for a single track
def compute_single_track_correlation(ds, track_df, track_id, buffer_km = 100):
    """
    - Pearson correlation (linear relationship)
    - Spearman correlation (monotonic relationship)
    - Linear regression (slope, intercept, R²)
    """
    track_id = int(track_id)

    # Find landfall
    if 'is_land' not in track_df.columns:
        track_df['is_land'] = track_df.apply(lambda row: is_point_land(row['latitude'], row['longitude']), axis=1)
    landfall_time = find_landfall_time(track_df)

    # Extract time series
    tp_times, tp_vals = extract_regional_data(ds, track_df, track_id, 'tp', buffer_km)
    sm_times, sm_vals = extract_regional_data(ds, track_df, track_id, 'swvl1', buffer_km)
    common_times = np.intersect1d(tp_times, sm_times)
    if len(common_times) < 3:
        return None

    # Align time series to common times
    tp_idx = np.isin(tp_times, common_times)
    sm_idx = np.isin(sm_times, common_times)

    tp_aligned = tp_vals[tp_idx] * 1000  # Convert m to mm
    sm_aligned = sm_vals[sm_idx]
    if len(tp_aligned) < 3:
        return None

    # check for zero variance
    if np.std(tp_aligned) == 0 or np.std(sm_aligned) == 0:
        std_print('WARN', 'ca', f'Track {track_id}', 'Zero variance in one or both variables')
        return None

    # Pearson correlation (linear)
    pearson_r, pearson_p = pearsonr(tp_aligned, sm_aligned)
    # Spearman correlation (monotonic)
    spearman_r, spearman_p = spearmanr(tp_aligned, sm_aligned)
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(tp_aligned, sm_aligned)
    r_squared_val = float(r_value) ** 2
    return {
        'track_id': track_id,
        'n_samples': len(tp_aligned),
        'mean_tp': float(np.mean(tp_aligned)),
        'mean_swvl1': float(np.mean(sm_aligned)),
        'std_tp_mm': float(np.std(tp_aligned)),
        'std_swvl1': float(np.std(sm_aligned)),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'regression_slope': float(slope),
        'regression_intercept': float(intercept),
        'r_squared': r_squared_val,
        'times': common_times,
        'tp_values': tp_aligned,
        'swvl1_values': sm_aligned
    }

# Compute correlations for all tracks.
def analyze_all_tracks(ds, full_track_df, buffer_km=100, output_dir="output"):
    std_print('INFO', 'ca', 'Analyzing all tracks for soil moisture patterns')
    unique_tracks = sorted(full_track_df['track_id'].unique())
    all_results = []
    for track_id in unique_tracks:
        track_df = full_track_df[full_track_df['track_id'] == track_id].copy()
        result = compute_single_track_correlation(ds, track_df, track_id, buffer_km)
        if result:
            all_results.append(result)
    if not all_results:
        std_print('WARN', 'ca', 'All Tracks', 'No results from individual track analysis')
        return None
    results_df = pd.DataFrame(all_results)

    # Summary statistics
    summary = {
        'n_tracks': len(results_df),
        'mean_correlation': results_df['pearson_r'].mean(),
        'std_correlation': results_df['pearson_r'].std(),
        'tracks_with_positive_corr': (results_df['pearson_r'] > 0).sum(),
        'tracks_with_significant_corr': (results_df['pearson_p'] < 0.05).sum(),
        'mean_tp': results_df['mean_tp'].mean(),
        'mean_swvl1': results_df['mean_swvl1'].mean(),
    }
    std_print('INFO', 'ca', f"Soil Moisture Analysis across  {summary['n_tracks']} Tracks :")
    std_print(None, None, 'Mean Correlation (TP vs SM)', f"{summary['mean_correlation']:.3f} ± {summary['std_correlation']:.3f}")
    std_print(None, None, 'Positive Correlations', f"{summary['tracks_with_positive_corr']}/{summary['n_tracks']}")
    std_print(None, None, 'Significant Correlations (p < 0.05)', f"{summary['tracks_with_significant_corr']}/{summary['n_tracks']}")
    std_print(None, None, 'Mean TP', f"{summary['mean_tp']:.2f} mm/hr")
    std_print(None, None, 'Mean SM', f"{summary['mean_swvl1']:.3f} m³/m³")
    # Export results
    export_results_to_csv(results_df, output_dir, "sm_analysis", 'sm')
    return {
        'summary': summary,
        'detailed_results': results_df
    }

# FIX: use  export_results_to_csv
def export_results(results_df, output_dir = 'output'):
    """Export correlation results to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    export_cols = ['track_id', 'n_samples', 'mean_tp', 'mean_swvl1', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 'regression_slope', 'r_squared']

    tp_swvl1_correlations_path = os.path.join(output_dir, 'tp_swvl1_correlations.csv')
    available_cols = [col for col in export_cols if col in results_df.columns]
    if not available_cols:
        std_print('ERROR', 'ca', 'No valid columns to export')
        return
    results_df[available_cols].to_csv(tp_swvl1_correlations_path, index=False)
    std_print('EXPORT', 'ca', 'Correlation results CSV', tp_swvl1_correlations_path)

def plot_scatter_analysis(ds, track_df, track_id, buffer_km = 100, output_dir = 'output'):
    result = compute_single_track_correlation(ds, track_df, track_id, buffer_km)
    if result is None:
        std_print('WARN', 'ca', f'Track {track_id}', 'Insufficient flux data')
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    times = result['times']
    tp_vals = result['tp_values']
    sm_vals = result['swvl1_values']
    common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)

    ax.scatter(
        tp_vals,
        sm_vals,
        alpha=0.6,
        s=100,
        color='purple',
        edgecolors='black',
        linewidth=1,
        zorder=3
    )
    # Regression line
    if result['n_samples'] > 2:
        x_line = np.array([tp_vals.min(), tp_vals.max()])
        y_line = result['regression_slope'] * x_line + result['regression_intercept']
        ax.plot(
            x_line,
            y_line,
            'r--',
            linewidth=2.5,
            alpha=0.7,
            label=f'Linear Fit (r²={result["r_squared"]:.4f})'
        )

    # Statistics box
    stats_text = (
        f"Pearson r:  {result['pearson_r']:.4f}\n"
        f"Spearman ρ: {result['spearman_r']:.4f}\n"
        f"P-value:    {result['pearson_p']:.4e}\n"
        f"N samples:  {result['n_samples']}\n"
        f"Slope:      {result['regression_slope']:.4f}\n"
        f"Mean TP:    {result['mean_tp']:.2f} mm/hr\n"
        f"Mean SM:    {result['mean_swvl1']:.2f} m³/m³"
    )
    ax.text(
        1.02,
        0.85,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        family='monospace',
        bbox=common_bbox
    )

    ax.set_xlabel('Total Precipitation (tp) (mm/hr)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Soil Water Volumatric Level-1 (swvl1) (m³/m³)', fontsize=10, fontweight='bold')
    ax.set_title(f'Track {track_id}: tp - swvl1 Correlation', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(fontsize=8, loc='best', framealpha=0.7)
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "correlation_scatter_plots")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_TP-SWVL1_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ca', 'TP-SWVL1 Correlation Scatter plot', plot_path)

def plot_lagged_correlation(tp, sm, max_lag_hours, track_id, output_dir):
    """
    Plot lag correlation showing temporal sequencing between TP and SWVL1.
    Negative lag: TP leads SWVL1 (precipitation causes soil moisture increase)
    Positive lag: SWVL1 leads TP (antecedent moisture affects rainfall)
    """
    lags, correlations_lag = lag_correlation(tp, sm, max_lag=max_lag_hours)

    # Ensure lags and correlations have same length
    if len(lags) != len(correlations_lag):
        std_print('WARN', 'ca', f'Mismatch between lags ({len(lags)}) and correlations', len(correlations_lag))
        # Truncate to use the shorter length
        min_len = min(len(lags), len(correlations_lag))
        lags = lags[:min_len]
        correlations_lag = correlations_lag[:min_len]

    # Filter out NaN values
    valid_indices = [i for i, val in enumerate(correlations_lag) if not np.isnan(val)]
    if len(valid_indices) == 0:
        std_print('WARN', 'ca', f'Track {track_id}', 'No valid correlation values for lag plot')
        return

    lags = [lags[i] for i in valid_indices]
    correlations_lag = [correlations_lag[i] for i in valid_indices]

    max_corr_idx = np.argmax(np.abs(correlations_lag))
    max_corr_lag = lags[max_corr_idx]
    max_corr_value = correlations_lag[max_corr_idx]
    if max_corr_lag < 0:
        lag_interpretation = f"TP LEADS SWVL1 by {abs(max_corr_lag)}h\n(Infiltration lag)"
    elif max_corr_lag > 0:
        lag_interpretation = f"SWVL1 LEADS TP by {max_corr_lag}h\n(Antecedent moisture)"
    else:
        lag_interpretation = "Simultaneous\n(Immediate feedback)"

    fig, ax = plt.subplots(figsize=(12, 6))
    if len(lags) == len(correlations_lag):
        ax.fill_between(lags, correlations_lag, alpha=0.2, color='steelblue')
    else:
        std_print('WARN', 'ca', f'Track {track_id} Lag mismatch', 'lags and r_lag lengths differ')

    common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)
    # Plot correlation
    ax.plot(
        lags,
        correlations_lag,
        'o-',
        lw=2.5,
        markersize=4,
        color='blue',
        zorder=3
    )
    ax.fill_between(lags, correlations_lag, alpha=0.2, color='steelblue')
    # Reference lines
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6, label='Zero lag')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(
        1.02,
        0.92,
        lag_interpretation,
        transform=ax.transAxes,
        fontsize=10,
        va='bottom',
        ha='right',
        bbox=common_bbox
    )
    ax.set_xlabel("Lag (hours)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Correlation (r)", fontsize=10, fontweight='bold')
    ax.set_title(f"Track {track_id}: Lagged Precipitation-Soil Moisture Correlation", fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.5, linestyle='--')
    # ax.set_ylim([-1, 1])
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "lag_correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{int(track_id)}_lag_corr.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ca', 'Lag Ccrelation plot', plot_path)

def plot_swvl1_heat_flux_correlations(swvl1, latent_hf, sensible_hf, valid_times, landfall_index, track_id, output_dir, originates_over_land=True):
    """
    Plot SWVL1 correlations with surface heat fluxes (Latent, Sensible, Total).
    Relationships:
    - SWVL1 <-> Latent HF: Soil moisture affects evaporative cooling
    - SWVL1 <-> Sensible HF: Dry soil increases sensible heating
    - SWVL1 <-> Total HF: Combined surface energy budget impact
    """
    # Calculate total heat flux
    total_hf = latent_hf + sensible_hf
    
    # Remove NaN values for correlation calculation
    valid_mask = ~(np.isnan(swvl1) | np.isnan(latent_hf) | np.isnan(sensible_hf) | np.isnan(total_hf))
    valid_count = np.sum(valid_mask)
    
    if valid_count < 5:
        std_print('WARN', 'ca', f'Track {track_id}', f'Insufficient heat flux data ({valid_count} valid points)')
        return False
    
    # Calculate correlations
    try:
        r_latent, p_latent = pearsonr(swvl1[valid_mask], latent_hf[valid_mask])
        r_sensible, p_sensible = pearsonr(swvl1[valid_mask], sensible_hf[valid_mask])
        r_total, p_total = pearsonr(swvl1[valid_mask], total_hf[valid_mask])
    except:
        std_print('WARN', 'ca', f'Track {track_id}', 'Could not compute heat flux correlations')
        return False
    
    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Determine landfall status text
    if originates_over_land:
        status_text = "Land-Originated"
        status_color = 'blue'
    elif landfall_index == -1:
        status_text = "No Landfall"
        status_color = 'gray'
    else:
        status_text = "Landfall"
        status_color = 'darkred'
    
    # Panel 1: SWVL1 vs Latent HF
    ax = axes[0]
    ax.scatter(swvl1[valid_mask], latent_hf[valid_mask], alpha=0.6, s=80, color='green', edgecolors='black', linewidth=0.5)
    
    # Regression line
    if np.std(swvl1[valid_mask]) > 0:
        z = np.polyfit(swvl1[valid_mask], latent_hf[valid_mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(swvl1[valid_mask].min(), swvl1[valid_mask].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Soil Moisture (m³/m³)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Latent Heat Flux (W/m²)', fontsize=10, fontweight='bold')
    ax.set_title(f'SWVL1 ↔ Latent HF\nr={r_latent:.3f}, p={p_latent:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # Panel 2: SWVL1 vs Sensible HF
    ax = axes[1]
    ax.scatter(swvl1[valid_mask], sensible_hf[valid_mask], alpha=0.6, s=80, color='orange', edgecolors='black', linewidth=0.5)
    
    # Regression line
    if np.std(swvl1[valid_mask]) > 0:
        z = np.polyfit(swvl1[valid_mask], sensible_hf[valid_mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(swvl1[valid_mask].min(), swvl1[valid_mask].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Soil Moisture (m³/m³)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Sensible Heat Flux (W/m²)', fontsize=10, fontweight='bold')
    ax.set_title(f'SWVL1 ↔ Sensible HF\nr={r_sensible:.3f}, p={p_sensible:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # Panel 3: SWVL1 vs Total HF
    ax = axes[2]
    ax.scatter(swvl1[valid_mask], total_hf[valid_mask], alpha=0.6, s=80, color='purple', edgecolors='black', linewidth=0.5)
    
    # Regression line
    if np.std(swvl1[valid_mask]) > 0:
        z = np.polyfit(swvl1[valid_mask], total_hf[valid_mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(swvl1[valid_mask].min(), swvl1[valid_mask].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Soil Moisture (m³/m³)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Total Heat Flux (W/m²)', fontsize=10, fontweight='bold')
    ax.set_title(f'SWVL1 ↔ Total HF\nr={r_total:.3f}, p={p_total:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    
    fig.suptitle(f'Track {track_id}: Soil Moisture ↔ Surface Heat Flux Correlations ({status_text})', 
                 fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save
    track_dir = os.path.join(output_dir, "heat_flux_soil_moisture_correlations")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{int(track_id)}_SWVL1-HF_correlations.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ca', 'SWVL1-HF Correlations plot', plot_path)
    return True

def analyze_flux_correlations(ds, track_df, track_id, buffer_km=100):
    ds, time_coord = get_time_coord(ds)
    ds_times = pd.to_datetime(ds[time_coord].values)

    time_series = []
    landfall_time = find_landfall_time(track_df)
    if landfall_time:
       std_print('INFO', 'ca', f'Track {track_id} Landfall detected', landfall_time.strftime('%Y-%m-%d %H:%M'))
    else:
       std_print('WARN', 'ca', f'Track {track_id} No landfall detected')
    if len(time_series) < 5:
       std_print('WARN', 'ca', f'Track {track_id} Insufficient data')
       return None
    df = pd.DataFrame(time_series)

    # correlations
    correlations = {}
    for v1 in ["tp", "swvl1", "total_hf"]:
        for v2 in ["tp", "swvl1", "total_hf"]:
            if v1 != v2:
                mask = df[[v1, v2]].notna().all(axis=1)
                if mask.sum() > 3:
                    r, p = pearsonr(df.loc[mask, v1], df.loc[mask, v2])
                    correlations[f"{v1}_vs_{v2}"] = {"r": r, "p": p, "n": mask.sum()}

    # before / after landfall split
    results = {
        "correlations": correlations,
        "time_series": df,
        "track_id": track_id,
    }

    if "is_land" in track_df.columns:
        land_mask = track_df["is_land"].diff().eq(1)
        if land_mask.any():
            lf_time = track_df.loc[land_mask.idxmax(), "time"]
            pre = df[df["time"] < lf_time]
            post = df[df["time"] >= lf_time]
            results["pre_landfall"] = {
                "mean_tp": pre["tp"].mean(),
                "mean_swvl1": pre["swvl1"].mean(),
                "mean_total_hf": pre["total_hf"].mean(),
            }
            results["post_landfall"] = {
                "mean_tp": post["tp"].mean(),
                "mean_swvl1": post["swvl1"].mean(),
                "mean_total_hf": post["total_hf"].mean(),
            }
    if correlations:
        std_print('INFO', 'ca', f'Flux correlation for Track {track_id} :')
        for k, v in correlations.items():
            std_print(None, None, f"{k}: r={v['r']:.3f}, p={v['p']:.4f}, n={v['n']}")
    return results

# def plot_flux_analysis(results, output_dir, track_id):
#     if results is None or len(results) < 4:  # Require minimum points for a meaningful plot
#        std_print('WARN', 'ca', f'Track {track_id}', 'Insufficient data')
#        return
#     if not results or 'time_series' not in results:
#        std_print('WARN', 'ca', f'Track {track_id}' 'No time series data')
#        return
#
#     df = results['time_series']
#     # Validate DataFrame
#     if not isinstance(df, pd.DataFrame) or df.empty:
#        std_print('WARN', 'ca', f'Track {track_id}', 'Invalid or empty DataFrame')
#        return
#
#     if df.empty or len(df) < 3:
#        std_print('WARN', 'ca', f'Track {track_id}', f'Insufficient data {len(df)} points')
#        return
#
#     # Check for required columns
#     required_cols = ['time', 'tp', 'swvl1']
#     missing_cols = [col for col in required_cols if col not in df.columns]
#     if missing_cols:
#        std_print('WARN', 'ca', f'Track {track_id} Missing required columns', missing_cols)
#        return
#     # Ensure is_land column exists
#     if 'is_land' not in df.columns:
#         df['is_land'] = False  # Default to ocean if not present
#
#     track_dir = os.path.join(output_dir, f"track_{track_id}_analysis")
#     os.makedirs(track_dir, exist_ok=True)
#
#     # PLOT 1: Precipitation & Soil Moisture
#     try:
#         ## Precip (Primary Y)
#         plt.figure(figsize=(10, 5))
#         ax1 = plt.gca()
#         p1, = ax1.plot(
#             df['time'],
#             df['tp'],
#             'b-',
#             linewidth=2
#         )
#         ax1.set_ylabel('Total Precipitation (mm/hr)', fontweight='bold')
#         ax1.tick_params(axis='y', labelcolor='b')
#         ## Soil Moisture (Secondary Y)
#         ax1_twin = ax1.twinx()
#         p2, = ax1_twin.plot(
#             df['time'],
#             df['swvl1'],
#             'r-',
#             linewidth=2
#         )
#         ax1_twin.set_ylabel('Soil Moisture (m³/m³)', fontweight='bold')
#         ax1_twin.tick_params(axis='y', labelcolor='r')
#
#         add_land_ocean_shading(ax1, df['time'], df['is_land'])
#
#         plt.title(f"Track {track_id}: Total Precipitation & Soil Moisture", fontsize=13, fontweight='bold')
#
#         # Get legend items from shading (land/ocean patches) and the data line handles with proper labels
#         handles, labels = ax1.get_legend_handles_labels()
#         all_handles = handles + [p1, p2]
#         all_labels = labels + ['Total Precipitation', 'Soil Moisture']
#         ax1.legend(all_handles, all_labels, loc='best', framealpha=0.7)
#         ax1.grid(True, alpha=0.5, linestyle='--')
#
#         plot1_path = os.path.join(track_dir, f"track_{track_id}_TP-SWVL1.png")
#         plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         std_print('SAVED', 'ca', 'Precipitation & Soil Moisture plot', plot1_path)
#     except Exception as e:
#         std_print('ERROR', 'ca', f'Track {track_id} Failed to create TP & SWVL1 plot:', str(e))
#         traceback.print_exc()
#
#     # PLOT 2: Heat Fluxes (The one with the Land/Ocean Legend)
#     try:
#         fig = plt.figure(figsize=(10, 5))
#         ax2 = fig.gca()
#         ax2.plot(
#             df['time'],
#             df['latent_hf'],
#             'blue',
#             label='Latent Heat Flux',
#             linewidth=2
#         )
#         ax2.plot(
#             df['time'],
#             df['sensible_hf'],
#             'red',
#             label='Sensible Heat Flux',
#             linewidth=2
#         )
#         ax2.plot(
#             df['time'],
#             df['total_hf'],
#             'purple',
#             label='Total Heat Flux',
#             linewidth=2
#         )
#
#         add_land_ocean_shading(ax2, df['time'], df['is_land'])
#         ax2.set_ylabel('Flux (W/m²)', fontweight='bold')
#         ax2.set_xlabel('Time', fontweight='bold')
#         ax2.legend(loc='upper left', framealpha=0.7)
#         ax2.grid(True, alpha=0.5, linestyle='--')
#
#         plt.title(f"Track {track_id}: Surface Heat Flux Transition", fontsize=13, fontweight='bold')
#         plot2_path = os.path.join(track_dir, f"track_{track_id}_heat_fluxes_trasnsition.png")
#         plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         std_print('SAVED', 'ca', 'Surface Heat Flux Transitions plot', plot2_path)
#     except Exception as e:
#         std_print('ERROR', 'ca', f'Track {track_id} Failed to create heat flux plot', str(e))
#         traceback.print_exc()
#
#     # PLOT 3: Scatter Plot (Flux vs. Soil Water Volumatric Level-1)
#     try:
#         fig = plt.figure(figsize=(10, 5))
#         if len(df) > 2:
#             scatter = plt.scatter(
#                 df['swvl1'],
#                 df['total_hf'],
#                 c=df['tp'],
#                 cmap='viridis',
#                 s=70,
#                 alpha=0.8,
#                 edgecolors='k',
#                 label='All Points'
#             )
#             # Calculate Correlation
#             try:
#                 r, p = pearsonr(df['swvl1'], df['total_hf'])
#                 corr_text = f"Pearson r = {r:.3f} (p={p:.4f})"
#             except:
#                 corr_text = "Unable to calculate correlation"
#             # Differentiate land vs ocean points with markers if data exists
#             if 'is_land' in df.columns and df['is_land'].any():
#                 land_pts = df[df['is_land'] == True]
#                 if not land_pts.empty:
#                     plt.scatter(
#                         land_pts['swvl1'],
#                         land_pts['total_hf'],
#                         c='orange',
#                         s=100,
#                         marker='s',
#                         alpha=0.6,
#                         edgecolors='k',
#                         linewidth=1.5,
#                         label='Land Points',
#                         zorder=5
#                     )
#             plt.title(
#                 f"Track {track_id}: SWVL1 vs Total Flux\n{corr_text}",
#                 fontsize=13,
#                 fontweight='bold'
#             )
#             cbar = plt.colorbar(scatter)
#             cbar.set_label('Precipitation (mm/hr)', fontsize=10)
#         else:
#             # Fallback if insufficient data
#             plt.text(
#                 0.5, 0.5,
#                 f"Insufficient Data for Track {track_id}\n({len(df)} points)",
#                 ha='center',
#                 va='center',
#                 fontsize=12,
#                 transform=plt.gca().transAxes
#             )
#             plt.title(f"Track {track_id}: SWVL1 vs Flux (Insufficient Data)")
#
#         plt.xlabel('Soil Moisture (m³/m³)', fontweight='bold', fontsize=11)
#         plt.ylabel('Total Heat Flux (W/m²)', fontweight='bold', fontsize=11)
#         plt.legend(loc='best', framealpha=0.7)
#         plt.grid(True, alpha=0.5, linestyle='--')
#
#         plot3_path = os.path.join(track_dir, f"track_{track_id}_SWVL1-THF_correlation_scatter.png")
#         plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         std_print('SAVED', 'ca', 'SWVL1 vs Flux Correlation plot', plot3_path)
#     except Exception as e:
#         std_print('ERROR', 'ca', f'Track {track_id} Failed to create Scatter plot', str(e))
#         traceback.print_exc()
def plot_flux_analysis(results, output_dir, track_id):
    try:
        # Validate input
        if results is None or not isinstance(results, dict):
            std_print('WARN', 'ca', f'Track {track_id}', 'Invalid or None results dictionary')
            return False
        if 'time_series' not in results:
            std_print('WARN', 'ca', f'Track {track_id}', 'No time_series key in results')
            return False

        df = results['time_series']
        # Validate DataFrame
        if not isinstance(df, pd.DataFrame) or df.empty:
            std_print('WARN', 'ca', f'Track {track_id}', 'Empty or invalid DataFrame')
            return False
        if len(df) < 3:
            std_print('WARN', 'ca', f'Track {track_id} Insufficient data points:', f'{len(df)} < 3')
            return False

        # Ensure required columns exist
        required_cols = ['time', 'tp', 'swvl1']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            std_print('WARN', 'ca', f'Track {track_id} Missing columns', missing_cols)
            return False

        # Ensure is_land column
        if 'is_land' not in df.columns:
            std_print('DEBUG', 'ca', f'Track {track_id}', 'Creating is_land column')
            df['is_land'] = False

        # Convert time to datetime
        try:
            df['time'] = pd.to_datetime(df['time'])
        except Exception as e:
            std_print('ERROR', 'ca', f'Track {track_id}  Time conversion failed', str(e))
            return False

        track_dir = os.path.join(output_dir, f"track_{track_id}_analysis")
        os.makedirs(track_dir, exist_ok=True)

        success_plots = 0
        # Precipitation & Soil Moisture
        try:
            if 'tp' not in df.columns or 'swvl1' not in df.columns:
                raise ValueError("Missing tp or swvl1 columns")

            fig, ax1 = plt.subplots(figsize=(14, 7))
            # Precipitation (primary Y)
            color_tp = 'tab:blue'
            ax1.set_xlabel('Time', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Total Precipitation (mm/hr)', color=color_tp, fontsize=11, fontweight='bold')
            p1, = ax1.plot(
                df['time'],
                df['tp'],
                '-o',
                color=color_tp,
                linewidth=2.5,
                markersize=4,
                label='Precipitation',
                alpha=0.8,
                zorder=3
            )
            ax1.tick_params(axis='y', labelcolor=color_tp)

            # Soil Moisture (secondary Y)
            ax2 = ax1.twinx()
            color_swvl1 = 'tab:brown'
            ax2.set_ylabel('Soil Moisture (m³/m³)', color=color_swvl1, fontsize=11, fontweight='bold')
            p2, = ax2.plot(
                df['time'],
                df['swvl1'],
                's-',
                color=color_swvl1,
                linewidth=2.5,
                markersize=4,
                label='Soil Moisture',
                alpha=0.8,
                zorder=3
            )
            ax2.tick_params(axis='y', labelcolor=color_swvl1)

            # Add land/ocean shading
            add_land_ocean_shading(ax1, df['time'], df['is_land'].values, land_color='wheat', ocean_color='lightblue')

            # Legend
            lines = [p1, p2]
            labels = ['Precipitation', 'Soil Moisture']
            lines.append(Patch(facecolor='lightblue', alpha=0.5, label='Ocean'))
            lines.append(Patch(facecolor='wheat', alpha=0.5, label='Land'))
            labels += ['Ocean', 'Land']
            ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.95)
            ax1.grid(True, alpha=0.4, linestyle='--', zorder=0)
            ax1.set_title(f'Track {track_id}: Precipitation & Soil Moisture', fontsize=13, fontweight='bold', pad=10)

            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.tight_layout()

            plot1_path = os.path.join(track_dir, f"track_{track_id}_TP-SWVL1.png")
            plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            std_print('SAVED', 'ca', 'TP-SWVL1 plot', plot1_path)
            success_plots += 1
        except Exception as e:
            std_print('ERROR', 'ca', f'Track {track_id} TP-SWVL1 plot failed', str(e))

        # PLOT 2: Heat Fluxes with Ocean/Land Shading
        try:
            if 'latent_hf' not in df.columns or 'sensible_hf' not in df.columns or 'total_hf' not in df.columns:
                raise ValueError("Missing flux columns")

            fig, ax = plt.subplots(figsize=(14, 7))
            # Plot fluxes
            ax.plot(df['time'], df['latent_hf'], '-o', color='green', linewidth=2.5, markersize=4, label='Latent Heat Flux', alpha=0.8, zorder=3)
            ax.plot(df['time'], df['sensible_hf'], '-s', color='orange', linewidth=2.5, markersize=4, label='Sensible Heat Flux', alpha=0.8, zorder=3)
            ax.plot(df['time'], df['total_hf'], '-D', color='purple', linewidth=2.5, markersize=4, label='Total Heat Flux', alpha=0.8, zorder=3)

            # Add land/ocean shading
            add_land_ocean_shading(ax, df['time'], df['is_land'].values, land_color='wheat', ocean_color='lightblue')

            ax.set_ylabel('Heat Flux (W/m²)', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time', fontsize=11, fontweight='bold')

            lines = ax.get_lines()
            lines.append(Patch(facecolor='lightblue', alpha=0.5, label='Ocean'))
            lines.append(Patch(facecolor='wheat', alpha=0.5, label='Land'))
            labels = [l.get_label() for l in ax.get_lines()] + ['Ocean', 'Land']
            ax.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.95)

            ax.grid(True, alpha=0.4, linestyle='--', zorder=0)
            ax.set_title(f'Track {track_id}: Surface Heat Flux Transition', fontsize=13, fontweight='bold', pad=10)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.tight_layout()

            plot2_path = os.path.join(track_dir, f"track_{track_id}_heat_fluxes_transition.png")
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            std_print('SAVED', 'ca', 'Heat flux plot', plot2_path)
            success_plots += 1
        except Exception as e:
            std_print('WARN', 'ca', f'Track {track_id} Heat flux plot failed', str(e))

        # PLOT 3: Scatter Plot (SM vs Total HF)
        try:
            if 'swvl1' not in df.columns or 'total_hf' not in df.columns:
                raise ValueError("Missing scatter plot columns")

            fig = plt.figure(figsize=(12, 8))
            # Create scatter with TP as colorbar
            if 'tp' in df.columns and len(df) > 1:
                scatter = plt.scatter(
                    df['swvl1'],
                    df['total_hf'],
                    c=df['tp'],
                    cmap='viridis',
                    s=100,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )
                cbar = plt.colorbar(scatter)
                cbar.set_label('Precipitation (mm/hr)', fontsize=10, fontweight='bold')
            else:
                plt.scatter(df['swvl1'], df['total_hf'], s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

            # Differentiate land points
            if df['is_land'].any():
                land_pts = df[df['is_land'] == True]
                if not land_pts.empty:
                    plt.scatter(
                        land_pts['swvl1'],
                        land_pts['total_hf'],
                        c='orange',
                        s=150,
                        marker='s',
                        alpha=0.5,
                        edgecolors='darkred',
                        linewidth=1.5,
                        label='Land',
                        zorder=5
                    )

            try:
                r, p = pearsonr(df['swvl1'].dropna(), df['total_hf'].dropna())
                corr_text = f'Pearson r = {r:.3f} (p={p:.4f})'
            except:
                corr_text = 'Correlation unavailable'

            plt.title(f'Track {track_id}: Soil Water Volumatric Level 1 vs Total Heat Flux\n{corr_text}', fontsize=13, fontweight='bold', pad=10)
            plt.xlabel('Soil Water Volumatric Level-1 (m³/m³)', fontsize=10, fontweight='bold')
            plt.ylabel('Total Heat Flux (W/m²)', fontsize=10, fontweight='bold')
            plt.legend(fontsize=10, loc='best', framealpha=0.95)
            plt.grid(True, alpha=0.4, linestyle='--')
            plt.tight_layout()

            plot3_path = os.path.join(track_dir, f"track_{track_id}_SWVL1-THF-correlation_scatter.png")
            plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            std_print('SAVED', 'ca', 'SWVL1 - Totla HF Scatter plot', plot3_path)
            success_plots += 1
        except Exception as e:
            std_print('ERROR', 'ca', f'Track {track_id} SWVL1 - Total HF Scatter plot failed', str(e))

        if success_plots == 0:
            std_print('WARN', 'ca', f'Track {track_id}', 'No plots generated successfully')
            return False

        std_print('INFO', 'ca', f'Track {track_id}', f'Generated {success_plots}/3 flux analysis plots')
        return True
    except Exception as e:
        std_print('ERROR', 'ca', f'Track {track_id} flux analysis', str(e))
        traceback.print_exc()
        return False
