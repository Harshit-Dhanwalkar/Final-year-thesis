# TODO: add TP-HF corrleation plot (done TP-HF lag correaltion)

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
    add_land_ocean_shading,
    get_default_buffer_km,
    extract_spatial_subset,
    extract_regional_data,
    extract_and_average_variable,
    get_time_coord,
    export_results_to_csv
)

# Helper functions
def lag_correlation(x, y, max_lag=48):
    """Computes correlation between x and y with temporal offsets"""
    lags = range(-max_lag, max_lag+1)
    correlations = []
    for lag in lags:
        if lag < 0:
            x_slice, y_slice = x[:lag], y[-lag:]
        elif lag > 0:
            x_slice, y_slice = x[lag:], y[:-lag]
        else:
            x_slice, y_slice = x, y
        if len(x_slice) < 5 or np.std(x_slice) == 0 or np.std(y_slice) == 0:
            correlations.append(np.nan)
        else:
            r, _ = pearsonr(x_slice, y_slice)
            correlations.append(r)
    return list(lags), correlations

def compute_autocorrelation(data, nlags=96, fft=True):
    # Remove NaN values
    data_clean = np.array(data)
    data_clean = data_clean[~np.isnan(data_clean)]
    
    if len(data_clean) < nlags + 1:
        nlags = max(len(data_clean) // 2, 10)
    
    try:
        acf_values = acf(data_clean, nlags=nlags, fft=fft)
        lags = np.arange(len(acf_values))  # in hours (assuming hourly data)
        return acf_values, lags
    except Exception as e:
        print(f"Error computing ACF: {e}")
        return None, None

############
def compute_lagged_correlation(data, max_lag=96):
    """Compute correlation of a variable with itself at different time lags"""
    data_clean = np.array(data)
    data_clean = data_clean[~np.isnan(data_clean)]
    if len(data_clean) < max_lag + 1:
        max_lag = len(data_clean) // 2
    lags = []
    correlations = []
    pvalues = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            # Perfect correlation with itself
            lags.append(0)
            correlations.append(1.0)
            pvalues.append(0.0)
        else:
            x = data_clean[:-lag]
            y = data_clean[lag:]
            if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
                lags.append(lag)
                correlations.append(np.nan)
                pvalues.append(np.nan)
            else:
                try:
                    r, p = pearsonr(x, y)
                    lags.append(lag)
                    correlations.append(r)
                    pvalues.append(p)
                except:
                    lags.append(lag)
                    correlations.append(np.nan)
                    pvalues.append(np.nan)
    return np.array(lags), np.array(correlations), np.array(pvalues)
############

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

def analyze_single_track(ds, track_df, track_id, buffer_km, max_lag_hours=50, output_dir="output"):
    std_print('DEBUG', 'ca', f' --- Track {track_id} : Processing Correlation Analysis ---')
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
    total_hf_series = []
    valid_times = []

    for i, row in current_track.iterrows():
        t = pd.to_datetime(row['time'])
        lat, lon = row['latitude'], row['longitude']
        # Use pre-computed is_land if available, otherwise calculate

        subset_tp = extract_spatial_subset(ds, lat, lon, t, buffer_km, time_coord)
        tp = float(subset_tp['tp'].mean().values)
        subset_swvl1 = extract_spatial_subset(ds, lat, lon, t, buffer_km, time_coord)
        swvl1 = float(subset_swvl1['swvl1'].mean().values)
        subset_hf = extract_spatial_subset(ds, lat, lon, t, buffer_km, time_coord)
        latent_hf = float(subset_hf['avg_slhtf'].mean().values)
        sensible_hf = float(subset_hf['avg_ishf'].mean().values)
        total_hf = latent_hf + sensible_hf if np.isfinite(latent_hf) and np.isfinite(sensible_hf) else np.nan

        if np.isfinite(tp) and np.isfinite(swvl1):
            tp_series.append(tp)
            swvl1_series.append(swvl1)
            latent_hf_series.append(latent_hf)
            sensible_hf_series.append(sensible_hf)
            total_hf_series.append(total_hf)
            valid_times.append(t)

    if len(valid_times) < 4:
        std_print('WARN', 'ca', f'Track {track_id} Insufficient data' , f'only {len(valid_times)} valid points')
        return None

    # Units conversion
    tp_series = np.array(tp_series) * 1000 # m -> mm
    swvl1_series = np.array(swvl1_series)
    latent_hf_series = np.array(latent_hf_series)
    sensible_hf_series = np.array(sensible_hf_series)
    total_hf_series = np.array(total_hf_series)

    # Generate plots
    os.makedirs(output_dir, exist_ok=True)
    std_print('INFO', 'ca', f' --- Generating plots for Track {track_id} ---')
    # 1. TP-SWVL1 Feedback
    try:
        plot_tp_swvl1_time_series(tp_series, swvl1_series, valid_times, landfall_index, track_id, output_dir, originates_over_land)
        plot_tp_swvl1_correlation(ds, track_df, track_id, buffer_km=100, output_dir=output_dir)
        plot_tp_swvl1_lag_correlation(tp_series, swvl1_series, max_lag_hours, track_id, output_dir)
    except Exception as e:
        std_print('ERROR', 'ca', f'TP-SWVL1 analysis failed for Track {track_id}', str(e))
    # 2. SWVL1-Heat Flux Feedback
    try:
        if not np.all(np.isnan(latent_hf_series)) or not np.all(np.isnan(sensible_hf_series)):
            plot_swvl1_hf_correlation(swvl1_series, latent_hf_series, sensible_hf_series, valid_times, landfall_index, track_id, output_dir, originates_over_land)
            plot_swvl1_hf_lag_correlation(swvl1_series, latent_hf_series, max_lag_hours, track_id, 'LHF', output_dir)
            plot_swvl1_hf_lag_correlation(swvl1_series, sensible_hf_series, max_lag_hours, track_id, 'SHF', output_dir)
            plot_swvl1_hf_lag_correlation(swvl1_series, total_hf_series, max_lag_hours, track_id, 'THF', output_dir)
    except Exception as e:
        std_print('ERROR', 'ca', f'SWVL1-Heat Flux lag analysis failed for Track {track_id}', str(e))
    # 3. TP-Heat Flux Feedback
    try:
        if not np.all(np.isnan(latent_hf_series)) or not np.all(np.isnan(sensible_hf_series)):
            plot_tp_hf_lag_correlation(tp_series, latent_hf_series, max_lag_hours, track_id, 'LHF', output_dir)
            plot_tp_hf_lag_correlation(tp_series, sensible_hf_series, max_lag_hours, track_id, 'SHF', output_dir)
            plot_tp_hf_lag_correlation(tp_series, total_hf_series, max_lag_hours, track_id, 'THF', output_dir)
    except Exception as e:
        std_print('ERROR', 'ca', f'TP-Heat Flux lag analysis failed for Track {track_id}', str(e))
    # 4. Autocorrelation Analysis
    try:
        std_print('DEBUG', 'ca', f' --- Generating Autocorrelation plots for Track {track_id} ---')
        plot_lagged_correlation_single_variable(tp_series, var_name='tp', track_id=track_id, output_dir=output_dir)
        plot_lagged_correlation_single_variable(swvl1_series, var_name='swvl1', track_id=track_id, output_dir=output_dir)
        plot_lagged_correlation_single_variable(latent_hf_series, var_name='latent_hf', track_id=track_id, output_dir=output_dir)
        plot_lagged_correlation_single_variable(sensible_hf_series, var_name='sensible_hf', track_id=track_id, output_dir=output_dir)
    except Exception as e:
        std_print('ERROR', 'ca', f'Autocorrelation analysis failed for Track {track_id}', str(e))
        traceback.print_exc()

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

# def plot_scatter_with_regression(x, y, x_label, y_label, title, track_id, output_dir, filename):
#     mask = ~np.isnan(x) & ~np.isnan(y)
#     x, y = x[mask], y[mask]
#     if len(x) < 3: return
#     slope, intercept, r_val, p_val, _ = linregress(x, y)
#
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.scatter(x, y, alpha=0.6, edgecolors='k')
#     ax.plot(x, slope*x + intercept, color='red', label=f'r={r_val:.2f}, p={p_val:.3f}')
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     ax.set_title(f'Track {track_id}: {title}')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     path = os.path.join(output_dir, f'track_{track_id}_{filename}.png')
#     plt.savefig(path, dpi=300)
#     plt.close()
# def analyze_track_correlations(ds, track_df, track_id, output_dir="output"):
#     track_dir = os.path.join(output_dir, f"track_{track_id}")
#     os.makedirs(track_dir, exist_ok=True)
#     # 1. Primary Feedback Loop
#     plot_scatter_with_regression(tp_vals, sm_vals, 'TP (mm/hr)', 'SWVL1 (m³/m³)', 'TP vs Soil Moisture', track_id, track_dir, 'TP_SWVL1_correlations')
#     # 2. Lag Correlation
#     plot_lagged_correlation(tp_vals, sm_vals, 48, track_id, track_dir)
#     # 3. SWVL1 ↔ Heat Fluxes
#     plot_scatter_with_regression(sm_vals, latent_hf, 'SWVL1 (m³/m³)')', 'Latent Heat Flux (W/m²)', 'SM vs Latent HF', track_id, track_dir, 'SWVL1_LatentHF_scatter')
#     plot_scatter_with_regression(sm_vals, sensible_hf, 'SWVL1 (m³/m³)', 'Sensible Heat Flux (W/m²)', 'SM vs Sensible HF', track_id, track_dir, 'SWVL1_SensibleHF_scatter')
#     plot_scatter_with_regression(sm_vals, total_hf, 'SWVL1 (m³/m³)', 'Total Heat Flux (W/m²)', 'SM vs Total HF', track_id, track_dir, 'SWVL1_TotalHF_scatter')
#     std_print('INFO', 'ca', f'Completed all correlation plots for Track {track_id}')

def plot_tp_swvl1_time_series(tp, swvl1, valid_times, landfall_index, track_id, output_dir, originates_over_land=True, track_df=None):
    track_id = int(track_id)
    std_print('INFO', 'ca', f' --- Track {track_id} TP-SWVL1 Time series plot ---')
    try:
        # Validate inputs
        if not isinstance(tp, np.ndarray):
            tp = np.array(tp)
        if not isinstance(swvl1, np.ndarray):
            swvl1 = np.array(swvl1)
        if not isinstance(valid_times, (list, np.ndarray, pd.DatetimeIndex)):
            raise ValueError("valid_times must be datetime-like")

        # Check for sufficient data
        valid_data_mask = ~(np.isnan(tp) | np.isnan(swvl1))
        valid_count = np.sum(valid_data_mask)

        if valid_count < 2:
            std_print('WARN', 'ca', f'Track {track_id} Insufficient valid data', f'only {valid_count} points')
            return False

        if len(tp) != len(swvl1) or len(tp) != len(valid_times):
            std_print('WARN', 'ca', f'Track {track_id}', 'Data length mismatch between tp, swvl1, and times')
            return False

        # Create figure with dual axes
        fig, ax1 = plt.subplots(figsize=(15, 7))

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
        ax1.set_ylabel('Total Precipitation (tp) (mm/hr)', color=color_tp, fontsize=10, fontweight='bold')
        line1 = ax1.plot(valid_times, tp, '-o', color=color_tp, linewidth=2.5, markersize=5, label='Total Precipitation', alpha=0.8, zorder=3)
        ax1.tick_params(axis='y', labelcolor=color_tp)
        ax1.grid(True, alpha=0.4, linestyle='--', zorder=0)
        # land/ocean shading
        add_land_ocean_shading(ax1, pd.to_datetime(valid_times), is_land_array, land_color='wheat', ocean_color='lightblue')

        # Landfall vertical line (only if actual landfall occurred)
        if landfall_index != -1 and landfall_index < len(valid_times) and not originates_over_land:
            landfall_time = valid_times[landfall_index]
            ax1.axvline(x=landfall_time, color=status_color, linestyle='--', linewidth=2.5, alpha=0.8, label=status_text, zorder=4)

        # Soil Water Content on secondary axis
        ax2 = ax1.twinx()
        color_swvl1 = 'tab:brown'
        ax2.set_ylabel('Soil Water Volumatric Layer 1 (swvl1) (m³/m³)', color=color_swvl1, fontsize=11, fontweight='bold')
        line2 = ax2.plot(valid_times, swvl1, 's-', color=color_swvl1, linewidth=2.5, markersize=5, label='Soil Water Volumatric Level-1', alpha=0.8, zorder=3)
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

        ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.95, edgecolor='none')
        # Title based on track characteristics
        title = f'Track {track_id}: Precipitation & Soil Water Content Evolution'
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
#         ds, time_coord = get_time_coord(ds)
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
    std_print('INFO', 'ca', f' --- Track {track_id} Computing correlation ---')
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
    std_print('INFO', 'ca', ' --- Analyzing all tracks for soil water content patterns ---')
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
    std_print('INFO', 'ca', f"Soil Water Content Analysis across  {summary['n_tracks']} Tracks :")
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

# FIX: use export_results_to_csv
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
    std_print('EXPORT', 'ca', 'Correlation results to CSV', tp_swvl1_correlations_path)

def plot_tp_swvl1_correlation(ds, track_df, track_id, buffer_km = 100, output_dir = 'output'):
    result = compute_single_track_correlation(ds, track_df, track_id, buffer_km)
    std_print('INFO', 'ca', f' --- Track {track_id} TP-SWVL1 Correlation plots analysis ---')
    if result is None:
        std_print('WARN', 'ca', f'Track {track_id}', 'Insufficient flux data')
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    times = result['times']
    tp_vals = result['tp_values']
    sm_vals = result['swvl1_values']

    ax.scatter(tp_vals, sm_vals, alpha=0.6, s=100, color='purple', edgecolors='black', linewidth=1, zorder=3)
    # Regression line
    if result['n_samples'] > 2:
        x_line = np.array([tp_vals.min(), tp_vals.max()])
        y_line = result['regression_slope'] * x_line + result['regression_intercept']
        ax.plot(x_line, y_line, 'r--', linewidth=2.5, alpha=0.7, label=f'Linear Fit (r²={result["r_squared"]:.4f})')

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
    common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)
    ax.text(1.02, 0.85, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', family='monospace', bbox=common_bbox)
    ax.set_xlabel('Total Precipitation (tp) (mm/hr)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Soil Water Volumatric Level-1 (swvl1) (m³/m³)', fontsize=10, fontweight='bold')
    ax.set_title(f'Track {track_id}: TP - SWVL1 Correlation', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(fontsize=8, loc='best', framealpha=0.7)
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_TP-SWVL1_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ca', 'TP-SWVL1 Correlation Scatter plot', plot_path)

def plot_tp_swvl1_lag_correlation(tp, sm, max_lag_hours, track_id, output_dir):
    """
    Negative lag: TP leads SWVL1 (precipitation causes soil water content increase)
    Positive lag: SWVL1 leads TP (antecedent moisture affects rainfall)
    """
    track_id = int(track_id)
    std_print('INFO', 'ca', f' --- Track {track_id} Lag-Correlation plots ---')
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
        lag_interpretation = f"TP leads SWVL1 by {abs(max_corr_lag)}h\n(Infiltration lag)"
    elif max_corr_lag > 0:
        lag_interpretation = f"SWVL1 leads TP by {max_corr_lag}h\n(Antecedent moisture)"
    else:
        lag_interpretation = "Simultaneous\n(Immediate feedback)"

    fig, ax = plt.subplots(figsize=(12, 6))
    if len(lags) == len(correlations_lag):
        ax.fill_between(lags, correlations_lag, alpha=0.2, color='steelblue')
    else:
        std_print('WARN', 'ca', f'Track {track_id} Lag mismatch', 'lags and r_lag lengths differ')

    # Plot lag correlation
    ax.plot(lags, correlations_lag, 'o-', lw=2.5, markersize=4, color='blue', zorder=3)
    ax.fill_between(lags, correlations_lag, alpha=0.2, color='steelblue')
    # Reference lines
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6, label='Zero lag')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)
    ax.text(0.98, 0.94, lag_interpretation, transform=ax.transAxes, fontsize=9, va='bottom', ha='right', bbox=common_bbox)
    ax.set_xlabel("Lag (hours)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Correlation (r)", fontsize=10, fontweight='bold')
    ax.set_title(f"Track {track_id}: Lagged TP-SWVL1  Correlation", fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.5, linestyle='--')
    # ax.set_ylim([-1, 1])
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "lag_correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{int(track_id)}_TP-SWVL1_lag_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ca', 'Lag Ccrelation plot', plot_path)

def plot_swvl1_hf_correlation(swvl1, latent_hf, sensible_hf, valid_times, landfall_index, track_id, output_dir, originates_over_land=True):
    """
    Plot SWVL1 correlation with surface heat fluxes (Latent, Sensible, Total).
    Relationships:
    - SWVL1 <-> Latent HF: Soil moisture affects evaporative cooling
    - SWVL1 <-> Sensible HF: Dry soil increases sensible heating
    - SWVL1 <-> Total HF: Combined surface energy budget impact
    """
    track_id = int(track_id)
    std_print('INFO', 'ca', f' --- Track {track_id} SWVL1-HF Correlation plot ---')
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
        std_print('WARN', 'ca', f'Track {track_id}', 'Could not compute heat flux correlation')
        return False

    # 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
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

    common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)
    # Panel 1: SWVL1 vs Latent HF
    ax = axes[0]
    ax.scatter(swvl1[valid_mask], latent_hf[valid_mask], alpha=0.6, s=80, color='green', edgecolors='black', linewidth=0.5)
    # Regression line with slope label
    if np.std(swvl1[valid_mask]) > 0:
        z = np.polyfit(swvl1[valid_mask], latent_hf[valid_mask], 1)
        slope_latent = z[0]
        p = np.poly1d(z)
        x_line = np.linspace(swvl1[valid_mask].min(), swvl1[valid_mask].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label=f'Slope = {slope_latent:.3f}')
        # slope annotation to the plot
        # ax.annotate(f'Slope: {slope_latent:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='red', bbox=common_bbox, verticalalignment='top')
    ax.set_xlabel('Soil Water Volumatric Level 1 (m³/m³)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Latent Heat Flux (W/m²)', fontsize=10, fontweight='bold')
    ax.set_title(f'SWVL1 ↔ Latent HF\nr={r_latent:.3f}, p={p_latent:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='lower right', fontsize=9)

    # Panel 2: SWVL1 vs Sensible HF
    ax = axes[1]
    ax.scatter(swvl1[valid_mask], sensible_hf[valid_mask], alpha=0.6, s=80, color='orange', edgecolors='black', linewidth=0.5)
    # Regression line with slope label
    if np.std(swvl1[valid_mask]) > 0:
        z = np.polyfit(swvl1[valid_mask], sensible_hf[valid_mask], 1)
        slope_sensible = z[0]
        p = np.poly1d(z)
        x_line = np.linspace(swvl1[valid_mask].min(), swvl1[valid_mask].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label=f'Slope = {slope_sensible:.3f}')
        # slope annotation to the plot
        # ax.annotate(f'Slope: {slope_sensible:.3f}', xy=(0.05, 0.05), xycoords='axes fraction', fontsize=10, fontweight='bold', color='red', bbox=common_bbox,verticalalignment='bottom')
        ax.set_xlabel('Soil Water Volumatric Level 1 (m³/m³)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Sensible Heat Flux (W/m²)', fontsize=10, fontweight='bold')
    ax.set_title(f'SWVL1 ↔ Sensible HF\nr={r_sensible:.3f}, p={p_sensible:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='lower right', fontsize=9)

    # Panel 3: SWVL1 vs Total HF
    ax = axes[2]
    ax.scatter(swvl1[valid_mask], total_hf[valid_mask], alpha=0.6, s=80, color='purple', edgecolors='black', linewidth=0.5)
    # Regression line with slope label
    if np.std(swvl1[valid_mask]) > 0:
        z = np.polyfit(swvl1[valid_mask], total_hf[valid_mask], 1)
        slope_total = z[0]
        p = np.poly1d(z)
        x_line = np.linspace(swvl1[valid_mask].min(), swvl1[valid_mask].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label=f'Slope = {slope_total:.3f}')
        # slope annotation to the plot
        # ax.annotate(f'Slope: {slope_total:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='red', bbox=common_bbox, verticalalignment='top')
    ax.set_xlabel('Soil Water Volumatric Level 1 (m³/m³)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Total Heat Flux (W/m²)', fontsize=10, fontweight='bold')
    ax.set_title(f'SWVL1 ↔ Total HF\nr={r_total:.3f}, p={p_total:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle(f'Track {track_id}: SWVL1 ↔ Surface HF Correlation ({status_text})', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{int(track_id)}_SWVL1-HF_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ca', 'SWVL1-HF Correlation plot', plot_path)
    return True

def plot_tp_hf_correlation(tp, latent_hf, sensible_hf, valid_times, landfall_index, track_id, output_dir, originates_over_land=True):
    """
    Plot TP correlation with surface heat fluxes (Latent, Sensible, Total).
    Relationships:
    - TP <-> Latent HF: Precipitation affects evaporative cooling
    - TP <-> Sensible HF: increases in sensible heating
    - TP <-> Total HF: Combined surface energy budget impact
    """
    track_id = int(track_id)
    std_print('INFO', 'ca', f' --- Track {track_id} SWVL1-HF Correlation plot ---')
    # Calculate total heat flux
    total_hf = latent_hf + sensible_hf
    # Remove NaN values for correlation calculation
    valid_mask = ~(np.isnan(tp) | np.isnan(latent_hf) | np.isnan(sensible_hf) | np.isnan(total_hf))
    valid_count = np.sum(valid_mask)
    if valid_count < 5:
        std_print('WARN', 'ca', f'Track {track_id}', f'Insufficient heat flux data ({valid_count} valid points)')
        return False

    # Calculate correlations
    try:
        r_latent, p_latent = pearsonr(tp[valid_mask], latent_hf[valid_mask])
        r_sensible, p_sensible = pearsonr(tp[valid_mask], sensible_hf[valid_mask])
        r_total, p_total = pearsonr(tp[valid_mask], total_hf[valid_mask])
    except:
        std_print('WARN', 'ca', f'Track {track_id}', 'Could not compute heat flux correlation')
        return False

    # 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
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

    common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)
    # Panel 1: TP vs Latent HF
    ax = axes[0]
    ax.scatter(tp[valid_mask], latent_hf[valid_mask], alpha=0.6, s=80, color='green', edgecolors='black', linewidth=0.5)
    # Regression line with slope label
    if np.std(tp[valid_mask]) > 0:
        z = np.polyfit(tp[valid_mask], latent_hf[valid_mask], 1)
        slope_latent = z[0]
        p = np.poly1d(z)
        x_line = np.linspace(tp[valid_mask].min(), tp[valid_mask].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label=f'Slope = {slope_latent:.3f}')
        # slope annotation to the plot
        # ax.annotate(f'Slope: {slope_latent:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='red', bbox=common_bbox, verticalalignment='top')
    ax.set_xlabel('Totlal Precipitation (mm/hr)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Latent Heat Flux (W/m²)', fontsize=10, fontweight='bold')
    ax.set_title(f'TP ↔ Latent HF\nr={r_latent:.3f}, p={p_latent:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='lower right', fontsize=9)

    # Panel 2: TP vs Sensible HF
    ax = axes[1]
    ax.scatter(tp[valid_mask], sensible_hf[valid_mask], alpha=0.6, s=80, color='orange', edgecolors='black', linewidth=0.5)
    # Regression line with slope label
    if np.std(tp[valid_mask]) > 0:
        z = np.polyfit(tp[valid_mask], sensible_hf[valid_mask], 1)
        slope_sensible = z[0]
        p = np.poly1d(z)
        x_line = np.linspace(tp[valid_mask].min(), tp[valid_mask].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label=f'Slope = {slope_sensible:.3f}')
        # slope annotation to the plot
        # ax.annotate(f'Slope: {slope_sensible:.3f}', xy=(0.05, 0.05), xycoords='axes fraction', fontsize=10, fontweight='bold', color='red', bbox=common_bbox,verticalalignment='bottom')
        ax.set_xlabel('Totla precipitation (m/hr)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Sensible Heat Flux (W/m²)', fontsize=10, fontweight='bold')
    ax.set_title(f'TP ↔ Sensible HF\nr={r_sensible:.3f}, p={p_sensible:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='lower right', fontsize=9)

    # Panel 3: TP vs Total HF
    ax = axes[2]
    ax.scatter(tp[valid_mask], total_hf[valid_mask], alpha=0.6, s=80, color='purple', edgecolors='black', linewidth=0.5)
    # Regression line with slope label
    if np.std(tp[valid_mask]) > 0:
        z = np.polyfit(tp[valid_mask], total_hf[valid_mask], 1)
        slope_total = z[0]
        p = np.poly1d(z)
        x_line = np.linspace(tp[valid_mask].min(), tp[valid_mask].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label=f'Slope = {slope_total:.3f}')
        # slope annotation to the plot
        # ax.annotate(f'Slope: {slope_total:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='red', bbox=common_bbox, verticalalignment='top')
    ax.set_xlabel('Totla precipitation (m/hr)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Total Heat Flux (W/m²)', fontsize=10, fontweight='bold')
    ax.set_title(f'SWVL1 ↔ Total HF\nr={r_total:.3f}, p={p_total:.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle(f'Track {track_id}: TP ↔ Surface HF Correlation ({status_text})', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{int(track_id)}_TP-HF_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ca', 'TP-HF Correlation plot', plot_path)
    return True

def plot_swvl1_hf_lag_correlation(sm, hf, max_lag_hours, track_id, hf_name, output_dir):
    """
    Negative lag: SM leads HF (Brown Ocean Effect: Moisture fuels energy flux)
    Positive lag: HF leads SM (Energy flux dries out soil)
    """
    track_id = int(track_id)
    std_print('INFO', 'ca', f' --- Track {track_id} SWVL1-{hf_name} Lag-Correlation analysis ---')
    lags, correlations_lag = lag_correlation(sm, hf, max_lag=max_lag_hours)

    # Filter out NaN values and ensure alignment
    valid_indices = [i for i, val in enumerate(correlations_lag) if not np.isnan(val)]
    if not valid_indices:
        std_print('WARN', 'ca', f'Track {track_id}', f'No valid data for SWVL1-{hf_name} lag plot')
        return

    lags = [lags[i] for i in valid_indices]
    correlations_lag = [correlations_lag[i] for i in valid_indices]

    if hf_name.upper() == 'LHF' or 'LATENT' in hf_name.upper():
        line_color, fill_color = 'forestgreen', 'green'
    elif hf_name.upper() == 'SHF' or 'SENSIBLE' in hf_name.upper():
        line_color, fill_color = 'darkorange', 'orange'
    elif hf_name.upper() == 'THF' or 'TOTAL' in hf_name.upper():
        line_color, fill_color = 'rebeccapurple', 'mediumpurple'
    else:
        line_color, fill_color = 'blue', 'steelblue' # Default

    # Interpretation Logic
    max_corr_idx = np.argmax(np.abs(correlations_lag))
    max_corr_lag = lags[max_corr_idx]

    if max_corr_lag < 0:
        interp = f"SWVL1 leads {hf_name} by {abs(max_corr_lag)}h\n(Energy Feedback)"
    elif max_corr_lag > 0:
        interp = f"{hf_name} leads SWVL1 by {max_corr_lag}h\n(Surface Drying)"
    else:
        interp = "Simultaneous\n(Coupled Phase)"

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot correlation
    ax.plot(lags, correlations_lag, 'o-', lw=2.5, markersize=4, color=line_color, zorder=3)
    ax.fill_between(lags, correlations_lag, alpha=0.2, color=fill_color)
    # Reference lines
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)
    ax.text(0.98, 0.94, interp, transform=ax.transAxes, fontsize=9, va='top', ha='right', bbox=common_bbox)
    ax.set_xlabel("Lag (hours)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Correlation (r)", fontsize=10, fontweight='bold')
    ax.set_title(f"Track {track_id}: Lagged SWVL1-{hf_name} Correlation", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')

    track_dir = os.path.join(output_dir, "lag_correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_SWVL1-{hf_name}_lag_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ca', f'SWVL1-{hf_name} Lag plot', plot_path)

def plot_tp_hf_lag_correlation(flux, tp, max_lag_hours, track_id, flux_name, output_dir):
    """
    Negative lag: Flux leads TP (energy flux influences rainfall)
    Positive lag: TP leads Flux (rainfall affects energy flux)
    """
    track_id = int(track_id)
    std_print('INFO', 'ca', f' --- Track {track_id} {flux_name}-TP Lag-Correlation analysis ---')
    # Calculate lag correlation
    lags, correlations_lag = lag_correlation(flux, tp, max_lag=max_lag_hours)

    # Filter out NaN values and ensure alignment
    valid_indices = [i for i, val in enumerate(correlations_lag) if not np.isnan(val)]
    if not valid_indices:
        std_print('WARN', 'ca', f'Track {track_id}', f'No valid data for {flux_name}-TP lag plot')
        return

    lags = [lags[i] for i in valid_indices]
    correlations_lag = [correlations_lag[i] for i in valid_indices]

    if flux_name.upper() == 'LHF' or 'LATENT' in flux_name.upper():
        line_color, fill_color = 'forestgreen', 'green'
        flux_label = 'Latent Heat Flux'
    elif flux_name.upper() == 'SHF' or 'SENSIBLE' in flux_name.upper():
        line_color, fill_color = 'darkorange', 'orange'
        flux_label = 'Sensible Heat Flux'
    elif flux_name.upper() == 'THF' or 'TOTAL' in flux_name.upper():
        line_color, fill_color = 'rebeccapurple', 'mediumpurple'
        flux_label = 'Total Heat Flux'
    else:
        line_color, fill_color = 'blue', 'steelblue'
        flux_label = flux_name
    # Find maximum correlation and interpret
    if correlations_lag:
        max_corr_idx = np.argmax(np.abs(correlations_lag))
        max_corr_lag = lags[max_corr_idx]
        max_corr_value = correlations_lag[max_corr_idx]
        if max_corr_lag < 0:
            interp = f"{flux_label} leads TP by {abs(max_corr_lag)}h\n(Energy influences precipitation)"
        elif max_corr_lag > 0:
            interp = f"TP leads {flux_label} by {max_corr_lag}h\n(Rainfall affects energy flux)"
        else:
            interp = "Simultaneous\n(Strong coupling)"
    else:
        interp = "No significant correlation"
        max_corr_lag = 0
        max_corr_value = 0

    fig, ax = plt.subplots(figsize=(12, 6))
    # Plot correlation curve
    ax.plot(lags, correlations_lag, 'o-', lw=2.5, markersize=4, color=line_color, zorder=3, label=f'Lag correlation')
    ax.fill_between(lags, correlations_lag, alpha=0.2, color=fill_color)
    # Mark maximum correlation point
    if correlations_lag:
        ax.plot(max_corr_lag, max_corr_value, 'ro', markersize=8, 
                zorder=4, label=f'Max |r| = {abs(max_corr_value):.3f} at {max_corr_lag}h')
    # Reference lines
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6, label='Zero lag')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)
    ax.text(0.98, 0.94, interp, transform=ax.transAxes, fontsize=9,  va='top', ha='right', bbox=common_bbox)
    ax.set_xlabel("Lag (hours)", fontsize=10, fontweight='bold')
    ax.set_ylabel("Correlation (r)", fontsize=10, fontweight='bold')
    ax.set_title(f"Track {track_id}: Lagged {flux_label} vs Total Precipitation", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.legend(loc='best', fontsize=9)

    track_dir = os.path.join(output_dir, "lag_correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_TP-{flux_name}_lag_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ca', f'TP-{flux_name} Lag plot', plot_path)

def analyze_flux_correlations(ds, track_df, track_id, buffer_km=100):
    track_id = int(track_id)
    std_print('INFO', 'ca', f' --- Track {track_id} Analyzing Flux Correlation ---')

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

def plot_flux_analysis(results, output_dir, track_id):
    track_id = int(track_id)
    std_print('INFO', 'ca', f' --- Track {track_id} Plotting Analyzing Flux Correlation ---')
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
        # Total Precipitation & Soil Water Volumatric Level1
        try:
            if 'tp' not in df.columns or 'swvl1' not in df.columns:
                raise ValueError("Missing tp or swvl1 columns")

            fig, ax1 = plt.subplots(figsize=(14, 7))
            # Precipitation (primary Y)
            color_tp = 'tab:blue'
            ax1.set_xlabel('Time', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Total Precipitation (mm/hr)', color=color_tp, fontsize=11, fontweight='bold')
            p1, = ax1.plot(df['time'], df['tp'], '-o', color=color_tp, linewidth=2.5, markersize=4, label='Total Precipitation', alpha=0.8, zorder=3)
            ax1.tick_params(axis='y', labelcolor=color_tp)

            # Soil Water Volumatric Level-1 (secondary Y)
            ax2 = ax1.twinx()
            color_swvl1 = 'tab:brown'
            ax2.set_ylabel('Soil Water Volumatric Level-1 (m³/m³)', color=color_swvl1, fontsize=10, fontweight='bold')
            p2, = ax2.plot(df['time'], df['swvl1'], 's-', color=color_swvl1, linewidth=2.5, markersize=4, label='Soil Water Volumatric Level-1', alpha=0.8, zorder=3)
            ax2.tick_params(axis='y', labelcolor=color_swvl1)
            # land/ocean shading
            add_land_ocean_shading(ax1, df['time'], df['is_land'].values, land_color='wheat', ocean_color='lightblue')

            # Legend
            lines = [p1, p2]
            labels = ['Totlal Precipitation (tp)', 'Soil Water Volumatric Level-1 (swvl1)']
            lines.append(Patch(facecolor='lightblue', alpha=0.5, label='Ocean'))
            lines.append(Patch(facecolor='wheat', alpha=0.5, label='Land'))
            labels += ['Ocean', 'Land']
            ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.95)
            ax1.grid(True, alpha=0.4, linestyle='--', zorder=0)
            ax1.set_title(f'Track {track_id}: Total Precipitation & Soil Water Volumatric Level-1', fontsize=13, fontweight='bold', pad=10)

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
            # land/ocean shading
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
                scatter = plt.scatter(df['swvl1'], df['total_hf'], c=df['tp'], cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
                cbar = plt.colorbar(scatter)
                cbar.set_label('Total Precipitation (mm/hr)', fontsize=10, fontweight='bold')
            else:
                plt.scatter(df['swvl1'], df['total_hf'], s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

            # Differentiate land points
            if df['is_land'].any():
                land_pts = df[df['is_land'] == True]
                if not land_pts.empty:
                    plt.scatter(land_pts['swvl1'], land_pts['total_hf'], c='orange', s=150, marker='s', alpha=0.5, edgecolors='black', linewidth=1.5, label='Land', zorder=5)
            try:
                r, p = pearsonr(df['swvl1'].dropna(), df['total_hf'].dropna())
                corr_text = f'Pearson r = {r:.3f} (p={p:.4f})'
            except:
                corr_text = 'Correlation unavailable'

            plt.title(f'Track {track_id}: Soil Water Volumatric Level-1 vs Total Heat Flux\n{corr_text}', fontsize=13, fontweight='bold', pad=10)
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

def plot_autocorrelation(ds, track_df, track_id, output_dir='output'):
    variables_to_plot = [
        ('tp', 'Total Precipitation', 350),
        ('swvl1', 'Soil Water Volumatric Level-1', 350),
        ('avg_slhtf', 'Surface Latent heat flux', 350),
        ('avg_ishf', 'Surface Sensible heat flux', 350),
    ]
    track_id = int(track_id)
    current_track = track_df[track_df['track_id'] == track_id].sort_values('time')
    if current_track.empty:
        std_print('WARN', 'ca', f'Empty track for autocorrelation: {track_id}')
        return None
    # Extract data for each variable
    nlags = 96  # 4 days of lags (for hourly data)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12,), constrained_layout=True)
    axes = axes.flatten()
    for idx, (var_name, var_label, buffer_km) in enumerate(variables_to_plot):
        # Extract time series
        times = []
        values = []
        for _, row in current_track.iterrows():
            try:
                val = extract_and_average_variable(
                    ds, 
                    lat=row['latitude'],
                    lon=row['longitude'],
                    time=pd.to_datetime(row['time']),
                    var_name=var_name,
                    buffer_km=buffer_km,
                    aggregation='mean'
                )
                if not np.isnan(val):
                    times.append(pd.to_datetime(row['time']))
                    values.append(val)
            except Exception as e:
                continue
        if len(values) < 10:
            axes[idx].text(0.5, 0.5, f'{var_label}\nInsufficient data', ha='center', va='center', fontsize=10)
            axes[idx].set_xlim(0, 1)
            axes[idx].set_ylim(0, 1)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            continue
        # Compute ACF
        acf_values, lags = compute_autocorrelation(values, nlags=min(nlags, len(values)//2))
        if acf_values is None:
            axes[idx].text(0.5, 0.5, f'{var_label}\nError computing ACF', ha='center', va='center', fontsize=10)
            continue
        # Plot ACF
        axes[idx].stem(lags, acf_values, basefmt=' ')
        # Add confidence intervals (95%)
        conf_interval = 1.96 / np.sqrt(len(values))
        axes[idx].axhline(y=conf_interval, color='r', linestyle='--', alpha=0.5, linewidth=1, label='95% CI')
        axes[idx].axhline(y=-conf_interval, color='r', linestyle='--', alpha=0.5, linewidth=1)
        axes[idx].axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Format
        axes[idx].set_xlabel('Lag (hours)', fontsize=10)
        axes[idx].set_ylabel('Autocorrelation', fontsize=10)
        axes[idx].set_title(f'{var_label}\n(n={len(values)} samples)', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-0.6, 1.05)

        # Add memory metrics
        # acf_at_24h = acf_values[24] if len(acf_values) > 24 else np.nan
        # acf_at_48h = acf_values[48] if len(acf_values) > 48 else np.nan
        #
        # textstr = f'ACF @ 24h: {acf_at_24h:.3f}\nACF @ 48h: {acf_at_48h:.3f}'
        # axes[idx].text(0.98, 0.35, textstr, transform=axes[idx].transAxes,
        #               fontsize=9, verticalalignment='top', horizontalalignment='right',
        #               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    # fig.suptitle(f'Track {track_id}: Autocorrelation Functions (Variable Memory)', fontsize=14, fontweight='bold', y=0.995)

    # Save figure
    output_dir = os.path.join(output_dir, 'autocorrelation')
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f'track_{track_id}_autocorrelation.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ca', f'Autocorrelation plot for Track {track_id}', outpath)

def plot_lagged_correlation_single_variable(data, var_name, track_id, output_dir="output"):
    # Ensure data is a numpy array and has enough points
    values = np.array(data)
    values = values[~np.isnan(values)]
    if len(values) < 10:
        std_print('WARN', 'ca', f'Insufficient data for {var_name}: {len(values)} points')
        return None

    # Compute lagged correlations (Self-correlation)
    # Using 48 hours or half the track length as max lag
    max_lag = min(48, len(values) // 2)
    lags, correlations, pvalues = compute_lagged_correlation(values, max_lag=max_lag)
    valid = ~np.isnan(correlations)
 
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Plot auto correlation
    ax.plot(lags[valid], correlations[valid], 'o-', linewidth=2.5, markersize=4, color='blue', label='Lag Correlation', zorder=3)
    ax.fill_between(lags, correlations, alpha=0.2, color='steelblue')
    # Reference lines
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6, label='Zero lag')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)

    ax.set_xlabel("Lag (hours)", fontsize=10, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient (r)', fontsize=10, fontweight='bold')
    ax.set_title(f'{var_name}: Temporal Auto-correlation (Track {track_id})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.set_ylim(-1.0, 1.05)
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()

    plot_dir = os.path.join(output_dir, 'autocorrelation')
    os.makedirs(plot_dir, exist_ok=True)
    clean_name = var_name.replace(" ", "_").replace("(", "").replace(")", "")
    outpath = os.path.join(plot_dir, f'track_{track_id}_autocorr_{clean_name}.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'ca', f'Autocorrelation: {var_name}', outpath)
    return lags, correlations, pvalues
