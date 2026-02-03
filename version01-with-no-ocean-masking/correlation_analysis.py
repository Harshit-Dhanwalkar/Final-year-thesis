"""
Correlation Analysis: Precipitation vs Soil Moisture
Analyzes the temporal relationship between TP and SWVL1 during track passage

- Direct correlation (overall, pre-landfall, post-landfall)
- Lagged correlation analysis
- Visual comparisons with landfall markers
- Separate time series and scatter plots

1. TP vs SWVL1
2. TP vs Surface Heat Fluxes
3. SWVL1 vs Vorticity/Intensity
4. Heat Fluxes vs Track Characteristics
"""

# FIX: Instead of recalculating is_point_land() repeatedly, use the is_land column already computed in the track dataframe.
# # In antecedent_analysis.py & correlation_analysis.py
# # Use the pre-computed column from the dataframe
# landfall_row = track[track['is_land'] == True].iloc[0] if any(track['is_land']) else None
# if landfall_row is not None:
#     landfall_time = pd.to_datetime(landfall_row['time'])

# FIX: Add ocean mask for swl1

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf
import traceback

from utils.land_utils import is_point_land

# Helper functions
def lag_correlation(
    x,
    y,
    max_lag=48, # hrs
    # max_lag=72, # hrs
):
    x = np.asarray(x)
    y = np.asarray(y)
    # Remove NaN values while keeping alignment
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 10:
        # Too few points for meaningful correlation
        print(f" > [WARNING] Insufficient non-NaN data for lag correlation: {len(x_clean)} points")
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
    lat_buffer = buffer_km / 111.0
    lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat)))
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    try:
        # Handle longitude wrapping
        ds_lon = ds.longitude.values
        if ds_lon.max() > 180 and lon < 0:
            lon_adj = lon + 360
        else:
            lon_adj = lon
        # Extract data
        subset = ds.sel(
            **{time_coord: t},
            latitude=slice(lat + lat_buffer, lat - lat_buffer),
            longitude=slice(lon_adj - lon_buffer, lon_adj + lon_buffer)
        )
        if var_name in subset:
            return float(subset[var_name].mean().values)
        else:
            return np.nan
    except Exception:
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

def analyze_single_track(
    ds,
    track_df,
    track_id,
    buffer_km=51,
    max_lag_hours=50,
    output_dir="output"
):
    print(f" >  [INFO ca    ] Processing Correlation Analysis for Track {track_id}")
    # Ensure is_land column exists
    if 'is_land' not in track_df.columns:
        track_df['is_land'] = track_df.apply(
            lambda row: is_point_land(row['latitude'], row['longitude']),
            axis=1
        )
    has_is_land = 'is_land' in track_df.columns

    # Split into phases
    ocean_phase = track_df[track_df['is_land'] == False].copy()
    land_phase = track_df[track_df['is_land'] == True].copy()

    if len(track_df['track_id'].unique()) == 1:
        # Already filtered to single track
        current_track = track_df.sort_values('time').copy()
    else:
        # Full DataFrame, needs filtering
        current_track = track_df[track_df['track_id'] == track_id].sort_values('time').copy()
    if current_track.empty:
        print(f" >  [ERROR ca   ] Track {track_id} not found.")
        return None

    tp_series = []
    swvl1_series = []
    valid_times = []
    landfall_index = -1
    originates_over_land = False
    reported_landfall = False
    lat_buffer = buffer_km / 111.0

    for i, row in current_track.iterrows():
        t = pd.to_datetime(row['time'])
        lat_c, lon_c = row['latitude'], row['longitude']
        # Use pre-computed is_land if available, otherwise calculate
        if has_is_land:
            is_land = row['is_land']
        else:
            is_land = is_point_land(lat_c, lon_c)
        # Origin check
        if i == 0:
            if is_land:
                originates_over_land = is_land
                print("Check")
                # print(f" >  [INFO ca    ] Time at Track {track_id} originates over LAND:       {t}")
                print(f" >  [INFO ca    ] Track {track_id} originates over {'LAND' if is_land else 'OCEAN'} at {t}")
        # First landfall detection (only if originated over ocean)
        if not originates_over_land and not reported_landfall and is_land:
            landfall_index = len(tp_series)
            reported_landfall = True
            print(f" >  [INFO ca    ] Track {track_id} Landfall detected at      :          {t}")

        # Extract data
        lat_buffer = buffer_km / 111.0
        lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat_c)))
        try:
            # Handle longitude wrapping if dataset is 0-360 and track is -180-180
            ds_lon = ds.longitude.values
            lon_c_adj = lon_c
            if ds_lon.max() > 180 and lon_c < 0:
                lon_c_adj = lon_c + 360

            time_coord = 'time' if 'time' in ds.coords else 'valid_time'
            subset = ds.sel(
                **{time_coord: t},
                latitude=slice(lat_c + lat_buffer, lat_c - lat_buffer),
                longitude=slice(lon_c_adj - lon_buffer, lon_c_adj + lon_buffer)
            )

            if 'tp' in subset.data_vars:
                tp = float(subset['tp'].mean().values)
            else:
                tp = np.nan
            if 'swvl1' in subset.data_vars:
                swvl1 = float(subset['swvl1'].mean().values)
            else:
                swvl1 = np.nan
            if np.isfinite(tp) and np.isfinite(swvl1):
                tp_series.append(tp)
                swvl1_series.append(swvl1)
                valid_times.append(t)
        except Exception as e:
            print(f" >  [ERROR ca  ] Error extracting data at time {t}: {e}")
            continue

    if len(valid_times) < 5:
        print(f" >  [WARNING ] Insufficient data for Track {track_id}     :  only {len(valid_times)} valid points")
        return None

    # Units conversion
    tp_series = np.array(tp_series) * 1000 # m -> mm
    swvl1_series = np.array(swvl1_series)

    ############################################################
    # print(f"\n >  [DEBUG   ] Raw tp data for track {track_id}")
    # print(f"Number of samples: {len(tp_series)}")
    # tp_mm = tp_series # m → mm (already converted)
    # print(f"  TP values (converted to mm):")
    # print(f"    Min: {np.min(tp_mm):.6f} mm")
    # print(f"    Max: {np.max(tp_mm):.6f} mm")
    # print(f"    Mean: {np.mean(tp_mm):.6f} mm")
    # print(f"    Median: {np.median(tp_mm):.6f} mm")
    # print(f"  First 10 TP values (raw):")
    # for i in range(min(10, len(tp_series))):
    #     print(f"    {i}: {tp_series[i]:.8f} m = {tp_mm[i]:.6f} mm")
    # print(f"  First 10 SWVL1 values:")
    # for i in range(min(10, len(swvl1_series))):
    #     print(f"    {i}: {swvl1_series[i]:.6f} m³/m³")
    ############################################################

    # Generate plots
    os.makedirs(output_dir, exist_ok=True)
    print(f" >  [INFO ca    ] Generating plots for Track {track_id}")
    try:
        plot_tp_sm_time_series(tp_series, swvl1_series, valid_times, landfall_index, track_id, output_dir, originates_over_land)
    except Exception as e:
        print(f" >  [WARNING ] Failed to create time series plot: {e}")
    try:
        plot_scatter_analysis(tp_series, swvl1_series, track_id, output_dir)
    except Exception as e:
        print(f" >  [WARNING ] Failed to create scatter plot: {e}")
    try:
        plot_lagged_correlation(tp_series, swvl1_series, max_lag_hours, track_id, output_dir)
    except Exception as e:
        print(f" >  [WARNING ] Failed to create lag correlation plot: {e}")

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
    # n_effective = effective_sample_size(tp_series, swvl1_series)
    # if len(tp_series) >= 3 and np.std(tp_series) > 0 and np.std(swvl1_series) > 0:
    #     try:
    #         corr, p_val_uncorrected = pearsonr(tp_series, swvl1_series)
    #         # Adjust degrees of freedom
    #         dof_effective = n_effective - 2
    #         dof_original = len(tp_series) - 2
    #         if dof_effective > 0:
    #             # Calculate t-statistic
    #             t_stat = corr * np.sqrt(dof_original / (1 - corr**2))
    #             # Adjust p-value
    #             p_val_adjusted = 2 * (1 - stats.t.cdf(abs(t_stat), dof_effective))
    #         else:
    #             p_val_adjusted = p_val_uncorrected
    #         correlation_results['correlation'] = corr
    #         correlation_results['p_value'] = p_val_adjusted
    #         correlation_results['p_value_uncorrected'] = p_val_uncorrected
    #         correlation_results['n_effective'] = n_effective
    #         correlation_results['reduction_factor'] = n_effective / len(tp_series)
    #         # Print warning if autocorrelation is high
    #         if n_effective / len(tp_series) < 0.5:
    #             print(f" > [WARNING ] High autocorrelation reduces effective sample size to {n_effective:.1f} "
    #                   f"({100*n_effective/len(tp_series):.0f}% of original)")
    #     except Exception as e:
    #         print(f" > [ERROR ca  ] Failed to calculate correlation: {e}")

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

    print(f" >> [CORRELATION] Track {track_id}                        : \n"
          f"                 r={correlation_results['correlation']:.3f},\n"
          f"                 p={correlation_results['p_value']:.4f},\n"
          f"                 n={len(valid_times)}"
    )
    return correlation_results


def plot_tp_sm_time_series(
    tp,
    sm,
    valid_times,
    landfall_index,
    track_id,
    output_dir,
    originates_over_land=False
):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Determine landfall status
    if originates_over_land:
        status_text = "Originates over land"
        status_color = 'blue'
    elif landfall_index == -1:
        status_text = "No Landfall"
        status_color = 'gray'
    else:
        status_text = "Landfall"
        status_color = 'black'

    # Time Series (Dual Axis)
    color_tp = 'tab:blue'
    ax1.set_xlabel(
        'Time',
        fontsize=10,
        fontweight='bold'
    )
    ax1.set_ylabel(
        'Precipitation (mm/hr)',
        color='black',
        fontsize=10,
        fontweight='bold'
    )
    line1 = ax1.plot(
        valid_times,
        tp,
        '-',
        color=color_tp,
        linewidth=2,
        marker='o',
        markersize=4,
        label='Precipitation',
        alpha=0.8,
        zorder=3
    )
    ax1.tick_params(axis='y', labelcolor=color_tp)
    ax1.grid(True, alpha=0.5, linestyle='--')

    # Landfall line
    if landfall_index != -1 and landfall_index < len(valid_times):
        landfall_time = valid_times[landfall_index]
        ax1.axvline(
            x=landfall_time,
            color=status_color,
            linestyle='--',
            linewidth=2,
            alpha=0.8,
            zorder=4
        )
    # Soil moisture on secondary axis
    ax2 = ax1.twinx()
    color_sm = 'tab:brown'
    ax2.set_ylabel(
        'Soil Moisture (m³/m³)',
        color='black',
        fontsize=10,
        fontweight='bold'
    )
    line2 = ax2.plot(
        valid_times,
        sm,
        's-',
        color=color_sm,
        linewidth=2,
        markersize=4,
        label='Soil Moisture',
        alpha=0.8,
        zorder=3
    )
    ax2.tick_params(axis='y', labelcolor=color_sm)

    # Legend
    lines = line1 + line2
    labels = ['Precipitation', 'Soil Moisture']
    if landfall_index != -1 and landfall_index < len(valid_times):
        lines.append(plt.Line2D([0], [0], color=status_color, linestyle='--', linewidth=2))
        labels.append(status_text)
    ax1.legend(lines, labels, loc='best', framealpha=0.7)

    ax1.set_title(f'Track {track_id}: Precipitation & Soil Moisture Time Series', fontsize=13, fontweight='bold', pad=10)
    plt.setp(
        ax1.xaxis.get_majorticklabels(),
        rotation=45,
        ha='right'
    )
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "time_series")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{int(track_id)}_tp_sm_time_series.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" >> [SAVED   ] Time series plot                  :          {plot_path}")


def plot_scatter_analysis(
    tp,
    sm,
    track_id,
    output_dir,
):
    ############################################################
    print(f" >  [DEBUG   ] TP values (first 10):")
    print(f"                {'Index':<8} {'TP (mm/hr)':<20} {'SM (m³/m³)':<15}")
    for i in range(min(10, len(tp))):
        print(f"                {i:<8} {tp[i]:<20.6f} {sm[i]:<15.6f}")
    print(f" >  [DEBUG   ] Statistics:")
    print(f"                TP min: {np.min(tp):.6f} mm/hr")
    print(f"                TP max: {np.max(tp):.6f} mm/hr")
    print(f"                TP mean: {np.mean(tp):.6f} mm/hr")
    print(f"                TP median: {np.median(tp):.6f} mm/hr")
    print(f"                TP std: {np.std(tp):.6f} mm/hr")
    print(f" > [DEBUG    ] SM min: {np.min(sm):.6f} m³/m³")
    print(f"                SM max: {np.max(sm):.6f} m³/m³")
    print(f"                SM mean: {np.mean(sm):.6f} m³/m³")
    print(f" > [DEBUG    ] Number of zero TP values: {np.sum(tp == 0)}/{len(tp)}")
    print(f"                Number of TP > 1 mm/hr: {np.sum(tp > 1)}/{len(tp)}")
    print(f"                Number of TP > 0.1 mm/hr: {np.sum(tp > 0.1)}/{len(tp)}")
    ############################################################

    fig, ax3 = plt.subplots(figsize=(10, 8))
    corr, p_val = pearsonr(tp, sm)
    ax3.scatter(
        tp,
        sm,
        alpha=0.6,
        s=100,
        color='blue',
        edgecolors='black',
        linewidth=1,
        zorder=3
    )
    # Regression line
    if len(tp) > 2:
        z = np.polyfit(tp, sm, 1)
        p = np.poly1d(z)
        x_line = np.array([tp.min(), tp.max()])
        y_line = p(x_line)
        r_squared = np.corrcoef(tp, p(tp))[0,1]**2
        ax3.plot(
            x_line,
            y_line,
            'r--',
            linewidth=2,
            alpha=0.85,
            zorder=4,
            label=f'Linear Fit (r²={r_squared:.3f})'
        )
    ax3.set_xlabel('Precipitation (mm/hr)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Soil Moisture (m³/m³)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Track {track_id}: TP - SWVL1 Correlation', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.5, linestyle='--')
    ax3.legend(fontsize=8, loc='best', framealpha=0.7)

    # Statistics box
    stats_text = (
        f"Pearson r: {corr:.3f}\n"
        f"P-value:   {p_val:.4f}\n"
        f"N samples: {len(tp)}\n"
        f"Mean TP:   {np.mean(tp):.2f} mm/hr\n"
        f"Mean SM:   {np.mean(sm):.3f} m³/m³"
    )
    ax3.text(
        1.08,
        0.9,
        stats_text,
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment='top',
        # horizontalalignment='right',
        horizontalalignment='left',
        family='monospace',
        bbox=dict(boxstyle='round', alpha=0.3)
    )
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "scatter_plots")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{int(track_id)}_scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" >> [SAVED   ] Scatter plot                      :          {plot_path}")


def plot_lagged_correlation(
    tp,
    sm,
    max_lag_hours,
    track_id,
    output_dir,
):
    lags, correlations_lag = lag_correlation(tp, sm, max_lag=max_lag_hours)

    # Ensure lags and correlations have same length
    if len(lags) != len(correlations_lag):
        print(f" >  [WARNING ] Mismatch between lags ({len(lags)}) and correlations ({len(correlations_lag)})")
        # Truncate to use the shorter length
        min_len = min(len(lags), len(correlations_lag))
        lags = lags[:min_len]
        correlations_lag = correlations_lag[:min_len]

    # Filter out NaN values
    valid_indices = [i for i, val in enumerate(correlations_lag) if not np.isnan(val)]
    if len(valid_indices) == 0:
        print(f" > [WARNING] No valid correlation values for lag plot")
        return

    lags = [lags[i] for i in valid_indices]
    correlations_lag = [correlations_lag[i] for i in valid_indices]

    fig, ax = plt.subplots(figsize=(12, 6))
    if len(lags) == len(correlations_lag):
        ax.fill_between(lags, correlations_lag, alpha=0.2, color='steelblue')
    else:
        print(" >  [WARNING ] Lag mismatch: lags and r_lag lengths differ.")

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
    ax.set_xlabel("Lag (hours)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Correlation (r)", fontsize=12, fontweight='bold')
    ax.set_title(f"Track {track_id}: Lagged Precipitation-Soil Moisture Correlation", fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.set_ylim([-1, 1])
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "lag_correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{int(track_id)}_lag_corr.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" >> [SAVED   ] Lag correlation plot              :          {plot_path}")


def analyze_flux_correlations(
    ds,
    track_df,
    track_id,
    buffer_km=100,
):
    time_coord = "time" if "time" in ds.coords else "valid_time"
    ds_times = pd.to_datetime(ds[time_coord].values)

    time_series = []
    lat_buffer = buffer_km / 111.0

    for _, row in track_df.iterrows():
        t = pd.to_datetime(row["time"])
        lat = row["latitude"]
        lon = row["longitude"]
        lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat)))
        is_land = row.get("is_land", is_point_land(lat, lon))

        # nearest ERA5 times
        idx_now  = np.argmin(np.abs(ds_times - t))
        idx_prev = np.argmin(np.abs(ds_times - (t - pd.Timedelta(hours=1))))
        # t_now = ds_times[idx_now]
        # t_prev = ds_times[idx_prev]

        try:
            # spatial box
            sub_now = ds.isel({time_coord: idx_now}).sel(
                latitude=slice(lat + lat_buffer, lat - lat_buffer),
                longitude=slice(lon - lon_buffer, lon + lon_buffer),
            )
            sub_prev = ds.isel({time_coord: idx_prev}).sel(
                latitude=slice(lat + lat_buffer, lat - lat_buffer),
                longitude=slice(lon - lon_buffer, lon + lon_buffer),
            )
            # Check which flux variables are available
            if 'avg_slhtf' in ds.data_vars and 'avg_ishf' in ds.data_vars:
                # Use average fluxes directly (already in W/m²)
                slhf_w = float(sub_now["avg_slhtf"].mean().values)
                sshf_w = float(sub_now["avg_ishf"].mean().values)
            elif 'slhf' in ds.data_vars and 'sshf' in ds.data_vars:
                # De-accumulate accumulated fluxes  J m-2  ->  W m-2
                dt_seconds = 3600.0
                slhf_w = -(float(sub_now["slhf"].mean().values) -
                           float(sub_prev["slhf"].mean().values)) / dt_seconds
                sshf_w = -(float(sub_now["sshf"].mean().values) -
                           float(sub_prev["sshf"].mean().values)) / dt_seconds
            else:
                continue
            tp_mm = float(sub_now["tp"].mean().values) * 1000.0   # m -> mm
            swvl1 = float(sub_now["swvl1"].mean().values)
            if np.any(np.isnan([tp_mm, swvl1, slhf_w, sshf_w])):
                continue

            time_series.append(
                {
                    "time": t,
                    "tp": tp_mm,
                    "swvl1": swvl1,
                    "latent_hf": slhf_w,
                    "sensible_hf": sshf_w,
                    "total_hf": slhf_w + sshf_w,
                    "is_land": is_land,
                    "latitude": lat,
                    "longitude": lon,
                }
            )
        except Exception:
            continue
    if len(time_series) < 5:
        print(f" >  [WARNING ] Insufficient flux data for Track {track_id}")
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
        print(f" >> [INFO ca    ] Flux correlations for Track {track_id}    :")
        for k, v in correlations.items():
            print(f"             {k}: r={v['r']:.3f}, p={v['p']:.4f}, n={v['n']}")
    return results

# Helper function to add Land/Ocean shading
def add_land_ocean_shading(ax, time_data, is_land_data):
    ax.fill_between(time_data, 0, 1, where=is_land_data, color='bisque', alpha=0.5, transform=ax.get_xaxis_transform(), label='Land')
    ax.fill_between(time_data, 0, 1, where=~is_land_data, color='azure', alpha=0.5, transform=ax.get_xaxis_transform(), label='Ocean')

def plot_flux_analysis(
    results,
    output_dir,
    track_id,
):
    if results is None or len(results) < 4:  # Require minimum points for a meaningful plot
       print(f" >  [WARNING ] Insufficient data to plot flux analysis for Track {track_id}")
       return
    if not results or 'time_series' not in results:
        print(f" >  [WARNING ] No time series data for Track {track_id}")
        return

    df = results['time_series']
    # Validate DataFrame
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f" >  [WARNING ] Invalid or empty DataFrame for Track {track_id}")
        return

    if df.empty or len(df) < 3:
        print(f" >  [WARNING ] Insufficient data points for Track {track_id} (only {len(df)} points)")
        return

    # Check for required columns
    required_cols = ['time', 'tp', 'swvl1']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f" >  [WARNING ] Missing required columns {missing_cols} for Track {track_id}")
        return
    # Ensure is_land column exists
    if 'is_land' not in df.columns:
        df['is_land'] = False  # Default to ocean if not present

    track_dir = os.path.join(output_dir, f"track_{track_id}_analysis")
    os.makedirs(track_dir, exist_ok=True)

    # PLOT 1: Precipitation & Soil Moisture
    try:
        ## Precip (Primary Y)
        plt.figure(figsize=(10, 5))
        ax1 = plt.gca()
        p1, = ax1.plot(
            df['time'],
            df['tp'],
            'b-',
            linewidth=2
        )
        ax1.set_ylabel('Total Precipitation (mm/hr)', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='b')
        ## Soil Moisture (Secondary Y)
        ax1_twin = ax1.twinx()
        p2, = ax1_twin.plot(
            df['time'],
            df['swvl1'],
            'r-',
            linewidth=2
        )
        ax1_twin.set_ylabel('Soil Moisture (m³/m³)', fontweight='bold')
        ax1_twin.tick_params(axis='y', labelcolor='r')

        add_land_ocean_shading(ax1, df['time'], df['is_land'])
        plt.title(f"Track {track_id}: Total Precipitation & Soil Moisture", fontsize=13, fontweight='bold')

        # Get legend items from shading (land/ocean patches) and the data line handles with proper labels
        handles, labels = ax1.get_legend_handles_labels()
        all_handles = handles + [p1, p2]
        all_labels = labels + ['Total Precipitation', 'Soil Moisture']
        ax1.legend(all_handles, all_labels, loc='best', framealpha=0.7)
        ax1.grid(True, alpha=0.5, linestyle='--')

        plot1_path = os.path.join(track_dir, f"track_{track_id}_tp_sm.png")
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" >> [SAVED   ] Precipitation & Soil Moisture Plot:          {os.path.basename(plot1_path)}")
    except Exception as e:
        print(f" >  [WARNING ] Failed to create TP & SM plot for Track {track_id}: {e}")

    # PLOT 2: Heat Fluxes (The one with the Land/Ocean Legend)
    try:
        plt.figure(figsize=(10, 5))
        ax2 = plt.gca()
        ax2.plot(
            df['time'],
            df['latent_hf'],
            'blue',
            label='Latent Heat Flux',
            linewidth=2
        )
        ax2.plot(
            df['time'],
            df['sensible_hf'],
            'red',
            label='Sensible Heat Flux',
            linewidth=2
        )
        ax2.plot(
            df['time'],
            df['total_hf'],
            'purple',
            label='Total Heat Flux',
            linewidth=2
        )

        add_land_ocean_shading(ax2, df['time'], df['is_land'])
        ax2.set_ylabel('Flux (W/m²)', fontweight='bold')
        ax2.set_xlabel('Time', fontweight='bold')
        ax2.legend(loc='upper left', framealpha=0.7)
        ax2.grid(True, alpha=0.5, linestyle='--')

        plt.title(f"Track {track_id}: Surface Heat Flux Transition", fontsize=13, fontweight='bold')
        plot2_path = os.path.join(track_dir, f"track_{track_id}_heat_fluxes_trasnsition.png")
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" >> [SAVED   ] Surface Heat Flux Transitions Plot:          {os.path.basename(plot2_path)}")
    except Exception as e:
        print(f" >  [WARNING ] Failed to create heat flux plot for Track {track_id}: {e}")

    # PLOT 3: Scatter Plot (Flux vs. Soil Moisture)
    try:
        plt.figure(figsize=(10, 5))
        if len(df) > 2:
            scatter = plt.scatter(
                df['swvl1'],
                df['total_hf'],
                c=df['tp'],
                cmap='viridis',
                s=70,
                alpha=0.8,
                edgecolors='k',
                label='All Points'
            )
            # Calculate Correlation
            try:
                r, p = pearsonr(df['swvl1'], df['total_hf'])
                corr_text = f"Pearson r = {r:.3f} (p={p:.4f})"
            except:
                corr_text = "Unable to calculate correlation"
            # Differentiate land vs ocean points with markers if data exists
            if 'is_land' in df.columns and df['is_land'].any():
                land_pts = df[df['is_land'] == True]
                if not land_pts.empty:
                    plt.scatter(
                        land_pts['swvl1'],
                        land_pts['total_hf'],
                        c='orange',
                        s=100,
                        marker='s',
                        alpha=0.6,
                        edgecolors='k',
                        linewidth=1.5,
                        label='Land Points',
                        zorder=5
                    )
            plt.title(
                f"Track {track_id}: SM vs Total Flux\n{corr_text}",
                fontsize=13,
                fontweight='bold'
            )
            cbar = plt.colorbar(scatter)
            cbar.set_label('Precipitation (mm/hr)', fontsize=10)
        else:
            # Fallback if insufficient data
            plt.text(
                0.5, 0.5,
                f"Insufficient Data for Track {track_id}\n({len(df)} points)",
                ha='center',
                va='center',
                fontsize=12,
                transform=plt.gca().transAxes
            )
            plt.title(f"Track {track_id}: SM vs Flux (Insufficient Data)")

        plt.xlabel('Soil Moisture (m³/m³)', fontweight='bold', fontsize=11)
        plt.ylabel('Total Heat Flux (W/m²)', fontweight='bold', fontsize=11)
        plt.legend(loc='best', framealpha=0.7)
        plt.grid(True, alpha=0.5, linestyle='--')

        plot3_path = os.path.join(track_dir, f"track_{track_id}_correlation_scatter.png")
        plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" >> [SAVED   ] SM vs Flux Correlation plot       :          {os.path.basename(plot3_path)}")
    except Exception as e:
        print(f" >  [WARNING ] Failed to create scatter plot for Track {track_id}: {e}")
        traceback.print_exc()
