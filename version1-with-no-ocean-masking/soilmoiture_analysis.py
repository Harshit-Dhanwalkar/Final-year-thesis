"""
Analyzes the relationship between TP and SWVL1 during entire MD track passage.

Key Questions:
- Is there a correlation between TP and SWVL1?
- Does more precipitation lead to higher soil moisture?
- How strong is the linear relationship?
- Spatial patterns of both variables during passage?
"""

# TODO: add shade region for land (like in latent heat flux transiton and in speed plots)

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr, spearmanr, linregress


def extract_regional_data(
    ds,
    track_df,
    track_id,
    variable,
    buffer_km = 100
):
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    values = []
    valid_times = []
    time_coord = "time" if "time" in ds.coords else "valid_time"
    lat_buffer = buffer_km / 111.0

    for _, row in track.iterrows():
        t = pd.to_datetime(row['time'])
        lat_c = row['latitude']
        lon_c = row['longitude']

        # Latitude-dependent longitude buffer
        lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat_c)))

        try:
            # Select spatial box and extract variable
            subset = ds.sel({
                time_coord: t,
                'latitude': slice(lat_c + lat_buffer, lat_c - lat_buffer),
                'longitude': slice(lon_c - lon_buffer, lon_c + lon_buffer)
            }, method='nearest')

            if variable in subset.data_vars:
                val = float(subset[variable].mean().values)
                if not np.isnan(val):
                    values.append(val)
                    valid_times.append(t)
        except Exception:
            continue

    return np.array(valid_times), np.array(values)


# Compute comprehensive correlation between TP and SWVL1 for a single track
def compute_single_track_correlation(
    ds,
    track_df,
    track_id,
    buffer_km = 100
):
    """
    - Pearson correlation (linear relationship)
    - Spearman correlation (monotonic relationship)
    - Linear regression (slope, intercept, R²)
    """
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
def analyze_all_tracks(
    ds,
    full_track_df,
    buffer_km=100,
    output_dir="output"
):
    print(" >  [INFO sm    ] Analyzing all tracks for soil moisture patterns")
    unique_tracks = sorted(full_track_df['track_id'].unique())
    all_results = []
    for track_id in unique_tracks:
        track_df = full_track_df[full_track_df['track_id'] == track_id].copy()
        result = compute_single_track_correlation(ds, track_df, track_id, buffer_km)
        if result:
            all_results.append(result)
    if not all_results:
        print(" >  [WARNING ] No results from individual track analysis")
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
    print(f" >  [SUMMARY ] Soil Moisture Analysis Across {summary['n_tracks']} Tracks:")
    print(f"                Mean Correlation (TP vs SM):       :                {summary['mean_correlation']:.3f} ± {summary['std_correlation']:.3f}")
    print(f"                Positive Correlations:             :                {summary['tracks_with_positive_corr']}/{summary['n_tracks']}")
    print(f"                Significant Correlations (p < 0.05):                {summary['tracks_with_significant_corr']}/{summary['n_tracks']}")
    print(f"                Mean TP:                           :                {summary['mean_tp']:.2f} mm/hr")
    print(f"                Mean SM:                           :                {summary['mean_swvl1']:.3f} m³/m³")
    # Export results
    export_results(results_df, output_dir)
    return {
        'summary': summary,
        'detailed_results': results_df
    }


def export_results(
    results_df,
    output_dir = 'output'
) -> None:
    """Export correlation results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    export_cols = [
        'track_id', 'n_samples', 'mean_tp', 'mean_swvl1',
        'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p',
        'regression_slope', 'r_squared'
    ]

    outpath = os.path.join(output_dir, 'tp_swvl1_correlations.csv')
    available_cols = [col for col in export_cols if col in results_df.columns]
    if not available_cols:
        print(f" >  [ERROR sm  ] No valid columns to export")
        return
    results_df[available_cols].to_csv(outpath, index=False)
    print(f" >> [EXPORTED] Correlation results CSV: {outpath}")


# Plot TP vs SWVL1 timeseries and scatter for a single track
def plot_track_timeseries(
    ds: xr.Dataset,
    track_df: pd.DataFrame,
    track_id: int,
    buffer_km: float = 100,
    output_dir: str = 'output'
) -> None:
    """Plot dual-axis timeseries and scatter plot for TP vs SWVL1."""
    result = compute_single_track_correlation(ds, track_df, track_id, buffer_km)

    if result is None:
        print(f" >  [WARNING ] Insufficient data for track {track_id}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    times = result['times']
    tp_vals = result['tp_values']
    sm_vals = result['swvl1_values']

    # Plot 1
    ax1 = axes[0]
    color_tp = 'tab:blue'
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Precipitation (mm/hr)', color=color_tp, fontsize=12)

    line1 = ax1.plot(
        times,
        tp_vals,
        'o-',
        color=color_tp,
        linewidth=2.5,
        markersize=7,
        label='Precipitation',
        alpha=0.8
    )
    ax1.tick_params(axis='y', labelcolor=color_tp)
    ax1.grid(True, alpha=0.5, linestyle='--')

    ax2 = ax1.twinx()
    color_sm = 'tab:brown'
    ax2.set_ylabel(
        'Soil Water Content (m³/m³)',
        color=color_sm,
        fontsize=12,
        fontweight='bold'
    )
    line2 = ax2.plot(
        times,
        sm_vals,
        's-',
        color=color_sm,
        linewidth=2.5,
        markersize=7,
        label='Soil Moisture',
        alpha=0.8
    )
    ax2.tick_params(axis='y', labelcolor=color_sm)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.95)

    ax1.set_title(f'Track {track_id}: TP vs SWVL1 Time Series\n'
                  f'Pearson r={result["pearson_r"]:.3f} (p={result["pearson_p"]:.4f})',
                  fontsize=13, fontweight='bold', pad=10)

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Scatter plot with regression
    ax3 = axes[1]
    scatter = ax3.scatter(
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
        ax3.plot(
            x_line,
            y_line,
            'r--',
            linewidth=2.5,
            alpha=0.8,
            label=f'Linear Fit (R²={result["r_squared"]:.3f})'
        )

    ax3.set_xlabel('Precipitation (mm/hr)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Soil Water Content (m³/m³)', fontsize=12, fontweight='bold')
    ax3.set_title(
        'Scatter Analysis: Precipitation vs Soil Moisture',
        fontsize=12,
        fontweight='bold'
    )
    ax3.grid(True, alpha=0.5, linestyle='--')
    ax3.legend(fontsize=10, loc='best', framealpha=0.95)

    # Statistics box
    stats_text = (
        f"Pearson r: {result['pearson_r']:.3f}\n"
        f"Spearman ρ: {result['spearman_r']:.3f}\n"
        f"N samples: {result['n_samples']}\n"
        f"Slope: {result['regression_slope']:.4f}\n"
        f"Mean TP: {result['mean_tp']:.2f} mm/hr\n"
        f"Mean SM: {result['mean_swvl1']:.3f} m³/m³"
    )
    ax3.text(
        0.98,
        0.02,
        stats_text,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f'track_{int(track_id)}_tp_swvl1_analysis.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" >> [SAVED   ] TP-SWVL1 plot                             : {outpath}")


def plot_spatial_tp_sm(
    ds: xr.Dataset,
    track_df: pd.DataFrame,
    track_id: int,
    output_dir: str = 'output',
    buffer_deg: float = 5.0
) -> None:
    """2-panel spatial maps of TP and SWVL1 during track passage."""
    track = track_df.sort_values('time')

    # Get spatial extent
    min_lat = track['latitude'].min() - buffer_deg
    max_lat = track['latitude'].max() + buffer_deg
    min_lon = track['longitude'].min() - buffer_deg
    max_lon = track['longitude'].max() + buffer_deg

    time_coord = 'time' if 'time' in ds.coords else 'valid_time'

    # Get averaged fields
    tp_data = ds['tp'].sel({
        time_coord: slice(track['time'].min(), track['time'].max()),
        'latitude': slice(max_lat, min_lat),
        'longitude': slice(min_lon, max_lon)
    }).sum(dim=time_coord) * 1000  # Total, converted to mm

    sm_data = ds['swvl1'].sel({
        time_coord: slice(track['time'].min(), track['time'].max()),
        'latitude': slice(max_lat, min_lat),
        'longitude': slice(min_lon, max_lon)
    }).mean(dim=time_coord)

    # Create figure
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # Panel 1: Total Precipitation
    im1 = axes[0].contourf(
        tp_data.longitude, tp_data.latitude, tp_data.values,
        levels=20, cmap='YlOrRd', transform=ccrs.PlateCarree()
    )
    contours1 = axes[0].contour(
        tp_data.longitude, tp_data.latitude, tp_data.values,
        levels=10, colors='black', linewidths=0.5, alpha=0.4,
        transform=ccrs.PlateCarree()
    )
    axes[0].clabel(contours1, inline=True, fontsize=8, fmt='%1.0f')
    axes[0].plot(
        track['longitude'],
        track['latitude'],
        'b-',
        linewidth=2,
        transform=ccrs.PlateCarree(),
        label='Track',
        zorder=5
    )
    axes[0].coastlines(resolution='50m', linewidth=1)
    axes[0].add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    axes[0].set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    axes[0].set_title('Total Precipitation (mm)', fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0], pad=0.02)
    cbar1.set_label('Total Precip (mm)', fontsize=10)
    axes[0].legend(loc='upper left')

    # Panel 2: Mean Soil Moisture
    im2 = axes[1].contourf(
        sm_data.longitude, sm_data.latitude, sm_data.values,
        levels=20, cmap='YlGnBu', transform=ccrs.PlateCarree()
    )
    contours2 = axes[1].contour(
        sm_data.longitude, sm_data.latitude, sm_data.values,
        levels=10, colors='black', linewidths=0.5, alpha=0.4,
        transform=ccrs.PlateCarree()
    )
    axes[1].clabel(contours2, inline=True, fontsize=8, fmt='%1.3f')
    axes[1].plot(
        track['longitude'],
        track['latitude'],
        'r-',
        linewidth=2,
        transform=ccrs.PlateCarree(),
        label='Track',
        zorder=5
    )
    axes[1].coastlines(resolution='50m', linewidth=1)
    axes[1].add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    axes[1].set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    axes[1].set_title('Mean Soil Moisture (m³/m³)', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[1], pad=0.02)
    cbar2.set_label('Soil Moisture', fontsize=10)
    axes[1].legend(loc='upper left')

    fig.suptitle(
        f'Track {track_id}: Spatial Distribution of TP and SWVL1',
        fontsize=13,
        fontweight='bold'
    )
    plt.tight_layout()

    outdir = os.path.join(output_dir, 'spatial_tp_sm')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'track_{int(track_id)}_spatial_tp_sm.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" >> [SAVED   ] Spatial TP-SM map                         : {outpath}")
