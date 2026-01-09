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

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from utils.land_utils import is_point_land

def lag_corr(x, y, max_lag=24):
    lags = range(-max_lag, max_lag+1)
    r = []
    x = np.asarray(x)
    y = np.asarray(y)
    for lag in lags:
        if lag < 0:
            abs_lag = -lag
            x_slice = x[:-abs_lag]
            y_slice = y[abs_lag:]
        elif lag > 0:
            x_slice = x[lag:]
            y_slice = y[:-lag]
        else:
            x_slice = x
            y_slice = y
        # Check for correlation: need at least 3 points and non-zero variance
        if (len(x_slice) < 3 or
            np.std(x_slice) == 0 or
            np.std(y_slice) == 0 or
            not (np.all(np.isfinite(x_slice)) and np.all(np.isfinite(y_slice)))):
            r.append(np.nan)
        else:
            r.append(np.corrcoef(x_slice, y_slice)[0,1])
    return lags, r

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

def analyze_single_track(
    ds, track_df,
    track_id,
    buffer_km=51,
    max_lag_hours=24,
    output_dir="output"
):
    print(f" >  [INFO    ] Processing Correlation Analysis for Track {track_id}")
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    current_track = track_df[track_df['track_id'] == track_id].sort_values('time').copy()
    if current_track.empty:
        print(f" >  [ERROR    ] Track {track_id} not found.")
        return None

    tp_series = []
    swvl1_series = []
    valid_times = []
    landfall_index = -1
    originates_over_land: bool = False
    reported_ocean: bool = False
    reported_landfall: bool = False
    lat_buffer = buffer_km / 111.0

    for i, row in current_track.iterrows():
        t = pd.to_datetime(row['time'])
        lat_c, lon_c = row['latitude'], row['longitude']
        is_land = is_point_land(lat_c, lon_c)
        # Origin check
        if i == 0:
            if is_land:
                originates_over_land = True
                print(f" >  [INFO    ] Track {track_id} originates over LAND at {t}")
            else:
                print(f" >  [INFO    ] Track {track_id} originates over OCEAN at {t}")
                reported_ocean = True
        # First landfall detection
        if not originates_over_land and not reported_landfall:
            if is_land:
                landfall_index = len(tp_series)
                reported_landfall = True
                print(f" >  [INFO    ] Track {track_id} Landfall detected at {t}")
        # ocean tracking before landfall
        if not originates_over_land and not reported_ocean and not is_land:
            print(f" >  [INFO    ] Track {track_id} is over Ocean at {t}")
            reported_ocean = True

        # Latitude-dependent longitude buffer
        lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat_c)))
        try:
            # Handle longitude wrapping if dataset is 0-360 and track is -180-180
            ds_lon = ds.longitude.values
            if ds_lon.max() > 180 and lon_c < 0:
                lon_c_adj = lon_c + 360
            else:
                lon_c_adj = lon_c
            subset = ds.sel(
                **{time_coord: t},
                latitude=slice(lat_c + lat_buffer, lat_c - lat_buffer),
                longitude=slice(lon_c_adj - lon_buffer, lon_c_adj + lon_buffer)
            )
            time_subset = ds.sel(**{time_coord: t}, method='nearest')
            if 'tp' in time_subset.data_vars and 'swvl1' in time_subset.data_vars:
                spatial_subset = time_subset.sel(
                    latitude=slice(lat_c + lat_buffer, lat_c - lat_buffer),
                    longitude=slice(lon_c_adj - lon_buffer, lon_c_adj + lon_buffer)
                )
                if spatial_subset['tp'].size > 0 and spatial_subset['swvl1'].size > 0:
                    tp = spatial_subset['tp'].mean().values
                    swvl1 = spatial_subset['swvl1'].mean().values
                    if np.isfinite(tp) and np.isfinite(swvl1):
                        tp_series.append(float(tp))
                        swvl1_series.append(float(swvl1))
                        valid_times.append(t)
        except Exception as e:
            print(f" >  [ERROR   ] {e}")
            continue

    if len(valid_times) < 5:
        print(f" >  [WARNING ] Insufficient data for Track {track_id}")
        return None

    tp_series = np.array(tp_series)
    swvl1_series = np.array(swvl1_series)

    # Global Correlation
    corr, p_val = pearsonr(tp_series, swvl1_series) if np.std(tp_series) > 0 else (np.nan, np.nan)

    # Generate plots
    plot_time_series(tp_series, swvl1_series, valid_times, landfall_index, track_id, output_dir, originates_over_land)
    plot_scatter_analysis(tp_series, swvl1_series, track_id, output_dir)
    plot_lagged_correlation(tp_series, swvl1_series, max_lag_hours, track_id, output_dir)

    # Phase-based Correlations
    corr_pre, corr_post = np.nan, np.nan
    if landfall_index != -1:
        tp_pre, sm_pre = tp_series[:landfall_index], swvl1_series[:landfall_index]
        tp_post, sm_post = tp_series[landfall_index:], swvl1_series[landfall_index:]
        if len(tp_pre) >= 3 and np.std(tp_pre) > 0 and np.std(sm_pre) > 0:
            corr_pre, _ = pearsonr(tp_pre, sm_pre)
        if len(tp_post) >= 3 and np.std(tp_post) > 0 and np.std(sm_post) > 0:
            corr_post, _ = pearsonr(tp_post, sm_post)

    return {
        'track_id': track_id,
        'correlation': corr,
        'p_value': p_val,
        'corr_pre': corr_pre,
        'corr_post': corr_post,
        'mean_tp': np.mean(tp_series),
        'mean_swvl1': np.mean(swvl1_series),
        'n_samples': len(valid_times),
        'landfall_time': valid_times[landfall_index] if landfall_index != -1 else None,
        'landfall_index': landfall_index,
        'originates_over_land': originates_over_land
    }


def plot_time_series(
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
    ax1.grid(True, alpha=0.3, linestyle='--')

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
        lines.append(plt.Line2D([0], [0], color=status_color, linestyle='--', linewidth=2.5))
        labels.append(status_text)
    ax1.legend(
        lines,
        labels,
        loc='upper left',
        fontsize=8,
        framealpha=0.7
    )
    ax1.set_title(
        f'Track {track_id}: Precipitation & Soil Moisture Time Series',
        fontsize=12,
        fontweight='bold',
        pad=10
    )
    plt.setp(
        ax1.xaxis.get_majorticklabels(),
        rotation=45,
        ha='right'
    )
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "time_series")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{int(track_id)}_time_series.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" >> [SAVED   ] Time series plot: {plot_path}")


def plot_scatter_analysis(
    tp,
    sm,
    track_id,
    output_dir,
):
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
            linewidth=2.5,
            alpha=0.85,
            zorder=4,
            label=f'Linear Fit (r²={r_squared:.3f})'
        )
    ax3.set_xlabel('Precipitation (mm/hr)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Soil Moisture (m³/m³)', fontsize=11, fontweight='bold')
    ax3.set_title(
        f'Track {track_id}: TP vs SWVL1 Scatter Analysis',
        fontsize=12,
        fontweight='bold'
    )
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(fontsize=10, loc='best', framealpha=0.95)

    # Statistics box
    stats_text = (
        f"Pearson r: {corr:.3f}\n"
        f"P-value: {p_val:.4f}\n"
        f"N samples: {len(tp)}\n"
        f"Mean TP: {np.mean(tp):.2f} mm/hr\n"
        f"Mean SM: {np.mean(sm):.3f} m³/m³"
    )
    ax3.text(
        0.98,
        0.02,
        stats_text,
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        family='monospace',
        bbox=dict(boxstyle='round', alpha=0.9)
    )
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "scatter_plots")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{int(track_id)}_scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" >> [SAVED   ] Scatter plot: {plot_path}")


def plot_lagged_correlation(tp, sm, max_lag_hours, track_id, output_dir):
    lags, r_lag = lag_corr(tp, sm, max_lag=max_lag_hours)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot correlation
    ax.plot(
        lags,
        r_lag,
        'o-',
        lw=2.5,
        markersize=4,
        color='blue',
        zorder=3
    )
    ax.fill_between(lags, r_lag, alpha=0.2, color='steelblue')

    # Reference lines
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Zero lag')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.set_xlabel("Lag (hours)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Correlation (r)", fontsize=12, fontweight='bold')
    ax.set_title(
        f"Track {track_id}: Lagged Precipitation-Soil Moisture Correlation",
        fontsize=13,
        fontweight='bold',
        pad=10
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-1, 1])
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "lag_correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{int(track_id)}_lag_corr.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" >> [SAVED   ] Lag correlation plot: {plot_path}")


def analyze_flux_correlations(ds, track_df, track_id, buffer_km=100):
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
        print(f" >> [FLUX CORRELATIONS] Track {track_id}:")
        for k, v in correlations.items():
            print(f"      {k}: r={v['r']:.3f}, p={v['p']:.4f}, n={v['n']}")
    return results

# def plot_flux_analysis(results, output_dir):
#     if not results or 'time_series' not in results:
#         return
#     track_id = int(results['track_id'])
#     df = results['time_series']
#
#     # Ensure is_land column exists
#     if 'is_land' not in df.columns:
#         from utils.land_utils import is_point_land
#         df['is_land'] = [is_point_land(lat, lon) for lat, lon in zip(df['latitude'], df['longitude'])]
#
#     # Check if total_hf exists, otherwise calculate it
#     if 'total_hf' not in df.columns and all(col in df.columns for col in ['latent_hf', 'sensible_hf']):
#         print(" >  [INFO    ] Calculating total_hf")
#         df['total_hf'] = df['latent_hf'] + df['sensible_hf']
#
#     # 1. Time series plot with multiple variables
#     fig, axes = plt.subplots(3, 1, figsize=(14, 12))
#
#     # Precipitation and Soil Moisture
#     ax1 = axes[0]
#     ax1.plot(
#         df['time'],
#         df['tp'],
#         'b-',
#         label='Precipitation (mm/hr)',
#         linewidth=2
#     )
#     ax1.fill_between(df['time'], 0, 1, where=df['is_land'], color='bisque', alpha=0.3, transform=ax1.get_xaxis_transform(), label='Land')
#     ax1.fill_between(df['time'], 0, 1, where=~df['is_land'],color='azure', alpha=0.3, transform=ax1.get_xaxis_transform(), label='Ocean')
#     ax1.set_ylabel('Precipitation (mm/hr)', color='b', fontsize=11)
#     ax1.tick_params(axis='y', labelcolor='b')
#     ax1.grid(True, alpha=0.3)
#     ax1_twin = ax1.twinx()
#     ax1_twin.plot(df['time'], df['swvl1'], 'r-', label='Soil Moisture', linewidth=2)
#     ax1_twin.set_ylabel('Soil Moisture (m³/m³)', color='r', fontsize=11)
#     ax1_twin.tick_params(axis='y', labelcolor='r')
#     # Heat Fluxes
#     ax2 = axes[1]
#     ax2.plot(
#         df['time'],
#         df['latent_hf'],
#         'g-',
#         label='Latent Heat Flux',
#         linewidth=2
#     )
#     ax2.plot(df['time'], df['sensible_hf'], 'orange', label='Sensible Heat Flux', linewidth=2)
#     ax2.plot(df['time'], df['total_hf'], 'purple', label='Total Heat Flux', linewidth=3)
#     ax2.fill_between(df['time'], 0, 1, where=df['is_land'], color='bisque', alpha=0.3, transform=ax2.get_xaxis_transform(), label='Land')
#     ax2.fill_between(df['time'], 0, 1, where=~df['is_land'],color='azure', alpha=0.3, transform=ax2.get_xaxis_transform(), label='Ocean')
#     ax2.set_ylabel('Heat Flux (W/m²)', fontsize=11)
#     ax2.legend(fontsize=9)
#     ax2.grid(True, alpha=0.3)
#     # Scatter matrix
#     ax3 = axes[2]
#     scatter = ax3.scatter(
#         df['swvl1'],
#         df['total_hf'],
#         c=df['tp'],
#         cmap='viridis',
#         s=50,
#         alpha=0.7,
#         edgecolors='black'
#     )
#     ax3.set_xlabel('Soil Moisture (m³/m³)', fontsize=11)
#     ax3.set_ylabel('Total Heat Flux (W/m²)', fontsize=11)
#     ax3.grid(True, alpha=0.3)
#     cbar = plt.colorbar(scatter, ax=ax3)
#     cbar.set_label('Precipitation (mm/hr)', fontsize=10)
#     fig.suptitle(
#         f'Track {track_id}: Flux Analysis',
#         fontsize=14,
#         fontweight='bold'
#     )
#     plt.tight_layout()
#
#     os.makedirs(output_dir, exist_ok=True)
#     plot_path = os.path.join(output_dir, f"track_{track_id}_flux_analysis.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print(f" >> [SAVED   ] Heat Flux Plot: {os.path.basename(plot_path)}")
#

def plot_flux_analysis(
    results,
    output_dir,
    track_id,
):
    if not results or 'time_series' not in results:
        print(f" >  [WARNING ] No time series data for Track {track_id}")
        return
    df = results['time_series']
    if len(df) < 3:
        print(f" >  [WARNING ] Insufficient data points ({len(df)}) for Track {track_id}")
        return
    # Ensure is_land column exists
    if 'is_land' not in df.columns:
        df['is_land'] = False  # Default to ocean if not present

    track_dir = os.path.join(output_dir, f"track_{track_id}_analysis")
    os.makedirs(track_dir, exist_ok=True)

    # Helper function to add Land/Ocean shading
    def add_land_ocean_shading(ax, time_data, is_land_data):
        ax.fill_between(time_data, 0, 1, where=is_land_data, 
                        color='bisque', alpha=0.3, transform=ax.get_xaxis_transform(), label='Land')
        ax.fill_between(time_data, 0, 1, where=~is_land_data, 
                        color='azure', alpha=0.3, transform=ax.get_xaxis_transform(), label='Ocean')

    # PLOT 1: Precipitation & Soil Moisture
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.plot(df['time'], df['tp'], 'b-', label='Precipitation', linewidth=2)
    ax1.set_ylabel('Precipitation (mm/hr)', color='b')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['time'], df['swvl1'], 'r--', label='Soil Moisture', linewidth=2)
    ax1_twin.set_ylabel('Soil Moisture (m³/m³)', color='r')
    add_land_ocean_shading(ax1, df['time'], df['is_land'])
    plt.title(f"Track {track_id}: Precipitation & Soil Moisture")
    ax1.legend(loc='upper left')
    plot1_path = os.path.join(track_dir, f"track_{track_id}_precip_sm.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f" >> [SAVED   ] Precipitation & Soil Moisture Plot: {os.path.basename(plot1_path)}")

    # PLOT 2: Heat Fluxes (The one with the Land/Ocean Legend)
    plt.figure(figsize=(10, 5))
    ax2 = plt.gca()
    ax2.plot(df['time'], df['latent_hf'], 'g-', label='Latent Heat Flux', linewidth=2)
    ax2.plot(df['time'], df['sensible_hf'], 'orange', label='Sensible Heat Flux', linewidth=2)
    ax2.plot(df['time'], df['total_hf'], 'purple', label='Total Heat Flux', linewidth=3)
    add_land_ocean_shading(ax2, df['time'], df['is_land'])
    ax2.set_ylabel('Flux (W/m²)')
    ax2.set_xlabel('Time')
    ax2.legend(loc='upper right')
    plt.title(f"Track {track_id}: Surface Heat Flux Transitions")
    plot2_path = os.path.join(track_dir, f"track_{track_id}_heat_fluxes.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f" >> [SAVED   ] Surface Heat Flux Transitions Plot: {os.path.basename(plot2_path)}")

    # PLOT 3: Scatter Plot (Flux vs. Soil Moisture)
    plt.figure(figsize=(8, 6))
    plt.scatter(df['swvl1'], df['total_hf'], c=df['tp'], cmap='viridis', s=60, alpha=0.8, edgecolors='k')
    plt.colorbar(label='Precipitation (mm/hr)')
    plt.xlabel('Soil Moisture (m³/m³)')
    plt.ylabel('Total Heat Flux (W/m²)')
    plt.title(f"Track {track_id}: BOE Correlation")
    plt.grid(True, alpha=0.3)
    plot3_path = os.path.join(track_dir, f"track_{track_id}_flux_sm_scatter.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f" >> [SAVED   ] BOE Correlation plot: {os.path.basename(plot3_path)}")
