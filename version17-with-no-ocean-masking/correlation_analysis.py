import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr, linregress
from statsmodels.tsa.stattools import acf

import traceback

from utils import (
    std_print,
    is_point_land,
    find_landfall_time,
    add_land_ocean_shading,
    extract_spatial_subset,
    extract_regional_data,
    get_time_coord,
    export_results_to_csv,
)


# helper functions
def lag_correlation(x, y, max_lag=48):
    """Computes correlation between x and y with temporal offsets"""
    lags = range(-max_lag, max_lag + 1)
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
    """Compute autocorrelation function"""
    data_clean = np.array(data)
    data_clean = data_clean[~np.isnan(data_clean)]
    if len(data_clean) < nlags + 1:
        nlags = max(len(data_clean) // 2, 10)
    try:
        acf_values = acf(data_clean, nlags=nlags, fft=fft)
        lags = np.arange(len(acf_values))
        return acf_values, lags
    except Exception as e:
        print(f"Error computing ACF: {e}")
        return None, None


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


def extract_spatial_mean(ds, var_name, t, lat, lon, buffer_km):
    """Extract spatial mean of variable at point"""
    ds, time_coord = get_time_coord(ds)
    subset = extract_spatial_subset(
        ds, lat, lon, t, buffer_km=buffer_km, time_coord=time_coord
    )
    if subset is None or subset.sizes.get("latitude", 0) == 0:
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
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 20:
        return len(x_clean)
    try:
        acf_x = acf(x_clean, nlags=max_lag, fft=True)
        acf_y = acf(y_clean, nlags=max_lag, fft=True)
        rho1 = acf_x[1] if len(acf_x) > 1 else 0
        rho2 = acf_y[1] if len(acf_y) > 1 else 0
        n = len(x_clean)
        n_eff = n * (1 - rho1 * rho2) / (1 + rho1 * rho2)
        n_eff = max(min(n_eff, n), n / 10)
        return n_eff
    except:
        return len(x_clean)


def format_colorbar_time(cbar, t_numeric):
    """Format colorbar ticks with datetime strings - DD/MM/YYYY HH:MM format"""
    tick_locs = cbar.ax.get_yticks()
    tick_dates = pd.to_datetime(
        np.linspace(t_numeric.min(), t_numeric.max(), len(tick_locs))
    )
    cbar.ax.set_yticklabels([d.strftime("%d/%m/%Y %H:%M") for d in tick_dates])
    return cbar


def compute_single_track_correlation(ds, track_df, track_id, buffer_km=100):
    """Compute Pearson and Spearman correlation between TP and SWVL1"""
    track_id = int(track_id)
    std_print("INFO", "ca", f" --- Track {track_id} Computing correlation ---")
    if "is_land" not in track_df.columns:
        track_df["is_land"] = track_df.apply(
            lambda row: is_point_land(row["latitude"], row["longitude"]), axis=1
        )
    landfall_time = find_landfall_time(track_df)

    tp_times, tp_vals = extract_regional_data(ds, track_df, track_id, "tp", buffer_km)
    sm_times, sm_vals = extract_regional_data(
        ds, track_df, track_id, "swvl1", buffer_km
    )
    common_times = np.intersect1d(tp_times, sm_times)
    if len(common_times) < 3:
        return None

    tp_idx = np.isin(tp_times, common_times)
    sm_idx = np.isin(sm_times, common_times)

    tp_aligned = tp_vals[tp_idx] * 1000
    sm_aligned = sm_vals[sm_idx]
    if len(tp_aligned) < 3:
        return None
    if np.std(tp_aligned) == 0 or np.std(sm_aligned) == 0:
        std_print(
            "WARN", "ca", f"Track {track_id}", "Zero variance in one or both variables"
        )
        return None

    pearson_r, pearson_p = pearsonr(tp_aligned, sm_aligned)
    spearman_r, spearman_p = spearmanr(tp_aligned, sm_aligned)
    slope, intercept, r_value, p_value, std_err = linregress(tp_aligned, sm_aligned)
    r_squared_val = float(r_value) ** 2

    return {
        "track_id": track_id,
        "n_samples": len(tp_aligned),
        "mean_tp": float(np.mean(tp_aligned)),
        "mean_swvl1": float(np.mean(sm_aligned)),
        "std_tp_mm": float(np.std(tp_aligned)),
        "std_swvl1": float(np.std(sm_aligned)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
        "r_squared": r_squared_val,
        "p_value": p_value,
        "std_err": std_err,
        "times": common_times,
        "tp_values": tp_aligned,
        "swvl1_values": sm_aligned,
    }


def analyze_all_tracks(ds, full_track_df, buffer_km=100, output_dir="output"):
    """Compute correlations for all tracks"""
    std_print(
        "INFO", "ca", " --- Analyzing all tracks for soil water content patterns ---"
    )
    unique_tracks = sorted(full_track_df["track_id"].unique())
    all_results = []

    for track_id in unique_tracks:
        track_df = full_track_df[full_track_df["track_id"] == track_id].copy()
        result = compute_single_track_correlation(ds, track_df, track_id, buffer_km)
        if result:
            all_results.append(result)

    if not all_results:
        std_print(
            "WARN", "ca", "All Tracks", "No results from individual track analysis"
        )
        return None

    results_df = pd.DataFrame(all_results)

    summary = {
        "n_tracks": len(results_df),
        "mean_correlation": results_df["pearson_r"].mean(),
        "std_correlation": results_df["pearson_r"].std(),
        "tracks_with_positive_corr": (results_df["pearson_r"] > 0).sum(),
        "tracks_with_significant_corr": (results_df["pearson_p"] < 0.05).sum(),
        "mean_tp": results_df["mean_tp"].mean(),
        "mean_swvl1": results_df["mean_swvl1"].mean(),
    }

    std_print(
        "INFO",
        "ca",
        f"Soil Water Content Analysis across {summary['n_tracks']} Tracks :",
    )
    std_print(
        None,
        None,
        "Mean Correlation (TP vs SM)",
        f"{summary['mean_correlation']:.3f} ± {summary['std_correlation']:.3f}",
    )
    std_print(
        None,
        None,
        "Positive Correlations",
        f"{summary['tracks_with_positive_corr']}/{summary['n_tracks']}",
    )
    std_print(
        None,
        None,
        "Significant Correlations (p < 0.05)",
        f"{summary['tracks_with_significant_corr']}/{summary['n_tracks']}",
    )
    std_print(None, None, "Mean TP", f"{summary['mean_tp']:.2f} mm/hr")
    std_print(None, None, "Mean SM", f"{summary['mean_swvl1']:.3f} m³/m³")
    export_results_to_csv(results_df, output_dir, "sm_analysis", "sm")
    return {"summary": summary, "detailed_results": results_df}


# time series
def plot_tp_swvl1_time_series(
    tp,
    swvl1,
    valid_times,
    landfall_index,
    track_id,
    output_dir,
    originates_over_land=True,
    track_df=None,
):
    track_id = int(track_id)
    std_print("INFO", "ca", f" --- Track {track_id} TP-SWVL1 Time series plot ---")
    try:
        if not isinstance(tp, np.ndarray):
            tp = np.array(tp)
        if not isinstance(swvl1, np.ndarray):
            swvl1 = np.array(swvl1)
        if not isinstance(valid_times, (list, np.ndarray, pd.DatetimeIndex)):
            raise ValueError("valid_times must be datetime-like")

        valid_data_mask = ~(np.isnan(tp) | np.isnan(swvl1))
        valid_count = np.sum(valid_data_mask)
        if valid_count < 2:
            std_print(
                "WARN",
                "ca",
                f"Track {track_id} Insufficient valid data",
                f"only {valid_count} points",
            )
            return False
        if len(tp) != len(swvl1) or len(tp) != len(valid_times):
            std_print("WARN", "ca", f"Track {track_id}", "Data length mismatch")
            return False

        fig, ax1 = plt.subplots(figsize=(18, 6))
        # Determine landfall status
        if originates_over_land:
            status_text = "Originates over land"
            status_color = "blue"
        elif landfall_index == -1:
            status_text = "No Landfall"
            status_color = "gray"
        else:
            status_text = "Landfall"
            status_color = "red"

        # Get land array for shading
        if track_df is not None and "is_land" in track_df.columns:
            is_land_array = track_df["is_land"].values
        else:
            is_land_array = np.zeros(len(valid_times), dtype=bool)
            if originates_over_land:
                is_land_array[:] = True
            elif landfall_index > 0:
                is_land_array[landfall_index:] = True

        # Plot TP on primary axis
        color_tp = "tab:blue"
        ax1.set_xlabel("Time", fontsize=11, fontweight="bold")
        ax1.set_ylabel(
            "Total Precipitation (tp) (mm/hr)",
            color=color_tp,
            fontsize=10,
            fontweight="bold",
        )
        line1 = ax1.plot(
            valid_times,
            tp,
            "-o",
            color=color_tp,
            linewidth=2.5,
            markersize=5,
            label="Total Precipitation",
            alpha=0.8,
            zorder=3,
        )
        ax1.tick_params(axis="y", labelcolor=color_tp)
        ax1.grid(True, alpha=0.5, linestyle="--", zorder=0)
        add_land_ocean_shading(
            ax1,
            pd.to_datetime(valid_times),
            is_land_array,
            land_color="wheat",
            ocean_color="lightblue",
        )
        if (
            landfall_index != -1
            and landfall_index < len(valid_times)
            and not originates_over_land
        ):
            landfall_time = valid_times[landfall_index]
            ax1.axvline(
                x=landfall_time,
                color=status_color,
                linestyle="--",
                linewidth=2.5,
                alpha=0.8,
                label=status_text,
                zorder=4,
            )

        # Plot SWVL1 on secondary axis
        ax2 = ax1.twinx()
        color_swvl1 = "tab:brown"
        ax2.set_ylabel(
            "Soil Water Volumetric Layer 1 (swvl1) (m³/m³)",
            color=color_swvl1,
            fontsize=11,
            fontweight="bold",
        )
        line2 = ax2.plot(
            valid_times,
            swvl1,
            "s-",
            color=color_swvl1,
            linewidth=2.5,
            markersize=5,
            label="Soil Water Volumetric Level-1",
            alpha=0.8,
            zorder=3,
        )
        ax2.tick_params(axis="y", labelcolor=color_swvl1)

        # Set y-limits to include zero on both axes
        tp_clean = tp[~np.isnan(tp)]
        swvl1_clean = swvl1[~np.isnan(swvl1)]
        tp_max = np.max(tp_clean) if len(tp_clean) > 0 else 1
        swvl1_max = np.max(swvl1_clean) if len(swvl1_clean) > 0 else 0.1
        tp_pad = (tp_max) * 0.1 if tp_max > 0 else 0.5
        swvl1_pad = (swvl1_max) * 0.1 if swvl1_max > 0 else 0.05

        ax1.set_ylim(bottom=0, top=tp_max + tp_pad)
        ax2.set_ylim(bottom=0, top=swvl1_max + swvl1_pad)
        ax1.axhline(
            y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.3, zorder=1
        )
        ax2.axhline(
            y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.3, zorder=1
        )

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        if (
            landfall_index != -1
            and landfall_index < len(valid_times)
            and not originates_over_land
        ):
            lines.append(
                plt.Line2D([0], [0], color=status_color, linestyle="--", linewidth=2.5)
            )
            labels.append(status_text)

        lines.append(Patch(facecolor="lightblue", alpha=0.5, label="Ocean Phase"))
        lines.append(Patch(facecolor="wheat", alpha=0.5, label="Land Phase"))
        labels += ["Ocean Phase", "Land Phase"]

        ax1.legend(
            lines,
            labels,
            loc="upper left",
            fontsize=10,
            framealpha=0.95,
            edgecolor="none",
        )

        title = f"Track {track_id}: Precipitation & Soil Water Content Evolution"
        if originates_over_land:
            title += "\n(Land-Originated MD)"
        elif landfall_index != -1:
            title += "\n(Ocean-Originated, Landfall Detected)"
        else:
            title += "\n(Ocean-Originated, No Landfall)"
        ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)

        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax1.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%d/%m/%Y %H:%M")
        )
        plt.tight_layout()

        track_dir = os.path.join(output_dir, "time_series")
        os.makedirs(track_dir, exist_ok=True)
        plot_path = os.path.join(
            track_dir, f"track_{track_id}_TP-SWVL1_time_series.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "ca", "TP-SWVL1 time series", plot_path)
        return True
    except Exception as e:
        std_print("ERROR", "ca", f"Track {track_id} TP-SWVL1 plot failed", str(e))
        traceback.print_exc()
        return False


# correlation scatter with time gradient
def plot_tp_swvl1_correlation(
    ds, track_df, track_id, buffer_km=100, output_dir="output"
):
    result = compute_single_track_correlation(ds, track_df, track_id, buffer_km)
    std_print(
        "INFO", "ca", f" --- Track {track_id} TP-SWVL1 Correlation scatter plot ---"
    )
    if result is None:
        std_print(
            "WARN", "ca", f"Track {track_id}", "Insufficient data for correlation plot"
        )
        return

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, width_ratios=[4, 0.15, 1], wspace=0.3)
    ax_main = plt.subplot(gs[0])
    ax_cbar = plt.subplot(gs[1])
    ax_stats = plt.subplot(gs[2])
    ax_stats.axis("off")

    times = result["times"]
    t_numeric = pd.to_datetime(times).astype(np.int64)
    tp_vals = result["tp_values"]
    sm_vals = result["swvl1_values"]

    sc = ax_main.scatter(
        tp_vals,
        sm_vals,
        c=t_numeric,
        cmap="viridis",
        alpha=0.8,
        s=100,
        edgecolors="black",
        linewidth=0.8,
        zorder=3,
    )
    cbar = plt.colorbar(sc, cax=ax_cbar)
    cbar.set_label("Track Time", fontsize=10, fontweight="bold")
    format_colorbar_time(cbar, t_numeric)

    # Statistics box
    stats_text = (
        f"Pearson r:  {result['pearson_r']:.4f}\n"
        f"Spearman ρ: {result['spearman_r']:.4f}\n"
        f"P-value:    {result['pearson_p']:.4e}\n"
        f"N samples:  {result['n_samples']}\n"
        f"Mean TP:    {result['mean_tp']:.2f} mm/hr\n"
        f"Mean SM:    {result['mean_swvl1']:.2f} m³/m³"
    )
    bbox_props = dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        alpha=0.9,
        edgecolor="gray",
        linewidth=1,
    )
    ax_stats.text(
        0.1,
        0.95,
        stats_text,
        transform=ax_stats.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        bbox=bbox_props,
    )

    ax_main.set_xlabel("Total Precipitation (mm/hr)", fontsize=11, fontweight="bold")
    ax_main.set_ylabel(
        "Soil Water Volumetric Level-1 (m³/m³)", fontsize=11, fontweight="bold"
    )
    ax_main.set_title(
        f"Track {track_id}: TP-SWVL1 Correlation",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax_main.grid(True, alpha=0.5, linestyle="--")
    plt.suptitle("", fontsize=12)
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_TP-SWVL1_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", "TP-SWVL1 Correlation scatter plot", plot_path)


def plot_swvl1_hf_correlation(
    swvl1,
    latent_hf,
    sensible_hf,
    valid_times,
    landfall_index,
    track_id,
    output_dir,
    originates_over_land=True,
):
    track_id = int(track_id)
    std_print(
        "INFO", "ca", f" --- Track {track_id} SWVL1-HF Correlation scatter plot ---"
    )
    total_hf = latent_hf + sensible_hf
    valid_mask = ~(
        np.isnan(swvl1)
        | np.isnan(latent_hf)
        | np.isnan(sensible_hf)
        | np.isnan(total_hf)
    )
    if np.sum(valid_mask) < 5:
        std_print(
            "WARN", "ca", f"Track {track_id}", "Insufficient data for HF correlation"
        )
        return False

    t_numeric = pd.to_datetime(valid_times).astype(np.int64)
    t_plot = t_numeric[valid_mask]

    # Figure
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 4, width_ratios=[4, 4, 4, 0.3], wspace=0.25)
    axes = [plt.subplot(gs[i]) for i in range(3)]
    ax_cbar = plt.subplot(gs[3])
    y_vars = [latent_hf, sensible_hf, total_hf]
    y_labels = [
        "Latent Heat Flux (W/m²)",
        "Sensible Heat Flux (W/m²)",
        "Total Heat Flux (W/m²)",
    ]
    titles = [
        "SWVL1 ↔ Latent Heat Flux",
        "SWVL1 ↔ Sensible Heat Flux",
        "SWVL1 ↔ Total Heat Flux",
    ]
    colors = ["green", "orange", "purple"]
    vmin, vmax = t_plot.min(), t_plot.max()
    sc = None
    for i, ax in enumerate(axes):
        sc = ax.scatter(
            swvl1[valid_mask],
            y_vars[i][valid_mask],
            c=t_plot,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            s=90,
            edgecolors="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.set_xlabel("SWVL1 (m³/m³)", fontsize=11, fontweight="bold")
        ax.set_ylabel(y_labels[i], fontsize=11, fontweight="bold")
        ax.set_title(titles[i], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.5, linestyle="--")
        # try:
        #     r, p = pearsonr(swvl1[valid_mask], y_vars[i][valid_mask])
        #     ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.4f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round',  facecolor='white', alpha=0.8, edgecolor='gray'))
        # except:
        #     pass

    cbar = plt.colorbar(sc, cax=ax_cbar)
    cbar.set_label("Track Time", fontsize=11, fontweight="bold")
    format_colorbar_time(cbar, t_numeric)

    status_text = "Land-Originated" if originates_over_land else "After Landfall"
    fig.suptitle(
        f"Track {track_id}: SWVL1 ↔ Surface Heat Flux Correlation ({status_text})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_SWVL1-HF_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", "SWVL1-HF Correlation scatter plot", plot_path)
    return True


def plot_tp_hf_correlation(
    tp,
    latent_hf,
    sensible_hf,
    valid_times,
    landfall_index,
    track_id,
    output_dir,
    originates_over_land=True,
):
    track_id = int(track_id)
    std_print("INFO", "ca", f" --- Track {track_id} TP-HF Correlation scatter plot ---")
    total_hf = latent_hf + sensible_hf
    valid_mask = ~(
        np.isnan(tp) | np.isnan(latent_hf) | np.isnan(sensible_hf) | np.isnan(total_hf)
    )
    if np.sum(valid_mask) < 5:
        std_print(
            "WARN", "ca", f"Track {track_id}", "Insufficient data for TP-HF correlation"
        )
        return False

    t_numeric = pd.to_datetime(valid_times).astype(np.int64)
    t_plot = t_numeric[valid_mask]

    # Figure
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 4, width_ratios=[4, 4, 4, 0.3], wspace=0.25)
    axes = [plt.subplot(gs[i]) for i in range(3)]
    ax_cbar = plt.subplot(gs[3])
    y_vars = [latent_hf, sensible_hf, total_hf]
    y_labels = [
        "Latent Heat Flux (W/m²)",
        "Sensible Heat Flux (W/m²)",
        "Total Heat Flux (W/m²)",
    ]
    titles = [
        "TP ↔ Latent Heat Flux",
        "TP ↔ Sensible Heat Flux",
        "TP ↔ Total Heat Flux",
    ]
    vmin, vmax = t_plot.min(), t_plot.max()
    sc = None
    for i, ax in enumerate(axes):
        sc = ax.scatter(
            tp[valid_mask],
            y_vars[i][valid_mask],
            c=t_plot,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
            s=90,
            edgecolors="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.set_xlabel("Total Precipitation (mm/hr)", fontsize=11, fontweight="bold")
        ax.set_ylabel(y_labels[i], fontsize=11, fontweight="bold")
        ax.set_title(titles[i], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.5, linestyle="--")
        # try:
        #     r, p = pearsonr(tp[valid_mask], y_vars[i][valid_mask])
        #     ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.4f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round',
        #            facecolor='white', alpha=0.8, edgecolor='gray'))
        # except:
        #     pass

    cbar = plt.colorbar(sc, cax=ax_cbar)
    cbar.set_label("Track Time", fontsize=11, fontweight="bold")
    format_colorbar_time(cbar, t_numeric)
    status_text = "Land-Originated" if originates_over_land else "After Landfall"
    fig.suptitle(
        f"Track {track_id}: TP ↔ Surface Heat Flux Correlation ({status_text})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_TP-HF_correlation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", "TP-HF Correlation scatter plot", plot_path)
    return True


# lag correlation
def plot_tp_swvl1_lag_correlation(tp, sm, max_lag_hours, track_id, output_dir):
    """
    lag correlation between TP and SWVL1
     - Negative lag: TP leads SWVL1 (infiltration)
     - Positive lag: SWVL1 leads TP (antecedent moisture)
    """
    track_id = int(track_id)
    std_print("INFO", "ca", f" --- Track {track_id} TP-SWVL1 Lag-Correlation plot ---")
    lags, correlations_lag = lag_correlation(tp, sm, max_lag=max_lag_hours)
    if len(lags) != len(correlations_lag):
        min_len = min(len(lags), len(correlations_lag))
        lags = lags[:min_len]
        correlations_lag = correlations_lag[:min_len]

    valid_indices = [i for i, val in enumerate(correlations_lag) if not np.isnan(val)]
    if len(valid_indices) == 0:
        std_print(
            "WARN",
            "ca",
            f"Track {track_id}",
            "No valid correlation values for lag plot",
        )
        return

    lags = [lags[i] for i in valid_indices]
    correlations_lag = [correlations_lag[i] for i in valid_indices]

    max_corr_idx = np.argmax(np.abs(correlations_lag))
    max_corr_lag = lags[max_corr_idx]
    if max_corr_lag < 0:
        lag_interpretation = (
            f"TP leads SWVL1 by {abs(max_corr_lag)}h\n(Infiltration lag)"
        )
    elif max_corr_lag > 0:
        lag_interpretation = f"SWVL1 leads TP by {max_corr_lag}h\n(Antecedent moisture)"
    else:
        lag_interpretation = "Simultaneous\n(Immediate feedback)"

    fig, ax = plt.subplots(figsize=(12, 6))
    bbox = dict(
        boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"
    )

    ax.plot(lags, correlations_lag, "o-", lw=2.5, markersize=4, color="blue", zorder=3)
    ax.fill_between(lags, correlations_lag, alpha=0.2, color="steelblue")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(
        0.98,
        0.94,
        lag_interpretation,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=bbox,
    )
    ax.set_xlabel("Lag (hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Correlation (r)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: Lagged TP-SWVL1 Correlation",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.axhline(y=0.2, color="red", linestyle=":", alpha=0.5)
    ax.axhline(y=-0.2, color="red", linestyle=":", alpha=0.5)
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "lag_correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(
        track_dir, f"track_{track_id}_TP-SWVL1_lag_correlation.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", "TP-SWVL1 Lag correlation plot", plot_path)


def plot_swvl1_hf_lag_correlation(sm, hf, max_lag_hours, track_id, hf_name, output_dir):
    """
    lag correlation between SWVL1 and Heat Flux
     - Negative lag: SM leads HF (Brown Ocean Effect)
     - Positive lag: HF leads SM (Surface Drying)
    """
    track_id = int(track_id)
    std_print(
        "INFO", "ca", f" --- Track {track_id} SWVL1-{hf_name} Lag-Correlation plot ---"
    )
    lags, correlations_lag = lag_correlation(sm, hf, max_lag=max_lag_hours)
    valid_indices = [i for i, val in enumerate(correlations_lag) if not np.isnan(val)]
    if not valid_indices:
        std_print(
            "WARN",
            "ca",
            f"Track {track_id}",
            f"No valid data for SWVL1-{hf_name} lag plot",
        )
        return

    lags = [lags[i] for i in valid_indices]
    correlations_lag = [correlations_lag[i] for i in valid_indices]

    # Color scheme based on flux type
    if "LHF" in hf_name.upper() or "LATENT" in hf_name.upper():
        line_color, fill_color = "forestgreen", "lightgreen"
    elif "SHF" in hf_name.upper() or "SENSIBLE" in hf_name.upper():
        line_color, fill_color = "darkorange", "navajowhite"
    elif "THF" in hf_name.upper() or "TOTAL" in hf_name.upper():
        line_color, fill_color = "rebeccapurple", "thistle"
    else:
        line_color, fill_color = "blue", "lightblue"

    max_corr_idx = np.argmax(np.abs(correlations_lag))
    max_corr_lag = lags[max_corr_idx]

    if max_corr_lag < 0:
        interp = f"SWVL1 leads {hf_name} by {abs(max_corr_lag)}h\n(Energy Feedback)"
    elif max_corr_lag > 0:
        interp = f"{hf_name} leads SWVL1 by {max_corr_lag}h\n(Surface Drying)"
    else:
        interp = "Simultaneous\n(Coupled Phase)"

    fig, ax = plt.subplots(figsize=(12, 6))
    bbox = dict(
        boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"
    )

    ax.plot(
        lags, correlations_lag, "o-", lw=2.5, markersize=4, color=line_color, zorder=3
    )
    ax.fill_between(lags, correlations_lag, alpha=0.2, color=fill_color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(
        0.98,
        0.94,
        interp,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=bbox,
    )
    ax.set_xlabel("Lag (hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Correlation (r)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: Lagged SWVL1-{hf_name} Correlation",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.axhline(y=0.2, color="red", linestyle=":", alpha=0.5)
    ax.axhline(y=-0.2, color="red", linestyle=":", alpha=0.5)
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "lag_correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(
        track_dir, f"track_{track_id}_SWVL1-{hf_name}_lag_correlation.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", f"SWVL1-{hf_name} Lag plot", plot_path)
    return True


def plot_tp_hf_lag_correlation(
    flux, tp, max_lag_hours, track_id, flux_name, output_dir
):
    """
    Plot lag correlation between Heat Flux and TP
     - Negative lag: Flux leads TP (energy influences precipitation)
     - Positive lag: TP leads Flux (rainfall affects energy flux)
    """
    track_id = int(track_id)
    std_print(
        "INFO", "ca", f" --- Track {track_id} {flux_name}-TP Lag-Correlation plot ---"
    )
    lags, correlations_lag = lag_correlation(flux, tp, max_lag=max_lag_hours)
    valid_indices = [i for i, val in enumerate(correlations_lag) if not np.isnan(val)]
    if not valid_indices:
        std_print(
            "WARN",
            "ca",
            f"Track {track_id}",
            f"No valid data for {flux_name}-TP lag plot",
        )
        return

    lags = [lags[i] for i in valid_indices]
    correlations_lag = [correlations_lag[i] for i in valid_indices]

    if "LHF" in flux_name.upper() or "LATENT" in flux_name.upper():
        line_color, fill_color = "forestgreen", "lightgreen"
        flux_label = "Latent Heat Flux"
    elif "SHF" in flux_name.upper() or "SENSIBLE" in flux_name.upper():
        line_color, fill_color = "darkorange", "navajowhite"
        flux_label = "Sensible Heat Flux"
    elif "THF" in flux_name.upper() or "TOTAL" in flux_name.upper():
        line_color, fill_color = "rebeccapurple", "thistle"
        flux_label = "Total Heat Flux"
    else:
        line_color, fill_color = "blue", "lightblue"
        flux_label = flux_name

    if correlations_lag:
        max_corr_idx = np.argmax(np.abs(correlations_lag))
        max_corr_lag = lags[max_corr_idx]
        if max_corr_lag < 0:
            interp = f"{flux_label} leads TP by {abs(max_corr_lag)}h\n(Energy influences precipitation)"
        elif max_corr_lag > 0:
            interp = f"TP leads {flux_label} by {max_corr_lag}h\n(Rainfall affects energy flux)"
        else:
            interp = "Simultaneous\n(Strong coupling)"
    else:
        interp = "No significant correlation"

    fig, ax = plt.subplots(figsize=(12, 6))
    bbox = dict(
        boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"
    )

    ax.plot(
        lags, correlations_lag, "o-", lw=2.5, markersize=4, color=line_color, zorder=3
    )
    ax.fill_between(lags, correlations_lag, alpha=0.2, color=fill_color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(
        0.98,
        0.94,
        interp,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=bbox,
    )
    ax.set_xlabel("Lag (hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Correlation (r)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: Lagged {flux_label} vs Total Precipitation",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.axhline(y=0.2, color="red", linestyle=":", alpha=0.5)
    ax.axhline(y=-0.2, color="red", linestyle=":", alpha=0.5)
    plt.tight_layout()

    track_dir = os.path.join(output_dir, "lag_correlation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(
        track_dir, f"track_{track_id}_TP-{flux_name}_lag_correlation.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", f"TP-{flux_name} Lag plot", plot_path)
    return True


def plot_lagged_correlation_single_variable(
    data, var_name, track_id, output_dir="output"
):
    """Plot autocorrelation (lagged self-correlation) for a single variable"""
    values = np.array(data)
    values = values[~np.isnan(values)]
    if len(values) < 10:
        std_print(
            "WARN", "ca", f"Insufficient data for {var_name}: {len(values)} points"
        )
        return None

    max_lag = min(48, len(values) // 2)
    lags, correlations, pvalues = compute_lagged_correlation(values, max_lag=max_lag)
    valid = ~np.isnan(correlations)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        lags[valid],
        correlations[valid],
        "o-",
        linewidth=2.5,
        markersize=4,
        color="blue",
        label="Autocorrelation",
        zorder=3,
    )
    ax.fill_between(lags, correlations, alpha=0.2, color="steelblue")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Lag (hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Correlation Coefficient (r)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"{var_name}: Temporal Autocorrelation (Track {track_id})",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.set_ylim(-1.0, 1.05)
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()

    plot_dir = os.path.join(output_dir, "autocorrelation")
    os.makedirs(plot_dir, exist_ok=True)
    clean_name = var_name.replace(" ", "_").replace("(", "").replace(")", "")
    outpath = os.path.join(plot_dir, f"track_{track_id}_autocorr_{clean_name}.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", f"Autocorrelation: {var_name}", outpath)
    return lags, correlations, pvalues


# main analysis function
def analyze_single_track(
    ds, track_df, track_id, buffer_km, max_lag_hours=50, output_dir="output"
):
    """Main function to analyze a single track with all correlation analyses"""
    std_print(
        "DEBUG", "ca", f" --- Track {track_id} : Processing Correlation Analysis ---"
    )
    # Filter to single track
    if len(track_df["track_id"].unique()) == 1:
        current_track = track_df.sort_values("time").copy()
    else:
        current_track = (
            track_df[track_df["track_id"] == track_id].sort_values("time").copy()
        )
    if current_track.empty:
        std_print("ERROR", "ca", f"track_df is None for track {track_id}")
        traceback.print_exc()
        return None

    # Find landfall
    landfall_index = -1
    originates_over_land = True
    if "is_land" in current_track.columns:
        is_land_values = current_track["is_land"].values
        for i in range(1, len(is_land_values)):
            if not is_land_values[i - 1] and is_land_values[i]:
                landfall_index = i
                break
        originates_over_land = is_land_values[0] if len(is_land_values) > 0 else False

    # Check required variables
    required_vars = ["tp", "swvl1"]
    missing_vars = [var for var in required_vars if var not in ds.data_vars]
    if missing_vars:
        std_print("ERROR", "ca", "Missing variables in dataset", missing_vars)
        return None

    # Extract time series
    ds, time_coord = get_time_coord(ds)
    tp_series = []
    swvl1_series = []
    latent_hf_series = []
    sensible_hf_series = []
    total_hf_series = []
    valid_times = []

    for i, row in current_track.iterrows():
        t = pd.to_datetime(row["time"])
        lat, lon = row["latitude"], row["longitude"]

        subset_tp = extract_spatial_subset(ds, lat, lon, t, buffer_km, time_coord)
        tp = float(subset_tp["tp"].mean().values)
        subset_swvl1 = extract_spatial_subset(ds, lat, lon, t, buffer_km, time_coord)
        swvl1 = float(subset_swvl1["swvl1"].mean().values)

        subset_hf = extract_spatial_subset(ds, lat, lon, t, buffer_km, time_coord)
        latent_hf = float(subset_hf["avg_slhtf"].mean().values)
        sensible_hf = float(subset_hf["avg_ishf"].mean().values)
        total_hf = (
            latent_hf + sensible_hf
            if np.isfinite(latent_hf) and np.isfinite(sensible_hf)
            else np.nan
        )
        if np.isfinite(tp) and np.isfinite(swvl1):
            tp_series.append(tp)
            swvl1_series.append(swvl1)
            latent_hf_series.append(latent_hf)
            sensible_hf_series.append(sensible_hf)
            total_hf_series.append(total_hf)
            valid_times.append(t)
    if len(valid_times) < 4:
        std_print(
            "WARN",
            "ca",
            f"Track {track_id} Insufficient data",
            f"only {len(valid_times)} valid points",
        )
        return None

    # Unit conversions
    tp_series = np.array(tp_series) * 1000
    swvl1_series = np.array(swvl1_series)
    latent_hf_series = np.array(latent_hf_series)
    sensible_hf_series = np.array(sensible_hf_series)
    total_hf_series = np.array(total_hf_series)

    # Set SWVL1 = 0 before landfall (physically impossible over ocean)
    swvl1_series_modified = swvl1_series.copy()
    if landfall_index > 0:
        swvl1_series_modified[:landfall_index] = 0
        std_print(
            "INFO",
            "ca",
            f"Track {track_id}",
            f"SWVL1 set to 0 for ocean phase (indices 0-{landfall_index})",
        )
    else:
        std_print(
            "DEBUG", "ca", f"Track {track_id}", "No landfall - SWVL1 valid throughout"
        )

    # Generate all plots
    os.makedirs(output_dir, exist_ok=True)
    std_print("INFO", "ca", f" --- Generating plots for Track {track_id} ---")

    # 1. TP-SWVL1 Feedback
    try:
        plot_tp_swvl1_time_series(
            tp_series,
            swvl1_series_modified,
            valid_times,
            landfall_index,
            track_id,
            output_dir,
            originates_over_land,
        )
        plot_tp_swvl1_correlation(ds, track_df, track_id, output_dir=output_dir)
        plot_tp_swvl1_lag_correlation(
            tp_series, swvl1_series_modified, max_lag_hours, track_id, output_dir
        )
    except Exception as e:
        std_print(
            "ERROR", "ca", f"TP-SWVL1 analysis failed for Track {track_id}", str(e)
        )

    # 2. SWVL1-Heat Flux Feedback
    try:
        if not np.all(np.isnan(latent_hf_series)) or not np.all(
            np.isnan(sensible_hf_series)
        ):
            plot_swvl1_hf_correlation(
                swvl1_series_modified,
                latent_hf_series,
                sensible_hf_series,
                valid_times,
                landfall_index,
                track_id,
                output_dir,
                originates_over_land,
            )
            plot_swvl1_hf_lag_correlation(
                swvl1_series_modified,
                latent_hf_series,
                max_lag_hours,
                track_id,
                "LHF",
                output_dir,
            )
            plot_swvl1_hf_lag_correlation(
                swvl1_series_modified,
                sensible_hf_series,
                max_lag_hours,
                track_id,
                "SHF",
                output_dir,
            )
            plot_swvl1_hf_lag_correlation(
                swvl1_series_modified,
                total_hf_series,
                max_lag_hours,
                track_id,
                "THF",
                output_dir,
            )
    except Exception as e:
        std_print(
            "ERROR",
            "ca",
            f"SWVL1-Heat Flux lag analysis failed for Track {track_id}",
            str(e),
        )

    # 3. TP-Heat Flux Feedback
    try:
        if not np.all(np.isnan(latent_hf_series)) or not np.all(
            np.isnan(sensible_hf_series)
        ):
            plot_tp_hf_correlation(
                tp_series,
                latent_hf_series,
                sensible_hf_series,
                valid_times,
                landfall_index,
                track_id,
                output_dir,
                originates_over_land,
            )
            plot_tp_hf_lag_correlation(
                tp_series, latent_hf_series, max_lag_hours, track_id, "LHF", output_dir
            )
            plot_tp_hf_lag_correlation(
                tp_series,
                sensible_hf_series,
                max_lag_hours,
                track_id,
                "SHF",
                output_dir,
            )
            plot_tp_hf_lag_correlation(
                tp_series, total_hf_series, max_lag_hours, track_id, "THF", output_dir
            )
    except Exception as e:
        std_print(
            "ERROR",
            "ca",
            f"TP-Heat Flux lag analysis failed for Track {track_id}",
            str(e),
        )

    # 4. Autocorrelation Analysis
    try:
        std_print(
            "DEBUG",
            "ca",
            f" --- Generating Autocorrelation plots for Track {track_id} ---",
        )
        plot_lagged_correlation_single_variable(
            tp_series, var_name="tp", track_id=track_id, output_dir=output_dir
        )
        plot_lagged_correlation_single_variable(
            swvl1_series_modified,
            var_name="swvl1",
            track_id=track_id,
            output_dir=output_dir,
        )
        plot_lagged_correlation_single_variable(
            latent_hf_series,
            var_name="latent_hf",
            track_id=track_id,
            output_dir=output_dir,
        )
        plot_lagged_correlation_single_variable(
            sensible_hf_series,
            var_name="sensible_hf",
            track_id=track_id,
            output_dir=output_dir,
        )
        plot_lagged_correlation_single_variable(
            total_hf_series,
            var_name="total_hf",
            track_id=track_id,
            output_dir=output_dir,
        )
    except Exception as e:
        std_print(
            "ERROR",
            "ca",
            f"Autocorrelation analysis failed for Track {track_id}",
            str(e),
        )
        traceback.print_exc()

    # Compute correlation ONLY on land phase
    corr, p_val = np.nan, np.nan
    corr_pre, corr_post = np.nan, np.nan
    if landfall_index > 0 and len(tp_series[landfall_index:]) >= 5:
        tp_land = tp_series[landfall_index:]
        sm_land = swvl1_series_modified[landfall_index:]
        if np.std(tp_land) > 0 and np.std(sm_land) > 0:
            corr, p_val = pearsonr(tp_land, sm_land)
            corr_post = corr
            std_print(
                "INFO",
                "ca",
                f"Track {track_id}",
                f"TP-SWVL1 Correlation (LAND PHASE ONLY): r={corr:.3f}, p={p_val:.4f}",
            )
        corr_pre = np.nan
    elif originates_over_land:
        if np.std(tp_series) > 0 and np.std(swvl1_series_modified) > 0:
            corr, p_val = pearsonr(tp_series, swvl1_series_modified)
            corr_post = corr
            std_print(
                "INFO",
                "ca",
                f"Track {track_id}",
                f"TP-SWVL1 Correlation (Land-Originated, Full Track): r={corr:.3f}, p={p_val:.4f}",
            )
    else:
        std_print(
            "ERROR",
            "ca",
            f"Track {track_id}",
            "No landfall - TP-SWVL1 correlation NOT computed",
        )

    # Store results
    correlation_results = {
        "track_id": track_id,
        "correlation": corr if not np.isnan(corr) else np.nan,
        "p_value": p_val if not np.isnan(p_val) else np.nan,
        "corr_pre": corr_pre,
        "corr_post": corr_post,
        "mean_tp": np.mean(tp_series),
        "mean_swvl1": np.mean(swvl1_series_modified),
        "n_samples": len(valid_times),
        "landfall_time": valid_times[landfall_index] if landfall_index != -1 else None,
        "landfall_index": landfall_index,
        "originates_over_land": originates_over_land,
    }

    std_print("INFO", "ca", f"Correlation for Track {track_id}:")
    std_print(None, None, f"r={correlation_results['correlation']:.3f}")
    std_print(None, None, f"p={correlation_results['p_value']:.4f}")
    std_print(None, None, f"n={len(valid_times)}")
    return correlation_results
