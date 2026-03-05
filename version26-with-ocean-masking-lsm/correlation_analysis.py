import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr, linregress
from statsmodels.tsa.stattools import acf

import traceback

from utils import (
    std_print,
    is_point_land,
    extract_spatial_subset,
    extract_regional_data,
    get_time_coord,
    export_results_to_csv,
    common_bbox,
    legend_style,
)


def plot_rolling_correlation(
    tp_series,
    swvl1_series,
    latent_hf_series,
    sensible_hf_series,
    valid_times,
    landfall_index,
    track_id,
    output_dir,
    window_hours=24,
):
    """
    Pearson r in a sliding window along the track.

    Shows HOW coupling between SM, flux and precipitation evolves across the track lifetime

    Two pairs plotted:
      - SWVL1 <-> LHF  (moisture -> evaporation coupling)
      - SWVL1 <-> TP   (antecedent moisture -> rainfall)
    """
    track_id = int(track_id)
    # Estimate timestep from valid_times (assumes uniform 6h ERA5)
    times_dt = pd.to_datetime(valid_times)
    if len(times_dt) > 1:
        dt_hours = (times_dt[1] - times_dt[0]).total_seconds() / 3600
    else:
        dt_hours = 6.0
    window = max(3, int(round(window_hours / dt_hours)))  # samples per window

    n = len(tp_series)
    if n < window + 2:
        std_print("WARN", "ca", "Too few points for rolling correlation")
        return

    r_swvl1_lhf = rolling_r(swvl1_series, latent_hf_series, n, window)
    r_swvl1_shf = rolling_r(swvl1_series, sensible_hf_series, n, window)
    r_swvl1_tp = rolling_r(swvl1_series, tp_series, n, window)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=100, sharex=True)
    time_axis = np.arange(n)  # index; x-ticks formatted below

    pairs = [
        (axes[0], r_swvl1_lhf, "SWVL1 ↔ LHF", "forestgreen", "lightgreen"),
        (axes[0], r_swvl1_shf, "SWVL1 ↔ SHF", "darkorange", "navajowhite"),
        (axes[1], r_swvl1_tp, "SWVL1 ↔ TP", "steelblue", "lightblue"),
    ]
    for ax, r_vals, label, color, fill in pairs:
        ax.plot(
            time_axis,
            r_vals,
            "-o",
            color=color,
            lw=2,
            markersize=3,
            label=label,
            zorder=3,
        )
        ax.fill_between(time_axis, r_vals, 0, alpha=0.18, color=fill)

    for ax in axes:
        ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.6)
        ax.axhline(0.5, color="red", lw=0.7, linestyle=":", alpha=0.5) # Threshold line at +0.5
        ax.axhline(-0.5, color="red", lw=0.7, linestyle=":", alpha=0.5) # Threshold line at -0.5
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Pearson r", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.4, linestyle="--")
        legend_handles, leg_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(leg_labels, legend_handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower left", fontsize=9)

    # Landfall line
    if landfall_index > 0:
        landfall_time = pd.to_datetime(valid_times[landfall_index]).strftime("%d/%m %H:%M")
        for ax in axes:
            ax.axvline(landfall_index, color="black", lw=1.5, linestyle="-", zorder=5)
            ax.text(
                landfall_index + 0.2,
                0.92,
                f"Landfall\n{landfall_time}",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                va="top",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
            )

    # x-axis tick labels: every 4th point as dd/mm HH:MM
    tick_step = max(1, n // 8)
    tick_idx = range(0, n, tick_step)
    tick_labs = [
        pd.to_datetime(valid_times[i]).strftime("%d/%m\n%H:%M") for i in tick_idx
    ]
    axes[1].set_xticks(list(tick_idx))
    axes[1].set_xticklabels(tick_labs, fontsize=8)
    axes[1].set_xlabel("Track Time", fontsize=11, fontweight="bold")

    axes[0].set_title(
        f"Track {track_id}: {window_hours}h Rolling Pearson Correlation\n"
        f"(SWVL1 ↔ Surface Fluxes)",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_title(
        f"Track {track_id}: {window_hours}h Rolling Pearson Correlation\n"
        f"(SWVL1 ↔ Total Precipitation)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    plot_dir = os.path.join(output_dir, "correlation")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(
        plot_dir, f"track_{track_id}_rolling_correlation_{window_hours}h.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", f"Rolling {window_hours}h correlation", out_path)


def plot_correlation_matrix(
    tp_series,
    swvl1_series,
    latent_hf_series,
    sensible_hf_series,
    # total_hf_series,
    valid_times,
    landfall_index,
    track_id,
    output_dir,
):
    """
    Pearson correlation matrix across all variable pairs (land phase only).

    output - r-value matrix with p-value significance stars.
    Significance codes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1
    """
    track_id = int(track_id)
    # Land phase only
    if landfall_index > 0:
        sl = slice(landfall_index, None)
        phase_label = "Post-Landfall (Land Phase)"
    else:
        sl = slice(None)
        phase_label = "Full Track (Land-Originated)"

    data = {
        "TP\n(mm/hr)": tp_series[sl],
        "SWVL1\n(m³/m³)": swvl1_series[sl],
        "LHF\n(W/m²)": latent_hf_series[sl],
        "SHF\n(W/m²)": sensible_hf_series[sl],
        # "THF\n(W/m²)": total_hf_series[sl],
    }
    labels = list(data.keys())
    n = len(labels)
    arrays = [np.array(v, dtype=float) for v in data.values()]

    # Compute r-matrix and p-matrix
    r_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            # Mask NaNs jointly
            mask = ~(np.isnan(arrays[i]) | np.isnan(arrays[j]))
            if mask.sum() < 5:
                continue
            if np.std(arrays[i][mask]) == 0 or np.std(arrays[j][mask]) == 0:
                continue
            r, p = pearsonr(arrays[i][mask], arrays[j][mask])
            r_matrix[i, j] = r
            p_matrix[i, j] = p

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    im = ax.imshow(r_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Cell annotations: r value + significance star
    for i in range(n):
        for j in range(n):
            r_val = r_matrix[i, j]
            star = sig_star(p_matrix[i, j])
            if np.isnan(r_val):
                txt = "N/A"
                color = "gray"
            else:
                txt = f"{r_val:.3f}{star}"
                color = "white" if abs(r_val) > 0.55 else "black"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pearson r", fontsize=11, fontweight="bold")
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    # Significance legend
    fig.text(
        0.01,
        0.01,
        "Significance:  *** p<0.001   ** p<0.01   * p<0.05   . p<0.1",
        fontsize=8,
        color="gray",
        va="bottom",
    )

    ax.set_title(
        f"Track {track_id}: Pearson Correlation Matrix\n"
        f"({phase_label}, n≥5 per pair)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    plt.tight_layout()
    plot_dir = os.path.join(output_dir, "correlation")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"track_{track_id}_correlation_matrix.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", "Correlation matrix", out_path)
    return r_matrix, p_matrix


def analyze_all_tracks(ds, full_track_df, buffer_km, output_dir):
    """Compute relationship for all tracks"""
    std_print(
        "INFO", "ca", " --- Analyzing all tracks for Soil Water Content patterns ---"
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

    # FIX: this is just relationship and not correlation
    # results_df = pd.DataFrame(all_results)

    # summary = {
    #     "n_tracks": len(results_df),
    #     "mean_relation": results_df["pearson_r"].mean(),
    #     "std_relation": results_df["pearson_r"].std(),
    #     "tracks_with_positive_r": (results_df["pearson_r"] > 0).sum(),
    #     "tracks_with_significant_r": (results_df["pearson_p"] < 0.05).sum(),
    #     "mean_tp": results_df["mean_tp"].mean(),
    #     "mean_swvl1": results_df["mean_swvl1"].mean(),
    # }

    # std_print("INFO", "ca", f"Soil Water Content Analysis across {summary['n_tracks']} Tracks")
    # std_print(None, None, "Mean relationship (TP vs SM)", f"{summary['mean_relation']:.3f} ± {summary['std_relation']:.3f}")
    # std_print(None, None, "Positive Correlations", f"{summary['tracks_with_positive_r']}/{summary['n_tracks']}")
    # std_print(None, None, "Significant Correlations (p < 0.05)", f"{summary['tracks_with_significant_r']}/{summary['n_tracks']}")
    # std_print(None, None, "Mean TP", f"{summary['mean_tp']:.2f} mm/hr")
    # std_print(None, None, "Mean SM", f"{summary['mean_swvl1']:.3f} m³/m³")
    # export_results_to_csv(results_df, "sm", output_dir, "sm_analysis")
    # return {"summary": summary, "detailed_results": results_df}


# Relation scatter plot with time gradient
def plot_tp_swvl1_relation(ds, track_df, track_id, buffer_km, output_dir):
    result = compute_single_track_relation(ds, track_df, track_id, buffer_km)
    # std_print("INFO", "ca", f" --- TP-SWVL1 Relation scatter plot ---")
    if result is None:
        std_print("WARN", "ca", "Insufficient data for relation plot")
        return

    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, width_ratios=[4, 0.1, 1], wspace=0.3)
    ax_main = plt.subplot(gs[0])
    ax_cbar = plt.subplot(gs[1])
    ax_stats = plt.subplot(gs[2])
    ax_stats.axis("off")

    times = result["times"]
    t_numeric = pd.to_datetime(times).astype(np.int64)
    tp_vals = result["tp_values"]
    sm_vals = result["swvl1_values"]

    # Filter for SWVL1 > 0.03 (land points only)
    land_mask = sm_vals > 0.03
    tp_land = tp_vals[land_mask]
    swvl1_land = sm_vals[land_mask]
    t_land = t_numeric[land_mask]

    if len(tp_land) and len(swvl1_land) < 3:
        std_print("WARN", "ca", "Insufficient data for relation plot", f"only {len(tp_land)} tp and {len(swvl1_land)} < 3 land points")
        sc = ax_main.scatter(
            tp_vals,
            sm_vals,
            c=t_numeric,
            cmap="viridis",
            alpha=0.6,
            s=100,
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
        )
    else:
        sc = ax_main.scatter(
            tp_land,
            swvl1_land,
            c=t_land,
            cmap="viridis",
            alpha=0.8,
            s=100,
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
            label=f"Land (SWVL1 > 0.03, n={len(tp_land)})"
        )
        # Overlay ocean points (if any) with low opacity
        ocean_mask = ~land_mask
        if np.any(ocean_mask):
            ax_main.scatter(
                tp_vals[ocean_mask],
                sm_vals[ocean_mask],
                c='gray',
                alpha=0.3,
                s=50,
                edgecolors='none',
                zorder=2,
                label=f"Ocean (n={np.sum(ocean_mask)})"
            )
        # Legend
        # ax_main.legend(loc="bottom right", **legend_style())

    cbar = plt.colorbar(sc, cax=ax_cbar)
    cbar.set_label("Track Time", fontsize=11, fontweight="bold")
    format_colorbar_time(cbar, t_numeric if len(tp_land) >= 3 else t_numeric)

    # Recalculate statistics using only land points (SWVL1 > 0.03)
    if len(tp_land) >= 3:
        pearson_r, pearson_p = pearsonr(tp_land, swvl1_land)
        slope, intercept, r_value, p_value, std_err = linregress(tp_land, swvl1_land)
        r_squared = r_value ** 2
    else:
        pearson_r, pearson_p, slope, intercept, r_squared = np.nan, np.nan, np.nan, np.nan, np.nan

    # Statistics box
    stats_text = (
        f"Pearson r:  {pearson_r:.4f}\n"
        f"P-value:    {pearson_p:.4e}\n"
        f"R²:         {r_squared:.4f}\n"
        f"Slope:      {slope:.4f}\n"
        f"Intercept:  {intercept:.4f}\n"
        f"N (land):   {len(tp_land)}\n"
        f"N (total):  {len(tp_vals)}\n"
        f"Mean TP:    {np.mean(tp_land):.2f} mm/hr\n"
        f"Mean SM:    {np.mean(swvl1_land):.2f} m³/m³"
    )
    bbox_style = common_bbox()
    ax_stats.text(
        0.1,
        0.95,
        stats_text,
        transform=ax_stats.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        bbox=bbox_style,
    )

    ax_main.set_xlabel("Total Precipitation (mm/hr)", fontsize=11, fontweight="bold")
    ax_main.set_ylabel("Volumetric Soil Water Layer-1 (0-7cm) (m³/m³)", fontsize=11, fontweight="bold")
    ax_main.set_title(
        f"Track {track_id}: TP-SWVL1 Relation (Land Only: SWVL1 > 0.03)",
        fontsize=13,
        fontweight="bold"
    )
    ax_main.grid(True, alpha=0.5, linestyle="--")

    track_dir = os.path.join(output_dir, "relation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_TP-SWVL1_relation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", "Relation scatter plot (land only) : TP-SWVL1", plot_path)


def plot_swvl1_hf_relation(swvl1, latent_hf, sensible_hf, valid_times, landfall_index, track_id, output_dir, originates_over_land=True):
    track_id = int(track_id)
    # std_print("INFO", "ca", f" --- SWVL1-HF Relation scatter plot ---")
    total_hf = latent_hf + sensible_hf
    valid_mask = ~(
        np.isnan(swvl1)
        | np.isnan(latent_hf)
        | np.isnan(sensible_hf)
        | np.isnan(total_hf)
    )
    if np.sum(valid_mask) < 5:
        std_print("WARN", "ca", "Insufficient data for SWVL1-HF relation", f"Mask points {valid_mask} < 5")
        return False

    t_numeric = pd.to_datetime(valid_times).astype(np.int64)
    t_plot = t_numeric[valid_mask]

    # Figure
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 4, width_ratios=[4, 4, 4, 0.3], wspace=0.3)
    axes = [plt.subplot(gs[i]) for i in range(3)]
    ax_cbar = plt.subplot(gs[3])
    y_vars = [latent_hf, sensible_hf, total_hf]
    y_labels = [
        "Latent Heat Flux (W/m²)",
        "Sensible Heat Flux (W/m²)",
        "Total Heat Flux (W/m²)"
    ]
    titles = [
        "SWVL1 ↔ Latent Heat Flux",
        "SWVL1 ↔ Sensible Heat Flux",
        "SWVL1 ↔ Total Heat Flux"
    ]
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
        ax.set_title(
            titles[i],
            fontsize=13,
            fontweight="bold"
        )
        ax.grid(True, alpha=0.5, linestyle="--")
        # try:
        #    bbox_style = common_bbox()
        #     r, p = pearsonr(swvl1[valid_mask], y_vars[i][valid_mask])
        #     ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.4f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=common_bbox)
        # except:
        #     pass

    cbar = plt.colorbar(sc, cax=ax_cbar)
    cbar.set_label("Track Time", fontsize=11, fontweight="bold")
    format_colorbar_time(cbar, t_numeric)

    status_text = "Land-Originated" if originates_over_land else "After Landfall"
    fig.suptitle(
        f"Track {track_id}: SWVL1 ↔ Surface Heat Flux Relation ({status_text})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    track_dir = os.path.join(output_dir, "relation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_SWVL1-HF_relation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", "Relation scatter plot : SWVL1-HF", plot_path)
    return True


def plot_tp_hf_relation(tp, latent_hf, sensible_hf, valid_times, landfall_index, track_id, output_dir, originates_over_land=True):
    track_id = int(track_id)
    # std_print("INFO", "ca", " --- TP-HF Relation scatter plot ---")
    total_hf = latent_hf + sensible_hf
    valid_mask = ~(
        np.isnan(tp) | np.isnan(latent_hf) | np.isnan(sensible_hf) | np.isnan(total_hf)
    )
    if np.sum(valid_mask) < 5:
        std_print("WARN", "ca", "Insufficient data for TP-HF relation", f"Mask points {valid_mask} < 5")
        return False

    t_numeric = pd.to_datetime(valid_times).astype(np.int64)
    t_plot = t_numeric[valid_mask]

    # Figure
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 4, width_ratios=[4, 4, 4, 0.3], wspace=0.3)
    axes = [plt.subplot(gs[i]) for i in range(3)]
    ax_cbar = plt.subplot(gs[3])
    y_vars = [latent_hf, sensible_hf, total_hf]
    y_labels = [
        "Latent Heat Flux (W/m²)",
        "Sensible Heat Flux (W/m²)",
        "Total Heat Flux (W/m²)"
    ]
    titles = [
        "TP ↔ Latent Heat Flux",
        "TP ↔ Sensible Heat Flux",
        "TP ↔ Total Heat Flux"
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
            zorder=3
        )
        ax.set_xlabel("Total Precipitation (mm/hr)", fontsize=11, fontweight="bold")
        ax.set_ylabel(y_labels[i], fontsize=11, fontweight="bold")
        ax.set_title(
            titles[i],
            fontsize=13,
            fontweight="bold"
        )
        ax.grid(True, alpha=0.5, linestyle="--")
        # try:
        #     bbox_style = common_bbox()
        #     r, p = pearsonr(tp[valid_mask], y_vars[i][valid_mask])
        #     ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.4f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=bbox_style)
        # except:
        #     pass

    cbar = plt.colorbar(sc, cax=ax_cbar)
    cbar.set_label("Track Time", fontsize=11, fontweight="bold")
    format_colorbar_time(cbar, t_numeric)
    status_text = "Land-Originated" if originates_over_land else "After Landfall"
    fig.suptitle(
        f"Track {track_id}: TP ↔ Surface Heat Flux Relation ({status_text})",
        fontsize=13,
        fontweight="bold",
        y=1.01
    )

    track_dir = os.path.join(output_dir, "relation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_TP-HF_relation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", "Relation scatter plot : TP-HF", plot_path)
    return True


# lag relation
def plot_tp_swvl1_lag_relation(tp, sm, max_lag_hours, track_id, output_dir):
    """
    lag relation between TP and SWVL1
     - Negative lag: TP leads SWVL1 (infiltration)
     - Positive lag: SWVL1 leads TP (antecedent moisture)
    """
    track_id = int(track_id)
    # std_print("INFO", "ca", f" --- Track {track_id} TP-SWVL1 Lag-Relation plot ---")
    lags, relations_lag = lag_relation(tp, sm, max_lag_hours=max_lag_hours)
    if len(lags) != len(relations_lag):
        min_len = min(len(lags), len(relations_lag))
        lags = lags[:min_len]
        relations_lag = relations_lag[:min_len]

    valid_indices = [i for i, val in enumerate(relations_lag) if not np.isnan(val)]
    if len(valid_indices) == 0:
        std_print("WARN", "ca", f"Track {track_id}", "No valid relation values for lag plot")
        return

    lags = [lags[i] for i in valid_indices]
    relations_lag = [relations_lag[i] for i in valid_indices]

    max_r_idx = np.argmax(np.abs(relations_lag))
    max_r_lag = lags[max_r_idx]
    if max_r_lag < 0:
        lag_interpretation = (f"TP leads SWVL1 by {abs(max_r_lag)}h\n(Infiltration lag)")
    elif max_r_lag > 0:
        lag_interpretation = f"SWVL1 leads TP by {max_r_lag}h\n(Antecedent moisture)"
    else:
        lag_interpretation = "Simultaneous\n(Immediate feedback)"

    fig, ax = plt.subplots(figsize=(12, 6))
    bbox_style = common_bbox()

    ax.plot(lags, relations_lag, "o-", lw=2.5, markersize=4, color="blue", zorder=3)
    ax.fill_between(lags, relations_lag, alpha=0.2, color="steelblue")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(
        0.98,
        0.94,
        lag_interpretation,
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="right",
        bbox=bbox_style,
    )
    ax.set_xlabel("Lag (hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Relation (r)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: TP-SWVL1 Lagged Relation",
        fontsize=13,
        fontweight="bold"
    )

    ax.grid(True, alpha=0.5, linestyle="--")
    ax.axhline(y=0.2, color="red", linestyle=":", alpha=0.5)
    ax.axhline(y=-0.2, color="red", linestyle=":", alpha=0.5)

    track_dir = os.path.join(output_dir, "lag_relation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_TP-SWVL1_lag_relation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", "Lag Relation plot : TP-SWVL1", plot_path)


def plot_swvl1_hf_lag_relation(sm, hf, max_lag_hours, track_id, hf_name, output_dir):
    """
    lag relation between SWVL1 and Heat Flux
     - Negative lag: SM leads HF (Brown Ocean Effect)
     - Positive lag: HF leads SM (Surface Drying)
    """
    track_id = int(track_id)
    # std_print("INFO", "ca", f" --- Track {track_id} SWVL1-{hf_name} Lag-relation plot ---")
    lags, relations_lag = lag_relation(sm, hf, max_lag_hours)
    valid_indices = [i for i, val in enumerate(relations_lag) if not np.isnan(val)]
    if not valid_indices:
        std_print("WARN", "ca", f"Track {track_id}", f"No valid data for SWVL1-{hf_name} lag plot")
        return

    lags = [lags[i] for i in valid_indices]
    relations_lag = [relations_lag[i] for i in valid_indices]

    # Color scheme based on flux type
    if "LHF" in hf_name.upper() or "LATENT" in hf_name.upper():
        line_color, fill_color = "forestgreen", "lightgreen"
    elif "SHF" in hf_name.upper() or "SENSIBLE" in hf_name.upper():
        line_color, fill_color = "darkorange", "navajowhite"
    elif "THF" in hf_name.upper() or "TOTAL" in hf_name.upper():
        line_color, fill_color = "rebeccapurple", "thistle"
    else:
        line_color, fill_color = "blue", "lightblue"

    max_idx = np.argmax(np.abs(relations_lag))
    max_lag_hours = lags[max_idx]
    if max_lag_hours < 0:
        interp = f"SWVL1 leads {hf_name} by {abs(max_lag_hours)}h\n(Energy Feedback)"
    elif max_lag_hours > 0:
        interp = f"{hf_name} leads SWVL1 by {max_lag_hours}h\n(Surface Drying)"
    else:
        interp = "Simultaneous\n(Coupled Phase)"

    fig, ax = plt.subplots(figsize=(12, 6))
    bbox_style = common_bbox()

    ax.plot(lags, relations_lag, "o-", lw=2.5, markersize=4, color=line_color, zorder=3)
    ax.fill_between(lags, relations_lag, alpha=0.2, color=fill_color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(
        0.98,
        0.94,
        interp,
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="right",
        bbox=bbox_style,
    )
    ax.set_xlabel("Lag (hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("relation (r)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: SWVL1-{hf_name} Lagged relation",
        fontsize=13,
        fontweight="bold"
    )
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.axhline(y=0.2, color="red", linestyle=":", alpha=0.5)
    ax.axhline(y=-0.2, color="red", linestyle=":", alpha=0.5)

    track_dir = os.path.join(output_dir, "lag_relation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_SWVL1-{hf_name}_lag_relation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", f"Lag plot : SWVL1-{hf_name}", plot_path)
    return True


def plot_tp_hf_lag_relation(flux, tp, max_lag_hours, track_id, flux_name, output_dir):
    """
    Plot lag relation between Heat Flux and TP
     - Negative lag: Flux leads TP (energy influences precipitation)
     - Positive lag: TP leads Flux (rainfall affects energy flux)
    """
    track_id = int(track_id)
    # std_print("INFO", "ca", f" --- Track {track_id} {flux_name}-TP Lag-relation plot ---")
    lags, relations_lag = lag_relation(flux, tp, max_lag_hours)
    valid_indices = [i for i, val in enumerate(relations_lag) if not np.isnan(val)]
    if not valid_indices:
        std_print("WARN", "ca", f"Track {track_id}", f"No valid data for {flux_name}-TP lag plot")
        return

    lags = [lags[i] for i in valid_indices]
    relations_lag = [relations_lag[i] for i in valid_indices]

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

    if relations_lag:
        max_idx = np.argmax(np.abs(relations_lag))
        max_lag_hours = lags[max_idx]
        if max_lag_hours < 0:
            interp = f"{flux_label} leads TP by {abs(max_lag_hours)}h\n(Energy influences precipitation)"
        elif max_lag_hours > 0:
            interp = f"TP leads {flux_label} by {max_lag_hours}h\n(Rainfall affects energy flux)"
        else:
            interp = "Simultaneous\n(Strong coupling)"
    else:
        interp = "No significant relation"

    fig, ax = plt.subplots(figsize=(12, 6))
    bbox_style = common_bbox()

    ax.plot(lags, relations_lag, "o-", lw=2.5, markersize=4, color=line_color, zorder=3)
    ax.fill_between(lags, relations_lag, alpha=0.2, color=fill_color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(
        0.98,
        0.94,
        interp,
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="right",
        bbox=bbox_style
    )
    ax.set_xlabel("Lag (hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("relation (r)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: Lagged {flux_label} vs Total Precipitation",
        fontsize=13,
        fontweight="bold"
    )
    ax.grid(True, alpha=0.5, linestyle="--")
    ax.axhline(y=0.2, color="red", linestyle=":", alpha=0.5)
    ax.axhline(y=-0.2, color="red", linestyle=":", alpha=0.5)

    track_dir = os.path.join(output_dir, "lag_relation")
    os.makedirs(track_dir, exist_ok=True)
    plot_path = os.path.join(track_dir, f"track_{track_id}_TP-{flux_name}_lag_relation.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ca", f"Lag plot : TP-{flux_name}", plot_path)
    return True


# def plot_lag_autorelation(data, var_name, track_id, output_dir="output"):
#     """Plot autorelation (lagged self-relation) for a single variable"""
#     values = np.array(data)
#     values = values[~np.isnan(values)]
#     if len(values) < 10:
#         std_print("WARN", "ca", f"Insufficient data for {var_name} autorelation", f"Points {values} < 10")
#         return None
#
#     max_lag_hours = min(48, len(values) // 2)
#     lags, relations, pvalues = compute_lagged_relation(values, max_lag_hours=max_lag_hours)
#     valid = ~np.isnan(relations)
#
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     ax.plot(
#         lags[valid],
#         relations[valid],
#         "o-",
#         linewidth=2.5,
#         markersize=4,
#         color="blue",
#         label="Autorelation",
#         zorder=3
#     )
#     ax.fill_between(lags, relations, alpha=0.2, color="steelblue")
#     ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
#     ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
#     ax.set_xlabel("Lag (hours)", fontsize=11, fontweight="bold")
#     ax.set_ylabel("Relation Coefficient (r)", fontsize=11, fontweight="bold")
#     ax.grid(True, alpha=0.5, linestyle="--")
#     ax.set_ylim(-1.0, 1.05)
#
#     # Legend
#     legend_handles, labels = ax.get_legend_handles_labels()
#     if legend_handles:
#         ax.legend(handles=legend_handles, loc="best", **legend_style())
#
#     titles = {
#         "avg_slhtf": f"Track {track_id}: Latent Heat Flux Temporal Autorelation",
#         "avg_ishf": f"Track {track_id}: Sensible Heat Flux Temporal Autorelation",
#         "total_hf": f"Track {track_id}: Total Heat Flux Temporal Autorelation"
#     }
#     title = titles.get(var_name, f"Track {track_id}: {var_name} Temporal Autorelation")
#     ax.set_title(title, fontsize=13, fontweight="bold")
#
#     plot_dir = os.path.join(output_dir, "autorelation")
#     os.makedirs(plot_dir, exist_ok=True)
#     clean_name = var_name.replace(" ", "_").replace("(", "").replace(")", "")
#     outpath = os.path.join(plot_dir, f"track_{track_id}_autor_{clean_name}.png")
#     plt.savefig(outpath, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     std_print("SAVED", "ca", f"Autorelation plot: {var_name}", outpath)
#     return lags, relations, pvalues


# Main relation analysis function
def analyze_single_track(ds, track_df, track_id, buffer_km, max_lag_hours=50, output_dir="output"):
    """Main function to analyze a single track with all relation analyses"""
    std_print("INFO", "ca", " --- Processing Relation Analysis ---")
    # Filter to single track
    if len(track_df["track_id"].unique()) == 1:
        current_track = track_df.sort_values("time").copy()
    else:
        current_track = (track_df[track_df["track_id"] == track_id].sort_values("time").copy())
    if current_track.empty:
        std_print("ERROR", "ca", "track_df is None")
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
        time_step = pd.to_datetime(row["time"])
        lat, lon = row["latitude"], row["longitude"]

        subset = extract_spatial_subset(ds, lat, lon, time_step, time_coord, buffer_km)
        if subset is None:
            std_print("ERROR", "ca", "Extracted spatical subset is None")
            continue
        tp = float(subset["tp"].mean().values)
        swvl1 = float(subset["swvl1"].mean().values)
        latent_hf = float(subset["avg_slhtf"].mean().values)
        sensible_hf = float(subset["avg_ishf"].mean().values)
        total_hf = (
            latent_hf + sensible_hf
            if np.isfinite(latent_hf) and np.isfinite(sensible_hf)
            else np.nan
        )
        if np.isfinite(tp) and np.isfinite(swvl1):
            valid_times.append(time_step)
            tp_series.append(tp)
            swvl1_series.append(swvl1)
            latent_hf_series.append(latent_hf)
            sensible_hf_series.append(sensible_hf)
            total_hf_series.append(total_hf)
    if len(valid_times) < 5:
        std_print("WARN", "ca", "Insufficient data", f"Valid points {len(valid_times)} < 5")
        return None

    # Unit conversions
    tp_series = np.array(tp_series) * 1000
    swvl1_series = np.array(swvl1_series)
    latent_hf_series = np.array(latent_hf_series)
    sensible_hf_series = np.array(sensible_hf_series)
    total_hf_series = np.array(total_hf_series)

    # Set SWVL1 = 0 before landfall (physically impossible over ocean)
    swvl1_series_land = swvl1_series.copy()
    if landfall_index > 0:
        swvl1_series_land[:landfall_index] = 0
        std_print("INFO", "ca", "SWVL1 set to 0 for ocean phase", f"indices 0-{landfall_index}")
    else:
        std_print("WARN", "ca", "No landfall detected", "No valid SWVL1 along track")

    # Plots
    os.makedirs(output_dir, exist_ok=True)
    std_print("INFO", "ca", " --- Generating relation plots ---")
    # 1. TP-SWVL1 Feedback
    try:
        plot_tp_swvl1_relation(ds, track_df, track_id, buffer_km=100, output_dir=output_dir)
        plot_tp_swvl1_lag_relation(tp_series, swvl1_series_land, max_lag_hours, track_id, output_dir)
    except Exception as e:
        std_print("ERROR", "ca", "TP-SWVL1 analysis failed", str(e))
        traceback.print_exc()

    # 2. SWVL1-Heat Flux Feedback
    try:
        if not np.all(np.isnan(latent_hf_series)) or not np.all(
            np.isnan(sensible_hf_series)
        ):
            plot_swvl1_hf_relation(swvl1_series_land, latent_hf_series, sensible_hf_series, valid_times, landfall_index, track_id, output_dir, originates_over_land)
            plot_swvl1_hf_lag_relation(swvl1_series_land, latent_hf_series, max_lag_hours, track_id, "LHF", output_dir)
            plot_swvl1_hf_lag_relation(swvl1_series_land, sensible_hf_series, max_lag_hours, track_id, "SHF", output_dir)
            plot_swvl1_hf_lag_relation(swvl1_series_land, total_hf_series, max_lag_hours, track_id, "THF", output_dir)
    except Exception as e:
        std_print("ERROR", "ca", "SWVL1-Heat Flux lag analysis failed", str(e))

    # 3. TP-Heat Flux Feedback
    try:
        if not np.all(np.isnan(latent_hf_series)) or not np.all(np.isnan(sensible_hf_series)):
            plot_tp_hf_relation(tp_series, latent_hf_series, sensible_hf_series, valid_times, landfall_index, track_id, output_dir, originates_over_land)
            plot_tp_hf_lag_relation(tp_series, latent_hf_series, max_lag_hours, track_id, "LHF", output_dir)
            plot_tp_hf_lag_relation(tp_series, sensible_hf_series, max_lag_hours, track_id, "SHF", output_dir)
            plot_tp_hf_lag_relation(tp_series, total_hf_series, max_lag_hours, track_id, "THF", output_dir)
    except Exception as e:
        std_print("ERROR", "ca", "TP-Heat Flux lag analysis failed", str(e))

    # 4. Correlation Matrix (actual relation analysis)
    try:
        plot_correlation_matrix(tp_series, swvl1_series_land, latent_hf_series, sensible_hf_series, valid_times, landfall_index, track_id, output_dir)
    except Exception as e:
        std_print("ERROR", "ca", "relation matrix failed", str(e))
        traceback.print_exc()

    # 5. Rolling windowed relation (temporal evolution of coupling)
    try:
        plot_rolling_correlation(tp_series, swvl1_series_land, latent_hf_series, sensible_hf_series, valid_times, landfall_index, track_id, output_dir, window_hours=24)
    except Exception as e:
        std_print("ERROR", "ca", "Rolling relation failed", str(e))
        traceback.print_exc()

    # # 6. Autorelation Analysis
    # std_print("INFO", "ca", " --- Generating Autorelation plots ---")
    # try:
    #     plot_lag_autorelation(tp_series, "tp", track_id, output_dir)
    #     plot_lag_autorelation(swvl1_series_land, "swvl1", track_id, output_dir)
    #     plot_lag_autorelation(latent_hf_series, "latent_hf", track_id, output_dir)
    #     plot_lag_autorelation(sensible_hf_series, "sensible_hf", track_id, output_dir)
    #     plot_lag_autorelation(total_hf_series, "total_hf", track_id, output_dir)
    # except Exception as e:
    #     std_print("ERROR", "ca", "Autorelation analysis failed", str(e))
    #     traceback.print_exc()

    # Compute relation ONLY on land phase
    r, p_val = np.nan, np.nan
    r_pre, r_post = np.nan, np.nan
    if landfall_index > 0 and len(tp_series[landfall_index:]) >= 5:
        tp_land = tp_series[landfall_index:]
        swvl1_land = swvl1_series_land[landfall_index:]
        if np.std(tp_land) > 0 and np.std(swvl1_land) > 0:
            r, p_val = pearsonr(tp_land, swvl1_land)
            r_post = r
            std_print("INFO", "ca", "TP-SWVL1 relation (Land only)", f"r={r:.4f}, p={p_val:.4f}")
        r_pre = np.nan
    elif originates_over_land:
        if np.std(tp_series) > 0 and np.std(swvl1_series_land) > 0:
            r, p_val = pearsonr(tp_series, swvl1_series_land)
            r_post = r
            std_print("INFO", "ca", "TP-SWVL1 relation (Land-Originated)", f"r={r:.4f}, p={p_val:.4f}")
    else:
        std_print("WARN", "ca", "No landfall detected", "TP-SWVL1 relation NOT computed")

    # Store results
    relation_results = {
        "track_id": track_id,
        "relation": r if not np.isnan(r) else np.nan,
        "p_value": p_val if not np.isnan(p_val) else np.nan,
        "r_pre": r_pre,
        "r_post": r_post,
        "mean_tp": np.mean(tp_series),
        "mean_swvl1": np.mean(swvl1_series_land),
        "n_samples": len(valid_times),
        "landfall_time": valid_times[landfall_index] if landfall_index != -1 else None,
        "landfall_index": landfall_index,
        "originates_over_land": originates_over_land,
        "tp_series": tp_series.tolist() if hasattr(tp_series, "tolist") else list(tp_series),
        "swvl1_series": swvl1_series.tolist() if hasattr(swvl1_series, "tolist") else list(swvl1_series),
        "valid_times": list(valid_times),
    }
    # std_print("DEBUG", "ca", "TP-SWVL1 relation", f"r={relation_results['relation']:.4f}, p={relation_results['p_value']:.4f}, n={len(valid_times)}")
    return relation_results


def compute_single_track_relation(ds, track_df, track_id, buffer_km):
    """Compute Pearson and Spearman Relation between TP and SWVL1 (land only)"""
    track_id = int(track_id)
    std_print("INFO", "ca", " --- Computing Relation (land only) ---")
    if "is_land" not in track_df.columns:
        track_df["is_land"] = track_df.apply(
            lambda row: is_point_land(row["latitude"], row["longitude"]), axis=1
        )

    tp_times, tp_vals = extract_regional_data(ds, track_df, track_id, "tp", buffer_km)
    sm_times, sm_vals = extract_regional_data(ds, track_df, track_id, "swvl1", buffer_km)
    common_times = np.intersect1d(tp_times, sm_times)
    if len(common_times) < 3:
        return None

    tp_idx = np.isin(tp_times, common_times)
    sm_idx = np.isin(sm_times, common_times)

    tp_aligned = tp_vals[tp_idx] * 1000
    sm_aligned = sm_vals[sm_idx]

    # Filter for land points only (SWVL1 > 0.03)
    land_mask = sm_aligned > 0.03
    tp_land = tp_aligned[land_mask]
    swvl1_land = sm_aligned[land_mask]

    if len(tp_land) < 3:
        std_print("WARN", "ca", "Insufficient land points (SWVL1 > 0.03)", len(tp_land))
        return None

    if np.std(tp_land) == 0 or np.std(swvl1_land) == 0:
        std_print("WARN", "ca", "Zero variance in one or both variables (land only)")
        return None

    pearson_r, pearson_p = pearsonr(tp_land, swvl1_land)
    spearman_r, spearman_p = spearmanr(tp_land, swvl1_land)
    slope, intercept, r_value, p_value, std_err = linregress(tp_land, swvl1_land)
    r_squared_val = float(r_value) ** 2

    return {
        "track_id": track_id,
        "n_samples": len(tp_land),
        "n_samples_total": len(tp_aligned),
        "mean_tp": float(np.mean(tp_land)),
        "mean_swvl1": float(np.mean(swvl1_land)),
        "std_tp_mm": float(np.std(tp_land)),
        "std_swvl1": float(np.std(swvl1_land)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
        "r_squared": r_squared_val,
        "p_value": p_value,
        "std_err": std_err,
        "times": common_times[land_mask],  # Only land times
        "tp_values": tp_land,  # Only over land
        "swvl1_values": swvl1_land,  # Only over land
    }

# Helper functions
# Significance stars
def sig_star(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "·"
    return ""


# Rolling Pearson r
def rolling_r(x, y, n, window):
    """Returns array of r values, NaN where window has < 3 finite pairs."""
    r_vals = np.full(n, np.nan)
    for k in range(window - 1, n):
        xi = x[k - window + 1 : k + 1]
        yi = y[k - window + 1 : k + 1]
        mask = ~(np.isnan(xi) | np.isnan(yi))
        if mask.sum() < 3:
            continue
        if np.std(xi[mask]) == 0 or np.std(yi[mask]) == 0:
            continue
        r, _ = pearsonr(xi[mask], yi[mask])
        r_vals[k] = r
    return r_vals


def lag_relation(x, y, max_lag_hours=48):
    """Computes relation between x and y with temporal offsets"""
    lags = range(-max_lag_hours, max_lag_hours + 1)
    relations = []
    for lag in lags:
        if lag < 0:
            x_slice, y_slice = x[:lag], y[-lag:]
        elif lag > 0:
            x_slice, y_slice = x[lag:], y[:-lag]
        else:
            x_slice, y_slice = x, y
        if len(x_slice) < 5 or np.std(x_slice) == 0 or np.std(y_slice) == 0:
            relations.append(np.nan)
        else:
            r, _ = pearsonr(x_slice, y_slice)
            relations.append(r)
    return list(lags), relations


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
        std_print("ERROR", "ca", "Computing ACF", str(e))
        return None, None


def compute_lagged_relation(data, max_lag_hours=96):
    """Compute relation of a variable with itself at different time lags"""
    data_clean = np.array(data)
    data_clean = data_clean[~np.isnan(data_clean)]
    if len(data_clean) < max_lag_hours + 1:
        max_lag_hours = len(data_clean) // 2
    lags = []
    relations = []
    pvalues = []
    for lag in range(0, max_lag_hours + 1):
        if lag == 0:
            lags.append(0)
            relations.append(1.0)
            pvalues.append(0.0)
        else:
            x = data_clean[:-lag]
            y = data_clean[lag:]
            if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
                lags.append(lag)
                relations.append(np.nan)
                pvalues.append(np.nan)
            else:
                try:
                    r, p = pearsonr(x, y)
                    lags.append(lag)
                    relations.append(r)
                    pvalues.append(p)
                except:
                    lags.append(lag)
                    relations.append(np.nan)
                    pvalues.append(np.nan)
    return np.array(lags), np.array(relations), np.array(pvalues)


def format_colorbar_time(cbar, t_numeric):
    """Format colorbar ticks with datetime strings - DD/MM/YYYY HH:MM format"""
    t_min, t_max = t_numeric.min(), t_numeric.max()
    cbar.mappable.set_clim(t_min, t_max)

    tick_locs = np.linspace(t_min, t_max, 10) # 10 ticks
    tick_dates = pd.to_datetime(tick_locs)

    cbar.ax.set_yticks(tick_locs)
    cbar.ax.set_yticklabels([d.strftime("%d/%m/%Y %H:%M") for d in tick_dates])
    cbar.ax.set_ylim(t_min, t_max)
    return cbar
