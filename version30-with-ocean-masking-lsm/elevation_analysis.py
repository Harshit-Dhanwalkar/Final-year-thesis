"""
Elevation Analysis for MD Tracks using SRTM HGT Data

  1. Orographic speed reduction      - terrain slows MDs (barrier effect)
  2. Orographic precipitation ratio  - windward vs flat TP amplification
  3. Elevation vs penetration scatter - terrain controls inland reach
  4. Landfall elevation classification - low-coast vs terrain-facing MDs
  5. Soil texture map (Clapp & Hornberger 1978) - SM memory interpretation

TSI combines three terrain-dependent factors:
  TSI = elevation_effect x slope_effect x roughness_effect
  - elevation_effect: normalized elevation contrast
  - slope_effect: mean slope magnitude
  - roughness_effect: terrain variability (std of elevation)
  Each component normalised to [0, 1].

Dependencies: sudo apt install gdal-bin

Reference: Kirshbaum (2018) JGR-Atmos.
"""

# TODO:
# 1. Mean elevation along track vs. speed anomaly (speed relative to pre-landfall baseline) - not implemented yet.

import os
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.feature as cfeature
from scipy.ndimage import sobel
from scipy.stats import spearmanr
import traceback

from utils import (
    std_print,
    common_bbox,
    legend_style,
    haversine_distance,
    get_landfall_index,
    pearsonr_corrected,
    is_land,
)

# Soil texture constants (Clapp & Hornberger 1978, Table 2)
# Approximate centre-of-mass coordinates for the main pedological zones over the Indian subcontinent.
# \beta = Clapp-Hornberger exponent; higher \beta -> faster drainage -> shorter SM memory
# Hydraulic conductivity k_sat (m/s) for rough scale.
SOIL_TEXTURE_ZONES = {
    "Gangetic Plains (Clay Loam)": {
        "lat_range": (25, 28),
        "lon_range": (77, 88),
        "beta": 7.12,
        "k_sat": 3.8e-6,
        "color": "#8B4513",
    },
    "Deccan Plateau (Sandy Clay)": {
        "lat_range": (15, 22),
        "lon_range": (74, 80),
        "beta": 5.39,
        "k_sat": 6.3e-6,
        "color": "#D2691E",
    },
    "Indo-Gangetic (Silty Clay)": {
        "lat_range": (22, 25),
        "lon_range": (75, 85),
        "beta": 8.52,
        "k_sat": 2.0e-6,
        "color": "#A0522D",
    },
    "Coastal Alluvium (Loamy Sand)": {
        "lat_range": (10, 18),
        "lon_range": (79, 86),
        "beta": 4.38,
        "k_sat": 1.6e-5,
        "color": "#DEB887",
    },
    "Eastern Ghats (Sandy Loam)": {
        "lat_range": (11, 20),
        "lon_range": (77, 82),
        "beta": 4.90,
        "k_sat": 7.2e-6,
        "color": "#F4A460",
    },
}
# SM memory timescale (days) from Seneviratne et al. (2010):
#   \tau \aprrox 1 / (k_sat x \beta)   (qualitative order-of-magnitude)


hgt_dir = ""
tile_index = {}   # "N24E087.HGT" → full path
tiles_cache = {}   # tile_path → tile_data dict

# FIX: workaround using new_hgt_dir
def init_terrain(new_hgt_dir):
    """Initialise module-level HGT state"""
    global hgt_dir, tiles_cache, tile_index
    hgt_dir = new_hgt_dir
    tiles_cache = {}
    tile_index = {}
    if os.path.exists(hgt_dir):
        build_tile_index()
        std_print("INFO", "ea", "Terrain initialised", f"HGT dir: {hgt_dir} ({len(tile_index)} tiles indexed)")
    else:
        std_print("WARN", "ea", "HGT directory not found", hgt_dir)


# FIX: empty/blank left panel
def plot_soil_texture_interpretation(results_df, output_dir):
    """
    Bar chart of inferred soil texture zones along track corridors and their implied SM memory timescales.
    """
    os.makedirs(output_dir, exist_ok=True)
    if "lat_landfall" not in results_df.columns:
        std_print("WARN", "ea", "plot_soil_texture_interpretation", "no lat_landfall column")
        return

    zone_counts = {z: 0 for z in SOIL_TEXTURE_ZONES}
    for _, row in results_df.iterrows():
        lat = row.get("lat_landfall", np.nan)
        lon = row.get("lon_landfall", np.nan)
        if np.isnan(lat) or np.isnan(lon):
            continue
        for zone, info in SOIL_TEXTURE_ZONES.items():
            if (
                info["lat_range"][0] <= lat <= info["lat_range"][1]
                and info["lon_range"][0] <= lon <= info["lon_range"][1]
            ):
                zone_counts[zone] += 1
                break

    zones = list(zone_counts.keys())
    counts = [zone_counts[z] for z in zones]
    colors = [SOIL_TEXTURE_ZONES[z]["color"] for z in zones]
    betas = [SOIL_TEXTURE_ZONES[z]["beta"] for z in zones]
    k_sats = [SOIL_TEXTURE_ZONES[z]["k_sat"] for z in zones]
    taus = [soil_memory_timescale(b, k) for b, k in zip(betas, k_sats)]

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: track count per zone
    ax = axes[0]
    bars = ax.barh(zones, counts, color=colors, edgecolor="black", linewidth=1)
    ax.set_xlabel("Track count", fontsize=11, fontweight="bold")
    ax.set_title("Landfall Zone - Soil Texture Distribution", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    for i, v in enumerate(counts):
        ax.text(v + 0.05, i, str(v), va="center", fontsize=10)

    # Panel 2: SM memory timescale per zone
    ax = axes[1]
    bars = ax.barh(zones, taus, color=colors, edgecolor="black", linewidth=1, alpha=0.8)
    ax.set_xlabel("SM memory timescale (days)", fontsize=11, fontweight="bold")
    ax.set_title("Inferred SM Memory by Soil Texture", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    for i, (v, b) in enumerate(zip(taus, betas)):
        ax.text(v + 0.05, i, f"β={b:.2f}", va="center", fontsize=9)
    plt.tight_layout()

    # Save
    soil_texture_memory_path = os.path.join(output_dir, "soil_texture_memory.png")
    plt.savefig(soil_texture_memory_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Soil texture / memory plot", soil_texture_memory_path)


def plot_elevation_metrics(tsi_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    bbox_style = common_bbox()

    # TODO: do i really need histogram?
    # 1. TSI histogram
    ax = axes[0, 0]
    valid_tsi = tsi_df[tsi_df["TSI"].notna()]["TSI"]
    if len(valid_tsi) > 0:
        ax.hist(valid_tsi, bins=15, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(
            valid_tsi.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {valid_tsi.mean():.3f}",
        )
        ax.set_xlabel(
            "Topographic Sensitivity Index (TSI)", fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("TSI Distribution", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        # Legend
        legend_handles, labels = ax.get_legend_handles_labels()
        if legend_handles:
            ax.legend(handles=legend_handles, loc="best", **legend_style())

    # 2. TSI vs Duration
    ax = axes[0, 1]
    if "duration_h" in tsi_df.columns:
        valid_subset = tsi_df.dropna(subset=["TSI", "duration_h"])
        if len(valid_subset) > 0:
            c_vals = valid_subset["elevation_range"] if "elevation_range" in valid_subset.columns else "steelblue"
            sc1 = ax.scatter(
                valid_subset["TSI"],
                valid_subset["duration_h"],
                c=c_vals,
                cmap="viridis",
                s=90,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.8,
            )
            ax.set_xlabel("TSI", fontsize=11, fontweight="bold")
            ax.set_ylabel("Duration (hours)", fontsize=11, fontweight="bold")
            ax.set_title("TSI vs Track Duration", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)

            if "elevation_range" in valid_subset.columns:
                cbar = plt.colorbar(sc1, ax=ax)
                cbar.set_label("Elevation Range (m)", fontsize=10)

            # Correlation
            # corr = valid_subset["TSI"].corr(valid_subset["duration_h"])
            r, _, p_corr, n_eff = pearsonr_corrected(valid_subset["TSI"].values, valid_subset["duration_h"].values)
            sig = "**" if (not np.isnan(p_corr) and p_corr < 0.05) else "ns"
            ax.text(
                0.05,
                0.95,
                f"r = {r:.3f}\np_corr = {p_corr:.4e} {sig}\nn_eff = {n_eff}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=bbox_style,
                family="monospace"
            )

    # 3. Component effects
    ax = axes[1, 0]
    components = []
    means = []
    for col, lbl in [
        ("elevation_effect", "Elevation"),
        ("slope_effect", "Slope"),
        ("roughness_effect", "Roughness")
    ]:
        if col in tsi_df.columns:
            components.append(lbl)
            means.append(tsi_df[col].mean())

    if components:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        bars = ax.bar(
            components,
            means,
            color=colors[: len(components)],
            edgecolor="black",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Mean Effect Value", fontsize=11, fontweight="bold")
        ax.set_title("TSI Component Effects", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=bbox_style,
            )

    # 4. Elevation range vs TSI
    ax = axes[1, 1]
    if "elevation_range" in tsi_df.columns:
        valid_subset = tsi_df.dropna(subset=["TSI", "elevation_range"])
        if len(valid_subset) > 0:
            c_vals = valid_subset["duration_h"] if "duration_h" in valid_subset.columns else "steelblue"
            sc2 = ax.scatter(
                valid_subset["elevation_range"],
                valid_subset["TSI"],
                c=c_vals,
                cmap="plasma",
                s=90,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.8,
            )
            if "duration_h" in valid_subset.columns:
                plt.colorbar(sc2, ax=ax).set_label("Duration (h)", fontsize=9)

            r, p_raw, p_corr, n_eff = pearsonr_corrected(valid_subset["elevation_range"].values, valid_subset["TSI"].values)
            ax.text(
                0.05,
                0.95,
                f"r = {r:.3f}\np_corr = {p_corr:.4e}",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                bbox=bbox_style,
                family="monospace"
            )
            ax.set_xlabel("Elevation Range (m)", fontsize=11, fontweight="bold")
            ax.set_ylabel("TSI", fontsize=11, fontweight="bold")
            ax.set_title("Elevation Range vs TSI", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    elevation_metrics_path = os.path.join(output_dir, "elevation_metrics_analysis.png")
    plt.savefig(elevation_metrics_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Elevation metric plots", elevation_metrics_path)


# Orographic speed reduction
def compute_orographic_speed_reduction(tracks_df_dict, output_dir):
    """
    Check whether Track translation speed drops as elevation rises.
 
    For each track that has both 'elevation' and 'speed_ms' columns, computes:
      - Pearson r between elevation gradient and speed change
      - Mean speed in low-elevation (<600 m)
    """
    os.makedirs(output_dir, exist_ok=True)
    all_elev = []
    all_speed = []
    all_dElev = []
    all_dSpeed = []
 
    for track_id, track_df in tracks_df_dict.items():
        if "elevation" not in track_df.columns or "speed_ms" not in track_df.columns:
            continue
        df = track_df.dropna(subset=["elevation", "speed_ms"]).copy()
        if len(df) < 5:
            continue
        # Elevation gradient: change in elev per track step (m per ~step)
        d_elev = df["elevation"].diff().fillna(0).values
        d_speed = df["speed_ms"].diff().fillna(0).values
        all_elev.extend(df["elevation"].values.tolist())
        all_speed.extend(df["speed_ms"].values.tolist())
        all_dElev.extend(d_elev.tolist())
        all_dSpeed.extend(d_speed.tolist())
 
    if len(all_elev) < 10:
        std_print("WARN", "ea", "Insufficient data for orographic speed reduction")
        return
 
    all_elev = np.array(all_elev)
    all_speed = np.array(all_speed)
    all_dElev = np.array(all_dElev)
    all_dSpeed = np.array(all_dSpeed)
 
    # Plot 1: elevation vs speed scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(all_elev, all_speed, s=20, alpha=0.4, c="steelblue", edgecolors="none")
    r, p_raw, p_corr, n_eff = pearsonr_corrected(all_elev, all_speed)
    z = np.polyfit(all_elev, all_speed, 1)
    x_line = np.linspace(all_elev.min(), all_elev.max(), 100)
    ax.plot(
      x_line,
      np.poly1d(z)(x_line),
      "r--",
      lw=1.5
    )
    sig = "**" if (not np.isnan(p_corr) and p_corr < 0.05) else "ns"
    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}\np_corr = {p_corr:.4e} {sig}\nn_eff = {n_eff}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=common_bbox(),
        family="monospace"
    )
    ax.set_xlabel("Elevation (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Translation Speed (m/s)", fontsize=11, fontweight="bold")
    ax.set_title("Orographic Speed Reduction\n(elevation vs translation speed)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
 
    # Plot 2: elevation change vs speed change
    ax = axes[1]
    # Filter to upslope points only (dElev > 0 = storm is ascending terrain)
    mask = all_dElev > 0
    if mask.sum() >= 5:
        ax.scatter(
            all_dElev[mask],
            all_dSpeed[mask],
            s=20,
            alpha=0.4,
            c="darkorange",
            edgecolors="none",
            label="Ascending terrain"
        )

        r2, p_raw2, p_corr2, n_eff2 = pearsonr_corrected(all_dElev[mask], all_dSpeed[mask])
        z2 = np.polyfit(all_dElev[mask], all_dSpeed[mask], 1)
        x2 = np.linspace(all_dElev[mask].min(), all_dElev[mask].max(), 50)
        ax.plot(
          x2,
          np.poly1d(z2)(x2),
          "r--",
          lw=1.5
        )
        sig2 = "**" if (not np.isnan(p_corr2) and p_corr2 < 0.05) else "ns"
        ax.text(
            0.05,
            0.95,
            f"r = {r2:.3f}\np_corr = {p_corr2:.4e} {sig2}\nn_eff = {n_eff2}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=common_bbox(),
            family="monospace"
        )
    ax.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Elevation change per step (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Speed change per step (km/h)", fontsize=11, fontweight="bold")
    ax.set_title("Terrain Ascent vs Speed Change\n(positive dElev = moving uphill)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    # Save
    orographic_speed_reduction_path = os.path.join(output_dir, "orographic_speed_reduction.png")
    plt.savefig(orographic_speed_reduction_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Orographic speed reduction plot", orographic_speed_reduction_path)


# Orographic precipitation enhancement ratio
def compute_orographic_precip_ratio(tracks_df_dict, ds, output_dir):
    """
    Orographic enhancement ratio:
      OER = mean_TP_upslope / mean_TP_flat
 
    Upslope = track points where elevation gradient > threshold (ascending)
    Flat = track points where |elevation gradient| < threshold
    """
    os.makedirs(output_dir, exist_ok=True)
    GRAD_THRESHOLD = 10   # m per track step
 
    oer_results = []
    for track_id, df in tracks_df_dict.items():
        if "elevation" not in df.columns or "tp_mmhr" not in df.columns:
            continue
        df = df.dropna(subset=["elevation", "tp_mmhr"]).copy()
        if len(df) < 5:
            continue
        d_elev = df["elevation"].diff().fillna(0).values
        tp_vals = df["tp_mmhr"].values
 
        upslope_tp = tp_vals[d_elev >  GRAD_THRESHOLD]
        flat_tp = tp_vals[np.abs(d_elev) <= GRAD_THRESHOLD]
 
        if len(upslope_tp) < 2 or len(flat_tp) < 2 or np.mean(flat_tp) < 1e-6:
            continue
        oer = float(np.mean(upslope_tp) / np.mean(flat_tp))
        oer_results.append({
            "track_id": track_id,
            "oer": oer,
            "mean_tp_upslope": float(np.mean(upslope_tp)),
            "mean_tp_flat": float(np.mean(flat_tp)),
            "n_upslope": len(upslope_tp),
            "n_flat": len(flat_tp)
        })
        std_print("INFO", "ea", "OER", f"{oer:.3f}")
 
    if not oer_results:
        std_print("WARN", "ea", "No OER results - add tp_mmhr column to track DataFrames")
        return pd.DataFrame()
    oer_df = pd.DataFrame(oer_results)

    # Export
    oer_CSV= os.path.join(output_dir, "orographic_precip_ratio.csv")
    oer_df.to_csv(oer_CSV, index=False)
    std_print("EXPORT", "ea", "OER results", oer_CSV)
 
    # Figure
    fig, ax = plt.subplots(figsize=(10, 6))

    valid = oer_df["oer"].dropna()
    valid_mean = valid.mean()
    ax.hist(valid, bins=15, edgecolor="black", alpha=0.75, color="teal")
    ax.axvline(1.0, color="black", linestyle="--", lw=1.5, label="OER = 1.0 (no enhancement)")
    ax.axvline(valid_mean, color="red",  linestyle="--", lw=1.5, label=f"Mean OER = {valid_mean:.2f}")
    ax.set_xlabel("Orographic Enhancement Ratio (OER)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Track count", fontsize=11, fontweight="bold")
    ax.set_title("Orographic Precipitation Enhancement\n(upslope TP / flat TP)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    legend_handles, _ = ax.get_legend_handles_labels()
    if legend_handles:
        ax.legend(handles=legend_handles, **legend_style())
    plt.tight_layout()

    # Save
    oer_ratio_path = os.path.join(output_dir, "orographic_precip_ratio.png")
    plt.savefig(oer_ratio_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "OER plot", oer_ratio_path)
    return oer_df


# Elevation vs inland penetration scatter
# FIX: empty output plot
def plot_elevation_vs_penetration(tsi_df, output_dir):
    """
      1. Mean elevation  vs penetration distance
      2. TSI             vs penetration distance
      3. Elevation range vs penetration distance
    """
    if "penetration_km" not in tsi_df.columns:
        std_print("WARN", "ea", "plot_elevation_vs_penetration: missing penetration_km column")
        return
 
    pairs = [
        ("mean_elevation",  "Mean elevation (m)",    "steelblue"),
        ("TSI",             "TSI",                   "darkorange"),
        ("elevation_range", "Elevation range (m)",   "purple"),
    ]
    pairs = [(col, lbl, c) for col, lbl, c in pairs if col in tsi_df.columns]
    n = len(pairs)
    if n == 0:
        return
 
    # Figure
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]
 
    for ax, (col, lbl, color) in zip(axes, pairs):
        sub = tsi_df.dropna(subset=[col, "penetration_km"])
        if len(sub) < 3:
            continue
        x, y = sub[col].values, sub["penetration_km"].values
        ax.scatter(x, y, s=80, alpha=0.75, c=color, edgecolors="black", lw=0.7, zorder=3)

        r, p_raw, p_corr, n_eff = pearsonr_corrected(x, y)
        rho, p = spearmanr(x, y)
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(
          x_line,
          np.poly1d(z)(x_line),
          "r--",
          lw=1.5,
          alpha=0.7
        )
        sig = "**" if (not np.isnan(p_corr) and p_corr < 0.05) else "ns"
        p_str = f"{p_corr:.4e}" if not np.isnan(p_corr) else "N/A"
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}\nρ = {rho:.3f}\np_corr = {p_str} {sig}\nn_eff = {n_eff}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=common_bbox(),
            family="monospace"
        )
        ax.set_xlabel(lbl, fontsize=11, fontweight="bold")
        ax.set_title(f"{lbl} vs\nPenetration", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
 
    axes[0].set_ylabel("Inland penetration distance (km)", fontsize=11, fontweight="bold")
    fig.suptitle("Terrain Metrics vs Inland Penetration Distance",fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    elevation_vs_penetration_path = os.path.join(output_dir, "elevation_vs_penetration.png")
    plt.savefig(elevation_vs_penetration_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Elevation vs penetration scatter", elevation_vs_penetration_path)


def plot_speed_anomaly_vs_elevation(tsi_df, output_dir):
    if "mean_elevation" not in tsi_df or "speed_anomaly" not in tsi_df:
        std_print("WARN", "ea", "Missing columns for speed anomaly plot")
        return

    # Figure
    fig, ax = plt.subplots(figsize=(8,6))

    ax.scatter(tsi_df["mean_elevation"], tsi_df["speed_anomaly"], alpha=0.7, edgecolors='k')
    ax.set_xlabel("Mean elevation (m)")
    ax.set_ylabel("Speed anomaly (m/s)")
    ax.set_title("Mean elevation vs speed anomaly")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    elevation_vs_speed_anomaly_path = os.path.join(output_dir, "elevation_vs_speed_anomaly.png")
    plt.savefig(elevation_vs_speed_anomaly_path, dpi=300)
    plt.close(fig)
    std_print("SAVED", "ea", "Speed anomaly vs elevation", elevation_vs_speed_anomaly_path)


# Landfall elevation classification
def classify_landfall_elevation(tsi_df, output_dir):
    # FIX: verify 50-600 criterion for flat or elevated region
    """
    Classify MDs by landfall-point elevation:
      - Low coast  : elevation at landfall < 50 m (flat)
      - Elevated   : elevation at landfall 50–600 m
      - Terrain-facing: elevation > 600 m  (Western / Eastern Ghats exposure)
 
    Compares penetration and duration between classes.
    """
    if "elevation_at_landfall" not in tsi_df.columns:
        std_print("WARN", "ea", "classify_landfall_elevation: missing elevation_at_landfall column")
        return None
 
    tsi_df = tsi_df.copy()
    tsi_df["lf_class"] = tsi_df["elevation_at_landfall"].apply(lf_class)
    std_print("INFO", "ea", "Landfall class distribution", tsi_df["lf_class"].value_counts().to_dict())
 
    # Export
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "landfall_elevation_class.csv")
    tsi_df[["track_id", "lf_class", "elevation_at_landfall","penetration_km", "duration_h"]].to_csv(csv_path, index=False)
    std_print("EXPORT", "ea", "Landfall elevation classes", csv_path)
 
    # penetration by class
    classes = [c for c in ["Low coast (<50 m)", "Elevated (50-600 m)", "Terrain-facing (>600 m)"]
               if c in tsi_df["lf_class"].values]
    if "penetration_km" in tsi_df.columns and len(classes) >= 2:
        # Figure
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [tsi_df[tsi_df["lf_class"] == c]["penetration_km"].dropna().values for c in classes]
        bp = ax.boxplot(
            data,
            label=classes,
            patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2)
        )
        ax.set_ylabel("Inland penetration (km)", fontsize=11, fontweight="bold")
        ax.set_title("Inland Penetration by Landfall Elevation Class", fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        for i, d in enumerate(data):
            ax.text(
                i + 1,
                max(d) if len(d) else 0,
                f"n={len(d)}",
                ha="center",
                va="bottom",
                fontsize=9
            )
        plt.tight_layout()

        # Save
        landfall_class_penetration_path= os.path.join(output_dir, "landfall_class_penetration.png")
        plt.savefig(landfall_class_penetration_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "ea", "Landfall class penetration plot", landfall_class_penetration_path)
    return tsi_df


def plot_track_elevation_speed_map(track_df, track_id, output_dir):
    """
    DEM tiles, track line colored by translation speed
    Tracks ran into high terrain vs passed through low-elevation corridors
    """
    track_id = int(track_id)
    required = {"latitude", "longitude", "speed_ms", "is_land"}
    if not required.issubset(track_df.columns):
        std_print("WARN", "ea", "plot_track_elevation_speed_map missing cols",
                  str(required - set(track_df.columns)))
        return

    df = track_df.sort_values("time").reset_index(drop=True)
    lats = df["latitude"].values
    lons = df["longitude"].values
    speeds = df["speed_ms"].values.astype(float)

    # Bounding box with buffer
    buf = 1.5
    lat_min = lats.min() - buf
    lat_max = lats.max() + buf
    lon_min = lons.min() - buf
    lon_max = lons.max() + buf

    # Figure
    fig, ax = plt.subplots(figsize=(13, 11),subplot_kw={"projection": ccrs.PlateCarree()})

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # DEM tiles
    tile_patches = []
    for lat_int in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        for lon_int in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            tile_path = get_tile_filename(lat_int + 0.5, lon_int + 0.5)
            if tile_path is None:
                continue
            tile_data = load_hgt_tile(tile_path)
            if tile_data is None:
                continue
            elev = tile_data["data"].copy()
            elev[elev < 0] = 0     # treat void/ocean pixels as 0
            # Downsample for display speed (step ≈ 30 arc-seconds)
            step = max(1, elev.shape[0] // 120)
            elev_ds = elev[::step, ::step]
            tile_patches.append((elev_ds, lon_int, lat_int))

    if tile_patches:
        all_vals = [np.nanpercentile(p[0], 98) for p in tile_patches]
        vmax_elev = max(float(np.nanmax(all_vals)), 600.0)
    else:
        vmax_elev = 1000.0
        ax.add_feature(cfeature.LAND, facecolor="#E8D5B0", alpha=0.6)

    for (elev_ds, lon_int, lat_int) in tile_patches:
        img_extent = (lon_int, lon_int + 1, lat_int, lat_int + 1)
        ax.imshow(elev_ds, extent=img_extent, origin="upper",cmap="terrain", vmin=0, vmax=vmax_elev,transform=ccrs.PlateCarree(), alpha=0.85, zorder=1)

    # proxy for the elevation colorbar
    sm_elev = plt.cm.ScalarMappable(cmap="terrain",norm=plt.Normalize(0, vmax_elev))
    sm_elev.set_array([])
    cbar_elev = plt.colorbar(sm_elev, ax=ax, pad=0.01, shrink=0.55, aspect=20)
    cbar_elev.set_label("Elevation (m)", fontsize=10)

    # Elevation contours
    if "elevation" in df.columns:
        elev_vals = df["elevation"].values
        land_mask = df["is_land"].values.astype(bool)

        # for thresh, color, ls in [(600, "brown", "--"), (700, "sienna", "-.")]:
        for thresh, color, ls in [(600, "brown", "--")]:
            crossings = np.where(np.diff((elev_vals > thresh).astype(int)))[0]
            for ci in crossings:
                ax.plot(
                  lons[ci],
                  lats[ci],
                  "^",
                  color=color,
                  markersize=7,
                  transform=ccrs.PlateCarree(),
                  zorder=5,
                  label=f"{thresh} m crossing"
                )

    # Track line colored by speed
    valid_speed = np.where(np.isnan(speeds), np.nanmedian(speeds), speeds)
    speed_min = max(0, np.nanpercentile(valid_speed, 5))
    speed_max = np.nanpercentile(valid_speed, 95)

    points = np.array([lons, lats]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm_speed = plt.Normalize(speed_min, speed_max)
    lc = LineCollection(segments, cmap="RdYlGn",norm=norm_speed, linewidth=3.0,transform=ccrs.PlateCarree(), zorder=6)
    lc.set_array(valid_speed[:-1])
    ax.add_collection(lc)

    cbar_speed = plt.colorbar(lc, ax=ax, pad=0.08, shrink=0.55, aspect=20)
    cbar_speed.set_label("Translation Speed (m/s)", fontsize=10)

    # # Start / End / Landfall markers
    # ax.plot(lons[0], lats[0], "g*", markersize=10,transform=ccrs.PlateCarree(), zorder=8,label="Origin", markeredgecolor="black", markeredgewidth=0.5)
    # ax.plot(lons[-1], lats[-1], "rx", markersize=10, markeredgewidth=2,transform=ccrs.PlateCarree(), zorder=8, label="Dissipation")

    is_land_arr = df["is_land"].values.astype(bool)
    for i in range(1, len(df)):
        if not is_land_arr[i - 1] and is_land_arr[i]:
            ax.plot(
              lons[i],
              lats[i],
              "D",
              color="darkorange",
              markersize=7,
              transform=ccrs.PlateCarree(),
              zorder=8,
              label="Landfall",
              markeredgecolor="black",
              markeredgewidth=0.5
            )
            break

    ax.coastlines(resolution="50m", linewidth=1.0, color="black", zorder=7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=":", color="gray", zorder=7)
    ocean_feature = cfeature.NaturalEarthFeature("physical", "ocean", "50m", facecolor="lightcyan", edgecolor="none")
    ax.add_feature(ocean_feature, facecolor="lightcyan", alpha=0.4, zorder=2)
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4, linewidth=0.5)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="lower left", **legend_style())
    ax.set_title(f"Track {track_id}: Track over Terrain",fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    elevation_speed_map_path = os.path.join(output_dir, f"track_{track_id}_elevation_spped_map.png")
    plt.savefig(elevation_speed_map_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "elevation map", elevation_speed_map_path)


def plot_track_elevation_profile(track_df, track_id, output_dir):
    """
    Elevation along track (left) vs translation speed (right).
    Highlights where terrain rise coincides with speed drop.
    """
    track_id = int(track_id)
    required = {"time", "elevation", "speed_ms", "is_land"}
    missing = required - set(track_df.columns)
    if missing:
        std_print("WARN", "ea", "plot_track_elevation_profile missing cols", str(missing))
        return

    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])
    elevation = df["elevation"].values.astype(float)
    speed = df["speed_ms"].values.astype(float)
    is_land_arr = df["is_land"].values.astype(bool)

    # Landfall time (first ocean→land transition)
    landfall_time = None
    for i in range(1, len(df)):
        if not is_land_arr[i - 1] and is_land_arr[i]:
            landfall_time = times.iloc[i]
            break

    # Figure
    fig, ax1 = plt.subplots(figsize=(16, 7))

    ax1.fill_between(
      times,
      0,
      1,
      where=is_land_arr,
      transform=ax1.get_xaxis_transform(),
      color="wheat",
      alpha=0.45,
      label="Land phase",
      zorder=0
    )
    ax1.fill_between(
      times,
      0,
      1,
      where=~is_land_arr,
      transform=ax1.get_xaxis_transform(),
      color="lightblue",
      alpha=0.45,
      label="Ocean phase",
      zorder=0
    )

    # Elevation (left axis)
    elev_land = np.where(is_land_arr, elevation, np.nan)
    ax1.fill_between(times, 0, elev_land, color="#8B6914", alpha=0.55, label="Elevation (land)", zorder=2)
    ax1.plot(
      times,
      elev_land,
      color="#5C3D0A",
      linewidth=1.2,
      zorder=3
    )

    # Reference elevations
    # for ref_m, ls, label in [(600, "--", "600 m"), (500, "-.", "500 m")]:
    for ref_m, ls, label in [(600, "--", "600 m")]:
        ax1.axhline(ref_m, color="gray", linestyle=ls, linewidth=0.8, alpha=0.6, label=label)

    ax1.set_ylabel("Elevation (m)", fontsize=12, fontweight="bold", color="#5C3D0A")
    ax1.tick_params(axis="y", labelcolor="#5C3D0A")
    elev_max = np.nanmax(elev_land) if not np.all(np.isnan(elev_land)) else 600
    ax1.set_ylim(0, max(elev_max * 1.25, 600))

    # Speed (right axis)
    ax2 = ax1.twinx()
    ax2.plot(
      times,
      speed,
      "-o",
      color="royalblue",
      linewidth=2.2,
      markersize=4,
      alpha=0.85,
      label="Speed (m/s)",
      zorder=4
    )
    ax2.set_ylabel("Translation Speed (m/s)", fontsize=12, fontweight="bold", color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")

    # Landfall line
    if landfall_time is not None:
        ax1.axvline(landfall_time, color="red", linestyle="--", linewidth=1.8,label="Landfall", zorder=5)

    # Annotate peak elevation on land
    valid_land = np.where(is_land_arr & ~np.isnan(elevation))[0]
    if len(valid_land) > 0:
        peak_idx = valid_land[np.argmax(elevation[valid_land])]
        peak_elev = elevation[peak_idx]
        peak_speed = speed[peak_idx]
        ax1.annotate(
            f"Peak: {peak_elev:.0f} m\nSpeed: {peak_speed:.1f} m/s",
            xy=(times.iloc[peak_idx], peak_elev),
            xytext=(15, 10),
            textcoords="offset points",
            fontsize=9,
            color="#5C3D0A",
            arrowprops=dict(arrowstyle="->", color="#5C3D0A", lw=1.0),
            bbox=common_bbox(),
        )
        # Minimum speed on land
        min_speed_idx = valid_land[np.argmin(speed[valid_land])]
        ax2.annotate(
            f"Min: {speed[min_speed_idx]:.1f} m/s",
            xy=(times.iloc[min_speed_idx], speed[min_speed_idx]),
            xytext=(15, -20),
            textcoords="offset points",
            fontsize=9,
            color="royalblue",
            arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.0),
            bbox=common_bbox(),
        )

    # Pearson r: elevation vs speed (land phase only)
    land_elev = elevation[is_land_arr & ~np.isnan(elevation) & ~np.isnan(speed)]
    land_speed = speed[is_land_arr & ~np.isnan(elevation) & ~np.isnan(speed)]
    if len(land_elev) >= 5:
        r, _, p_corr, n_eff = pearsonr_corrected(land_elev, land_speed)
        sig = "**" if (not np.isnan(p_corr) and p_corr < 0.05) else "ns"
        p_str = f"{p_corr:.3e}" if not np.isnan(p_corr) else "N/A"
        stats_text = (
            f"Land phase (n_eff={n_eff})\n"
            f"r(elev, speed) = {r:.3f}\n"
            f"p_corr = {p_str} {sig}"
        )
        ax1.text(
          0.02,
          0.97,
          stats_text,
          transform=ax1.transAxes,
          fontsize=9,
          va="top",
          family="monospace",
          bbox=common_bbox()
        )

    # Lag-correlation: r(elevation[t], speed[t+k]) for k in -12..+12 h
    # INFO: lag = +3h means the MD speed drops about 3 hours after encountering high terrain
    lags_h = list(range(-12, 13, 1))
    lag_rs = []
    for lag in lags_h:
        if lag < 0:
            xe, xs = land_elev[:lag],  land_speed[-lag:]
        elif lag > 0:
            xe, xs = land_elev[lag:],  land_speed[:-lag]
        else:
            xe, xs = land_elev, land_speed
        if len(xe) < 5 or np.std(xe) < 1e-6 or np.std(xs) < 1e-6:
            lag_rs.append(np.nan)
        else:
            r_lag, _ = pearsonr_corrected(xe, xs)[:2]
            lag_rs.append(float(r_lag) if not np.isnan(r_lag) else np.nan)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,loc="upper right", **legend_style())

    ax1.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax1.set_title(f"Track {track_id}: Elevation Profile vs Translation Speed",fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--", zorder=0)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    elevation_profile_path = os.path.join(output_dir, f"track_{track_id}_elevation_profile.png")
    plt.savefig(elevation_profile_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Elevation profile", elevation_profile_path)

    if len(lag_rs) == len(lags_h) and not all(np.isnan(v) for v in lag_rs):
        best_lag = lags_h[np.nanargmin(lag_rs)]   # most negative r = strongest braking lag
        lag_rs_arr = np.array(lag_rs, dtype=float)

        fig_lag, ax_lag = plt.subplots(figsize=(10, 5))
        ax_lag.plot(
            lags_h,
            lag_rs_arr,
            color="steelblue",
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="r(elevation[t], speed[t+lag])"
        )
        ax_lag.fill_between(
            lags_h,
            lag_rs_arr,
            0,
            where=~np.isnan(lag_rs_arr),
            alpha=0.15,
            color="steelblue"
        )
        ax_lag.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax_lag.axvline(0, color="gray",  linewidth=0.8, linestyle="--", label="Lag = 0")
        ax_lag.axvline(best_lag, color="navy",  linewidth=1.5, linestyle=":",label=f"Min r at lag {best_lag:+d} h")
        ax_lag.set_xlabel("Lag (hours)", fontsize=11, fontweight="bold")
        ax_lag.set_ylabel("Pearson r", fontsize=11, fontweight="bold")
        ax_lag.set_title(
            f"Track {track_id}: Lag correlation – elevation vs translation speed\n"
            f"(negative lag = terrain leads speed reduction; min r at {best_lag:+d} h)",
            fontsize=13, fontweight="bold"
        )
        ax_lag.set_ylim(-1, 1)
        ax_lag.grid(True, alpha=0.3, linestyle="--")
        ax_lag.legend(**legend_style())
        plt.tight_layout()

        # Save
        lag_outpath = os.path.join(output_dir, f"track_{track_id}_elevation_lag.png")
        plt.savefig(lag_outpath, dpi=300, bbox_inches="tight")
        plt.close(fig_lag)
        std_print("SAVED", "ea", "elevation lag plot", lag_outpath)


def plot_all_tracks_elevation_comparison(track_dict, output_dir):
    """
    All tracks overlaid on DEM
    Line width proportional to speed (thin = slow = terrain-braked, thick = fast = open corridor)
    """
    if not track_dict:
        return

    all_lats = np.concatenate([df["latitude"].values for df in track_dict.values()])
    all_lons = np.concatenate([df["longitude"].values for df in track_dict.values()])
    buf = 2.0
    lat_min, lat_max = all_lats.min() - buf, all_lats.max() + buf
    lon_min, lon_max = all_lons.min() - buf, all_lons.max() + buf

    # Figure
    fig = plt.figure(figsize=(13, 10), dpi=100)
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # DEM background
    tile_patches = []
    for lat_int in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        for lon_int in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            tp = get_tile_filename(lat_int + 0.5, lon_int + 0.5)
            if tp is None:
                continue
            td = load_hgt_tile(tp)
            if td is None:
                continue
            elev = td["data"].copy()
            elev[elev < 0] = 0
            step = max(1, elev.shape[0] // 120)
            tile_patches.append((elev[::step, ::step], lon_int, lat_int))

    vmax_elev = max(
        (float(np.nanpercentile(p[0], 98)) for p in tile_patches),
        default=1000.0
    )
    for elev_ds, lon_int, lat_int in tile_patches:
        ax.imshow(
            elev_ds,
            extent=(lon_int, lon_int+1, lat_int, lat_int+1),
            origin="upper",
            cmap="terrain",
            vmin=0,
            vmax=vmax_elev,
            transform=ccrs.PlateCarree(),
            alpha=0.75,
            zorder=1
        )

    sm_elev = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(0, vmax_elev))
    sm_elev.set_array([])
    cbar = plt.colorbar(sm_elev, ax=ax, pad=0.06, shrink=0.5, aspect=18)
    cbar.set_label("Elevation (m)", fontsize=10)

    # Per-track overlays
    track_colors = plt.cm.tab10(np.linspace(0, 1, len(track_dict)))
    summary_rows = []
    legend_handles = []

    for (track_id, track_df), color in zip(sorted(track_dict.items()), track_colors):
        df = track_df.sort_values("time").reset_index(drop=True)
        lats = df["latitude"].values
        lons = df["longitude"].values
        is_land_arr = df["is_land"].values.astype(bool) if "is_land" in df.columns else np.ones(len(df), bool)

        # Speed column
        speed_col = (
            "speed_ms"  if "speed_ms"  in df.columns else
            "speed_kmh" if "speed_kmh" in df.columns else
            None
        )
        speeds = df[speed_col].values.astype(float) if speed_col else np.full(len(df), 15.0)
        speed_unit = "m/s" if speed_col == "speed_ms" else "km/h"

        # Track line - width mapped to speed (thin = slow, thick = fast)
        speed_norm = np.clip(
            (speeds - np.nanpercentile(speeds, 5))
            / (np.nanpercentile(speeds, 95) - np.nanpercentile(speeds, 5) + 1e-6),
            0, 1
        )
        lw_arr = 2.0 + speed_norm * 4.0

        points = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lws = lw_arr[:-1]

        for seg, lw in zip(segments, lws):
            ax.plot(
                seg[:, 0], seg[:, 1],
                color="white",
                linewidth=lw + 1.0,
                transform=ccrs.PlateCarree(),
                solid_capstyle="round",
                zorder=6,
            )
        for seg, lw in zip(segments, lws):
            ax.plot(
                seg[:, 0], seg[:, 1],
                color=color,
                linewidth=lw,
                transform=ccrs.PlateCarree(),
                solid_capstyle="round",
                zorder=6.01,
            )

        # Start / End markers
        ax.scatter(
          lons[0],
          lats[0],
          color='green',
          marker='o',
          s=100,
          edgecolors='white',
          linewidths=1,
          zorder=10,
          transform=ccrs.PlateCarree(),
          label="Start" if track_id == sorted(track_dict.keys())[0] else ""
        )
        ax.scatter(
          lons[-1],
          lats[-1],
          color='red',
          marker='o',
          s=100,
          edgecolors='white',
          linewidths=1,
          zorder=10,
          transform=ccrs.PlateCarree(),
          label="End" if track_id == sorted(track_dict.keys())[0] else ""
        )

        ax.annotate(
          f"T{track_id}",
          xy=(lons[0], lats[0]),
          xytext=(4, 6),
          textcoords="offset points",
          fontsize=6,
          fontweight="bold",
          color=color,
          transform=ccrs.PlateCarree(),
          bbox=common_bbox()
        )

        # Landfall marker
        for i in range(1, len(df)):
            if not is_land_arr[i-1] and is_land_arr[i]:
                ax.plot(
                  lons[i],
                  lats[i],
                  "D",
                  color=color,
                  markersize=6,
                  markeredgecolor="black",
                  markeredgewidth=0.7,
                  transform=ccrs.PlateCarree(),
                  zorder=9
                )
                break

        # Terrain stats for legend and CSV
        if "elevation" in df.columns:
            land_elev = df.loc[is_land_arr, "elevation"].dropna()
            max_elev = float(land_elev.max())  if len(land_elev) else np.nan
            mean_elev = float(land_elev.mean()) if len(land_elev) else np.nan
        else:
            max_elev = mean_elev = np.nan

        land_speed = speeds[is_land_arr & ~np.isnan(speeds)]
        mean_speed_land = float(np.mean(land_speed)) if len(land_speed) else np.nan

        pen_km = np.nan
        for i in range(1, len(df)):
            if not is_land_arr[i-1] and is_land_arr[i]:
                pen_km = haversine_distance(lons[i], lats[i], lons[-1], lats[-1])
                break

        summary_rows.append({
            "track_id":        track_id,
            "max_land_elev_m":  round(max_elev, 0),
            "mean_land_elev_m": round(mean_elev, 0),
            "mean_land_speed":  round(mean_speed_land, 1),
            "penetration_km":   round(pen_km, 0),
        })

        legend_handles.append(Line2D(
            [0], [0], color=color, linewidth=3,
            label=f"MD Track {track_id}"
                  # f"max:{max_elev:.0f}m  "
                  # f"pen:{pen_km:.0f}km  "
                  # f"spd:{mean_speed_land:.1f}{speed_unit}"
        ))

    # 600 m threshold legend proxy (no actual contour - only a legend entry)
    ax.plot([], [], linestyle="--", color="saddlebrown", linewidth=1.5, label="600 m threshold")

    ax.coastlines(resolution="50m", linewidth=1.0, color="black", zorder=7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=":", color="gray", zorder=7)
    ocean_feat = cfeature.NaturalEarthFeature("physical", "ocean", "50m", facecolor="lightcyan", edgecolor="none")
    ax.add_feature(ocean_feat, alpha=0.4, zorder=2)
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.35, linewidth=0.5)
    gl.right_labels = False

    ax.legend(handles=legend_handles, loc="lower left", **legend_style({"fontsize": 9}))
    ax.set_title("MD Tracks over Terrain", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    elevation_comparison_path = os.path.join(output_dir, "all_tracks_elevation_comparison.png")
    plt.savefig(elevation_comparison_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "All-tracks elevation comparison", elevation_comparison_path)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "track_terrain_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        std_print("EXPORT", "ea", "Track terrain summary", csv_path)
    return summary_rows


def plot_all_tracks_elevation_speed_map(track_dict, output_dir):
    if not track_dict:
        return

    all_lats = np.concatenate([df["latitude"].values  for df in track_dict.values()])
    all_lons = np.concatenate([df["longitude"].values for df in track_dict.values()])
    buf = 2.0
    lat_min, lat_max = all_lats.min() - buf, all_lats.max() + buf
    lon_min, lon_max = all_lons.min() - buf, all_lons.max() + buf

    # Figure
    fig, ax = plt.subplots(figsize=(16, 13), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # DEM background
    tile_patches = []
    for lat_int in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        for lon_int in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            tp = get_tile_filename(lat_int + 0.5, lon_int + 0.5)
            if tp is None:
                continue
            td = load_hgt_tile(tp)
            if td is None:
                continue
            elev = td["data"].copy()
            elev[elev < 0] = 0
            step = max(1, elev.shape[0] // 120)
            tile_patches.append((elev[::step, ::step], lon_int, lat_int))

    if tile_patches:
        vmax_elev = max(float(np.nanpercentile(p[0], 98)) for p in tile_patches)
    else:
        vmax_elev = 1000.0
        ax.add_feature(cfeature.LAND, facecolor="#E8D5B0", alpha=0.6)

    for elev_ds, lon_int, lat_int in tile_patches:
        ax.imshow(
            elev_ds,
            extent=(lon_int, lon_int + 1, lat_int, lat_int + 1),
            origin="upper",
            cmap="terrain",
            vmin=0,
            vmax=vmax_elev,
            transform=ccrs.PlateCarree(),
            alpha=0.80,
            zorder=1,
        )

    # Elevation colorbar on the left so it doesn't clash with the speed cbar on right
    sm_elev = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(0, vmax_elev))
    sm_elev.set_array([])
    cbar_elev = plt.colorbar(sm_elev, ax=ax, location="right", pad=0.06, shrink=0.5, aspect=18)
    cbar_elev.set_label("Elevation (m)", fontsize=10)

    # Shared speed normalisation across ALL tracks
    all_speeds = []
    for track_df in track_dict.values():
        col = "speed_ms" if "speed_ms" in track_df.columns else \
              "speed_kmh" if "speed_kmh" in track_df.columns else None
        if col:
            vals = track_df[col].values.astype(float)
            all_speeds.extend(vals[~np.isnan(vals)].tolist())
    if all_speeds:
        speed_min = max(0.0, float(np.percentile(all_speeds, 5)))
        speed_max = float(np.percentile(all_speeds, 95))
    else:
        speed_min, speed_max = 0.0, 15.0
    speed_norm = plt.Normalize(speed_min, speed_max)
    speed_unit = "m/s" if any("speed_ms" in df.columns for df in track_dict.values()) else "km/h"

    # Per-track markers
    track_colors = plt.cm.tab10(np.linspace(0, 1, len(track_dict)))
    for (track_id, track_df), marker_color in zip(sorted(track_dict.items()), track_colors):
        df = track_df.sort_values("time").reset_index(drop=True)
        lats = df["latitude"].values
        lons = df["longitude"].values
        is_land_arr = df["is_land"].values.astype(bool) if "is_land" in df.columns else np.ones(len(df), bool)

        speed_col = "speed_ms"  if "speed_ms"  in df.columns else "speed_kmh" if "speed_kmh" in df.columns else None
        speeds = df[speed_col].values.astype(float) if speed_col else np.full(len(df), (speed_min + speed_max) / 2)
        # Fill any NaN with median
        speeds = np.where(np.isnan(speeds), np.nanmedian(speeds), speeds)

        # Track line as LineCollection colored by speed
        points = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # White halo so the line reads over any terrain color
        lc_halo = LineCollection(segments, linewidth=5.5, color="white", transform=ccrs.PlateCarree(), zorder=5.9)
        ax.add_collection(lc_halo)

        lc = LineCollection(segments, cmap="RdYlGn", norm=speed_norm, linewidth=3.0, transform=ccrs.PlateCarree(), zorder=6.0)
        lc.set_array(speeds[:-1])
        ax.add_collection(lc)

        ax.plot(
          lons[0],
          lats[0],
          "o",
          color=marker_color,
          markersize=13,
          markeredgecolor="black",
          markeredgewidth=0.6,
          transform=ccrs.PlateCarree(),
          zorder=9
        )
        ax.plot(
          lons[-1],
          lats[-1],
          "o",
          color=marker_color,
          markersize=10,
          markeredgecolor="black",
          markeredgewidth=0.6,
          transform=ccrs.PlateCarree(),
          zorder=9
        )
        ax.annotate(
            f"T{track_id}",
            xy=(lons[0], lats[0]),
            xytext=(4, 6), textcoords="offset points",
            fontsize=8, fontweight="bold", color=marker_color,
            transform=ccrs.PlateCarree(),
            bbox=common_bbox(),
            zorder=10,
        )
        # Landfall ◆
        for i in range(1, len(df)):
            if not is_land_arr[i - 1] and is_land_arr[i]:
                ax.plot(lons[i], lats[i], "D", color="darkorange", markersize=8, markeredgecolor="black", markeredgewidth=0.5, transform=ccrs.PlateCarree(), zorder=9, label="Landfall" if track_id == sorted(track_dict)[0] else "_nolegend_")
                break

    # Speed colorbar
    sm_speed = plt.cm.ScalarMappable(cmap="RdYlGn", norm=speed_norm)
    sm_speed.set_array([])
    cbar_speed = plt.colorbar(sm_speed, ax=ax, location="right", pad=0.02, shrink=0.5, aspect=18)
    cbar_speed.set_label(f"Translation Speed ({speed_unit})", fontsize=10)

    ax.coastlines(resolution="50m", linewidth=1.0, color="black", zorder=7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=":", color="gray", zorder=7)
    ocean_feat = cfeature.NaturalEarthFeature("physical", "ocean", "50m", facecolor="lightcyan", edgecolor="none")
    ax.add_feature(ocean_feat, alpha=0.4, zorder=2)
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.35, linewidth=0.5)
    gl.right_labels = False

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="*",  color="gray", markersize=10,linestyle="none", label="Origin", markeredgecolor="black"),
        Line2D([0], [0], marker="X",  color="gray", markersize=9,linestyle="none", label="Dissipation", markeredgecolor="black"),
        Line2D([0], [0], marker="D",  color="darkorange", markersize=8,linestyle="none", label="Landfall", markeredgecolor="black")
    ]
    # One entry per track (marker color)
    for (track_id, _), mc in zip(sorted(track_dict.items()), track_colors):
        legend_handles.append(Line2D([0], [0], color=mc, linewidth=2.5, label=f"T{track_id}"))
    ax.legend(handles=legend_handles, loc="lower left", **legend_style({"fontsize": 9}))
    ax.set_title("Translation Speed over Elevation",fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "all_tracks_elevation_speed_map.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Global elevation-speed map", outpath)
    return outpath


def detect_barrier_encounters(elevation, times, threshold_m_per_h=15.0, min_steps=3):
    """
    Identify terrain barrier encounters along a land-phase track.

    A barrier encounter is a run of >= min_steps consecutive steps where the
    rate of elevation gain (dElev/dt in m h⁻¹) exceeds threshold_m_per_h.

    Returns a list of dicts:
      {start_idx, end_idx, start_time, end_time, max_dElev, mean_dElev}
    """
    dt_h = np.array([
        (pd.to_datetime(times.iloc[i]) - pd.to_datetime(times.iloc[i-1])).total_seconds() / 3600.0
        if i > 0 else np.nan
        for i in range(len(times))
    ])
    delev = np.diff(elevation, prepend=elevation[0])
    # Rate of elevation gain (m h⁻¹); NaN where dt == 0 or NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where((dt_h > 0) & ~np.isnan(dt_h), delev / dt_h, np.nan)

    # Find consecutive runs where rate > threshold
    ascending = (rate > threshold_m_per_h) & ~np.isnan(rate)
    encounters = []
    i = 0
    while i < len(ascending):
        if ascending[i]:
            j = i
            while j < len(ascending) and ascending[j]:
                j += 1
            if (j - i) >= min_steps:
                encounters.append({
                    "start_idx":  i,
                    "end_idx":    j - 1,
                    "start_time": times.iloc[i],
                    "end_time":   times.iloc[j - 1],
                    "max_dElev":  float(np.nanmax(rate[i:j])),
                    "mean_dElev": float(np.nanmean(rate[i:j])),
                })
            i = j
        else:
            i += 1
    return encounters


def plot_barrier_encounter_profile(track_df, track_id, output_dir,threshold_m_per_h=15.0, min_steps=3):
    """
    Elevation profile with terrain barrier encounters shaded in red and annotated.

    Same dual-axis layout as plot_track_elevation_profile but with:
      - Red shading over each barrier encounter window
      - Arrow annotation: "Barrier → speed ↓" at the first encounter
      - Speed response window shown for each encounter (speed 6 h after vs at start)

    Hypothesis test: Does each barrier encounter correspond to a speed drop
    in the following 6 h?
    """
    track_id = int(track_id)
    required = {"time", "elevation", "speed_ms", "is_land"}
    if not required.issubset(track_df.columns):
        std_print("WARN", "ea", "plot_barrier_encounter_profile missing cols",
                  str(required - set(track_df.columns)))
        return

    df = track_df.sort_values("time").reset_index(drop=True)
    times     = pd.to_datetime(df["time"])
    elevation = df["elevation"].values.astype(float)
    speed     = df["speed_ms"].values.astype(float)
    is_land   = df["is_land"].values.astype(bool)

    # Only analyse land-phase points
    land_elev  = np.where(is_land, elevation, np.nan)
    encounters = detect_barrier_encounters(
        land_elev, times,
        threshold_m_per_h=threshold_m_per_h,
        min_steps=min_steps,
    )

    if not encounters:
        std_print("INFO", "ea", "No barrier encounters detected",
                  f"(threshold={threshold_m_per_h} m/h, min_steps={min_steps})")

    # Figure
    fig, ax1 = plt.subplots(figsize=(16, 7))
    bbox_style = common_bbox()

    # Phase shading
    ax1.fill_between(
      times,
      0,
      1,
      where=is_land,
      transform=ax1.get_xaxis_transform(),
      color="wheat",
      alpha=0.35,
      label="Land phase",
      zorder=0
    )
    ax1.fill_between(
      times,
      0,
      1,
      where=~is_land,
      transform=ax1.get_xaxis_transform(),
      color="lightblue",
      alpha=0.35,
      label="Ocean phase",
      zorder=0
    )

    # Elevation
    ax1.fill_between(times, 0, land_elev, color="#8B6914", alpha=0.5,label="Elevation (land)", zorder=2)
    ax1.plot(times, land_elev, color="#5C3D0A", linewidth=1.2, zorder=3)
    ax1.axhline(600, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="600 m")

    elev_max = np.nanmax(land_elev) if not np.all(np.isnan(land_elev)) else 600
    ax1.set_ylim(0, max(elev_max * 1.3, 600))
    ax1.set_ylabel("Elevation (m)", fontsize=12, fontweight="bold", color="#5C3D0A")
    ax1.tick_params(axis="y", labelcolor="#5C3D0A")

    # Speed
    ax2 = ax1.twinx()
    ax2.plot(
      times,
      speed,
      "-o",
      color="royalblue",
      linewidth=2.2,
      markersize=4,
      alpha=0.85,
      label="Speed (m/s)",
      zorder=4
    )
    ax2.set_ylabel("Translation Speed (m/s)", fontsize=12, fontweight="bold",color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")

    # Landfall line
    for i in range(1, len(df)):
        if not is_land[i-1] and is_land[i]:
            ax1.axvline(times.iloc[i], color="red", linestyle="--",linewidth=1.8, label="Landfall", zorder=5)
            break

    # Shade barrier encounters + annotate speed response
    response_lines = []
    for enc in encounters:
        t0 = pd.to_datetime(enc["start_time"])
        t1 = pd.to_datetime(enc["end_time"])
        ax1.axvspan(t0, t1, color="tomato", alpha=0.25, zorder=1,label="_barrier" if enc != encounters[0] else "Barrier encounter")

        # Speed 6 h after encounter end vs at encounter start
        t_after = t1 + pd.Timedelta(hours=6)
        spd_at   = speed[enc["start_idx"]]
        idx_after = np.argmin(np.abs(times - t_after))
        spd_after = speed[idx_after]
        delta_spd = spd_after - spd_at
        sign = "↓" if delta_spd < 0 else "↑"
        label_txt = (f"Barrier\n{enc['mean_dElev']:.0f} m/h\nΔv={delta_spd:+.1f} m/s {sign}")
        ax2.annotate(
          label_txt,
          xy=(t0, spd_at),
          xytext=(10, 20),
          textcoords="offset points",
          fontsize=8,
          color="tomato",
          fontweight="bold",
          arrowprops=dict(arrowstyle="->", color="tomato", lw=1.0),
          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="tomato", alpha=0.85),
          zorder=8,
        )
        response_lines.append({
            "track_id": track_id,
            "encounter_start": str(t0),
            "encounter_end": str(t1),
            "mean_ascent_rate_m_per_h": enc["mean_dElev"],
            "speed_at_encounter_ms": float(spd_at),
            "speed_6h_after_ms": float(spd_after),
            "delta_speed_ms": float(delta_spd),
            "braking_detected": delta_spd < 0,
        })

    # Summary text
    if encounters:
        n_braked = sum(1 for r in response_lines if r["braking_detected"])
        ax1.text(
            0.02, 0.97,
            f"Barrier encounters: {len(encounters)}\n"
            f"Speed reduction detected: {n_braked}/{len(encounters)}",
            transform=ax1.transAxes, fontsize=9, va="top",
            family="monospace", bbox=bbox_style,
        )

    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2, loc="upper right", **legend_style())
    ax1.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Track {track_id}: Terrain Barrier Encounters & Speed Response\n"
        f"(Ascending ≥ {threshold_m_per_h:.0f} m/h for ≥ {min_steps} steps)",
        fontsize=13, fontweight="bold",
    )
    ax1.grid(True, alpha=0.3, linestyle="--", zorder=0)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"track_{track_id}_barrier_encounters.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Barrier encounter profile", outpath)

    # Export encounter table
    if response_lines:
        enc_csv = os.path.join(output_dir, f"track_{track_id}_barrier_encounters.csv")
        pd.DataFrame(response_lines).to_csv(enc_csv, index=False)
        std_print("EXPORT", "ea", "Barrier encounter table", enc_csv)
    return response_lines


def plot_dissipation_terrain_map(track_dict, output_dir):
    """
    Show where each track dissipates on the DEM.

    Primary spatial evidence for the hypothesis:
    "Short-lived MDs dissipate at terrain barriers; the long-lived MD
     dissipates on flat terrain after avoiding the Ghats."
    """
    if not track_dict:
        return

    all_lats = np.concatenate([df["latitude"].values for df in track_dict.values()])
    all_lons = np.concatenate([df["longitude"].values for df in track_dict.values()])
    buf = 2.0
    lat_min, lat_max = all_lats.min() - buf, all_lats.max() + buf
    lon_min, lon_max = all_lons.min() - buf, all_lons.max() + buf

    # Figure
    fig, ax = plt.subplots(figsize=(14, 12), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # DEM background
    tile_patches = []
    for lat_int in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        for lon_int in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            tp = get_tile_filename(lat_int + 0.5, lon_int + 0.5)
            if tp is None:
                continue
            td = load_hgt_tile(tp)
            if td is None:
                continue
            elev = td["data"].copy()
            elev[elev < 0] = 0
            step = max(1, elev.shape[0] // 120)
            tile_patches.append((elev[::step, ::step], lon_int, lat_int))

    if tile_patches:
        vmax_elev = max(float(np.nanpercentile(p[0], 98)) for p in tile_patches)
    else:
        vmax_elev = 1000.0
        ax.add_feature(cfeature.LAND, facecolor="#E8D5B0", alpha=0.6)

    for elev_ds, lon_int, lat_int in tile_patches:
        ax.imshow(
          elev_ds,
          extent=(lon_int, lon_int+1, lat_int, lat_int+1),
          origin="upper",
          cmap="terrain",
          vmin=0,
          vmax=vmax_elev,
          transform=ccrs.PlateCarree(),
          alpha=0.80,
          zorder=1
        )

    sm_elev = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(0, vmax_elev))
    sm_elev.set_array([])
    cbar = plt.colorbar(sm_elev, ax=ax, location="right", pad=0.06, shrink=0.5, aspect=18)
    cbar.set_label("Elevation (m)", fontsize=10)

    durations = {}
    for tid, df in track_dict.items():
        t = pd.to_datetime(df["time"])
        durations[tid] = (t.max() - t.min()).total_seconds() / 3600.0
    dur_vals = np.array(list(durations.values()))
    dur_norm = plt.Normalize(dur_vals.min(), dur_vals.max())
    dur_cmap = plt.cm.RdYlGn   # red = short-lived, green = long-lived

    legend_handles = []
    for track_id, df in sorted(track_dict.items()):
        df = df.sort_values("time").reset_index(drop=True)
        lats = df["latitude"].values
        lons = df["longitude"].values
        dur_h = durations[track_id]
        color = dur_cmap(dur_norm(dur_h))
        is_land_arr = df["is_land"].values.astype(bool) if "is_land" in df.columns else np.ones(len(df), bool)

        points   = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc_halo = LineCollection(
          segments,
          linewidth=3,
          color="white",
          transform=ccrs.PlateCarree(),
          zorder=5,
          alpha=0.7
        )
        lc_line = LineCollection(
          segments,
          linewidth=2,
          color=color,
          transform=ccrs.PlateCarree(),
          zorder=5.5,
          linestyle="--"
        )
        ax.add_collection(lc_halo)
        ax.add_collection(lc_line)

        # Landfall
        for i in range(1, len(df)):
            if not is_land_arr[i-1] and is_land_arr[i]:
                ax.plot(
                  lons[i],
                  lats[i],
                  "D",
                  color=color,
                  markersize=10,
                  markeredgecolor="black",
                  markeredgewidth=0.8,
                  transform=ccrs.PlateCarree(),
                  zorder=8
                )
                break

        # End point
        # FIX: end point to be circle
        diss_lon, diss_lat = lons[-1], lats[-1]
        diss_elev = df["elevation"].iloc[-1] if "elevation" in df.columns else np.nan
        ax.plot(
          diss_lon,
          diss_lat,
          "X",
          color=color,
          markersize=14,
          markeredgecolor="black",
          markeredgewidth=1.0,
          transform=ccrs.PlateCarree(),
          zorder=9
        )

        elev_str = f" @ {diss_elev:.0f} m" if not np.isnan(diss_elev) else ""
        ax.annotate(
          f"T{track_id}  {dur_h:.0f}h{elev_str}",
          xy=(diss_lon, diss_lat),
          xytext=(6, 8),
          textcoords="offset points",
          fontsize=7,
          fontweight="bold",
          color=color,
          transform=ccrs.PlateCarree(),
          bbox=common_bbox(),
          zorder=10,
        )

        legend_handles.append(
          Line2D(
            [0],
            [0],
            marker="X",
            color=color,
            markersize=9,
            linestyle="--",
            linewidth=1.5,
            label=f"T{track_id}  {dur_h:.0f} h",
            markeredgecolor="black",
            markeredgewidth=0.6
          )
        )

    # Duration colorbar
    sm_dur = plt.cm.ScalarMappable(cmap=dur_cmap, norm=dur_norm)
    sm_dur.set_array([])
    cbar2 = plt.colorbar(sm_dur, ax=ax, location="right", pad=0.02, shrink=0.5, aspect=18)
    cbar2.set_label("Track duration (h)\nred = short-lived, green = long-lived", fontsize=9)

    ax.coastlines(resolution="50m", linewidth=1.0, color="black", zorder=7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=":", color="gray", zorder=7)
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "ocean", "50m", facecolor="lightcyan", edgecolor="none"), alpha=0.4, zorder=2)
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.35, linewidth=0.5)
    gl.right_labels = False

    legend_handles += [
        Line2D(
          [0],
          [0],
          marker="D",
          color="gray",
          markersize=9,
          linestyle="none",
          label="Landfall",
          markeredgecolor="black"
        ),
        Line2D(
          [0],
          [0],
          marker="X",
          color="gray",
          markersize=10,
          linestyle="none",
          label="Dissipation",
          markeredgecolor="black"
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower left", **legend_style({"fontsize": 9}))
    ax.set_title("MD Dissipation Points on Terrain", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    diss_map_path = os.path.join(output_dir, "dissipation_terrain_map.png")
    plt.savefig(diss_map_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Dissipation terrain map", diss_map_path)
    return diss_map_path


def plot_bearing_change_at_terrain(track_df, track_id, output_dir):
    """
    Track bearing (direction of motion, °) over time, overlaid with elevation.

    A deflection or stall in bearing when crossing the 600 m terrain contour is
    evidence that the Eastern Ghats are steering or blocking the MD.

    Three panels:
      Top: elevation (land phase only)
      Mid: bearing (°, 0=N, 90=E, 270=W - NW-ward MDs sit around 270–315°)
      Bot: bearing change rate (°/h) - spikes = deflection events
    """
    track_id = int(track_id)
    required = {"time", "latitude", "longitude", "is_land"}
    if not required.issubset(track_df.columns):
        std_print("WARN", "ea", "plot_bearing_change_at_terrain missing cols",
                  str(required - set(track_df.columns)))
        return

    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])
    lats = np.radians(df["latitude"].values)
    lons = np.radians(df["longitude"].values)
    is_land = df["is_land"].values.astype(bool)

    # Forward bearing at each step (degrees, 0=N, clockwise)
    dlat = np.diff(lats, prepend=lats[0])
    dlon = np.diff(lons, prepend=lons[0])
    bearing = np.degrees(np.arctan2(dlon * np.cos(lats), dlat)) % 360

    dt_h = np.array([
        (times.iloc[i] - times.iloc[i-1]).total_seconds() / 3600.0 if i > 0 else np.nan
        for i in range(len(times))
    ])
    with np.errstate(invalid="ignore", divide="ignore"):
        bearing_rate = np.where(dt_h > 0, np.gradient(bearing) / dt_h, np.nan)

    elevation = df["elevation"].values.astype(float) \
        if "elevation" in df.columns else np.full(len(df), np.nan)
    land_elev = np.where(is_land, elevation, np.nan)

    # 600 m crossings
    crossings_600 = []
    if "elevation" in df.columns:
        elev_arr = elevation
        for i in range(1, len(df)):
            if is_land[i] and not np.isnan(elev_arr[i]) and not np.isnan(elev_arr[i-1]):
                if (elev_arr[i-1] < 600 <= elev_arr[i]) or (elev_arr[i-1] >= 600 > elev_arr[i]):
                    crossings_600.append(i)

    # Figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    bbox_style = common_bbox()

    # Panel 1: Elevation
    ax1.fill_between(times, 0, 1, where=is_land,transform=ax1.get_xaxis_transform(),color="wheat", alpha=0.35, zorder=0)
    ax1.fill_between(times, 0, land_elev, color="#8B6914", alpha=0.5,label="Elevation (m)", zorder=2)
    ax1.axhline(600, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,label="600 m")
    for ci in crossings_600:
        ax1.axvline(times.iloc[ci], color="purple", linestyle=":", linewidth=1.2,alpha=0.8)
    ax1.set_ylabel("Elevation (m)", fontsize=11, fontweight="bold", color="#5C3D0A")
    ax1.tick_params(axis="y", labelcolor="#5C3D0A")
    ax1.set_title(f"Track {track_id}: Bearing, Bearing-Change Rate & Terrain Crossing", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", **legend_style())
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Panel 2: Bearing
    land_bearing = np.where(is_land, bearing, np.nan)
    ax2.plot(times, land_bearing, "-o", color="steelblue", linewidth=2.0,markersize=3, alpha=0.85, label="Bearing (°)")
    ax2.axhline(315, color="darkblue", linestyle="--", linewidth=0.8, alpha=0.5,label="NW (315°) - typical MD track")
    ax2.axhline(270, color="navy", linestyle=":", linewidth=0.8, alpha=0.5,label="W  (270°)")
    for ci in crossings_600:
        ax2.axvline(times.iloc[ci], color="purple", linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("Bearing (°)", fontsize=11, fontweight="bold", color="steelblue")
    ax2.set_ylim(0, 360)
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.legend(loc="upper right", **legend_style())
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Panel 3: Bearing change rate
    land_rate = np.where(is_land, bearing_rate, np.nan)
    ax3.plot(times, land_rate, "-o", color="darkorange", linewidth=2.0,markersize=3, alpha=0.85, label="|dBearing/dt| (°/h)")
    ax3.axhline(0, color="black", linewidth=0.6)
    # Shade large deflections (|rate| > 15 °/h)
    ax3.fill_between(times, land_rate, 0,where=(np.abs(np.nan_to_num(land_rate)) > 15),color="darkorange", alpha=0.25, label="Deflection > 15°/h")
    for ci in crossings_600:
        ax3.axvline(times.iloc[ci], color="purple", linestyle=":", linewidth=1.2, alpha=0.8)
    ax3.set_ylabel("Bearing change (°/h)", fontsize=11, fontweight="bold",color="darkorange")
    ax3.set_xlabel("Time", fontsize=11, fontweight="bold")
    ax3.tick_params(axis="y", labelcolor="darkorange")
    ax3.legend(loc="upper right", **legend_style())
    ax3.grid(True, alpha=0.3, linestyle="--")
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    bearing_terrain_path = os.path.join(output_dir, f"track_{track_id}_bearing_terrain.png")
    plt.savefig(bearing_terrain_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Bearing change at terrain plot", bearing_terrain_path)

    # Export crossings table
    if crossings_600:
        rows = []
        for ci in crossings_600:
            rows.append({
                "track_id":      track_id,
                "time":          str(times.iloc[ci]),
                "elevation_m":   float(elevation[ci]) if not np.isnan(elevation[ci]) else np.nan,
                "bearing_deg":   float(bearing[ci]),
                "bearing_rate":  float(bearing_rate[ci]) if not np.isnan(bearing_rate[ci]) else np.nan,
                "latitude":      float(df["latitude"].iloc[ci]),
                "longitude":     float(df["longitude"].iloc[ci]),
            })
        terrain_crossings_CSV= os.path.join(output_dir, f"track_{track_id}_terrain_crossings.csv")
        pd.DataFrame(rows).to_csv(terrain_crossings_CSV, index=False)
        std_print("EXPORT", "ea", "Terrain crossing table", terrain_crossings_CSV)
    return bearing_terrain_path


def run_single_track_elevation_analysis(track_df, track_id, output_dir):
    """
    Per-track elevation analysis:
      1. Extract elevation profile
      2. Profile plot: elevation + speed vs time (orographic braking signal)
      3. Map plot: DEM background + track colored by speed (corridor vs barrier)

    Called inside the per-track loop in md_detection_tracking.py.
    Returns track_df enriched with 'elevation' column.
    """
    track_id = int(track_id)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract elevation
    if "speed_ms" not in track_df.columns:
        std_print("WARN", "ea", "speed_ms missing - run compute_track_speeds first")

    try:
        track_df = get_elevation_profile(track_df)
    except Exception as e:
        std_print("ERROR", "ea", "get_elevation_profile failed", str(e))
        return track_df

    valid = track_df["elevation"].notna().sum()
    if valid == 0:
        std_print("WARN", "ea","no elevation data - skipping per-track plots")
        return track_df

    # 2. Dual-axis profile: elevation + speed over time
    try:
        plot_track_elevation_profile(track_df, track_id, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Elevation profile plot failed", str(e))
        traceback.print_exc()

    # 3. Spatial map: DEM background + track colored by speed
    try:
        plot_track_elevation_speed_map(track_df, track_id, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "elevation map plot failed", str(e))
        traceback.print_exc()

    # 4. Barrier encounter annotation (terrain braking hypothesis)
    try:
        plot_barrier_encounter_profile(track_df, track_id, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Barrier encounter profile failed", str(e))
        traceback.print_exc()

    # 5. Bearing change at terrain crossings (deflection / stall hypothesis)
    try:
        plot_bearing_change_at_terrain(track_df, track_id, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Bearing change plot failed", str(e))
        traceback.print_exc()

    return track_df


def run_elevation_analysis(tracks_df, coupling_df, hgt_dir, output_dir):
    """
    1. Initialise terrain (if not already done)
    2. Extract elevation profiles for all tracks
    3. Compute TSI
    4. Orographic speed reduction (if speed_ms available)
    5. Elevation vs penetration scatter (if penetration_km available)
    6. Landfall elevation classification (if elevation_at_landfall available)
    7. Soil texture interpretation (if lat_landfall available)
    8. Export elevation_metrics.csv
    """
    if not tile_index:
      init_terrain(hgt_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Compute TSI for all tracks
    tsi_df, elevation_enriched_tracks_df = compute_tsi_for_all_tracks(tracks_df, coupling_df)
    if tsi_df.empty:
        std_print("ERROR", "ea", "No TSI results computed", "No elevation data")
        return pd.DataFrame()

    # Export results
    export_elevation_metrics(tsi_df, output_dir)
    plot_elevation_metrics(tsi_df, output_dir)

    # Orographic speed reduction
    try:
        compute_orographic_speed_reduction(elevation_enriched_tracks_df, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Orographic speed reduction failed", str(e))

    # Elevation vs penetration (only if column available)
    try:
        plot_elevation_vs_penetration(tsi_df, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Elevation vs penetration plot failed", str(e))

    # Landfall elevation classification
    try:
        classify_landfall_elevation(tsi_df, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Landfall elevation classification failed", str(e))

    # Soil texture
    try:
        plot_soil_texture_interpretation(tsi_df, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Soil texture plot failed", str(e))

    # Dissipation terrain map
    try:
        if elevation_enriched_tracks_df:
            plot_dissipation_terrain_map(elevation_enriched_tracks_df, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Dissipation terrain map failed", str(e))

    std_print("INFO", "ea", "Elevation analysis complete", output_dir)
    return tsi_df


# Helper functions
def soil_memory_timescale(beta, k_sat_ms):
    """
    Approximate SM memory timescale (days).
    Higher beta and higher k_sat -> water drains faster -> shorter memory.
    """
    k_sat_day = k_sat_ms * 86600  # m/day
    tau = 1.0 / (k_sat_day * beta + 1e-10)
    return round(tau, 1)


def lf_class(elev):
    if np.isnan(elev):
        return "Unknown"
    if elev < 50:
        return "Low coast (<50 m)"
    if elev < 600:
        return "Elevated (50-600 m)"
    return "Terrain-facing (>600 m)"


# HGT I/O helpers
def build_tile_index():
    """Walk _hgt_dir and populate _tile_index {canonical_name → full_path}."""
    pattern = re.compile(r'[NS]\d{2}[EW]\d{3}\.hgt', re.IGNORECASE)
    for root, dirs, files in os.walk(hgt_dir):
        for fname in files:
            full_path = os.path.join(root, fname)
            # Direct match: file is exactly "N24E087.hgt"
            if pattern.fullmatch(fname):
                tile_index[fname.upper()] = full_path
            else:
                # Embedded match: file is something like "H44N24E087.hgt"
                m = pattern.search(fname)
                if m:
                    tile_index[m.group().upper()] = full_path


def get_tile_filename(lat, lon):
    """Return full path to HGT tile for (lat, lon), or None if not indexed/available"""
    lat_int = int(np.floor(lat))
    lon_int = int(np.floor(lon))
    lat_str = f"{'N' if lat_int >= 0 else 'S'}{abs(lat_int):02d}"
    lon_str = f"{'E' if lon_int >= 0 else 'W'}{abs(lon_int):03d}"
    key = f"{lat_str}{lon_str}.HGT"   # index keys are upper-cased
    return tile_index.get(key)        # None if tile not available


def load_hgt_tile(tile_path):
    """Load an HGT tile from disk (or return cached copy)"""
    # Check cache first
    if tile_path in tiles_cache:
        return tiles_cache[tile_path]
    if not os.path.exists(tile_path):
        std_print("ERROR", "ea", "HGT tile not found", tile_path)
        return None
    try:
        with rasterio.open(tile_path) as src:
            # Read elevation data (SRTM uses 1 band)
            elevation = src.read(1).astype(np.float32)
            # Handle nodata values (SRTM uses -32768)
            nodata = src.nodata
            if nodata is not None:
                elevation[elevation == nodata] = np.nan
            tile_data = {
                "data":      elevation,
                "transform": src.transform,
                # 'crs': src.crs,
                # 'bounds': src.bounds,
                "shape":     elevation.shape,
            }
            tiles_cache[tile_path] = tile_data
            # elev_min = np.nanmin(elevation)
            # elev_max = np.nanmax(elevation)
            # valid_pct = 100 * (~np.isnan(elevation)).sum() / elevation.size
            # std_print("DEBUG", "ea", f"Loaded {os.path.basename(tile_path)}")
            # std_print("DEBUG", "ea", f"Elevation: [{elev_min:.0f}, {elev_max:.0f}] m")
            # std_print("DEBUG", "ea", f"Valid: {valid_pct:.1f}%")
            return tile_data
    except Exception as e:
        std_print("ERROR", "ea", f"Failed to load {tile_path}", str(e))
        return None


def get_elevation_at_point(lat, lon):
    """Return elevation (m) at a single lat/lon point, or None if tile unavailable."""
    tile_path = get_tile_filename(lat, lon)
    # if tile_name is None: # tile not available
    #     return None
    tile_data = load_hgt_tile(tile_path)
    if tile_data is None: # tile found but failed to read
        return None
    try:
        row, col = rasterio.transform.rowcol(tile_data["transform"], lon, lat)
        row, col = int(round(row)), int(round(col))
        h, w = tile_data["shape"]
        if 0 <= row < h and 0 <= col < w:
            v = tile_data["data"][row, col]
            return float(v) if not np.isnan(v) else None
    except Exception:
        pass
    return None


def get_elevation_profile(track_df):
    """
    Extract elevation profile along MD track, batched per tile.
    """
    track_df = track_df.copy().reset_index(drop=True)
    lats = track_df["latitude"].values
    lons = track_df["longitude"].values
    elevations = np.full(len(track_df), np.nan)

    # Classify each point as land or ocean once
    if "is_land" in track_df.columns:
        land_mask = track_df["is_land"].values.astype(bool)
    else:
        land_mask = is_land(lats, lons)

    # Group land point indices by (lat_int, lon_int) tile key for batched loading
    tile_groups: dict = {}
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if not land_mask[i]:
            continue   # ocean point - skip
        key = (int(np.floor(lat)), int(np.floor(lon)))
        tile_groups.setdefault(key, []).append(i)

    missing_land_tiles = []
    for (lat_int, lon_int), indices in tile_groups.items():
        tile_path = get_tile_filename(lat_int + 0.5, lon_int + 0.5)
        if tile_path is None:
            lat_str = f"{'N' if lat_int >= 0 else 'S'}{abs(lat_int):02d}"
            lon_str = f"{'E' if lon_int >= 0 else 'W'}{abs(lon_int):03d}"
            missing_land_tiles.append(f"{lat_str}{lon_str}")
            continue
        tile_data = load_hgt_tile(tile_path)
        if tile_data is None:
            continue
        for i in indices:
            try:
                row, col = rasterio.transform.rowcol(tile_data["transform"], lons[i], lats[i])
                row, col = int(round(row)), int(round(col))
                h, w = tile_data["shape"]
                if 0 <= row < h and 0 <= col < w:
                    v = tile_data["data"][row, col]
                    if not np.isnan(v):
                        elevations[i] = float(v)
            except Exception:
                pass

    if missing_land_tiles:
        std_print("WARN", "ea",f"Missing land HGT tiles ({len(missing_land_tiles)})", \
                  ",".join(sorted(missing_land_tiles)))

    track_df["elevation"] = elevations
    land_count = int(np.sum(land_mask))
    valid_count = int(np.sum(~np.isnan(elevations)))
    ocean_count = len(track_df) - land_count
    std_print("DEBUG", "ea", "Elevation profile",
              f"{valid_count}/{land_count} land points valid | "
              f"{ocean_count} ocean points (NaN, expected)")

    if valid_count == 0 and land_count > 0:
        std_print("WARN", "ea", "No elevation data for land points - missing tiles")
    elif valid_count == 0 and land_count == 0:
        std_print("WARN", "ea", "Track is entirely over ocean - no elevation data expected")
    return track_df


def calculate_slope(tile_name, cell_size=30.0):
    """Slope magnitude (degrees) via Sobel operator."""
    tile_data = load_hgt_tile(tile_name)
    if tile_data is None:
        return None
    elevation = tile_data["data"].copy()

    # Interpolate NaN values for gradient calculation
    nan_mask = np.isnan(elevation)
    if np.all(nan_mask):
        return None

    # Simple interpolation: replace NaN with local mean
    if nan_mask.any():
        valid = elevation[~nan_mask]
        elevation[nan_mask] = np.nanmean(valid)

    # Sobel operator for gradients
    dz_dx = sobel(elevation, axis=1) / (8 * cell_size)  # East-West
    dz_dy = sobel(elevation, axis=0) / (8 * cell_size)  # North-South

    # Slope magnitude in degrees
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope = np.degrees(slope_rad)

    # Cap unrealistic values
    # slope = np.clip(slope, 0, 90)
    return slope


def get_terrain_metrics_at_point(lat, lon, window_size=3):
    """Get terrain metrics at a point (lat, lon) with local neighborhood"""
    tile_name = get_tile_filename(lat, lon)
    if tile_name is None:
        return None
    tile_data = load_hgt_tile(tile_name)
    if tile_data is None:
        return None

    # Get pixel coordinates
    transform = tile_data["transform"]
    col, row = ~transform * (lon, lat)
    row, col = int(round(row)), int(round(col))

    # Check bounds
    shape = tile_data["shape"]
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        return None
    elevation = tile_data["data"]
    point_elev = elevation[row, col]
    if np.isnan(point_elev):
        return None
    # Local neighborhood
    half_win = window_size // 2
    r_min = max(0, row - half_win)
    r_max = min(shape[0], row + half_win + 1)
    c_min = max(0, col - half_win)
    c_max = min(shape[1], col + half_win + 1)
    neighborhood = elevation[r_min:r_max, c_min:c_max]
    valid_neighborhood = neighborhood[~np.isnan(neighborhood)]

    if len(valid_neighborhood) == 0:
        # No valid neighbors, use point elevation only
        return {
            "elevation": float(point_elev),
            "slope": 0.0,
            "aspect": -1.0,
            "roughness": 0.0,
            "elevation_std": 0.0,
            "n_valid_neighbors": 0,
        }

    # Aspect needs bivariate gradient
    if neighborhood.shape[0] >= 3 and neighborhood.shape[1] >= 3:
        # Use central differences
        gy, gx = np.gradient(neighborhood, 1, 1)  # spacing assumed 1 pixel
        aspect_rad = np.arctan2(-gx, gy)  # appropriate formula
        aspect_deg = np.degrees(aspect_rad).mean()
        aspect_deg = (aspect_deg + 360) % 360
    else:
        aspect_deg = -1.0
    std_print("DEBUG", "ea", "Using aspect_deg", aspect_deg)

    # Metrics
    metrics = {
        "elevation": float(point_elev),
        "slope": float(np.nanmean(np.abs(np.diff(valid_neighborhood)))),
        "roughness": float(np.nanstd(valid_neighborhood)),
        "elevation_std": float(np.nanstd(valid_neighborhood)),
        "n_valid_neighbors": len(valid_neighborhood),
        "aspect": float(aspect_deg),
    }
    return metrics


# TSI computation
def compute_tsi(track_df, track_coupling=None):
    """Compute Topographic Sensitivity Index (TSI) for a track from land-phase points only.
    TSI = elevation_effect x slope_effect x roughness_effect
    """
    # Use land points if the column is available
    if "is_land" in track_df.columns:
        land_df = track_df[track_df["is_land"] == True]
    else:
        land_df = track_df  # NaN filtering below

    # Extract valid elevations
    valid_elev = track_df["elevation"].dropna()
    tsi_result = {
        "TSI": np.nan,
        "TSI_class": "insufficient_data",
        "elevation_effect": np.nan,
        "slope_effect": np.nan,
        "roughness_effect": np.nan,
        "mean_elevation": np.nan,
        "elevation_range": np.nan,
        "mean_slope": np.nan,
        "elevation_std": float(valid_elev.std()),
        "n_valid_points": len(valid_elev),
        "n_land_points": len(land_df), 
    }
    if len(valid_elev) < 3:
        return tsi_result

    # 1. Elevation effect
    elev_range = valid_elev.max() - valid_elev.min()
    # INFO: total elevation change along a track rarely exceeds 3000 m
    # TODO: would it be better to use max_elevation along track instead of hardcoded 3000m?
    elev_eff = min(1.0, elev_range / 3000.0)

    # 2. Slope effect
    mean_slope = np.abs(valid_elev.diff().dropna()).mean()
    # INFO: 30m per step over ~0.25° grid
    slope_eff = min(1.0, mean_slope / 30.0)

    # 3. Roughness effect
    # INFO: 500 m represents a high degree of roughness, typical of the most complex topography in the region.
    roughness_eff = min(1.0, valid_elev.std() / 500.0)

    # Compute TSI
    tsi = elev_eff * slope_eff * roughness_eff
    # Classify TSI
    if tsi >= 0.05:
        tsi_class = "High (terrain-sensitive)"
    elif tsi >= 0.02:
        tsi_class = "Moderate"
    elif tsi >= 0.005:
        tsi_class = "Low (terrain-insensitive)"
    else:
        tsi_class = "Very low (flat)"

    result = {
        "TSI": float(tsi),
        "TSI_class": tsi_class,
        "elevation_effect": float(elev_eff),
        "slope_effect": float(slope_eff),
        "roughness_effect": float(roughness_eff),
        "mean_elevation": float(valid_elev.mean()),
        "elevation_range": float(elev_range),
        "mean_slope": float(mean_slope),
        "n_valid_points": len(valid_elev),
        "n_land_points": len(land_df),
    }
    # Coupling metrics if available
    if track_coupling is not None:
        if "FSI" in track_coupling:
            result["FSI"] = track_coupling["FSI"]
        if "duration_h" in track_coupling:
            result["duration_h"] = track_coupling["duration_h"]
    return result


def compute_tsi_for_all_tracks(tracks_df, coupling_df):
    """Compute TSI for all tracks"""
    if isinstance(tracks_df, pd.DataFrame):
        if "track_id" not in tracks_df.columns:
            std_print("ERROR", "ea", "tracks_df has no track_id column")
            return pd.DataFrame(), {}
        tracks_df = {
            int(track_id): tracks_df[tracks_df["track_id"] == track_id].copy()
            for track_id in tracks_df["track_id"].unique()
        }

    all_results = []
    elevation_enriched_tracks_df = {}
    for track_id, track_df in tracks_df.items():
        # Ensure track_df
        if isinstance(track_df, pd.Series):
            continue
        try:
            # Get elevation profile
            track_with_elev = get_elevation_profile(track_df)

            # Coupling metrics if available
            track_coupling = None
            if coupling_df is not None and not coupling_df.empty:
                if "track_id" in coupling_df.columns:
                    mask = coupling_df["track_id"] == track_id
                    if mask.any():
                        track_coupling = coupling_df[mask].iloc[0].to_dict()
                elif track_id in coupling_df.index:
                    track_coupling = coupling_df.loc[track_id].to_dict()

            # Compute TSI
            tsi_result = compute_tsi(track_with_elev, track_coupling)
            tsi_result["track_id"] = track_id

             # TODO:
            # tsi_result["speed_anomaly"] = compute_speed_anomaly(track_with_elev)

            # Calculate duration if available
            if "time" in track_df.columns:
                times = pd.to_datetime(track_df["time"])
                duration_h = (times.max() - times.min()).total_seconds() / 3600
                tsi_result["duration_h"] = duration_h
            all_results.append(tsi_result)
            std_print("DEBUG", "ea", "TSI", f"{tsi_result['TSI']:.3f} ({tsi_result['TSI_class']})")
        except Exception as e:
            std_print("ERROR", "ea", "Computing TSI Failed", str(e))
            traceback.print_exc()
            continue

        try:
            landfall_idx, _, is_valid = get_landfall_index(track_with_elev)
            if is_valid and landfall_idx >= 0:
                lf = track_with_elev.iloc[landfall_idx]
                end = track_with_elev.iloc[-1]
                tsi_result["lat_landfall"] = float(lf["latitude"])
                tsi_result["lon_landfall"] = float(lf["longitude"])
                tsi_result["elevation_at_landfall"] = (
                  float(lf["elevation"]) if pd.notna(lf.get("elevation", np.nan)) else np.nan
                )
                tsi_result["penetration_km"] = float(haversine_distance(
                    lf["longitude"],
                    lf["latitude"],
                    end["longitude"],
                    end["latitude"]
                ))
            else:
                for col in ("lat_landfall", "lon_landfall","elevation_at_landfall", "penetration_km"):
                    tsi_result[col] = np.nan
        except Exception as lf_err:
            std_print("ERROR", "ea", "landfall columns failed", str(lf_err))
            for col in ("lat_landfall", "lon_landfall","elevation_at_landfall", "penetration_km"):
                tsi_result[col] = np.nan
        elevation_enriched_tracks_df[track_id] = track_with_elev
    return (pd.DataFrame(all_results) if all_results else pd.DataFrame()), elevation_enriched_tracks_df


# Export
def export_elevation_metrics(tsi_df, output_dir):
    if tsi_df.empty:
        std_print("WARN", "ea", "No TSI results to export")
        return
    os.makedirs(output_dir, exist_ok=True)
    # Save CSV
    # FIX: use export_to_csv()
    csv_path = os.path.join(output_dir, "elevation_metrics.csv")
    tsi_df.to_csv(csv_path, index=False)
    std_print("SAVED", "ea", "Elevation metrics", csv_path)
    # Save summary report
    report_path = os.path.join(output_dir, "elevation_summary.txt")
    with open(report_path, "w") as f:
        f.write("Elevation & Terrain analysis summary\n\n")
        f.write(f"Total tracks analyzed: {len(tsi_df)}\n")
        f.write(f"Valid elevation data: {(tsi_df['n_valid_points'] > 0).sum()}\n\n")

        # valid_tsi = tsi_df[tsi_df["TSI"].notna()]["TSI"]
        valid_tsi = tsi_df["TSI"].dropna()
        if len(valid_tsi):
            f.write("Topographic Sensitivity Index (TSI) statistics\n\n")
            f.write(f"  Mean TSI: {valid_tsi.mean():.3f}\n")
            f.write(f"  Std Dev:  {valid_tsi.std():.3f}\n")
            # f.write(f"  Min TSI:  {valid_tsi.min():.3f}\n")
            # f.write(f"  Max TSI:  {valid_tsi.max():.3f}\n\n")
            f.write(f"  Range: [{valid_tsi.min():.3f}, {valid_tsi.max():.3f}]\n\n")

            f.write("TSI Classification Distribution:\n")
            class_dist = tsi_df["TSI_class"].value_counts()
            for cls, count in class_dist.items():
                pct = 100 * count / len(tsi_df)
                f.write(f"  {cls}: {count} tracks ({pct:.1f}%)\n")
            f.write("\n")

        # Component effects
        f.write("Terrain component effects\n\n")
        if "elevation_effect" in tsi_df.columns:
            f.write(
                f"  Elevation effect (mean): {tsi_df['elevation_effect'].mean():.3f}\n"
            )
        if "slope_effect" in tsi_df.columns:
            f.write(
                f"  Slope effect (mean):     {tsi_df['slope_effect'].mean():.3f}\n"
            )
        if "roughness_effect" in tsi_df.columns:
            f.write(
                f"  Roughness effect (mean): {tsi_df['roughness_effect'].mean():.3f}\n"
            )
        f.write("\n")

        # Elevation statistics
        f.write("Elevation statistics\n\n")
        if "mean_elevation" in tsi_df.columns:
            f.write(
                f"  Mean track elevation: {tsi_df['mean_elevation'].mean():.1f} m\n"
            )
        if "max_elevation" in tsi_df.columns:
            f.write(
                f"  Max track elevation:  {tsi_df['max_elevation'].max():.1f} m\n"
            )
        if "elevation_range" in tsi_df.columns:
            f.write(
                f"  Mean elevation range: {tsi_df['elevation_range'].mean():.1f} m\n"
            )
        if "elevation_std" in tsi_df.columns:
            f.write(
                f"  Mean elevation std:   {tsi_df['elevation_std'].mean():.1f} m\n"
            )
        f.write("\n")

        # TSI vs duration correlation
        if "duration_h" in tsi_df.columns:
            sub = tsi_df.dropna(subset=["TSI", "duration_h"])
            if len(sub) >= 5:
                r, p_raw, p_corr, n_eff = pearsonr_corrected(
                    sub["TSI"].values, sub["duration_h"].values
                )
                sig = (
                    "significant"
                    if (not np.isnan(p_corr) and p_corr < 0.05)
                    else "not significant"
                )
                f.write("\n")

                f.write("TSI vs Duration\n")
                f.write(f"  r = {r:.3f}, p_corr = {p_corr:.4e} [{sig}], n_eff = {n_eff}")
    std_print("SAVED", "ea", "Elevation summary report", report_path)


def load_elevation_metrics(output_dir):
    """Load TSI metrics computed from HGT elevation data for statistical analysis"""
    path = os.path.join(output_dir, "Global_elevation_analysis", "elevation_metrics.csv")
    try:
        df = pd.read_csv(path)
        std_print("INFO", "ea", "Loaded elevation metrics", path)
        return df
    except Exception as e:
        std_print("ERROR", "ea", f"Failed to load {path}", str(e))

    std_print("WARN", "ea", "Elevation metrics not found", "TSI analysis will be skipped")
    return None
