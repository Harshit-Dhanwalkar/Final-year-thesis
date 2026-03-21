"""
Antecedent Soil Moisture Analysis for Monsoon Depressions.

Implements the Kishtawal et al. (2013a) methodology:
  - ASMI computed over the future inland track corridor
  - Exponential temporal decay: recent SM weighted more than older SM
  - Multiple lag windows before landfall (24 h -> 2 weeks)
  - Wet / Dry track classification and comparative statistics

Dry antecedent conditions -> Soil can absorb a lot of the incoming rain -> lower immediate flood risk (more infiltration) but also lower LHF -> less Brown Ocean feedback -> faster storm decay inland.
Wet antecedent conditions -> Soil is already near saturation -> much more of the MD's heavy rainfall becomes surface runoff -> higher flood potential even if the total rainfall amount is not extreme but also sustained LHF to storm -> longer inland persistence (Brown Ocean Effect).

References :
Kishtawal et al. (2013a): Water Resour. Res. 49, 3573-3589.
Andersen & Shepherd   (2013): J. Climate 26, 6629-6642.
"""

# FIX:
# 1. Currently computes a single spatial mean of SWVL1 in the 24h before system formation. This is not antecedent soil moisture analysis -> it's a snapshot
# 2. The future_track_bbox function extracts the bounding box of the inland portion (after landfall). This is correct for measuring pre-storm soil state along the future track. However, compute_asmi_multi_lag is called with ref_time = landfall_time if available else times.min(). For ocean-originated tracks with a clear landfall, this is correct. But for land-originated tracks (no landfall), it uses times.min() - the very start of the track - which means the 24h window extends before the track even exists, potentially reading data from an ocean region. Add a guard: if origin is land_originated and no landfall, use times.min() + timedelta(hours=6) as ref_time so at least the early inland points are covered.


# TODO:
# Wet/Dry classification and comparison
# # Classify tracks by antecedent conditions
# median_asmi = asmi_values.median()
# wet_tracks  = results[results['asmi'] > median_asmi]
# dry_tracks  = results[results['asmi'] <= median_asmi]
#
# - Compare:
#   - inland penetration distance (km)
#   - duration after landfall (hours)
#   - speed change at landfall (km/h)
#   - mean LHF during land phase (W/m²)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr, mannwhitneyu, ttest_ind
from cartopy.mpl.geoaxes import GeoAxes
import traceback

from utils import (
    std_print,
    find_origin_type,
    get_time_coord,
    common_bbox,
    legend_style,
    clamp_to_domain,
    calculate_buffer_deg,
    pearsonr_corrected,
)

# FIX: use this constants from main script
CLIPPED_MIN_LAT = 10.0
CLIPPED_MAX_LAT = 30.0
CLIPPED_MIN_LON = 70.0
CLIPPED_MAX_LON = 95.0


def compute_asmi(ds, track_df, landfall_time, lag_days=7, decay_k=0.85, buffer_km=100):
    """
    Antecedent Soil Moisture Index (ASMI).

    Exponentially weighted average of SWVL1 extracted over the future inland
    track corridor* for `lag_days` before `landfall_time`
    """
    landfall_time = pd.to_datetime(landfall_time)
    lat_min, lat_max, lon_min, lon_max = future_track_bbox(track_df, buffer_km)

    weights = []
    sm_vals = []
    for d in range(lag_days, 0, -1):
        t = landfall_time - pd.Timedelta(days=d)
        w = decay_k**d
        sm = sm_at_time(ds, t, lat_min, lat_max, lon_min, lon_max)
        if not np.isnan(sm):
            weights.append(w)
            sm_vals.append(sm)

    if not weights:
        return np.nan
    return float(np.average(sm_vals, weights=weights))


def compute_asmi_multi_lag(ds, track_df, landfall_time, buffer_km=100):
    """
    Compute ASMI at multiple antecedent windows:
      24 h, 48 h, 72 h, 1 week (7 days), 2 weeks (14 days)
    """
    windows = {
        "asmi_24h": (1, 0.85),
        "asmi_48h": (2, 0.85),
        "asmi_72h": (3, 0.85),
        "asmi_7d": (7, 0.85),
        "asmi_14d": (14, 0.90),  # slower decay for longer window
    }
    results = {}
    for label, (days, k) in windows.items():
        results[label] = compute_asmi(
            ds, track_df, landfall_time, lag_days=days, decay_k=k, buffer_km=buffer_km
        )
    return results


# # Single-track spatial map (antecedent snapshot)
# FIX: tracks are out of plot
# def plot_antecedent_spatial_map(ds, track_df, track_id, hours_before, output_dir):
#     """
#     Spatial map of SWVL1 at `hours_before` before landfall (or system start).
#     Shows future MD track overlaid so the corridor-based logic is visible.
#     """
#     track_id = int(track_id)
#     track = (
#         track_df[track_df["track_id"] == track_id].sort_values("time")
#         if "track_id" in track_df.columns
#         else track_df.sort_values("time")
#     )
#     if track.empty:
#         std_print("ERROR", "as", f"Track {track_id} not found")
#         return False
#
#     # Reference time: landfall if available, else formation
#     if "is_land" in track.columns:
#         land_rows = track[track["is_land"] == True]
#         ref_time = (
#             pd.to_datetime(land_rows.iloc[0]["time"])
#             if not land_rows.empty
#             else pd.to_datetime(track.iloc[0]["time"])
#         )
#     else:
#         ref_time = pd.to_datetime(track.iloc[0]["time"])
#
#     snap_time = ref_time - pd.Timedelta(hours=hours_before)
#
#     # Map extent
#     lat_center = track["latitude"].mean()
#     lon_center = track["longitude"].mean()
#     ext_buf = max(5.0, (track["latitude"].max() - track["latitude"].min()) * 1.5)
#     min_lat = max(CLIPPED_MIN_LAT, lat_center - ext_buf / 2)
#     max_lat = min(CLIPPED_MAX_LAT, lat_center + ext_buf / 2)
#     min_lon = max(CLIPPED_MIN_LON, lon_center - ext_buf / 2)
#     max_lon = min(CLIPPED_MAX_LON, lon_center + ext_buf / 2)
#
#     ds, time_coord = get_time_coord(ds)
#     try:
#         ds_t = ds.sel({time_coord: snap_time}, method="nearest")
#         sm_data = ds_t["swvl1"].sel(
#             latitude=slice(max_lat, min_lat),
#             longitude=slice(min_lon, max_lon),
#         )
#         if sm_data.size == 0:
#             std_print("WARN", "as", f"Track {track_id}", "No SM data at snapshot time")
#             return False
#     except Exception as e:
#         std_print(
#             "ERROR", "as", f"Track {track_id}: snapshot extraction failed", str(e)
#         )
#         return False
#
#     # Figure
#     fig = plt.figure(figsize=(12, 9))
#     ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
#
#     ax.coastlines(resolution="50m", linewidth=1.0)
#     ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
#     ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=6)
#     ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle="--")
#     gl = ax.gridlines(
#         crs=ccrs.PlateCarree(),
#         draw_labels=True,
#         linewidth=0.5,
#         color="gray",
#         alpha=0.5,
#         linestyle="--",
#     )
#     gl.top_labels = False
#     gl.right_labels = False
#
#     im = ax.contourf(
#         sm_data.longitude,
#         sm_data.latitude,
#         sm_data.values,
#         levels=21,
#         cmap="YlGnBu",
#         transform=ccrs.PlateCarree(),
#         extend="both",
#     )
#     ax.contour(
#         sm_data.longitude,
#         sm_data.latitude,
#         sm_data.values,
#         levels=10,
#         colors="black",
#         linewidths=0.5,
#         alpha=0.4,
#         transform=ccrs.PlateCarree(),
#     )
#
#     # Future track
#     ax.plot(
#         track["longitude"],
#         track["latitude"],
#         "r-",
#         linewidth=2.5,
#         transform=ccrs.PlateCarree(),
#         label="Future MD track",
#         zorder=5,
#     )
#     ax.scatter(
#         track.iloc[0]["longitude"],
#         track.iloc[0]["latitude"],
#         s=250,
#         c="gold",
#         marker="s",
#         edgecolors="black",
#         linewidth=1.5,
#         transform=ccrs.PlateCarree(),
#         label="Formation point",
#         zorder=6,
#     )
#
#     ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
#     cbar = plt.colorbar(im, ax=ax, pad=0.02)
#     cbar.set_label("Volumatric Soil Water Layer (0-7cm) (m³/m³)", fontsize=10)
#
#     legend_handles, _ = ax.get_legend_handles_labels()
#     if legend_handles:
#         ax.legend(handles=legend_handles, loc="upper left", **legend_style())
#
#     ax.set_title(
#         f"Track {track_id}: Antecedent Soil Moisture  ({hours_before}h before landfall/formation)\n"
#         f"Snapshot: {snap_time.strftime('%d %b %Y %H:%M UTC')}",
#         fontsize=12,
#         fontweight="bold",
#     )
#     plt.tight_layout()
#     os.makedirs(output_dir, exist_ok=True)
#     outpath = os.path.join(
#         output_dir, f"track_{track_id}_antecedent_spatial_{hours_before}h.png"
#     )
#     plt.savefig(outpath, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     std_print("SAVED", "as", "Antecedent spatial map", outpath)
#     return True
#
#
# # Antecedent trend (48 h before formation/landfall)
# def analyze_antecedent_trend(ds, track_df, track_id, output_dir):
#     """SM time-series in the 48 h before landfall/formation over the future corridor."""
#     track_id = int(track_id)
#     track = (
#         track_df[track_df["track_id"] == track_id].sort_values("time")
#         if "track_id" in track_df.columns
#         else track_df.sort_values("time")
#     )
#     if track.empty:
#         return None
#
#     if "is_land" in track.columns:
#         land_rows = track[track["is_land"] == True]
#         ref_time = (
#             pd.to_datetime(land_rows.iloc[0]["time"])
#             if not land_rows.empty
#             else pd.to_datetime(track.iloc[0]["time"])
#         )
#     else:
#         ref_time = pd.to_datetime(track.iloc[0]["time"])
#
#     lat_min, lat_max, lon_min, lon_max = future_track_bbox(track, buffer_km=100)
#
#     # Sample every 6 h over [-48h, 0h]
#     hours_before_list = list(range(48, -1, -6))
#     sm_vals = []
#     times = []
#     for h in hours_before_list:
#         t = ref_time - pd.Timedelta(hours=h)
#         sm = sm_at_time(ds, t, lat_min, lat_max, lon_min, lon_max)
#         if not np.isnan(sm):
#             sm_vals.append(sm)
#             times.append(h)
#
#     if len(sm_vals) < 3:
#         std_print("WARN", "as", f"Track {track_id}: insufficient antecedent data")
#         return None
#
#     # Figure
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(times, sm_vals, "o-", linewidth=2.5, markersize=6, color="darkgreen")
#     ax.fill_between(times, sm_vals, alpha=0.2, color="lightgreen")
#
#     # Trend line
#     z = np.polyfit(times, sm_vals, 1)
#     p_fit = np.poly1d(z)
#     ax.plot(
#         times,
#         p_fit(times),
#         "r--",
#         linewidth=1.5,
#         alpha=0.7,
#         label=f"Trend: {z[0]:.5f} m³/m³ per hour",
#     )
#
#     trend_label = (
#         "Decreasing toward landfall"
#         if z[0] > 0.001
#         else "Increasing toward landfall" if z[0] < -0.001 else "Stable"
#     )
#     ax.text(
#         1.02,
#         0.98,
#         f"{trend_label} SM\nbefore landfall",
#         transform=ax.transAxes,
#         fontsize=12,
#         va="top",
#         bbox=common_bbox(),
#     )
#
#     ax.set_xlabel("Hours before landfall/formation", fontsize=12, fontweight="bold")
#     ax.set_ylabel("SWVL1 (m³/m³)", fontsize=12, fontweight="bold")
#     ax.set_title(
#         f"Track {track_id}: Antecedent SM Trend over Future Inland Corridor\n(48 h window)",
#         fontsize=12,
#         fontweight="bold",
#         pad=15,
#     )
#     ax.grid(True, alpha=0.3, linestyle="--")
#     ax.invert_xaxis()
#     legend_handles, _ = ax.get_legend_handles_labels()
#     if legend_handles:
#         ax.legend(handles=legend_handles, **legend_style())
#     plt.tight_layout()
#
#     os.makedirs(output_dir, exist_ok=True)
#     outpath = os.path.join(output_dir, f"track_{track_id}_antecedent_trend.png")
#     plt.savefig(outpath, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     std_print("SAVED", "as", "Antecedent trend plot", outpath)
#     return sm_vals


# Multi-lag ASMI vs penetration scatter
def plot_asmi_vs_penetration(results_df, output_dir):
    """
    Scatter plot: ASMI (various lags) vs inland penetration distance.
    One panel per lag window; regression line + corrected r shown.
    """
    lag_cols = ["asmi_24h", "asmi_48h", "asmi_72h", "asmi_7d", "asmi_14d"]
    lag_cols = [c for c in lag_cols if c in results_df.columns]
    if "penetration_km" not in results_df.columns or not lag_cols:
        std_print("WARN", "as", "plot_asmi_vs_penetration: missing columns")
        return

    # Figure
    n = len(lag_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, lag_cols):
        sub = results_df.dropna(subset=[col, "penetration_km"])
        if len(sub) < 3:
            continue
        x = sub[col].values
        y = sub["penetration_km"].values
        r, p_raw, p_corr, n_eff = pearsonr_corrected(x, y)
        ax.scatter(
            x,
            y,
            s=80,
            alpha=0.75,
            edgecolors="black",
            linewidth=0.7,
            c="steelblue",
            zorder=3,
        )
        if len(x) >= 3:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, np.poly1d(z)(x_line), "r--", lw=1.5, alpha=0.7)
        sig = "**" if (not np.isnan(p_corr) and p_corr < 0.05) else ""
        p_str = f"{p_corr:.4f}" if not np.isnan(p_corr) else "N/A"
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}\np_corr = {p_str} {sig}\nn_eff = {n_eff}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=common_bbox(),
            family="monospace",
        )
        lag_label = col.replace("asmi_", "").upper()
        ax.set_xlabel(f"ASMI {lag_label} (m³/m³)", fontsize=12, fontweight="bold")
        ax.set_title(f"ASMI {lag_label}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Inland penetration (km)", fontsize=12, fontweight="bold")
    fig.suptitle(
        "Antecedent SM vs Inland Penetration Distance", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "asmi_vs_penetration.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "as", "ASMI vs penetration scatter", outpath)


def plot_asmi_multi_lag_summary(results_df, output_dir):
    """ASMI at each lag window - shows how soil moisture memory decays as the antecedent window widens"""
    lag_cols = ["asmi_24h", "asmi_48h", "asmi_72h", "asmi_7d", "asmi_14d"]
    lag_cols = [c for c in lag_cols if c in results_df.columns]
    if not lag_cols:
        return

    lag_hours = {
        "asmi_24h": 24,
        "asmi_48h": 48,
        "asmi_72h": 72,
        "asmi_7d": 168,
        "asmi_14d": 336,
    }
    x_vals = [lag_hours[c] for c in lag_cols]
    labels = [c.replace("asmi_", "").upper() for c in lag_cols]

    # Figure
    fig, ax = plt.subplots(figsize=(11, 6))
    track_colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))

    # Individual tracks as faint lines + dots
    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row.get(c, np.nan) for c in lag_cols]
        track_id = row.get("track_id", i)
        ax.plot(
            x_vals,
            vals,
            "-o",
            color=track_colors[i],
            linewidth=1.2,
            markersize=6,
            alpha=0.55,
            label=f"T{track_id}",
        )

    # Mean line on top
    means = [results_df[c].mean() for c in lag_cols]
    ax.plot(
        x_vals,
        means,
        "-o",
        color="black",
        linewidth=2.5,
        markersize=9,
        zorder=5,
        label="Mean (all tracks)",
    )

    ax.axhline(
        0.30,
        color="darkgreen",
        linestyle="--",
        lw=1.5,
        alpha=0.8,
        label="Field capacity (~0.30 m³/m³)",
    )
    ax.axhline(
        0.45,
        color="saddlebrown",
        linestyle="--",
        lw=1.5,
        alpha=0.8,
        label="Saturation (~0.45 m³/m³)",
    )

    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_xlabel("Antecedent window", fontsize=12, fontweight="bold")
    ax.set_ylabel("ASMI (m³/m³)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Antecedent SM across lag windows - soil memory decay\n"
        "(dots = individual tracks; line = mean)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    legend_handles, _ = ax.get_legend_handles_labels()
    if legend_handles:
        ax.legend(handles=legend_handles, **legend_style())
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "asmi_multi_lag_summary.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "as", "ASMI multi-lag summary", outpath)


# Scatter: antecedent SM vs duration
def plot_antecedent_vs_duration(results_df, output_dir):
    """Scatter: ASMI_7d vs total track duration, coloured by origin type"""
    if (
        "asmi_7d" not in results_df.columns
        or "duration_hours" not in results_df.columns
    ):
        std_print("WARN", "as", "plot_antecedent_vs_duration: missing columns")
        return

    sub = results_df.dropna(subset=["asmi_7d", "duration_hours"])
    if len(sub) < 3:
        std_print("WARN", "as", "Insufficient data for ASMI-duration scatter")
        return

    # Figure
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"ocean_originated": "steelblue", "land_originated": "darkorange"}
    for origin, grp in (
        sub.groupby("origin_type") if "origin_type" in sub.columns else [("all", sub)]
    ):
        c = colors.get(origin, "gray")
        ax.scatter(
            grp["asmi_7d"],
            grp["duration_hours"],
            s=90,
            c=c,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.7,
            label=origin.replace("_", " ").title(),
            zorder=3,
        )

    x = sub["asmi_7d"].values
    y = sub["duration_hours"].values
    r, p_raw, p_corr, n_eff = pearsonr_corrected(x, y)

    if len(x) >= 3:
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_line, np.poly1d(z)(x_line), "r--", lw=2, alpha=0.7)

    sig = "**" if (not np.isnan(p_corr) and p_corr < 0.05) else "ns"
    p_str = f"{p_corr:.4f}" if not np.isnan(p_corr) else "N/A"
    summary = (
        f"Tracks analyzed: {len(sub)}\n"
        f"Mean ASMI 7d: {sub['asmi_7d'].mean():.4f} m³/m³\n"
        f"r = {r:.3f}\n"
        f"p_corr = {p_str}  {sig}\n"
        f"n_eff = {n_eff}"
    )
    ax.text(
        1.02,
        0.98,
        summary,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=common_bbox(),
        family="monospace",
    )

    ax.set_xlabel("ASMI (7-day, m³/m³)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Track Duration (hours)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Antecedent Soil Moisture vs MD Duration\n(ASMI: 7-day exponential window)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    legend_handles, _ = ax.get_legend_handles_labels()
    if legend_handles:
        ax.legend(handles=legend_handles, **legend_style())
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "antecedent_sm_vs_duration.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "as", "Antecedent SM vs duration", outpath)


def analyze_all_tracks_antecedent(ds, full_track_df, output_dir):
    unique_tracks = (
        sorted(full_track_df["track_id"].unique())
        if "track_id" in full_track_df.columns
        else [0]
    )

    all_results = []
    for track_id in unique_tracks:
        try:
            track_sub = (
                full_track_df[full_track_df["track_id"] == track_id].copy()
                if "track_id" in full_track_df.columns
                else full_track_df.copy()
            )
            result = compute_antecedent_for_track(ds, track_sub, track_id, output_dir)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            std_print("ERROR", "as", f"Track {track_id}: analysis failed", str(e))
            traceback.print_exc()

    if not all_results:
        std_print("WARN", "as", "No antecedent results to export")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)

    # Export
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "antecedent_analysis.csv")
    results_df.to_csv(csv_path, index=False)
    std_print("EXPORT", "as", "Antecedent analysis results", csv_path)

    # Summary plots
    if len(results_df) >= 3:
        plot_asmi_multi_lag_summary(results_df, output_dir)
        plot_antecedent_vs_duration(results_df, output_dir)

    if len(results_df) >= 4:
        compare_wet_dry_tracks(results_df, output_dir)
        create_antecedent_summary_report(results_df, output_dir)
    return results_df


def compare_wet_dry_tracks(results_df, output_dir):
    """
    Split tracks into Wet / Dry by median ASMI and compare:
      - inland penetration distance (km)
      - duration after landfall (hours)
      - track speed change at landfall (km/h)
      - mean LHF during land phase (W/m²)

    Uses Mann-Whitney U test (non-parametric, for small n)
    """
    if "asmi_7d" not in results_df.columns:
        std_print("WARN", "as", "No asmi_7d column found")
        return None

    results_df = results_df.copy()
    results_df["is_wet"] = classify_antecedent_conditions(results_df["asmi_7d"])
    wet = results_df[results_df["is_wet"] == True]
    dry = results_df[results_df["is_wet"] == False]

    std_print("INFO", "as", "Wet/Dry split", f"Wet: {len(wet)}, Dry: {len(dry)}")

    metric_cols = {
        "penetration_km": "Inland penetration (km)",
        "duration_land_hours": "Duration after landfall (h)",
        "speed_change_kmh": "Speed change at landfall (km/h)",
        "mean_lhf_land": "Mean LHF - land phase (W/m²)",
    }

    comparison = {}
    for col, label in metric_cols.items():
        if col not in results_df.columns:
            continue
        w_vals = wet[col].dropna().values
        d_vals = dry[col].dropna().values
        if len(w_vals) < 2 or len(d_vals) < 2:
            continue
        try:
            stat, p = mannwhitneyu(w_vals, d_vals, alternative="two-sided")
        except Exception:
            stat, p = np.nan, np.nan
        comparison[col] = {
            "label": label,
            "wet_mean": float(np.mean(w_vals)),
            "dry_mean": float(np.mean(d_vals)),
            "wet_median": float(np.median(w_vals)),
            "dry_median": float(np.median(d_vals)),
            "mann_whitney_u": stat,
            "p_value": float(p),
            "significant": p < 0.05 if not np.isnan(p) else False,
        }
        std_print(
            "INFO",
            "as",
            f"{label}",
            f"Wet: {np.mean(w_vals):.2f} vs Dry: {np.mean(d_vals):.2f}  p={p:.4f}",
        )

    # Figure
    if comparison:
        n_metrics = len(comparison)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 7))
        if n_metrics == 1:
            axes = [axes]
        for ax, (col, info) in zip(axes, comparison.items()):
            w_vals = wet[col].dropna().values if col in wet.columns else np.array([])
            d_vals = dry[col].dropna().values if col in dry.columns else np.array([])

            # Jitter x positions so points don't overlap
            rng = np.random.default_rng(42)
            jitter_w = rng.uniform(-0.06, 0.06, size=len(w_vals))
            jitter_d = rng.uniform(-0.06, 0.06, size=len(d_vals))

            ax.scatter(
                0 + jitter_w,
                w_vals,
                s=120,
                color="steelblue",
                edgecolors="black",
                linewidth=0.8,
                zorder=4,
                label="Wet",
            )
            ax.scatter(
                1 + jitter_d,
                d_vals,
                s=120,
                color="sandybrown",
                edgecolors="black",
                linewidth=0.8,
                zorder=4,
                label="Dry",
            )

            # Mean markers (horizontal line)
            if len(w_vals):
                ax.plot(
                    [-0.15, 0.15],
                    [info["wet_mean"]] * 2,
                    color="steelblue",
                    linewidth=2.5,
                    zorder=5,
                )
            if len(d_vals):
                ax.plot(
                    [0.85, 1.15],
                    [info["dry_mean"]] * 2,
                    color="sandybrown",
                    linewidth=2.5,
                    zorder=5,
                )

            sig_str = "**" if info["significant"] else "ns"
            ax.text(
                0.5,
                0.97,
                f"p={info['p_value']:.4f}  {sig_str}\n"
                f"Wet: {info['wet_mean']:.2f}   Dry: {info['dry_mean']:.2f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                bbox=common_bbox(),
                family="monospace",
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Wet ASMI", "Dry ASMI"], fontsize=12)
            ax.set_ylabel(info["label"], fontsize=12, fontweight="bold")
            ax.set_title(info["label"], fontsize=12, fontweight="bold", pad=8)
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        fig.suptitle(
            "Wet vs Dry Antecedent Conditions - Track Metrics\n"
            "(dots = individual tracks, horizontal bar = mean; n too small for bars)",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        # Save
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, "wet_dry_comparison.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "as", "Wet/Dry comparison plot", outpath)
    return comparison


def create_antecedent_summary_report(results_df, output_dir):
    """Summary of the antecedent analysis"""
    os.makedirs(output_dir, exist_ok=True)
    antecedent_analysis_report_path = os.path.join(
        output_dir, "antecedent_analysis_report.txt"
    )

    with open(antecedent_analysis_report_path, "w") as f:
        f.write("Antecedent Soil Moisture Analysis - Summary Report\n")
        f.write(f"Total tracks analyzed: {len(results_df)}\n\n")

        for lag in ["asmi_24h", "asmi_48h", "asmi_72h", "asmi_7d", "asmi_14d"]:
            if lag not in results_df.columns:
                continue
            valid = results_df[lag].dropna()
            if valid.empty:
                continue
            f.write(f"{lag.upper()}\n")
            f.write(f"  Mean  : {valid.mean():.4f} m³/m³\n")
            f.write(f"  Std   : {valid.std():.4f} m³/m³\n")
            f.write(f"  Min   : {valid.min():.4f} m³/m³\n")
            f.write(f"  Max   : {valid.max():.4f} m³/m³\n\n")

        # Correlation with duration (corrected p-value)
        if "asmi_7d" in results_df.columns and "duration_hours" in results_df.columns:
            sub = results_df.dropna(subset=["asmi_7d", "duration_hours"])
            if len(sub) >= 5:
                r, p_raw, p_corr, n_eff = pearsonr_corrected(
                    sub["asmi_7d"].values, sub["duration_hours"].values
                )
                sig = (
                    "significant"
                    if (not np.isnan(p_corr) and p_corr < 0.05)
                    else "not significant"
                )
                f.write("ASMI_7d vs Duration\n")
                f.write(f"  Pearson r              : {r:.4f}\n")
                f.write(
                    f"  Pearson p (raw)        : {p_raw:.4e}  [inflated by autocorrelation]\n"
                )
                f.write(f"  Pearson p (corrected)  : {p_corr:.4e}  [{sig}]\n")
                f.write(f"  n_eff                  : {n_eff}\n\n")
    std_print(
        "SAVED", "as", "Antecedent summary report", antecedent_analysis_report_path
    )


# Helper functions
def sm_at_time(ds, time_step, lat_min, lat_max, lon_min, lon_max):
    """Return spatial-mean SWVL1 over a bounding box at a given time."""
    ds, time_coord = get_time_coord(ds)
    try:
        ds_t = ds.sel({time_coord: time_step}, method="nearest")
        sm_sub = ds_t["swvl1"].sel(
            latitude=slice(lat_max, lat_min),  # ERA5 lat is descending
            longitude=slice(lon_min, lon_max),
        )
        if sm_sub.size == 0:
            return np.nan
        val = float(sm_sub.mean().values)
        return val if not np.isnan(val) else np.nan
    except Exception as e:
        std_print("ERROR", "as", "function sm_at_time failed", str(e))
        return np.nan


def classify_soil_moisture(sm_value):
    """Classify soil moisture level"""
    if np.isnan(sm_value):
        return "No Data"
    elif sm_value < 0.1:
        return "Very Dry"
    elif sm_value < 0.2:
        return "Dry"
    elif sm_value < 0.3:
        return "Moderate"
    elif sm_value < 0.4:
        return "Wet"
    else:
        return "Very Wet"


def future_track_bbox(track_df, buffer_km=100):
    """
    Bounding box of the inland portion of a track with buffer.
    Falls back to the full track if no is_land column is available.
    """
    if "is_land" in track_df.columns:
        inland = track_df[track_df["is_land"] == True]
        if inland.empty:
            inland = track_df  # fallback: land-originated
    else:
        inland = track_df

    lat_center = inland["latitude"].mean()
    lat_buf, lon_buf = calculate_buffer_deg(lat_center, buffer_km)

    lat_min = max(inland["latitude"].min() - lat_buf, CLIPPED_MIN_LAT)
    lat_max = min(inland["latitude"].max() + lat_buf, CLIPPED_MAX_LAT)
    lon_min = max(inland["longitude"].min() - lon_buf, CLIPPED_MIN_LON)
    lon_max = min(inland["longitude"].max() + lon_buf, CLIPPED_MAX_LON)
    return lat_min, lat_max, lon_min, lon_max


def classify_antecedent_conditions(asmi_series):
    """
    Classify tracks as Wet / Dry based on median ASMI split.
    Returns a boolean Series: True = Wet.
    """
    valid = asmi_series.dropna()
    if len(valid) < 4:
        return (asmi_series >= asmi_series.median()).astype(bool)
    median_val = valid.median()
    return asmi_series >= median_val


def compute_antecedent_for_track(ds, track_df, track_id, output_dir):
    """
    Compute all antecedent SM metrics for a single track.

    Returns a dict with:
      - asmi_{lag} for each lag window
      - origin_type, duration_hours, has_landfall
      - landfall_time (pd.Timestamp or None)
    """
    track_id = int(track_id)
    track = (
        track_df[track_df["track_id"] == track_id].sort_values("time")
        if "track_id" in track_df.columns
        else track_df.sort_values("time")
    )

    if track.empty:
        std_print("ERROR", "as", f"Track {track_id}: not found")
        return None

    times = pd.to_datetime(track["time"])
    duration_h = (times.max() - times.min()).total_seconds() / 3600

    # Landfall
    landfall_time = None
    has_landfall = False
    if "is_land" in track.columns:
        land_rows = track[track["is_land"] == True]
        if not land_rows.empty:
            landfall_time = pd.to_datetime(land_rows.iloc[0]["time"])
            has_landfall = True

    ref_time = landfall_time if landfall_time is not None else times.min()
    # INFO: for land-originated tracks (no landfall), times.min() is the very first detection point - the 24h window before it reaches into a pre-existence period that may be over ocean. Shift forward by 6h so the earliest inland points are included in the antecedent window.
    origin = find_origin_type(track)
    if landfall_time is None and origin == "land_originated":
        ref_time = times.min() + pd.Timedelta(hours=6)

    # Multi-lag ASMI
    asmi_dict = compute_asmi_multi_lag(ds, track, ref_time, buffer_km=100)

    result = {
        "track_id": track_id,
        "origin_type": find_origin_type(track),
        "has_landfall": has_landfall,
        "landfall_time": landfall_time,
        "duration_hours": duration_h,
        **asmi_dict,
    }

    # # Spatial map at 24 h before
    # try:
    #     plot_antecedent_spatial_map(ds, track_df, track_id, hours_before=24, output_dir)
    # except Exception as e:
    #     std_print("WARN", "as", "spatial map failed", str(e))

    # # Antecedent trend plot
    # try:
    #     analyze_antecedent_trend(ds, track_df, track_id, output_dir)
    # except Exception as e:
    #     std_print("WARN", "as", "Antecedent Trend plot failed", str(e))
    #

    std_print(
        "INFO",
        "as",
        f"Track {track_id} ASMI",
        f"24h={asmi_dict.get('asmi_24h', np.nan):.4f} "
        f"7d={asmi_dict.get('asmi_7d', np.nan):.4f}",
    )
    return result
