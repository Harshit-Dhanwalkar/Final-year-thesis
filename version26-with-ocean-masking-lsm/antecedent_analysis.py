# This script computes soil moisture conditions in the 24 hours before landfall.
# Is the antecedent soil moisture (wet vs dry) related to rainfall/flooding intensity during MD passage?

# INFO:
# - Dry antecedent conditions -> Soil can absorb a lot of the incoming rain -> lower immediate flood risk (more infiltration).
# - Wet antecedent conditions -> Soil is already near saturation → much more of the MD's heavy rainfall becomes surface runoff -> higher flood potential even if the total rainfall amount is not extreme.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from scipy.stats import pearsonr

import traceback

from utils import (
    std_print,
    find_origin_type,
    get_time_coord,
    common_bbox,
    legend_style,
)


# Helper function
def compute_antecedent_soil_moisture(ds, track_df, track_id, hours_before):
    track_id = int(track_id)
    # Get track data
    track = track_df[track_df["track_id"] == track_id].sort_values("time")
    if track.empty:
        std_print("ERROR", "as", f"Track {track_id} not found")
        return np.nan

    # Get system start time (first point)
    start_time = pd.to_datetime(track.iloc[0]["time"])
    antecedent_start = start_time - pd.Timedelta(hours=hours_before)
    start_time = start_time.strftime("%d/%m/%Y %H:%M:%S")
    antecedent_start = antecedent_start.strftime("%d/%m/%Y %H:%M:%S")

    # Define spatial domain: 200km buffer around future track
    lat_min, lat_max = track["latitude"].min(), track["latitude"].max()
    lon_min, lon_max = track["longitude"].min(), track["longitude"].max()

    # buffer (approximately 2° = ~220km at equator)
    lat_buffer = 2.0
    lon_buffer = 2.0
    lat_min_buf = max(10.0, lat_min - lat_buffer)
    lat_max_buf = min(30.0, lat_max + lat_buffer)
    lon_min_buf = max(70.0, lon_min - lon_buffer)
    lon_max_buf = min(95.0, lon_max + lon_buffer)

    # Extract soil moisture data before system
    ds, time_coord = get_time_coord(ds)

    try:
        # Get temporal subset
        ds_antecedent = ds.sel(
            {
                time_coord: slice(antecedent_start, start_time),
                "latitude": slice(lat_max_buf, lat_min_buf),  # ERA5: lat descending
                "longitude": slice(lon_min_buf, lon_max_buf),
            }
        )

        if ds_antecedent[time_coord].size == 0:
            std_print(
                "WARN",
                "as",
                f"Track {track_id}",
                "No temporal data in antecedent period",
            )
            return np.nan

        if "swvl1" not in ds_antecedent.data_vars:
            std_print("ERROR", "as", "SWVL1 not in dataset")
            return np.nan

        # Compute spatial and temporal average
        sm_mean = float(ds_antecedent["swvl1"].mean().values)

        std_print("INFO", "as", f"Antecedent SM ({hours_before}h before)")
        std_print(None, None, "Period", f"{antecedent_start} to {start_time}")
        std_print(None, None, "Mean SM", f"{sm_mean:.4f} m³/m³")
        std_print(
            None,
            None,
            "Spatial domain",
            f"Lat: [{lat_min_buf:.1f}, {lat_max_buf:.1f}], Lon: [{lon_min_buf:.1f}, {lon_max_buf:.1f}]",
        )
        return sm_mean
    except Exception as e:
        std_print("ERROR", "as", "Antecedent computation failed", str(e))
        return np.nan


# Helper function
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


def plot_antecedent_spatial_map(ds, track_df, track_id, hours_before, output_dir):
    track_id = int(track_id)
    # Get track data
    track = track_df[track_df["track_id"] == track_id].sort_values("time")
    if track.empty:
        std_print("ERROR", "as", f"Track {track_id} not found")
        return False

    # Get system start time
    system_start = pd.to_datetime(track.iloc[0]["time"])
    antecedent_time = system_start - pd.Timedelta(hours=hours_before)

    # Define spatial extent with buffer
    lat_center = track["latitude"].mean()
    lon_center = track["longitude"].mean()
    lat_range = max(5.0, track["latitude"].max() - track["latitude"].min()) * 1.5
    lon_range = max(5.0, track["longitude"].max() - track["longitude"].min()) * 1.5

    min_lat = max(10.0, lat_center - lat_range / 2)
    max_lat = min(30.0, lat_center + lat_range / 2)
    min_lon = max(70.0, lon_center - lon_range / 2)
    max_lon = min(95.0, lon_center + lon_range / 2)

    ds, time_coord = get_time_coord(ds)
    try:
        # Get soil moisture at antecedent time
        ds_at_time = ds.sel({time_coord: antecedent_time}, method="nearest")
        sm_data = ds_at_time["swvl1"].sel(
            latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
        )
        if sm_data.size == 0:
            std_print(
                "WARN", "as", f"Track {track_id}", "No SM data at antecedent time"
            )
            return False

        # Figure
        fig = plt.figure(figsize=(13, 10))
        ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]

        ax.coastlines(resolution="50m", linewidth=1.0)
        ax.add_feature(
            cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.5
        )
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.LAKES, facecolor="lightblue")
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=6)
        ax.add_feature(
            cfeature.BORDERS, linestyle=":", alpha=0.5, linewidth=0.8, zorder=6
        )
        ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle="--", alpha=0.7)
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.6,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

        # Plot soil moisture
        im = ax.contourf(
            sm_data.longitude,
            sm_data.latitude,
            sm_data.values,
            levels=21,
            cmap="YlGnBu",
            transform=ccrs.PlateCarree(),
            extend="both",
        )

        # Contours
        contours = ax.contour(
            sm_data.longitude,
            sm_data.latitude,
            sm_data.values,
            levels=10,
            colors="black",
            linewidths=0.5,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(contours, inline=True, fontsize=8, fmt="%1.2f")

        # Plot FUTURE track
        ax.plot(
            track["longitude"],
            track["latitude"],
            "r-",
            linewidth=3,
            transform=ccrs.PlateCarree(),
            label="Future MD Track",
            zorder=5,
        )

        if len(track) > 0:
            # MD position at antecedent time (not yet formed), mark where it WILL form
            ax.scatter(
                track.iloc[0]["longitude"],
                track.iloc[0]["latitude"],
                s=300,
                c="gold",
                marker="s",
                edgecolors="black",
                linewidth=2,
                transform=ccrs.PlateCarree(),
                label="Future MD Genesis",
                zorder=6,
            )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label("Soil Moisture (m³/m³)", fontsize=11, fontweight="bold")

        # Title
        title = f"Track {track_id}: Antecedent Soil Moisture\n{hours_before}h before System Formation\n({antecedent_time.strftime('%d/%m/%Y %H:%M')})"
        ax.set_title(title, fontsize=13, fontweight="bold")

        # Legend
        legend_handles, labels = ax.get_legend_handles_labels()
        if legend_handles:
            ax.legend(handles=legend_handles, loc="lower left", **legend_style())
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(
            output_dir, f"track_{track_id}_antecedent_spatial_map.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "as", "Antecedent spatial map", plot_path)
    except Exception as e:
        std_print("ERROR", "as", "Spatial map failed", str(e))
        traceback.print_exc()


def plot_all_tracks_antecedent_map(ds, track_df, hours_before, output_dir, extent):
    """Plot antecedent SM for all tracks on a single map"""
    std_print("INFO", "as", "Plotting all-tracks antecedent spatial map")
    ds, time_coord = get_time_coord(ds)
    unique_tracks = sorted(track_df["track_id"].unique())

    if len(unique_tracks) == 0:
        std_print("WARN", "as", "No tracks to plot")
        return False

    # Map extent
    if extent is not None:
        # full_extent = [min_lon, max_lon, min_lat, max_lat]
        min_lon, max_lon, min_lat, max_lat = extent
    else:
        # fallback: union of track bounding boxes + buffer
        buf = 4.0
        min_lat = max(2.0, track_df["latitude"].min() - buf)
        max_lat = min(35.0, track_df["latitude"].max() + buf)
        min_lon = max(60.0, track_df["longitude"].min() - buf)
        max_lon = min(100.0, track_df["longitude"].max() + buf)
    std_print(
        "DEBUG",
        "as",
        "Map extent",
        f"lon=[{min_lon},{max_lon}] lat=[{min_lat},{max_lat}]",
    )

    # SM snapshot: median antecedent time across all tracks
    antecedent_times = []
    for tid in unique_tracks:
        trk = track_df[track_df["track_id"] == tid].sort_values("time")
        if trk.empty:
            continue
        t0 = pd.to_datetime(trk.iloc[0]["time"]) - pd.Timedelta(hours=hours_before)
        antecedent_times.append(t0)

    if not antecedent_times:
        std_print("ERROR", "as", "No valid antecedent times found")
        return False

    antecedent_times_sorted = sorted(antecedent_times)
    mid_idx = len(antecedent_times_sorted) // 2
    ref_time = antecedent_times_sorted[mid_idx]
    std_print("DEBUG", "as", "SM snapshot time", ref_time.strftime("%d/%m/%Y %H:%M"))

    # Load SM field over full extent
    try:
        ds_at_time = ds.sel({time_coord: ref_time}, method="nearest")
        sm_data = ds_at_time["swvl1"].sel(
            latitude=slice(max_lat, min_lat),  # ERA5: descending lat order
            longitude=slice(min_lon, max_lon),
        )
        if sm_data.size == 0:
            std_print("WARN", "as", "No SM data for reference time", str(ref_time))
            return False
    except Exception as e:
        std_print("ERROR", "as", "Failed to load SM field", str(e))
        return False

    # Figure
    track_colors = plt.cm.tab10.colors
    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=1.0)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor="lightblue", zorder=0)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=5)
    ax.add_feature(cfeature.STATES, linewidth=0.4, linestyle="--", zorder=5)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    # SM contourf
    im = ax.contourf(
        sm_data.longitude,
        sm_data.latitude,
        sm_data.values,
        levels=21,
        cmap="YlGnBu",
        transform=ccrs.PlateCarree(),
        extend="both",
        alpha=0.85,
        zorder=1,
    )
    ax.contour(
        sm_data.longitude,
        sm_data.latitude,
        sm_data.values,
        levels=8,
        colors="black",
        linewidths=0.4,
        alpha=0.35,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("Soil Moisture SWVL1 (m³/m³)", fontsize=11, fontweight="bold")

    # Overlay each track
    for i, tid in enumerate(unique_tracks):
        color = track_colors[i % len(track_colors)]
        trk = track_df[track_df["track_id"] == tid].sort_values("time")
        if trk.empty:
            continue

        # Track line
        ax.plot(
            trk["longitude"],
            trk["latitude"],
            color=color,
            linewidth=2.5,
            transform=ccrs.PlateCarree(),
            label=f"Track {int(tid)}",
            zorder=6,
        )
        # Genesis point (square)
        ax.scatter(
            trk.iloc[0]["longitude"],
            trk.iloc[0]["latitude"],
            s=250,
            c=[color],
            marker="s",
            edgecolors="black",
            linewidth=1,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )
        # Dissipation point (X)
        ax.scatter(
            trk.iloc[-1]["longitude"],
            trk.iloc[-1]["latitude"],
            s=200,
            c=[color],
            marker="X",
            edgecolors="black",
            linewidth=1,
            transform=ccrs.PlateCarree(),
            zorder=7,
        )
        # Label
        ax.annotate(
            f"T{int(tid)}",
            xy=(trk.iloc[0]["longitude"], trk.iloc[0]["latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=color,
            transform=ccrs.PlateCarree(),
            bbox=common_bbox(),
            zorder=8,
        )

    # Legend & title
    legend_handles, labels = ax.get_legend_handles_labels()
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", **legend_style())

    title = (
        f"Antecedent Soil Moisture — All Tracks ({len(unique_tracks)} MDs)\n"
        f"SM snapshot: {ref_time.strftime('%d/%m/%Y %H:%M')} "
        f"(≈{hours_before}h before earliest genesis)"
    )
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "all_tracks_antecedent_spatial_map.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "as", "All-tracks antecedent spatial map", plot_path)
    return True


def analyze_antecedent_trend(ds, track_df, track_id, output_dir):
    """Analyze how soil moisture changes before system formation"""
    track_id = int(track_id)
    track = track_df[track_df["track_id"] == track_id].sort_values("time")
    if track.empty:
        return np.nan

    system_start = pd.to_datetime(track.iloc[0]["time"])

    # time points: before system formation
    time_points = [48, 36, 24, 12, 0]
    sm_values = []
    times = []
    ds, time_coord = get_time_coord(ds)

    for hours_before in time_points:
        antecedent_time = system_start - pd.Timedelta(hours=hours_before)
        # Spatial domain around future track
        lat_center = track["latitude"].mean()
        lon_center = track["longitude"].mean()
        try:
            ds_at_time = ds.sel({time_coord: antecedent_time}, method="nearest")
            # Extract region (1° buffer)
            lat_min = max(10.0, lat_center - 1.0)
            lat_max = min(30.0, lat_center + 1.0)
            lon_min = max(70.0, lon_center - 1.0)
            lon_max = min(95.0, lon_center + 1.0)
            sm_region = ds_at_time["swvl1"].sel(
                latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
            )
            if sm_region.size > 0:
                sm_mean = float(sm_region.mean().values)
                sm_values.append(sm_mean)
                times.append(hours_before)
        except Exception as e:
            std_print("WARN", "as", f"Hour {hours_before}", str(e))
            continue

    if len(sm_values) >= 3:
        # Plot trend
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, sm_values, "o-", linewidth=2.5, markersize=6, color="darkgreen")
        ax.fill_between(times, sm_values, alpha=0.2, color="lightgreen")
        ax.set_xlabel("Hours Before System Formation", fontsize=11, fontweight="bold")
        ax.set_ylabel(
            "Soil Water Volumetric Layer-1 (0-7cm) (m³/m³)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_title(
            f"Track {track_id}: Antecedent Soil Moisture Trend\n48h Before MD Formation",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.grid(True, alpha=0.3, linestyle="--")

        # Trend line
        if len(times) >= 3:
            z = np.polyfit(times, sm_values, 1)
            p_fit = np.poly1d(z)
            ax.plot(
                times,
                p_fit(times),
                "r--",
                linewidth=2,
                alpha=0.7,
                label=f"Trend: {z[0]:.4f} m³/m³ per hour",
            )

            # Interpret trend
            if z[0] > 0.001:
                trend_text = "Increasing moisture before MD"
            elif z[0] < -0.001:
                trend_text = "Decreasing moisture before MD"
            else:
                trend_text = "Stable moisture before MD"

            bbox_style = common_bbox()
            ax.text(
                1.02,
                0.98,
                trend_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=bbox_style,
            )

        # Legend
        legend_handles, labels = ax.get_legend_handles_labels()
        if legend_handles:
            ax.legend(handles=legend_handles, loc="lower left", **legend_style())

        ax.invert_xaxis()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f"track_{track_id}_antecedent_trend.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        std_print("SAVED", "as", "Antecedent trend plot", plot_path)
        return sm_values
    else:
        std_print("WARN", "as", "Insufficient data for trend analysis (< 3)")
        return None


def analyze_antecedent_for_all_tracks(
    ds, track_df, hours_before, output_dir, extent=None
):
    std_print("INFO", "as", "Antecedent analysis for all tracks")
    unique_tracks = sorted(track_df["track_id"].unique())
    results = []
    for track_id in unique_tracks:
        # Compute antecedent soil moisture
        asm = compute_antecedent_soil_moisture(ds, track_df, track_id, hours_before)

        # Plot spatial map
        plot_antecedent_spatial_map(ds, track_df, track_id, hours_before, output_dir)

        # Get track characteristics
        track = track_df[track_df["track_id"] == track_id].sort_values("time")
        duration_hours = len(track)
        origin_type = find_origin_type(track)

        # Classify as wet/dry
        sm_class = classify_soil_moisture(asm)
        results.append(
            {
                "track_id": track_id,
                "antecedent_sm": asm,
                "duration_hours": duration_hours,
                "origin_type": origin_type,
                "sm_class": sm_class,
                "system_start": pd.to_datetime(track.iloc[0]["time"]),
                "n_points": len(track),
            }
        )

        std_print("INFO", "as", f"Track {track_id} Summary")
        std_print(None, None, "Antecedent SM", f"{asm:.4f} m³/m³ ({sm_class})")
        std_print(None, None, "Duration", f"{duration_hours} hours")
        std_print(None, None, "Origin", origin_type)

    results_df = pd.DataFrame(results)

    # Combined map — all tracks on one plot over full dataset extent
    plot_all_tracks_antecedent_map(
        ds,
        track_df,
        hours_before=hours_before,
        output_dir=output_dir,
        extent=extent,
    )

    # Analyze correlations and export
    analyze_antecedent_correlations(results_df, output_dir)
    create_antecedent_summary_report(results_df, output_dir)
    export_path = os.path.join(output_dir, "antecedent_analysis_results.csv")
    results_df.to_csv(export_path, index=False)
    std_print("SAVED", "as", "Antecedent results to CSV", export_path)
    return results_df


# Helper functions
# def analyze_antecedent_correlations(results_df, output_dir):
#     """Analyze correlations between antecedent SM and track characteristics"""
#     std_print("INFO", "as", "Analyzing antecedent correlations")
#
#     # Filter tracks with valid antecedent data
#     valid_df = results_df[results_df["antecedent_sm"].notna()].copy()
#     if len(valid_df) < 3:
#         std_print("WARN", "as", "Insufficient data for correlation analysis")
#         return
#     # 1. Correlation with duration
#     if len(valid_df) >= 3:
#         r_duration, p_duration = pearsonr(
#             valid_df["antecedent_sm"], valid_df["duration_hours"]
#         )
#         std_print("INFO", "as", "Antecedent SM vs Duration")
#         std_print(None, None, "Pearson r", f"{r_duration:.3f}")
#         std_print(None, None, "p-value", f"{p_duration:.4f}")
#         if p_duration < 0.05:
#             if r_duration > 0:
#                 interpretation = "Wetter soils → Longer MD duration"
#             else:
#                 interpretation = "Drier soils → Longer MD duration"
#         else:
#             interpretation = "No significant correlation"
#         std_print(None, None, "Interpretation", interpretation)
#
#     # 2. Compare ocean vs land origin
#     if "origin_type" in valid_df.columns:
#         ocean_tracks = valid_df[valid_df["origin_type"] == "ocean_originated"]
#         land_tracks = valid_df[valid_df["origin_type"] == "land_originated"]
#         std_print("INFO", "as", "Antecedent SM by Origin Type")
#         if not ocean_tracks.empty:
#             std_print(
#                 None,
#                 None,
#                 "Ocean-origin mean SM",
#                 f"{ocean_tracks['antecedent_sm'].mean():.4f} m³/m³",
#             )
#             std_print(None, None, "Ocean-origin count", len(ocean_tracks))
#         if not land_tracks.empty:
#             std_print(
#                 None,
#                 None,
#                 "Land-origin mean SM",
#                 f"{land_tracks['antecedent_sm'].mean():.4f} m³/m³",
#             )
#             std_print(None, None, "Land-origin count", len(land_tracks))
#
#     # 3. Scatter plot
#     antecedent_scatter_plot(valid_df, output_dir)
#
#
# def antecedent_scatter_plot(results_df, output_dir):
#     """Scatter plot of antecedent SM vs duration"""
#     if len(results_df) < 3:
#         return
#
#     fig, ax = plt.subplots(figsize=(10, 8))
#
#     # Color by origin type
#     colors = {"ocean_originated": "blue", "land_originated": "green"}
#     for origin, color in colors.items():
#         subset = results_df[results_df["origin_type"] == origin]
#         if not subset.empty:
#             ax.scatter(
#                 subset["antecedent_sm"],
#                 subset["duration_hours"],
#                 c=color,
#                 s=150,
#                 alpha=0.7,
#                 edgecolors="black",
#                 linewidth=1.5,
#                 label=origin.replace("_", " ").title(),
#             )
#
#     # Regression line if significant correlation
#     if len(results_df) >= 3:
#         try:
#             r, p = pearsonr(results_df["antecedent_sm"], results_df["duration_hours"])
#             if p < 0.05:
#                 z = np.polyfit(
#                     results_df["antecedent_sm"], results_df["duration_hours"], 1
#                 )
#                 p_fit = np.poly1d(z)
#                 x_line = np.linspace(
#                     results_df["antecedent_sm"].min(),
#                     results_df["antecedent_sm"].max(),
#                     100,
#                 )
#                 ax.plot(
#                     x_line,
#                     p_fit(x_line),
#                     "r--",
#                     linewidth=2,
#                     alpha=0.7,
#                     label=f"r = {r:.3f}, p = {p:.4f}",
#                 )
#         except:
#             pass
#
#     ax.set_xlabel("Antecedent Soil Moisture (m³/m³)", fontsize=11, fontweight="bold")
#     ax.set_ylabel("Track Duration (hours)", fontsize=11, fontweight="bold")
#     ax.grid(True, alpha=0.3, linestyle="--")
#
#     # Legend
#     legend_handles, labels = ax.get_legend_handles_labels()
#     if legend_handles:
#         ax.legend(handles=legend_handles, loc="lower left", **legend_style())
#
#     ax.set_title(
#         "Antecedent Soil Moisture vs. MD Duration",
#         fontsize=13,
#         fontweight="bold",
#         pad=15,
#     )
#
#     summary_text = f"Tracks analyzed: {len(results_df)}\n"
#     summary_text += (
#         f"Mean antecedent SM: {results_df['antecedent_sm'].mean():.4f} m³/m³\n"
#     )
#     summary_text += f"Mean duration: {results_df['duration_hours'].mean():.0f} hours"
#
#     bbox_style = common_bbox()
#     ax.text(
#         1.02,
#         0.98,
#         summary_text,
#         transform=ax.transAxes,
#         fontsize=11,
#         verticalalignment="top",
#         bbox=bbox_style,
#     )
#     plt.tight_layout()
#
#     # Save
#     plot_path = os.path.join(output_dir, "antecedent_sm_vs_duration.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     std_print("SAVED", "as", "Antecedent scatter plot", plot_path)


def create_antecedent_summary_report(results_df, output_dir):
    report_path = os.path.join(output_dir, "antecedent_analysis_report.txt")

    with open(report_path, "w") as f:
        f.write("Antecedent soil moisture analysis report\n")
        # Basic statistics
        f.write("1. Basic statistics\n")
        f.write(f"Total tracks analyzed: {len(results_df)}\n")

        valid_sm = results_df["antecedent_sm"].dropna()
        if len(valid_sm) > 0:
            f.write(f"Tracks with valid antecedent data: {len(valid_sm)}\n")
            f.write(f"Mean antecedent SM: {valid_sm.mean():.4f} m³/m³\n")
            f.write(f"Std deviation: {valid_sm.std():.4f} m³/m³\n")
            f.write(f"Minimum: {valid_sm.min():.4f} m³/m³\n")
            f.write(f"Maximum: {valid_sm.max():.4f} m³/m³\n\n")

        # Soil moisture classification
        f.write("2. Soil moisture classification\n")
        if "sm_class" in results_df.columns:
            class_counts = results_df["sm_class"].value_counts()
            for sm_class, count in class_counts.items():
                f.write(
                    f"{sm_class}: {count} tracks ({count / len(results_df) * 100:.1f}%)\n"
                )
            f.write("\n")

        # By origin type
        f.write("3. By origin type\n")
        if "origin_type" in results_df.columns:
            origin_counts = results_df["origin_type"].value_counts()
            for origin, count in origin_counts.items():
                subset = results_df[results_df["origin_type"] == origin]
                sm_mean = subset["antecedent_sm"].mean()
                duration_mean = subset["duration_hours"].mean()
                f.write(f"{origin.replace('_', ' ').title()}: {count} tracks\n")
                f.write(f"  Mean antecedent SM: {sm_mean:.4f} m³/m³\n")
                f.write(f"  Mean duration: {duration_mean:.0f} hours\n\n")

        # Key findings
        f.write("4. Key findings\n")
        # Check for correlation with duration
        if len(valid_sm) >= 3:
            try:
                r, p = pearsonr(
                    results_df["antecedent_sm"].dropna(),
                    results_df["duration_hours"].dropna(),
                )
                f.write(f"Correlation with duration: r = {r:.3f}, p = {p:.4f}\n")
                if p < 0.05:
                    if r > 0.5:
                        f.write("  STRONG POSITIVE: Wetter soils support longer MDs\n")
                    elif r > 0.3:
                        f.write(
                            "  MODERATE POSITIVE: Wetter soils may extend MD life\n"
                        )
                    elif r < -0.3:
                        f.write("  NEGATIVE: Drier soils support longer MDs\n")
                    else:
                        f.write("  WEAK but significant relationship\n")
                else:
                    f.write("  No significant correlation with duration\n")
            except:
                f.write("  Correlation analysis failed\n")
    std_print("SAVED", "as", "Antecedent analysis report", report_path)
