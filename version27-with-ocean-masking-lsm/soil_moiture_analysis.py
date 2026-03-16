"""
Analyzes the relationship between TP and SWVL1 during entire MD track passage.

Key Questions:
- Is there a correlation between TP and SWVL1?
- Does more precipitation lead to higher soil moisture?
- How strong is the linear relationship?
- Spatial patterns of both variables during passage?
"""

# TODO: add shade region for land (like in latent heat flux transiton and in speed plots)
# FIX: use buffer_km instaed of buffer_deg
# FIX: ACF lag selection : Requires>=48 points (24h lag + buffer)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from statsmodels.tsa.stattools import acf

from utils import (
    std_print,
    get_default_buffer_km,
    extract_track_variable,
    setup_map_extent,
    clamp_to_domain,
    get_time_coord,
    legend_style,
)


# def plot_spatial_tp_swvl1(ds, track_df, track_id, output_dir="output"):
#     """2-panel spatial maps of TP and SWVL1 during track passage"""
#     track_id = int(track_id)
#     track = track_df.sort_values("time")
#
#     min_lon, max_lon, min_lat, max_lat = setup_map_extent(track, buffer_deg=5.0)
#     min_lat, max_lat, min_lon, max_lon = clamp_to_domain(
#         min_lat, max_lat, min_lon, max_lon
#     )
#     ds, time_coord = get_time_coord(ds)
#
#     # FIX: use extract_track_variable
#     time_slice = slice(track["time"].min(), track["time"].max())
#     lat_slice = slice(max_lat, min_lat)
#     lon_slice = slice(min_lon, max_lon)
#
#     # Get averaged fields
#     tp_data = (
#         ds["tp"]
#         .sel({time_coord: time_slice, "latitude": lat_slice, "longitude": lon_slice})
#         .sum(dim=time_coord)
#         * 1000
#     )  # Total, converted to mm
#
#     sm_data = (
#         ds["swvl1"]
#         .sel({time_coord: time_slice, "latitude": lat_slice, "longitude": lon_slice})
#         .mean(dim=time_coord)
#     )
#
#     # Figure
#     fig, axes = plt.subplots(
#         1, 2, figsize=(13, 7), subplot_kw={"projection": ccrs.PlateCarree()}
#     )
#
#     # Panel 1: Total Precipitation
#     im1 = axes[0].contourf(
#         tp_data.longitude,
#         tp_data.latitude,
#         tp_data.values,
#         levels=20,
#         cmap="YlOrRd",
#         transform=ccrs.PlateCarree(),
#     )
#     contours1 = axes[0].contour(
#         tp_data.longitude,
#         tp_data.latitude,
#         tp_data.values,
#         levels=10,
#         colors="black",
#         linewidths=0.5,
#         alpha=0.4,
#         transform=ccrs.PlateCarree(),
#     )
#     axes[0].clabel(contours1, inline=True, fontsize=8, fmt="%1.0f")
#     axes[0].plot(
#         track["longitude"],
#         track["latitude"],
#         "b-",
#         linewidth=2,
#         transform=ccrs.PlateCarree(),
#         label="Track",
#         zorder=5,
#     )
#     axes[0].coastlines(resolution="50m", linewidth=1)
#     axes[0].add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
#     axes[0].set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
#     cbar1 = plt.colorbar(im1, ax=axes[0], pad=0.02)
#     cbar1.set_label("TP (mm)", fontsize=10)
#
#     # Legend
#     legend_handles, labels = axes[0].get_legend_handles_labels()
#     if legend_handles:
#         axes[0].legend(
#             handles=legend_handles,
#             loc="upper left",
#             **legend_style(),
#         )
#     axes[0].set_title("Total Precipitation (mm)", fontsize=12, fontweight="bold")
#
#     # Panel 2: Mean Soil Moisture
#     im2 = axes[1].contourf(
#         sm_data.longitude,
#         sm_data.latitude,
#         sm_data.values,
#         levels=20,
#         cmap="YlGnBu",
#         transform=ccrs.PlateCarree(),
#     )
#     contours2 = axes[1].contour(
#         sm_data.longitude,
#         sm_data.latitude,
#         sm_data.values,
#         levels=10,
#         colors="black",
#         linewidths=0.5,
#         alpha=0.4,
#         transform=ccrs.PlateCarree(),
#     )
#     axes[1].clabel(contours2, inline=True, fontsize=8, fmt="%1.3f")
#     axes[1].plot(
#         track["longitude"],
#         track["latitude"],
#         "r-",
#         linewidth=2,
#         transform=ccrs.PlateCarree(),
#         label="Track",
#         zorder=5,
#     )
#     axes[1].coastlines(resolution="50m", linewidth=1)
#     axes[1].add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
#     axes[1].set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
#     cbar2 = plt.colorbar(im2, ax=axes[1], pad=0.02)
#     cbar2.set_label("SWVL1 (m³/m³)", fontsize=10)
#
#     # Legend
#     legend_handles, labels = axes[1].get_legend_handles_labels()
#     if legend_handles:
#         axes[1].legend(
#             handles=legend_handles,
#             loc="upper left",
#             **legend_style(),
#         )
#     axes[1].set_title(
#         "Volumatric Soil Water Layer-1 (0-7 cm) (m³/m³)", fontsize=12, fontweight="bold"
#     )
#     fig.suptitle(
#         f"Track {track_id}: Spatial Distribution of TP and SWVL1",
#         fontsize=13,
#         fontweight="bold",
#     )
#     plt.tight_layout()
#
#     spactial_tp_sm_dir = os.path.join(output_dir, "spatial_tp_sm")
#     os.makedirs(spactial_tp_sm_dir, exist_ok=True)
#     outpath = os.path.join(
#         spactial_tp_sm_dir, f"track_{int(track_id)}_spatial_tp_sm.png"
#     )
#     plt.savefig(outpath, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     std_print("SAVED", "sm", "Spatial TP-SWVL1 plot", outpath)
def plot_spatial_tp_swvl1(ds, track_df, track_id, output_dir):
    """Spatial maps of TP and SWVL1 during track passage"""
    track_id = int(track_id)
    track = track_df.sort_values("time")

    min_lon, max_lon, min_lat, max_lat = setup_map_extent(track, buffer_deg=5.0)
    min_lat, max_lat, min_lon, max_lon = clamp_to_domain(
        min_lat, max_lat, min_lon, max_lon
    )
    ds, time_coord = get_time_coord(ds)

    # FIX: use extract_track_variable
    time_slice = slice(track["time"].min(), track["time"].max())
    lat_slice = slice(max_lat, min_lat)
    lon_slice = slice(min_lon, max_lon)

    # Get averaged fields
    tp_data = (
        ds["tp"]
        .sel({time_coord: time_slice, "latitude": lat_slice, "longitude": lon_slice})
        .sum(dim=time_coord)
        * 1000
    )  # Total, converted to mm

    sm_data = (
        ds["swvl1"]
        .sel({time_coord: time_slice, "latitude": lat_slice, "longitude": lon_slice})
        .mean(dim=time_coord)
    )

    # Plot 1: Total Precipitation
    fig1, ax1 = plt.subplots(
        figsize=(11, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    im1 = ax1.contourf(
        tp_data.longitude,
        tp_data.latitude,
        tp_data.values,
        levels=20,
        cmap="YlOrRd",
        transform=ccrs.PlateCarree(),
    )
    contours1 = ax1.contour(
        tp_data.longitude,
        tp_data.latitude,
        tp_data.values,
        levels=10,
        colors="black",
        linewidths=0.5,
        alpha=0.4,
        transform=ccrs.PlateCarree(),
    )
    ax1.clabel(contours1, inline=True, fontsize=8, fmt="%1.0f")

    ax1.plot(
        track["longitude"],
        track["latitude"],
        "b-",
        linewidth=2,
        transform=ccrs.PlateCarree(),
        label="Track",
        zorder=5,
    )

    ax1.coastlines(resolution="50m", linewidth=1)
    ax1.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax1.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label("Total Precipitation (mm)", fontsize=11)

    # Legend
    legend_handles, labels = ax1.get_legend_handles_labels()
    if legend_handles:
        ax1.legend(
            handles=legend_handles,
            loc="upper left",
            **legend_style(),
        )

    ax1.set_title(
        f"Track {track_id}: Total Precipitation During Passage",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # plot 2: Soil Moisture
    fig2, ax2 = plt.subplots(
        figsize=(11, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    im2 = ax2.contourf(
        sm_data.longitude,
        sm_data.latitude,
        sm_data.values,
        levels=20,
        cmap="YlGnBu",
        transform=ccrs.PlateCarree(),
    )
    contours2 = ax2.contour(
        sm_data.longitude,
        sm_data.latitude,
        sm_data.values,
        levels=10,
        colors="black",
        linewidths=0.5,
        alpha=0.4,
        transform=ccrs.PlateCarree(),
    )
    ax2.clabel(contours2, inline=True, fontsize=8, fmt="%1.3f")

    ax2.plot(
        track["longitude"],
        track["latitude"],
        "r-",
        linewidth=2,
        transform=ccrs.PlateCarree(),
        label="Track",
        zorder=5,
    )

    ax2.coastlines(resolution="50m", linewidth=1)
    ax2.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax2.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label("SWVL1 (m³/m³)", fontsize=11)

    # Legend
    legend_handles, labels = ax2.get_legend_handles_labels()
    if legend_handles:
        ax2.legend(
            handles=legend_handles,
            loc="upper left",
            **legend_style(),
        )

    ax2.set_title(
        f"Track {track_id}: Mean Soil Moisture (0-7cm) During Passage",
        fontsize=13,
        fontweight="bold",
    )
    ax2.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tp_file = os.path.join(output_dir, f"track_{track_id}_spatial_tp.png")
    plt.savefig(tp_file, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    std_print("SAVED", "sm", "Spatial TP plot", tp_file)
    sm_file = os.path.join(output_dir, f"track_{track_id}_spatial_swvl1.png")
    plt.savefig(sm_file, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    std_print("SAVED", "sm", "Spatial SWVL1 plot", sm_file)


def compute_soil_moisture_memory(swvl1_series, lag_hours=24):
    """Compute autocorrelation of SWVL1 at specified lag.
    Higher autocorrelation = longer memory = more sustained feedback.
    """
    # Remove NaN values
    swvl1_clean = np.array(swvl1_series)
    swvl1_clean = swvl1_clean[~np.isnan(swvl1_clean)]
    if len(swvl1_clean) < lag_hours * 2:
        return np.nan
    try:
        # Compute autocorrelation function
        acf_vals = acf(swvl1_clean, nlags=lag_hours, fft=True)
        # Return autocorrelation at specified lag
        if lag_hours < len(acf_vals):
            memory_index = acf_vals[lag_hours]
            return memory_index
        else:
            return np.nan
    except Exception as e:
        std_print("ERROR", "sm", "Returing NaN for memory_index", str(e))
        return np.nan


def analyze_soil_memory_by_track(ds, track_df_dict, output_dir):
    """Analyze soil moisture memory effect for each track"""
    memory_results = []
    buffer_km = get_default_buffer_km("swvl1")
    for track_id, track_df in track_df_dict.items():
        swvl1_vals = []
        ds, time_coord = get_time_coord(ds)
        try:
            _, swvl1_vals_arr = extract_track_variable(
                ds, track_df, track_id, time_coord, "swvl1", buffer_km
            )
        except Exception as e:
            std_print("ERROR", "sm", "extract_track_variable failed", str(e))
            continue
        swvl1_vals = swvl1_vals_arr.tolist()
        if len(swvl1_vals) > 48:  # Need enough data for 24h lag
            memory_24h = compute_soil_moisture_memory(swvl1_vals, lag_hours=24)
            memory_results.append(
                {
                    "track_id": track_id,
                    "memory_index_24h": memory_24h,
                    "n_samples": len(swvl1_vals),
                    "mean_swvl1": np.mean(swvl1_vals),
                    "std_swvl1": np.std(swvl1_vals),
                    "buffer_km": buffer_km,
                }
            )
    # Export results
    if memory_results:
        export_df = pd.DataFrame(memory_results)
        output_filename = os.path.join(output_dir, "soil_memory_analysis.csv")
        export_df.to_csv(output_filename, index=False)
        std_print("EXPORT", "sm", "Soil memory analysis to CSV", output_filename)
    else:
        std_print("ERROR", "sm", "No memory_results to export")
    return memory_results
