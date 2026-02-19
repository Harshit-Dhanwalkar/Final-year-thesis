"""
Overlay Monsoon Depression Tracks on SRTM Elevation Mosaic
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import rasterio
import subprocess

from utils import std_print

import traceback

MONTH_COLORS = {
    "June": "#FFA500",
    "July": "#00cc55",
    "August": "#ee3311",
    "September": "#ff017f",
}


def create_elevation_vrt(
    hgt_dir="../Data/Elevation/HGT-files", vrt_filename="elevation_mosaic.vrt"
):
    """Build Virtual Raster (VRT) from all HGT files in directory"""
    search_path = os.path.join(hgt_dir, "**/*.hgt")
    hgt_files = glob.glob(search_path, recursive=True)
    if not hgt_files:
        std_print("WARN", "vis", "No HGT files found", hgt_dir)
        return None
    std_print("INFO", "vis", "Creating elevation VRT", f"{len(hgt_files)} tiles")
    try:
        # Build VRT using gdalbuildvrt
        result = subprocess.run(
            ["gdalbuildvrt", "-overwrite", vrt_filename] + hgt_files,
            check=True,
            capture_output=True,
            text=True,
        )
        if os.path.exists(vrt_filename):
            std_print("INFO", "vis", "VRT created successfully", vrt_filename)
            return vrt_filename
        else:
            std_print("ERROR", "vis", "VRT file not created", "")
            return None
    except FileNotFoundError:
        std_print("ERROR", "vis", "gdalbuildvrt not found")
        return None
    except subprocess.CalledProcessError as e:
        std_print("ERROR", "vis", "VRT build failed", e.stderr[:80])
        return None
    except Exception as e:
        std_print("ERROR", "vis", "Unexpected error", str(e)[:80])
        return None


def plot_tracks_on_elevation(track_df, vrt_path, output_dir="output", year=2024):
    """Create elevation mosaic with MD tracks overlaid"""
    if track_df is None or track_df.empty:
        std_print("WARN", "vis", "No tracks to visualize", "")
        return None
    std_print("INFO", "vis", "--- Elevation Track Visualization ---", "")
    std_print(
        "INFO",
        "vis",
        "Input tracks",
        f"{len(track_df)} points, {len(track_df['track_id'].unique())} tracks",
    )
    elev_dir = os.path.join(output_dir, "elevation_analysis")
    os.makedirs(elev_dir, exist_ok=True)

    track_df = track_df.copy()
    if "month" not in track_df.columns:
        track_df["time"] = pd.to_datetime(track_df["time"])
        track_df["month"] = track_df["time"].dt.strftime("%B")
    std_print(
        "INFO", "vis", "Track months", ", ".join(track_df["month"].dropna().unique())
    )
    try:
        with rasterio.open(vrt_path) as src:
            std_print(
                "DEBUG",
                "vis",
                "VRT extent",
                f"E:{src.bounds.left:.1f}°-{src.bounds.right:.1f}°, "
                f"N:{src.bounds.bottom:.1f}°-{src.bounds.top:.1f}°",
            )
            # Read elevation at reduced resolution to avoid memory issues
            scale_factor = 0.05
            out_shape = (int(src.height * scale_factor), int(src.width * scale_factor))
            std_print(
                "INFO",
                "vis",
                "Reading elevation mosaic",
                f"{out_shape[1]}x{out_shape[0]} pixels",
            )
            elev_data = src.read(
                1, out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear
            ).astype(float)
            # Clean invalid data (SRTM void value)
            elev_data[elev_data < -50] = 0
            # Plot
            fig, ax = plt.subplots(figsize=(20, 14), dpi=100)
            # Calculate colormap limits (clip at 95th percentile to avoid washout)
            vmin_val = np.nanpercentile(elev_data, 1)
            vmax_val = np.nanpercentile(elev_data, 95)
            vmin = max(vmin_val, 0.1)
            vmax = vmax_val
            # Plot elevation with logarithmic scale
            im = ax.imshow(
                elev_data + 0.1,
                cmap="terrain",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                extent=[
                    src.bounds.left,
                    src.bounds.right,
                    src.bounds.bottom,
                    src.bounds.top,
                ],
                origin="upper",
                zorder=1,
            )
            # Plot tracks by month
            std_print("INFO", "vis", "Plotting tracks", "")
            track_count = 0
            for month_name, color in MONTH_COLORS.items():
                month_subset = track_df[track_df["month"] == month_name]
                if not month_subset.empty:
                    for tid in month_subset["track_id"].unique():
                        track = month_subset[
                            month_subset["track_id"] == tid
                        ].sort_values("time")
                        label = f"MD Track ({month_name})" if track_count == 0 else ""
                        ax.plot(
                            track["longitude"].values,
                            track["latitude"].values,
                            color=color,
                            linewidth=2.0,
                            alpha=0.9,
                            label=label if label else None,
                            zorder=10,
                        )
                        track_count += 1
            std_print(
                "INFO", "vis", "Tracks plotted", f"{track_count} individual tracks"
            )

            cbar = plt.colorbar(im, fraction=0.03, pad=0.04)
            cbar.set_label("Elevation (m)", fontweight="bold", fontsize=11)
            title = f"Monsoon Depression Tracks Over SRTM Elevation ({year})"
            ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
            ax.set_xlabel("Longitude (°E)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Latitude (°N)", fontsize=12, fontweight="bold")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc="lower left",
                    framealpha=0.9,
                    fontsize=9,
                    edgecolor="none",
                    fancybox=True,
                    shadow=True,
                )
            ax.grid(True, alpha=0.3, linestyle="--", zorder=1)
            plt.tight_layout()
            # Save
            output_filename = f"MD_tracks_elevation_mosaic_{year}.png"
            output_path = os.path.join(elev_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            std_print("SAVED", "vis", "Elevation visualization", output_path)
            return output_path
    except Exception as e:
        std_print("ERROR", "vis", "Visualization failed", str(e)[:80])
        traceback.print_exc()
        return None


# Backwards compatibility wrapper
def plot_tracks_elevation_mosaic(
    csv_path, vrt_path, output_path="tracks_on_elevation.png"
):
    """
    Legacy function for standalone usage
    Use plot_tracks_on_elevation() instead for integrated usage.
    """
    try:
        track_df = pd.read_csv(csv_path)
        track_df["time"] = pd.to_datetime(track_df["time"])
        track_df["month"] = track_df["time"].dt.strftime("%B")
        std_print(
            "INFO",
            "vis",
            f"Loaded{len(track_df)}track points",
            f"{track_df['month'].nunique()}months",
        )
        with rasterio.open(vrt_path) as src:
            scale_factor = 0.05
            out_shape = (
                int(src.height * scale_factor),
                int(src.width * scale_factor),
            )
            std_print(
                "INFO", "vis", "Reading elevation", f"{out_shape[1]}x{out_shape[0]}"
            )
            elev_data = src.read(
                1, out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear
            ).astype(float)
            elev_data[elev_data < -50] = 0
            fig, ax = plt.subplots(figsize=(20, 14), dpi=100)
            vmin, vmax = np.nanpercentile(elev_data, [1, 95])
            vmin = max(vmin, 0.1)
            im = ax.imshow(
                elev_data + 0.1,
                cmap="terrain",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                extent=[
                    src.bounds.left,
                    src.bounds.right,
                    src.bounds.bottom,
                    src.bounds.top,
                ],
                origin="upper",
                zorder=1,
            )
            # Plot tracks
            for month_name, color in MONTH_COLORS.items():
                month_subset = track_df[track_df["month"] == month_name]
                if not month_subset.empty:
                    for tid in month_subset["track_id"].unique():
                        track = month_subset[month_subset["track_id"] == tid]
                        label = (
                            f"MD Track ({month_name})"
                            if tid == month_subset["track_id"].iloc[0]
                            else ""
                        )
                        ax.plot(
                            track["longitude"],
                            track["latitude"],
                            color=color,
                            linewidth=2.0,
                            alpha=0.9,
                            label=label if label else None,
                            zorder=10,
                        )
            cb = plt.colorbar(im, fraction=0.03, pad=0.04)
            cb.set_label("Elevation (m)", fontweight="bold")
            ax.set_title(
                "Monsoon Depression Tracks Over SRTM Elevation",
                fontsize=18,
                fontweight="bold",
                pad=20,
            )
            ax.set_xlabel("Longitude (°E)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Latitude (°N)", fontsize=12, fontweight="bold")
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc="lower left",
                framealpha=0.9,
                fontsize=9,
                edgecolor="none",
                fancybox=True,
                shadow=True,
            )
            ax.grid(True, alpha=0.3, linestyle="--", zorder=1)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            std_print("SAVED", "vis", "Plot saved", output_path)
            return output_path
    except Exception as e:
        std_print("ERROR", "vis", "Failed", str(e)[:80])
        return None
