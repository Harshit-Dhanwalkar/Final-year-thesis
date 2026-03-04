"""MD Tracks on SRTM Elevation Mosaic"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import rasterio
import subprocess

from utils import std_print, legend_style

import traceback

MONTH_COLORS = {"June": "#000", "July": "#000", "August": "#000", "September": "#000"}


def create_elevation_vrt(hgt_dir, vrt_filename="elevation_mosaic.vrt"):
    """Build Virtual Raster (VRT) from all HGT files in directory"""
    search_path = os.path.join(hgt_dir, "**/*.hgt")
    hgt_files = glob.glob(search_path, recursive=True)
    if not hgt_files:
        std_print("WARN", "ev", "No HGT files found", hgt_dir)
        return None

    std_print("INFO", "ev", "Creating elevation VRT", f"{len(hgt_files)} tiles")
    try:
        # Build VRT using gdalbuildvrt
        # result = subprocess.run(
        #     ["gdalbuildvrt", "-overwrite", vrt_filename] + hgt_files,
        #     check=True,
        #     capture_output=True,
        #     text=True,
        # )
        # Check if file was created
        if os.path.exists(vrt_filename):
            std_print("INFO", "ev", "VRT created", vrt_filename)
            return vrt_filename
        else:
            std_print("ERROR", "ev", "VRT file not created")
            return None
    except FileNotFoundError:
        std_print(
            "ERROR",
            "ev",
            "gdalbuildvrt not found",
            "Install with: sudo apt install gdal-bin",
        )
        return None
    except subprocess.CalledProcessError as e:
        std_print("ERROR", "ev", "VRT build failed", e.stderr)
        return None
    except Exception as e:
        std_print("ERROR", "ev", "Unexpected error", str(e))
        return None


def plot_tracks_on_elevation(track_id, track_df, vrt_path, output_dir):
    """Create elevation mosaic with MD tracks overlaid"""
    track_id = int(track_id)
    if track_df is None or track_df.empty:
        std_print("WARN", "ev", "No tracks to visualize")
        return None
    std_print("INFO", "ev", "--- Elevation Track visualization ---")
    std_print(
        "INFO",
        "ev",
        f"Input track {track_id}",
        f"{len(track_df)} points, {len(track_df['track_id'].unique())} tracks",
    )
    elev_dir = os.path.join(output_dir, "elevation_analysis")
    os.makedirs(elev_dir, exist_ok=True)

    track_df = track_df.copy()
    # if "month" not in track_df.columns:
    #     track_df["time"] = pd.to_datetime(track_df["time"])
    #     track_df["month"] = track_df["time"].dt.strftime("%B")
    # std_print(
    #     "DEBUG", "ev", "Track months", ", ".join(track_df["month"].dropna().unique())
    # )
    try:
        with rasterio.open(vrt_path) as src:
            std_print(
                "DEBUG",
                "ev",
                "VRT extent",
                f"E:{src.bounds.left:.1f}°-{src.bounds.right:.1f}°, "
                f"N:{src.bounds.bottom:.1f}°-{src.bounds.top:.1f}°",
            )
            # Read elevation at reduced resolution to avoid memory issues
            scale_factor = 0.05
            out_shape = (int(src.height * scale_factor), int(src.width * scale_factor))
            std_print(
                "INFO",
                "ev",
                "Reading elevation mosaic",
                f"{out_shape[1]}x{out_shape[0]} pixels",
            )
            elev_data = src.read(
                1, out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear
            ).astype(float)

            # Clean invalid data (SRTM void value)
            elev_data[elev_data < -50] = 0

            # Figure
            fig, ax = plt.subplots(figsize=(13, 10), dpi=100)
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
            std_print("INFO", "ev", "Plotting tracks")
            for month_name, color in MONTH_COLORS.items():
                month_subset = track_df[track_df["month"] == month_name]
                if not month_subset.empty:
                    for tid in month_subset["track_id"].unique():
                        track = month_subset[
                            month_subset["track_id"] == tid
                        ].sort_values("time")
                        label = f"MD Track ({month_name})"
                        ax.plot(
                            track["longitude"].values,
                            track["latitude"].values,
                            color=color,
                            linewidth=2.0,
                            alpha=0.9,
                            label=label if label else None,
                            zorder=10,
                        )

            cbar = plt.colorbar(im, fraction=0.03, pad=0.04)
            cbar.set_label("Elevation (m)", fontweight="bold", fontsize=11)
            title = f"MD Track {track_id} Over SRTM Elevation"
            ax.set_xlabel("Longitude (°E)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Latitude (°N)", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5, zorder=1)

            # Legend
            legend_handles, labels = ax.get_legend_handles_labels()
            if legend_handles:
                by_label = dict(zip(labels, legend_handles))
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    # handles=legend_handles,
                    loc="lower left",
                    **legend_style(),
                )

            ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
            plt.tight_layout()

            # Save
            output_filename = f"MD_track_{track_id}_elevation_mosaic.png"
            output_path = os.path.join(elev_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            std_print("SAVED", "ev", "Elevation visualization", output_path)
            return output_path
    except Exception as e:
        std_print("ERROR", "ev", "Elevation visualization failed", str(e))
        traceback.print_exc()
        return None


# def plot_tracks_elevation_mosaic(csv_path, vrt_path, output_path):
#     """Use plot_tracks_on_elevation() instead for integrated usage"""
#     try:
#         track_df = pd.read_csv(csv_path)
#         track_df["time"] = pd.to_datetime(track_df["time"])
#         track_df["month"] = track_df["time"].dt.strftime("%B")
#         std_print(
#             "INFO",
#             "ev",
#             f"Loaded{len(track_df)}track points",
#             f"{track_df['month'].nunique()}months",
#         )
#         with rasterio.open(vrt_path) as src:
#             scale_factor = 0.05
#             out_shape_width = int(src.width * scale_factor)
#             out_shape_height = int(src.height * scale_factor)
#             out_shape = [out_shape_height, out_shape_width]
#             std_print(
#                 "INFO",
#                 "ev",
#                 "Reading elevation",
#                 f"{out_shape_width}x{out_shape_height}",
#             )
#             elev_data = src.read(
#                 1, out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear
#             ).astype(float)
#             elev_data[elev_data < -50] = 0
#
#             # Figure
#             fig, ax = plt.subplots(figsize=(13, 10), dpi=100)
#             # Calculate colormap limits (clip at 95th percentile to avoid washout)
#             vmin, vmax = np.nanpercentile(elev_data, [1, 95])
#             vmin = max(vmin, 0.1)
#             im = ax.imshow(
#                 elev_data + 0.1,
#                 cmap="terrain",
#                 norm=LogNorm(vmin=vmin, vmax=vmax),
#                 extent=[
#                     src.bounds.left,
#                     src.bounds.right,
#                     src.bounds.bottom,
#                     src.bounds.top,
#                 ],
#                 origin="upper",
#                 zorder=1,
#             )
#             for month_name, color in MONTH_COLORS.items():
#                 month_subset = track_df[track_df["month"] == month_name]
#                 if not month_subset.empty:
#                     for tid in month_subset["track_id"].unique():
#                         track = month_subset[month_subset["track_id"] == tid]
#                         label = (
#                             f"MD Track ({month_name})"
#                             if tid == month_subset["track_id"].iloc[0]
#                             else ""
#                         )
#                         ax.plot(
#                             track["longitude"],
#                             track["latitude"],
#                             color=color,
#                             linewidth=2.0,
#                             alpha=0.9,
#                             label=label if label else None,
#                             zorder=10,
#                         )
#             cb = plt.colorbar(im, fraction=0.03, pad=0.04)
#             cb.set_label("Elevation (m)", fontweight="bold")
#             ax.set_xlabel("Longitude (°E)", fontsize=12, fontweight="bold")
#             ax.set_ylabel("Latitude (°N)", fontsize=12, fontweight="bold")
#             ax.grid(True, alpha=0.3, linestyle="--", zorder=1)
#
#             # Legend
#             legend_handles, labels = ax.get_legend_handles_labels()
#             if legend_handles:
#                 by_label = dict(zip(labels, legend_handles))
#                 ax.legend(
#                     by_label.values(),
#                     by_label.keys(),
#                     # handles=legend_handles,
#                     loc="lower left",
#                     **legend_style(),
#                 )
#             ax.set_title("SRTM Elevation", fontsize=18, fontweight="bold")
#
#             plt.tight_layout()
#             plt.savefig(output_path, dpi=300, bbox_inches="tight")
#             plt.close(fig)
#             std_print("SAVED", "ev", "Elevation mosaic plot", output_path)
#             return output_path
#     except Exception as e:
#         std_print("ERROR", "ev", "Failed `plot_tracks_elevation_mosaic`", str(e))
#         return None
