"""MD Tracks on SRTM Elevation Mosaic"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import subprocess
import rasterio
from rasterio.merge import merge as rio_merge
import traceback

from utils import std_print, common_bbox, legend_style

month_colors = {
    "June": "#FF6B6B",
    "July": "#FF4500",
    "August": "#1E90FF",
    "September": "#FF00FF",
}

# TODO: make clipping global so every modules can use `full_extent`
# Use clipped domain instead of full dataset bounds
clipped_min_lat = 7.0  # N
clipped_max_lat = 30.0  # N
clipped_min_lon = 70.0  # E
clipped_max_lon = 95.0  # E
full_extent = [clipped_min_lon, clipped_max_lon, clipped_min_lat, clipped_max_lat]


def create_elevation_vrt(hgt_dir, vrt_filename="elevation_mosaic.vrt"):
    """Build Virtual Raster (VRT) from all HGT files in directory

    1. Return existing VRT if already built (fast path)
    2. gdalbuildvrt command-line tool
    3. rasterio.merge written to a plain GeoTIFF mosaic (fallback)
    """
    search_path = os.path.join(hgt_dir, "**/*.hgt")
    hgt_files = glob.glob(search_path, recursive=True)
    if not hgt_files:
        std_print("WARN", "ev", "No HGT files found", hgt_dir)
        return None
    std_print("INFO", "ev", "HGT files/tiles found", len(hgt_files))

    if os.path.exists(vrt_filename):
        std_print("INFO", "ev", "Using cached VRT", vrt_filename)
        return vrt_filename

    try:
        # Build VRT using gdalbuildvrt
        result = subprocess.run(
            ["gdalbuildvrt", "-overwrite", vrt_filename] + hgt_files,
            check=True,
            capture_output=True,
            text=True,
        )
        # std_print("DEBUG", "ev", "Elevation HGT files to convert to VRT", result)

        # Check if file was created
        if os.path.exists(vrt_filename):
            std_print("INFO", "ev", "VRT built via gdalbuildvrt", vrt_filename)
            return vrt_filename
        else:
            std_print("ERROR", "ev", "gdalbuildvrt ran but VRT not created")
            return None
    except FileNotFoundError:
        std_print(
            "ERROR",
            "ev",
            "gdalbuildvrt not found",
            "Install via `sudo apt install gdal-bin`",
        )
    except subprocess.CalledProcessError as e:
        std_print("ERROR", "ev", "VRT build failed", e.stderr)
    except Exception as e:
        std_print("ERROR", "ev", "Unexpected error", str(e))

    # rasterio merge -> GeoTIFF (If GDAL CLI not present)
    try:
        mosaic_path = vrt_filename.replace(".vrt", "_mosaic.tif")
        std_print(
            "INFO",
            "ev",
            "Building mosaic with rasterio",
            f"{len(hgt_files)} tiles -> {mosaic_path}",
        )
        datasets = [rasterio.open(f) for f in hgt_files]
        mosaic, out_transform = rio_merge(datasets)
        out_meta = datasets[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            }
        )
        with rasterio.open(mosaic_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        for ds in datasets:
            ds.close()
        std_print("INFO", "ev", "Mosaic GeoTIFF created", mosaic_path)
        return mosaic_path
    except Exception as e:
        std_print("ERROR", "ev", "rasterio merge fallback failed", str(e))
        return None


def plot_tracks_elevation(track_df, track_id, vrt_path, output_dir):
    """Elevation mosaic with MD tracks overlaid"""
    track_id = int(track_id)
    if track_df is None or track_df.empty:
        std_print("WARN", "ev", "No tracks to visualize")
        return None
    std_print("INFO", "ev", f"Input track {track_id} points", len(track_df))
    std_print(
        "INFO",
        "ev",
        f"Input track {track_id} length",
        len(track_df["track_id"].unique()),
    )
    plot_dir = os.path.join(output_dir, f"track_{track_id}", "elevation_analysis")
    os.makedirs(plot_dir, exist_ok=True)

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
            fig = plt.figure(figsize=(13, 10), dpi=100)
            ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]
            ax.set_extent(full_extent, crs=ccrs.PlateCarree())

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
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()

            track_data = track_df.sort_values("time")
            track_times = pd.to_datetime(track_df["time"])
            month = track_data["month"].iloc[0]
            track_color = month_colors.get(month, "black")
            bbox_style = common_bbox()

            # Start/End (for annotatation)
            start_point = track_data.iloc[0]
            end_point = track_data.iloc[-1]
            start_datetime = pd.to_datetime(start_point["time"])
            end_datetime = pd.to_datetime(end_point["time"])
            start_time_str = start_datetime.strftime("%d/%m %H:%M")
            end_time_str = end_datetime.strftime("%d/%m %H:%M")
            start_lon = start_point["longitude"]
            end_lon = end_point["longitude"]
            start_lat = start_point["latitude"]
            end_lat = end_point["latitude"]

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

            # # Plot tracks by month
            # for month_name, track_color in month_colors.items():
            #     month_subset = track_df[track_df["month"] == month_name]
            #     if not month_subset.empty:
            #         for track_id in month_subset["track_id"].unique():
            #             track = month_subset[
            #                 month_subset["track_id"] == track_id
            #             ].sort_values("time")
            #             label = f"MD Track ({month_name})"
            #             ax.plot(
            #                 track["longitude"].values,
            #                 track["latitude"].values,
            #                 color=track_color,
            #                 linewidth=1.0,
            #                 alpha=0.9,
            #                 label=label if label else None,
            #                 zorder=10,
            #             )

            # Plot
            track_ids = sorted(track_df["track_id"].unique())
            for tracl_id in track_ids:
                # track = track_df[track_df["track_id"] == track_id].sort_values("time")
                # Connecting lines
                ax.plot(
                    track_df["longitude"],
                    track_df["latitude"],
                    "-o",
                    color=track_color,
                    transform=ccrs.PlateCarree(),
                    linewidth=1.5,
                    solid_capstyle="round",
                    markersize=2,
                    # alpha=0.8,
                    label=f"MD Track {track_id} ({month})",
                    zorder=2,
                )

                # 12-hour interval markers
                hours_elapsed = (track_times - start_datetime).dt.total_seconds() / 3600
                for elapsed_hours in np.arange(0, hours_elapsed.max() + 12, 12):
                    idx = np.argmin(np.abs(hours_elapsed - elapsed_hours))
                    if idx < len(track_df):
                        point = track_df.iloc[idx]
                        ax.plot(
                            point["longitude"],
                            point["latitude"],
                            "o",
                            color=track_color,
                            solid_capstyle="round",
                            markersize=4,
                            transform=ccrs.PlateCarree(),
                            zorder=5,
                        )
                        point_time = pd.to_datetime(point["time"])
                        time_label = f"{point_time.strftime('%d/%m %H:%M')}"
                        ax.text(
                            point["longitude"] - 0.1,
                            point["latitude"] + 0.1,
                            time_label,
                            transform=ccrs.PlateCarree(),
                            fontsize=5,
                            fontweight="bold",
                            color="black",
                            ha="left",
                            va="bottom",
                            bbox=bbox_style,
                            zorder=7,
                        )

            # Annotate Start point
            ax.plot(
                start_lon,
                start_lat,
                marker="o",
                color="green",
                solid_capstyle="round",
                markersize=3,
                markeredgewidth=5,
                transform=ccrs.PlateCarree(),
                zorder=8,
            )
            # Annotate End point
            ax.plot(
                end_lon,
                end_lat,
                marker="o",
                color="orange",
                solid_capstyle="round",
                markersize=3,
                markeredgewidth=5,
                transform=ccrs.PlateCarree(),
                zorder=8,
            )

            cbar = plt.colorbar(im, fraction=0.03, pad=0.04)
            cbar.set_label("Elevation (m)", fontsize=11, fontweight="bold")

            # Legend
            legend_handles, labels = ax.get_legend_handles_labels()
            if legend_handles:
                ax.legend(handles=legend_handles, loc="lower left", **legend_style())

            title = f"MD Track {track_id} over SRTM Elevation \n{start_time_str} to {end_time_str}"  # , {MONSOON_YEAR}"
            ax.set_title(title, fontsize=13, fontweight="bold")
            plt.tight_layout()

            # Save
            output_filename = f"track_{track_id}_elevation_mosaic.png"
            output_path = os.path.join(plot_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            std_print("SAVED", "ev", "Elevation visualization", output_path)
            return output_path
    except Exception as e:
        std_print("ERROR", "ev", "Elevation visualization failed", str(e))
        traceback.print_exc()
        return None


def plot_all_tracks_elevation(track_df, vrt_path, output_dir, year):
    """Plot all MD tracks on a single elevation mosaic"""
    if track_df is None or track_df.empty:
        std_print("WARN", "ev", "No tracks to visualize")
        return None

    std_print("INFO", "ev", "Total tracks", len(track_df["track_id"].unique()))
    std_print("INFO", "ev", "Total points", len(track_df))

    plot_dir = os.path.join(output_dir, "elevation_analysis")
    os.makedirs(plot_dir, exist_ok=True)
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
            fig = plt.figure(figsize=(13, 10), dpi=100)
            ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]
            ax.set_extent(full_extent, crs=ccrs.PlateCarree())

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
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()

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

            # Plot
            track_ids = sorted(track_df["track_id"].unique())
            for track_id in track_ids:
                track = track_df[track_df["track_id"] == track_id].sort_values("time")
                month = track["month"].iloc[0]
                track_color = month_colors.get(month, "black")

                # Connecting lines
                ax.plot(
                    track["longitude"].values,
                    track["latitude"].values,
                    "-o",
                    color=track_color,
                    transform=ccrs.PlateCarree(),
                    linewidth=1.5,
                    solid_capstyle="round",
                    markersize=2,
                    # alpha=0.8,
                    label=f"MD Track {track_id} ({month})",
                    zorder=2,
                )

            cbar = plt.colorbar(im, fraction=0.03, pad=0.04)
            cbar.set_label("Elevation (m)", fontsize=11, fontweight="bold")

            # Legend
            legend_handles, labels = ax.get_legend_handles_labels()
            if legend_handles:
                ax.legend(handles=legend_handles, loc="lower left", **legend_style())

            title = f"Monsoon Depression Tracks ({year}) over SRTM Elevation"
            ax.set_title(title, fontsize=13, fontweight="bold")
            plt.tight_layout()

            # Save
            output_filename = f"All_tracks_elevation_mosaic_{year}.png"
            output_path = os.path.join(plot_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            std_print("SAVED", "ev", "All tracks elevation map", output_path)
            return output_path
    except Exception as e:
        std_print("ERROR", "ev", "Elevation visualization failed", str(e))
        traceback.print_exc()
        return None


# def plot_tracks_elevation_mosaic(csv_path, vrt_path, output_path):
#     """Use plot_tracks_elevation() instead for integrated usage"""
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
#             for month_name, color in month_colors.items():
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
#                             linewidth=1.0,
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
#                 ax.legend(handles=legend_handles, loc="lower left", **legend_style())
#
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
