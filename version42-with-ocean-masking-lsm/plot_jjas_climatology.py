# NOTE:
# python plot_jjas_climatology.py
# python plot_jjas_climatology.py --data_dir /path/to/era5/ncfiles
# python plot_jjas_climatology.py --vrt elevation_mosaic.vrt --tracks_root .
# python3 plot_jjas_climatology.py \
#     --data_dir ../Data/single-levels/ncfiles \
#     --tracks_root . \
#     --vrt Overall_years/elevation_mosaic.vrt
#
# Fig : india_jjas_climatology.png         -> Chapter 1

import argparse
import glob
import os
import sys
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.windows import from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from scipy.stats import gaussian_kde
from skimage import measure

from utils import std_print

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Output
OUTPUT_DIR = "Thesis_schematics"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "india_jjas_climatology.png")

# Map domain
FULL_EXTENT = [70.0, 95.0, 7.0, 30.0]  # lon_min, lon_max, lat_min, lat_max
JJAS_MONTHS = [6, 7, 8, 9]

# Font sizes
FS_TITLE = 14
FS_LABEL = 12
FS_TICK = 10
FS_LEGEND = 10
FS_ANNOT = 9

# Approximate climatological monsoon trough axis (Sikka 1977)
TROUGH_LONS = [67.5, 72.0, 77.0, 81.5, 85.5, 88.5]
TROUGH_LATS = [28.2, 27.6, 25.8, 24.2, 22.6, 20.5]


# Data loading
def rename_time(ds):
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "time" in ds.coords:
        ds["time"] = ds["time"].astype("datetime64[ns]")
    return ds


def load_jjas_mean(data_root):
    """Load ERA5 tp files, filter JJAS"""
    pattern = os.path.join(data_root, "*", "*-*", "*accum*.nc")
    nc_files = sorted(glob.glob(pattern))
    if not nc_files:
        # Fallback: flat directory of .nc files
        nc_files = sorted(glob.glob(os.path.join(data_root, "*.nc")))
    if not nc_files:
        raise FileNotFoundError(
            f"No *.nc files found under '{data_root}'. "
            "Expected ../Data/single-levels/ncfiles/YYYY/Month-YYYY/*accum*.nc"
        )
    std_print("INFO", "pc", f"Found {len(nc_files)} NetCDF file(s)")

    ds = xr.open_mfdataset(
        nc_files,
        combine="by_coords",
        join="outer",
        compat="override",
        data_vars="minimal",
        coords="minimal",
        engine="netcdf4",
        chunks={"time": 24},
        preprocess=rename_time,
    )
    if "tp" not in ds:
        raise KeyError(f"Variable 'tp' not found. Available: {list(ds.data_vars)}")

    tp_jjas = ds["tp"].sel(time=ds["tp"]["time"].dt.month.isin(JJAS_MONTHS))
    if tp_jjas.time.size == 0:
        raise ValueError("No JJAS data after month filtering.")

    years = np.unique(tp_jjas["time"].dt.year.values).tolist()
    std_print("INFO", "pc", "JJAS years", f"{years[0]}–{years[-1]} ({len(years)} yrs)")

    annual_totals = [
        tp_jjas.sel(time=tp_jjas["time"].dt.year == yr).sum(dim="time", skipna=True)
        for yr in years
    ]
    stacked = xr.concat(annual_totals, dim="year").assign_coords(year=years)
    mean_mm = stacked.mean(dim="year", skipna=True) * 1000.0
    mean_mm.attrs.update(
        {"long_name": "Mean JJAS total precipitation", "units": "mm season⁻¹"}
    )
    ds.close()
    return mean_mm, years


# Elevation contour
def get_ghats_contour_from_vrt(vrt_path, contour_m=800):
    """Extract contour segments from SRTM VRT and convert to geographic coords"""
    try:
        with rasterio.open(vrt_path) as src:
            # Define the spatial window
            win = from_bounds(
                left=FULL_EXTENT[0] - 1,
                bottom=FULL_EXTENT[2] - 1,
                right=FULL_EXTENT[1] + 1,
                top=FULL_EXTENT[3] + 1,
                transform=src.transform,
            )
            # Downsample for performance
            h_out = max(1, int(win.height // 10))
            w_out = max(1, int(win.width // 10))
            elev = src.read(
                1,
                window=win,
                out_shape=(h_out, w_out),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            win_bounds = src.window_bounds(win)
            new_tf = transform_from_bounds(*win_bounds, width=w_out, height=h_out)

        # 4. Generate contours
        contours = measure.find_contours(elev.astype(float), level=contour_m)
        all_lons, all_lats = [], []
        for path in contours:
            # path is (row, col) array
            rows, cols = path[:, 0], path[:, 1]
            # Transform indices to (lon, lat) using the new transform
            # Affine * (col, row) returns (lon, lat)
            lons_, lats_ = new_tf * (cols, rows)
            # Geographic filtering (Eastern Ghats Band)
            mask = (lons_ >= 78) & (lons_ <= 88) & (lats_ >= 11) & (lats_ <= 22)
            if np.sum(mask) > 5:
                all_lons.append(lons_[mask])
                all_lats.append(lats_[mask])
        return (all_lons, all_lats) if all_lons else (None, None)

    except Exception as e:
        std_print("WARN", "pc", "VRT contour failed", str(e))
        return None, None


def fallback_ghats_line():
    """Approximate Eastern Ghats axis when VRT is unavailable"""
    lons = [88.0, 86.8, 85.6, 84.2, 82.8, 81.5, 80.0, 78.8, 78.0]
    lats = [21.5, 20.2, 18.8, 17.4, 16.0, 14.8, 13.5, 12.5, 11.5]
    return lons, lats


# MD track density
def load_all_tracks(output_root):
    frames = []
    for d in sorted(glob.glob(os.path.join(output_root, "output_*"))):
        try:
            year = int(os.path.basename(d).replace("output_", ""))
        except ValueError:
            continue
        csv = os.path.join(d, f"MD_tracks_{year}.csv")
        if not os.path.exists(csv):
            csv = os.path.join(d, "MD_coordinates_ALL.csv")
        if not os.path.exists(csv):
            continue
        try:
            df = pd.read_csv(csv)
            df.columns = [c.strip().lower() for c in df.columns]
            if {"latitude", "longitude"}.issubset(df.columns):
                df["year"] = year
                frames.append(df[["latitude", "longitude", "year"]])
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Main plotting functions
def plot_jjas_climatology(
    mean_mm, years, vrt_path=None, tracks_df=None, output_path=OUTPUT_FILE
):
    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(FULL_EXTENT, crs=ccrs.PlateCarree())

    # Background
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#C8E8F5", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F5F0E8", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8, zorder=5)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"), linewidth=0.8, edgecolor="0.3", zorder=5
    )
    ax.add_feature(
        cfeature.STATES.with_scale("10m"),
        linewidth=0.35,
        linestyle="--",
        edgecolor="0.5",
        zorder=5,
    )
    ax.add_feature(
        cfeature.RIVERS.with_scale("50m"),
        linewidth=0.3,
        edgecolor="#6baed6",
        alpha=0.55,
        zorder=4,
    )

    # Precipitation shading
    lat_name = "latitude" if "latitude" in mean_mm.dims else "lat"
    lon_name = "longitude" if "longitude" in mean_mm.dims else "lon"
    lats = mean_mm[lat_name].values
    lons = mean_mm[lon_name].values
    data = mean_mm.values

    lat_mask = (lats >= FULL_EXTENT[2]) & (lats <= FULL_EXTENT[3])
    lon_mask = (lons >= FULL_EXTENT[0]) & (lons <= FULL_EXTENT[1])
    lats_c = lats[lat_mask]
    lons_c = lons[lon_mask]
    data_c = (
        data[np.ix_(lat_mask, lon_mask)]
        if data.shape == (len(lats), len(lons))
        else data[np.ix_(lon_mask, lat_mask)].T
    )
    LONS, LATS = np.meshgrid(lons_c, lats_c)

    levels = np.concatenate(
        [
            np.arange(0, 500, 50),
            np.arange(500, 1500, 100),
            np.arange(1500, 3000, 200),
            np.arange(3000, 5001, 500),
        ]
    )
    cmap = plt.cm.YlGnBu
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    pcm = ax.pcolormesh(
        LONS,
        LATS,
        data_c,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        zorder=1,
        shading="auto",
    )

    ct = ax.contour(
        LONS,
        LATS,
        data_c,
        levels=[200, 500, 1000, 1500, 2000, 3000],
        colors=["black"],
        linewidths=[0.5],
        alpha=0.50,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.clabel(ct, fmt="%d mm", fontsize=FS_TICK - 1, inline=True, inline_spacing=4)

    # Colorbar
    cbar = fig.colorbar(
        pcm,
        ax=ax,
        orientation="vertical",
        fraction=0.025,
        pad=0.025,
        extend="max",
        ticks=[0, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
    )
    cbar.set_label(
        "Mean JJAS precipitation  (mm season\u207b\u00b9)", fontsize=FS_LABEL
    )
    cbar.ax.tick_params(labelsize=FS_TICK)

    # Eastern Ghats 800 m contour
    ghats_from_vrt = False
    if vrt_path and os.path.exists(vrt_path):
        cont_lons, cont_lats = get_ghats_contour_from_vrt(vrt_path, 800)
        if cont_lons:
            for cl, la in zip(cont_lons, cont_lats):
                ax.plot(
                    cl,
                    la,
                    color="#8c510a",
                    linewidth=1.6,
                    alpha=0.88,
                    transform=ccrs.PlateCarree(),
                    zorder=6,
                )
            std_print("INFO", "pc", "Eastern Ghats 800m contour drawn from VRT")
            ghats_from_vrt = True

    if not ghats_from_vrt:
        gl_lons, gl_lats = fallback_ghats_line()
        ax.plot(
            gl_lons,
            gl_lats,
            color="#8c510a",
            linewidth=1.8,
            alpha=0.85,
            linestyle="-",
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        std_print("WARN", "pc", "Using approximate Eastern Ghats axis line")

    # Monsoon trough axis
    ax.plot(
        TROUGH_LONS,
        TROUGH_LATS,
        color="#7f0000",
        linewidth=2.0,
        linestyle="--",
        dashes=(9, 4),
        alpha=0.92,
        transform=ccrs.PlateCarree(),
        zorder=7,
    )

    # MD track density
    has_density = False
    if tracks_df is not None and len(tracks_df) > 0:
        try:
            lt = tracks_df["latitude"].values.astype(float)
            ln = tracks_df["longitude"].values.astype(float)
            mask = (
                (ln >= FULL_EXTENT[0])
                & (ln <= FULL_EXTENT[1])
                & (lt >= FULL_EXTENT[2])
                & (lt <= FULL_EXTENT[3])
            )
            lt, ln = lt[mask], ln[mask]
            gl = np.arange(FULL_EXTENT[0], FULL_EXTENT[1] + 0.5, 0.5)
            ga = np.arange(FULL_EXTENT[2], FULL_EXTENT[3] + 0.5, 0.5)
            GL, GA = np.meshgrid(gl, ga)
            kde = gaussian_kde(np.vstack([ln, lt]), bw_method=0.15)
            density = kde(np.vstack([GL.ravel(), GA.ravel()])).reshape(GL.shape)
            density /= density.max()
            ax.contourf(
                GL,
                GA,
                density,
                levels=np.linspace(0.15, 1.0, 8),
                cmap="Reds",
                alpha=0.20,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
            has_density = True
            std_print("INFO", "pc", "MD track density overlay added")
        except Exception as e:
            std_print("WARN", "pc", "Track density failed", str(e))

    # Text labels
    path_effects = [pe.withStroke(linewidth=2.5, foreground="white")]
    ax.text(
        89.0,
        14.5,
        "Bay of\nBengal",
        transform=ccrs.PlateCarree(),
        fontsize=FS_ANNOT + 1,
        color="#1a6696",
        ha="center",
        style="italic",
        zorder=8,
        path_effects=path_effects,
    )
    ax.text(
        72.0,
        10.5,
        "Arabian\nSea",
        transform=ccrs.PlateCarree(),
        fontsize=FS_ANNOT + 1,
        color="#1a6696",
        ha="center",
        style="italic",
        zorder=8,
        path_effects=path_effects,
    )
    ax.text(
        84.0,
        17.5,
        "Eastern\nGhats",
        transform=ccrs.PlateCarree(),
        fontsize=FS_ANNOT,
        color="#8c510a",
        ha="center",
        rotation=55,
        fontweight="bold",
        zorder=9,
        path_effects=path_effects,
    )
    ax.text(
        80.0,
        25.5,
        "Monsoon Trough",
        transform=ccrs.PlateCarree(),
        fontsize=FS_ANNOT,
        color="#7f0000",
        ha="center",
        style="italic",
        zorder=9,
        path_effects=path_effects,
    )

    # Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.35,
        color="gray",
        alpha=0.50,
        linestyle=":",
        zorder=4,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(70, 96, 5))
    gl.ylocator = mticker.FixedLocator(range(5, 31, 5))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": FS_TICK}
    gl.ylabel_style = {"size": FS_TICK}

    # legend
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#7f0000",
            linewidth=2.0,
            linestyle="--",
            dashes=(9, 4),
            label="Monsoon trough axis (mean Jul–Aug; Sikka 1977)",
        ),
        Line2D(
            [0],
            [0],
            color="#8c510a",
            linewidth=1.8,
            label="Eastern Ghats 800m contour (SRTM)",
        ),
        Line2D(
            [0],
            [0],
            color="darkorange",
            linewidth=3.0,
            label="Preferred MD landfall zone (Odisha–Bengal coast)",
        ),
    ]
    # if has_density:
    #     legend_handles.append(
    #         Patch(
    #             facecolor="red",
    #             alpha=0.22,
    #             label=f"MD track density, {year[0]}–{years[-1]}",
    #         )
    #     )

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=FS_LEGEND,
        framealpha=0.8,
        facecolor="white",
        borderpad=0.7,
        handlelength=2.2,
    )

    # Title
    yr_str = f"{years[0]}-{years[-1]}" if years else "2001-2025"
    ax.set_title(
        # "$\overline{P}_{\rm JJAS}$  [mm season$^{-1}$]"
        f"ERA5 Mean JJAS Precipitation (June–September, {yr_str})",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=10,
    )

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    std_print("SAVED", "pc", "JJAS climatology", output_path)


# CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="ERA5 JJAS precipitation climatology for thesis Figure 1.1"
    )
    p.add_argument(
        "--data_dir", default=os.path.join("..", "Data", "single-levels", "ncfiles")
    )
    p.add_argument(
        "--vrt", default=os.path.join("Overall_years", "elevation_mosaic.vrt")
    )
    p.add_argument("--tracks_root", default=".")
    p.add_argument("--output", default=OUTPUT_FILE)
    p.add_argument("--no_tracks", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    std_print("INFO", "pc", "Loading ERA5 tp", args.data_dir)
    try:
        mean_mm, years = load_jjas_mean(args.data_dir)
    except (FileNotFoundError, KeyError, ValueError) as e:
        std_print("ERROR", "pc", str(e))
        sys.exit(1)

    vrt = args.vrt if os.path.exists(args.vrt) else None
    tracks_df = None
    if not args.no_tracks and args.tracks_root.lower() != "none":
        tracks_df = load_all_tracks(args.tracks_root)
        if tracks_df.empty:
            tracks_df = None
            std_print("WARN", "pc", "No track data — skipping density overlay")

    plot_jjas_climatology(
        mean_mm, years, vrt_path=vrt, tracks_df=tracks_df, output_path=args.output
    )


if __name__ == "__main__":
    main()
