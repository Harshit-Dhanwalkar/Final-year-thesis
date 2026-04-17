"""
python plot_jjas_climatology.py
python plot_jjas_climatology.py --data_dir /path/to/era5/ncfiles
python plot_jjas_climatology.py --vrt elevation_mosaic.vrt --tracks_root .
python3 plot_jjas_climatology.py \
    --data_dir ../Data/single-levels/ncfiles \
    --tracks_root . \
    --vrt Overall_years/elevation_mosaic.vrt
"""

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
from scipy.stats import gaussian_kde
from skimage import measure

from utils import std_print

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")

# Output
OUTPUT_DIR = "Overall_years"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "india_jjas_climatology.png")

# Map domain (lon_min, lon_max, lat_min, lat_max)
clipped_min_lat = 7.0
clipped_max_lat = 30.0
clipped_min_lon = 70.0
clipped_max_lon = 95.0
FULL_EXTENT = [clipped_min_lon, clipped_max_lon, clipped_min_lat, clipped_max_lat]

JJAS_MONTHS = [6, 7, 8, 9]  # June July August September


def _time_coord(ds):
    """Return the name of the time coordinate in an ERA5 dataset"""
    for candidate in ("valid_time", "time"):
        if candidate in ds.coords or candidate in ds:
            return candidate
    raise KeyError(
        "Cannot find a time coordinate in the dataset. "
        f"Available coordinates: {list(ds.coords)}"
    )


def load_jjas_mean(data_root):
    # Pattern: year/month-*/*accum*.nc  (only accumulation files)
    pattern = os.path.join(data_root, "*", "*-*", "*accum*.nc")
    nc_files = sorted(glob.glob(pattern))
    if not nc_files:
        raise FileNotFoundError(
            f"No *accum*.nc files found in {data_root}/YYYY/Month-YYYY/. "
            "Make sure your ERA5 data is stored as ../Data/single-levels/ncfiles/2001/June-2001/*accum*.nc etc."
        )
    std_print("INFO", "pc", f"Found {len(nc_files)} accumulation NetCDF file(s)")

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

    tp = ds["tp"]
    # Convert from m to mm, and keep only JJAS months
    tp_jjas = tp.sel(time=tp["time"].dt.month.isin([6, 7, 8, 9]))
    if tp_jjas.time.size == 0:
        raise ValueError("No JJAS data after month filtering.")

    years = np.unique(tp_jjas["time"].dt.year.values).tolist()
    std_print("INFO", "pc", "JJAS data years", f"{years[0]}–{years[-1]} ({len(years)})")

    annual_totals = []
    for yr in years:
        yr_data = tp_jjas.sel(time=tp_jjas["time"].dt.year == yr)
        total_m = yr_data.sum(dim="time", skipna=True)
        annual_totals.append(total_m)

    stacked = xr.concat(annual_totals, dim="year").assign_coords(year=years)
    mean_m = stacked.mean(dim="year", skipna=True)
    mean_mm = mean_m * 1000.0
    mean_mm.attrs["long_name"] = "Mean JJAS total precipitation"
    mean_mm.attrs["units"] = "mm season⁻¹"

    ds.close()
    return mean_mm, years


# Helper function
def rename_time(ds):
    """Preprocess callback: rename 'valid_time' -> 'time' if present"""
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    # Ensure the time coordinate is decoded as datetime
    if "time" in ds.coords:
        ds["time"] = ds["time"].astype("datetime64[ns]")
    return ds


# Elevation contour (SRTM)
def get_elevation_contour_from_vrt(vrt_path, contour_m=600):
    """Read SRTM VRT and extract the 600 m contour vertices as arrays"""
    try:
        with rasterio.open(vrt_path) as src:
            win = from_bounds(
                left=FULL_EXTENT[0] - 1,
                bottom=FULL_EXTENT[2] - 1,
                right=FULL_EXTENT[1] + 1,
                top=FULL_EXTENT[3] + 1,
                transform=src.transform,
            )
            # Read at 10% of original resolution
            out_shape = (1, int(src.height // 10), int(src.width // 10))
            elev = src.read(
                1, out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear
            )
            transform = src.transform * src.transform.scale(
                (src.width / elev.shape[1]), (src.height / elev.shape[0])
            )
            tf = src.window_transform(win)

        # marching squares -> list of contour paths at contour_m
        contours = measure.find_contours(elev.astype(float), level=contour_m)
        if not contours:
            return None, None

        # Convert pixel (row, col) -> geographic lon, lat
        all_lons, all_lats = [], []
        for path in contours:
            rows, cols = path[:, 0], path[:, 1]
            lons = tf.c + cols * tf.a
            lats = tf.f + rows * tf.e
            all_lons.append(lons)
            all_lats.append(lats)
        return all_lons, all_lats
    except ImportError as e:
        std_print("ERROR", "pc", "Skipping 600 m contour", str(e))
        return None, None
    except Exception as e:
        std_print("ERROR", "pc", "Could not read VRT for elevation contour", str(e))
        return None, None


# MD track density
def load_all_tracks(output_root):
    """Load MD track CSVs from output_YYYY directories (same logic as plot_all_years.py)."""
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
            if {"latitude", "longitude", "time"}.issubset(df.columns):
                df["year"] = year
                frames.append(df[["latitude", "longitude", "time", "year"]])
        except Exception:
            pass
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["latitude", "longitude", "year"])


# Main plotting function
def plot_jjas_climatology(mean_mm, years, vrt_path, tracks_df, output_path):
    """Generate JJAS precipitation climatology map"""
    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(FULL_EXTENT, crs=ccrs.PlateCarree())

    # Background features
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#C8E8F5", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F5F0E8", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, zorder=5)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.8,
        linestyle="-",
        edgecolor="0.3",
        zorder=5,
    )
    ax.add_feature(
        cfeature.STATES.with_scale("10m"),
        linewidth=0.35,
        linestyle="--",
        edgecolor="0.55",
        zorder=5,
    )
    ax.add_feature(
        cfeature.RIVERS.with_scale("50m"),
        linewidth=0.3,
        edgecolor="#6baed6",
        alpha=0.6,
        zorder=4,
    )

    # Precipitation shading
    # Detect lat/lon dimension names (ERA5 uses 'latitude'/'longitude' or 'lat'/'lon')
    lat_name = "latitude" if "latitude" in mean_mm.dims else "lat"
    lon_name = "longitude" if "longitude" in mean_mm.dims else "lon"

    lats = mean_mm[lat_name].values
    lons = mean_mm[lon_name].values
    data = mean_mm.values

    # Clip the domain to the map extent
    lat_mask = (lats >= FULL_EXTENT[2]) & (lats <= FULL_EXTENT[3])
    lon_mask = (lons >= FULL_EXTENT[0]) & (lons <= FULL_EXTENT[1])
    lats_c = lats[lat_mask]
    lons_c = lons[lon_mask]
    # handle both (lat,lon) and (lon,lat) array layouts
    if data.shape == (len(lats), len(lons)):
        data_c = data[np.ix_(lat_mask, lon_mask)]
    else:
        data_c = data[np.ix_(lon_mask, lat_mask)].T

    lons, lats = np.meshgrid(lons_c, lats_c)

    # Colour levels: 0–3000 mm/season in steps; extend beyond 3000
    levels = np.concatenate(
        [
            np.arange(0, 500, 50),  #   0 – 450 mm (semi-arid)
            np.arange(500, 1500, 100),  # 500 – 1400 mm (sub-humid)
            np.arange(1500, 3000, 200),  # 1500 – 2800 mm (humid)
            np.arange(3000, 5001, 500),  # 3000 – 5000 mm (very wet coasts)
        ]
    )

    cmap = plt.cm.YlGnBu
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    pcm = ax.pcolormesh(
        lons,
        lats,
        data_c,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        zorder=1,
        shading="auto",
    )

    # Overlay smoothed contour lines
    ct = ax.contour(
        lons,
        lats,
        data_c,
        levels=[200, 500, 1000, 1500, 2000, 3000],
        colors=["white"],
        linewidths=[0.4],
        alpha=0.55,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.clabel(ct, fmt="%d mm", fontsize=6.5, inline=True, inline_spacing=4)

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
    cbar.set_label("Mean JJAS precipitation (mm season⁻¹)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Eastern Ghats 600 m contour
    if vrt_path and os.path.exists(vrt_path):
        std_print("INFO", "pc", "Extracting Eastern Ghats 600 m contour from SRTM VRT")
        cont_lons, cont_lats = get_elevation_contour_from_vrt(vrt_path, 600)
        if cont_lons is not None:
            for cl, la in zip(cont_lons, cont_lats):
                # only draw segments in the Eastern Ghats longitude band
                in_band = (cl >= 78) & (cl <= 88) & (la >= 11) & (la <= 24)
                if np.sum(in_band) > 5:
                    ax.plot(
                        cl[in_band],
                        la[in_band],
                        color="saddlebrown",
                        linewidth=1.4,
                        alpha=0.85,
                        transform=ccrs.PlateCarree(),
                        zorder=6,
                        label="_nolegend_",
                    )
            std_print("INFO", "pc", "Eastern Ghats 600 m contour drawn.")

    # MD track density
    if tracks_df is not None and len(tracks_df) > 0:
        try:
            lons_t = tracks_df["longitude"].values.astype(float)
            lats_t = tracks_df["latitude"].values.astype(float)
            mask = (
                (lons_t >= FULL_EXTENT[0])
                & (lons_t <= FULL_EXTENT[1])
                & (lats_t >= FULL_EXTENT[2])
                & (lats_t <= FULL_EXTENT[3])
            )
            lons_t, lats_t = lons_t[mask], lats_t[mask]

            # KDE on a 0.5° grid
            grid_lon = np.arange(FULL_EXTENT[0], FULL_EXTENT[1] + 0.5, 0.5)
            grid_lat = np.arange(FULL_EXTENT[2], FULL_EXTENT[3] + 0.5, 0.5)
            GL, GA = np.meshgrid(grid_lon, grid_lat)
            pts = np.vstack([lons_t, lats_t])
            kde = gaussian_kde(pts, bw_method=0.15)
            density = kde(np.vstack([GL.ravel(), GA.ravel()])).reshape(GL.shape)
            density = density / density.max()

            ax.contourf(
                GL,
                GA,
                density,
                levels=np.linspace(0.15, 1.0, 8),
                cmap="Reds",
                alpha=0.18,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
            std_print("INFO", "pc", "MD track density overlay added.")
        except Exception as e:
            std_print("ERROR", "pc", "Could not compute track density", str(e))

    # Bay of Bengal label and landfall zone marker
    ax.text(
        88.5,
        14.0,
        "Bay of\nBengal",
        transform=ccrs.PlateCarree(),
        fontsize=9.5,
        color="#1a6696",
        ha="center",
        style="italic",
        zorder=8,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )
    # ax.text(
    #     66.5,
    #     20.0,
    #     "Arabian\nSea",
    #     transform=ccrs.PlateCarree(),
    #     fontsize=9.5,
    #     color="#1a6696",
    #     ha="center",
    #     style="italic",
    #     zorder=8,
    #     path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    # )

    # Eastern Ghats label
    ax.text(
        82.8,
        17.0,
        "Eastern\nGhats",
        transform=ccrs.PlateCarree(),
        fontsize=8.0,
        color="saddlebrown",
        ha="center",
        rotation=55,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        zorder=9,
    )

    # Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.45,
        linestyle=":",
        zorder=4,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(range(65, 101, 5))
    gl.ylocator = mticker.FixedLocator(range(5, 36, 5))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="#990000",
            lw=1.8,
            ls="--",
            dashes=(8, 4),
            label="Monsoon trough axis (mean Jul–Aug; Sikka 1977)",
        ),
        Line2D(
            [0],
            [0],
            color="saddlebrown",
            lw=1.6,
            label="Eastern Ghats 600 m contour (SRTM)",
        ),
        Line2D(
            [0],
            [0],
            color="darkorange",
            lw=2.5,
            label="Preferred MD landfall zone (Odisha–Bengal coast)",
        ),
    ]
    if tracks_df is not None and len(tracks_df) > 0:
        legend_elements.append(
            Patch(facecolor="red", alpha=0.2, label="MD track density (2001–2025)")
        )
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=8,
        framealpha=0.88,
        edgecolor="0.4",
        facecolor="white",
    )

    # Title
    yr_str = f"{years[0]}–{years[-1]}" if years else "2001–2025"
    ax.set_title(
        f"ERA5 Mean JJAS Precipitation ({yr_str})\n"
        r"$\overline{P}_{\rm JJAS}$ [mm season$^{-1}$] (June–September)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "pc", "JJAS precipitation plot", output_path)


# CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot ERA5 JJAS mean precipitation climatology over India"
    )
    p.add_argument(
        "--data_dir",
        default=os.path.join("..", "Data", "single-levels", "ncfiles"),
        help="Directory containing ERA5 single-level *.nc files with 'tp' variable "
        "(default: ../Data/single-levels/ncfiles)",
    )
    p.add_argument(
        "--vrt",
        default=os.path.join("Overall_years", "elevation_mosaic.vrt"),
        help="Path to SRTM elevation VRT file for 600 m contour "
        "(default: Overall_years/elevation_mosaic.vrt)",
    )
    p.add_argument(
        "--tracks_root",
        default=".",
        help="Root directory containing output_YYYY folders for track density "
        "overlay (default: . ; pass 'none' to skip)",
    )
    p.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help=f"Output PNG path (default: {OUTPUT_FILE})",
    )
    p.add_argument(
        "--no_tracks",
        action="store_true",
        help="Skip MD track density overlay",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Load ERA5 precipitation data
    std_print("INFO", "pc", "Loading ERA5 tp", args.data_dir)
    try:
        mean_mm, years = load_jjas_mean(args.data_dir)
    except (FileNotFoundError, KeyError) as e:
        std_print("ERROR", "pc", "Loading jjas mean", str(e))
        sys.exit(1)

    std_print(
        "INFO",
        "pc",
        f"Climatology computed over {len(years)} years ({years[0]}–{years[-1]})",
    )
    std_print(
        "INFO",
        "pc",
        f"Precipitation range"
        f"{float(mean_mm.min()):.1f} – {float(mean_mm.max()):.1f} mm/season",
    )

    # Load SRTM VRT
    vrt = args.vrt if os.path.exists(args.vrt) else None
    if vrt is None:
        std_print(
            "WARN", "pc", f"VRT not found at '{args.vrt}' - using fallback Ghats line"
        )

    # Load MD tracks
    tracks_df = None
    if not args.no_tracks and args.tracks_root.lower() != "none":
        std_print("INFO", "pa", f"Loading MD tracks from: {args.tracks_root}")
        tracks_df = load_all_tracks(args.tracks_root)
        if len(tracks_df) == 0:
            std_print("WARN", "pc", "No track data found - skipping density overlay")
            tracks_df = None
        else:
            std_print("INFO", "pa", f"{len(tracks_df)} track points loaded")

    # Plot
    std_print("INFO", "pa", "Generating figure", args.output)
    plot_jjas_climatology(
        mean_mm=mean_mm,
        years=years,
        vrt_path=vrt,
        tracks_df=tracks_df,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
