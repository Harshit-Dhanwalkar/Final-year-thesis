# NOTE:
#   1. Plain map  — tracks over land/ocean cartopy base map, coloured by year
#   2. Elevation  — same tracks over SRTM elevation mosaic (requires VRT or HGT files)
#
#   python plot_all_years.py                        # auto-discovers output_* dirs
#   python plot_all_years.py --output_root .        # explicit root
#   python plot_all_years.py --HGT_DIR /path/to/HGT_tiles --vrt elevation_mosaic.vrt

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from elevation_track_vis import create_elevation_vrt

# Domain
EXTENT = [70, 95, 7, 30]  # [lon_min, lon_max, lat_min, lat_max]

# Colour schemes
YEAR_COLORS = {
    2018: "#06D6A0",  # mint
    2019: "#FFB703",  # amber
    2020: "#E63946",  # red
    2021: "#457B9D",  # steel blue
    2022: "#2A9D8F",  # teal
    2023: "#E9C46A",  # yellow
    2024: "#F4A261",  # orange
    2025: "#A8DADC",  # sky blue
}
# Month sub-colouring for tracks on the elevation map
MONTH_COLORS = {
    "June": "#FF6B6B",
    "July": "#FF4500",
    "August": "#1E90FF",
    "September": "#FF00FF",
    "October": "#32CD32",
}

HGT_DIR = os.path.join("..", "Data", "elevation", "hgtfiles")
vrt_path = "elevation_mosaic.vrt"


def load_all_tracks(output_root):
    """from output_YYYY directory load the track coordinate CSV"""
    dirs = sorted(glob.glob(os.path.join(output_root, "output_*")))
    if not dirs:
        print(f"No output_* directories found in '{output_root}'")

    frames = []
    for d in dirs:
        # Extract year from directory name
        basename = os.path.basename(d)
        try:
            year = int(basename.replace("output_", ""))
        except ValueError:
            continue

        # Prefer MD_coordinates_ALL.csv; fall back to MD_tracks_YYYY.csv
        csv_path = os.path.join(d, "MD_coordinates_ALL.csv")
        if not os.path.exists(csv_path):
            fallback = os.path.join(d, f"MD_tracks_{year}.csv")
            if os.path.exists(fallback):
                csv_path = fallback
            else:
                print(f"[WARN] No track CSV found in {d} — skipping")
                continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Could not read {csv_path}: {e}")
            continue

        # Normalise column names (time might be named valid_time etc.)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"time", "latitude", "longitude", "track_id"}
        if not required.issubset(df.columns):
            print(
                f"[WARN] {csv_path} missing columns {required - set(df.columns)} — skipping"
            )
            continue

        df["time"] = pd.to_datetime(df["time"], dayfirst=True, errors="coerce")
        df["year"] = year
        df["month"] = df["time"].dt.strftime("%B")
        df["track_uid"] = df["year"].astype(str) + "_" + df["track_id"].astype(str)
        frames.append(df)
        print(
            f"[INFO] Loaded year {year}: {df['track_id'].nunique()} tracks, {len(df)} points"
        )

    if not frames:
        raise RuntimeError("No track data could be loaded.")

    return pd.concat(frames, ignore_index=True)


# Plot 1: Plain cartopy map
def plot_tracks_map(all_df, output_path):
    """All tracks over a cartopy base map, one colour per year"""
    years = sorted(all_df["year"].unique())
    n_years = len(years)

    # Figure
    fig = plt.figure(figsize=(16, 11))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    # Base map
    ax.add_feature(cfeature.LAND, facecolor="#F0EDE5", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#D6EAF8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":", alpha=0.6, zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle="--", alpha=0.4, zorder=3)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.4, zorder=2)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
        zorder=1,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Plot each track as a continuous line
    total_tracks = 0
    for year in years:
        color = YEAR_COLORS.get(year, "black")
        yr_df = all_df[all_df["year"] == year]
        first = True
        for track_uid, tr in yr_df.groupby("track_uid"):
            tr = tr.sort_values("time")
            lons = tr["longitude"].values
            lats = tr["latitude"].values

            ax.plot(
                lons,
                lats,
                "-o",
                color=color,
                linewidth=1,
                markersize=2,
                alpha=0.8,
                transform=ccrs.PlateCarree(),
                label=str(year) if first else "_nolegend_",
                zorder=4,
                solid_capstyle="round",
            )

            # Start marker (origin)
            ax.plot(
                lons[0],
                lats[0],
                "^",
                color=color,
                markersize=6,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            # End marker
            ax.plot(
                lons[-1],
                lats[-1],
                "s",
                color=color,
                markersize=5,
                alpha=0.7,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            first = False
            total_tracks += 1

    # Legend: one entry per year
    year_handles = [
        mpatches.Patch(color=YEAR_COLORS.get(y, "black"), label=str(y)) for y in years
    ]
    marker_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="gray",
            linestyle="None",
            markersize=6,
            label="Track origin",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="gray",
            linestyle="None",
            markersize=5,
            label="Track end",
        ),
    ]
    ax.legend(
        handles=year_handles + marker_handles,
        loc="lower left",
        fontsize=9,
        framealpha=0.8,
        title="Year",
        title_fontsize=13,
    )

    # Title
    ax.set_title(
        f"Monsoon Depression Tracks {min(years)}-{max(years)}\n"
        f"{total_tracks} tracks across {n_years} seasons (June-September)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plain map : {output_path}")


# Plot 2: Elevation map
def plot_tracks_elevation(all_df, vrt_path, output_path):
    """All tracks over SRTM elevation mosaic. Tracks coloured by month."""
    if not os.path.exists(vrt_path):
        print(f"[WARN] VRT not found at '{vrt_path}' — skipping elevation plot")
        return

    years = sorted(all_df["year"].unique())
    n_years = len(years)

    with rasterio.open(vrt_path) as src:
        # Downsample to ~5% for speed
        scale = 0.05
        shape = (int(src.height * scale), int(src.width * scale))
        elev = src.read(
            1, out_shape=shape, resampling=rasterio.enums.Resampling.bilinear
        ).astype(float)
        elev[elev < -50] = 0  # SRTM void fill
        bounds = src.bounds

    # Figure
    fig = plt.figure(figsize=(16, 11))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    # Elevation background
    vmin = max(float(np.nanpercentile(elev, 2)), 1.0)
    vmax = float(np.nanpercentile(elev, 97))
    im = ax.imshow(
        elev + 0.1,
        cmap="terrain",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        zorder=1,
        transform=ccrs.PlateCarree(),
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.03, shrink=0.75)
    cbar.set_label("Elevation (m)", fontsize=12, fontweight="bold")

    ax.add_feature(cfeature.COASTLINE, linewidth=0.9, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":", alpha=0.7, zorder=4)
    ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle="--", alpha=0.5, zorder=4)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color="gray",
        alpha=0.6,
        linestyle="--",
        zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Plot tracks - colour by month, style by year
    YEAR_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    year_list = sorted(all_df["year"].unique())
    ls_map = {
        y: YEAR_LINESTYLES[i % len(YEAR_LINESTYLES)] for i, y in enumerate(year_list)
    }

    plotted_months = set()
    plotted_years = set()
    total_tracks = 0

    for (year, track_uid), tr in all_df.groupby(["year", "track_uid"]):
        tr = tr.sort_values("time")
        month = tr["month"].iloc[0] if "month" in tr.columns else "Unknown"
        color = MONTH_COLORS.get(month, "#FFFFFF")
        ls = ls_map[year]

        lons = tr["longitude"].values
        lats = tr["latitude"].values

        ax.plot(
            lons,
            lats,
            color=color,
            linewidth=1.8,
            linestyle=ls,
            markersize=0,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            zorder=5,
            solid_capstyle="round",
        )

        # Start
        ax.plot(
            lons[0],
            lats[0],
            "o",
            color=color,
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.4,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        # End
        ax.plot(
            lons[-1],
            lats[-1],
            "s",
            color=color,
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.4,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

        plotted_months.add(month)
        plotted_years.add(year)
        total_tracks += 1

    # Legend: month colours
    month_order = ["June", "July", "August", "September", "October"]
    month_handles = [
        mpatches.Patch(color=MONTH_COLORS[m], label=m)
        for m in month_order
        if m in plotted_months
    ]

    # Legend: year line-styles
    year_handles = [
        plt.Line2D(
            [0], [0], color="black", linestyle=ls_map[y], linewidth=2, label=str(y)
        )
        for y in year_list
        if y in plotted_years
    ]

    leg1 = ax.legend(
        handles=month_handles,
        loc="lower left",
        title="Month",
        title_fontsize=9,
        fontsize=9,
        framealpha=0.85,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=year_handles,
        loc="lower right",
        title="Year (line style)",
        title_fontsize=9,
        fontsize=9,
        framealpha=0.85,
    )

    # Title
    ax.set_title(
        f"Monsoon Depression Tracks {min(years)}-{max(years)} over SRTM Elevation\n"
        f"{total_tracks} tracks across {n_years} seasons (June-September)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Elevation map : {output_path}")


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Plot all MD tracks across all years")
    p.add_argument(
        "--output_root",
        default=".",
        help="Directory containing output_YYYY folders (default: current dir)",
    )
    p.add_argument(
        "--HGT_DIR",
        default=os.path.join("..", "Data", "elevation", "hgtfiles"),
        help="Directory with SRTM .hgt tile files (for elevation plot)",
    )
    p.add_argument(
        "--vrt",
        default="elevation_mosaic.vrt",
        help="Path to existing VRT mosaic (built by create_elevation_vrt)",
    )
    p.add_argument("--out_dir", default=".", help="To save the output PNG files")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[INFO] Scanning '{args.output_root}' for output_* directories")
    all_df = load_all_tracks(args.output_root)
    print(
        f"\n[INFO] Total: {all_df['track_uid'].nunique()} tracks, "
        f"{len(all_df)} points, {all_df['year'].nunique()} years "
        f"({all_df['year'].min()}-{all_df['year'].max()})\n"
    )

    # Plot 1 — plain map
    map_path = os.path.join(args.out_dir, "all_years_tracks_map.png")
    plot_tracks_map(all_df, map_path)

    # Plot 2 — elevation map
    # Use the VRT argument directly; if absent try to find one in output dirs
    vrt = args.vrt
    if not os.path.exists(vrt):
        # Try to find a VRT inside any output dir
        candidates = glob.glob(
            os.path.join(args.output_root, "output_*", "elevation_mosaic.vrt")
        )
        if candidates:
            vrt = candidates[0]
            print(f"[INFO] Found VRT at {vrt}")

    # create VRT if missing
    if not os.path.exists(args.vrt):
        print(f"[INFO] VRT not found, creating from HGT tiles in {args.HGT_DIR}")
        try:
            args.vrt = create_elevation_vrt(hgt_dir=args.HGT_DIR, vrt_filename=args.vrt)
            print(f"[INFO] VRT created at {args.vrt}")
        except Exception as e:
            print(f"[WARN] Could not create VRT: {e}")
            args.vrt = None  # so we skip elevation plot

    elev_path = os.path.join(args.out_dir, "all_years_tracks_elevation.png")
    if args.vrt and os.path.exists(args.vrt):
        plot_tracks_elevation(all_df, args.vrt, elev_path)
    else:
        print("[INFO] Skipping elevation plot - no VRT available.")


if __name__ == "__main__":
    main()
