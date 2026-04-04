# NOTE:
#   1. Plain map  - tracks over land/ocean cartopy base map, coloured by year
#   2. Elevation  - same tracks over SRTM elevation mosaic (requires VRT or HGT files)
#   3. Elevation interactive map - html file

#   python plot_all_years.py                        # auto-discovers output_* dirs
#   python plot_all_years.py --output_root .        # explicit root
#   python plot_all_years.py --HGT_DIR /path/to/HGT_tiles --vrt elevation_mosaic.vrt

import os
import glob
import argparse
from cartopy.mpl.gridliner import mticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from PIL import Image as PIL
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

import io as io
import base64
import json
import urllib.request as request

from elevation_track_vis import create_elevation_vrt
from utils import std_print, common_bbox

OUTPUT_DIR = "Overall_years"

# Domain
EXTENT = [70, 95, 7, 30]  # [lon_min, lon_max, lat_min, lat_max]

YEAR_COLORS = {
    # 1990: "#4A4A4A",
    # 1991: "#636363",
    # 1992: "#7D7D7D",
    # 1993: "#969696",
    # 1994: "#AFAFAF",
    # 1995: "#8D6E63",
    # 1996: "#795548",
    # 1997: "#6D4C41",
    # 1998: "#5D4037",
    # 1999: "#4E342E",
    # 2000: "#3E2723",
    # 2001: "#5A0000",
    # 2002: "#700000",
    # 2003: "#800000",
    # 2004: "#A50000",
    2005: "#8B0000",
    2006: "#DB1A1A",
    2007: "#E63946",
    2008: "#FF4400",
    2009: "#FF8C00",
    2010: "#FE7743",
    2011: "#FFB703",
    2012: "#E9C46A",
    2013: "#90BE6D",
    2014: "#43AA8B",
    2015: "#2A9D8F",
    2016: "#237227",
    2017: "#1F77B4",
    2018: "#457B9D",
    2019: "#134E8E",
    2020: "#8C00FF",
    2021: "#9400D3",
    2022: "#6E026F",
    2023: "#4B0082",
    2024: "#2A1458",
    2025: "#1A0A3D",
    2026: "#3C096C",
}

MOUNTAIN_POINTS = {
    # --- Eastern ghats ---
    "Jindhagada (1690m)": (
        18.23,
        82.72,
    ),  # Highest in Eastern Ghats (Andhra Pradesh) # 1690m
    "Nimgiri (1515m)": (19.575, 83.42),  # Odisha # 1515m
    "Mahendragiri (1501m)": (18.96, 84.36),  # Odisha coast # 1501m
    "Parasnath (1366m)": (
        23.96,
        86.13,
    ),  # Chota Nagpur Plateau boundary, Jharkhnad # 1366m
    "Malayagiri (1187m)": (21.37, 85.268),  # Odisha # 1187m
    # --- Western ghats ---
    "Anamudi (2695m)": (10.17, 77.06),  # Highest in Western Ghats, Kerala # 2695m
    "Doddabetta (2637m)": (11.40, 76.737),  # Nilgiri Mountains, Kerala # 2637m
    "Mullayanagiri (1930m)": (13.391, 75.721),  # Highest in Karnataka # 1930m
    "Kalsubai (1646m)": (19.6, 73.71),  # Highest in Maharashtra # 1646m
    "Mahabaleshwar (1439m)": (17.93, 73.66),  # Maharashtra # 1439m
    "Agasthyamalai (1868m)": (8.37, 77.15),  # Kerala # 1868m
    "Brahmagiri (1608m)": (11.57, 75.57),  # Kerala # 1608m
    # --- Northern & central ---
    "Guru Shikhar (1722m)": (24.65, 72.71),  # Highest in Aravali (Mount Abu)
    "Dhupgarh (1350m)": (
        22.45,
        78.37,
    ),  # Highest in Satpura Range, # Madhya Pradesh # 1350
    "Amarkantak (1067m)": (22.67, 81.75),  # Madhya Pradesh # 1067m
    "Kanchenjunga (8586m)": (27.70, 88.15),  # Sikkim # 8586m
    "Goakhnath (1117m)": (21.528, 70.52),  # Gujarat # 1117m
    "Lugu pahar (1070m)": (23.77, 85.71),  # Jharkhnad # 1070m
    "Saru pahar (1113m)": (23.045, 84.49),  # Jharkhand # 1113m
    # --- East India ---
    "Nokrek (1412m)": (25.46, 90.322),  # Meghalaya # 1412m
    "Blue Peak (2157m)": (22.63, 93.04),  # Mizkoram # 2157m
    "Saramati (3830m)": (25.74, 95.037),  # Nagaland # 3830m
    # --- Andaman Nicobar ---
    "Saddle Peak (732m)": (13.159, 93.00),  # Anadman # 732m
    "Mt. Hariet (370m)": (11.72, 92.73),  # Anadman # 370m
}


def load_all_tracks(output_root):
    """from output_YYYY directory load the track coordinate CSV"""
    dirs = sorted(glob.glob(os.path.join(output_root, "output_*")))
    if not dirs:
        std_print("ERROR", "pa", f"No output_* directories found in '{output_root}'")

    frames = []
    for d in dirs:
        # Extract year from directory name
        basename = os.path.basename(d)
        try:
            year = int(basename.replace("output_", ""))
        except ValueError:
            continue

        # Prefer MD_coordinates_ALL.csv; fall back to MD_tracks_YYYY.csv
        coordinates_CSV = os.path.join(d, f"MD_tracks_{year}.csv")
        if not os.path.exists(coordinates_CSV):
            fallback = os.path.join(d, "MD_coordinates_ALL.csv")
            if os.path.exists(fallback):
                coordinates_CSV = fallback
            else:
                std_print("ERROR", "pa", f" No track CSV found in {d} - skipping")
                continue

        try:
            df = pd.read_csv(coordinates_CSV)
        except Exception as e:
            std_print("ERROR", "pa", f"Could not read {coordinates_CSV}", str(e))
            continue

        # Normalise column names (time might be named valid_time etc.)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"time", "latitude", "longitude", "track_id"}
        if not required.issubset(df.columns):
            std_print(
                "WARN",
                "pa",
                f"{coordinates_CSV} missing columns {required - set(df.columns)} - skipping",
            )
            continue

        df["time"] = pd.to_datetime(df["time"], dayfirst=True, errors="coerce")
        df["year"] = year
        df["month"] = df["time"].dt.strftime("%B")
        df["track_uid"] = df["year"].astype(str) + "_" + df["track_id"].astype(str)
        frames.append(df)
        std_print(
            "INFO",
            "pa",
            f"Loaded year {year}: {df['track_id'].nunique()} tracks, {len(df)} points",
        )
    if not frames:
        raise RuntimeError("No track data could be loaded.")
    return pd.concat(frames, ignore_index=True)


# Plot 1: Plain cartopy map
def plot_tracks_map(all_df, output_path):
    """All tracks over a cartopy base map, one colour per year"""
    years = sorted(all_df["year"].unique())

    # Figure
    fig = plt.figure(figsize=(16, 11))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

    # Base map
    ax.add_feature(cfeature.LAND, facecolor="#F0EDE5", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#D6EAF8", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8, zorder=3)
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
                "-",
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

    # Legend
    year_handles = [
        mpatches.Patch(color=YEAR_COLORS.get(y, "#333333"), label=str(y)) for y in years
    ]
    ax.legend(
        handles=year_handles,
        loc="lower right",
        title="Year",
        title_fontsize=9,
        fontsize=8,
        framealpha=0.8,
        ncol=3,
        borderpad=0.4,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.6,
        labelspacing=0.3,
    )

    # Title
    ax.set_title(
        f"Monsoon Depression Tracks {min(years)}-{max(years)}\n"
        f"{total_tracks} tracks across {max(years) - min(years) + 1} seasons (June-September)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "pa", "Plain map", output_path)


# Plot 2: Elevation map
def plot_tracks_elevation(all_df, vrt_path, output_path):
    """All tracks over SRTM elevation mosaic"""
    if not os.path.exists(vrt_path):
        std_print(
            "ERROR", "pa", f"VRT not found at '{vrt_path}'", "skipping elevation plot"
        )
        return

    years = sorted(all_df["year"].unique())
    months = ["June", "July", "August", "September"]

    # Line styles for months
    MONTH_LINESTYLES = {
        "June": ":",
        "July": "-.",
        "August": "--",
        "September": "-",
    }

    with rasterio.open(vrt_path) as src:
        # Downsample for speed
        scale = 0.05
        shape = (int(src.height * scale), int(src.width * scale))
        elev = src.read(
            1, out_shape=shape, resampling=rasterio.enums.Resampling.bilinear
        ).astype(float)
        elev[elev < -50] = 0  # void fill
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

    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.9, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":", alpha=0.7, zorder=4)
    ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle="--", alpha=0.5, zorder=4)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
        zorder=2,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Plot tracks
    plotted_months = set()
    total_tracks = 0

    for (year, track_uid), tr in all_df.groupby(["year", "track_uid"]):
        tr = tr.sort_values("time")
        month = tr["month"].iloc[0] if "month" in tr.columns else "Unknown"
        color = YEAR_COLORS.get(year, "#333333")
        ls = MONTH_LINESTYLES.get(month, "-")

        lons = tr["longitude"].values
        lats = tr["latitude"].values

        ax.plot(
            lons,
            lats,
            color=color,
            linewidth=1.5,
            linestyle=ls,
            alpha=0.8,
            transform=ccrs.PlateCarree(),
            zorder=5,
            solid_capstyle="round",
        )

        # Start and End markers
        ax.plot(
            lons[0],
            lats[0],
            "o",
            color=color,
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.5,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        ax.plot(
            lons[-1],
            lats[-1],
            "s",
            color=color,
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.5,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

        plotted_months.add(month)
        total_tracks += 1

    # Mountain points
    for name, (lat, lon) in MOUNTAIN_POINTS.items():
        ax.plot(
            lon,
            lat,
            "^",
            color="white",
            markeredgecolor="black",
            markersize=7,
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        ax.text(
            lon - 0.2,
            lat,
            name,
            transform=ccrs.PlateCarree(),
            fontsize=6,
            fontweight="bold",
            color="black",
            ha="right",
            va="center",
            bbox=common_bbox(),
            zorder=11,
        )

    # Legends
    # 1. Month
    month_handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle=MONTH_LINESTYLES.get(m, "-"),
            linewidth=2,
            label=m,
        )
        for m in months
        if m in plotted_months
    ]
    if month_handles:
        month_legend = ax.legend(
            handles=month_handles,
            loc="lower left",
            title="Month",
            title_fontsize=9,
            fontsize=8,
            framealpha=0.8,
            borderpad=0.4,
            handlelength=1.5,
            handletextpad=0.4,
            labelspacing=0.3,
        )
        ax.add_artist(month_legend)

    # 2. Year
    year_handles = [
        mpatches.Patch(color=YEAR_COLORS.get(y, "#333333"), label=str(y)) for y in years
    ]
    ax.legend(
        handles=year_handles,
        loc="lower right",
        title="Year",
        title_fontsize=9,
        fontsize=8,
        framealpha=0.8,
        ncol=3,
        borderpad=0.4,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.6,
        labelspacing=0.3,
    )

    # Title
    ax.set_title(
        f"Monsoon Depression Tracks {min(years)}–{max(years)} over SRTM Elevation\n"
        f"{total_tracks} tracks across {max(years) - min(years) + 1} seasons (June-September)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "pa", f"Elevation map saved", output_path)


def plot_ocean_origin_only(all_df, vrt_path, output_path):
    """Filters tracks to only those originating over the ocean"""
    if "is_land" not in all_df.columns:
        std_print(
            "WARN",
            "pa",
            "'is_land' column not found in data - cannot filter ocean-only tracks. Skipping.",
        )
        return

    ocean_track_ids = []
    for uid, tr in all_df.groupby("track_uid"):
        first_point = tr.sort_values("time").iloc[0]

        is_land_val = first_point.get("is_land", None)

        if is_land_val is None or pd.isna(is_land_val):
            continue

        # Only keep tracks whose first point is explicitly NOT over land
        if not bool(is_land_val):
            ocean_track_ids.append(uid)

    ocean_df = all_df[all_df["track_uid"].isin(ocean_track_ids)]
    if not ocean_df.empty:
        std_print("INFO", "pa", f"Plotting {len(ocean_track_ids)} ocean-origin tracks")
        plot_tracks_elevation(ocean_df, vrt_path, output_path)
    else:
        std_print("WARN", "pa", "No ocean-origin tracks found to plot.")


def fetch(url):
    try:
        with request.urlopen(url, timeout=15) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def export_interactive_map(all_df, output_path, vrt_path=None):
    """
    Generate a fully self-contained HTML interactive map that works when
    opened directly as file:// (no web server required).

    Controls:
      1. Year toggles          (checkbox per year, colour swatch)
      2. Month toggles         (June / July / August / September)
      3. Track-origin toggle   (Ocean / Land / Unknown)
      4. SRTM Elevation opacity     (slider, only shown when VRT is available)
      5. Mountains toggle
    """
    LEAFLET_VER = "1.9.4"

    # Download Leaflet once so the HTML works as a local file:// with no server
    leaflet_css = fetch(f"https://unpkg.com/leaflet@{LEAFLET_VER}/dist/leaflet.css")
    leaflet_js = fetch(f"https://unpkg.com/leaflet@{LEAFLET_VER}/dist/leaflet.js")
    std_print(
        "INFO",
        "pa",
        "Leaflet assets " + ("inlined" if leaflet_js else "CDN fallback"),
    )

    MONTH_DASH = {"June": "4,7", "July": "12,4,3,4", "August": "10,6", "September": ""}
    MONTHS = ["June", "July", "August", "September"]
    # Build track data list
    tracks_list = []
    for (year, track_uid), tr in all_df.groupby(["year", "track_uid"]):
        tr = tr.sort_values("time")
        pts = tr[["latitude", "longitude"]].dropna().values.tolist()
        if len(pts) < 2:
            continue
        yr = int(year)
        mon = str(tr["month"].iloc[0]) if "month" in tr.columns else "Unknown"
        ilv = tr.iloc[0].get("is_land", None)
        try:
            is_na = ilv is None or bool(pd.isna(ilv))
        except Exception:
            is_na = True
        orig = "Unknown" if is_na else ("Land" if bool(ilv) else "Ocean")
        tracks_list.append(
            {
                "uid": str(track_uid),
                "year": yr,
                "month": mon,
                "origin": orig,
                "color": YEAR_COLORS.get(yr, "#FF0000"),
                "dash": MONTH_DASH.get(mon, ""),
                "pts": pts,
                "start": pts[0],
                "end": pts[-1],
            }
        )

    mtns_list = [
        {"n": name, "lat": lat, "lon": lon}
        for name, (lat, lon) in MOUNTAIN_POINTS.items()
    ]

    # Elevation raster
    elev_url_js = "null"  # JS literal: null or "data:image/png;base64,..."
    elev_bnds_js = "null"  # JS literal: null or [[s,w],[n,e]]
    if vrt_path and os.path.exists(vrt_path):
        try:
            with rasterio.open(vrt_path) as src:
                sc = 0.05
                sh = (int(src.height * sc), int(src.width * sc))
                elev = src.read(
                    1, out_shape=sh, resampling=rasterio.enums.Resampling.bilinear
                ).astype(float)
                elev[elev < -50] = 0
                bnds = src.bounds
            vmin = float(np.nanpercentile(elev, 2))
            vmax = float(np.nanpercentile(elev, 98))
            norm = np.clip((elev - vmin) / (vmax - vmin + 1e-6), 0, 1)
            rgba = (cm.terrain(norm) * 255).astype(np.uint8)
            rgba[..., 3] = 140  # ~55% opacity channel
            buf = io.BytesIO()
            PIL.fromarray(rgba).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            elev_url_js = '"data:image/png;base64,' + b64 + '"'
            elev_bnds_js = json.dumps(
                [[bnds.bottom, bnds.left], [bnds.top, bnds.right]]
            )
            std_print("INFO", "pa", "VRT elevation raster embedded")
        except Exception as e:
            std_print("WARN", "pa", "VRT raster skipped", str(e))

    # Panel HTML snippets
    years_sorted = sorted(int(y) for y in all_df["year"].unique())

    yr_rows = "".join(
        '<label class="cb-row">'
        f'<input type="checkbox" class="yr-cb" value="{yr}" checked>'
        f'<span class="sw" style="background:{YEAR_COLORS.get(yr, "#666")}"></span>'
        f"{yr}</label>"
        for yr in years_sorted
    )
    mo_rows = ""
    for mo in MONTHS:
        dash = MONTH_DASH.get(mo, "")
        # Create SVG line preview
        if mo == "September":
            svg = '<svg width="20" height="4" viewBox="0 0 20 4"><line x1="0" y1="2" x2="20" y2="2" stroke="#333" stroke-width="2"/></svg>'
        elif mo == "August":
            svg = '<svg width="20" height="4" viewBox="0 0 20 4"><line x1="0" y1="2" x2="20" y2="2" stroke="#333" stroke-width="2" stroke-dasharray="6,4"/></svg>'
        elif mo == "July":
            svg = '<svg width="20" height="4" viewBox="0 0 20 4"><line x1="0" y1="2" x2="20" y2="2" stroke="#333" stroke-width="2" stroke-dasharray="4,4,8,4"/></svg>'
        elif mo == "June":
            svg = '<svg width="20" height="4" viewBox="0 0 20 4"><line x1="0" y1="2" x2="20" y2="2" stroke="#333" stroke-width="2" stroke-dasharray="2,4"/></svg>'
        mo_rows += (
            f'<label class="cb-row">'
            f'<input type="checkbox" class="mo-cb" value="{mo}" checked>'
            f"{svg}"
            f'<span style="margin-left:4px">{mo}</span>'
            f"</label>"
        )
        elev_ctrl = (
            '<div class="sec">Elevation overlay</div>'
            '<div class="slider-row">'
            "<span>SRTM Opacity</span>"
            '<input type="range" id="elev-sl" min="0" max="100" value="55" style="width:82px">'
            '<span id="elev-pct">55%</span>'
            "</div>"
            if elev_url_js != "null"
            else '<div class="sec">Elevation overlay</div><span class="note">No VRT file</span>'
        )

    # Leaflet asset tags
    css_tag = (
        f"<style>{leaflet_css}</style>"
        if leaflet_css
        else f'<link rel="stylesheet" href="https://unpkg.com/leaflet@{LEAFLET_VER}/dist/leaflet.css">'
    )
    js_tag = (
        f"<script>{leaflet_js}</script>"
        if leaflet_js
        else f'<script src="https://unpkg.com/leaflet@{LEAFLET_VER}/dist/leaflet.js"></script>'
    )

    # Define domain bounds (70°E, 95°E, 7°N, 30°N)
    # Leaflet bounds are [[south, west], [north, east]]
    BOUNDS = [[7, 70], [30, 95]]

    # HTML template
    TEMPLATE = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>Monsoon Depression \u2013 Interactive Track Map</title>\n"
        "__LEAFLET_CSS__\n"
        "<style>\n"
        "html,body{height:100%;margin:0;padding:0;font-family:sans-serif}\n"
        "#map{height:100vh;width:100%}\n"
        "#ctrl{position:absolute;top:10px;right:10px;z-index:1000;"
        "background:rgba(255,255,255,.95);border-radius:8px;"
        "box-shadow:0 2px 12px rgba(0,0,0,.3);padding:10px 12px;"
        "width:182px;max-height:92vh;overflow-y:auto;font-size:12px;line-height:1.5}\n"
        "#ctrl h3{margin:0 0 5px;font-size:13px;border-bottom:1px solid #ccc;padding-bottom:4px}\n"
        ".sec{font-weight:700;font-size:10.5px;color:#555;margin:8px 0 3px;"
        "text-transform:uppercase;letter-spacing:.05em}\n"
        ".cb-row{display:flex;align-items:center;gap:4px;cursor:pointer;padding:1px 0}\n"
        ".sw{display:inline-block;width:11px;height:11px;border-radius:2px;flex-shrink:0}\n"
        ".slider-row{display:flex;align-items:center;gap:5px;margin-top:3px;flex-wrap:wrap}\n"
        ".note{color:#aaa;font-size:11px}\n"
        ".sel-btns{display:flex;gap:4px;margin-bottom:4px}\n"
        ".sel-btns button{flex:1;font-size:10px;padding:2px 0;cursor:pointer;"
        "border:1px solid #ccc;border-radius:3px;background:#f5f5f5}\n"
        ".sel-btns button:hover{background:#e0e0e0}\n"
        ".mtn-tt{background:rgba(255,255,255,.92)!important;border:none!important;"
        "box-shadow:0 1px 4px rgba(0,0,0,.3)!important;font-size:11px;padding:2px 6px!important}\n"
        ".line-preview-svg { display: inline-block; vertical-align: middle; margin-right: 4px; }\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        '<div id="map"></div>\n'
        '<div id="ctrl">\n'
        "  <h3>MD Tracks</h3>\n"
        '  <div class="sec">Base map</div>\n'
        '  <label class="cb-row"><input type="radio" name="bm" value="carto" checked> Plain (CartoDB)</label>\n'
        '  <label class="cb-row"><input type="radio" name="bm" value="topo"> Topo (OpenTopoMap)</label>\n'
        "  __ELEV_CTRL__\n"
        '  <div class="sec">Mountains</div>\n'
        '  <label class="cb-row"><input type="checkbox" id="mtn-cb" checked> Show peaks \u25b2</label>\n'
        '  <div class="sec">Year</div>\n'
        '  <div class="sel-btns">\n'
        "    <button onclick=\"setAll('.yr-cb',true)\">All</button>\n"
        "    <button onclick=\"setAll('.yr-cb',false)\">None</button>\n"
        "  </div>\n"
        "  __YR_ROWS__\n"
        '  <div class="sec">Month</div>\n'
        "  __MO_ROWS__\n"
        '  <div class="sec">Track origin</div>\n'
        '  <label class="cb-row"><input type="checkbox" class="orig-cb" value="Ocean" checked>'
        '    <span class="sw" style="background:#1a78c2"></span>Ocean</label>\n'
        '  <label class="cb-row"><input type="checkbox" class="orig-cb" value="Land" checked>'
        '    <span class="sw" style="background:#b5440a"></span>Land</label>\n'
        '  <label class="cb-row"><input type="checkbox" class="orig-cb" value="Unknown" checked>'
        '    <span class="sw" style="background:#888"></span>Unknown</label>\n'
        "</div>\n"
        "__LEAFLET_JS__\n"
        "<script>\n"
        "var TRACKS    = __TRACKS_JSON__;\n"
        "var MTNS      = __MTNS_JSON__;\n"
        "var ELEV_URL  = __ELEV_URL__;\n"
        "var ELEV_BNDS = __ELEV_BNDS__;\n"
        "\n"
        "var map = L.map('map').setView([20,82],5);\n"
        "// Restrict panning to the defined bounds\n"
        "map.setMaxBounds(__BOUNDS__);\n"
        "map.on('drag', function() { map.panInsideBounds(__BOUNDS__, {animate: false}); });\n"
        "\n"
        "var bCarto = L.tileLayer(\n"
        "  'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',\n"
        "  {attribution:'&copy; <a href=\"https://www.openstreetmap.org/copyright\">OSM</a>'"
        "   +' &copy; <a href=\"https://carto.com/\">CARTO</a>',"
        "   subdomains:'abcd',maxZoom:19}\n"
        ").addTo(map);\n"
        "\n"
        "var bTopo = L.tileLayer(\n"
        "  'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',\n"
        "  {attribution:'Map data: &copy; OSM, SRTM | Style: &copy;"
        '   <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)\',maxZoom:17}\n'
        ");\n"
        "\n"
        "var elevLyr=null;\n"
        "if(ELEV_URL&&ELEV_BNDS){\n"
        "  elevLyr=L.imageOverlay(ELEV_URL,ELEV_BNDS,{opacity:0.55,interactive:false});\n"
        "  elevLyr.addTo(map);\n"
        "}\n"
        "\n"
        "var gTracks =L.layerGroup().addTo(map);\n"
        "var gMarkers=L.layerGroup().addTo(map);\n"
        "var gMtns   =L.layerGroup().addTo(map);\n"
        "\n"
        "function drawMtns(){\n"
        "  gMtns.clearLayers();\n"
        "  if(!document.getElementById('mtn-cb').checked)return;\n"
        "  MTNS.forEach(function(m){\n"
        "    var ico=L.divIcon({\n"
        "      html:'<div style=\"font-size:11px;line-height:1;color:#333;'"
        "           +'text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff\">\u25b2</div>',\n"
        "      className:'',iconSize:[12,12],iconAnchor:[6,11]\n"
        "    });\n"
        "    L.marker([m.lat,m.lon],{icon:ico,zIndexOffset:500})\n"
        "     .bindTooltip(m.n,{direction:'top',className:'mtn-tt',offset:[0,-8]})\n"
        "     .addTo(gMtns);\n"
        "  });\n"
        "}\n"
        "drawMtns();\n"
        "\n"
        "function activeVals(cls){\n"
        "  var s={};\n"
        "  document.querySelectorAll(cls+':checked').forEach(function(e){s[e.value]=1;});\n"
        "  return s;\n"
        "}\n"
        "\n"
        "function redraw(){\n"
        "  gTracks.clearLayers(); gMarkers.clearLayers();\n"
        "  var aYr =activeVals('.yr-cb');\n"
        "  var aMo =activeVals('.mo-cb');\n"
        "  var aOrg=activeVals('.orig-cb');\n"
        "  TRACKS.forEach(function(t){\n"
        "    if(!aYr[t.year]||!aMo[t.month]||!aOrg[t.origin])return;\n"
        "    L.polyline(t.pts,{color:t.color,weight:2.5,opacity:0.75,dashArray:t.dash||null})\n"
        "     .bindTooltip(t.year+' | '+t.month+' | '+t.origin"
        "                  +'<br><small>'+t.uid+'</small>',{sticky:true})\n"
        "     .addTo(gTracks);\n"
        "    L.circleMarker(t.start,{\n"
        "      radius:4,color:t.color,weight:2,fillColor:'#fff',fillOpacity:1\n"
        "    }).bindTooltip('Start: '+t.uid).addTo(gMarkers);\n"
        "    L.marker(t.end,{icon:L.divIcon({\n"
        "      html:'<div style=\"width:7px;height:7px;background:'+t.color"
        "           +';border:1.5px solid #fff;box-shadow:0 0 2px rgba(0,0,0,.4)\"></div>',\n"
        "      className:'',iconSize:[7,7],iconAnchor:[4,4]\n"
        "    })}).bindTooltip('End: '+t.uid).addTo(gMarkers);\n"
        "  });\n"
        "}\n"
        "redraw();\n"
        "\n"
        "function setAll(cls,val){\n"
        "  document.querySelectorAll(cls).forEach(function(e){e.checked=val;});\n"
        "  redraw();\n"
        "}\n"
        "document.querySelectorAll('.yr-cb,.mo-cb,.orig-cb')\n"
        "  .forEach(function(e){e.addEventListener('change',redraw);});\n"
        "document.getElementById('mtn-cb').addEventListener('change',drawMtns);\n"
        "document.querySelectorAll('input[name=\"bm\"]').forEach(function(el){\n"
        "  el.addEventListener('change',function(){\n"
        "    if(this.value==='topo'){map.removeLayer(bCarto);bTopo.addTo(map);}\n"
        "    else{map.removeLayer(bTopo);bCarto.addTo(map);}\n"
        "  });\n"
        "});\n"
        "var esl=document.getElementById('elev-sl');\n"
        "if(esl&&elevLyr){\n"
        "  esl.addEventListener('input',function(){\n"
        "    elevLyr.setOpacity(this.value/100);\n"
        "    document.getElementById('elev-pct').textContent=this.value+'%';\n"
        "  });\n"
        "}\n"
        "</script>\n"
        "</body>\n"
        "</html>"
    )

    html = (
        TEMPLATE.replace("__LEAFLET_CSS__", css_tag)
        .replace("__LEAFLET_JS__", js_tag)
        .replace("__ELEV_CTRL__", elev_ctrl)
        .replace("__YR_ROWS__", yr_rows)
        .replace("__MO_ROWS__", mo_rows)
        .replace("__TRACKS_JSON__", json.dumps(tracks_list))
        .replace("__MTNS_JSON__", json.dumps(mtns_list))
        .replace("__ELEV_URL__", elev_url_js)
        .replace("__ELEV_BNDS__", elev_bnds_js)
        .replace("__BOUNDS__", json.dumps(BOUNDS))
    )

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    std_print("SAVED", "pa", "Interactive map", output_path)


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
        help="Directory with SRTM .hgt tile files",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    std_print("INFO", "pa", "Using output directory", OUTPUT_DIR)
    map_path = os.path.join(OUTPUT_DIR, "all_years_tracks_map.png")
    elev_path = os.path.join(OUTPUT_DIR, "all_years_tracks_elevation.png")
    vrt_path = os.path.join(OUTPUT_DIR, "elevation_mosaic.vrt")

    std_print("INFO", "pa", f"Scanning '{args.output_root}' for output_* directories")
    all_df = load_all_tracks(args.output_root)
    if all_df.empty:
        std_print("ERROR", "pa", "No track data found. Exiting...")
        return

    print()
    std_print(
        "INFO",
        "pa",
        f"Total: {all_df['track_uid'].nunique()} tracks, "
        f"{len(all_df)} points, {all_df['year'].nunique()} years "
        f"({all_df['year'].min()}-{all_df['year'].max()})",
    )

    # Plot 1 - plain map
    plot_tracks_map(all_df, map_path)

    # Plot 2 - elevation map
    # Use VRT inside Overall_years folder
    if not os.path.exists(vrt_path):
        std_print("WARN", "pa", "VRT not found. Creating ...")
        try:
            # You can still pass custom HGT_DIR via argument
            create_elevation_vrt(hgt_dir=args.HGT_DIR, vrt_filename=vrt_path)
            std_print("INFO", "pa", "VRT created successfully", vrt_path)
        except Exception as e:
            std_print("ERROR", "pa", "Failed to create VRT", str(e))
            vrt_path = None

    if vrt_path and os.path.exists(vrt_path):
        plot_tracks_elevation(all_df, vrt_path, elev_path)
    else:
        std_print("WARN", "pa", "Skipping elevation plot - no VRT available")

    # Plot 3 - elevation map - ocean-origin tracks only
    if vrt_path and os.path.exists(vrt_path):
        plot_ocean_origin_only(all_df, vrt_path, "Overall_years/ocean_only_tracks.png")
    else:
        std_print("WARN", "pa", "Skipping elevation plot - no VRT available")

    # Plot 4 - generate the interactive map
    folium_path = os.path.join(OUTPUT_DIR, "interactive_tracks_map.html")
    export_interactive_map(
        all_df,
        folium_path,
        vrt_path=vrt_path if (vrt_path and os.path.exists(vrt_path)) else None,
    )

    std_print("INFO", "pa", "All outputs saved", f"./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
