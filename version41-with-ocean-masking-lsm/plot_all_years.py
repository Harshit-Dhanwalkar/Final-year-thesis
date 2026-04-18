# NOTE:
#   1. Plain map  - tracks over land/ocean cartopy base map, coloured by year
#   2. Elevation  - same tracks over SRTM elevation mosaic (requires VRT or HGT files)
#   3. Elevation interactive map - html file
#
#   python plot_all_years.py                        # auto-discovers output_* dirs
#   python plot_all_years.py --output_root .        # explicit root
#   python plot_all_years.py --HGT_DIR /path/to/HGT_tiles --vrt elevation_mosaic.vrt

import os
import io
import glob
import argparse
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import rasterio
from matplotlib.colors import LogNorm
from PIL import Image as PIL
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import mticker
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

import base64
import json
import urllib.request as request

from elevation_track_vis import create_elevation_vrt
from utils import std_print, common_bbox

OUTPUT_DIR = "Overall_years"

# Domain
FULL_EXTENT = [70, 95, 7, 30]  # [lon_min, lon_max, lat_min, lat_max]

# Font sizes
FS_TITLE = 14
FS_LABEL = 12
FS_TICK = 10
FS_LEGEND = 10
FS_ANNOT = 9


# Dynamic colour palette based on year order
def get_year_colors(years_list, cmap_name="inferno", reverse=False):
    """Generate a colour for each year using a continuous colormap"""
    years_sorted = sorted(years_list)
    if len(years_sorted) == 0:
        return {}
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=min(years_sorted), vmax=max(years_sorted))
    colors = {}
    for y in years_sorted:
        t = norm(y)
        if reverse:
            t = 1 - t
        rgba = cmap(t)
        colors[y] = mcolors.to_hex(rgba)
    return colors


# Mountain points
MOUNTAIN_POINTS = {
    "Jindhagada (1690m)": (18.23, 82.72),
    "Nimgiri (1515m)": (19.575, 83.42),
    "Mahendragiri (1501m)": (18.96, 84.36),
    "Parasnath (1366m)": (23.96, 86.13),
    "Malayagiri (1187m)": (21.37, 85.268),
    "Anamudi (2695m)": (10.17, 77.06),
    "Doddabetta (2637m)": (11.40, 76.737),
    "Mullayanagiri (1930m)": (13.391, 75.721),
    "Kalsubai (1646m)": (19.6, 73.71),
    "Mahabaleshwar (1439m)": (17.93, 73.66),
    "Agasthyamalai (1868m)": (8.37, 77.15),
    "Brahmagiri (1608m)": (11.57, 75.57),
    "Guru Shikhar (1722m)": (24.65, 72.71),
    "Dhupgarh (1350m)": (22.45, 78.37),
    "Amarkantak (1067m)": (22.67, 81.75),
    "Kanchenjunga (8586m)": (27.70, 88.15),
    "Goakhnath (1117m)": (21.528, 70.52),
    "Lugu pahar (1070m)": (23.77, 85.71),
    "Saru pahar (1113m)": (23.045, 84.49),
    "Nokrek (1412m)": (25.46, 90.322),
    "Blue Peak (2157m)": (22.63, 93.04),
    "Saramati (3830m)": (25.74, 95.037),
    "Saddle Peak (732m)": (13.159, 93.00),
    "Mt. Hariet (370m)": (11.72, 92.73),
}


def load_all_tracks(output_root):
    """from output_YYYY directory load the track coordinate CSV"""
    dirs = sorted(glob.glob(os.path.join(output_root, "output_*")))
    if not dirs:
        std_print("ERROR", "pa", f"No output_* directories found in '{output_root}'")

    frames = []
    for d in dirs:
        basename = os.path.basename(d)
        try:
            year = int(basename.replace("output_", ""))
        except ValueError:
            continue

        # Prefer MD_tracks_YYYY.csv; fall back to MD_coordinates_ALL.csv
        coordinates_CSV = os.path.join(d, f"MD_tracks_{year}.csv")
        if not os.path.exists(coordinates_CSV):
            fallback = os.path.join(d, "MD_coordinates_ALL.csv")
            if os.path.exists(fallback):
                coordinates_CSV = fallback
            else:
                std_print("WARN", "pa", f"No track CSV found in {d} - skipping")
                continue

        try:
            df = pd.read_csv(coordinates_CSV)
        except Exception as e:
            std_print("ERROR", "pa", f"Could not read {coordinates_CSV}", str(e))
            continue

        df.columns = [c.strip().lower() for c in df.columns]
        required = {"time", "latitude", "longitude", "track_id"}
        if not required.issubset(df.columns):
            std_print(
                "WARN",
                "pa",
                f"{coordinates_CSV} missing columns {required - set(df.columns)} - skipping",
            )
            continue

        df["time"] = pd.to_datetime(
            df["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
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
def plot_tracks_map(all_df, output_path, cmap_name="inferno"):
    years = sorted(all_df["year"].unique())
    year_colors = get_year_colors(years, cmap_name=cmap_name)

    # Figure
    fig = plt.figure(figsize=(16, 11))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(FULL_EXTENT, crs=ccrs.PlateCarree())

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

    total_tracks = 0
    for year in years:
        color = year_colors[year]
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
                alpha=0.8,
                transform=ccrs.PlateCarree(),
                label=str(year) if first else "_nolegend_",
                zorder=4,
                solid_capstyle="round",
            )
            ax.plot(
                lons[0],
                lats[0],
                "^",
                color=color,
                markersize=6,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
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
    year_handles = [mpatches.Patch(color=year_colors[y], label=str(y)) for y in years]
    ax.legend(
        handles=year_handles,
        loc="lower right",
        title="Year",
        title_fontsize=FS_LEGEND + 1,
        fontsize=FS_LABEL,
        framealpha=0.8,
        ncol=3,
        borderpad=0.4,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.6,
        labelspacing=0.3,
    )

    ax.set_title(
        f"Monsoon Depression Tracks {min(years)}-{max(years)}\n"
        f"{total_tracks} tracks across {max(years)-min(years)+1} seasons (June-September)",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "pa", "Plain map", output_path)


# Plot 2: Elevation map
def plot_tracks_elevation(all_df, vrt_path, output_path, cmap_name="inferno"):
    if not os.path.exists(vrt_path):
        std_print(
            "ERROR", "pa", f"VRT not found at '{vrt_path}'", "skipping elevation plot"
        )
        return

    years = sorted(all_df["year"].unique())
    year_colors = get_year_colors(years, cmap_name=cmap_name)
    months = ["June", "July", "August", "September"]
    MONTH_LINESTYLES = {"June": ":", "July": "-.", "August": "--", "September": "-"}

    with rasterio.open(vrt_path) as src:
        scale = 0.05
        shape = (int(src.height * scale), int(src.width * scale))
        elev = src.read(
            1, out_shape=shape, resampling=rasterio.enums.Resampling.bilinear
        ).astype(float)
        elev[elev < -50] = 0  # fill voids
        bounds = src.bounds

    # Figure
    fig = plt.figure(figsize=(16, 11))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(FULL_EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor("#D6EAF8")
    # Mask out ocean (elevation <= 0) so it becomes transparent
    elev_masked = np.ma.masked_where(elev <= 0, elev)

    vmin = max(float(np.nanpercentile(elev[elev > 0], 2)), 1.0)  # ignore ocean for vmin
    vmax = float(np.nanpercentile(elev[elev > 0], 97))

    im = ax.imshow(
        elev_masked,
        cmap="terrain",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        zorder=1,
        transform=ccrs.PlateCarree(),
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.03, shrink=0.75)
    cbar.set_label("Elevation (m)", fontsize=FS_LABEL, fontweight="bold")

    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.9, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":", alpha=0.7, zorder=4)
    ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle="--", alpha=0.5, zorder=4)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.4, zorder=2)

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

    # Plot tracks (same as before)
    plotted_months = set()
    total_tracks = 0
    for (year, track_uid), tr in all_df.groupby(["year", "track_uid"]):
        tr = tr.sort_values("time")
        month = tr["month"].iloc[0] if "month" in tr.columns else "Unknown"
        color = year_colors.get(year, "#333333")
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
            fontsize=FS_ANNOT,
            fontweight="bold",
            color="black",
            ha="right",
            va="center",
            bbox=common_bbox(),
            zorder=11,
        )

    # Legends
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
            title_fontsize=FS_LEGEND + 1,
            fontsize=FS_LEGEND,
            framealpha=0.8,
            borderpad=0.4,
            handlelength=1.5,
            handletextpad=0.4,
            labelspacing=0.3,
        )
        ax.add_artist(month_legend)

    year_handles = [mpatches.Patch(color=year_colors[y], label=str(y)) for y in years]
    ax.legend(
        handles=year_handles,
        loc="lower right",
        title="Year",
        title_fontsize=FS_LEGEND + 1,
        fontsize=FS_LEGEND,
        framealpha=0.8,
        ncol=3,
        borderpad=0.4,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.6,
        labelspacing=0.3,
    )

    ax.set_title(
        f"Monsoon Depression Tracks {min(years)}-{max(years)} over SRTM Elevation\n"
        f"{total_tracks} tracks across {max(years)-min(years)+1} seasons (June-September)",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "pa", "Elevation map", output_path)


def plot_ocean_origin_only(all_df, vrt_path, output_path, cmap_name="inferno"):
    if "is_land" not in all_df.columns:
        std_print(
            "WARN",
            "pa",
            "'is_land' column not found - cannot filter ocean-only tracks",
            "Skipping.",
        )
        return
    ocean_track_ids = []
    for uid, tr in all_df.groupby("track_uid"):
        first_point = tr.sort_values("time").iloc[0]
        is_land_val = first_point.get("is_land", None)
        if is_land_val is None or pd.isna(is_land_val):
            continue
        if not bool(is_land_val):
            ocean_track_ids.append(uid)
    ocean_df = all_df[all_df["track_uid"].isin(ocean_track_ids)]
    if not ocean_df.empty:
        std_print("INFO", "pa", f"Plotting {len(ocean_track_ids)} ocean-origin tracks")
        plot_tracks_elevation(ocean_df, vrt_path, output_path, cmap_name=cmap_name)
    else:
        std_print("WARN", "pa", "No ocean-origin tracks found to plot.")


def fetch(url):
    try:
        with request.urlopen(url, timeout=15) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def export_interactive_map(all_df, output_path, vrt_path=None, cmap_name="inferno"):
    LEAFLET_VER = "1.9.4"
    leaflet_css = fetch(f"https://unpkg.com/leaflet@{LEAFLET_VER}/dist/leaflet.css")
    leaflet_js = fetch(f"https://unpkg.com/leaflet@{LEAFLET_VER}/dist/leaflet.js")
    std_print(
        "INFO", "pa", "Leaflet assets " + ("inlined" if leaflet_js else "CDN fallback")
    )

    years = sorted(all_df["year"].unique())
    year_colors = get_year_colors(years, cmap_name=cmap_name)

    MONTH_DASH = {
        "June": "4,7",
        "July": "12,4,3,4",
        "August": "10,6",
        "September": "",
    }
    MONTHS = ["June", "July", "August", "September"]

    tracks_list = []
    for (year, track_uid), tr in all_df.groupby(["year", "track_uid"]):
        tr = tr.sort_values("time")
        pts = tr[["latitude", "longitude"]].dropna().values.tolist()
        if len(pts) < 2:
            continue
        yr = int(year)
        _raw_mon = str(tr["month"].iloc[0]) if "month" in tr.columns else "Unknown"
        mon = _raw_mon if _raw_mon in MONTHS else "Other"

        # Handle origin
        ilv = tr.iloc[0].get("is_land", None)
        if ilv is None or pd.isna(ilv):
            orig = "Unknown"
        else:
            try:
                is_land_bool = bool(ilv)
            except:
                is_land_bool = str(ilv).strip().lower() == "true"
            orig = "Land" if is_land_bool else "Ocean"

        # Dash array (empty for Other, will be solid)
        dash = MONTH_DASH.get(mon, "")
        tracks_list.append(
            {
                "uid": str(track_uid),
                "year": yr,
                "month": mon,
                "origin": orig,
                "color": year_colors.get(yr, "#FF0000"),
                "dash": dash,
                "pts": pts,
                "start": pts[0],
                "end": pts[-1],
            }
        )

    mtns_list = [
        {"n": name, "lat": lat, "lon": lon}
        for name, (lat, lon) in MOUNTAIN_POINTS.items()
    ]

    # Elevation raster embedding
    elev_url_js = "null"
    elev_bnds_js = "null"
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
            rgba[..., 3] = 140
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

    # HTML snippets
    years_sorted = sorted(years)
    yr_rows = "".join(
        f'<label class="cb-row">'
        f'<input type="checkbox" class="yr-cb" value="{yr}" checked>'
        f'<span class="sw" style="background:{year_colors[yr]}"></span>{yr}</label>'
        for yr in years_sorted
    )
    mo_rows = ""
    for mo in MONTHS + ["Other"]:  # Other for non-JJAS months
        if mo == "September":
            svg = '<svg width="20" height="4" viewBox="0 0 20 4"><line x1="0" y1="2" x2="20" y2="2" stroke="#333" stroke-width="2"/></svg>'
        elif mo == "August":
            svg = '<svg width="20" height="4" viewBox="0 0 20 4"><line x1="0" y1="2" x2="20" y2="2" stroke="#333" stroke-width="2" stroke-dasharray="6,4"/></svg>'
        elif mo == "July":
            svg = '<svg width="20" height="4" viewBox="0 0 20 4"><line x1="0" y1="2" x2="20" y2="2" stroke="#333" stroke-width="2" stroke-dasharray="4,4,8,4"/></svg>'
        elif mo == "June":
            svg = '<svg width="20" height="4" viewBox="0 0 20 4"><line x1="0" y1="2" x2="20" y2="2" stroke="#333" stroke-width="2" stroke-dasharray="2,4"/></svg>'
        else:
            # Other: dotted line preview
            svg = '<svg width="20" height="4" viewBox="0 0 20 4"><line x1="0" y1="2" x2="20" y2="2" stroke="#333" stroke-width="2" stroke-dasharray="2,2"/></svg>'
        mo_rows += f'<label class="cb-row"><input type="checkbox" class="mo-cb" value="{mo}" checked>{svg}<span style="margin-left:4px">{mo}</span></label>'

    elev_ctrl = (
        (
            '<div class="sec">Elevation overlay</div><div class="slider-row">'
            '<span>SRTM Opacity</span><input type="range" id="elev-sl" min="0" max="100" value="55" style="width:82px">'
            '<span id="elev-pct">55%</span></div>'
        )
        if elev_url_js != "null"
        else '<div class="sec">Elevation overlay</div><span class="note">No VRT file</span>'
    )

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

    BOUNDS = [[7, 70], [30, 95]]

    TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Monsoon Depression - Interactive Track Map</title>
__LEAFLET_CSS__
<style>
html,body{height:100%;margin:0;padding:0;font-family:sans-serif}
#map{height:100vh;width:100%}
#ctrl{position:absolute;top:10px;right:10px;z-index:1000;background:rgba(255,255,255,.95);border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.3);padding:10px 12px;width:182px;max-height:92vh;overflow-y:auto;font-size:12px;line-height:1.5}
#ctrl h3{margin:0 0 5px;font-size:13px;border-bottom:1px solid #ccc;padding-bottom:4px}
.sec{font-weight:700;font-size:10.5px;color:#555;margin:8px 0 3px;text-transform:uppercase;letter-spacing:.05em}
.cb-row{display:flex;align-items:center;gap:4px;cursor:pointer;padding:1px 0}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;flex-shrink:0}
.slider-row{display:flex;align-items:center;gap:5px;margin-top:3px;flex-wrap:wrap}
.note{color:#aaa;font-size:11px}
.sel-btns{display:flex;gap:4px;margin-bottom:4px}
.sel-btns button{flex:1;font-size:10px;padding:2px 0;cursor:pointer;border:1px solid #ccc;border-radius:3px;background:#f5f5f5}
.sel-btns button:hover{background:#e0e0e0}
.mtn-tt{background:rgba(255,255,255,.92)!important;border:none!important;box-shadow:0 1px 4px rgba(0,0,0,.3)!important;font-size:11px;padding:2px 6px!important}
svg { vertical-align: middle; }
</style>
</head>
<body>
<div id="map"></div>
<div id="ctrl">
  <h3>MD Tracks</h3>
  <div class="sec">Base map</div>
  <label class="cb-row"><input type="radio" name="bm" value="carto" checked> Plain (CartoDB)</label>
  <label class="cb-row"><input type="radio" name="bm" value="topo"> Topo (OpenTopoMap)</label>
  __ELEV_CTRL__
  <div class="sec">Mountains</div>
  <label class="cb-row"><input type="checkbox" id="mtn-cb" checked> Show peaks ▲</label>
  <div class="sec">Year</div>
  <div class="sel-btns"><button onclick="setAll('.yr-cb',true)">All</button><button onclick="setAll('.yr-cb',false)">None</button></div>
  __YR_ROWS__
  <div class="sec">Month</div>
  __MO_ROWS__
  <div class="sec">Track origin</div>
  <label class="cb-row"><input type="checkbox" class="orig-cb" value="Ocean" checked><span class="sw" style="background:#1a78c2"></span>Ocean</label>
  <label class="cb-row"><input type="checkbox" class="orig-cb" value="Land" checked><span class="sw" style="background:#b5440a"></span>Land</label>
  <label class="cb-row"><input type="checkbox" class="orig-cb" value="Unknown" checked><span class="sw" style="background:#888"></span>Unknown</label>
</div>
__LEAFLET_JS__
<script>
var TRACKS    = __TRACKS_JSON__;
var MTNS      = __MTNS_JSON__;
var ELEV_URL  = __ELEV_URL__;
var ELEV_BNDS = __ELEV_BNDS__;

var map = L.map('map').setView([20,82],5);
map.setMaxBounds(__BOUNDS__);
map.on('drag', function() { map.panInsideBounds(__BOUNDS__, {animate: false}); });

var bCarto = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>', subdomains:'abcd',maxZoom:19}).addTo(map);
var bTopo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
  {attribution:'Map data: &copy; OSM, SRTM | Style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)',maxZoom:17});

var elevLyr=null;
if(ELEV_URL&&ELEV_BNDS) elevLyr=L.imageOverlay(ELEV_URL,ELEV_BNDS,{opacity:0.55,interactive:false}).addTo(map);

var gTracks = L.layerGroup().addTo(map);
var gMarkers= L.layerGroup().addTo(map);
var gMtns   = L.layerGroup().addTo(map);

function drawMtns(){
  gMtns.clearLayers();
  if(!document.getElementById('mtn-cb').checked)return;
  MTNS.forEach(function(m){
    var ico=L.divIcon({html:'<div style="font-size:11px;line-height:1;color:#333;text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff">▲</div>',className:'',iconSize:[12,12],iconAnchor:[6,11]});
    L.marker([m.lat,m.lon],{icon:ico,zIndexOffset:500}).bindTooltip(m.n,{direction:'top',className:'mtn-tt',offset:[0,-8]}).addTo(gMtns);
  });
}
drawMtns();

function activeVals(cls){
  var s={};
  document.querySelectorAll(cls+':checked').forEach(function(e){s[e.value]=1;});
  return s;
}

function redraw(){
  gTracks.clearLayers(); gMarkers.clearLayers();
  var aYr =activeVals('.yr-cb');
  var aMo =activeVals('.mo-cb');
  var aOrg=activeVals('.orig-cb');
  TRACKS.forEach(function(t){
    if(!aYr[t.year]||!aMo[t.month]||!aOrg[t.origin])return;
    L.polyline(t.pts,{color:t.color,weight:2.5,opacity:0.75,dashArray:t.dash||null})
     .bindTooltip(t.year+' | '+t.month+' | '+t.origin+'<br><small>'+t.uid+'</small>',{sticky:true}).addTo(gTracks);
    L.circleMarker(t.start,{radius:4,color:t.color,weight:2,fillColor:'#fff',fillOpacity:1})
     .bindTooltip('Start: '+t.uid).addTo(gMarkers);
    L.marker(t.end,{icon:L.divIcon({html:'<div style="width:7px;height:7px;background:'+t.color+';border:1.5px solid #fff;box-shadow:0 0 2px rgba(0,0,0,.4)"></div>',className:'',iconSize:[7,7],iconAnchor:[4,4]})})
     .bindTooltip('End: '+t.uid).addTo(gMarkers);
  });
}
redraw();

function setAll(cls,val){ document.querySelectorAll(cls).forEach(function(e){e.checked=val;}); redraw(); }
document.querySelectorAll('.yr-cb,.mo-cb,.orig-cb').forEach(function(e){e.addEventListener('change',redraw);});
document.getElementById('mtn-cb').addEventListener('change',drawMtns);
document.querySelectorAll('input[name="bm"]').forEach(function(el){
  el.addEventListener('change',function(){
    if(this.value==='topo'){map.removeLayer(bCarto);bTopo.addTo(map);}
    else{map.removeLayer(bTopo);bCarto.addTo(map);}
  });
});
var esl=document.getElementById('elev-sl');
if(esl&&elevLyr){ esl.addEventListener('input',function(){ elevLyr.setOpacity(this.value/100); document.getElementById('elev-pct').textContent=this.value+'%'; }); }
</script>
</body>
</html>"""

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
    p.add_argument(
        "--cmap",
        default="inferno",
        choices=[
            "viridis",
            "inferno",
            "bwr",
            "Spectral",
        ],
        help="Colormap for year colours (default: inferno)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_root = os.path.join("..", "Data", "single-levels", "ncfiles")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    std_print("INFO", "pa", f"Using output directory: {OUTPUT_DIR}")
    std_print("INFO", "pa", f"Using colour map: {args.cmap}")
    map_path = os.path.join(OUTPUT_DIR, "all_years_tracks_map.png")
    elev_path = os.path.join(OUTPUT_DIR, "all_years_tracks_elevation.png")
    vrt_path = os.path.join(OUTPUT_DIR, "elevation_mosaic.vrt")
    precip_output = os.path.join(OUTPUT_DIR, "mean_jjas_precipitation.png")

    std_print("INFO", "pa", f"Scanning '{args.output_root}' for output_* directories")
    all_df = load_all_tracks(args.output_root)
    if all_df.empty:
        std_print("ERROR", "pa", "No track data found. Exiting...")
        return

    print()
    std_print(
        "INFO",
        "pa",
        f"Total: {all_df['track_uid'].nunique()} tracks, {len(all_df)} points, "
        f"{all_df['year'].nunique()} years ({all_df['year'].min()}-{all_df['year'].max()})",
    )

    # Plain map
    plot_tracks_map(all_df, map_path, cmap_name=args.cmap)

    # Elevation map
    if not os.path.exists(vrt_path):
        std_print("WARN", "pa", "VRT not found. Creating ...")
        try:
            create_elevation_vrt(hgt_dir=args.HGT_DIR, vrt_filename=vrt_path)
            std_print("INFO", "pa", "VRT created successfully", vrt_path)
        except Exception as e:
            std_print("ERROR", "pa", "Failed to create VRT", str(e))
            vrt_path = None

    if vrt_path and os.path.exists(vrt_path):
        plot_tracks_elevation(all_df, vrt_path, elev_path, cmap_name=args.cmap)
        plot_ocean_origin_only(
            all_df, vrt_path, "Overall_years/ocean_only_tracks.png", cmap_name=args.cmap
        )
    else:
        std_print("ERROR", "pa", "Skipping elevation plots - no VRT available")

    # Interactive map
    folium_path = os.path.join(OUTPUT_DIR, "interactive_tracks_map.html")
    export_interactive_map(
        all_df,
        folium_path,
        vrt_path=vrt_path if (vrt_path and os.path.exists(vrt_path)) else None,
        cmap_name=args.cmap,
    )
    std_print("INFO", "pa", "All outputs saved", f"./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
