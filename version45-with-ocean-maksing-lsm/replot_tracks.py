"""
Redraw MD track maps from the 8-track detection CSV inside each BUF_* dir.

Produces:
  output_2024_BUF_{B}/MD_tracks_2024_8tracks.png       combined 8-track map
  output_2024_BUF_{B}/track_N/track_N_full_map.png     per-track map
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

# --- config ---
YEAR = 2024
BUFFERS = [100, 300, 500]
FULL_EXTENT = [70.0, 95.0, 7.0, 30.0]

FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT = 14, 12, 10, 10, 9
month_colors = {"June": "#FF6B6B", "July": "#FF4500",
                "August": "#1E90FF", "September": "#FF00FF"}

# --- draw one track on an existing GeoAxes ---
def draw_track(ax, g, color, label=True):
    g = g.sort_values("time").reset_index(drop=True)
    lons, lats = g["longitude"].values, g["latitude"].values

    # land/ocean colored polyline
    land = g["is_land"].values.astype(bool) if "is_land" in g else np.zeros(len(g), bool)
    for i in range(len(g) - 1):
        c = "saddlebrown" if land[i] else "dodgerblue"
        ax.plot(lons[i:i+2], lats[i:i+2], color=c, linewidth=1.6,
                solid_capstyle="round", transform=ccrs.PlateCarree(), zorder=3)

    # start / end
    ax.plot(lons[0], lats[0], "o", color="green", ms=7, mec="black",
            transform=ccrs.PlateCarree(), zorder=6)
    ax.plot(lons[-1], lats[-1], "o", color="orange", ms=7, mec="black",
            transform=ccrs.PlateCarree(), zorder=6)

    if label:
        tid = int(g["track_id"].iloc[0])
        ax.text(lons[0] - 0.3, lats[0], f"T{tid}", fontsize=FS_ANNOT,
                fontweight="bold", ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
                transform=ccrs.PlateCarree(), zorder=7)
        # first timestamp
        t0 = pd.to_datetime(g["time"].iloc[0]).strftime("%d/%m %H:%M")
        ax.text(lons[0] + 0.3, lats[0], t0, fontsize=FS_ANNOT - 1, ha="left",
                va="center", transform=ccrs.PlateCarree(), zorder=7)

# --- setup a cartopy axes ---
def setup_axes(fig, extent=FULL_EXTENT):
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.4)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, linewidth=0.7, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.4, linestyle="--", alpha=0.6)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    return ax

# --- combined map: all tracks in one figure ---
def combined_map(df, output_dir, year):
    fig = plt.figure(figsize=(14, 11))
    ax = setup_axes(fig)

    legend_handles = []
    for tid, g in df.groupby("track_id"):
        month = g.sort_values("time")["month"].iloc[0] if "month" in g else "August"
        color = month_colors.get(month, "black")
        draw_track(ax, g, color, label=True)
        # add one legend entry per track
        legend_handles.append(
            plt.Line2D([0], [0], color=color, lw=2, label=f"T{int(tid)} ({month})")
        )

    ax.legend(handles=legend_handles, loc="lower left", fontsize=FS_LEGEND,
              framealpha=0.9, ncol=2)
    ax.set_title(f"Detected MD Tracks ({year}) — 8 tracks",
                 fontsize=FS_TITLE, fontweight="bold", pad=10)
    plt.tight_layout()

    out = os.path.join(output_dir, f"MD_tracks_{year}_8tracks.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  SAVED  {out}")

# --- per-track map, full domain ---
def per_track_map(g, output_dir, year):
    tid = int(g["track_id"].iloc[0])
    g = g.sort_values("time").reset_index(drop=True)
    month = g["month"].iloc[0] if "month" in g else "August"
    color = month_colors.get(month, "black")

    fig = plt.figure(figsize=(13, 10))
    ax = setup_axes(fig)

    # full range within monsoon domain (same style as md_detect.py)
    draw_track(ax, g, color, label=False)

    # annotate every 12 h
    t0 = pd.to_datetime(g["time"].iloc[0])
    hrs = (pd.to_datetime(g["time"]) - t0).dt.total_seconds() / 3600
    for h in np.arange(0, hrs.max() + 12, 12):
        idx = int(np.argmin(np.abs(hrs.values - h)))
        pt = g.iloc[idx]
        ax.plot(pt["longitude"], pt["latitude"], "o", color=color,
                ms=4, transform=ccrs.PlateCarree(), zorder=5)
        ax.text(pt["longitude"] - 0.1, pt["latitude"] + 0.1,
                f"{int(h)}h {pd.to_datetime(pt['time']).strftime('%d/%m %H:%M')}",
                fontsize=FS_ANNOT - 1, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.8),
                transform=ccrs.PlateCarree(), zorder=7)

    # landfall marker (first ocean -> land transition)
    land = g["is_land"].values.astype(bool) if "is_land" in g else np.zeros(len(g), bool)
    trans = np.diff(land.astype(int))
    o2l = np.where(trans == 1)[0]
    if len(o2l):
        lf = g.iloc[o2l[0] + 1]
        ax.plot(lf["longitude"], lf["latitude"], "s", color="yellow", ms=8,
                mec="black", transform=ccrs.PlateCarree(), zorder=8, label="Landfall")
        ax.text(lf["longitude"] + 0.3, lf["latitude"] + 0.2,
                f"Landfall\n{pd.to_datetime(lf['time']).strftime('%d/%m %H:%M')}",
                fontsize=FS_ANNOT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="orange", alpha=0.9),
                transform=ccrs.PlateCarree(), zorder=9)

    start = pd.to_datetime(g["time"].iloc[0]).strftime("%d/%m %H:%M")
    end = pd.to_datetime(g["time"].iloc[-1]).strftime("%d/%m %H:%M")
    ax.set_title(f"MD Track {tid}  |  {start} → {end}, {year}",
                 fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax.legend(loc="lower left", fontsize=FS_LEGEND, framealpha=0.9)
    plt.tight_layout()

    out = os.path.join(output_dir, f"track_{tid}", f"track_{tid}_full_map.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  SAVED  {out}")

# --- main ---
def main():
    for B in BUFFERS:
        root = f"output_{YEAR}_BUF_{B}"
        csv = os.path.join(root, f"MD_tracks_{YEAR}.csv")
        if not os.path.exists(csv):
            print(f"skip {root} (no CSV)")
            continue
        print(f"\n=== BUFFER {B} km ===")
        df = pd.read_csv(csv, parse_dates=["time"])
        df["track_id"] = df["track_id"].astype(int)
        if "month" not in df.columns:
            df["month"] = df["time"].dt.strftime("%B")
        if "is_land" in df.columns:
            df["is_land"] = df["is_land"].astype(bool)

        combined_map(df, root, YEAR)
        for tid, g in df.groupby("track_id"):
            per_track_map(g.copy(), root, YEAR)

if __name__ == "__main__":
    main()
