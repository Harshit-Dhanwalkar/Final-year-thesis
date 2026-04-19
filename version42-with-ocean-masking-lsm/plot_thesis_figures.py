"""
Generates every figure needed for the thesis that are not already produced by
existing analysis pipeline.

    python plot_thesis_figures.py --output_root . --year 2024 --track 1

1. coriolis_md_rotation.png           -> Chapter 1
2. # boe_feedback_loop.png              -> Chapter 1
3. jet_streams_schematic.png          -> Chapter 2,  After Findlater (1969) and Mishra (2021)
4. ## ghats_blocking_schematic.png      -> Chapter 2  [cross-section schematic]
5. hovmoller_geometry.png             -> Chapter 3  [polar grid]
6. ## detection_flowchart.png           -> Chapter 3  [flowchart]
7. annual_md_count_boe.png            -> Chapter 4  [bar + line from CSVs]
8. duration_histogram.png             -> Chapter 4  [histogram from CSVs]
9. boe_vs_nonboe_hf_comparison.png    -> Chapter 4  [2-panel comparison]
10. bowen_regime_diagram.png           -> Chapter 4  [conceptual \\beta diagram]
"""

import argparse
import glob
import os
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from utils import std_print

warnings.filterwarnings("ignore")

OUTPUT_DIR = "Thesis_schematics"

# Map domain
FULL_EXTENT = [70.0, 95.0, 7.0, 30.0]  # lon_min, lon_max, lat_min, lat_max
JJAS_MONTHS = [6, 7, 8, 9]

# Font sizes
FS_TITLE = 14
FS_LABEL = 12
FS_TICK = 10
FS_LEGEND = 10
FS_ANNOT = 9


# Helper functions
def save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    std_print("SAVED", "pf", f"{name} Output Path", path)
    return path

# Schematic: ocean phase β ≈ 0.2, BOE stays low, non-BOE rises
def beta_boe(t):
    b = np.where(
        t < 0,
        0.20,
        np.where(t < 6, 0.20 + 0.05 * (t / 6), 0.25 + 0.08 * np.tanh(t / 15)),
    )
    return b

def beta_nonboe(t):
    b = np.where(
        t < 0,
        0.20,
        np.where(
            t < 6,
            0.20 + 0.3 * (t / 6),
            np.where(
                t < 24,
                0.50 + 0.035 * (t - 6),
                np.minimum(1.85, 1.13 + 0.02 * (t - 24)),
            ),
        ),
    )
    return b

# def box(x, y, w, h, text, color="#2166ac", textcolor="white"):
#     rect = FancyBboxPatch(
#         (x, y),
#         w,
#         h,
#         boxstyle="round,pad=0.15",
#         facecolor=color,
#         edgecolor="white",
#         linewidth=1.5,
#         zorder=3,
#     )
#     ax.add_patch(rect)
#     ax.text(
#         x + w / 2,
#         y + h / 2,
#         text,
#         ha="center",
#         va="center",
#         fontsize=FS_ANNOT,
#         fontweight="bold",
#         color=textcolor,
#         zorder=4,
#         multialignment="center",
#     )
#
# def arrow(x0, y0, x1, y1, color="#333333", label="", lw=2.0):
#     ax.annotate(
#         "",
#         xy=(x1, y1),
#         xytext=(x0, y0),
#         arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=18),
#         zorder=5,
#     )
#     if label:
#         mx, my = (x0 + x1) / 2, (y0 + y1) / 2
#         ax.text(
#             mx,
#             my + 0.18,
#             label,
#             ha="center",
#             fontsize=FS_ANNOT,
#             color=color,
#             path_effects=[pe.withStroke(linewidth=2, foreground="white")],
#         )

def load_all_track_csvs(output_root):
    """Aggregate MD_tracks_YYYY.csv across all output_YYYY dirs."""
    frames = []
    for d in sorted(glob.glob(os.path.join(output_root, "output_*"))):
        try:
            year = int(os.path.basename(d).replace("output_", ""))
        except ValueError:
            continue
        for fname in [f"MD_tracks_{year}.csv", "MD_coordinates_ALL.csv"]:
            fpath = os.path.join(d, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                df.columns = [c.strip().lower() for c in df.columns]
                df["year"] = year
                frames.append(df)
                break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

## Main ##

# Schematic figures
def fig_coriolis_schematic():
    """Coriolis force -> anti-clockwise cyclonic rotation schematic"""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.axis("off")
    ax.set_facecolor("#f8f8f8")

    # Low-pressure centre circle
    circ = Circle((0, 0), 0.35, color="#d6604d", zorder=5)
    ax.add_patch(circ)
    ax.text(
        0,
        0,
        "L",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="white",
        zorder=6,
    )

    # Anti-clockwise spiral arrows (4 quadrants)
    theta_starts = [0, 90, 180, 270]
    arrow_colors = ["#2166ac", "#2166ac", "#2166ac", "#2166ac"]
    for i, (ts, ac) in enumerate(zip(theta_starts, arrow_colors)):
        # Inward straight arrow
        ang_r = np.radians(ts + 45)
        x0 = 2.0 * np.cos(np.radians(ts))
        y0 = 2.0 * np.sin(np.radians(ts))
        dx = -0.9 * np.cos(np.radians(ts))
        dy = -0.9 * np.sin(np.radians(ts))
        ax.annotate(
            "",
            xy=(x0 + dx, y0 + dy),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=ac, lw=2.0, mutation_scale=18),
            zorder=4,
        )
        # Label "inflow"
        if i == 0:
            ax.text(
                x0 + 0.1,
                y0 + 0.15,
                "Inflow ->",
                fontsize=9.5,
                color=ac,
                ha="left",
                va="bottom",
            )

    # Large curved CCW arrow showing rotation
    theta = np.linspace(0.15, 2 * np.pi - 0.15, 200)
    r = 1.05
    ax.plot(
        r * np.cos(theta),
        r * np.sin(theta),
        color="#4a4a4a",
        linewidth=2.5,
        linestyle="-",
        zorder=3,
    )
    # Arrowhead at end of curve (CCW -> arrowhead slightly before 2π)
    ah_angle = 2 * np.pi - 0.30
    ax.annotate(
        "",
        xy=(r * np.cos(ah_angle), r * np.sin(ah_angle)),
        xytext=(r * np.cos(ah_angle - 0.18), r * np.sin(ah_angle - 0.18)),
        arrowprops=dict(arrowstyle="-|>", color="#4a4a4a", lw=2.0, mutation_scale=20),
        zorder=4,
    )

    # Coriolis deflection arrows (rightward deflection shown)
    for ang_deg in [45, 135, 225, 315]:
        ang = np.radians(ang_deg)
        x_start = 1.6 * np.cos(ang)
        y_start = 1.6 * np.sin(ang)
        # rightward deflection = -90° turn
        def_ang = ang - np.pi / 2
        dx = 0.35 * np.cos(def_ang)
        dy = 0.35 * np.sin(def_ang)
        ax.annotate(
            "",
            xy=(x_start + dx, y_start + dy),
            xytext=(x_start, y_start),
            arrowprops=dict(
                arrowstyle="-|>", color="#c51b7d", lw=1.5, mutation_scale=14
            ),
            zorder=5,
        )

    # Labels
    ax.text(
        0,
        -2.3,
        "Anti-clockwise (cyclonic) rotation\n"
        r"Coriolis: $f = 2\Omega\sin\phi$",
        ha="center",
        fontsize=11.5,
        color="#333333",
    )
    ax.text(
        1.85,
        0.55,
        "Coriolis\ndeflection\n(rightward)",
        ha="left",
        fontsize=9.5,
        color="#c51b7d",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )
    ax.set_title(
        "Coriolis Force: Why Monsoon Depressions Rotate Anti-Clockwise\n"
        "(Northern Hemisphere)",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=10,
    )
    return save(fig, "coriolis_md_rotation.png")


# def fig_boe_feedback_loop():
#     """BOE precipitation–soil moisture positive feedback loop."""
#     fig, ax = plt.subplots(figsize=(10, 7))
#     ax.set_xlim(0, 10)
#     ax.set_ylim(0, 7)
#     ax.axis("off")
#
#     # Ocean panel (left)
#     ax.add_patch(
#         FancyBboxPatch(
#             (0.1, 1.5),
#             2.4,
#             4.0,
#             boxstyle="round,pad=0.1",
#             facecolor="#d1e5f0",
#             edgecolor="#2166ac",
#             linewidth=1.5,
#             zorder=1,
#             alpha=0.5,
#         )
#     )
#     ax.text(
#         1.3,
#         5.65,
#         "OCEAN PHASE",
#         ha="center",
#         fontsize=FS_ANNOT,
#         fontweight="bold",
#         color="#2166ac",
#     )
#     box(0.3, 4.2, 2.0, 0.9, "BoB SST\n28–30 °C", color="#2166ac")
#     box(0.3, 2.9, 2.0, 0.9, "Evaporation\n100–150 W m⁻²", color="#2166ac")
#     box(0.3, 1.6, 2.0, 0.9, "Deep Convection\nsustained", color="#2166ac")
#     arrow(1.3, 4.2, 1.3, 3.8, color="#2166ac")
#     arrow(1.3, 2.9, 1.3, 2.5, color="#2166ac")
#
#     # Landfall arrow
#     ax.annotate(
#         "",
#         xy=(3.8, 3.5),
#         xytext=(2.7, 3.5),
#         arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2.5, mutation_scale=22),
#         zorder=5,
#     )
#     ax.text(
#         3.25,
#         3.75,
#         "LANDFALL",
#         ha="center",
#         fontsize=FS_ANNOT,
#         fontweight="bold",
#         color="#333333",
#     )
#     ax.text(3.25, 3.25, "->", ha="center", fontsize=FS_ANNOT, color="#333333")
#
#     # Land (BOE) panel (right)
#     ax.add_patch(
#         FancyBboxPatch(
#             (3.9, 0.4),
#             5.8,
#             5.8,
#             boxstyle="round,pad=0.1",
#             facecolor="#f6e8c3",
#             edgecolor="#8c510a",
#             linewidth=1.5,
#             zorder=1,
#             alpha=0.5,
#         )
#     )
#     ax.text(
#         6.8,
#         6.35,
#         "LAND PHASE - Brown Ocean Effect",
#         ha="center",
#         fontsize=FS_ANNOT,
#         fontweight="bold",
#         color="#8c510a",
#     )
#
#     # Four boxes in a cycle
#     box(4.2, 4.5, 2.5, 1.0, "MD Heavy\nRainfall", color="#4393c3")
#     box(
#         7.0,
#         4.5,
#         2.5,
#         1.0,
#         "SWVL1 -> Field\nCapacity",
#         color="#4dac26",
#         textcolor="white",
#     )
#     box(7.0, 2.0, 2.5, 1.0, "LHF ≥ 70 W m⁻²\n(BOE Active)", color="#d6604d")
#     box(4.2, 2.0, 2.5, 1.0, "Convection\nSustained", color="#4393c3")
#
#     # Cycle arrows
#     arrow(5.5, 4.5, 7.0, 5.0, color="#333333", label="wets soil")
#     arrow(8.25, 4.5, 8.25, 3.0, color="#333333", label="near saturation")
#     arrow(7.0, 2.5, 5.5, 2.1, color="#333333", label="sustains CAPE")
#     arrow(4.95, 2.0, 4.95, 4.5, color="#333333", label="drives precip")
#
#     # Antecedent SM entry arrow
#     ax.annotate(
#         "",
#         xy=(6.5, 4.5),
#         xytext=(6.5, 6.1),
#         arrowprops=dict(
#             arrowstyle="-|>",
#             color="#8c510a",
#             lw=1.8,
#             mutation_scale=15,
#             linestyle="dashed",
#         ),
#         zorder=5,
#     )
#     ax.text(
#         6.5,
#         6.15,
#         "Antecedent\nSM pre-conditions\n(ASMI ≥ 0.35 m³/m³)",
#         ha="center",
#         fontsize=FS_ANNOT,
#         color="#8c510a",
#         fontweight="bold",
#         path_effects=[pe.withStroke(linewidth=2, foreground="white")],
#     )
#
#     ax.set_title(
#         "Brown Ocean Effect: Precipitation–Soil Moisture Positive Feedback",
#         fontsize=FS_TITLE,
#         fontweight="bold",
#         pad=10,
#     )
#     ax.text(
#         5.0,
#         0.05,
#         "After Andersen & Shepherd (2013) and Kishtawal et al. (2013)",
#         ha="center",
#         fontsize=FS_TITLE,
#         style="italic",
#         color="#555555",
#     )
#     return save(fig, "boe_feedback_loop.png")


def fig_jet_streams_schematic():
    """Somali LLJ + STJ retreat + monsoon trough -> NW MD steering"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([45, 100, 0, 40], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#C8E8F5", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F5F0E8", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, zorder=3)

    # Somali LLJ (850 hPa southwesterly)
    llj_lons = [48, 55, 63, 70, 77, 80, 84, 87]
    llj_lats = [2, 6, 11, 16, 20, 22, 22, 21]
    for i in range(len(llj_lons) - 1):
        ax.annotate(
            "",
            xy=(llj_lons[i + 1], llj_lats[i + 1]),
            xytext=(llj_lons[i], llj_lats[i]),
            arrowprops=dict(
                arrowstyle="-|>", color="#2166ac", lw=3.5, mutation_scale=20
            ),
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
    ax.text(
        58,
        9,
        "Somali Low-Level Jet\n(Findlater Jet, 850 hPa)",
        fontsize=FS_ANNOT,
        color="#2166ac",
        fontweight="bold",
        transform=ccrs.PlateCarree(),
        ha="center",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )

    # Monsoon trough (dashed)
    tr_lon = [67, 72, 77, 82, 86, 88]
    tr_lat = [28, 27, 25, 24, 22, 20]
    ax.plot(
        tr_lon,
        tr_lat,
        color="#7f0000",
        linewidth=2.0,
        linestyle="--",
        dashes=(8, 4),
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.text(
        76,
        26.5,
        "Monsoon Trough",
        fontsize=FS_ANNOT,
        color="#7f0000",
        style="italic",
        transform=ccrs.PlateCarree(),
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )

    # MD symbol + NW track arrow
    ax.plot(
        87,
        19.5,
        "o",
        ms=18,
        color="#d6604d",
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=7,
    )
    ax.text(
        87,
        19.5,
        "MD",
        ha="center",
        va="center",
        fontsize=FS_ANNOT,
        fontweight="bold",
        color="white",
        transform=ccrs.PlateCarree(),
        zorder=8,
    )
    ax.annotate(
        "",
        xy=(79, 24),
        xytext=(87, 19.5),
        arrowprops=dict(arrowstyle="-|>", color="#d6604d", lw=2.5, mutation_scale=20),
        transform=ccrs.PlateCarree(),
        zorder=7,
    )
    ax.text(
        82,
        23.5,
        "NW track",
        fontsize=FS_ANNOT,
        color="#d6604d",
        transform=ccrs.PlateCarree(),
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )

    # STJ (retreated poleward of Himalayas at 200 hPa)
    stj_lon = np.linspace(50, 100, 50)
    stj_lat = np.full_like(stj_lon, 37) + 2 * np.sin(np.radians(stj_lon) * 3)
    ax.plot(
        stj_lon,
        stj_lat,
        color="#f4a582",
        linewidth=2.8,
        linestyle="-",
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(stj_lon[-1], stj_lat[-1]),
        xytext=(stj_lon[-3], stj_lat[-3]),
        arrowprops=dict(arrowstyle="-|>", color="#f4a582", lw=2.0, mutation_scale=16),
        transform=ccrs.PlateCarree(),
    )
    ax.text(
        72,
        38.5,
        "Sub-Tropical Westerly Jet (200 hPa) -> retreated poleward (low shear over BoB)",
        fontsize=FS_ANNOT,
        color="#c47a3a",
        transform=ccrs.PlateCarree(),
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )

    # Himalayan arc
    him_lon = np.linspace(73, 97, 50)
    him_lat = 28 + 4 * np.sin(np.pi * (him_lon - 73) / 24)
    ax.plot(
        him_lon,
        him_lat,
        color="#6a3d9a",
        linewidth=2.5,
        linestyle="-",
        transform=ccrs.PlateCarree(),
        zorder=4,
        alpha=0.7,
    )
    ax.text(
        82,
        31,
        "Himalayas",
        fontsize=FS_ANNOT,
        color="#6a3d9a",
        transform=ccrs.PlateCarree(),
        style="italic",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )

    # Gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="gray", alpha=0.4, linestyle=":"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}

    # Legend
    handles = [
        Line2D([0], [0], color="#2166ac", lw=3.5, label="Somali LLJ (850 hPa)"),
        Line2D(
            [0],
            [0],
            color="#f4a582",
            lw=2.8,
            label="Sub-Tropical Jet (200 hPa, retreated N)",
        ),
        Line2D(
            [0],
            [0],
            color="#7f0000",
            lw=2.0,
            ls="--",
            dashes=(8, 4),
            label="Monsoon trough axis",
        ),
        Line2D([0], [0], color="#d6604d", lw=2.5, label="MD NW track direction"),
    ]
    ax.legend(
        handles=handles, loc="lower left", fontsize=FS_LEGEND, framealpha=0.9, edgecolor="0.4"
    )

    ax.set_title(
        "Monsoon Jet Streams Controlling MD Development and North-Westward Track",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=10,
    )
    return save(fig, "jet_streams_schematic.png")


# def fig_ghats_cross_section():
#     """East-West cross-section showing all 4 Ghats blocking mechanisms"""
#     fig, ax = plt.subplots(figsize=(13, 7))
#     ax.set_xlim(0, 13)
#     ax.set_ylim(-0.5, 7.5)
#     ax.axis("off")
#
#     # Terrain profile
#     x_coast = 4.5
#     x_crest = 6.5
#     x_leeward = 13.0
#     terrain_x = [0, x_coast, x_coast, x_crest - 0.3, x_crest, x_crest + 0.3, 13]
#     terrain_y = [0, 0, 0.05, 1.5, 1.8, 0.8, 0.1]
#     ax.fill_between(terrain_x, terrain_y, color="#c7a97c", alpha=0.7, zorder=2)
#     ax.fill_between(
#         [0, x_coast], [0, 0], [-0.5, -0.5], color="#C8E8F5", alpha=0.6, zorder=1
#     )
#     ax.fill_between(
#         [x_coast, 13], [0.05, 0.1], [-0.5, -0.5], color="#F5F0E8", alpha=0.5, zorder=1
#     )
#
#     ax.text(
#         x_crest,
#         1.95,
#         "Eastern Ghats\n600–1000 m",
#         ha="center",
#         fontsize=FS_ANNOT,
#         fontweight="bold",
#         color="#8c510a",
#     )
#     ax.text(
#         x_coast / 2,
#         -0.35,
#         "Bay of Bengal",
#         ha="center",
#         fontsize=FS_ANNOT,
#         color="#2166ac",
#         style="italic",
#     )
#     ax.text(
#         9.5,
#         -0.35,
#         "Deccan Plateau / Gangetic Plains",
#         ha="center",
#         fontsize=FS_ANNOT,
#         color="#555555",
#         style="italic",
#     )
#
#     # MD vortex symbols
#     # Ocean MD
#     circ_o = Circle((1.5, 2.0), 0.55, color="#4393c3", alpha=0.8, zorder=5)
#     ax.add_patch(circ_o)
#     ax.text(
#         1.5,
#         2.0,
#         "MD",
#         ha="center",
#         va="center",
#         fontsize=FS_ANNOT,
#         fontweight="bold",
#         color="white",
#         zorder=6,
#     )
#     ax.annotate(
#         "",
#         xy=(3.5, 2.2),
#         xytext=(2.1, 2.1),
#         arrowprops=dict(arrowstyle="-|>", color="#4393c3", lw=2.5, mutation_scale=16),
#         zorder=6,
#     )
#
#     # Terrain (stalled) MD
#     circ_t = Circle((6.0, 2.5), 0.45, color="#d6604d", alpha=0.8, zorder=5)
#     ax.add_patch(circ_t)
#     ax.text(
#         6.0,
#         2.5,
#         "MD",
#         ha="center",
#         va="center",
#         fontsize=FS_ANNOT,
#         fontweight="bold",
#         color="white",
#         zorder=6,
#     )
#
#     # Leeward MD (decoupled upper)
#     circ_l = Circle((9.5, 3.8), 0.4, color="#f4a582", alpha=0.8, zorder=5)
#     ax.add_patch(circ_l)
#     ax.text(
#         9.5,
#         3.8,
#         "Upper",
#         ha="center",
#         va="center",
#         fontsize=FS_ANNOT,
#         color="#7f0000",
#         zorder=6,
#     )
#     circ_ll = Circle((8.8, 2.1), 0.4, color="#d6604d", alpha=0.6, zorder=5)
#     ax.add_patch(circ_ll)
#     ax.text(
#         8.8, 2.1, "Low", ha="center", va="center", fontsize=FS_ANNOT, color="white", zorder=6
#     )
#     # Tilt arrow
#     ax.annotate(
#         "",
#         xy=(9.5, 3.4),
#         xytext=(8.8, 2.5),
#         arrowprops=dict(arrowstyle="->", color="#c51b7d", lw=1.8, mutation_scale=14),
#         zorder=7,
#     )
#     ax.text(9.7, 2.9, "Tilt\n(Mech.4)", fontsize=FS_ANNOT, color="#c51b7d")
#
#     # Mechanism labels
#     # M1: Orographic precip enhancement
#     ax.annotate(
#         "",
#         xy=(5.5, 4.0),
#         xytext=(5.5, 5.5),
#         arrowprops=dict(arrowstyle="-|>", color="#2166ac", lw=2.0, mutation_scale=14),
#     )
#     ax.text(
#         5.5,
#         5.6,
#         "Orographic precip\nenhancement",
#         ha="center",
#         fontsize=FS_ANNOT,
#         color="#2166ac",
#         fontweight="bold",
#     )
#     # Rain symbols
#     for rx in [4.8, 5.2, 5.5, 5.8]:
#         ax.text(rx, 3.5, "💧", fontsize=FS_ANNOT, ha="center", zorder=8)
#
#     # M2: Speed deceleration
#     ax.annotate(
#         "",
#         xy=(6.5, 4.2),
#         xytext=(7.2, 5.2),
#         arrowprops=dict(arrowstyle="-|>", color="#8c510a", lw=2.0, mutation_scale=14),
#     )
#     ax.text(
#         7.5,
#         5.3,
#         "Deceleration\n(TSI; Mech.2)",
#         ha="left",
#         fontsize=FS_ANNOT,
#         color="#8c510a",
#         fontweight="bold",
#     )
#
#     # M3: BoB moisture blocked
#     ax.annotate(
#         "",
#         xy=(x_coast, 1.5),
#         xytext=(3.0, 1.5),
#         arrowprops=dict(arrowstyle="-|>", color="#4393c3", lw=2.5, mutation_scale=16),
#     )
#     ax.text(
#         2.0,
#         1.7,
#         "BoB moisture inflow\n-> BLOCKED at Ghats (Mech.3)",
#         fontsize=FS_ANNOT,
#         color="#4393c3",
#     )
#
#     # M4: Vortex decoupling label
#     ax.text(
#         10.5,
#         5.0,
#         "Vortex tilt\nUpper core decouples\nfrom surface low",
#         fontsize=FS_ANNOT,
#         color="#c51b7d",
#         fontweight="bold",
#         bbox=dict(facecolor="#fff0f3", edgecolor="#c51b7d", boxstyle="round,pad=0.3"),
#     )
#     ax.annotate(
#         "",
#         xy=(9.5, 4.2),
#         xytext=(10.5, 4.8),
#         arrowprops=dict(arrowstyle="-|>", color="#c51b7d", lw=1.5, mutation_scale=13),
#     )
#
#     # BOE counter label
#     ax.text(
#         9.5,
#         0.5,
#         "BOE: Sustained LHF from wet soil\npartially offsets Mechanisms 3 & 4",
#         fontsize=FS_ANNOT,
#         color="#4dac26",
#         ha="center",
#         fontweight="bold",
#         bbox=dict(facecolor="#f0fff0", edgecolor="#4dac26", boxstyle="round,pad=0.4"),
#     )
#
#     # Direction arrow
#     ax.annotate(
#         "",
#         xy=(12.5, 0.3),
#         xytext=(0.5, 0.3),
#         arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2),
#     )
#     ax.text(
#         6.5,
#         0.05,
#         "<- East (BoB coast)      West (Deccan) ->",
#         ha="center",
#         fontsize=FS_ANNOT,
#         color="#888888",
#     )
#
#     ax.set_title(
#         "Eastern Ghats: Four Mechanisms of MD Orographic Modulation\n"
#         "After Houze (1994), Kirshbaum & Durran (2005)",
#         fontsize=FS_TITLE,
#         fontweight="bold",
#         pad=10,
#     )
#     return save(fig, "ghats_blocking_schematic.png")


def fig_hovmoller_geometry():
    """Polar grid showing azimuthal Hovm\"oller construction"""
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_facecolor("#f9f9f9")

    radii = np.arange(0, 401, 25)
    n_sectors = 16
    sector_angles = np.linspace(0, 2 * np.pi, n_sectors + 1)

    # Radial rings
    theta_full = np.linspace(0, 2 * np.pi, 360)
    for r in radii[1:]:
        ax.plot(theta_full, [r] * 360, color="#aaaaaa", linewidth=0.5, alpha=0.6)

    # Sector dividers
    for sa in sector_angles:
        ax.plot([sa, sa], [0, 400], color="#aaaaaa", linewidth=0.5, alpha=0.6)

    # Highlight 2 example sectors at r=100 km
    for s_i in [2, 3]:
        th = np.linspace(sector_angles[s_i], sector_angles[s_i + 1], 50)
        inner_r = 75
        outer_r = 125
        th_fill = np.concatenate([th, th[::-1]])
        r_fill = np.concatenate([[outer_r] * 50, [inner_r] * 50])
        ax.fill(th_fill, r_fill, color="#2166ac", alpha=0.35, zorder=3)

    # Storm centre
    ax.plot([0], [0], marker="o", ms=18, color="#d6604d", zorder=8)

    # Radius labels
    for r in [100, 200, 300, 400]:
        ax.text(
            np.radians(15),
            r + 12,
            f"{r} km",
            fontsize=FS_ANNOT,
            ha="left",
            va="bottom",
            color="#555555",
        )

    # Sector label
    ax.text(
        sector_angles[2] + 0.10,
        100,
        "Sector\n(22.5°)",
        fontsize=FS_ANNOT,
        color="#2166ac",
        fontweight="bold",
        ha="left",
        va="center",
    )

    # Annotations
    ax.text(
        np.radians(285),
        230,
        r"16 sectors \times 17 annuli\n= 272 cells per time step",
        fontsize=FS_ANNOT,
        ha="center",
        va="center",
        color="#333333",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )

    ax.set_rticks([100, 200, 300, 400])
    ax.set_rlabel_position(200)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_thetagrids(
        np.degrees(sector_angles[:-1]),
        [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ],
        fontsize=FS_ANNOT-1,
    )
    ax.set_title(
        r"Azimuthal Hovm$\"o$ller Grid",
        # r"25\u202fkm annuli to 400\u202fkm | 16 sectors \times 22.5°",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=18,
    )
    return save(fig, "hovmoller_geometry.png")


# def fig_detection_flowchart():
#     """MD detection algorithm flowchart"""
#     fig, ax = plt.subplots(figsize=(9, 12))
#     ax.set_xlim(0, 9)
#     ax.set_ylim(0, 12)
#     ax.axis("off")
#
#     def box(
#         x,
#         y,
#         w,
#         h,
#         text,
#         color="#2166ac",
#         tcolor="white",
#         fs=10.5,
#         style="round,pad=0.2",
#     ):
#         r = FancyBboxPatch(
#             (x, y),
#             w,
#             h,
#             boxstyle=style,
#             facecolor=color,
#             edgecolor="white",
#             linewidth=1.5,
#             zorder=3,
#         )
#         ax.add_patch(r)
#         ax.text(
#             x + w / 2,
#             y + h / 2,
#             text,
#             ha="center",
#             va="center",
#             fontsize=FS_ANNOT,
#             fontweight="bold",
#             color=tcolor,
#             zorder=4,
#             multialignment="center",
#         )
#
#     def arr(x0, y0, x1, y1, label=""):
#         ax.annotate(
#             "",
#             xy=(x1, y1),
#             xytext=(x0, y0),
#             arrowprops=dict(
#                 arrowstyle="-|>", color="#444444", lw=1.8, mutation_scale=16
#             ),
#             zorder=5,
#         )
#         if label:
#             ax.text(
#                 (x0 + x1) / 2 + 0.15, (y0 + y1) / 2, label, fontsize=FS_LABEL, color="#444444"
#             )
#
#     # Boxes (y positions from bottom)
#     box(2.5, 10.5, 4, 0.9, "ERA5 850 hPa\nRelative Vorticity (hourly)", color="#4393c3")
#     box(
#         2.5,
#         9.0,
#         4,
#         0.9,
#         "① ζ₈₅₀ ≥ 4.0 × 10⁻⁵ s⁻¹\n(Vorticity Threshold)",
#         color="#2166ac",
#     )
#     box(2.5, 7.5, 4, 0.9, "② Local Max within\n3° × 3° neighbourhood", color="#2166ac")
#     box(2.5, 6.0, 4, 0.9, "③ Closed MSLP\nCirculation (r < 400 km)", color="#2166ac")
#     box(2.5, 4.5, 4, 0.9, "④ Persistence ≥ 12 h", color="#2166ac")
#     box(2.5, 3.0, 4, 0.9, "⑤ Domain: 10–30°N, 70–95°E", color="#2166ac")
#     box(
#         2.5,
#         1.5,
#         4,
#         0.9,
#         "Track Link (NN ≤ 5°/6 h)\n-> Land/Sea Classification",
#         color="#5aae61",
#     )
#     box(2.5, 0.1, 4, 0.9, "IMD Cross-Validation\n(±24 h, ±2° lat/lon)", color="#d6604d")
#
#     # Arrows
#     arr(4.5, 10.5, 4.5, 9.9)
#     arr(4.5, 9.0, 4.5, 8.4)
#     arr(4.5, 7.5, 4.5, 6.9)
#     arr(4.5, 6.0, 4.5, 5.4)
#     arr(4.5, 4.5, 4.5, 3.9)
#     arr(4.5, 3.0, 4.5, 2.4)
#     arr(4.5, 1.5, 4.5, 1.0)
#
#     # Reject branches
#     for y, text in [
#         (9.45, "Reject"),
#         (7.95, "Reject"),
#         (6.45, "Reject"),
#         (4.95, "Reject"),
#         (3.45, "Reject"),
#     ]:
#         ax.annotate(
#             "",
#             xy=(7.2, y),
#             xytext=(6.5, y),
#             arrowprops=dict(
#                 arrowstyle="-|>", color="#999999", lw=1.2, mutation_scale=12
#             ),
#         )
#         ax.text(7.25, y, text, fontsize=FS_ANNOT, color="#999999", va="center")
#
#     ax.set_title(
#         "ERA5 Monsoon Depression Detection Algorithm",
#         fontsize=FS_TITLE,
#         fontweight="bold",
#         pad=10,
#     )
#     return save(fig, "detection_flowchart.png")


#  Data-driven figures (read from output_* csvs)
def fig_annual_count(output_root):
    """Annual MD count + BOE fraction time series"""
    df = load_all_track_csvs(output_root)
    if df.empty:
        std_print("WARN", "tf", "No track data - skipping (annual count)")
        return

    # Deduplicate to one row per track
    id_col = "track_uid" if "track_uid" in df.columns else "track_id"
    if id_col not in df.columns:
        std_print("WARN", "tf", "No track_id column - skipping")
        return

    dur_col = "duration_h"
    if dur_col not in df.columns:
        # compute duration from time if available
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            grp = df.groupby(["year", id_col])["time"]
            df_dur = (grp.max() - grp.min()).dt.total_seconds() / 3600
            df_dur = df_dur.reset_index().rename(columns={"time": dur_col})
            df = df.drop_duplicates(subset=["year", id_col])
            df = df.merge(df_dur, on=["year", id_col], how="left")
        else:
            std_print("WARN", "tf", "No duration_h or time column - cannot compute duration")
            return

    # BOE column (if available)
    boe_col = "boe_candidate"
    if boe_col not in df.columns and dur_col in df.columns:
        # proxy: long-lived (>=96h) as BOE proxy
        df[boe_col] = df[dur_col] >= 96
        std_print("WARN", "tf", "Using long-lived criterion as BOE proxy")

    df_trk = df.drop_duplicates(subset=["year", id_col]).copy()

    # Aggregate annual count and BOE fraction
    if boe_col in df_trk:
        annual = (
            df_trk.groupby("year")
            .agg(count=(id_col, "nunique"), boe_frac=(boe_col, "mean"))
            .reset_index()
        )
    else:
        annual = (
            df_trk.groupby("year")
            .agg(count=(id_col, "nunique"))
            .reset_index()
        )
        annual["boe_frac"] = 0

    fig, ax1 = plt.subplots(figsize=(12, 5))
    years = annual["year"].values
    ax1.bar(years, annual["count"], color="#4393c3", alpha=0.75, label="Annual MD count")
    ax1.set_xlabel("Year", fontsize=FS_LABEL)
    ax1.set_ylabel("MD Count per Season", fontsize=FS_LABEL, color="#4393c3")
    ax1.tick_params(axis="y", labelcolor="#4393c3")

    ax2 = ax1.twinx()
    ax2.plot(
        years,
        annual["boe_frac"] * 100,
        "o-",
        color="#d6604d",
        linewidth=2.2,
        markersize=7,
        label="Long-lived fraction (≥96 h, %)",
    )
    ax2.set_ylabel("Long-lived / BOE fraction (%)", fontsize=FS_LABEL, color="#d6604d")
    ax2.tick_params(axis="y", labelcolor="#d6604d")
    ax2.set_ylim(0, 100)

    plt.xticks(years, rotation=45, ha="right", fontsize=FS_TICK)
    ax1.set_title(
        "Annual Monsoon Depression Count and Long-Lived Fraction\n"
        f"ERA5, {years[0]}–{years[-1]}",
        fontsize=FS_TITLE,
        fontweight="bold",
    )

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=FS_LEGEND, framealpha=0.9
    )
    plt.tight_layout()
    return save(fig, "annual_md_count_boe.png")


def fig_duration_histogram(output_root):
    """Duration frequency histogram with 96 h threshold"""
    df = load_all_track_csvs(output_root)
    if df.empty:
        std_print("WARN", "tf", "No data - skipping duration histogram")
        return

    id_col = "track_uid" if "track_uid" in df.columns else "track_id"
    if "duration_h" not in df.columns and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        dur = (
            df.groupby(["year", id_col])["time"]
            .apply(lambda t: (t.max() - t.min()).total_seconds() / 3600)
            .reset_index()
            .rename(columns={"time": "duration_h"})
        )
        df = df.drop_duplicates(subset=["year", id_col]).merge(
            dur, on=["year", id_col], how="left"
        )

    if "duration_h" not in df.columns:
        std_print("WARN", "tf", "No duration data - skipping histogram")
        return

    durs = df.drop_duplicates(subset=["year", id_col])["duration_h"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, durs.max() + 24, 24)
    # Compute histogram counts and bin edges
    counts, bin_edges = np.histogram(durs, bins=bins)
    # Assign colors per bin: blue for bins <96, red for >=96
    colors = ["#4393c3" if b < 96 else "#d6604d" for b in bin_edges[:-1]]
    # Draw bars manually to allow per‑bar coloring
    ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges),
           align="edge", color=colors, edgecolor="white",
           linewidth=0.6, alpha=0.85)

    ax.axvline(
        96,
        color="#7f0000",
        linewidth=2.0,
        linestyle="--",
        dashes=(7, 3),
        label="96\u202fh classification threshold",
    )
    ax.set_xlabel("Post-landfall duration (hours)", fontsize=FS_LABEL)
    ax.set_ylabel("Number of MD events", fontsize=FS_LABEL)
    ax.set_title(
        "Monsoon Depression Duration Distribution (2001–2025)\n"
        "Blue: short-lived (<96\u202fh)  |  Red: long-lived (≥96\u202fh)",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax.legend(
        handles=[
            Patch(facecolor="#4393c3", alpha=0.85, label="Short-lived (<96 h)"),
            Patch(
                facecolor="#d6604d",
                alpha=0.85,
                label="Long-lived (≥96 h, primary focus)",
            ),
            Line2D(
                [0],
                [0],
                color="#7f0000",
                lw=2.0,
                ls="--",
                dashes=(7, 3),
                label="96 h threshold",
            ),
        ],
        fontsize=FS_LEGEND,
        loc="upper right",
    )
    plt.tight_layout()
    return save(fig, "duration_histogram.png")


def fig_bowen_regime_diagram():
    """Conceptual Bowen ratio regime diagram (ocean -> BOE -> non-BOE)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    t = np.linspace(-48, 72, 300)  # hours relative to landfall

    ax.plot(t, beta_boe(t), color="#4393c3", lw=2.8, label="BOE candidate (long-lived)")
    ax.plot(t, beta_nonboe(t), color="#d6604d", lw=2.8, label="Non-BOE (short-lived)")

    # Reference lines
    ax.axhline(0.3, color="#888888", lw=1.0, ls=":", label="Oceanic value (β ≈ 0.3)")
    ax.axhline(0.5, color="#4dac26", lw=1.2, ls="--", label="BOE upper bound (β ≤ 0.5)")
    ax.axhline(
        1.0, color="#7f0000", lw=1.5, ls="--", label="Sensible/latent parity (β = 1.0)"
    )
    ax.axvline(
        0, color="#333333", lw=1.5, ls="--", dashes=(8, 3), label="Landfall (t = 0)"
    )

    # Phase shading
    ax.axvspan(-48, 0, color="#d1e5f0", alpha=0.25, label="_ocean")
    ax.axvspan(0, 72, color="#f6e8c3", alpha=0.25, label="_land")
    ax.text(
        -24,
        1.7,
        "Ocean Phase",
        ha="center",
        fontsize=FS_ANNOT,
        color="#2166ac",
        style="italic",
    )
    ax.text(
        36,
        1.7,
        "Land Phase",
        ha="center",
        fontsize=FS_ANNOT,
        color="#8c510a",
        style="italic"
    )

    # Regime annotations
    ax.fill_between(t[t >= 0], 0, 0.5, alpha=0.08, color="#4393c3", label="_boe_zone")
    ax.text(
        50,
        0.22,
        "Energy-limited\n(BOE regime)",
        fontsize=FS_ANNOT,
        color="#4393c3",
        ha="center",
    )
    ax.fill_between(t[t >= 0], 1.0, 2.0, alpha=0.06, color="#d6604d")
    ax.text(
        50,
        1.45,
        "Water-limited\n(Sensible heating)",
        fontsize=FS_ANNOT,
        color="#d6604d",
        ha="center",
    )

    ax.set_xlabel("Time relative to landfall (hours)", fontsize=FS_LABEL)
    ax.set_ylabel(r"Bowen ratio  $\beta = Q_H / Q_E$", fontsize=FS_LABEL)
    ax.set_ylim(0, 2.0)
    ax.set_xlim(-48, 72)
    ax.set_title(
        "Bowen Ratio Trajectory: BOE vs Conventional Post-Landfall Decay\n"
        "After Seneviratne et al. (2010)",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax.legend(fontsize=FS_LEGEND, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    return save(fig, "bowen_regime_diagram.png")


#  CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate all thesis schematic and data figures"
    )
    p.add_argument(
        "--output_root", default=".", help="Root dir with output_YYYY folders"
    )
    p.add_argument(
        "--schematics_only",
        action="store_true",
        help="Only generate schematic figures (no CSV data needed)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    # Schematics (no data)
    std_print("INFO", "tf", "[1/7] Coriolis schematic")
    fig_coriolis_schematic()

    # std_print("INFO", "tf", "[2/7] BOE feedback loop")
    # fig_boe_feedback_loop()

    std_print("INFO", "tf", "[3/7] Jet streams schematic")
    fig_jet_streams_schematic()

    # std_print("INFO", "tf", "[4/7] Eastern Ghats cross-section")
    # fig_ghats_cross_section()

    std_print("INFO", "tf", "[5/7] Hovmöller polar grid")
    fig_hovmoller_geometry()

    # std_print("INFO", "tf", "[6/7] Detection flowchart")
    # fig_detection_flowchart()

    std_print("INFO", "tf", "[7/7] Bowen ratio regime diagram")
    fig_bowen_regime_diagram()

    if not args.schematics_only:
        std_print("INFO", "tf", "[8] Annual count")
        fig_annual_count(args.output_root)

        std_print("INFO", "tf", "[9] Duration histogram")
        fig_duration_histogram(args.output_root)

    std_print("INFO", "tf", f"All figures saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
