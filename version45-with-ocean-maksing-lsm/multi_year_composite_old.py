# NOTE:
# Multi-Year Landfall Composite - elevation (eastern ghat) Survivors vs Non-Survivors
#
# Outputs (saved to composite_output/):
#   composite_msl_hpa.png
#   composite_speed_kmh.png
#   composite_all_variables.png   (4-panel combined)
#   composite_tracks_all_years.csv

# FIX: use std_print

import os
import glob
import argparse
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Ghats survivor registry
# (year, track_id) pairs confirmed from elevation analysis.
GHATS_SURVIVORS = {
    (2005, 6),   # T6, September 2005, Short-lived
    (2017, 4),   # T4, August 2017,    Short-lived
    (2021, 5),   # T5, September 2021, Long-lived
    (2024, 4),   # T4, August 2024,    Short-lived
}

# Style constants
SURVIVOR_COLOR     = "#2ca02c"   # green
NON_SURVIVOR_COLOR = "#1f77b4"   # blue
MEAN_ALPHA         = 0.95
SPAGHETTI_ALPHA    = 0.45
SPAGHETTI_LW       = 1.2
MEAN_LW            = 2.8
FS_TITLE  = 14
FS_LABEL  = 12
FS_LEGEND = 10
FS_ANNOT  = 9


# Data loading
def load_all_years(root_dir: str = ".") -> pd.DataFrame:
    """
    Glob output_*/MD_tracks_*.csv, load each, tag with year.
    """
    pattern = os.path.join(root_dir, "output_*", "MD_tracks_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No MD_tracks_*.csv files found under {root_dir}/output_*\n"
            f"Pattern used: {pattern}"
        )

    frames = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            # Extract year from directory name  output_YEAR
            year = int(os.path.basename(os.path.dirname(fpath)).replace("output_", ""))
            df["year"] = year
            frames.append(df)
            print(f"  Loaded {fpath}  ({len(df)} rows, {df['track_id'].nunique()} tracks)")
        except Exception as e:
            print(f"  WARN: Could not load {fpath}: {e}")

    if not frames:
        raise ValueError("All CSV files failed to load.")

    combined = pd.concat(frames, ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"])
    print(f"\nTotal rows: {len(combined)}  |  Years: {sorted(combined['year'].unique())}")
    return combined


# Filter: ocean-originated + has landfall
def _has_landfall(group: pd.DataFrame) -> bool:
    """True when there is at least one ocean -> land transition."""
    if "is_land" not in group.columns:
        return False
    land = group.sort_values("time")["is_land"].values.astype(bool)
    for i in range(1, len(land)):
        if not land[i - 1] and land[i]:
            return True
    return False


def _is_ocean_originated(group: pd.DataFrame) -> bool:
    """True when the first point is over ocean."""
    if "is_land" not in group.columns:
        return False
    first_land = group.sort_values("time")["is_land"].iloc[0]
    return not bool(first_land)


def filter_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ocean-originated tracks with at least one landfall."""
    records = []
    for (year, tid), grp in df.groupby(["year", "track_id"]):
        grp = grp.sort_values("time").reset_index(drop=True)
        if _is_ocean_originated(grp) and _has_landfall(grp):
            records.append(grp)

    if not records:
        raise ValueError("No ocean+landfall tracks found across all years.")

    eligible = pd.concat(records, ignore_index=True)
    n_tracks = eligible.groupby(["year", "track_id"]).ngroups
    print(f"Eligible (ocean + landfall): {n_tracks} tracks")
    return eligible


# Build landfall-relative time series (\pm window_h)
def _landfall_time(group: pd.DataFrame) -> pd.Timestamp | None:
    """Return timestamp of first ocean -> land transition."""
    land = group["is_land"].values.astype(bool)
    for i in range(1, len(land)):
        if not land[i - 1] and land[i]:
            return group["time"].iloc[i]
    return None


def build_relative_series(eligible: pd.DataFrame, window_h: int = 24) -> list[dict]:
    """
    For every eligible track, compute hours relative to landfall and
    slice to ±window_h.
    """
    records = []
    for (year, tid), grp in eligible.groupby(["year", "track_id"]):
        grp = grp.sort_values("time").reset_index(drop=True)
        lf_time = _landfall_time(grp)
        if lf_time is None:
            continue

        is_survivor  = (int(year), int(tid)) in GHATS_SURVIVORS
        duration_h   = (grp["time"].max() - grp["time"].min()).total_seconds() / 3600
        is_long      = duration_h >= 96.0

        for _, row in grp.iterrows():
            rel_h = (row["time"] - lf_time).total_seconds() / 3600
            if abs(rel_h) > window_h:
                continue

            msl_hpa = float(row["msl"]) / 100.0 if float(row["msl"]) > 10000 else float(row["msl"])

            records.append({
                "year":         int(year),
                "track_id":     int(tid),
                "is_survivor":  is_survivor,
                "is_long_lived": is_long,
                "rel_hours":    rel_h,
                "msl_hpa":      msl_hpa,
                "speed_kmh":    float(row.get("speed_kmh", np.nan)),
                "latitude":     float(row["latitude"]),
                "longitude":    float(row["longitude"]),
                "is_land":      bool(row.get("is_land", False)),
            })

    df = pd.DataFrame(records)
    n_survivors     = df[df["is_survivor"]].groupby(["year", "track_id"]).ngroups
    n_non_survivors = df[~df["is_survivor"]].groupby(["year", "track_id"]).ngroups
    print(f"Relative series built: {n_survivors} survivors | {n_non_survivors} non-survivors")
    return df


# Composite mean ± σ on a regular hour grid
def composite_stats(
    rel_df: pd.DataFrame,
    col: str,
    is_survivor: bool,
    window_h: int = 24,
    step_h: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each hour on [-window_h, +window_h] grid, collect all track values
    within ±step_h/2 of that hour, then compute mean and std.

    Returns (hour_grid, means, stds, counts).
    """
    sub   = rel_df[rel_df["is_survivor"] == is_survivor]
    hour_grid = np.arange(-window_h, window_h + step_h, step_h)
    means, stds, counts = [], [], []

    for h in hour_grid:
        vals = sub.loc[np.abs(sub["rel_hours"] - h) <= step_h / 2, col].dropna().values
        means.append(np.mean(vals) if len(vals) else np.nan)
        stds.append(np.std(vals)   if len(vals) >= 2 else 0.0)
        counts.append(len(vals))

    return (
        hour_grid,
        np.array(means),
        np.array(stds),
        np.array(counts, dtype=int),
    )


# Plotting
def _draw_panel(
    ax,
    rel_df: pd.DataFrame,
    col: str,
    ylabel: str,
    window_h: int,
    show_n: bool = True,
):
    """
    Draw one composite panel (spaghetti + mean \\pm std) for both groups.
    Elavation (Eastern ghat)survivors are always plotted on top with thicker lines.
    """
    # Spaghetti
    for (year, tid), grp in rel_df.groupby(["year", "track_id"]):
        is_surv = grp["is_survivor"].iloc[0]
        color   = SURVIVOR_COLOR if is_surv else NON_SURVIVOR_COLOR
        lw      = 2.4 if is_surv else SPAGHETTI_LW
        zo      = 5   if is_surv else 2
        ax.plot(
            grp["rel_hours"], grp[col],
            color=color, linewidth=lw,
            alpha=SPAGHETTI_ALPHA if not is_surv else 0.85,
            zorder=zo,
            label=f"{year} T{tid}" if is_surv else None,
        )
        # Annotate survivor lines at their final point
        if is_surv:
            last = grp.sort_values("rel_hours").iloc[-1]
            ax.annotate(
                f"{year} T{tid}",
                xy=(last["rel_hours"], last[col]),
                fontsize=FS_ANNOT,
                fontweight="bold",
                color=SURVIVOR_COLOR,
                xytext=(3, 2),
                textcoords="offset points",
            )

    # Composite mean \pm 1σ for each group
    for is_surv, color, lbl in [
        (False, NON_SURVIVOR_COLOR, "Non-survivors"),
        (True,  SURVIVOR_COLOR,     "Survivors"),
    ]:
        h_grid, means, stds, cnts = composite_stats(rel_df, col, is_surv, window_h)
        valid = ~np.isnan(means)
        if valid.sum() < 2:
            continue
        ax.plot(
            h_grid[valid], means[valid],
            color=color, linewidth=MEAN_LW, linestyle="--",
            alpha=MEAN_ALPHA, zorder=8,
            label=f"{lbl} mean",
        )
        ax.fill_between(
            h_grid[valid],
            means[valid] - stds[valid],
            means[valid] + stds[valid],
            color=color, alpha=0.15, zorder=3,
        )

        # Sample count annotation at T=0
        idx0 = np.argmin(np.abs(h_grid - 0))
        if show_n and cnts[idx0] > 0:
            ax.annotate(
                f"n={cnts[idx0]}",
                xy=(0, means[idx0]),
                fontsize=FS_ANNOT - 1,
                color=color,
                xytext=(4, 6 if is_surv else -10),
                textcoords="offset points",
                fontweight="bold",
            )

    # Reference lines
    ax.axvline(
        0,
        color="red",
        linewidth=2.0,
        linestyle="--",
        label="Landfall (T = 0)",
        zorder=9
    )
    ax.axvspan(
        -window_h,
        0,
        color="lightblue",
        alpha=0.08,
        zorder=1
    )
    ax.axvspan(
        0,
        window_h,
        color="wheat",
        alpha=0.08,
        zorder=1
    )

    ax.set_xlim(-window_h, window_h)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, fontweight="bold")
    ax.set_xlabel("Hours relative to landfall (T = 0)", fontsize=FS_LABEL, fontweight="bold")
    ax.grid(True, alpha=0.30, linestyle="--")


def plot_single_variable(
    rel_df:   pd.DataFrame,
    col:      str,
    ylabel:   str,
    output_dir: str,
    window_h: int = 24,
    year_range: str = "2001–2025",
):
    if col not in rel_df.columns or rel_df[col].isna().all():
        print(f"  SKIP: column '{col}' not available or all-NaN")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    _draw_panel(ax, rel_df, col, ylabel, window_h)

    legend_patches = [
        Patch(facecolor=SURVIVOR_COLOR,     edgecolor="none", alpha=0.75,
              label="Elevation survivors (n=4)"),
        Patch(facecolor=NON_SURVIVOR_COLOR, edgecolor="none", alpha=0.75,
              label="Non-survivors"),
        Line2D([0], [0], color=SURVIVOR_COLOR,     linestyle="--", linewidth=MEAN_LW,
               label="Survivors mean"),
        Line2D([0], [0], color=NON_SURVIVOR_COLOR, linestyle="--", linewidth=MEAN_LW,
               label="Non-survivors mean"),
        Patch(facecolor="lightblue", alpha=0.20, label="Pre-landfall (ocean phase)"),
        Patch(facecolor="wheat",     alpha=0.20, label="Post-landfall (land phase)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=FS_LEGEND, framealpha=0.92)

    ax.set_title(
        f"Composite {ylabel}  |  Landfall-relative (±{window_h} h)  |  {year_range}",
        # "Elevation survivors vs Non-survivors  -  Ocean-originated MDs with landfall",
        fontsize=FS_TITLE, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"composite_{col}.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  SAVED: {outpath}")


def plot_combined_panel(
    rel_df:   pd.DataFrame,
    output_dir: str,
    window_h: int = 24,
    year_range: str = "2001–2025",
):
    """
    2-panel combined figure: MSL (top) + speed (bottom).
    """
    variables = [
        ("msl_hpa",   "MSL Pressure (hPa)"),
        ("speed_kmh", "Translation Speed (km/h)"),
    ]
    # Only keep panels where data exists
    variables = [(c, l) for c, l in variables
                 if c in rel_df.columns and not rel_df[c].isna().all()]

    if not variables:
        print("  SKIP combined panel: no plottable variables")
        return

    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (col, ylabel) in zip(axes, variables):
        _draw_panel(ax, rel_df, col, ylabel, window_h, show_n=(col == variables[0][0]))

    # Single shared legend on top panel
    legend_patches = [
        Patch(facecolor=SURVIVOR_COLOR,     edgecolor="none", alpha=0.75,
              label="Elevation survivors"),
        Patch(facecolor=NON_SURVIVOR_COLOR, edgecolor="none", alpha=0.75,
              label="Non-survivors"),
        Line2D([0], [0], color=SURVIVOR_COLOR,     linestyle="--", linewidth=MEAN_LW,
               label="Survivors mean"),
        Line2D([0], [0], color=NON_SURVIVOR_COLOR, linestyle="--", linewidth=MEAN_LW,
               label="Non-survivors mean"),
        Patch(facecolor="lightblue", alpha=0.20, label="Ocean phase (T<0)"),
        Patch(facecolor="wheat",     alpha=0.20, label="Land phase (T>0)"),
    ]
    axes[0].legend(handles=legend_patches, loc="upper right",
                   fontsize=FS_LEGEND, framealpha=0.92)

    fig.suptitle(
        f"Landfall-Relative Composite  |  ±{window_h} h  |  {year_range}",
        # "Ghats survivors vs Non-survivors  -  Ocean-originated MDs with landfall",
        fontsize=FS_TITLE + 1, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "composite_all_variables.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  SAVED: {outpath}")


# 6.  Summary CSV
def export_composite_csv(rel_df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "composite_tracks_all_years.csv")
    rel_df.to_csv(path, index=False)
    print(f"  SAVED: {path}")


def print_group_summary(rel_df: pd.DataFrame):
    """Print stats for each group at T=0 and T=+12h."""
    print("\n--- Composite summary ---")
    for is_surv, label in [(True, "Survivors"), (False, "Non-survivors")]:
        grp = rel_df[rel_df["is_survivor"] == is_surv]
        n_tracks = grp.groupby(["year", "track_id"]).ngroups
        print(f"\n{label}  (n={n_tracks} tracks)")
        for col, unit in [("msl_hpa", "hPa"), ("speed_kmh", "km/h")]:
            if col not in grp.columns:
                continue
            at_lf = grp[np.abs(grp["rel_hours"]) <= 1][col].dropna()
            at_12 = grp[np.abs(grp["rel_hours"] - 12) <= 1][col].dropna()
            print(f"  {col:<12}: at landfall = {at_lf.mean():.1f} ± {at_lf.std():.1f} {unit}"
                  f"  |  T+12h = {at_12.mean():.1f} ± {at_12.std():.1f} {unit}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-year landfall composite: Elevation survivors vs non-survivors."
    )
    p.add_argument("--root",   default=".", help="Project root (contains output_* dirs)")
    p.add_argument("--out",    default="composite_output", help="Output directory")
    p.add_argument("--window", type=int, default=24, help="Half-window hours around landfall")
    return p.parse_args()


def main():
    args = parse_args()

    print("Multi-Year Landfall Composite")
    print(f"  Window : ±{args.window} h around landfall (T=0)")
    print(f"  Elevation survivors registered: {len(GHATS_SURVIVORS)}")
    for yr, tid in sorted(GHATS_SURVIVORS):
        print(f"    ({yr}, T{tid})")

    # 1. Load
    print("\nLoading all years...")
    all_df = load_all_years(args.root)

    # 2. Filter
    print("\nFiltering ocean+landfall tracks...")
    eligible = filter_eligible(all_df)

    # 3. Build relative series
    print("\nBuilding landfall-relative time series...")
    rel_df = build_relative_series(eligible, window_h=args.window)
    if rel_df.empty:
        print("ERROR: No data after building relative series. Check CSV columns.")
        return

    # Verify at least one survivor is present
    surv_found = set(zip(rel_df[rel_df["is_survivor"]]["year"],
                         rel_df[rel_df["is_survivor"]]["track_id"]))
    if not surv_found:
        print("WARN: None of the registered elevation survivors found in CSVs.")
        print(f"      Registered: {GHATS_SURVIVORS}")
        print(f"      Found (year, track_id) pairs: "
              f"{set(zip(rel_df['year'], rel_df['track_id']))}")

    year_range = (f"{rel_df['year'].min()}–{rel_df['year'].max()}")

    # 4. Summary
    print_group_summary(rel_df)

    # 5. Export CSV
    print("\nExporting composite CSV...")
    export_composite_csv(rel_df, args.out)

    # 6. Individual variable plots
    print("\nPlotting individual variables...")
    for col, ylabel in [
        ("msl_hpa",   "MSL Pressure (hPa)"),
        ("speed_kmh", "Translation Speed (km/h)"),
    ]:
        try:
            plot_single_variable(rel_df, col, ylabel, args.out, args.window, year_range)
        except Exception as e:
            print(f"  ERROR plotting {col}: {e}")
            traceback.print_exc()

    # 7. Combined 2-panel figure
    print("\nPlotting combined panel...")
    try:
        plot_combined_panel(rel_df, args.out, args.window, year_range)
    except Exception as e:
        print(f"  ERROR combined panel: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
