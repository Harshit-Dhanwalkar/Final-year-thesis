#   1. Combine all year/track speed plots into single composite
#   2. Compare groups (Eastern Ghat survivors vs non-survivors)
#   3. Create publication-quality multi-panel figures

import os
import glob
import argparse
import traceback
from typing import Dict, List, Tuple, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# CONFIGURATION


# Define your groups here
EASTERN_GHAT_SURVIVORS = {
    (2005, 6),   # T6, September 2005 - Short-lived
    (2017, 4),   # T4, August 2017 - Short-lived
    (2021, 5),   # T5, September 2021 - Long-lived
    (2024, 4),   # T4, August 2024 - Short-lived
}

COLOR_SURVIVORS = "#2ca02c"      # green
COLOR_NON_SURVIVORS = "#1f77b4"  # blue
COLOR_MEAN = "#d62728"           # red

ALPHA_SPAGHETTI = 0.35
ALPHA_MEAN = 0.95
LW_SPAGHETTI = 1.0
LW_MEAN = 2.5
FS_TITLE = 14
FS_LABEL = 12
FS_LEGEND = 9



# DATA LOADING & FILTERING


def load_all_years(root_dir: str = ".") -> pd.DataFrame:
    """Load all MD tracks from output_*/MD_tracks_*.csv"""
    pattern = os.path.join(root_dir, "output_*", "MD_tracks_*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No MD_tracks_*.csv found in {root_dir}")
    
    frames = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            year = int(os.path.basename(os.path.dirname(fpath)).replace("output_", ""))
            df["year"] = year
            frames.append(df)
            print(f"  ✓ {fpath}: {df['track_id'].nunique()} tracks")
        except Exception as e:
            print(f"  ✗ {fpath}: {e}")
    
    if not frames:
        raise ValueError("No files loaded")
    
    combined = pd.concat(frames, ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"])
    return combined


def filter_ocean_landfall(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to ocean-origin tracks with landfall"""
    records = []
    
    for (year, tid), grp in df.groupby(["year", "track_id"]):
        grp = grp.sort_values("time").reset_index(drop=True)
        
        # Check ocean origin
        if "is_land" in grp.columns:
            first_land = bool(grp["is_land"].iloc[0])
            if first_land:
                continue
            
            # Check for landfall
            land_vals = grp["is_land"].astype(bool).values
            has_landfall = False
            for i in range(1, len(land_vals)):
                if not land_vals[i-1] and land_vals[i]:
                    has_landfall = True
                    break
            
            if not has_landfall:
                continue
        
        records.append(grp)
    
    if not records:
        print("WARNING: No ocean+landfall tracks found")
        return df
    
    return pd.concat(records, ignore_index=True)



# LANDFALL-RELATIVE TIME SERIES BUILDING


def find_landfall_time(group: pd.DataFrame) -> pd.Timestamp | None:
    """Find first ocean->land transition"""
    if "is_land" not in group.columns:
        return None
    
    land = group["is_land"].astype(bool).values
    for i in range(1, len(land)):
        if not land[i-1] and land[i]:
            return group["time"].iloc[i]
    return None


def build_relative_series(
    df: pd.DataFrame,
    group_fn: Callable,
    window_h: int = 24
) -> pd.DataFrame:
    """
    Build landfall-relative time series.
    group_fn: function that takes (year, track_id) and returns group label
    """
    records = []
    
    for (year, tid), grp in df.groupby(["year", "track_id"]):
        grp = grp.sort_values("time").reset_index(drop=True)
        lf_time = find_landfall_time(grp)
        
        if lf_time is None:
            continue
        
        group_label = group_fn(int(year), int(tid))
        duration_h = (grp["time"].max() - grp["time"].min()).total_seconds() / 3600
        
        for _, row in grp.iterrows():
            rel_h = (row["time"] - lf_time).total_seconds() / 3600
            if abs(rel_h) > window_h:
                continue
            
            # Handle MSL pressure
            msl_val = float(row["msl"])
            msl_hpa = msl_val / 100 if msl_val > 10000 else msl_val
            
            records.append({
                "year": int(year),
                "track_id": int(tid),
                "group": group_label,
                "duration_h": duration_h,
                "rel_hours": rel_h,
                "msl_hpa": msl_hpa,
                "speed_kmh": float(row.get("speed_kmh", np.nan)),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "is_land": bool(row.get("is_land", False)),
            })
    
    rel_df = pd.DataFrame(records)
    print(f"Built relative series: {rel_df.groupby(['year', 'track_id']).ngroups} tracks")
    return rel_df



# COMPOSITE STATISTICS


def compute_stats(
    rel_df: pd.DataFrame,
    col: str,
    group_label: str,
    window_h: int = 24,
    step_h: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean ± std at each hour for a group"""
    sub = rel_df[rel_df["group"] == group_label]
    
    if sub.empty:
        return (np.array([]), np.array([]), np.array([]))
    
    hour_grid = np.arange(-window_h, window_h + step_h, step_h)
    means, stds = [], []
    
    for h in hour_grid:
        vals = sub.loc[np.abs(sub["rel_hours"] - h) <= step_h/2, col].dropna().values
        if len(vals) > 0:
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
        else:
            means.append(np.nan)
            stds.append(0)
    
    return (hour_grid, np.array(means), np.array(stds))



# PLOTTING FUNCTIONS


def plot_composite_panel(
    ax,
    rel_df: pd.DataFrame,
    col: str,
    ylabel: str,
    group_colors: Dict,
    window_h: int = 24,
):
    """Plot composite with spaghetti + mean ± std"""
    
    groups = sorted(rel_df["group"].unique())
    
    # Spaghetti lines (individual tracks)
    for (year, tid), grp in rel_df.groupby(["year", "track_id"]):
        group_label = grp["group"].iloc[0]
        color = group_colors.get(group_label, "#999999")
        is_survivor = "Survivor" in group_label
        
        ax.plot(
            grp["rel_hours"], grp[col],
            color=color, linewidth=LW_SPAGHETTI,
            alpha=ALPHA_SPAGHETTI, zorder=2
        )
    
    # Mean ± std for each group
    for group_label in groups:
        hour_grid, means, stds = compute_stats(rel_df, col, group_label, window_h)
        if len(hour_grid) == 0:
            continue
        
        color = group_colors.get(group_label, "#999999")
        n_tracks = rel_df[rel_df["group"] == group_label]["track_id"].nunique()
        
        # Error band
        ax.fill_between(
            hour_grid, means - stds, means + stds,
            color=color, alpha=0.25, zorder=3
        )
        
        # Mean line
        ax.plot(
            hour_grid, means,
            color=color, linestyle="--", linewidth=LW_MEAN,
            alpha=ALPHA_MEAN, zorder=4,
            label=f"{group_label} (n={n_tracks})"
        )
    
    # Styling
    ax.axvline(0, color="black", linestyle="-", linewidth=1.5, 
               label="Landfall", zorder=9, alpha=0.7)
    ax.axvspan(-window_h, 0, color="lightblue", alpha=0.05, zorder=1)
    ax.axvspan(0, window_h, color="wheat", alpha=0.05, zorder=1)
    
    ax.set_xlim(-window_h, window_h)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, fontweight="bold")
    ax.set_xlabel("Hours relative to landfall", fontsize=FS_LABEL-1, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend(loc="best", fontsize=FS_LEGEND, framealpha=0.9)


def plot_single_variable(
    rel_df: pd.DataFrame,
    col: str,
    ylabel: str,
    group_colors: Dict,
    output_dir: str,
    window_h: int = 24,
    year_range: str = "",
):
    """Plot single variable composite"""
    if col not in rel_df.columns or rel_df[col].isna().all():
        print(f"  SKIP: {col} not available")
        return

    fig, ax = plt.subplots(figsize=(13, 6))
    plot_composite_panel(ax, rel_df, col, ylabel, group_colors, window_h)

    title = f"{ylabel} | Landfall-relative composite | {year_range}"
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"composite_{col}.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_multi_panel(
    rel_df: pd.DataFrame,
    variables: List[Tuple[str, str]],
    group_colors: Dict,
    output_dir: str,
    window_h: int = 24,
    year_range: str = "",
):
    """Plot multiple variables in subpanels"""
    variables = [
        (col, label) for col, label in variables
        if col in rel_df.columns and not rel_df[col].isna().all()
    ]
    
    if not variables:
        print("  No plottable variables")
        return
    
    n_panels = len(variables)
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 5*n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    
    for ax, (col, ylabel) in zip(axes, variables):
        plot_composite_panel(ax, rel_df, col, ylabel, group_colors, window_h)
    
    fig.suptitle(
        f"Landfall-relative composite | {year_range}",
        fontsize=FS_TITLE+1, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "composite_all_variables.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")



# GROUPING FUNCTIONS


def group_eastern_ghat(year: int, track_id: int) -> str:
    """Classify as Eastern Ghat Survivor or Non-survivor"""
    if (year, track_id) in EASTERN_GHAT_SURVIVORS:
        return "Eastern Ghat Survivors (4)"
    return "Non-survivors (127)"


def group_by_year(year: int, track_id: int) -> str:
    """Group by year"""
    return f"{year}"



# SUMMARY STATISTICS
def print_summary(rel_df: pd.DataFrame):
    """Print group statistics"""
    for group in sorted(rel_df["group"].unique()):
        grp = rel_df[rel_df["group"] == group]
        n_tracks = grp["track_id"].nunique()
        n_points = len(grp)
        print(f"\n{group}:")
        print(f"  Tracks: {n_tracks} | Points: {n_points}")
        for col in ["msl_hpa", "speed_kmh"]:
            if col in grp.columns:
                vals = grp[col].dropna()
                if len(vals) > 0:
                    print(f"  {col:12s}: {vals.mean():.2f} ± {vals.std():.2f}")



# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Composite analysis: consolidate multiple tracks into single plots"
    )
    parser.add_argument("--root", default=".", help="Project root (contains output_*)")
    parser.add_argument("--out", default="composite_output", help="Output directory")
    parser.add_argument("--window", type=int, default=24, help="Hours around landfall")
    parser.add_argument("--group", choices=["eastern_ghat", "year"], default="eastern_ghat", help="Grouping method")
    args = parser.parse_args()

    print(f"Window: ±{args.window}h")
    print(f"Grouping: {args.group}")

    # Load data
    print("Loading data...")
    all_df = load_all_years(args.root)
    print(f"  Total: {all_df['track_id'].nunique()} tracks")

    # Filter
    print("\nFiltering ocean+landfall tracks...")
    eligible = filter_ocean_landfall(all_df)
    print(f"  Eligible: {eligible['track_id'].nunique()} tracks")

    # Build relative series
    print("\nBuilding landfall-relative time series...")
    group_fn = group_eastern_ghat if args.group == "eastern_ghat" else group_by_year
    rel_df = build_relative_series(eligible, group_fn, args.window)

    if rel_df.empty:
        print("ERROR: Empty relative series")
        return

    year_range = f"{rel_df['year'].min()}–{rel_df['year'].max()}"

    # Summary
    print_summary(rel_df)

    # Color scheme
    group_colors = {
        "Eastern Ghat Survivors (4)": COLOR_SURVIVORS,
        "Non-survivors (127)": COLOR_NON_SURVIVORS,
    }

    # Plot
    print("\nCreating plots...")
    variables = [
        ("msl_hpa", "MSL Pressure (hPa)"),
        ("speed_kmh", "Translation Speed (km/h)"),
    ]
    
    for col, ylabel in variables:
        try:
            plot_single_variable(rel_df, col, ylabel, group_colors, args.out, args.window, year_range)
        except Exception as e:
            print(f"  ERROR: {col} - {e}")
            traceback.print_exc()
    
    try:
        plot_multi_panel(rel_df, variables, group_colors, args.out, args.window, year_range)
    except Exception as e:
        print(f"  ERROR multi-panel: {e}")
        traceback.print_exc()

    # Export
    print("\nExporting data...")
    os.makedirs(args.out, exist_ok=True)
    rel_df.to_csv(os.path.join(args.out, "composite_data.csv"), index=False)
    print(f" Saved: {args.out}/composite_data.csv")


if __name__ == "__main__":
    main()
