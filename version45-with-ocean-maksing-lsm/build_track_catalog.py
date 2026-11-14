# build_track_catalog.py
"""
Track catalog + metric extraction for buffer sensitivity comparison.

Outputs per buffer directory (output_2024_BUF_{100,300,500}/):
  - track_catalog.csv                 summary: dates, duration, locations
  - track_N/track_N_identity_card.png map + TP/SWVL1 + annotated stats
  - track_N/track_N_metrics.csv       all table metrics for that track

Also writes:
  - buffer_comparison_table.csv       combined table across all buffers
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BUFFERS = [100, 300, 500]
YEAR = 2024

# -----------------------------------------------------------------------------
# 1. Summary catalog per buffer
# -----------------------------------------------------------------------------
def build_catalog(root):
    csv = os.path.join(root, f"MD_tracks_{YEAR}.csv")
    if not os.path.exists(csv):
        print(f"  [skip] {csv} missing")
        return None
    df = pd.read_csv(csv, parse_dates=["time"])
    if "is_land" in df.columns:
        df["is_land"] = df["is_land"].astype(bool)
    rows = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("time").reset_index(drop=True)
        start = g["time"].iloc[0]
        end = g["time"].iloc[-1]
        rows.append({
            "track_id": int(tid),
            "start": start.strftime("%Y-%m-%d %H:%M"),
            "end": end.strftime("%Y-%m-%d %H:%M"),
            "dur_h": round((end - start).total_seconds() / 3600, 1),
            "start_lat": round(g["latitude"].iloc[0], 2),
            "start_lon": round(g["longitude"].iloc[0], 2),
            "end_lat": round(g["latitude"].iloc[-1], 2),
            "end_lon": round(g["longitude"].iloc[-1], 2),
            "n_pts": len(g),
            "starts_on_land": bool(g["is_land"].iloc[0]) if "is_land" in g else None,
        })
    cat = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    cat.to_csv(os.path.join(root, "track_catalog.csv"), index=False)
    return cat

# -----------------------------------------------------------------------------
# 2. Metric extraction (the numbers you need for the table)
# -----------------------------------------------------------------------------
def extract_metrics(root, track_id):
    """Collect every metric for one track at one buffer."""
    row = {"track_id": track_id, "buffer_km": int(root.split("_BUF_")[-1])}
    tdir = os.path.join(root, f"track_{track_id}")

    # --- correlation metrics (mean_tp, mean_swvl1, pearson r, p) ------------
    import glob
    f_corr = glob.glob(os.path.join(tdir, "correlation_analysis", "*.csv"))
    if f_corr:
        d = pd.read_csv(f_corr[0])
        if len(d):
            r = d.iloc[0]
            for k in ["pearson_r", "pearson_p", "pearson_p_corrected",
                      "mean_tp", "mean_swvl1", "n_samples"]:
                if k in r:
                    row[k] = r[k]

    # --- heat flux metrics (mean LHF, SHF, total HF) ------------------------
    f_hf = os.path.join(tdir, "heat_flux_analysis",
                        f"track_{track_id}_landfall_flux.csv")
    if os.path.exists(f_hf):
        try:
            d = pd.read_csv(f_hf)
            if len(d):
                r = d.iloc[0]
                # These are string-dicts in the CSV; parse carefully
                for side in ["before", "after"]:
                    if side in r and isinstance(r[side], str):
                        try:
                            import ast
                            s = ast.literal_eval(r[side])
                            for k in ["latent", "sensible", "total"]:
                                if k in s:
                                    row[f"{side}_{k}_hf"] = s[k]
                        except Exception:
                            pass
        except Exception as e:
            print(f"  heat flux parse failed: {e}")

    # --- BOE classification (peak / mean LHF land) --------------------------
    f_boe = os.path.join(tdir, "heat_flux_analysis",
                         f"track_{track_id}_boe_classification.csv")
    if os.path.exists(f_boe):
        d = pd.read_csv(f_boe)
        if len(d):
            r = d.iloc[0]
            for k in ["is_boe_candidate", "peak_lhf_land",
                      "mean_lhf_land", "max_consecutive_h", "total_above_h"]:
                if k in r:
                    row[k] = r[k]

    # --- soil moisture memory ----------------------------------------------
    f_sm = os.path.join(tdir, "soilmoiture_analysis",
                        f"track_{track_id}_soil_memory.csv")
    if os.path.exists(f_sm):
        d = pd.read_csv(f_sm)
        if len(d):
            r = d.iloc[0]
            for k in ["memory_index_24h", "mean_swvl1", "std_swvl1", "n_samples"]:
                if k in r:
                    row[f"sm_{k}"] = r[k]

    return row

# -----------------------------------------------------------------------------
# 3. Identity card PNG — map + quick TP/SWVL1 timeseries + stats box
# -----------------------------------------------------------------------------
def identity_card(root, track_id, catalog_row):
    tdir = os.path.join(root, f"track_{track_id}")
    csv = os.path.join(tdir, f"track_{track_id}_detections.csv")
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv, parse_dates=["time"]).sort_values("time").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    # ---- Left: map of track ----
    ax = axes[0]
    land = df["is_land"].values.astype(bool) if "is_land" in df else np.zeros(len(df), bool)
    # draw land/ocean segments
    for i in range(len(df) - 1):
        c = "saddlebrown" if land[i] else "dodgerblue"
        ax.plot(df["longitude"].iloc[i:i+2], df["latitude"].iloc[i:i+2],
                color=c, linewidth=2.0, solid_capstyle="round", zorder=3)
    ax.scatter(df["longitude"].iloc[0], df["latitude"].iloc[0],
               s=140, c="green", marker="o", edgecolors="black", label="Start", zorder=5)
    ax.scatter(df["longitude"].iloc[-1], df["latitude"].iloc[-1],
               s=140, c="orange", marker="o", edgecolors="black", label="End", zorder=5)
    # annotate every 12h
    t0 = df["time"].iloc[0]
    hrs = (df["time"] - t0).dt.total_seconds() / 3600
    for h in np.arange(0, hrs.max() + 12, 12):
        idx = int(np.argmin(np.abs(hrs.values - h)))
        ax.annotate(f"{int(h)}h", (df["longitude"].iloc[idx], df["latitude"].iloc[idx]),
                    fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Track {track_id}  |  {catalog_row['start']} → {catalog_row['end']}")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")

    # ---- Right: stats box ----
    ax = axes[1]
    ax.axis("off")
    stats_lines = [
        f"Buffer: {catalog_row['buffer_km']} km",
        f"Duration: {catalog_row['dur_h']} h",
        f"Points: {catalog_row['n_pts']}",
        f"Start: ({catalog_row['start_lat']}°N, {catalog_row['start_lon']}°E)",
        f"End:   ({catalog_row['end_lat']}°N, {catalog_row['end_lon']}°E)",
        f"Starts on land: {catalog_row['starts_on_land']}",
        "",
    ]
    # append whatever metrics we found
    m = extract_metrics(root, track_id)
    for k, v in m.items():
        if k in ("track_id", "buffer_km"):
            continue
        try:
            v_str = f"{float(v):.4g}" if isinstance(v, (int, float, np.floating)) else str(v)
        except Exception:
            v_str = str(v)
        stats_lines.append(f"{k}: {v_str}")

    ax.text(0.02, 0.98, "\n".join(stats_lines),
            transform=ax.transAxes, fontsize=10, va="top", family="monospace")

    plt.tight_layout()
    out = os.path.join(tdir, f"track_{track_id}_identity_card.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------
def main():
    all_rows = []
    for B in BUFFERS:
        root = f"output_{YEAR}_BUF_{B}"
        if not os.path.exists(root):
            print(f"skip {root} (missing)")
            continue
        print(f"\n=== BUFFER {B} km ===")
        cat = build_catalog(root)
        if cat is None or cat.empty:
            continue
        # attach buffer col to catalog before identity cards
        cat["buffer_km"] = B
        print(cat.to_string(index=False))

        for _, row in cat.iterrows():
            tid = int(row["track_id"])
            print(f"  generating card for track {tid} … ", end="")
            p = identity_card(root, tid, row)
            print("ok" if p else "skip")
            # metrics row for combined table
            m = extract_metrics(root, tid)
            all_rows.append(m)

    if all_rows:
        combined = pd.DataFrame(all_rows)
        combined = combined.sort_values(["track_id", "buffer_km"])
        combined.to_csv("buffer_comparison_table.csv", index=False)
        print("\nWrote buffer_comparison_table.csv")
        print(combined.to_string(index=False))

if __name__ == "__main__":
    main()
