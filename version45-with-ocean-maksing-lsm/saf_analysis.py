# saf_analysis.py
# Focused analysis of ONE land depression (default: 21 Jul - 6 Aug 2024).
#
# Stages (each one is independent; a failure in one does not stop the others)
#   1. Box-mean series (400 km box centred on the MD): SWVL1, TP, LHF, SHF, mixing ratio q
#   2. Time series: LHF/SHF (+70 W/m2 BOE line), SWVL1, TP
#   3. Rolling 12 h (trailing, no look-ahead) r(SM,q), sigma(q) and SAF = r(SM,q) * sigma(q)
#   4. Spatial SAF maps every 12 h on a 400 km x 400 km box centred on the MD
#   5. Daily-mean (24 h) azimuthal Hovmoller diagrams
#
# SAF (Chen et al. 2026):  SAF = r(SM, q) * sigma(q)
#   SM = volumetric soil water layer 1 (swvl1), q = mixing ratio from d2m and sp (g/kg)
#
# Usage (default stages ts hov sens; add saf for the SAF part):
#   python saf_analysis.py --year 2024 --start 2024-07-21 --end 2024-08-06 [--track-id 4]

import argparse
import os
import sys
import traceback
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
warnings.filterwarnings("ignore", category=RuntimeWarning)

FULL_EXTENT = [70.0, 95.0, 7.0, 30.0]  # lon_min, lon_max, lat_min, lat_max
PAD_DEG = 6.0  # ERA5 crop padding around the track (covers 400 km Hovmoller radius)
BOE_LHF = 70.0  # W/m2 (Andersen & Shepherd 2013)
RADII_KM = list(range(0, 401, 25))
HOV_VARS = [
    "wind10", "vr10", "vt10", "vor_rel", "vor_abs", "msl", "tp", "t2m", "d2m",
    "swvl1", "avg_slhtf", "avg_ishf", "total_hf", "w",
]
NEEDED_VARS = [
    "swvl1", "tp", "avg_slhtf", "avg_ishf", "d2m", "sp", "t2m", "u10", "v10", "msl", "stl1",
]
FS_TITLE, FS_LABEL, FS_ANNOT = 14, 12, 9


def log(level, msg, val=""):
    from utils import std_print

    std_print(level, "saf", msg, val)


# Loading
def load_track(csv_path, track_id, t0, t1):
    """Slice the detection CSV to the window and return an hourly track for one MD."""
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df[(df["time"] >= t0) & (df["time"] <= t1)]
    if df.empty:
        raise RuntimeError(f"No track points between {t0} and {t1} in {csv_path}")
    if track_id is None:
        track_id = int(df["track_id"].value_counts().idxmax())
        log("INFO", "No --track-id given, using track with most points in window", track_id)
    tr = df[df["track_id"] == track_id].sort_values("time").drop_duplicates("time")
    if tr.empty:
        raise RuntimeError(f"Track {track_id} has no points in the window")
    tr = tr.set_index("time")
    idx = pd.date_range(tr.index.min().floor("h"), tr.index.max().ceil("h"), freq="h")
    union = idx.union(tr.index)
    out = tr[["latitude", "longitude"]].reindex(union).interpolate("time").reindex(idx)
    if "is_land" in tr.columns:
        out["is_land"] = tr["is_land"].astype(bool).reindex(idx, method="nearest")
    else:
        from utils import is_point_land

        out["is_land"] = [is_point_land(a, b) for a, b in zip(out.latitude, out.longitude)]
    out["track_id"] = int(track_id)
    out.index.name = "time"
    out = out.reset_index()
    log("INFO", f"Track {track_id} hourly points", f"{len(out)}  {out.time.min()} -> {out.time.max()}")
    return out


def load_era5_window(data_dir, t0, t1, track, hist_h=24):
    """Open only the months needed, crop to the track + padding and load into RAM."""
    ta, tb = t0 - pd.Timedelta(hours=hist_h), t1 + pd.Timedelta(hours=1)
    months = [f"{p.strftime('%B')}-{p.year}" for p in pd.period_range(ta, tb, freq="M")]
    files = []
    for m in months:
        d = os.path.join(data_dir, m)
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".nc")]
    if not files:
        raise RuntimeError(f"No ERA5 files in {data_dir} for {months}")
    ds = xr.open_mfdataset(
        files, chunks={}, combine="by_coords", parallel=True,
        coords="minimal", data_vars="minimal", compat="override", engine="h5netcdf",
    )
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    ds = ds.sortby("time")

    lat_lo = max(track.latitude.min() - PAD_DEG, FULL_EXTENT[2])
    lat_hi = min(track.latitude.max() + PAD_DEG, FULL_EXTENT[3])
    lon_lo = max(track.longitude.min() - PAD_DEG, FULL_EXTENT[0])
    lon_hi = min(track.longitude.max() + PAD_DEG, FULL_EXTENT[1])
    desc = float(ds.latitude[0]) > float(ds.latitude[-1])
    ds = ds.sel(
        time=slice(ta, tb),
        latitude=slice(lat_hi, lat_lo) if desc else slice(lat_lo, lat_hi),
        longitude=slice(lon_lo, lon_hi),
    )
    keep = [v for v in NEEDED_VARS if v in ds.data_vars]
    missing = {"swvl1", "tp", "avg_slhtf", "avg_ishf", "d2m", "sp"} - set(keep)
    if missing:
        raise RuntimeError(f"ERA5 is missing required variables: {sorted(missing)}")
    ds = ds[keep].load()

    from azimuthal_hovmoller import compute_mixing_ratio

    ds["q"] = xr.apply_ufunc(compute_mixing_ratio, ds["d2m"], ds["sp"])  # g/kg
    ds["q"].attrs["units"] = "g/kg"
    log("INFO", "ERA5 window loaded", f"{ds.sizes['time']} h, {ds.sizes['latitude']}x{ds.sizes['longitude']} grid")
    return ds


def load_lsm(path):
    if not os.path.exists(path):
        log("WARN", "LSM not found, Hovmoller ocean masking disabled", path)
        return None
    lsm = xr.open_dataset(path)
    var = "lsm" if "lsm" in lsm.data_vars else list(lsm.data_vars)[0]
    arr = lsm[var]
    for d in [d for d in arr.dims if d not in ("latitude", "longitude")]:
        arr = arr.isel({d: 0}, drop=True)
    return arr


# Box helpers
def box_mask(ds, lat_c, lon_c, half_km):
    lat = ds.latitude.values[:, None]
    lon = ds.longitude.values[None, :]
    dy = (lat - lat_c) * 111.0
    dx = (lon - lon_c) * 111.0 * np.cos(np.radians(lat_c))
    return (np.abs(dy) <= half_km) & (np.abs(dx) <= half_km)


def box_mean_series(ds, track, box_km):
    """Box-mean of the fields around the MD centre at every track hour."""
    half = box_km / 2.0
    times_ds = pd.to_datetime(ds.time.values)
    fields = [("swvl1", 1.0), ("tp", 1000.0), ("avg_slhtf", 1.0), ("avg_ishf", 1.0),
              ("q", 1.0), ("t2m", 1.0), ("d2m", 1.0)]
    rows = []
    for _, r in track.iterrows():
        it = times_ds.get_indexer([r["time"]], method="nearest")[0]
        m = box_mask(ds, r["latitude"], r["longitude"], half)
        rec = {"time": r["time"], "latitude": r["latitude"], "longitude": r["longitude"],
               "is_land": bool(r["is_land"]), "n_cells": int(m.sum())}
        for v, scale in fields:
            vals = ds[v].isel(time=it).values[m]
            rec[v] = float(np.nanmean(vals)) * scale if np.isfinite(vals).any() else np.nan
        rows.append(rec)
    ts = pd.DataFrame(rows).rename(columns={"tp": "tp_mmhr", "avg_slhtf": "lhf", "avg_ishf": "shf"})
    # ERA5 avg_slhtf / avg_ishf are positive DOWNWARD. If evaporation shows up as negative
    # over land, flip so that upward flux is positive (needed for the 70 W/m2 test).
    land_lhf = ts.loc[ts.is_land, "lhf"]
    if len(land_lhf) and land_lhf.median() < 0:
        log("WARN", "Land LHF median is negative (ERA5 downward-positive); flipping sign of LHF and SHF")
        ts["lhf"] *= -1.0
        ts["shf"] *= -1.0
    ts["total_hf"] = ts["lhf"] + ts["shf"]
    return ts.set_index("time")


# SAF maths
def rolling_saf(ts, win=12):
    """Trailing-window statistics: at hour t only hours t-win+1 ... t are used."""
    out = ts.copy()
    kw = dict(window=win, min_periods=win)
    out["r_sm_q"] = out["swvl1"].rolling(**kw).corr(out["q"])
    out["std_q"] = out["q"].rolling(**kw).std()
    out["std_sm"] = out["swvl1"].rolling(**kw).std()
    out["r_sm_lhf"] = out["swvl1"].rolling(**kw).corr(out["lhf"])
    out["saf"] = out["r_sm_q"] * out["std_q"]
    return out


def pearson_map(a, b):
    """Pearson r along axis 0 for (T, y, x) arrays; NaN where any sample is missing/constant."""
    ok = np.isfinite(a) & np.isfinite(b)
    full = ok.all(axis=0)
    a = np.where(ok, a, np.nan)
    b = np.where(ok, b, np.nan)
    da = a - np.nanmean(a, axis=0)
    db = b - np.nanmean(b, axis=0)
    cov = np.nansum(da * db, axis=0)
    va = np.nansum(da**2, axis=0)
    vb = np.nansum(db**2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.sqrt(va * vb)
    r[~full | (va <= 0) | (vb <= 0)] = np.nan
    return r


def saf_map(ds, it, win, mask):
    """SAF = r(SM,q) * sigma(q) per grid cell, using the `win` hours ending at index `it`."""
    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    ysl, xsl = slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)
    tsl = slice(it - win + 1, it + 1)
    sm = ds["swvl1"].isel(time=tsl, latitude=ysl, longitude=xsl).values
    q = ds["q"].isel(time=tsl, latitude=ysl, longitude=xsl).values
    r = pearson_map(sm, q)
    sq = np.nanstd(q, axis=0, ddof=1)
    sq[~np.isfinite(r)] = np.nan
    lat = ds.latitude.values[ysl]
    lon = ds.longitude.values[xsl]
    return lat, lon, r, sq, r * sq


def compute_saf_maps(ds, track, box_km, win, step_h):
    half = box_km / 2.0
    times_ds = pd.to_datetime(ds.time.values)
    maps = []
    for _, row in track.iterrows():
        t = row["time"]
        if t.hour % step_h != 0:
            continue
        it = times_ds.get_indexer([t], method="nearest")[0]
        if it - win + 1 < 0:
            continue
        span = (times_ds[it] - times_ds[it - win + 1]).total_seconds() / 3600
        if abs(span - (win - 1)) > 1e-6:
            log("WARN", f"{t}: ERA5 history window is not contiguous, skipping map")
            continue
        mask = box_mask(ds, row["latitude"], row["longitude"], half)
        lat, lon, r, sq, saf = saf_map(ds, it, win, mask)
        maps.append(dict(time=t, lat_c=row["latitude"], lon_c=row["longitude"],
                         lat=lat, lon=lon, r=r, sq=sq, saf=saf))
    return maps


# Plots
def _shade(ax, ts):
    from utils import add_land_ocean_shading

    add_land_ocean_shading(ax, pd.Series(ts.index), ts["is_land"].values.astype(bool),
                           land_color="wheat", ocean_color="lightblue")


def _fmt_time(ax):
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m %H:%M"))


def plot_flux_sm_tp(ts, track_id, out_dir, boe18, boe72, boe96=None):
    from utils import legend_style

    t = ts.index
    land = ts["is_land"].values.astype(bool)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    ax = axes[0]
    _shade(ax, ts)
    ax.plot(t, ts["lhf"], "-o", color="green", lw=2.2, ms=3, label="Latent heat flux (LHF)", zorder=4)
    ax.plot(t, ts["shf"], "-s", color="orange", lw=2.2, ms=3, label="Sensible heat flux (SHF)", zorder=4)
    ax.axhline(BOE_LHF, color="red", ls="--", lw=1.8, label=f"LHF = {BOE_LHF:.0f} W/m² (BOE threshold)", zorder=5)
    ax.fill_between(t, ts["lhf"], BOE_LHF, where=(ts["lhf"] >= BOE_LHF) & land,
                    color="seagreen", alpha=0.25, interpolate=True, label="LHF ≥ 70 on land")
    ax.set_ylabel("Surface heat flux (W/m²)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: surface energy flux, soil moisture and precipitation\n"
        f"Longest land run LHF ≥ {BOE_LHF:.0f}: {boe18['max_consecutive_h']} h "
        f"(18 h criterion: {'YES' if boe18['is_boe_candidate'] else 'no'}, "
        f"72 h: {'YES' if boe72['is_boe_candidate'] else 'no'}"
        + (f", 96 h: {'YES' if boe96['is_boe_candidate'] else 'no'}" if boe96 else "") + ")",
        fontsize=FS_TITLE, fontweight="bold")
    h, l = ax.get_legend_handles_labels()
    h += [Patch(facecolor="lightblue", alpha=0.5), Patch(facecolor="wheat", alpha=0.5)]
    l += ["Ocean phase", "Land phase"]
    ax.legend(h, l, loc="upper right", **legend_style())

    ax = axes[1]
    _shade(ax, ts)
    ax.plot(t, ts["swvl1"], "-s", color="tab:brown", lw=2.2, ms=3, zorder=4)
    ax.set_ylabel("SWVL1 (m³/m³)", fontsize=FS_LABEL, fontweight="bold")

    ax = axes[2]
    _shade(ax, ts)
    ax.plot(t, ts["tp_mmhr"], "-o", color="tab:blue", lw=2.2, ms=3, zorder=4)
    ax.set_ylabel("Total precipitation (mm/hr)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_xlabel("Time", fontsize=FS_LABEL, fontweight="bold")
    for a in axes:
        a.grid(True, alpha=0.35, ls="--", zorder=0)
        _fmt_time(a)
    plt.tight_layout()
    p = os.path.join(out_dir, f"track_{track_id}_HF_SM_TP_timeseries.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log("SAVED", "Flux / SM / TP time series", p)


def plot_rolling_saf(ts, track_id, win, out_dir):
    from utils import legend_style

    t = ts.index
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    spec = [
        ("r_sm_q", "tab:purple", f"r(SM, q), previous {win} h", True),
        ("std_q", "tab:teal" if "tab:teal" in matplotlib.colors.CSS4_COLORS else "teal",
         f"σ(q), previous {win} h (g/kg)", False),
        ("saf", "crimson", "SAF = r(SM, q) × σ(q) (g/kg)", True),
    ]
    for ax, (col, color, label, zero) in zip(axes, spec):
        _shade(ax, ts)
        ax.plot(t, ts[col], "-o", color=color, lw=2.2, ms=3, zorder=4)
        if zero:
            ax.axhline(0, color="k", lw=0.8, alpha=0.6)
        ax.set_ylabel(label, fontsize=FS_LABEL, fontweight="bold")
        ax.grid(True, alpha=0.35, ls="--", zorder=0)
        _fmt_time(ax)
    axes[0].set_title(f"Track {track_id}: trailing {win} h SM–q coupling (no look-ahead)",
                      fontsize=FS_TITLE, fontweight="bold")
    axes[0].set_ylim(-1.05, 1.05)
    axes[-1].set_xlabel("Time", fontsize=FS_LABEL, fontweight="bold")
    axes[0].legend([Patch(facecolor="lightblue", alpha=0.5), Patch(facecolor="wheat", alpha=0.5)],
                   ["Ocean phase", "Land phase"], loc="upper right", **legend_style())
    plt.tight_layout()
    p = os.path.join(out_dir, f"track_{track_id}_rolling_{win}h_SAF.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log("SAVED", "Rolling SAF plot", p)


def plot_saf_maps(maps, track_id, box_km, win, out_dir):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    pc = ccrs.PlateCarree()
    allv = np.concatenate([m["saf"][np.isfinite(m["saf"])].ravel() for m in maps if np.isfinite(m["saf"]).any()])
    if allv.size == 0:
        log("WARN", "All SAF maps are empty (no land cells with valid SM?)")
        return
    vmax = float(np.nanpercentile(np.abs(allv), 98)) or 1e-6
    log("INFO", "SAF colour scale (common to all maps)", f"±{vmax:.3f} g/kg")

    def draw(ax, m, field, cmap, vmin, vmax_, title):
        ax.set_extent([m["lon"].min() - 0.2, m["lon"].max() + 0.2, m["lat"].min() - 0.2, m["lat"].max() + 0.2], crs=pc)
        im = ax.pcolormesh(m["lon"], m["lat"], field, cmap=cmap, vmin=vmin, vmax=vmax_,
                           shading="nearest", transform=pc, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
        ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=6)
        ax.plot(m["lon_c"], m["lat_c"], marker="*", color="k", ms=12, mfc="yellow", transform=pc, zorder=8)
        ax.set_title(title, fontsize=FS_LABEL, fontweight="bold")
        return im

    for m in maps:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), subplot_kw={"projection": pc})
        tstr = m["time"].strftime("%d/%m/%Y %H:%M")
        ims = [
            draw(axes[0], m, m["r"], "RdBu_r", -1, 1, f"r(SM, q)  [prev {win} h]"),
            draw(axes[1], m, m["sq"], "viridis", 0, np.nanpercentile(m["sq"], 98) if np.isfinite(m["sq"]).any() else 1,
                 "σ(q) (g/kg)"),
            draw(axes[2], m, m["saf"], "RdBu_r", -vmax, vmax, "SAF = r × σ(q) (g/kg)"),
        ]
        for ax, im in zip(axes, ims):
            plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.06, shrink=0.85)
        fig.suptitle(f"Track {track_id}: SAF map, {box_km:.0f} km box centred on MD — {tstr} UTC",
                     fontsize=FS_TITLE, fontweight="bold")
        p = os.path.join(out_dir, f"SAF_map_{m['time'].strftime('%Y%m%d_%H%M')}.png")
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # NetCDF copy of the maps for later work
        xr.Dataset(
            {"r": (("latitude", "longitude"), m["r"]), "std_q": (("latitude", "longitude"), m["sq"]),
             "saf": (("latitude", "longitude"), m["saf"])},
            coords={"latitude": m["lat"], "longitude": m["lon"]},
            attrs={"time": str(m["time"]), "centre_lat": m["lat_c"], "centre_lon": m["lon_c"]},
        ).to_netcdf(p.replace(".png", ".nc"), engine="h5netcdf")
    log("SAVED", f"{len(maps)} SAF maps (PNG + NetCDF)", out_dir)

    # Contact sheet (SAF only, common colour scale)
    n = len(maps)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow), subplot_kw={"projection": pc}, squeeze=False)
    im = None
    for ax, m in zip(axes.ravel(), maps):
        im = draw(ax, m, m["saf"], "RdBu_r", -vmax, vmax, m["time"].strftime("%d/%m %H:%M"))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", pad=0.03, shrink=0.5,
                     label="SAF (g/kg)")
    fig.suptitle(f"Track {track_id}: SAF every 12 h ({box_km:.0f} km box)", fontsize=FS_TITLE, fontweight="bold")
    p = os.path.join(out_dir, f"track_{track_id}_SAF_contact_sheet.png")
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log("SAVED", "SAF contact sheet", p)


def saf_map_summary(maps):
    rows = []
    for m in maps:
        v = m["saf"][np.isfinite(m["saf"])]
        rows.append({"time": m["time"], "lat_c": m["lat_c"], "lon_c": m["lon_c"], "n_valid_cells": int(v.size),
                     "mean_saf": float(v.mean()) if v.size else np.nan,
                     "min_saf": float(v.min()) if v.size else np.nan,
                     "max_saf": float(v.max()) if v.size else np.nan,
                     "frac_positive": float((v > 0).mean()) if v.size else np.nan})
    return pd.DataFrame(rows)


# Daily Hovmoller
def plot_daily_hovmoller(arr, days, counts, radii, var, track_id, lf_day, per_r_lf_day, out_dir):
    from azimuthal_hovmoller import VARIABLE_NAMES, VARIABLE_UNITS

    label = VARIABLE_NAMES.get(var, var)
    unit = VARIABLE_UNITS.get(var, "")
    arr = np.asarray(arr, float)
    if var in ("vor_rel", "vor_abs"):
        arr = arr * 1e5
    cmap = {"tp": "plasma", "wind10": "plasma", "avg_slhtf": "plasma", "avg_ishf": "plasma",
            "total_hf": "plasma", "t2m": "coolwarm", "stl1": "coolwarm", "msl": "viridis_r",
            "sp": "viridis_r", "swvl1": "YlGnBu", "w": "Blues"}.get(var, "viridis")
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        log("WARN", f"All-NaN daily Hovmoller for {var}, skipping")
        return
    if var in ("vor_rel", "vor_abs", "vr10", "dswvl1_dt"):
        cmap = "RdBu_r"
        vmax = float(np.nanpercentile(np.abs(arr), 97)) or 1e-6
        vmin = -vmax
    else:
        vmin, vmax = float(np.nanpercentile(arr, 2)), float(np.nanpercentile(arr, 98))
        if vmin == vmax:
            vmin, vmax = vmin - 1e-6, vmax + 1e-6
    X, Y = np.meshgrid(radii, np.arange(len(days)))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(X, Y, arr, levels=np.linspace(vmin, vmax, 36), cmap=cmap, extend="both")
    cs = ax.contour(X, Y, arr, levels=np.linspace(vmin, vmax, 12), colors="black", linewidths=0.5, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=FS_ANNOT, fmt="%1.2f")
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(f"{label} ({unit})", fontsize=FS_LABEL)
    ax.set_yticks(np.arange(len(days)))
    ax.set_yticklabels([f"{d.strftime('%d/%m')}" + ("" if c >= 24 else f" ({c}h)") for d, c in zip(days, counts)],
                       fontsize=FS_LABEL - 2)
    ax.set_xticks(radii)
    ax.set_xticklabels([str(int(r)) for r in radii], rotation=45, ha="right", fontsize=FS_LABEL - 2)
    ax.set_xlabel("Radius from centre (km)", fontsize=FS_LABEL)
    ax.set_ylabel("Day (24 h mean; partial days show hours used)", fontsize=FS_LABEL)
    ax.set_title(f"Track {track_id}: {label} — daily mean", fontsize=FS_TITLE, fontweight="bold")
    if per_r_lf_day is not None and np.any(per_r_lf_day > 0):
        ok = per_r_lf_day >= 0
        ax.plot(np.asarray(radii)[ok], per_r_lf_day[ok], "k--", lw=2, label="First land contact", zorder=5)
        ax.legend(loc="upper left", bbox_to_anchor=(0.78, 1.05), fontsize=9)
    plt.tight_layout()
    p = os.path.join(out_dir, f"hov_{var}_daily_mean.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log("SAVED", f"Daily Hovmoller {var}", p)


def run_daily_hovmoller(ds, track, lsm_2d, track_id, out_dir):
    from azimuthal_hovmoller import (average_azimuthally, compute_azimuthal_hovmoller_sectors,
                                     compute_swvl1_tendency)

    variables = [v for v in HOV_VARS if v in ds.data_vars or v in ("wind10", "vr10", "vt10", "vor_rel",
                                                                    "vor_abs", "total_hf", "w")]
    hov, times, lf_idx, per_r = compute_azimuthal_hovmoller_sectors(
        ds, track.reset_index(drop=True), radii_km=RADII_KM, variables=variables, n_theta=16, lsm_2d=lsm_2d)
    avg = average_azimuthally(hov, variables, radii_km=RADII_KM)
    if "swvl1" in avg:
        avg["dswvl1_dt"] = compute_swvl1_tendency(avg["swvl1"], times)

    day_of_step = pd.to_datetime(times).floor("D")
    days = pd.DatetimeIndex(day_of_step.unique())
    counts = [int((day_of_step == d).sum()) for d in days]
    day_index = {d: i for i, d in enumerate(days)}
    # hourly landfall index -> index of the day containing it
    per_r_day = np.array([day_index[day_of_step[i]] if i >= 0 else -1 for i in per_r])

    daily = {v: np.vstack([np.nanmean(a[day_of_step == d], axis=0) for d in days]) for v, a in avg.items()}
    for v, a in daily.items():
        plot_daily_hovmoller(a, days, counts, RADII_KM, v, track_id, None, per_r_day, out_dir)
    np.savez(os.path.join(out_dir, f"track_{track_id}_daily_hovmoller.npz"),
             days=days.values.astype("datetime64[D]"), radii_km=np.array(RADII_KM), hours_per_day=np.array(counts),
             **{k: v for k, v in daily.items()})
    log("SAVED", "Daily Hovmoller arrays", os.path.join(out_dir, f"track_{track_id}_daily_hovmoller.npz"))


# 72 h vs 96 h criteria
LV = 2.5e6  # J/kg
SENS_DURATIONS = (18, 24, 48, 72, 96, 120)


def longest_run(mask):
    best = run = 0
    for v in np.asarray(mask, bool):
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def sustained_max_threshold(series, land, n):
    """Highest x such that series >= x for n consecutive LAND hours (sliding-window min, then max).
    Non-increasing in n by construction: a longer window can only lower the sustainable level."""
    s = pd.Series(np.where(land, series, np.nan), dtype=float)
    r = s.rolling(n, min_periods=n).min()
    return float(r.max()) if r.notna().any() else np.nan


def criteria_sensitivity(ts, durations=SENS_DURATIONS, lhf_thr=BOE_LHF, sm_thr=(0.30, 0.325, 0.35, 0.375, 0.40)):
    land = ts["is_land"].values.astype(bool)
    lhf = ts["lhf"].values
    sm = ts["swvl1"].values
    rows = []
    for n in durations:
        row = {"duration_h": n,
               "lhf_max_sustained_Wm2": sustained_max_threshold(lhf, land, n),
               "sm_max_sustained_m3m3": sustained_max_threshold(sm, land, n),
               "lhf_ge_70_pass": longest_run((lhf >= lhf_thr) & land) >= n}
        for th in sm_thr:
            row[f"sm_ge_{th:.3f}_pass"] = longest_run((sm >= th) & land) >= n
            row[f"lhf70_and_sm_ge_{th:.3f}_pass"] = longest_run((lhf >= lhf_thr) & (sm >= th) & land) >= n
        rows.append(row)
    return pd.DataFrame(rows)


def drawdown_budget(lhf_wm2=BOE_LHF, durations=(72, 96), depth_m=0.07):
    """No-resupply drawdown of ERA5 layer 1 (0-7 cm) if ALL of the latent flux were drawn from it.
    Upper bound: ignores rain, capillary rise and root uptake from layers 2-4."""
    e_mm_h = lhf_wm2 / LV * 3600.0
    return pd.DataFrame([{"duration_h": n, "evap_mm": e_mm_h * n,
                          "drawdown_m3m3": e_mm_h * n / (depth_m * 1000.0)} for n in durations])


def plot_sensitivity(sens, track_id, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, lab, ref in [(axes[0], "sm_max_sustained_m3m3", "Highest SWVL1 sustained (m³/m³)", None),
                              (axes[1], "lhf_max_sustained_Wm2", "Highest LHF sustained (W/m²)", BOE_LHF)]:
        ax.plot(sens["duration_h"], sens[col], "-o", lw=2.2, color="tab:brown" if ref is None else "green")
        for n in (72, 96):
            ax.axvline(n, color="k", ls=":", lw=1)
        if ref:
            ax.axhline(ref, color="red", ls="--", lw=1.5, label=f"{ref:.0f} W/m²")
            ax.legend()
        ax.set_xlabel("Required consecutive land hours", fontsize=FS_LABEL)
        ax.set_ylabel(lab, fontsize=FS_LABEL)
        ax.grid(True, alpha=0.35, ls="--")
    fig.suptitle(f"Track {track_id}: level sustainable for N consecutive land hours", fontsize=FS_TITLE, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(out_dir, f"track_{track_id}_criteria_sensitivity.png")
    plt.savefig(p, dpi=250, bbox_inches="tight")
    plt.close(fig)
    log("SAVED", "Criteria sensitivity plot", p)


def mask_ocean_swvl1(ds, lsm_2d):
    """ERA5 swvl1 can carry fill values over ocean; keep land cells only so box means are land means."""
    if lsm_2d is None:
        log("WARN", "No LSM: box-mean SWVL1 may include ocean cells")
        return ds
    near = lsm_2d.sel(latitude=ds.latitude.values, longitude=ds.longitude.values, method="nearest")
    land2d = xr.DataArray(near.values.squeeze() >= 0.5, dims=("latitude", "longitude"),
                          coords={"latitude": ds.latitude, "longitude": ds.longitude})
    ds["swvl1"] = ds["swvl1"].where(land2d)
    return ds


# Main
STAGES = ("ts", "hov", "sens", "saf")


def parse_args():
    ap = argparse.ArgumentParser(description="Single land-depression analysis (daily Hovmoller, flux/SM/TP series, 72 vs 96 h criteria, SAF)")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--start", default="2024-07-21")
    ap.add_argument("--end", default="2024-08-06")
    ap.add_argument("--track-id", type=int, default=None, help="default: track with most points in window")
    ap.add_argument("--csv", default=None, help="detection CSV (default output_<year>_BUF_500/MD_tracks_<year>.csv)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--box-km", type=float, default=400.0)
    ap.add_argument("--roll-h", type=int, default=12)
    ap.add_argument("--map-step-h", type=int, default=12)
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=["ts", "hov", "sens"],
                    help="ts = flux/SM/TP series; hov = daily-mean Hovmoller; sens = 72 vs 96 h check; saf = SAF series + maps")
    return ap.parse_args()


def main():
    a = parse_args()
    stages = set(a.stages)
    t0 = pd.Timestamp(a.start)
    t1 = pd.Timestamp(a.end) + pd.Timedelta(hours=23)  # inclusive of the whole end day
    base = a.output_dir or f"output_{a.year}_BUF_500"
    csv = a.csv or os.path.join(base, f"MD_tracks_{a.year}.csv")
    data_dir = os.path.join("..", "Data", "single-levels", "ncfiles", str(a.year))
    lsm_path = os.path.join("..", "Data", "single-levels", "ncfiles", "lsm-soiltype-z.nc")

    track = load_track(csv, a.track_id, t0, t1)
    tid = int(track["track_id"].iloc[0])
    out = os.path.join(base, f"track_{tid}", f"landdep_{t0:%Y%m%d}_{t1:%Y%m%d}")
    os.makedirs(out, exist_ok=True)

    ds = load_era5_window(data_dir, track["time"].min(), track["time"].max(), track)
    lsm_2d = load_lsm(lsm_path)
    ds = mask_ocean_swvl1(ds, lsm_2d)

    ts = box_mean_series(ds, track, a.box_km)
    if "saf" in stages:
        ts = rolling_saf(ts, a.roll_h)
    ts.to_csv(os.path.join(out, f"track_{tid}_box_timeseries.csv"))

    # 1. time series: LHF + SHF (70 W/m2 line), SWVL1, TP
    if "ts" in stages:
        try:
            from heatflux_analysis import classify_boe_candidate

            fl = ts.reset_index().rename(columns={"lhf": "latent_heat_flux"})[["time", "latent_heat_flux", "is_land"]]
            boe = {n: classify_boe_candidate(fl, lhf_threshold=BOE_LHF, min_hours=n) for n in (18, 72, 96)}
            pd.DataFrame([{"criterion_h": n, **b} for n, b in boe.items()]).to_csv(
                os.path.join(out, f"track_{tid}_boe_classification.csv"), index=False)
            plot_flux_sm_tp(ts, tid, out, boe[18], boe[72], boe[96])
        except Exception as e:
            log("ERROR", "Flux/SM/TP plot failed", str(e))
            traceback.print_exc()

    # 2. daily-mean Hovmoller
    if "hov" in stages:
        try:
            hov_dir = os.path.join(out, "hovmoller_daily")
            os.makedirs(hov_dir, exist_ok=True)
            run_daily_hovmoller(ds, track, lsm_2d, tid, hov_dir)
        except Exception as e:
            log("ERROR", "Daily Hovmoller failed", str(e))
            traceback.print_exc()

    # 3. 72 h vs 96 h criteria
    if "sens" in stages:
        try:
            sens = criteria_sensitivity(ts)
            sens.to_csv(os.path.join(out, f"track_{tid}_criteria_sensitivity.csv"), index=False)
            bud = drawdown_budget()
            bud.to_csv(os.path.join(out, "drawdown_budget_70Wm2.csv"), index=False)
            plot_sensitivity(sens, tid, out)
            log("INFO", "No-resupply drawdown, 70 W/m2 from 0-7 cm", bud.round(3).to_dict("records"))
        except Exception as e:
            log("ERROR", "Criteria sensitivity failed", str(e))
            traceback.print_exc()

    # 4. SAF (opt-in)
    if "saf" in stages:
        try:
            plot_rolling_saf(ts, tid, a.roll_h, out)
            maps = compute_saf_maps(ds, track, a.box_km, a.roll_h, a.map_step_h)
            log("INFO", "SAF maps computed", len(maps))
            if maps:
                map_dir = os.path.join(out, "saf_maps")
                os.makedirs(map_dir, exist_ok=True)
                saf_map_summary(maps).to_csv(os.path.join(out, f"track_{tid}_saf_map_summary.csv"), index=False)
                plot_saf_maps(maps, tid, a.box_km, a.roll_h, map_dir)
        except Exception as e:
            log("ERROR", "SAF failed", str(e))
            traceback.print_exc()
    log("INFO", "Done", out)


if __name__ == "__main__":
    sys.exit(main())
