"""
Vortex Structure Analyses for Monsoon Depressions
Uses ERA5 pressure-level data (u, v, vo, t, q, w, z) at 200/300/400/500/600/700/850 hPa.

1. Vortex tilt vector                - displacement of vo max at 850 hPa vs 200 hPa over time
2. Cold-core intensity               - 850 hPa temperature anomaly relative to 300 km ambient ring
3. Warm-core index                   - 400 hPa temperature anomaly (MD warm core peaks near 400 hPa)
4. Vortex structure metric           - vo(850) <-> vo(200) difference tracks system coherence
5. Integrated Vapour Transport (IVT) - moisture flux column-integral 850->300 hPa
6. Orographic omega cross-section    - 700 hPa vertical velocity ±150 km across track
7. All-tracks tilt comparison plot   - tilt magnitude for all ocean-originated tracks

References:
Hunt & Turner (2016) MWR - MD 3D composite structure, 850 hPa vorticity centre definition
Rashid & Srinivasan (2005) MAP - vorticity budget, cold-core/warm-core characterisation
Mishra (2021) MWR - MD development and large-scale interaction
"""

# FIX:
# 1. Time series panel is empty

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import traceback

from utils import (
    std_print,
    haversine_distance,
    common_bbox,
    legend_style,
    pearsonr_corrected,
)

warnings.filterwarnings("ignore", message="facecolor will have no effect", category=UserWarning, module="cartopy")

# Constants
G = 9.80665  # m s⁻²
L_v = 2.5e6      # J kg⁻¹ — latent heat of vaporisation
RHO_W = 1000.0   # kg m⁻³ — water density

SEARCH_RADIUS_DEG = 3.0  # vortex centre search radius around single-level track (°)
AMBIENT_RING_KM   = 300.0  # radius for temperature anomaly background ring
LEVELS = [850, 700,600, 500, 400, 300, 200]
OMEGA_TRANSECT_KM = 150.0  # perpendicular transect half-width for omega cross-section

# Module-level dataset cache
DS_CACHE = {}

# Data loading
def load_level_at_time(pl_dir, t, level_hpa, year):
    """Return a single-timestep Dataset slice from the cached month file"""
    month_str = month_str_from_time(t, year)
    ds = load_pl_level(pl_dir, month_str, level_hpa)
    if ds is None:
        return None
    return ds.sel(time=t, method="nearest")


# TODO: add all intermidiate levels
# Vortex tilt
def compute_vortex_tilt(track_df, pl_dir,year, output_dir):
    """
    Compute the vortex tilt vector (850 hPa -> 200 hPa) at each track time step.

    The tilt vector is the horizontal displacement between the 850 hPa and
    200 hPa relative vorticity maxima.

    Reference:
    Hunt & Turner (2016) show that Indian  MDs have a characteristic WNW tilt with height in their mature phase;
    an anomalous increase in tilt magnitude during the land phase - particularly  collocated with terrain
    - indicates low-level frictional decoupling.
    """
    track_id = int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])

    records = []
    for i, t in enumerate(times):
        tlon = float(df["longitude"].iloc[i])
        tlat = float(df["latitude"].iloc[i])

        # Load 850 and 200 hPa vorticity at this time step
        ds850 = load_level_at_time(pl_dir, t, 850, year)
        ds200 = load_level_at_time(pl_dir, t, 200, year)
        if ds850 is None or ds200 is None:
            records.append({
                "time": t,
                "lat_850": np.nan,
                "lon_850": np.nan,
                "lat_200": np.nan,
                "lon_200": np.nan,
                "tilt_magnitude_km": np.nan,
                "tilt_bearing_deg": np.nan
            })
            continue

        vo850 = ds850["vo"].squeeze()
        vo200 = ds200["vo"].squeeze()
        lat850, lon850 = find_vortex_centre(vo850, tlat, tlon)
        lat200, lon200 = find_vortex_centre(vo200, tlat, tlon)

        if lat850 is None or lat200 is None:
            records.append({
                "time": t,
                "lat_850": lat850,
                "lon_850": lon850,
                "lat_200": lat200,
                "lon_200": lon200,
                "tilt_magnitude_km": np.nan,
                "tilt_bearing_deg": np.nan
            })
            continue

        # Tilt vector: from 850 hPa centre toward 200 hPa centre
        tilt_km = haversine_distance(lon850, lat850, lon200, lat200)

        # Bearing of tilt (degrees, 0=N, clockwise)
        dlat = lat200 - lat850
        dlon = lon200 - lon850
        bearing = np.degrees(np.arctan2(dlon * np.cos(np.radians(lat850)), dlat)) % 360

        records.append({
            "time": t,
            "lat_850": lat850,
            "lon_850": lon850,
            "lat_200": lat200,
            "lon_200": lon200,
            "tilt_lat_deg": dlat,
            "tilt_lon_deg": dlon,
            "tilt_magnitude_km": tilt_km,
            "tilt_bearing_deg": bearing,
        })
    tilt_df = pd.DataFrame(records)

    # Export
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"track_{track_id}_vortex_tilt.csv")
    tilt_df.to_csv(csv_path, index=False)
    std_print("EXPORT", "va", f"Track {track_id} vortex tilt", csv_path)

    plot_vortex_tilt(tilt_df, track_df, track_id, output_dir)
    return tilt_df


# Thermal structure - cold-core + warm-core
def compute_thermal_structure(track_df, pl_dir,year, output_dir):
    """
    At each track time step extract:
      - T_850_anomaly : 850 hPa temperature minus ambient ring mean (cold core signal)
      - T_200_anomaly : 200 hPa temperature minus ambient ring mean (warm core signal)

    Reference:
    Classical MD thermal structure (Rashid & Srinivasan 2005):
      Cold core: T_850_anomaly < 0  (−2 to −4 K at peak)
      Warm core: T_200_anomaly > 0  (+1 to +3 K at peak)

    When the cold core warms (anomaly -> 0) the system is weakening
    When it warms faster over terrain than over flat land -> terrain-induced dissipation
    """
    track_id = int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])

    theraml_sturcture_records = []
    for i, t in enumerate(times):
        tlon = float(df["longitude"].iloc[i])
        tlat = float(df["latitude"].iloc[i])

        rec = {"time": t}
        for level in LEVELS:
            rec[f"T{level}_centre"] = np.nan
            rec[f"T{level}_anom"]   = np.nan

        for level in LEVELS:
            ds = load_level_at_time(pl_dir, t, level , year)
            if ds is None or "t" not in ds:
                continue
            t_slice = ds["t"].squeeze()

            try:
                # Vortex centre temperature (interpolate to track position)
                t_centre = float(t_slice.sel(latitude=tlat, longitude=tlon, method="nearest").values)
            except Exception:
                t_centre = np.nan
            t_amb = ambient_temperature(t_slice, tlat, tlon)
            rec[f"T{level}_centre"] = t_centre
            rec[f"T{level}_anom"]   = t_centre - t_amb

        theraml_sturcture_records.append(rec)
    therm_df = pd.DataFrame(theraml_sturcture_records)

    # Export
    os.makedirs(output_dir, exist_ok=True)
    thermal_structure_CSV = os.path.join(output_dir, f"Track_{track_id}_thermal_structure.csv")
    therm_df.to_csv(thermal_structure_CSV, index=False)
    std_print("EXPORT", "va", "Thermal structure", thermal_structure_CSV)

    plot_thermal_structure(therm_df, track_df, track_id, output_dir)
    return therm_df


# Vorticity column coherence (850 -> 200 hPa full column)
def compute_vorticity_column(track_df, pl_dir,year, output_dir):
    """
    At each time step, extract the relative vorticity at the track centre
    for 850, 700, 600, 500, 400, 300, 200hPa

    Δvo = vo_850 − vo_200 measures vortex column coherence.
    When Δvo drops sharply (upper-level vorticity lags or reverses sign),
    the system is losing vertical coherence - a dissipation precursor.

    Reference:
    Rashid & Srinivasan (2005) showed that positive vorticity at 850+700+500 hPa
    with negative vorticity at 200 hPa characterises a mature MD.
    """
    track_id = int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])

    vorticity_column_records = []
    for i, t in enumerate(times):
        lon_at_t = float(df["longitude"].iloc[i])
        lat_at_t = float(df["latitude"].iloc[i])
        rec = {"time": t}

        for level in LEVELS:
            ds = load_level_at_time(pl_dir, t, level, year)
            if ds is None or "vo" not in ds:
                rec[f"vo_{level}"] = np.nan
                continue
            vo_slice = ds["vo"].squeeze()

            try:
                vo_val = float(vo_slice.sel(latitude=lat_at_t, longitude=lon_at_t, method="nearest").values)
            except Exception:
                vo_val = np.nan
            rec[f"vo_{level}"] = vo_val
        vorticity_column_records.append(rec)

    col_df = pd.DataFrame(vorticity_column_records)
    col_df["delta_vo_850_200"] = col_df["vo_850"] - col_df["vo_200"]

    # Export
    os.makedirs(output_dir, exist_ok=True)
    vorticity_column_csv= os.path.join(output_dir, f"track_{track_id}_vorticity_column.csv")
    col_df.to_csv(vorticity_column_csv, index=False)
    std_print("EXPORT", "va", f"Track {track_id} vorticity column", vorticity_column_csv)

    plot_vorticity_column(col_df, track_df, track_id, output_dir)
    return col_df


# Integrated Vapour Transport (IVT)
def compute_ivt(track_df, pl_dir, year, output_dir):
    """
    IVT = (1/g) * sqrt( (Σ q·u·Δp)² + (Σ q·v·Δp)² )

    Integrate from 850 to 300 hPa using available levels.
    Units: kg m⁻¹ s⁻¹

    A drop in IVT during the land phase - especially over terrain - indicates that orographic 
    blocking is cutting off the moisture flux that sustains the MD's convection
    Reference threshold: IVT < 250 kg m⁻¹ s⁻¹ is considered below the threshold for sustained
    deep convection in tropical systems (high-end estimates for intense monsoon systems are 200–800)
    """
    track_id = int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])

    # Pressure differences for trapezoidal integration (Pa)
    levels_hpa = LEVELS
    dp = []
    for j in range(len(levels_hpa)):
        if j == 0:
            dp.append((levels_hpa[j] - levels_hpa[j+1]) * 100 / 2)
        elif j == len(levels_hpa) - 1:
            dp.append((levels_hpa[j-1] - levels_hpa[j]) * 100 / 2)
        else:
            dp.append((levels_hpa[j-1] - levels_hpa[j+1]) * 100 / 2)

    ivt_records = []
    for i, t in enumerate(times):
        tlon = float(df["longitude"].iloc[i])
        tlat = float(df["latitude"].iloc[i])

        sum_qu = 0.0
        sum_qv = 0.0
        valid = True
        for level, dpj in zip(levels_hpa, dp):
            ds = load_level_at_time(pl_dir, t, level, year)
            if ds is None or "q" not in ds or "u" not in ds or "v" not in ds:
                valid = False
                break
            try:
                q = float(ds["q"].squeeze().sel(latitude=tlat, longitude=tlon, method="nearest").values)
                u = float(ds["u"].squeeze().sel(latitude=tlat, longitude=tlon, method="nearest").values)
                v = float(ds["v"].squeeze().sel(latitude=tlat, longitude=tlon, method="nearest").values)
                q = max(q, 0)   # clip spectral artefact
                sum_qu += q * u * dpj
                sum_qv += q * v * dpj
            except Exception:
                valid = False
                break

        ivt = float(np.sqrt(sum_qu**2 + sum_qv**2) / G) if valid else np.nan
        ivt_records.append({
            "time": t,
            "IVT": ivt,
            "IVT_u": float(abs(sum_qu) / G) if valid else np.nan,
            "IVT_v": float(abs(sum_qv) / G) if valid else np.nan
        })
    ivt_df = pd.DataFrame(ivt_records)

    # Export
    os.makedirs(output_dir, exist_ok=True)
    ivt_CSV = os.path.join(output_dir, f"track_{track_id}_ivt.csv")
    ivt_df.to_csv(ivt_CSV, index=False)
    std_print("EXPORT", "va", f"Track {track_id}: IVT", ivt_CSV)

    plot_ivt(ivt_df, track_df, track_id, output_dir)
    return ivt_df

#  Precipitation Recycling Ratio
# TODO: this is the simplified single-column Budyko formulation, not the full
# two-dimensional recycling ratio which requires wind field integration across
# the domain.
def compute_recycling_ratio(track_df, track_id, output_dir, buffer_km=100.0):
    """
    Precipitation recycling ratio along the MD track.
 
    recycling_ratio (ρ) = local_evaporation / total_precipitation
 
    Uses ERA5 single-level fields already loaded in track_df:
      - LHF (avg_slhtf, W m⁻²) -> E (mm h⁻¹) via latent heat of vaporisation
      - TP (tp, m h⁻¹)
 
    ρ ≈ 0   -> rainfall is almost entirely advected moisture (oceanic source)
    ρ -> 1  -> rainfall is mostly recycled from local land evaporation -> Brown Ocean Effect signature
 
    Reference: Eltahir & Bras (1996) define ρ as the fraction of
    precipitation that originates from within the region of interest.
    For tropical land systems, ρ > 0.25 is considered significant local
    moisture recycling. 
    """
    track_id = int(track_id)
    df = track_df.sort_values("time").reset_index(drop=True)
 
    # Require both LHF and TP columns
    lhf_col = None
    for candidate in ["avg_slhtf", "slhf", "lhf"]:
        if candidate in df.columns:
            lhf_col = candidate
            break
    tp_col = None
    for candidate in ["tp", "total_precipitation"]:
        if candidate in df.columns:
            tp_col = candidate
            break
 
    if lhf_col is None or tp_col is None:
        # std_print("WARN", "va","Recycling ratio requires LHF and TP columns", f"available: {list(df.columns)}")
        return pd.DataFrame()
 
    times = pd.to_datetime(df["time"])
    is_land_arr = df["is_land"].values.astype(bool) \
        if "is_land" in df.columns else np.ones(len(df), bool)
 
    # ERA5 LHF is negative downward (convention: positive = upward = evaporation)
    lhf_vals = df[lhf_col].values.astype(float) # W m⁻²
    tp_vals  = df[tp_col].values.astype(float)  # m (accumulated per step)
 
    # Convert LHF (W m⁻²) -> evaporation rate (mm h⁻¹)
    # E = LHF / (L_v * rho_w) in m s⁻¹, 
    E_m_s = (np.abs(lhf_vals) / (L_v * RHO_W))
    E_mm_h = E_m_s * 3600.0 * 1000.0
 
    # TP is hourly accumulated (m) -> mm h⁻¹
    P_mm_h = tp_vals * 1000.0
    P_mm_h = np.clip(P_mm_h, 0, None)   # negative TP is deaccumulation artefact
 
    # Recycling ratio: ρ = E / P, clipped [0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(P_mm_h >= 0.01, np.clip(E_mm_h / P_mm_h, 0, 1), np.nan)
 
    records = pd.DataFrame({
        "time": times,
        "is_land": is_land_arr,
        "LHF_Wm2": lhf_vals,
        "E_mm_h": E_mm_h,
        "P_mm_h": P_mm_h,
        "recycling_ratio": rho,
    })
 
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"track_{track_id}_recycling_ratio.csv")
    records.to_csv(csv_path, index=False)
    std_print("EXPORT", "va", f"Track {track_id} recycling ratio", csv_path)
 
    plot_recycling_ratio(records, track_id, output_dir)
    return records


# All-tracks tilt comparison
def plot_all_tracks_tilt_comparison(tilt_results, track_dict, output_dir):
    """
    Overlay tilt magnitude time series for all tracks on a single plot,
    aligned relative to landfall time (t=0 = landfall).

      Terminating MDs show increasing tilt as they approach terrain.
      The long-lived MD shows stable or decreasing tilt.
    """
    if not tilt_results:
        return

    # Figure
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(tilt_results)))

    for (track_id, tilt_df), color in zip(sorted(tilt_results.items()), colors):
        track_df = track_dict.get(track_id)
        if track_df is None:
            continue
        is_land = track_df.sort_values("time")["is_land"].values.astype(bool) \
            if "is_land" in track_df.columns else np.ones(len(tilt_df), bool)

        # Find landfall index
        lf_idx = None
        for i in range(1, len(is_land)):
            if not is_land[i-1] and is_land[i]:
                lf_idx = i
                break

        times = pd.to_datetime(tilt_df["time"])
        if lf_idx is not None and lf_idx < len(times):
            lf_time = times.iloc[lf_idx]
            hours = [(t - lf_time).total_seconds() / 3600 for t in times]
        else:
            hours = [(t - times.iloc[0]).total_seconds() / 3600 for t in times]

        tilt_km = tilt_df["tilt_magnitude_km"].values
        ax.plot(
            hours,
            tilt_km,
            "-o",
            color=color,
            linewidth=2.0,
            markersize=5,
            alpha=0.85,
            label=f"Track {track_id}"
        )
        # Mark land phase
        land_hours = np.array(hours)[is_land[:len(hours)]]
        land_tilt  = tilt_km[is_land[:len(tilt_km)]]
        ax.scatter(
            land_hours,
            land_tilt,
            color=color,
            s=60,
            marker="D",
            zorder=5,
            alpha=0.6
        )

    ax.axvline(0, color="red", linestyle="--", linewidth=1.8, label="Landfall (t = 0)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Hours relative to landfall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Vortex tilt 850 -> 200 hPa (km)", fontsize=12, fontweight="bold")
    # "Diamonds = land-phase points.  Rising tilt after landfall = terrain-induced decoupling",
    ax.set_title("All Tracks: Vortex Tilt Evolution", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", **legend_style())
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "all_tracks_vortex_tilt_comparison.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "All-tracks tilt comparison", outpath)


# Main
def run_vortex_analysis(track_dict, pl_dir, year, output_dir):
    """Run all vortex analyses for all tracks"""
    os.makedirs(output_dir, exist_ok=True)
    tilt_results  = {}
    therm_results = {}

    for track_id, track_df in sorted(track_dict.items()):
        track_out = os.path.join(output_dir, f"track_{track_id}")
        os.makedirs(track_out, exist_ok=True)

        std_print("INFO", "va", f"Track {track_id} - vortex tilt")
        try:
            tilt_df = compute_vortex_tilt(track_df, pl_dir, year, track_out)
            tilt_results[track_id] = tilt_df
        except Exception as e:
            std_print("ERROR", "va", f"Track {track_id} tilt failed", str(e))
            traceback.print_exc()

        std_print("INFO", "va", f"Track {track_id} - thermal structure")
        try:
            therm_df = compute_thermal_structure(track_df, pl_dir, year, track_out)
            therm_results[track_id] = therm_df
        except Exception as e:
            std_print("ERROR", "va", f"Track {track_id} thermal failed", str(e))
            traceback.print_exc()

        std_print("INFO", "va", f"Track {track_id} - vorticity column")
        try:
            compute_vorticity_column(track_df, pl_dir, year, track_out)
        except Exception as e:
            std_print("ERROR", "va", f"Track {track_id} vorticity column failed", str(e))
            traceback.print_exc()

        std_print("INFO", "va", f"Track {track_id} - IVT")
        try:
            compute_ivt(track_df, pl_dir, year, track_out)
        except Exception as e:
            std_print("ERROR", "va", f"Track {track_id} IVT failed", str(e))
            traceback.print_exc()

        std_print("INFO", "va", f"Track {track_id} - omega cross-section")
        try:
            compute_omega_cross_section(track_df, pl_dir, year, track_out)
        except Exception as e:
            std_print("ERROR", "va", f"Track {track_id} omega failed", str(e))
            traceback.print_exc()

        std_print("INFO", "va", f"Track {track_id} - recycling ratio")
        try:
            compute_recycling_ratio(track_df, track_id, track_out)
        except Exception as e:
            std_print("ERROR", "va", f"Track {track_id} recycling ratio failed", str(e))
            traceback.print_exc()

    # Global comparison
    std_print("INFO", "va", "Global - all-tracks tilt comparison")
    try:
        plot_all_tracks_tilt_comparison(tilt_results, track_dict, output_dir)
    except Exception as e:
        std_print("ERROR", "va", "All-tracks tilt comparison failed", str(e))
        traceback.print_exc()

    std_print("INFO", "va", "Vortex analysis complete", output_dir)
    return {"tilt": tilt_results, "thermal": therm_results}


# Helper functions
def load_pl_level(pl_dir, month_str, level_hpa):
    """Load one pressure-level month file.  Returns None if file missing"""
    cache_key = (month_str, level_hpa)
    if cache_key in DS_CACHE:
        return DS_CACHE[cache_key] 

    path = pl_path(pl_dir, month_str, level_hpa)
    if not os.path.exists(path):
        std_print("WARN", "va", "Pressure-level file not found", path)
        return None
    try:
        ds = xr.open_dataset(path)
        # Normalise time coordinate name
        if "valid_time" in ds.coords and "time" not in ds.coords:
            ds = ds.rename({"valid_time": "time"})
        # Clip RH to physical range
        if "r" in ds:
            ds["r"] = ds["r"].clip(0, 100)
        # Clip specific humidity to non-negative (spectral artefact)
        if "q" in ds:
            ds["q"] = ds["q"].clip(0, None)

        # Store normalised dataset
        DS_CACHE[cache_key] = ds
        return ds
    except Exception as e:
        std_print("ERROR", "va", f"Failed to open {path}", str(e))
        return None

def pl_path(pl_dir , month_str, level_hpa):
    """Build path to a pressure-level file"""
    return os.path.join(pl_dir, month_str, f"{month_str}-{level_hpa}hPa.nc")


def month_str_from_time(t: pd.Timestamp, year) -> str:
    """Return 'August-2024' style string from a timestamp"""
    return f"{t.strftime('%B')}-{year}"


def select_time_nearest(ds: xr.Dataset, t: pd.Timestamp) -> xr.Dataset:
    """Select nearest time step in a dataset"""
    return ds.sel(time=t, method="nearest")


def plot_ivt(ivt_df, track_df, track_id, output_dir):
    """IVT time series with land shading and convection threshold line"""
    times = pd.to_datetime(ivt_df["time"])
    is_land = track_df.sort_values("time")["is_land"].values.astype(bool) \
        if "is_land" in track_df.columns else np.ones(len(ivt_df), bool)

    fig, ax = plt.subplots(figsize=(16, 7))
    ivt = ivt_df["IVT"].values
    ax.plot(
        times,
        ivt,
        "-o",
        color="teal",
        linewidth=2.2,
        markersize=5,
        alpha=0.85,
        label="IVT (kg m⁻¹ s⁻¹)"
    )
    ax.fill_between(
        times,
        ivt,
        0,
        where=~np.isnan(ivt),
        color="teal",
        alpha=0.15
    )
    ax.axhline(250, color="darkorange", linestyle="--", linewidth=1.5, label="Min. convection threshold")
    ax.fill_between(
        times,
        0,
        1,
        where=is_land,
        transform=ax.get_xaxis_transform(),
        color="wheat",
        alpha=0.35,
        label="Land phase",
        zorder=0
    )
    ax.fill_between(
        times,
        0,
        1,
        where=~is_land,
        transform=ax.get_xaxis_transform(),
        color="lightblue",
        alpha=0.35,
        label="Ocean phase",
        zorder=0
    )
    for i in range(1, len(is_land)):
        if not is_land[i-1] and is_land[i]:
            ax.axvline(times.iloc[i], color="red", linestyle="--", linewidth=1.8, label="Landfall", zorder=5)
            break
    ax.set_ylabel("IVT (kg m⁻¹ s⁻¹)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: Integrated Vapour Transport (IVT)  850–300 hPa\n"
        "IVT drop over terrain = orographic blocking of moisture supply",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", **legend_style())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    outpath = os.path.join(output_dir, f"track_{track_id}_ivt.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "IVT plot", outpath)


def find_vortex_centre(vo_slice,track_lat, track_lon , search_radius_deg = SEARCH_RADIUS_DEG):
    """
    Find the relative vorticity maximum within `search_radius_deg` of the  single-level track centre.
    Returns (lat, lon) of the vorticity maximum, or (None, None) if no valid data in the search window.

    Reference:
    Uses the approach of Hunt & Turner (2016): restrict to a bounded search box then find the global max within that box
    """
    lat_min = track_lat - search_radius_deg
    lat_max = track_lat + search_radius_deg
    lon_min = track_lon - search_radius_deg
    lon_max = track_lon + search_radius_deg

    # ERA5 latitude is descending
    sub = vo_slice.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
    )
    if sub.size == 0 or sub.isnull().all():
        return None, None

    vals = sub.values
    if np.all(np.isnan(vals)):
        return None, None

    idx = np.unravel_index(np.nanargmax(vals), vals.shape)
    c_lat = float(sub.latitude.values[idx[0]])
    c_lon = float(sub.longitude.values[idx[1]])
    return c_lat, c_lon


def ambient_temperature(t_slice, centre_lat, centre_lon, ring_km = AMBIENT_RING_KM):
    """Compute the mean temperature in an annulus from ring_km/2 to ring_km around (centre_lat, centre_lon)"""
    lats = t_slice.latitude.values
    lons = t_slice.longitude.values
    lon2d, lat2d = np.meshgrid(lons, lats)
    dist = haversine_distance(centre_lon, centre_lat, lon2d.ravel(), lat2d.ravel())
    dist = dist.reshape(lat2d.shape)
    mask = (dist > ring_km / 2) & (dist <= ring_km)
    vals = t_slice.values[mask]
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if vals.size > 0 else np.nan


def plot_vortex_tilt(tilt_df, track_df, track_id, output_dir):
    """
    Three-panel time series:
      Top:    tilt magnitude (km) - primary metric
      Middle: tilt bearing (°) - direction of upper-level centre relative to low-level
      Bottom: speed (m/s) + is_land shading - for direct comparison
    """
    times = pd.to_datetime(tilt_df["time"])
    track_sorted = track_df.sort_values("time").copy()
    track_sorted["time"] = pd.to_datetime(track_sorted["time"])
    merge_cols = ["time"]
    if "is_land"  in track_sorted.columns:
        merge_cols.append("is_land")
    if "speed_ms" in track_sorted.columns:
        merge_cols.append("speed_ms")

    tilt_merged = pd.merge_asof(
        tilt_df[["time"]].assign(time=pd.to_datetime(tilt_df["time"])),
        track_sorted[merge_cols].drop_duplicates("time"),
        on="time", direction="nearest", tolerance=pd.Timedelta("2h")
    )
    is_land = tilt_merged["is_land"].fillna(False).values.astype(bool) \
        if "is_land"  in tilt_merged.columns else np.ones(len(tilt_df), bool)
    speed = tilt_merged["speed_ms"].values.astype(float) \
        if "speed_ms" in tilt_merged.columns else np.full(len(tilt_df), np.nan)

    # Figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    bbox = common_bbox()

    # Top: tilt magnitude
    ax = axes[0]
    tilt_km = tilt_df["tilt_magnitude_km"].values
    ax.plot(
        times,
        tilt_km,
        "-o",
        color="purple",
        linewidth=2.0,
        markersize=4,
        alpha=0.85,
        label="Tilt magnitude (km)"
    )
    # Land/ocean shading
    ax.fill_between(
        times,
        0,
        1,
        where=is_land,
        transform=ax.get_xaxis_transform(),
        color="wheat",
        alpha=0.35,
        label="Land phase",
        zorder=0
    )
    ax.fill_between(
        times,
        0,
        1,
        where=~is_land,
        transform=ax.get_xaxis_transform(),
        color="lightblue",
        alpha=0.35,
        label="Ocean phase",
        zorder=0
    )
    # Landfall line
    for i in range(1, len(is_land)):
        if not is_land[i-1] and is_land[i]:
            ax.axvline(times.iloc[i], color="red", linestyle="--", linewidth=1.5,
                       label="Landfall", zorder=5)
            break
    # Annotate mean land-phase tilt vs ocean-phase tilt
    if is_land.any() and (~is_land).any():
        ml = np.nanmean(tilt_km[is_land])
        mo = np.nanmean(tilt_km[~is_land])
        ax.text(
            0.02,
            0.95,
            f"Mean tilt - Ocean: {mo:.0f} km   Land: {ml:.0f} km",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=bbox,
            family="monospace"
        )
    ax.set_ylabel("Tilt 850->200 hPa (km)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: Vortex Tilt\n(850 hPa -> 200 hPa vorticity centre displacement)",
        fontsize=13,
        fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", **legend_style({"fontsize": 9}))

    # Middle: tilt bearing
    ax = axes[1]
    bearing = tilt_df["tilt_bearing_deg"].values
    ax.plot(times, bearing, "-o", color="steelblue", linewidth=2.0,markersize=4, alpha=0.85, label="Tilt bearing (°)")
    # Reference: Hunt & Turner report WNW tilt (~285–300°) for mature MDs
    ax.axhline(292, color="navy", linestyle="--", linewidth=0.8, alpha=0.6,label="WNW (292°) - MD tilt direction")
    ax.fill_between(
        times,
        0,
        1,
        where=is_land,
        transform=ax.get_xaxis_transform(),
        color="wheat",
        alpha=0.35,
        zorder=0
    )
    ax.set_ylim(0, 360)
    ax.set_ylabel("Tilt bearing (°)", fontsize=12, fontweight="bold")
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_yticklabels(["N", "E", "S", "W", "N"])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", **legend_style({"fontsize": 9}))

    # Bottom: translation speed
    ax = axes[2]
    ax.plot(
        times,
        speed,
        "-o",
        color="royalblue",
        linewidth=2.0,
        markersize=4,
        alpha=0.85,
        label="Translation speed (m/s)"
    )
    ax.fill_between(
        times,
        0,
        1,
        where=is_land,
        transform=ax.get_xaxis_transform(),
        color="wheat",
        alpha=0.35,
        zorder=0
    )
    ax.set_ylabel("Speed (m/s)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", **legend_style({"fontsize": 9}))

    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save
    vortex_tilt_path = os.path.join(output_dir, f"track_{track_id}_vortex_tilt.png")
    plt.savefig(vortex_tilt_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "Vortex tilt plot", vortex_tilt_path)


# TODO: Add intermidiate levels and only to one plot (no subplots)
def plot_thermal_structure(therm_df, track_df, track_id, output_dir):
    """
    3-panel: cold-core anomaly (850 hPa) and warm-core anomaly (400 hPa) and top 200hPa
    with land/ocean shading and landfall marker.
    """
    times = pd.to_datetime(therm_df["time"])
    # is_land = track_df.sort_values("time")["is_land"].values.astype(bool) \
    #     if "is_land" in track_df.columns else np.ones(len(therm_df), bool)
    track_sorted = track_df.sort_values("time").copy()
    track_sorted["time"] = pd.to_datetime(track_sorted["time"])
    merged = pd.merge_asof(
        therm_df[["time"]].assign(time=times),
        track_sorted[["time", "is_land"]].drop_duplicates("time")
        if "is_land" in track_sorted.columns
        else track_sorted[["time"]].assign(is_land=True),
        on="time", direction="nearest", tolerance=pd.Timedelta("2h")
    )
    is_land = merged["is_land"].fillna(False).values.astype(bool)

    # Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    bbox = common_bbox()

    for ax, col, color, title, ref_line, ref_label in [
        (
            ax1,
            "T850_anom",
            "steelblue",
            "Cold-core anomaly at 850 hPa  (ΔT relative to 300 km ambient ring)",
            0,
            "ΔT = 0"
        ),
        (
            ax2,
            "T200_anom",
            "lightgreen",
            "Warm-core anomaly at 200 hPa  (ΔT relative to 300 km ambient ring)",
            0,
            "ΔT = 0"
        ),
    ]:
        vals = therm_df[col].values
        ax.plot(
            times,
            vals,
            "-o",
            color=color,
            linewidth=2.0,
            markersize=4,
            alpha=0.85
        )
        ax.fill_between(
            times,
            vals,
            0,
            where=~np.isnan(vals),
            color=color,
            alpha=0.15
        )
        ax.axhline(ref_line, color="black", linewidth=0.8, linestyle="--", label=ref_label)
        ax.fill_between(
            times,
            0,
            1,
            where=is_land,
            transform=ax.get_xaxis_transform(),
            color="wheat",
            alpha=0.35,
            label="Land phase",
            zorder=0
        )
        ax.fill_between(
            times,
            0,
            1,
            where=~is_land,
            transform=ax.get_xaxis_transform(),
            color="lightblue",
            alpha=0.35,
            label="Ocean phase",
            zorder=0
        )
        for i in range(1, len(is_land)):
            if not is_land[i-1] and is_land[i]:
                ax.axvline(times.iloc[i], color="red", linestyle="--",linewidth=1.5, label="Landfall", zorder=5)
                break
        ax.set_ylabel("ΔT (K)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", **legend_style({"fontsize": 9}))
    ax2.set_xlabel("Time", fontsize=12, fontweight="bold")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.suptitle(
        f"Track {track_id}: Thermal Structure\nCold-core 850 hPa, Warm-core 200 hPa",
        fontsize=13,
        fontweight="bold",
        y=1
    )
    plt.tight_layout()

    # Save
    theraml_sturcture_path = os.path.join(output_dir, f"track_{track_id}_thermal_structure.png")
    plt.savefig(theraml_sturcture_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "Thermal structure plot", theraml_sturcture_path)


def plot_vorticity_column(col_df, track_df, track_id, output_dir):
    """Two-panel: vorticity at 4 levels as line plot, and Δvo(850-200) below
    Mature MD: positive at 850+700+500 hPa, negative (anticyclonic) at 300+200 hPa
    """
    times = pd.to_datetime(col_df["time"])
    is_land = track_df.sort_values("time")["is_land"].values.astype(bool) \
        if "is_land" in track_df.columns else np.ones(len(col_df), bool)

    # Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    colors = {
        "vo_850": "royalblue",
        "vo_700": "blue",
        "vo_600": "green",
        "vo_500": "yellow",
        "vo_400": "gray",
        "vo_300": "darkorange",
        "vo_200": "red",
    }
    labels = {
        "vo_850": "850 hPa",
        "vo_700": "700 hPa",
        "vo_600": "600 hPa",
        "vo_500": "500 hPa",
        "vo_400": "400 hPa",
        "vo_300": "300 hPa",
        "vo_200": "200 hPa"
    }
    for col, color in colors.items():
        if col in col_df.columns:
            ax1.plot(
                times,
                col_df[col].values * 1e4,
                "-o",
                color=color,
                linewidth=1.8,
                markersize=3,
                alpha=0.85,
                label=f"vo {labels[col]} (x10⁻⁴ s⁻¹)"
            )
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.fill_between(
        times,
        0,
        1,
        where=is_land,
        transform=ax1.get_xaxis_transform(),
        color="wheat",
        alpha=0.35,
        label="Land phase",
        zorder=0
    )
    for i in range(1, len(is_land)):
        if not is_land[i-1] and is_land[i]:
            ax1.axvline(times.iloc[i], color="red", linestyle="--", linewidth=1.5, label="Landfall")
            break
    ax1.set_ylabel("Relative vorticity (x10⁻⁴ s⁻¹)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Track {track_id}: Relative vorticity vertical profile",
        fontsize=13,
        fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper right", **legend_style({"fontsize": 9}))

    # Δvo(850-200)
    delta = col_df["delta_vo_850_200"].values * 1e4
    ax2.plot(
        times,
        delta,
        "-o",
        color="purple",
        linewidth=2.0,
        markersize=4,
        alpha=0.85,
        label="Δvo = vo(850) − vo(200)  (x10⁻⁴ s⁻¹)"
    )
    ax2.fill_between(
        times,
        delta,
        0,
        where=~np.isnan(delta),
        color="purple",
        alpha=0.12
    )
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.fill_between(
        times,
        0,
        1,
        where=is_land,
        transform=ax2.get_xaxis_transform(),
        color="wheat",
        alpha=0.35,
        zorder=0
    )
    ax2.set_ylabel("Δvo (x10⁻⁴ s⁻¹)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Column coherence: Δvo(850−200)\n"
        "Drop to zero = system losing vertical structure",
        fontsize=13,
        fontweight="bold"
    )
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper right", **legend_style({"fontsize": 9}))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save
    vorticity_column_path = os.path.join(output_dir, f"Track_{track_id}_vorticity_column.png")
    plt.savefig(vorticity_column_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "Vorticity column plot", vorticity_column_path)


# Omega cross-section at 700 hPa
# INFO:
# Why 700 hPa for omega? 700 hPa is the level where orographic uplift signal is
# strongest over the Eastern Ghats (~1500 m peak elevation = ~850 hPa surface, so
# forced ascent signal propagates to ~700 hPa). It's also above boundary layer so
# it captures free-tropospheric vertical motion rather than surface-forced turbulence.
# 500 hPa would show weaker signal because it's above most of orographic forcing.
def compute_omega_cross_section(track_df, pl_dir,year, output_dir):
    """
    At each land-phase time step, extract 700 hPa omega (Pa/s) along a
    perpendicular transect +/-OMEGA_TRANSECT_KM across the track bearing

    Negative omega = ascent (Pa/s convention). A spike in upward omega on
    the upslope side of terrain indicates forced orographic lifting - which
    enhances precipitation but also drains moisture faster from the system
    """
    track_id = int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    df = track_df.sort_values("time").reset_index(drop=True)

    # Only land phase
    is_land = df["is_land"].values.astype(bool) if "is_land" in df.columns else np.ones(len(df), bool)
    land_df = df[is_land].reset_index(drop=True)
    if len(land_df) < 3:
        std_print("WARN", "va", f"Track {track_id}: insufficient land-phase points for omega cross-section")
        return

    times = pd.to_datetime(land_df["time"])

    # Transect distance axis (km, negative = left of track, positive = right)
    n_pts = 25
    dist_km = np.linspace(-OMEGA_TRANSECT_KM, OMEGA_TRANSECT_KM, n_pts)
    omega_matrix = np.full((len(land_df), n_pts), np.nan)

    for i in range(len(land_df)):
        t      = times.iloc[i]
        c_lat  = float(land_df["latitude"].iloc[i])
        c_lon  = float(land_df["longitude"].iloc[i])
        bear   = float(land_df["bearing"].iloc[i]) if "bearing" in land_df.columns else 270.0
        perp   = (bear + 90) % 360  # perpendicular direction

        ds700 = load_level_at_time(pl_dir, t, 700, year)
        if ds700 is None or "w" not in ds700:
            continue
        w_slice = ds700["w"].squeeze()

        for j, d in enumerate(dist_km):
            # Project along perpendicular bearing
            dlat_rad = np.radians(d / 111.0)   # approximate
            dlon_rad = np.radians(d / (111.0 * np.cos(np.radians(c_lat))))
            p_lat = c_lat + np.cos(np.radians(perp)) * np.degrees(dlat_rad)
            p_lon = c_lon + np.sin(np.radians(perp)) * np.degrees(dlon_rad)
            try:
                val = float(w_slice.sel(
                    latitude=p_lat, longitude=p_lon, method="nearest"
                ).values)
                omega_matrix[i, j] = val
            except Exception:
                pass

    # Figure - Hovmöller plot
    fig, ax = plt.subplots(figsize=(14, 7))

    vmax = np.nanpercentile(np.abs(omega_matrix), 97)
    vmax = max(vmax, 0.5)
    im = ax.pcolormesh(
        dist_km,
        np.arange(len(land_df)),
        omega_matrix,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto"
    )
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Track centre")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("ω at 700 hPa (Pa s⁻¹)\nnegative = ascent", fontsize=10)
    ax.set_yticks(np.arange(len(land_df))[::max(1, len(land_df)//8)])
    ax.set_yticklabels(
        [times.iloc[k].strftime("%d/%m %H:%M")
         for k in np.arange(len(land_df))[::max(1, len(land_df)//8)]],
        fontsize=8
    )
    ax.set_xlabel("Distance from track centre (km)\nnegative = left, positive = right of motion", fontsize=12)
    ax.set_ylabel("Time (land phase only)", fontsize=12)
    ax.set_title(
        f"Track {track_id}: 700 hPa Vertical Velocity Cross-Section  (perpendicular to track)\n"
        "Asymmetric ascent on upslope side = orographic forced lifting",
        fontsize=13,
        fontweight="bold"
    )
    ax.legend(loc="upper right", **legend_style())
    plt.tight_layout()
    outpath = os.path.join(output_dir, f"track_{track_id}_omega_cross_section.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "Omega cross-section Hovmöller", outpath)

    # Export
    os.makedirs(output_dir, exist_ok=True)
    omega_col = [f"omega_d{d:.0f}km" for d in dist_km]
    omega_out = pd.DataFrame(omega_matrix, columns=omega_col, index=times)
    omega_out.index.name = "time"
    omega_csv_path = os.path.join(output_dir, f"track_{track_id}_omega_cross_section.csv")
    omega_out.to_csv(omega_csv_path)
    std_print("EXPORT", "va", "Omega cross-section CSV", omega_csv_path)


def plot_recycling_ratio(rr_df, track_id, output_dir):
    """Plot recycling ratio and component fluxes alongside land/ocean phase"""
    times = pd.to_datetime(rr_df["time"])
    is_land = rr_df["is_land"].values.astype(bool)
    rho = rr_df["recycling_ratio"].values
    E = rr_df["E_mm_h"].values
    P = rr_df["P_mm_h"].values
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
 
    shade_kw = dict(transform=ax1.get_xaxis_transform(), zorder=0, alpha=0.35)
    for ax in (ax1, ax2):
        ax.fill_between(
            times,
            0,
            1,
            where=is_land,
            color="wheat",
            **shade_kw,
            label="Land phase"
        )
        ax.fill_between(
            times,
            0,
            1,
            where=~is_land,
            color="lightblue",
            **shade_kw,
            label="Ocean phase"
        )
        for i in range(1, len(is_land)):
            if not is_land[i-1] and is_land[i]:
                ax.axvline(
                    times.iloc[i],
                    color="red",
                    linestyle="--",
                    linewidth=1.8,
                    label="Landfall",
                    zorder=5
                )
                break
 
    # Panel 1: ρ (recycling ratio)
    ax1.plot(
        times,
        rho,
        "-o",
        color="teal",
        linewidth=2.2,
        markersize=5,
        alpha=0.85,
        label="Recycling ratio ρ = E / P"
    )
    ax1.axhline(0.25, color="darkorange", linestyle="--", linewidth=1.5, label="ρ = 0.25  (significant local recycling)")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Recycling ratio ρ", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Track {track_id}: Precipitation Recycling Ratio  ρ = E / P\n"
        "ρ → 1: rainfall is locally recycled (BOE signature)  |  "
        "ρ → 0: rainfall from advected oceanic moisture",
        fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper right", **legend_style({"fontsize": 9}))
 
    # Panel 2: E and P fluxes
    ax2.plot(
        times,
        P,
        "-o",
        color="royalblue",
        linewidth=2.0,
        markersize=4,
        alpha=0.85,
        label="Precipitation P (mm h⁻¹)"
    )
    ax2.plot(
        times,
        E,
        "-o",
        color="tomato",
        linewidth=2.0,
        markersize=4,
        alpha=0.85,
        label="Evaporation E from LHF (mm h⁻¹)"
    )
    ax2.set_ylabel("Flux (mm h⁻¹)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper right", **legend_style({"fontsize": 9}))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save
    recycling_ratio_path = os.path.join(output_dir, f"track_{track_id}_recycling_ratio.png")
    plt.savefig(recycling_ratio_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "Recycling ratio plot", recycling_ratio_path)
