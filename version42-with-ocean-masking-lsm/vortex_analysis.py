# NOTE:
# Vortex analysis - monsoon depression (md) 3-d structure
# Uses ERA5 pressure-level data (u, v, vo, t, q) at 850/700/600/500/400/300/200 hPa.
# All per-track analyses save outputs to track_dir/vortex_analysis/.
# The global comparison (all-tracks tilt) saves to Global_vortex_analysis/.
# 1. Vortex Tilt  [compute_vortex_tilt]
#     Horizontal displacement of the relative vorticity (vo) maximum between 850 and 200 hPa.
#     Typical MD mature-phase tilt: WNW direction (~285–300°), magnitude 50–300 km.
#     Increasing tilt after landfall over terrain = frictional decoupling of low-level centre.
#     Reference: Hunt & Turner (2016) MWR.
# 2. Thermal Structure  [compute_thermal_structure]
#     Temperature anomaly (centre minus 300 km ambient ring) at all pressure levels.
#     Cold core at 850 hPa (−2 to −4 K at peak) + warm core at 400 hPa (+1 to +3 K).
#     Warming of cold core toward 0 K = system weakening.
#     Reference: Rashid & Srinivasan (2005).
# 3. Vorticity Column Coherence  [compute_vorticity_column]
#     Relative vorticity vo at 850/700/600/500/400/300/200 hPa at the track centre.
#     Δvo = vo_850 − vo_200: positive = healthy low-level dominated vortex;
#     sharp drop = vertical coherence loss = dissipation precursor.
# 4. Integrated Vapour Transport (IVT) [compute_ivt]
#     IVT = (1/g) x \sqrt{(Σ q.u.Δp)^2 + (Σ q.v.Δp)^2}  [kg/ms]
#     Integrated 850->200 hPa using trapezoidal Δp weights.
#     IVT < 250 kg/ms = below threshold for sustained deep convection.
#     Drop over terrain = orographic blocking of moisture supply.
# 5. Column Moisture Tendency (∂W/∂t)  [compute_column_moisture_tendency]
#     Precipitable water W = (1/g) x Σ(q.Δp) [mm] integrated 850->200 hPa.
#     ∂W/∂t computed via np.gradient.
#     Negative ∂W/∂t on land = MD consuming moisture faster than replenishment -> decay signal.
# 6. Moisture Flux Convergence (MFC) at 850 hPa  [compute_moisture_flux_convergence]
#     MFC = −(1/g) x [∂(qu)/∂x + ∂(qv)/∂y] x Δp_850 -> mm/hr
#     Positive = moisture converging = convection sustained.
#     Negative = divergence = dissipation precursor.
#     Collapse over terrain = orographic wind deflection cutting off low-level moisture inflow.
# 7. θ_e Vertical Profile  [compute_theta_e_profile]
#     Equivalent potential temperature θ_e at each pressure level 850->200 hPa.
#     Formula: θ_e = T x (P0/P)^(Rd/cp) x exp(Lv x q_kg / (cp x T))   [Bolton 1980]
#     Note: q must be in kg/kg (specific humidity from ERA5 directly - NOT mixing ratio g/kg).
#     Uniform θ_e with height = well-mixed moist column = active convection.
#     Upper-level θ_e collapse (300–200 hPa) before surface -> stable cap -> weakening.
# 8. Mixing Ratio Vertical Profile  [compute_vertical_mixing_ratio]
#     w = q / (1 − q) x 1000  [g/kg] at each pressure level 850->200 hPa.
#     High w throughout 850–500 hPa = deep moisture column sustaining MD.
#     Sharp w drop at 850 hPa after terrain crossing = orographic blocking.
#     Time–height section plot as θ_e.
# 9. Omega Cross-Section  [compute_omega_cross_section]
#     700 hPa vertical velocity ω (Pa/s) along ±150 km perpendicular transect to track.
#     Land phase only. Negative ω = ascent.
#     Asymmetric ascent on upslope side = orographic forced lifting.
# Global comparison  [global_vortex_analysis]:
#     All-tracks vortex tilt evolution aligned to landfall (t=0).
#
# References:
#  - Hunt & Turner (2016) MWR - MD 3D composite, 850 hPa vorticity centre definition
#  - Rashid & Srinivasan (2005) MAP - vorticity budget, cold/warm-core characterisation
#  - Mishra (2021) MWR - MD development and large-scale interaction

import os
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize, PowerNorm

from utils import (
    add_land_ocean_shading,
    common_bbox,  # find_landfall_time,
    haversine_distance,
    legend_style,
    std_print,
)

warnings.filterwarnings(
    "ignore",
    message="facecolor will have no effect",
    category=UserWarning,
    module="cartopy",
)

LEVELS = [850, 700, 600, 500, 400, 300, 200]  # hPa
SEARCH_RADIUS_DEG = (
    3.0  # degrees, vortex centre search radius around single-level track
)
AMBIENT_RING_KM = 300.0  # km, radius for temperature anomaly background ring
OMEGA_TRANSECT_KM = (
    150.0  # km, perpendicular transect half-width for omega cross-section
)

# Constants
G = 9.80665  # m/s^2
L_v = 2.501e6  # J/kg, latent heat of vaporisation
RHO_W = 1000.0  # kg/m^3, water density
Rd = 287.05
cp = 1005.7
P0 = 100000.0

# # Map domain
# FULL_EXTENT = [70.0, 95.0, 7.0, 30.0]  # lon_min, lon_max, lat_min, lat_max
# JJAS_MONTHS = [6, 7, 8, 9]

# Font sizes
FS_TITLE = 14
FS_LABEL = 12
FS_TICK = 10
FS_LEGEND = 10
FS_ANNOT = 9

# Dataset cache
DS_CACHE = {}


# Data loading
def load_level_at_time(pl_dir, t, level_hpa, year):
    """Return a single-timestep Dataset slice from the cached month file"""
    month_str = month_str_from_time(t, year)
    ds = load_pl_level(pl_dir, month_str, level_hpa)
    if ds is None:
        return None
    return ds.sel(time=t, method="nearest")


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
        is_land_track = (
            track_df.sort_values("time")["is_land"].values.astype(bool)
            if "is_land" in track_df.columns
            else np.ones(len(tilt_df), bool)
        )

        times = pd.to_datetime(tilt_df["time"])
        tilt_km = tilt_df["tilt_magnitude_km"].values

        # Find landfall time
        landfall_time = None
        for i in range(1, len(is_land_track)):
            if not is_land_track[i - 1] and is_land_track[i]:
                landfall_time = times.iloc[i]
                break

        # Convert ALL x values to relative hours
        ref = landfall_time if landfall_time is not None else times.iloc[0]
        hours = np.array([(t - ref).total_seconds() / 3600 for t in times])

        track_times_arr = pd.to_datetime(track_df.sort_values("time")["time"]).values
        is_land_arr = np.zeros(len(times), dtype=bool)
        for k, tt in enumerate(times.values):
            nearest_idx = np.argmin(np.abs(track_times_arr - tt))
            is_land_arr[k] = (
                is_land_track[nearest_idx]
                if nearest_idx < len(is_land_track)
                else False
            )

        # Land / ocean background shading
        add_land_ocean_shading(
            ax, hours, is_land_arr, land_color="wheat", ocean_color="lightblue"
        )
        # # Landfall line
        # if landfall_time is not None:
        #     ax.axvline(
        #         landfall_time,
        #         color="red",
        #         linestyle="--",
        #         linewidth=1.8,
        #         label="Landfall",
        #         zorder=5,
        #     )

        ax.plot(
            hours,
            tilt_km,
            "-o",
            color=color,
            linewidth=2,
            markersize=4,
            alpha=0.8,
            label=f"Track {track_id}",
        )
        # Land-phase markers (diamonds)
        land_hours = hours[is_land_arr[: len(hours)]]
        land_tilt = tilt_km[is_land_arr[: len(tilt_km)]]
        ax.scatter(
            land_hours,
            land_tilt,
            color=color,
            s=60,
            marker="D",
            zorder=3,
            alpha=0.6,
        )

    # x-axis reference lines for \pm12h terrain window
    ax.axvspan(-6, 18, alpha=0.08, color="orange", label="Terrain interaction window")
    ax.axhline(
        150, color="gray", linestyle=":", linewidth=1, label="Tilt threshold (150 km)"
    )

    ax.axvline(
        0, color="black", linestyle="--", linewidth=1.5, label="Landfall (t = 0)"
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hours relative to landfall", fontsize=FS_LABEL, fontweight="bold")
    ax.set_ylabel(
        "Vortex tilt 850 -> 200 hPa (km)", fontsize=FS_LABEL, fontweight="bold"
    )
    # "Diamonds = land-phase points.  Rising tilt after landfall = terrain-induced decoupling",
    ax.set_title(
        "All Tracks: Vortex Tilt Evolution", fontsize=FS_TITLE, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", **legend_style())
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    vortex_tilt_path = os.path.join(output_dir, "All_tracks_vortex_tilt_comparison.png")
    plt.savefig(vortex_tilt_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "All-tracks tilt comparison", vortex_tilt_path)


# Run vortex analyese per track id
def run_track_vortex_analysis(track_df, track_id, pl_dir, year, output_dir):
    """Run all vortex analyses for MD track

    Per-track analyses:
      - Vortex tilt          (850 -> 200 hPa vorticity centre displacement)
      - Thermal structure    (cold-core 850 hPa + warm-core 400/200 hPa anomalies)
      - Vorticity column     (vo profile 850->200 hPa + Δvo coherence metric)
      - IVT                  (Integrated Vapour Transport 850->200 hPa)
      - Omega cross-section  (Vertical velocity perpendicular to track, land phase)
    """
    os.makedirs(output_dir, exist_ok=True)
    tilt_df = None
    therm_df = None
    col_df = None
    ivt_df = None
    cmt_df = None
    mfc_df = None
    theta_e_df = None
    mixing_ratio_df = None

    std_print("INFO", "va", "--- Vortex tilt ---")
    try:
        tilt_df = compute_vortex_tilt(track_df, pl_dir, year, output_dir)
    except Exception as e:
        std_print("ERROR", "va", "Tilt failed", str(e))
        traceback.print_exc()

    std_print("INFO", "va", "--- Thermal structure ---")
    try:
        therm_df = compute_thermal_structure(track_df, pl_dir, year, output_dir)
    except Exception as e:
        std_print("ERROR", "va", "Thermal failed", str(e))
        traceback.print_exc()

    std_print("INFO", "va", "--- Vorticity column ---")
    try:
        col_df = compute_vorticity_column(track_df, pl_dir, year, output_dir)
    except Exception as e:
        std_print("ERROR", "va", "Vorticity column failed", str(e))
        traceback.print_exc()

    std_print("INFO", "va", "--- IVT (Integrated Vapour Transport) ---")
    try:
        ivt_df = compute_ivt(track_df, pl_dir, year, output_dir)
    except Exception as e:
        std_print("ERROR", "va", "IVT failed", str(e))
        traceback.print_exc()

    std_print("INFO", "va", "--- Moisture Flux Convergence at 850 hPa ---")
    try:
        mfc_df = compute_moisture_flux_convergence(track_df, pl_dir, year, output_dir)
    except Exception as e:
        std_print("ERROR", "va", "MFC failed", str(e))
        traceback.print_exc()

    std_print("INFO", "va", "--- Column Moisture Tendency (∂W/∂t) ---")
    try:
        cmt_df = compute_column_moisture_tendency(track_df, pl_dir, year, output_dir)
    except Exception as e:
        std_print("ERROR", "va", "Column moisture tendency failed", str(e))
        traceback.print_exc()

    std_print("INFO", "va", "--- Mixing ratio vertical profile ---")
    try:
        mixing_ratio_df = compute_vertical_mixing_ratio(
            track_df, pl_dir, year, output_dir
        )
    except Exception as e:
        std_print("ERROR", "va", "Mixing ratio profile failed", str(e))
        traceback.print_exc()

    std_print("INFO", "va", "--- θ_e vertical profile ---")
    try:
        theta_e_df = compute_theta_e_profile(track_df, pl_dir, year, output_dir)
    except Exception as e:
        std_print("ERROR", "va", "θ_e profile failed", str(e))
        traceback.print_exc()

    std_print("INFO", "va", "--- Omega cross-section ---")
    try:
        compute_omega_cross_section(track_df, pl_dir, year, output_dir)
    except Exception as e:
        std_print("ERROR", "va", "Omega failed", str(e))
        traceback.print_exc()

    return {
        "track_id": track_id,
        "tilt": tilt_df,
        "thermal": therm_df,
        "vorticity_col": col_df,
        "ivt": ivt_df,
        "cmt": cmt_df,
        "mfc": mfc_df,
        "theta_e_profile": theta_e_df,
        "mixing_ratio_profile": mixing_ratio_df,
    }


# FIX: overlapping land-ocean shading
def global_vortex_analysis(tilt_results, track_dict, global_out):
    """
    All-tracks vortex comparison plots.
    - All-tracks vortex tilt evolution (aligned to landfall)
    """
    os.makedirs(global_out, exist_ok=True)
    std_print("INFO", "va", "Global - all-tracks tilt comparison")
    try:
        plot_all_tracks_tilt_comparison(tilt_results, track_dict, output_dir=global_out)
    except Exception as e:
        std_print("ERROR", "va", "All-tracks tilt comparison failed", str(e))
        traceback.print_exc()
    std_print("INFO", "va", "Global vortex finalization complete", global_out)


# Helper functions
# TODO: add all intermediate levels
# Vortex tilt
def compute_vortex_tilt(track_df, pl_dir, year, output_dir):
    """
    Compute the vortex tilt vector (850 hPa -> 200 hPa) at each track time step.

    The tilt vector is the horizontal displacement between the 850 hPa and
    200 hPa relative vorticity maxima.

    Reference:
    Hunt & Turner (2016) show that Indian  MDs have a characteristic WNW tilt with height in their mature phase;
    an anomalous increase in tilt magnitude during the land phase - particularly  collocated with terrain
    - indicates low-level frictional decoupling.
    """
    track_id = (
        int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    )
    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])

    # TODO: implement like this (from thermal struc.)
    #      rec = {"time": t}
    #      for level in LEVELS:
    #          rec[f"T{level}_centre"] = np.nan
    #          rec[f"T{level}_anom"]   = np.nan
    #
    #      for level in LEVELS:
    #          ds = load_level_at_time(pl_dir, t, level , year)
    #          if ds is None or "t" not in ds:
    #              continue
    #          t_slice = ds["t"].squeeze()
    #
    #          try:
    #              # Vortex centre temperature (interpolate to track position)
    #              t_centre = float(t_slice.sel(latitude=tlat, longitude=tlon, method="nearest").values)
    #          except Exception:
    #              t_centre = np.nan
    #          t_amb = ambient_temperature(t_slice, tlat, tlon)
    #          rec[f"T{level}_centre"] = t_centre
    #          rec[f"T{level}_anom"]   = t_centre - t_amb
    #
    #      theraml_sturcture_records.append(rec)
    # therm_df = pd.DataFrame(theraml_sturcture_records)

    # Build is_land lookup from track_df so every record carries it.
    # tilt_df has different timestamps to track_df (pressure-level data may have gaps),
    # so we carry the column forward and plot functions use tilt_df directly.
    is_land_series = (
        df["is_land"].values.astype(bool)
        if "is_land" in df.columns
        else np.ones(len(df), bool)
    )

    records = []
    for i, t in enumerate(times):
        tlon = float(df["longitude"].iloc[i])
        tlat = float(df["latitude"].iloc[i])
        land_at_t = bool(is_land_series[i])

        # Load 850 and 200 hPa vorticity at current time step
        ds850 = load_level_at_time(pl_dir, t, 850, year)
        ds200 = load_level_at_time(pl_dir, t, 200, year)
        if ds850 is None or ds200 is None:
            records.append(
                {
                    "time": t,
                    "lat_850": np.nan,
                    "lon_850": np.nan,
                    "lat_200": np.nan,
                    "lon_200": np.nan,
                    "tilt_magnitude_km": np.nan,
                    "tilt_bearing_deg": np.nan,
                    "is_land": land_at_t,
                }
            )
            continue

        vo850 = ds850["vo"].squeeze()
        vo200 = ds200["vo"].squeeze()
        lat850, lon850 = find_vortex_centre(vo850, tlat, tlon)
        lat200, lon200 = find_vortex_centre(vo200, tlat, tlon)

        if lat850 is None or lat200 is None:
            records.append(
                {
                    "time": t,
                    "lat_850": lat850,
                    "lon_850": lon850,
                    "lat_200": lat200,
                    "lon_200": lon200,
                    "tilt_magnitude_km": np.nan,
                    "tilt_bearing_deg": np.nan,
                    "is_land": land_at_t,
                }
            )
            continue

        # Tilt vector: from 850 hPa centre toward 200 hPa centre
        tilt_km = haversine_distance(lon850, lat850, lon200, lat200)

        # Bearing of tilt (degrees, 0=N, clockwise)
        dlat = lat200 - lat850
        dlon = lon200 - lon850
        bearing = np.degrees(np.arctan2(dlon * np.cos(np.radians(lat850)), dlat)) % 360

        records.append(
            {
                "time": t,
                "lat_850": lat850,
                "lon_850": lon850,
                "lat_200": lat200,
                "lon_200": lon200,
                "tilt_lat_deg": dlat,
                "tilt_lon_deg": dlon,
                "tilt_magnitude_km": tilt_km,
                "tilt_bearing_deg": bearing,
                "is_land": land_at_t,
            }
        )
    tilt_df = pd.DataFrame(records)

    # Export
    os.makedirs(output_dir, exist_ok=True)
    vortex_tilt_CSV = os.path.join(output_dir, f"track_{track_id}_vortex_tilt.csv")
    tilt_df.to_csv(vortex_tilt_CSV, index=False)
    std_print("EXPORT", "va", "Vortex tilt CSV", vortex_tilt_CSV)

    plot_vortex_tilt(tilt_df, track_df, track_id, output_dir)
    return tilt_df


# Thermal structure - cold-core + warm-core
def compute_thermal_structure(track_df, pl_dir, year, output_dir):
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
    track_id = (
        int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    )
    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])

    theraml_sturcture_records = []
    for i, t in enumerate(times):
        tlon = float(df["longitude"].iloc[i])
        tlat = float(df["latitude"].iloc[i])

        rec = {"time": t}
        for level in LEVELS:
            rec[f"T{level}_centre"] = np.nan
            rec[f"T{level}_anom"] = np.nan

        for level in LEVELS:
            ds = load_level_at_time(pl_dir, t, level, year)
            if ds is None or "t" not in ds:
                continue
            t_slice = ds["t"].squeeze()

            try:
                # Vortex centre temperature (interpolate to track position)
                t_centre = float(
                    t_slice.sel(latitude=tlat, longitude=tlon, method="nearest").values
                )
            except Exception:
                t_centre = np.nan
            t_amb = ambient_temperature(t_slice, tlat, tlon)
            rec[f"T{level}_centre"] = t_centre
            rec[f"T{level}_anom"] = t_centre - t_amb

        theraml_sturcture_records.append(rec)
    therm_df = pd.DataFrame(theraml_sturcture_records)

    if therm_df["T850_anom"].isnull().all() and therm_df["T200_anom"].isnull().all():
        std_print(
            "WARN",
            "va",
            f"Track {track_id}: All thermal structure data are NaN - skipping plot",
        )
        return therm_df

    # Export
    os.makedirs(output_dir, exist_ok=True)
    thermal_structure_CSV = os.path.join(
        output_dir, f"Track_{track_id}_thermal_structure.csv"
    )
    therm_df.to_csv(thermal_structure_CSV, index=False)
    std_print("EXPORT", "va", "Thermal structure CSV", thermal_structure_CSV)

    plot_thermal_structure(therm_df, track_df, track_id, output_dir)
    return therm_df


# Vorticity column coherence (850 -> 200 hPa full column)
def compute_vorticity_column(track_df, pl_dir, year, output_dir):
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
    track_id = (
        int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    )
    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])

    is_land_series_col = (
        df["is_land"].values.astype(bool)
        if "is_land" in df.columns
        else np.ones(len(df), bool)
    )

    vorticity_column_records = []
    for i, t in enumerate(times):
        lon_at_t = float(df["longitude"].iloc[i])
        lat_at_t = float(df["latitude"].iloc[i])
        rec = {"time": t, "is_land": bool(is_land_series_col[i])}

        for level in LEVELS:
            ds = load_level_at_time(pl_dir, t, level, year)
            if ds is None or "vo" not in ds:
                rec[f"vo_{level}"] = np.nan
                continue
            vo_slice = ds["vo"].squeeze()

            try:
                vo_val = float(
                    vo_slice.sel(
                        latitude=lat_at_t, longitude=lon_at_t, method="nearest"
                    ).values
                )
            except Exception:
                vo_val = np.nan
            rec[f"vo_{level}"] = vo_val
        vorticity_column_records.append(rec)

    col_df = pd.DataFrame(vorticity_column_records)
    col_df["delta_vo_850_200"] = col_df["vo_850"] - col_df["vo_200"]

    # Export
    os.makedirs(output_dir, exist_ok=True)
    vorticity_column_csv = os.path.join(
        output_dir, f"track_{track_id}_vorticity_column.csv"
    )
    col_df.to_csv(vorticity_column_csv, index=False)
    std_print("EXPORT", "va", "Vorticity column", vorticity_column_csv)

    plot_vorticity_column(col_df, track_df, track_id, output_dir)
    return col_df


# Integrated Vapour Transport (IVT)
def compute_ivt(track_df, pl_dir, year, output_dir):
    """
    IVT = (1/g) * sqrt( (Σ q.u.Δp)^2 + (Σ q.v.Δp)^2 )

    Integrate from 850 to 200 hPa using available levels.

    A drop in IVT during the land phase - especially over terrain - indicates that orographic
    blocking is cutting off the moisture flux that sustains the MD's convection
    Reference threshold: IVT < 250 kg/ms is considered below the threshold for sustained
    deep convection in tropical systems (high-end estimates for intense monsoon systems are 200-800)
    """
    track_id = (
        int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    )
    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])

    # Pressure differences for trapezoidal integration (Pa)
    levels_hpa = LEVELS
    dp = []
    for j in range(len(levels_hpa)):
        if j == 0:
            dp.append((levels_hpa[j] - levels_hpa[j + 1]) * 100 / 2)
        elif j == len(levels_hpa) - 1:
            dp.append((levels_hpa[j - 1] - levels_hpa[j]) * 100 / 2)
        else:
            dp.append((levels_hpa[j - 1] - levels_hpa[j + 1]) * 100 / 2)

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
                q = float(
                    ds["q"]
                    .squeeze()
                    .sel(latitude=tlat, longitude=tlon, method="nearest")
                    .values
                )
                u = float(
                    ds["u"]
                    .squeeze()
                    .sel(latitude=tlat, longitude=tlon, method="nearest")
                    .values
                )
                v = float(
                    ds["v"]
                    .squeeze()
                    .sel(latitude=tlat, longitude=tlon, method="nearest")
                    .values
                )
                q = max(q, 0)  # clip spectral artefact
                sum_qu += q * u * dpj
                sum_qv += q * v * dpj
            except Exception:
                valid = False
                break

        ivt = float(np.sqrt(sum_qu**2 + sum_qv**2) / G) if valid else np.nan
        ivt_records.append(
            {
                "time": t,
                "IVT": ivt,
                "IVT_u": float(abs(sum_qu) / G) if valid else np.nan,
                "IVT_v": float(abs(sum_qv) / G) if valid else np.nan,
            }
        )
    ivt_df = pd.DataFrame(ivt_records)

    # Export
    os.makedirs(output_dir, exist_ok=True)
    ivt_CSV = os.path.join(output_dir, f"track_{track_id}_ivt.csv")
    ivt_df.to_csv(ivt_CSV, index=False)
    std_print("EXPORT", "va", "IVT CSV", ivt_CSV)

    plot_ivt(ivt_df, track_df, track_id, output_dir)
    return ivt_df


def compute_vertical_mixing_ratio(track_df, pl_dir, year, output_dir):
    """
    Water vapour mixing ratio w = q / (1 − q) [g/kg] at each pressure level (850->200 hPa).

    High w throughout 850–500 hPa -> deep moisture column sustaining MD convection
    Sharp drop in 850 hPa w after crossing terrain -> orographic blocking of low-level easterly moisture inflow (Eastern Ghats / Chhota Nagpur signature)
    Characteristic timing: w drops at 850 hPa ~6–12 h after landfall over elevated regions
    """
    track_id = (
        int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    )
    profile_df = build_vertical_profile(track_df, pl_dir, year, mixing_ratio, "w")

    # Export
    os.makedirs(output_dir, exist_ok=True)
    mr_profile_CSV = os.path.join(
        output_dir, f"track_{track_id}_mixing_ratio_profile.csv"
    )
    profile_df.to_csv(mr_profile_CSV, index=False)
    std_print("EXPORT", "va", "Mixing ratio vertical profile CSV", mr_profile_CSV)

    w_cols = [f"w_{lv}" for lv in LEVELS if f"w_{lv}" in profile_df.columns]
    w_valid = profile_df[w_cols].values.ravel()
    w_valid = w_valid[~np.isnan(w_valid)]
    w_vmax = float(np.nanpercentile(w_valid, 98)) if w_valid.size > 0 else 1.0
    w_vmax = max(w_vmax, 0.1)

    plot_time_height_section(
        df=profile_df,
        col_prefix="w",
        track_id=track_id,
        cbar_label="w (g/kg)",
        cmap="YlGnBu",
        title=(
            f"Track {track_id}: Water Vapour Mixing Ratio Profile  (850–200 hPa)\n"
            "850 hPa w drop after landfall over terrain = orographic moisture blocking"
        ),
        filename=f"track_{track_id}_mixing_ratio_profile.png",
        output_dir=output_dir,
        norm=PowerNorm(gamma=0.4, vmin=0, vmax=w_vmax),
    )
    return profile_df


# Moisture Flux Convergence (MFC) at 850 hPa
def compute_moisture_flux_convergence(track_df, pl_dir, year, output_dir):
    """
    Moisture Flux Convergence at 850 hPa:
        MFC = -(1/g) x [∂(qu)/∂x + ∂(qv)/∂y] x Δp_850   [kg m⁻^2 s^(-1) -> mm hr^(-1)]

      Positive MFC -> moisture converging into the system -> convection sustained
      Negative MFC -> divergence -> moisture export -> dissipation precursor
      Collapse over terrain -> orographic wind deflection cutting off low-level inflow
    """
    BOX_DEG = 3.0
    track_id = (
        int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    )
    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])
    is_land_arr = (
        df["is_land"].values.astype(bool)
        if "is_land" in df.columns
        else np.ones(len(df), bool)
    )

    # Δp for 850 hPa layer (trapezoidal: half-distance to 700 hPa above)
    dp_850 = (LEVELS[0] - LEVELS[1]) * 100 / 2  # Pa = 7500 Pa

    mfc_records = []
    for i, t in enumerate(times):
        tlon = float(df["longitude"].iloc[i])
        tlat = float(df["latitude"].iloc[i])

        ds850 = load_level_at_time(pl_dir, t, 850, year)
        if ds850 is None or not all(v in ds850 for v in ("q", "u", "v")):
            mfc_records.append(
                {"time": t, "MFC_mm_hr": np.nan, "is_land": bool(is_land_arr[i])}
            )
            continue
        try:
            # Spatial box (ERA5 lats descending -> slice reversed)
            lat_slice = slice(tlat + BOX_DEG, tlat - BOX_DEG)
            lon_slice = slice(tlon - BOX_DEG, tlon + BOX_DEG)
            q2d = np.clip(
                ds850["q"]
                .squeeze()
                .sel(latitude=lat_slice, longitude=lon_slice)
                .values.astype(float),
                0,
                None,
            )
            u2d = (
                ds850["u"]
                .squeeze()
                .sel(latitude=lat_slice, longitude=lon_slice)
                .values.astype(float)
            )
            v2d = (
                ds850["v"]
                .squeeze()
                .sel(latitude=lat_slice, longitude=lon_slice)
                .values.astype(float)
            )
            lats = ds850["latitude"].sel(latitude=lat_slice).values
            lons = ds850["longitude"].sel(longitude=lon_slice).values

            if q2d.size == 0 or q2d.shape[0] < 2 or q2d.shape[1] < 2:
                raise ValueError("Spatial box too small")

            # Physical grid spacing (m)
            dlat_deg = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25
            dlon_deg = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.25
            lat_dir = 1.0 if lats[-1] > lats[0] else -1.0
            dy = lat_dir * dlat_deg * 111000.0  # m, uniform
            dx_2d = dlon_deg * 111000.0 * np.cos(np.radians(lats))[:, np.newaxis]

            # ∂(qu)/∂x and ∂(qv)/∂y
            dqu_dx = np.gradient(q2d * u2d, axis=1) / dx_2d
            dqv_dy = np.gradient(q2d * v2d, axis=0) / dy

            # Sample divergence at track centre
            lat_idx = np.argmin(np.abs(lats - tlat))
            lon_idx = np.argmin(np.abs(lons - tlon))
            div_ctr = float((dqu_dx + dqv_dy)[lat_idx, lon_idx])

            # MFC = -div x Δp / g  [kg m⁻^2 s^(-1)]  -> mm hr^(-1)
            mfc_mm_hr = -div_ctr * dp_850 / G * 3600.0
            mfc_records.append(
                {"time": t, "MFC_mm_hr": mfc_mm_hr, "is_land": bool(is_land_arr[i])}
            )
        except Exception as e:
            std_print("WARN", "va", f"Track {track_id}: MFC failed at {t}", str(e))
            mfc_records.append(
                {"time": t, "MFC_mm_hr": np.nan, "is_land": bool(is_land_arr[i])}
            )

    mfc_df = pd.DataFrame(mfc_records)
    mfc_df["time"] = pd.to_datetime(mfc_df["time"])

    # Export
    os.makedirs(output_dir, exist_ok=True)
    mfc_csv = os.path.join(output_dir, f"track_{track_id}_mfc_850hpa.csv")
    mfc_df.to_csv(mfc_csv, index=False)
    std_print("EXPORT", "va", "MFC 850 hPa CSV", mfc_csv)

    plot_moisture_flux_convergence(mfc_df, track_id, output_dir)
    return mfc_df


def compute_column_moisture_tendency(track_df, pl_dir, year, output_dir):
    """
    Column precipitable water W = (1/g) x Σ(q.Δp)  [kg/m^2 ≡ mm]
    Column moisture tendency ∂W/∂t  [mm/hr]

    Negative ∂W/∂t on land -> MD consuming moisture faster than replenishment -> dissipation precursor; sign-change correlates with decay onset
    Positive ∂W/∂t on land -> BOE-like recharge; land surface sustaining column moisture

    Uses the same trapezoidal Δp weights as compute_ivt (LEVELS = [850,700,600,500,400,300,200]).
    """
    track_id = (
        int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    )
    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])
    is_land_arr = (
        df["is_land"].values.astype(bool)
        if "is_land" in df.columns
        else np.ones(len(df), bool)
    )

    # Trapezoidal Δp weights (Pa) - identical to compute_ivt
    levels_hpa = LEVELS
    dp = []
    for j in range(len(levels_hpa)):
        if j == 0:
            dp.append((levels_hpa[j] - levels_hpa[j + 1]) * 100 / 2)
        elif j == len(levels_hpa) - 1:
            dp.append((levels_hpa[j - 1] - levels_hpa[j]) * 100 / 2)
        else:
            dp.append((levels_hpa[j - 1] - levels_hpa[j + 1]) * 100 / 2)

    cmt_records = []
    for i, t in enumerate(times):
        tlon = float(df["longitude"].iloc[i])
        tlat = float(df["latitude"].iloc[i])
        W = 0.0
        valid = True
        for level, dpj in zip(levels_hpa, dp):
            ds = load_level_at_time(pl_dir, t, level, year)
            if ds is None or "q" not in ds:
                valid = False
                break
            try:
                q = float(
                    ds["q"]
                    .squeeze()
                    .sel(latitude=tlat, longitude=tlon, method="nearest")
                    .values
                )
                q = max(q, 0.0)
                W += q * dpj / G
            except Exception:
                valid = False
                break
        cmt_records.append(
            {
                "time": t,
                "W_mm": W * 1000.0 if valid else np.nan,  # kg/m^2 -> mm
                "is_land": bool(is_land_arr[i]),
            }
        )

    cmt_df = pd.DataFrame(cmt_records)
    cmt_df["time"] = pd.to_datetime(cmt_df["time"])

    # ∂W/∂t [mm/hr] - np.gradient with actual cumulative-hour coordinate
    t_hours = np.array([(t - times.iloc[0]).total_seconds() / 3600.0 for t in times])
    W_vals = cmt_df["W_mm"].values
    valid_mask = ~np.isnan(W_vals)
    dWdt = np.full(len(W_vals), np.nan)
    if np.sum(valid_mask) >= 3:
        dWdt[valid_mask] = np.gradient(W_vals[valid_mask], t_hours[valid_mask])
    cmt_df["dWdt_mm_hr"] = dWdt

    os.makedirs(output_dir, exist_ok=True)
    cmt_csv = os.path.join(output_dir, f"track_{track_id}_column_moisture_tendency.csv")
    cmt_df.to_csv(cmt_csv, index=False)
    std_print("EXPORT", "va", "Column moisture tendency CSV", cmt_csv)

    plot_column_moisture_tendency(
        cmt_df=cmt_df, track_id=track_id, output_dir=output_dir
    )
    return cmt_df


def compute_theta_e_profile(track_df, pl_dir, year, output_dir):
    """
    Equivalent Potential Temperature θ_e at each pressure level (850->200 hPa).
    θ_e = T x (P0/P)^(Rd/cp) x exp(Lv x q / (cp x T))  [K]
    (simplified Bolton 1980 - uses q directly, avoids T_LCL computation)

    Uniform θ_e with height -> well-mixed moist column -> active deep convection
    θ_e decreasing sharply above 500 hPa -> stable cap forming -> weakening signal
    Upper-level θ_e (300–200 hPa) collapses before surface θ_e -> terrain-induced
    upper stabilisation (hallmark of BOE breakdown over elevated terrain
    """
    track_id = (
        int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    )
    profile_df = build_vertical_profile(track_df, pl_dir, year, theta_e, "theta_e")

    os.makedirs(output_dir, exist_ok=True)
    theta_e_CSV = os.path.join(output_dir, f"track_{track_id}_theta_e_profile.csv")
    profile_df.to_csv(theta_e_CSV, index=False)
    std_print("EXPORT", "va", "θ_e vertical profile CSV", theta_e_CSV)

    plot_time_height_section(
        df=profile_df,
        col_prefix="theta_e",
        track_id=track_id,
        cbar_label="θ_e (K)",
        cmap="RdYlBu_r",
        title=(
            f"Track {track_id}: Equivalent Potential Temperature Profile  (850–200 hPa)\n"
            "Uniform θ_e with height = well-mixed column | "
            "Upper collapse = stable cap (convection suppression)"
        ),
        output_dir=output_dir,
        filename=f"track_{track_id}_theta_e_profile.png",
    )
    return profile_df


# Omega cross-section at 700 hPa
# NOTE:
# Why 700 hPa for omega? 700 hPa is the level where orographic uplift signal is strongest over Eastern
# Ghats (~1500 m peak elevation = ~850 hPa surface, so forced ascent signal propagates to ~700 hPa).
# It's above boundary layer so it captures free-tropospheric vertical motion rather than surface-forced turbulence.
# - 500 hPa would show weaker signal because it's above most of orographic forcing.
# TODO: see for all pressure levels
def compute_omega_cross_section(track_df, pl_dir, year, output_dir):
    """
    At each land-phase time step, extract 700 hPa omega (Pa/s) along a
    perpendicular transect +/-OMEGA_TRANSECT_KM across the track bearing

    Negative omega = ascent (Pa/s convention). A spike in upward omega on
    the upslope side of terrain indicates forced orographic lifting - which
    enhances precipitation but also drains moisture faster from the system
    """
    track_id = (
        int(track_df["track_id"].iloc[0]) if "track_id" in track_df.columns else 0
    )
    df = track_df.sort_values("time").reset_index(drop=True)

    # Only land phase
    is_land = (
        df["is_land"].values.astype(bool)
        if "is_land" in df.columns
        else np.ones(len(df), bool)
    )
    land_df = df[is_land].reset_index(drop=True)
    if len(land_df) < 3:
        std_print(
            "WARN", "va", "Insufficient land-phase points for omega cross-section"
        )
        return

    times = pd.to_datetime(land_df["time"])

    # Transect distance axis (km, negative = left of track, positive = right)
    n_pts = 25
    dist_km = np.linspace(-OMEGA_TRANSECT_KM, OMEGA_TRANSECT_KM, n_pts)
    omega_matrix = np.full((len(land_df), n_pts), np.nan)

    for i in range(len(land_df)):
        t = times.iloc[i]
        c_lat = float(land_df["latitude"].iloc[i])
        c_lon = float(land_df["longitude"].iloc[i])
        bear = (
            float(land_df["bearing"].iloc[i]) if "bearing" in land_df.columns else 270.0
        )
        perp = (bear + 90) % 360  # perpendicular direction

        ds700 = load_level_at_time(pl_dir, t, 700, year)
        if ds700 is None or "w" not in ds700:
            continue
        w_slice = ds700["w"].squeeze()

        for j, d in enumerate(dist_km):
            # Project along perpendicular bearing
            dlat_rad = np.radians(d / 111.0)  # approximate
            dlon_rad = np.radians(d / (111.0 * np.cos(np.radians(c_lat))))
            p_lat = c_lat + np.cos(np.radians(perp)) * np.degrees(dlat_rad)
            p_lon = c_lon + np.sin(np.radians(perp)) * np.degrees(dlon_rad)
            try:
                val = float(
                    w_slice.sel(
                        latitude=p_lat, longitude=p_lon, method="nearest"
                    ).values
                )
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
        shading="auto",
    )
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Track centre")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("ω at 700 hPa (Pa/s)\nnegative = ascent", fontsize=FS_LABEL)
    ax.set_yticks(np.arange(len(land_df))[:: max(1, len(land_df) // 8)])
    ax.set_yticklabels(
        [
            times.iloc[k].strftime("%d/%m %H:%M")
            for k in np.arange(len(land_df))[:: max(1, len(land_df) // 8)]
        ],
        fontsize=FS_TICK,
    )
    ax.set_xlabel(
        "Distance from track centre (km)\nnegative = left, positive = right of motion",
        fontsize=FS_LABEL,
    )
    ax.set_ylabel("Time (land phase only)", fontsize=FS_LABEL)
    ax.set_title(
        f"Track {track_id}: 700 hPa Vertical Velocity Cross-Section",
        # (perpendicular to track) Asymmetric ascent on upslope side = orographic forced lifting",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax.legend(loc="upper right", **legend_style())
    plt.tight_layout()

    # Save
    ocs_path = os.path.join(output_dir, f"track_{track_id}_omega_cross_section.png")
    plt.savefig(ocs_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "Omega cross-section Hovmöller", ocs_path)

    # Export
    os.makedirs(output_dir, exist_ok=True)
    omega_col = [f"omega_d{d:.0f}km" for d in dist_km]
    omega_out = pd.DataFrame(omega_matrix, columns=omega_col, index=times)
    omega_out.index.name = "time"
    omega_CSV = os.path.join(output_dir, f"track_{track_id}_omega_cross_section.csv")
    omega_out.to_csv(omega_CSV)
    std_print("EXPORT", "va", "Omega cross-section CSV", omega_CSV)


def plot_ivt(ivt_df, track_df, track_id, output_dir):
    """IVT time series with land shading and convection threshold line"""
    times = pd.to_datetime(ivt_df["time"])
    ivt = ivt_df["IVT"].values
    if np.isnan(ivt).all():
        std_print("WARN", "va", f"Track {track_id}: No valid IVT data - skipping plot")
        return
    is_land_data = (
        track_df.sort_values("time")["is_land"].values.astype(bool)
        if "is_land" in track_df.columns
        else np.ones(len(ivt_df), bool)
    )

    # Figure
    fig, ax = plt.subplots(figsize=(16, 7))

    add_land_ocean_shading(
        ax,
        times,
        is_land_data[: len(times)]
        if len(is_land_data) >= len(times)
        else np.pad(is_land_data, (0, len(times) - len(is_land_data))),
    )
    ax.plot(
        times,
        ivt,
        "-o",
        color="teal",
        linewidth=2,
        markersize=3,
        alpha=0.8,
        label="IVT (kg/ms)",
    )
    # ax.axhline(
    #     250,
    #     color="black",
    #     linestyle="--",
    #     linewidth=1.5,
    #     label="Min. convection threshold",
    # )
    # Landfall line
    for i in range(1, len(is_land_data)):
        if not is_land_data[i - 1] and is_land_data[i]:
            ax.axvline(
                times.iloc[i],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Landfall",
                zorder=5,
            )
            break
    ax.set_ylabel("IVT (kg/ms)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_xlabel("Time", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: Integrated Vapour Transport (IVT)  850-200 hPa",
        # "IVT drop over terrain = orographic blocking of moisture supply",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", **legend_style())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save
    ivt_path = os.path.join(output_dir, f"track_{track_id}_ivt.png")
    plt.savefig(ivt_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "IVT (Integrated Vapour Transport)", ivt_path)


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

    if "speed_ms" not in track_sorted.columns:
        std_print("WARN", "va", "speed_ms missing from track_df")
        track_sorted["speed_ms"] = np.nan

    merge_cols = ["time", "speed_ms"]
    if "is_land" in track_sorted.columns:
        merge_cols.append("is_land")

    tilt_merged = pd.merge_asof(
        tilt_df[["time"]].assign(time=pd.to_datetime(tilt_df["time"])),
        track_sorted[merge_cols].drop_duplicates("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("2h"),
    )
    is_land = (
        tilt_merged["is_land"].fillna(False).values.astype(bool)
        if "is_land" in tilt_merged.columns
        else np.ones(len(tilt_df), bool)
    )
    speed = tilt_merged["speed_ms"].values.astype(float)

    # Figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    bbox_style = common_bbox()

    # Top: Tilt magnitude
    ax = axes[0]
    tilt_km = tilt_df["tilt_magnitude_km"].values
    # add_land_ocean_shading(ax, track_df['time'], track_df['is_land'])
    ax.plot(
        times,
        tilt_km,
        "-o",
        color="purple",
        linewidth=2.0,
        markersize=4,
        alpha=0.8,
        label="Tilt magnitude (km)",
    )
    # Land/ocean shading
    add_land_ocean_shading(ax, track_df["time"], track_df["is_land"])
    # Landfall line
    for i in range(1, len(is_land)):
        if not is_land[i - 1] and is_land[i]:
            ax.axvline(
                times.iloc[i],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Landfall",
                zorder=5,
            )
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
            fontsize=FS_ANNOT,
            va="top",
            bbox=bbox_style,
            family="monospace",
        )
    ax.set_ylabel("Tilt 850->200 hPa (km)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: Vortex Tilt\n(850 hPa -> 200 hPa vorticity centre displacement)",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", **legend_style({"fontsize": FS_LEGEND}))

    # Middle: Tilt bearing
    ax = axes[1]
    bearing = tilt_df["tilt_bearing_deg"].values
    ax.plot(
        times,
        bearing,
        "-o",
        color="steelblue",
        linewidth=2,
        markersize=3,
        alpha=0.8,
        label="Tilt bearing (°)",
    )
    # Reference: Hunt & Turner report WNW tilt (~285-300°) for mature MDs
    ax.axhline(
        292,
        color="blue",
        linestyle="--",
        linewidth=1,
        alpha=0.4,
        label="WNW (292°) - MD tilt direction",
    )
    # Land/ocean shading
    add_land_ocean_shading(ax, times, is_land)
    # Landfall line
    for i in range(1, len(is_land)):
        if not is_land[i - 1] and is_land[i]:
            ax.axvline(
                times.iloc[i],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Landfall",
                zorder=5,
            )
            break
    ax.set_ylim(0, 360)
    ax.set_ylabel("Tilt bearing (°)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_yticklabels(["N", "E", "S", "W", "N"])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", **legend_style({"fontsize": FS_LEGEND}))

    # Bottom: Translation speed
    ax = axes[2]
    ax.plot(
        times,
        speed,
        "-o",
        color="royalblue",
        linewidth=2,
        markersize=4,
        alpha=0.8,
        label="Translation speed (m/s)",
    )
    # Land/ocean shading
    add_land_ocean_shading(ax, times, is_land)
    # Landfall line
    for i in range(1, len(is_land)):
        if not is_land[i - 1] and is_land[i]:
            ax.axvline(
                times.iloc[i],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Landfall",
                zorder=5,
            )
            break
    ax.set_ylabel("Speed (m/s)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_xlabel("Time", fontsize=FS_LABEL, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", **legend_style({"fontsize": FS_LEGEND}))

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
    #      if "is_land" in track_df.columns else np.ones(len(therm_df), bool)
    track_sorted = track_df.sort_values("time").copy()
    track_sorted["time"] = pd.to_datetime(track_sorted["time"])
    merged = pd.merge_asof(
        therm_df[["time"]].assign(time=times),
        track_sorted[["time", "is_land"]].drop_duplicates("time")
        if "is_land" in track_sorted.columns
        else track_sorted[["time"]].assign(is_land=True),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("2h"),
    )
    is_land = merged["is_land"].fillna(False).values.astype(bool)

    # Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for ax, col, color, title, ref_line, ref_label in [
        (
            ax1,
            "T850_anom",
            "blue",
            "Cold-core anomaly at 850 hPa  (ΔT relative to 300 km ambient ring)",
            0,
            "ΔT = 0",
        ),
        (
            ax2,
            "T200_anom",
            "green",
            "Warm-core anomaly at 200 hPa  (ΔT relative to 300 km ambient ring)",
            0,
            "ΔT = 0",
        ),
    ]:
        vals = therm_df[col].values
        # Land/ocean shading
        add_land_ocean_shading(ax, track_df["time"], track_df["is_land"])
        # Landfall line
        for i in range(1, len(is_land)):
            if not is_land[i - 1] and is_land[i]:
                ax.axvline(
                    times.iloc[i],
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label="Landfall",
                    zorder=5,
                )
                break
        ax.plot(times, vals, "-o", color=color, linewidth=2, markersize=4, alpha=0.8)
        ax.fill_between(times, vals, 0, where=~np.isnan(vals), color=color, alpha=0.15)
        ax.axhline(ref_line, color="black", linewidth=0.8, label=ref_label)
        ax.set_ylabel("ΔT (K)", fontsize=FS_LABEL, fontweight="bold")
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", **legend_style({"fontsize": FS_LEGEND}))

    ax2.set_xlabel("Time", fontsize=FS_LABEL, fontweight="bold")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.suptitle(
        f"Track {track_id}: Thermal Structure",
        fontsize=FS_TITLE,
        fontweight="bold",
        y=1,
    )
    plt.tight_layout()

    # Save
    theraml_sturcture_path = os.path.join(
        output_dir, f"track_{track_id}_thermal_structure.png"
    )
    plt.savefig(theraml_sturcture_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "Thermal structure plot", theraml_sturcture_path)


def plot_vorticity_column(col_df, track_df, track_id, output_dir):
    """Two-panel: vorticity at 4 levels as line plot, and Δvo(850-200) below
    Mature MD: positive at 850+700+500 hPa, negative (anticyclonic) at 300+200 hPa
    """
    times = pd.to_datetime(col_df["time"])
    is_land = (
        track_df.sort_values("time")["is_land"].values.astype(bool)
        if "is_land" in track_df.columns
        else np.ones(len(col_df), bool)
    )

    # Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    colors = {
        "vo_850": "darkblue",
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
        "vo_200": "200 hPa",
    }
    for col, color in colors.items():
        if col in col_df.columns:
            ax1.plot(
                times,
                col_df[col].values * 1e4,
                "-o",
                color=color,
                linewidth=1.5,
                markersize=3,
                alpha=0.8,
                label=f"vo {labels[col]}",
            )
    ax1.axhline(0, color="black", linewidth=0.8)
    # Land/ocean shading
    add_land_ocean_shading(ax1, times, is_land)
    # Landfall line
    for i in range(1, len(is_land)):
        if not is_land[i - 1] and is_land[i]:
            ax1.axvline(
                times.iloc[i],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Landfall",
                zorder=5,
            )
            break
    ax1.set_ylabel("Relative vorticity (x10⁻⁴/s)", fontsize=FS_LABEL, fontweight="bold")
    ax1.set_title(
        f"Track {track_id}: Relative vorticity vertical profile",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper right", **legend_style({"fontsize": FS_LEGEND}))

    # Δvo(850-200)
    delta = col_df["delta_vo_850_200"].values * 1e4
    ax2.plot(
        times,
        delta,
        "-o",
        color="purple",
        linewidth=2,
        markersize=4,
        alpha=0.8,
        label="Δvo = vo(850) − vo(200)",
    )
    ax2.fill_between(
        times, delta, 0, where=~np.isnan(delta), color="purple", alpha=0.15
    )
    ax2.axhline(0, color="black", linewidth=0.8)
    # Land/ocean shading
    add_land_ocean_shading(ax2, times, is_land)
    # Landfall line
    for i in range(1, len(is_land)):
        if not is_land[i - 1] and is_land[i]:
            ax2.axvline(
                times.iloc[i],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Landfall",
                zorder=5,
            )
            break
    ax2.set_ylabel("Δvo (x10⁻⁴/s)", fontsize=FS_LABEL, fontweight="bold")
    ax2.set_xlabel("Time", fontsize=FS_LABEL, fontweight="bold")
    ax2.set_title(
        "Column coherence: Δvo(850−200)",
        # "Drop to zero = system losing vertical structure",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper right", **legend_style({"fontsize": FS_LEGEND}))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save
    vorticity_column_path = os.path.join(
        output_dir, f"Track_{track_id}_vorticity_column.png"
    )
    plt.savefig(vorticity_column_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "Vorticity column plot", vorticity_column_path)


def plot_moisture_flux_convergence(mfc_df, track_id, output_dir):
    """MFC timeseries with convergence/divergence fill, land/ocean shading, landfall line"""
    times = pd.to_datetime(mfc_df["time"])
    mfc = mfc_df["MFC_mm_hr"].values
    is_land = mfc_df["is_land"].values.astype(bool)

    # Find landfall index
    landfall_time = None
    for i in range(1, len(is_land)):
        if not is_land[i - 1] and is_land[i]:
            landfall_time = times.iloc[i]
            break

    # Figure
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(
        times,
        mfc,
        "-o",
        color="teal",
        linewidth=2,
        markersize=3,
        alpha=0.85,
        label="MFC (mm/hr)",
    )
    # Land/ocean shading
    add_land_ocean_shading(ax, mfc_df["time"], mfc_df["is_land"])
    # Landfall line
    for i in range(1, len(is_land)):
        if not is_land[i - 1] and is_land[i]:
            ax.axvline(
                times.iloc[i],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Landfall",
                zorder=5,
            )
            break
    ax.fill_between(
        times,
        mfc,
        0,
        where=~np.isnan(mfc) & (mfc >= 0),
        color="teal",
        alpha=0.15,
        label="Convergence (+)",
    )
    ax.fill_between(
        times,
        mfc,
        0,
        where=~np.isnan(mfc) & (mfc < 0),
        color="tomato",
        alpha=0.15,
        label="Divergence (−)",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MFC at 850 hPa  (mm/hr))", fontsize=FS_LABEL, fontweight="bold")
    ax.set_xlabel("Time", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: Moisture Flux Convergence at 850 hPa",
        # "Positive = convergence (convection sustained), Negative = divergence (dissipation)",
        fontsize=FS_TITLE,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", **legend_style({"fontsize": FS_LEGEND}))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m %H:%M"))
    plt.tight_layout()

    # Save
    mfc_path = os.path.join(output_dir, f"track_{track_id}_mfc_850hpa.png")
    plt.savefig(mfc_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "MFC 850 hPa plot", mfc_path)


def plot_column_moisture_tendency(cmt_df, track_id, output_dir):
    """Two-panel: precipitable water W (top) and ∂W/∂t (bottom)"""
    times = pd.to_datetime(cmt_df["time"])
    is_land = cmt_df["is_land"].values.astype(bool)
    W = cmt_df["W_mm"].values
    dWdt = cmt_df["dWdt_mm_hr"].values

    # Find landfall index
    landfall_time = None
    for i in range(1, len(is_land)):
        if not is_land[i - 1] and is_land[i]:
            landfall_time = times.iloc[i]
            break

    # Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    shade_kw = dict(alpha=0.3, zorder=0)

    for ax, vals, color, ylabel in [
        (ax1, W, "steelblue", "Precipitable Water W (mm)"),
        (ax2, dWdt, "tomato", "∂W/∂t  (mm/hr)"),
    ]:
        ax.plot(times, vals, "-o", color=color, linewidth=2, markersize=3, alpha=0.85)
        # Land/ocean shading
        add_land_ocean_shading(ax, cmt_df["time"], cmt_df["is_land"])
        # Landfall line
        for i in range(1, len(is_land)):
            if not is_land[i - 1] and is_land[i]:
                ax.axvline(
                    times.iloc[i],
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label="Landfall",
                    zorder=5,
                )
                break
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", **legend_style({"fontsize": FS_LEGEND}))

    ax2.set_xlabel("Time", fontsize=FS_LABEL, fontweight="bold")
    ax1.set_title(
        f"Track {track_id}: Column Moisture Budget  (850–200 hPa)",
        # "Negative ∂W/∂t on land = moisture consumed faster than replenished (dissipation signal)",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m %H:%M"))
    plt.tight_layout()

    # Save
    cmt_path = os.path.join(
        output_dir, f"track_{track_id}_column_moisture_tendency.png"
    )
    plt.savefig(cmt_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "Column moisture tendency plot", cmt_path)


def plot_time_height_section(
    df, col_prefix, track_id, cbar_label, cmap, title, filename, output_dir, norm=None
):
    """Time–height (pressure) section plot"""
    times = pd.to_datetime(df["time"])
    is_land = df["is_land"].values.astype(bool)

    # Sort pressure levels ascending for 200 hPa at top, 850 at bottom
    levels_desc = list(LEVELS)  # [850, 700, 600, 500, 400, 300, 200]
    levels_asc = sorted(levels_desc)  # [200, 300, 400, 500, 600, 700, 850]
    idx_asc = [levels_desc.index(l) for l in levels_asc]  # column mapping

    # Build matrix: (n_times x n_levels_desc) - original order
    matrix_desc = np.full((len(df), len(levels_desc)), np.nan)
    for j, lev in enumerate(levels_desc):
        col = f"{col_prefix}_{lev}"
        if col in df.columns:
            matrix_desc[:, j] = df[col].values.astype(float)

    # Reorder columns to ascending pressure: shape stays (n_times x n_levels_asc)
    matrix_asc = matrix_desc[:, idx_asc]

    # Skip plot if entirely NaN
    valid = matrix_asc[~np.isnan(matrix_asc)]
    if valid.size == 0:
        std_print("WARN", "va", f"Time-height section all-NaN - skipping: {filename}")
        return

    t_num = np.arange(len(times))
    L_asc = np.array(levels_asc, dtype=float)  # [200, 300, 400, 500, 600, 700, 850]

    # Figure
    fig, ax = plt.subplots(figsize=(16, 7))

    vmin = float(np.nanpercentile(valid, 2))
    vmax = float(np.nanpercentile(valid, 98))
    if vmin == vmax:
        vmin -= 0.1
        vmax += 0.1

    # pcolormesh: x=time, y=pressure ascending (200 at top, 850 at bottom)
    # matrix_asc.T shape = (n_levels_asc, n_times) - matches pcolormesh (Y, x)
    # im = ax.pcolormesh(t_num, L_asc, matrix_asc.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    if norm is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.pcolormesh(t_num, L_asc, matrix_asc.T, cmap=cmap, norm=norm, shading="auto")

    # Contour lines for iso-value readability
    n_contours = 8
    try:
        cs = ax.contour(
            t_num,
            L_asc,
            matrix_asc.T,
            levels=np.linspace(vmin, vmax, n_contours),
            colors="black",
            linewidths=0.5,
            alpha=0.5,
        )
        ax.clabel(cs, inline=True, fontsize=FS_LABEL, fmt="%.1f")
    except Exception:
        pass

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=FS_LABEL)

    # Detect landfall
    landfall_t_num = None
    for i in range(1, len(is_land)):
        if not is_land[i - 1] and is_land[i]:
            landfall_t_num = t_num[i]
            break

    # Landfall vertical line
    if landfall_t_num is not None:
        ax.axvline(
            landfall_t_num,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Landfall",
            zorder=6,
        )

    # x-axis: time labels
    step = max(1, len(times) // 10)
    ax.set_xticks(t_num[::step])
    ax.set_xticklabels(
        [times.iloc[k].strftime("%d/%m\n%H:%M") for k in range(0, len(times), step)],
        fontsize=FS_TICK,
    )

    # y-axis: ascending pressure (200 at top, 850 at bottom) - invert so 850 is visual bottom
    ax.set_yticks(levels_asc)
    ax.set_yticklabels([f"{lv} hPa" for lv in levels_asc])
    ax.invert_yaxis()  # now 200 hPa is visually at top, 850 at bottom - correct convention

    ax.set_xlabel("Time", fontsize=FS_LABEL, fontweight="bold")
    ax.set_ylabel("Pressure Level", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    ax.legend(loc="upper right", **legend_style({"fontsize": FS_LEGEND}))
    ax.grid(True, alpha=0.2, linestyle="--", color="white")
    plt.tight_layout()

    # Save
    time_height_path = os.path.join(output_dir, filename)
    plt.savefig(time_height_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "va", "Time-height section", time_height_path)


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


def pl_path(pl_dir, month_str, level_hpa):
    """Build path to a pressure-level file"""
    return os.path.join(pl_dir, month_str, f"{month_str}-{level_hpa}hPa.nc")


def month_str_from_time(t, year):
    """Return 'August-2024' style string from a timestamp"""
    return f"{t.strftime('%B')}-{year}"


def select_time_nearest(ds, t):
    """Select nearest time step in a dataset"""
    return ds.sel(time=t, method="nearest")


def find_vortex_centre(
    vo_slice, track_lat, track_lon, search_radius_deg=SEARCH_RADIUS_DEG
):
    """
    Find the relative vorticity maximum within `search_radius_deg` of the  single-level track centre.

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


def ambient_temperature(t_slice, centre_lat, centre_lon, ring_km=AMBIENT_RING_KM):
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


def mixing_ratio(ds, tlat, tlon, level_hpa):
    """Water vapour mixing ratio w = q/(1-q) [g/kg]"""
    q = max(
        float(
            ds["q"]
            .squeeze()
            .sel(latitude=tlat, longitude=tlon, method="nearest")
            .values
        ),
        0.0,
    )
    return (q / (1.0 - q)) * 1000.0  # g/kg


def theta_e(ds, tlat, tlon, level_hpa):
    """
    Equivalent potential temperature θ_e [K] at a pressure level.
    Bolton (1980): θ_e = T x (P0/P)^(Rd/cp) x exp(Lv x q_kg / (cp x T))
    q must be in kg/kg - reads specific humidity directly from dataset (not mixing ratio).
    """
    P_Pa = level_hpa * 100.0
    T = float(
        ds["t"].squeeze().sel(latitude=tlat, longitude=tlon, method="nearest").values
    )
    q_kg = max(
        float(
            ds["q"]
            .squeeze()
            .sel(latitude=tlat, longitude=tlon, method="nearest")
            .values
        ),
        0.0,
    )
    return T * (P0 / P_Pa) ** (Rd / cp) * np.exp(L_v * q_kg / (cp * T))


# Vertical Moisture Profile (θ_e) and mixing ratio time-height sections
def build_vertical_profile(track_df, pl_dir, year, extract_fn, col_prefix):
    """Generic column-profile extractor"""
    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])
    is_land_arr = (
        df["is_land"].values.astype(bool)
        if "is_land" in df.columns
        else np.ones(len(df), bool)
    )

    records = []
    for i, t in enumerate(times):
        tlon = float(df["longitude"].iloc[i])
        tlat = float(df["latitude"].iloc[i])
        rec = {"time": t, "is_land": bool(is_land_arr[i])}
        for level in LEVELS:
            ds = load_level_at_time(pl_dir, t, level, year)
            if ds is None:
                rec[f"{col_prefix}_{level}"] = np.nan
                continue
            try:
                rec[f"{col_prefix}_{level}"] = extract_fn(ds, tlat, tlon, level)
            except Exception:
                rec[f"{col_prefix}_{level}"] = np.nan
        records.append(rec)
    return pd.DataFrame(records)
