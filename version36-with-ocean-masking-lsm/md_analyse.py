# TODO:
# 0. Add the recycling result to global analysis
# 1. orographyconvection2005 (Kirshbaum & Durran) covers banded convection but doesn't specifically address the windward/leeward TP asymmetry.
#   - Medina & Houze (2003, Quarterly Journal) for orographic precipitation structures over barriers comparable to the Eastern Ghats
#   - Barros & Lang (2003) for the specific rain-shadow effect
# 2. Something to visualize : On the windward slope, expect depression to drop (moistening from forced ascent).
#    On the leeward side, it should spike as the dry Foehn-like descent warms and dries the descending air.
#    This T2m−D2m jump on the leeward side is a direct observable signature of orographic deception hypothesis.
# 2. Add land-surface-mask (Data/nc-files/lsm-soiltype-z.nc) for rest (used in azimuthal_hovmoller plot)
# 3. H4 GAM: pip install pygam; SM ~ s(TP) -> Plateau fig
# 4. H5 HWSD:
#    ds = xr.open_dataset('SoilGrids250m_CLAY.nc')
#    tracks['CLAY_%'] = ds.CLAY.isel(depth=0).interp(lon=tracks.lon,lat=tracks.lat)
#    # FSI ~ CLAY_% (r>0.6 expected)
# 5. H7 SRTM:
#    !pip install richdem rioxarray
#    dem = rioxarray.open_rasterio('SRTM_India.tif')
#    tracks['MEAN_ELEV'] = dem.interp(...)  # r(dur,elev)<0.7
# 6. Add duirnal HF (LHF, SHF and THF) amplitude time series plot -> \Delta(HF) = (HF_max -HF_min)_24h -> see veriation in this \Delta(HF)
# 7. Implement: analyze_asm_fsi_relationship, analyze_gam_nonlinearity, test_granger_causality
# 8. ding1994monsoon and rashid2005track: land-phase vorticity tendency from the Azimuthal Hovmöller data would tell whether the MD is spinning down due to surface friction or being maintained by convective stretching. This directly tests whether the terrain-braking is frictional or thermodynamic.
# 9. Energy rates/factors
#    - CAPE (Convective Available Potential Energy)
#    - CIN or CINE (Convective Inhibition Energy)
#    - LFC (Level of Free Convection)
#    - LCL (Lifted Condensation Level)
#    - EL (Equilibrium Level)

import os
import sys
import gc
import threading
import time
import traceback

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from utils.progress_utils import ProgressSpinner
from utils import (
    std_print,
    is_point_land,
    find_landfall_time,
    get_time_coord,
    extract_spatial_subset,
    extract_and_average_variable,
    common_bbox,
    legend_style,
)

# Analysis modules
from speed_analysis import (
    compute_track_speeds,
    plot_speed_analysis_kmh,
    plot_speed_analysis_ms,
)
from antecedent_analysis import analyze_all_tracks_antecedent
from azimuthal_hovmoller import (
    compute_azimuthal_hovmoller_sectors,
    average_azimuthally,
    compute_swvl1_tendency,
    plot_hovmoller_azimuthal_average,
)
from correlation_analysis import (
    analyze_single_track,
    compute_single_track_correlation,
    analyze_all_tracks,
)
from coupling_metrics import (
    compute_fsi_all_tracks,
    compute_diurnal_amplitude_all_tracks,
    compute_coupling_efficiency,
    export_coupling_metrics,
    generate_coupling_metrics_report,
)
from elevation_track_vis import (
    create_elevation_vrt,
    plot_tracks_elevation,
    plot_global_elevation,
)
from elevation_analysis import (
    run_elevation_analysis,
    run_track_elevation_analysis,
    plot_global_elevation_comparison,
    plot_global_elevation_speed_map,
    init_terrain,
)
from soil_moiture_analysis import (
    plot_spatial_tp_swvl1,
    compute_soil_moisture_memory,
    analyze_soil_memory_by_track,
)
from heatflux_analysis import (
    classify_md_duration,
    analyze_heat_flux_evolution,
    analyze_flux_transition,
    export_heat_flux_results,
    export_landfall_flux_results,
    print_md_classification_summary,
)
from timeseries import (
    tp_swvl1_timeseries,
    extract_heat_flux_timeseries,
    hf_timeseries,
    recycling_ratio_timeseries,
    extract_theta_e_timeseries,
    theta_e_timeseries,
)
from vortex_analysis import (
    run_track_vortex_analysis,
    global_vortex_analysis
)
from statistics_analysis import (
    analyze_asm_fsi_relationship,
    analyze_gam_nonlinearity,
    test_granger_causality,
    load_elevation_metrics,
    analyze_tsi,
    write_statistical_summary,
)

# Configuration
MONSOON_YEAR = 2020
MONSOON_SEASONS = {
    2018: ["June-2018", "July-2018", "August-2018", "September-2018"],
    2019: ["June-2019", "July-2019", "August-2019", "September-2019"],
    2020: ["June-2020", "July-2020", "August-2020", "September-2020"],
    2021: ["June-2021", "July-2021", "August-2021", "September-2021"],
    2022: ["June-2022", "July-2022", "August-2022", "September-2022"],
    2023: ["June-2023", "July-2023", "August-2023", "September-2023"],
    2024: ["June-2024", "July-2024", "August-2024", "September-2024"],
    2025: ["June-2025", "July-2025", "August-2025", "September-2025"],
}
MONTH_DIRS = MONSOON_SEASONS.get(MONSOON_YEAR, [])
DATA_DIR = os.path.join("..", "Data", "single-levels", "ncfiles", f"{MONSOON_YEAR}")
PL_DIR = os.path.join("..", "Data", "pressure-levels", "nc-files", f"{MONSOON_YEAR}")
HGT_DIR = os.path.join("..", "Data", "elevation", "hgtfiles")
LSM_PATH = os.path.join("..", "Data", "single-levels", "ncfiles", "lsm-soiltype-z.nc")

output_dir = f"output_{MONSOON_YEAR}"
DETECTION_CSV = os.path.join(output_dir, f"MD_tracks_{MONSOON_YEAR}.csv")

# Clipped domain (must match md_detect.py)
clipped_min_lat = 7.0
clipped_max_lat = 30.0
clipped_min_lon = 70.0
clipped_max_lon = 95.0

# Spatial/temporal padding for per-track ERA5 window
TRACK_LAT_PAD = 5.0  # degrees
TRACK_LON_PAD = 5.0  # degrees
TRACK_TIME_PRE_H = 48  # hours before track start
TRACK_TIME_POST_H = 6  # hours after track end

# Constants
# LHF to evaporation (mm/h)
L_v = 2.501e6   # J/kg
RHO_W = 1000.0  # kg/m³


# Load CSV
def load_csv(path):
    """Load and validate the CSV produced by md_detect.py."""
    if not os.path.exists(path):
        std_print("ERROR", "an", "CSV not found", path)
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["time"])
    required = ["time", "latitude", "longitude", "track_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        std_print("ERROR", "an", "CSV missing columns", str(missing))
        sys.exit(1)

    # Restore types
    df["track_id"] = df["track_id"].astype(int)
    if "is_land" in df.columns:
        df["is_land"] = df["is_land"].astype(bool)
    if "month" not in df.columns:
        df["month"] = df["time"].dt.strftime("%B")

    std_print(
        "INFO",
        "an",
        "CSV loaded",
        f"{df['track_id'].nunique()} tracks, {len(df)} rows",
    )
    return df


# Load ERA5
def load_era5_lazy(month_dirs, data_dir):
    """Open ERA5 as a lazy Dask-backed dataset clipped to monsoon domain."""
    file_list = []
    for month_dir in month_dirs:
        month_path = os.path.join(data_dir, month_dir)
        if os.path.isdir(month_path):
            file_list.extend(
                os.path.join(month_path, f)
                for f in os.listdir(month_path)
                if f.endswith(".nc")
            )
    if not file_list:
        std_print("ERROR", "an", "No ERA5 files found", data_dir)
        sys.exit(1)

    ds = xr.open_mfdataset(
        file_list,
        chunks={},
        combine="by_coords",
        parallel=True,
        coords="minimal",
        data_vars="minimal",
        compat="override",
        engine="h5netcdf",
    )
    ds = ds.sel(
        latitude=slice(clipped_max_lat, clipped_min_lat),
        longitude=slice(clipped_min_lon, clipped_max_lon),
    )
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    ds, _ = get_time_coord(ds)
    ds = ds.sortby("time")
    print()
    std_print("INFO", "an", "ERA5 opened (lazy)", f"{len(file_list)} files")
    return ds


# Per-track ERA5 window loader
def load_track_window(ds, track_df, track_id):
    """
    Load a small eager ERA5 slice around this track.
    All analysis functions receive track_ds - never the full lazy ds.
    """
    t_start = pd.to_datetime(track_df["time"].min()) - pd.Timedelta(hours=TRACK_TIME_PRE_H)
    t_end = pd.to_datetime(track_df["time"].max()) + pd.Timedelta(hours=TRACK_TIME_POST_H)
    lat_min = max(track_df["latitude"].min() - TRACK_LAT_PAD, float(ds.latitude.min()))
    lat_max = min(track_df["latitude"].max() + TRACK_LAT_PAD, float(ds.latitude.max()))
    lon_min = max(track_df["longitude"].min() - TRACK_LON_PAD, float(ds.longitude.min()))
    lon_max = min(track_df["longitude"].max() + TRACK_LON_PAD, float(ds.longitude.max()))

    t_start = max(t_start, pd.to_datetime(ds.time.values.min()))
    t_end = min(t_end, pd.to_datetime(ds.time.values.max()))

    track_ds = ds.sel(
        time=slice(t_start, t_end),
        latitude=slice(lat_max, lat_min),  # ERA5 descending
        longitude=slice(lon_min, lon_max),
    ).load()

    ram_mb = sum(v.nbytes for v in track_ds.data_vars.values()) / 1024**2
    std_print(
        "DEBUG",
        "an",
        f"Track {track_id}: window loaded",
        f"lat [{lat_min:.1f}–{lat_max:.1f}], lon [{lon_min:.1f}–{lon_max:.1f}],"
        f" {ram_mb:.0f} MB",
    )
    return track_ds


# Global summary report
def generate_summary_report(
    all_correlation_results,
    all_flux_results,
    all_landfall_results,
    all_classifications,
    all_sm_results,
    all_asm_results,
    all_memory_results,
    all_recycling_results,
    output_dir,
):
    report_path = os.path.join(output_dir, "Analysis_summary_report.txt")
    with open(report_path, "w") as f:
        f.write("Monsoon depression analysis - summary\n\n")

        f.write("1. Track statistics\n")
        if all_classifications:
            df_c = pd.DataFrame(all_classifications)
            long = df_c[df_c["is_long_lived"]]
            short = df_c[~df_c["is_long_lived"]]
            f.write(f"   Total MDs:          {len(df_c)}\n")
            f.write(
                f"   Short-lived (<96h): {len(short)} ({100*len(short)/len(df_c):.1f}%)\n"
            )
            f.write(
                f"   Long-lived (≥96h):  {len(long)}  ({100*len(long)/len(df_c):.1f}%)\n"
            )
            f.write(
                f"   Mean duration:      {df_c['duration_hours'].mean():.1f} ± "
                f"{df_c['duration_hours'].std():.1f} h\n\n"
            )
        else:
            f.write("   No classification data\n\n")

        f.write("2. TP–SM correlation\n")
        if all_correlation_results:
            df_r = pd.DataFrame(all_correlation_results)
            f.write(f"   Tracks: {len(df_r)}\n")
            f.write(f"   Mean r (TP vs SM): {df_r['pearson_r'].mean():.3f}\n")
            f.write(
                f"   Significant (p<0.05): {(df_r['pearson_p']<0.05).sum()}/{len(df_r)}\n\n"
            )
        else:
            f.write("   No correlation data\n\n")

        f.write("3. Heat flux\n")
        if all_flux_results:
            df_f = pd.DataFrame(all_flux_results)
            f.write(f"   Mean total HF: {df_f['mean_total_hf'].mean():.1f} W/m²\n\n")
        else:
            f.write("   No flux data\n\n")

        f.write("4. Landfall transitions\n")
        if all_landfall_results:
            bf = [r["before"]["total"] for r in all_landfall_results if r]
            af = [r["after"]["total"] for r in all_landfall_results if r]
            if bf and af:
                f.write(f"   Before: {np.nanmean(bf):.1f} W/m²\n")
                f.write(f"   After:  {np.nanmean(af):.1f} W/m²\n\n")
        else:
            f.write("   No landfall data\n\n")

        f.write("5. Soil moisture\n")
        if all_sm_results:
            df_s = (
                pd.DataFrame(all_sm_results)
                if not isinstance(all_sm_results, pd.DataFrame)
                else all_sm_results
            )
            if not df_s.empty:
                f.write(f"   Mean SWVL1: {df_s['mean_swvl1'].mean():.3f} m³/m³\n\n")
        else:
            f.write("   No SM data\n\n")

        f.write("6. SM memory (24 h ACF)\n")
        if all_memory_results:
            df_m = pd.DataFrame(all_memory_results)
            high = (df_m["memory_index_24h"] > 0.5).sum()
            f.write(f"   Strong memory (>0.5): {high}/{len(df_m)}\n\n")
        else:
            f.write("   No memory data\n\n")

        # TODO: add recycling ratio summary
        # f.write("7. Recycling ratio\n")
        # all_recycling_results

    std_print("SAVED", "an", "Summary report", report_path)
    return report_path



if __name__ == "__main__":
    t_start_all = time.time()

    # 1. Load CSV
    print("-- Loading track CSV --")
    full_track_df = load_csv(DETECTION_CSV)
    unique_track_ids = sorted(full_track_df["track_id"].unique())

    # Recompute speeds if missing (CSV may not have them)
    if "speed_kmh" not in full_track_df.columns:
        std_print("WARN", "an", "speed_kmh missing - computing from CSV coordinates")
        frames = []
        for tid in unique_track_ids:
            frames.append(compute_track_speeds(
                full_track_df[full_track_df["track_id"] == tid].copy()
            ))
        full_track_df = pd.concat(frames, ignore_index=True)

    # 2. Open ERA5 (lazy)
    print("-- Opening ERA5 (lazy) --")
    result = {"ds": None, "success": False, "error": None}

    def _load_era5():
        try:
            result["ds"]      = load_era5_lazy(MONTH_DIRS, DATA_DIR)
            result["success"] = True
        except Exception as e:
            result["error"]   = str(e)

    load_thread = threading.Thread(target=_load_era5)
    load_thread.daemon = True
    load_thread.start()
    spinner = ProgressSpinner(len(MONTH_DIRS) * 31, f"Opening ERA5 {MONSOON_YEAR}")
    step = 0
    while load_thread.is_alive():
        spinner.update(current=step)
        time.sleep(0.2)
        step += 1
    spinner.finish()

    if not result["success"]:
        std_print("ERROR", "an", "ERA5 open failed", result["error"])
        sys.exit(1)
    ds = result["ds"]

    # 3. Elevation VRT
    print("\n-- Elevation mosaic --")
    vrt_path = os.path.join(output_dir, "elevation_mosaic.vrt")
    if not os.path.exists(vrt_path):
        vrt_path = create_elevation_vrt(hgt_dir=HGT_DIR, vrt_filename=vrt_path)
    else:
        std_print("INFO", "an", "Using existing elevation VRT", vrt_path)
    init_terrain(HGT_DIR)

    # 4. Result containers
    all_correlation_results = []
    all_flux_results        = []
    all_landfall_results    = []
    all_classifications     = []
    all_sm_results          = []
    all_asm_results         = []
    all_memory_results      = []
    all_recycling_results   = []
    track_dict              = {}
    flux_dict               = {}
    vortex_tilt_results     = {}
    vortex_therm_results    = {}
    fsi_df = amplitude_df = coupling_efficiency_df = pd.DataFrame()

    # 5. Per-track loop
    for track_id in unique_track_ids:
        track_id = int(track_id)
        track_df = (full_track_df[full_track_df["track_id"] == track_id].reset_index(drop=True).copy())
        track_dict[track_id] = track_df.copy()
        track_dir = os.path.join(output_dir, f"track_{track_id}")
        os.makedirs(track_dir, exist_ok=True)

        print(f"\n\n  MD Track {track_id}  |  {len(track_df)} pts  |  "
              f"{pd.to_datetime(track_df['time'].min()).strftime('%d/%m/%Y')} – "
              f"{pd.to_datetime(track_df['time'].max()).strftime('%d/%m/%Y')}")

        # Load eager ERA5 window
        track_ds = None
        try:
            track_ds = load_track_window(ds, track_df, track_id)
        except Exception as e:
            std_print("ERROR", "an", "Window load failed, using lazy ds", str(e))
            track_ds = ds

        try:
            # 5.1 Export detections CSV
            track_csv = os.path.join(track_dir, f"track_{track_id}_detections.csv")
            track_df.to_csv(track_csv, index=False)
            std_print("EXPORT", "an", "Track detections CSV", track_csv)

            # 5.2 Speed plots
            print(f"\n-- Speed plots (track {track_id}) --")
            speed_dir = os.path.join(track_dir, "translational-speed")
            os.makedirs(speed_dir, exist_ok=True)
            # Export augmented CSV
            aug_csv = os.path.join(speed_dir, f"track_{track_id}_detections_augmented.csv")
            track_df.to_csv(aug_csv, index=False)
            std_print("EXPORT", "an", "Augmented track CSV", aug_csv)
            plot_speed_analysis_kmh(track_df, track_id, speed_dir)
            plot_speed_analysis_ms(track_df,  track_id, speed_dir)
            if "speed_kmh" in track_df.columns:
                spd = track_df["speed_kmh"].dropna()
                std_print("INFO", "an", "Speed stats",
                          f"mean {spd.mean():.1f} km/h  |  "
                          f"dur {(track_df['time'].max()-track_df['time'].min()).total_seconds()/3600:.0f} h")

            # 5.3 Correlation
            print(f"\n-- Correlation analysis (track {track_id}) --")
            correlation_result = None
            try:
                correlation_dir = os.path.join(track_dir, "correlation_analysis")
                os.makedirs(correlation_dir, exist_ok=True)
                correlation_result = analyze_single_track(
                    ds, track_df, track_id,
                    buffer_km=100, max_lag_hours=48,
                    output_dir=correlation_dir,
                )
                if correlation_result:
                    all_correlation_results.append({
                        "track_id":      track_id,
                        "pearson_r":     correlation_result.get("pearson_r"),
                        "pearson_p":     correlation_result.get("pearson_p"),
                        "mean_tp":       correlation_result.get("mean_tp"),
                        "mean_swvl1":    correlation_result.get("mean_swvl1"),
                        "n_samples":     correlation_result.get("n_samples"),
                        "landfall_time": correlation_result.get("landfall_time"),
                        "landfall_index":correlation_result.get("landfall_index"),
                        "tp_series":     correlation_result.get("tp_series"),
                        "swvl1_series":  correlation_result.get("swvl1_series"),
                        "valid_times":   correlation_result.get("valid_times"),
                        "originates_over_land": correlation_result.get("originates_over_land", True),
                    })
            except Exception as e:
                std_print("ERROR", "an", "Correlation failed", str(e))
                traceback.print_exc()

            # 5.4 Heat flux
            print(f"\n-- Heat flux analysis (track {track_id}) --")
            flux_df = None
            try:
                classification = classify_md_duration(track_df, short_threshold_hours=96)
                classification["track_id"] = track_id
                all_classifications.append(classification)
                std_print("INFO", "an", "Duration",
                          f"{classification['duration_hours']:.1f} h  "
                          f"({classification['num_points']} pts)")

                heatflux_dir = os.path.join(track_dir, "heat_flux_analysis")
                os.makedirs(heatflux_dir, exist_ok=True)
                flux_df = extract_heat_flux_timeseries(track_ds, track_df, track_id, buffer_km=100)
                if flux_df is not None:
                    flux_dict[track_id] = flux_df.copy()
                    flux_analysis = analyze_heat_flux_evolution(flux_df, track_id)
                    if flux_analysis:
                        all_flux_results.append({
                            "track_id":       track_id,
                            "is_long_lived":  classification["is_long_lived"],
                            "duration_hours": classification["duration_hours"],
                            "category":       classification["category"],
                            **flux_analysis,
                        })
                        std_print("INFO", "an", "Mean total HF", f"{flux_analysis['mean_total_hf']:.1f} W/m²")
                    # Landfall transition
                    lf_result = analyze_flux_transition(
                        track_ds, track_df, track_id, hours_before=24, hours_after=24
                    )
                    if lf_result:
                        all_landfall_results.append(lf_result)
                        lf_csv = os.path.join(heatflux_dir, f"track_{track_id}_landfall_flux.csv")
                        pd.DataFrame([lf_result]).to_csv(lf_csv, index=False)
                        std_print("EXPORT", "an", "Landfall flux CSV", lf_csv)
                else:
                    std_print("WARN", "an", "No heat flux data")
            except Exception as e:
                std_print("ERROR", "an", "Heat flux failed", str(e))
                traceback.print_exc()

            # 5.5 Time series
            print(f"\n-- Time series (track {track_id}) --")
            ts_dir = os.path.join(track_dir, "time_series")
            os.makedirs(ts_dir, exist_ok=True)
            try:
                if correlation_result and all(
                    k in correlation_result and correlation_result[k] is not None
                    for k in ("tp_series", "swvl1_series", "valid_times")
                ):
                    tp_swvl1_timeseries(
                        track_df, track_id,
                        tp=correlation_result["tp_series"],
                        swvl1=correlation_result["swvl1_series"],
                        landfall_index=correlation_result.get("landfall_index", -1),
                        output_dir=ts_dir,
                        originates_over_land=correlation_result.get("originates_over_land", True),
                    )
                else:
                    std_print("WARN", "an", "TP/SWVL1 time series skipped", "no correlation result with series data")
                if flux_df is not None:
                    hf_timeseries(flux_df, track_id, output_dir=ts_dir)
                theta_df = extract_theta_e_timeseries(
                    track_ds, track_df, track_id, buffer_km=100
                )
                theta_e_timeseries(theta_df, track_id, output_dir=ts_dir)

                rr_df = None
                if flux_df is not None and correlation_result is not None:
                    try:
                        # TP series is in mm/hr
                        tp_mmhr = np.array(correlation_result["tp_series"])
                        # LHF series in W/m² from flux_df 
                        lhf_w = flux_df["latent_heat_flux"].values

                        times_flux = pd.to_datetime(flux_df["time"])
                        times_tp = pd.to_datetime(correlation_result["valid_times"])
                        common_times = np.intersect1d(times_flux, times_tp)
                        if len(common_times) >= 5:
                            idx_flux = pd.Index(times_flux).get_indexer(common_times)
                            idx_tp = pd.Index(times_tp).get_indexer(common_times)
                            valid = (idx_flux != -1) & (idx_tp != -1)
                            idx_flux = idx_flux[valid]
                            idx_tp = idx_tp[valid]
                            if len(idx_flux) >= 5:
                                lhf_aligned = lhf_w[idx_flux]
                                tp_aligned = tp_mmhr[idx_tp]

                                E_mmh = np.abs(lhf_aligned) / (L_v * RHO_W) * 3600.0 * 1000.0
                                # Recycling ratio ρ = E / P
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    rho = np.where(tp_aligned >= 0.01, np.clip(E_mmh / tp_aligned, 0, 1), np.nan)

                                is_land_vals = []
                                for t in common_times[valid]:
                                    mask = pd.to_datetime(track_df["time"]) == t
                                    if mask.any():
                                        is_land_vals.append(bool(track_df.loc[mask, "is_land"].iloc[0]))
                                    else:
                                        is_land_vals.append(False)

                                rr_df = {
                                    "track_id": track_id,
                                    "time": common_times[valid],
                                    "rho": rho,
                                    "E_mm_h": E_mmh,
                                    "P_mm_h": tp_aligned,
                                    "is_land": is_land_vals,
                                }
                                recycling_ratio_timeseries(rr_df, track_id, output_dir=ts_dir)
                            else:
                                std_print("WARN", "an", "Insufficient aligned data for recycling ratio")
                    except Exception as e:
                        std_print("ERROR", "an", "Recycling ratio failed", str(e))
            except Exception as e:
                std_print("ERROR", "an", "Time series failed", str(e))
                traceback.print_exc()

            # 5.6 Soil moisture
            print(f"\n-- Soil moisture analysis (track {track_id}) --")
            try:
                sm_result = compute_single_track_correlation(
                    track_ds, track_df, track_id, buffer_km=100
                )
                if sm_result:
                    all_sm_results.append(sm_result)
                    sm_dir = os.path.join(track_dir, "soilmoiture_analysis")
                    os.makedirs(sm_dir, exist_ok=True)
                    plot_spatial_tp_swvl1(track_ds, track_df, track_id, output_dir=sm_dir)
                    std_print("SAVED", "an", "Soil moisture plots", sm_dir)

                    # SM memory (ACF)
                    swvl1_vals = []
                    for _, row in track_df.iterrows():
                        sub = extract_spatial_subset(
                            track_ds, row["latitude"], row["longitude"],
                            pd.to_datetime(row["time"]), buffer_km=100,
                        )
                        if sub is not None and "swvl1" in sub.data_vars:
                            v = float(sub["swvl1"].mean().values)
                            if not np.isnan(v):
                                swvl1_vals.append(v)
                    if len(swvl1_vals) > 48:
                        mem = compute_soil_moisture_memory(swvl1_vals, lag_hours=24)
                        mem_rec = {
                            "track_id":         track_id,
                            "memory_index_24h": mem,
                            "n_samples":        len(swvl1_vals),
                            "mean_swvl1":       np.mean(swvl1_vals),
                            "std_swvl1":        np.std(swvl1_vals),
                        }
                        all_memory_results.append(mem_rec)
                        mem_csv = os.path.join(sm_dir, f"track_{track_id}_soil_memory.csv")
                        pd.DataFrame([mem_rec]).to_csv(mem_csv, index=False)
                        std_print("EXPORT", "an", "Soil memory CSV", mem_csv)
                        std_print("INFO",   "an", "Memory index (24 h)", f"{mem:.3f}")
                        std_print(None, None, "Mean SWVL1", f"{np.mean(swvl1_vals):.3f} m³/m³")
                        std_print(None, None, "Samples", len(swvl1_vals))
                    else:
                        std_print("WARN", "an", "Insufficient data for SM memory analysis")
            except Exception as e:
                std_print("ERROR", "an", "Soil moisture failed", str(e))
                traceback.print_exc()

            # 5.7 Elevation
            print(f"\n-- Elevation analysis (track {track_id}) --")
            elev_dir = os.path.join(track_dir, "elevation_analysis")
            os.makedirs(elev_dir, exist_ok=True)
            try:
                if vrt_path and os.path.exists(vrt_path):
                    elev_plot_path = plot_tracks_elevation(track_df, track_id, vrt_path, output_dir)
                    if elev_plot_path:
                        std_print("SAVED", "an", "Elevation track plot", elev_plot_path)
                track_df = run_track_elevation_analysis(track_df, track_id, elev_dir)
                # Update track_dict so elevation cols are available to global analyses
                track_dict[track_id] = track_df.copy()
            except Exception as e:
                std_print("ERROR", "an", "Elevation failed", str(e))
                traceback.print_exc()

            # 5.8 Vortex
            print(f"\n-- Vortex analysis (track {track_id}) --")
            vortex_dir = os.path.join(track_dir, "vortex_analysis")
            os.makedirs(vortex_dir, exist_ok=True)
            try:
                vortex_result = run_track_vortex_analysis(
                    track_df, track_id, PL_DIR, MONSOON_YEAR, vortex_dir
                )
                if vortex_result:
                    if vortex_result.get("tilt")    is not None:
                        vortex_tilt_results[track_id]  = vortex_result["tilt"]
                    if vortex_result.get("thermal") is not None:
                        vortex_therm_results[track_id] = vortex_result["thermal"]
            except Exception as e:
                std_print("ERROR", "an", "Vortex failed", str(e))
                traceback.print_exc()
        finally:
            # Release per-track window
            if track_ds is not ds:
                del track_ds
            gc.collect()

    # 6. Global analyses
    print("\n\n-- Global analyses --")
    # 6.1 All-tracks correlation
    std_print("INFO", "an", "--- Global Coupling Metrics ---")
    correlation_df = pd.DataFrame(all_correlation_results) if all_correlation_results else pd.DataFrame()
    duration_df    = pd.DataFrame(all_classifications)     if all_classifications     else pd.DataFrame()
    coupling_metrics_dir = os.path.join(output_dir, "Global_coupling_metrics")
    os.makedirs(coupling_metrics_dir, exist_ok=True)
    flux_results_df = pd.DataFrame(all_flux_results) if all_flux_results else pd.DataFrame()

    try:
        # FSI
        if not correlation_df.empty and track_dict:
            std_print("INFO", "an", "Computing FSI")
            fsi_df = compute_fsi_all_tracks(ds, correlation_df, track_dict)
            if fsi_df is not None and not fsi_df.empty:
                fsi_csv = os.path.join(coupling_metrics_dir, "coupling_metrics_FSI.csv")
                # Merge TP/SWVL1 into FSI table if not already there
                for col in ["mean_tp", "mean_swvl1"]:
                    if col not in fsi_df.columns and col in correlation_df.columns:
                        fsi_df = fsi_df.merge(
                            correlation_df[["track_id", col]], on="track_id", how="left"
                        )
                fsi_df.to_csv(fsi_csv, index=False)
                std_print("EXPORT", "an", "FSI CSV", fsi_csv)
        else:
            std_print("WARN", "an", "Insufficient data for FSI")
            fsi_df = pd.DataFrame()

        # Diurnal amplitude
        if flux_dict:
            std_print("INFO", "an", "Computing diurnal SHF amplitude")
            amplitude_df = compute_diurnal_amplitude_all_tracks(flux_dict, track_dict)
        else:
            std_print("WARN", "an", "No flux data for diurnal amplitude")
            amplitude_df = pd.DataFrame()

        # Coupling efficiency
        if not fsi_df.empty and not amplitude_df.empty and not duration_df.empty:
            std_print("INFO", "an", "Computing coupling efficiency")
            coupling_efficiency_df = compute_coupling_efficiency(
                fsi_df, correlation_df, amplitude_df, duration_df
            )
            export_coupling_metrics(
                fsi_df, amplitude_df, coupling_efficiency_df, coupling_metrics_dir
            )
            generate_coupling_metrics_report(
                fsi_df, amplitude_df, coupling_efficiency_df, coupling_metrics_dir
            )
        else:
            std_print("WARN", "an", "Insufficient data for coupling efficiency")
    except Exception as e:
        std_print("ERROR", "an", "Coupling metrics failed", str(e))
        traceback.print_exc()

    # 6.2 Global flux export
    std_print("INFO", "an", "--- Global Flux Export ---")
    try:
        if all_flux_results:
            export_heat_flux_results(all_flux_results, output_dir)
        if all_landfall_results:
            export_landfall_flux_results(all_landfall_results, output_dir)
        else:
            std_print("WARN", "an", "No landfall flux results to export")
    except Exception as e:
        std_print("ERROR", "an", "Global flux export failed", str(e))
        traceback.print_exc()

    # 6.3 Global soil memory
    std_print("INFO", "an", "--- Global Soil Memory ---")
    try:
        if all_memory_results:
            mem_dir = os.path.join(output_dir, "Global_soil_memory")
            os.makedirs(mem_dir, exist_ok=True)
            mem_df = pd.DataFrame(all_memory_results)
            mem_all_csv = os.path.join(mem_dir, "all_tracks_soil_memory.csv")
            mem_df.to_csv(mem_all_csv, index=False)
            std_print("EXPORT", "an", "Global soil memory CSV", mem_all_csv)
            std_print(None, None, "Mean memory index (24 h)", f"{mem_df['memory_index_24h'].mean():.3f}")
        else:
            std_print("WARN", "an", "No soil memory results for global export")
    except Exception as e:
        std_print("ERROR", "an", "Global soil memory export failed", str(e))

    # 6.4 Global elevation
    std_print("INFO", "an", "--- Global Elevation ---")
    elevation_dir = os.path.join(output_dir, "Global_elevation_analysis")
    os.makedirs(elevation_dir, exist_ok=True)
    try:
        elevation_metrics = run_elevation_analysis(
            tracks_df=full_track_df,
            coupling_df=fsi_df,
            hgt_dir=HGT_DIR,
            output_dir=elevation_dir,
        )
        if elevation_metrics is not None and not elevation_metrics.empty:
            std_print("INFO", "an", "Elevation analysis done", f"{len(elevation_metrics)} tracks")
            # Takes track_dict, not full_track_df+vrt_path
            plot_global_elevation_comparison(track_dict, output_dir=elevation_dir)
            plot_global_elevation_speed_map( track_dict, output_dir=elevation_dir)
        if vrt_path and os.path.exists(vrt_path):
            plot_global_elevation(full_track_df, vrt_path, output_dir=elevation_dir, year=MONSOON_YEAR)
    except Exception as e:
        std_print("ERROR", "an", "Global elevation failed", str(e))
        traceback.print_exc()

    # 6.5 Global vortex
    std_print("INFO", "an", "--- Global Vortex ---")
    try:
        global_vortex_analysis(
            vortex_tilt_results, track_dict,
            os.path.join(output_dir, "Global_vortex_analysis"),
        )
    except Exception as e:
        std_print("ERROR", "an", "Global vortex failed", str(e))
        traceback.print_exc()

    # 6.6 Statistical analyses
    std_print("INFO", "an", "--- Statistical Analyses ---")
    try:
        fsi_csv_path = os.path.join(coupling_metrics_dir, "coupling_metrics_FSI.csv")
        asm_df = (
            pd.concat(all_asm_results, ignore_index=True)
            if all_asm_results and isinstance(all_asm_results[0], pd.DataFrame)
            else pd.DataFrame(all_asm_results) if all_asm_results
            else pd.DataFrame()
        )
        if not asm_df.empty and os.path.exists(fsi_csv_path):
            fsi_df_stats = pd.read_csv(fsi_csv_path)
            stats = {}
            try:
                stats["asm_fsi"] = analyze_asm_fsi_relationship(fsi_df_stats, asm_df, output_dir)
            except Exception as e:
                std_print("ERROR", "an", "ASM-FSI failed", str(e))
            if len(fsi_df_stats) >= 10:
                try:
                    stats["gam"] = analyze_gam_nonlinearity(fsi_df_stats, output_dir)
                except Exception as e:
                    std_print("ERROR", "an", "GAM failed", str(e))
            try:
                stats["granger"] = test_granger_causality(fsi_df_stats, output_dir)
            except Exception as e:
                std_print("ERROR", "an", "Granger failed", str(e))
            elev_metrics = load_elevation_metrics(output_dir)
            if elev_metrics is not None and not elev_metrics.empty:
                try:
                    stats["tsi"] = analyze_tsi(elev_metrics, fsi_df_stats, output_dir)
                except Exception as e:
                    std_print("ERROR", "an", "TSI failed", str(e))
            write_statistical_summary(stats, fsi_df_stats, output_dir)
        else:
            std_print("WARN", "an", "Skipping stats", "insufficient coupling or ASM data")
    except Exception as e:
        std_print("ERROR", "an", "Statistical analyses failed", str(e))
        traceback.print_exc()

    # 6.7 MD classification summary
    if all_classifications:
        print_md_classification_summary(all_classifications, output_dir)

    # 6.8 Overall summary report
    generate_summary_report(
        all_correlation_results, all_flux_results, all_landfall_results,
        all_classifications, all_sm_results,
        all_asm_results if not isinstance(all_asm_results, pd.DataFrame) else all_asm_results.to_dict("records"),
        all_memory_results, output_dir,
    )

    elapsed = time.time() - t_start_all
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    print(f"\nAnalysis finished in {h}h {m}m {s}s")
