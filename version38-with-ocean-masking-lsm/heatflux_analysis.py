# NOTE:
#  - Surface Heat Flux Analysis (Latent, Sensible, Total)
#  - Landfall Flux Transitions
#  - Flux-Precipitation-Soil Moisture Correlations
#  - Short vs Long-lived MD Classification
#  - Duration-dependent analysis of MD behavior
#  - BOE Candidate Classifier (andersen2013)

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from utils import (
    std_print,
    is_point_land,
    find_landfall_time,
    calculate_buffer_deg,
    get_time_coord,
    export_results_to_csv,
)


# Landfall flux transition
def analyze_flux_transition(ds, track_df, track_id, hours_before=24, hours_after=24):
    """
    ERA5 Heat Flux Sign Convention:
    - avg_slhtf, avg_ishf : W/m^2  (instantaneous average, positive = away from surface)
    - slhf, ishf          : J/m^2  (accumulated, ERA5 convention: negative downward)

    Conversion: W/m^2 = -dJ/m^2 / dt_seconds
    (Negative sign flips ERA5 downward-positive -> upward-positive)
    """
    track_id = int(track_id)
    ds, time_coord = get_time_coord(ds)
    track_df = track_df.sort_values("time")

    if "is_land" not in track_df.columns:
        track_df["is_land"] = track_df.apply(
            lambda row: is_point_land(row["latitude"], row["longitude"]), axis=1
        )
    landfall_time = find_landfall_time(track_df)
    if landfall_time is None:
        std_print(
            "INFO", "hf", "No landfall detected", "Using track midpoint as reference"
        )
        midpoint_idx = len(track_df) // 2
        landfall_time = pd.to_datetime(track_df.iloc[midpoint_idx]["time"])

    t_start = landfall_time - pd.Timedelta(hours=hours_before)
    t_end = landfall_time + pd.Timedelta(hours=hours_after)
    ds_times = pd.to_datetime(ds[time_coord].values)

    before = {"latent": [], "sensible": [], "total": []}
    after = {"latent": [], "sensible": [], "total": []}
    before_is_land = []
    after_is_land = []

    for _, row in track_df.iterrows():
        t = pd.to_datetime(row["time"])
        if t < t_start or t > t_end:
            continue

        lat_c = row["latitude"]
        lon_c = row["longitude"]
        is_land = row.get("is_land", False)
        lat_buf, lon_buf = calculate_buffer_deg(lat_c, buffer_km=100)

        idx = np.argmin(np.abs(ds_times - t))
        t_prev = ds_times[idx] - pd.Timedelta(hours=1)
        idx_prev = np.argmin(np.abs(ds_times - t_prev))

        try:
            lat_slice = slice(lat_c + lat_buf, lat_c - lat_buf)  # ERA5 descending
            lon_slice = slice(lon_c - lon_buf, lon_c + lon_buf)

            sn = ds.isel({time_coord: idx}).sel(latitude=lat_slice, longitude=lon_slice)
            sp = ds.isel({time_coord: idx_prev}).sel(
                latitude=lat_slice, longitude=lon_slice
            )

            if "avg_slhtf" in ds.data_vars and "avg_ishf" in ds.data_vars:
                # Instantaneous averages – already W/m^2, no de-accumulation needed
                slhf_w = float(sn["avg_slhtf"].mean().values)
                ishf_w = float(sn["avg_ishf"].mean().values)
            elif "slhf" in ds.data_vars and "ishf" in ds.data_vars:
                dt_seconds = 3600.0
                slhf_now = float(sn["slhf"].mean().values)
                slhf_prev = float(sp["slhf"].mean().values)
                ishf_now = float(sn["ishf"].mean().values)
                ishf_prev = float(sp["ishf"].mean().values)
                # FIX: Check sign convention. -(subset_now - subset_prev).
                # NOTE: ERA5 sign: negative = upward (away from surface = energy into atmosphere)
                # Positive gives upward flux (into surface = energy into land surface).
                slhf_w = (slhf_now - slhf_prev) / dt_seconds
                ishf_w = (ishf_now - ishf_prev) / dt_seconds
            else:
                std_print("WARN", "hf", f"No valid heat flux variables at {t}")
                continue

            if np.isnan(slhf_w) or np.isnan(ishf_w):
                continue

            total = slhf_w + ishf_w
            bucket = after if t >= landfall_time else before
            bucket["latent"].append(slhf_w)
            bucket["sensible"].append(ishf_w)
            bucket["total"].append(total)

            status_bucket = after_is_land if t >= landfall_time else before_is_land
            status_bucket.append(is_land)
        except Exception as e:
            std_print("ERROR", "hf", f"Error at {t}", str(e))
            continue

    has_actual_landfall = False
    if before["latent"] and after["latent"]:
        has_actual_landfall = (
            get_environment(before_is_land) == "Ocean"
            and get_environment(after_is_land) == "Land"
        )

    results = {
        "track_id": track_id,
        "landfall_time": landfall_time,
        "has_actual_landfall": has_actual_landfall,
        "before_environment": get_environment(before_is_land),
        "after_environment": get_environment(after_is_land),
        "before": {
            "latent": mean(before["latent"]),
            "sensible": mean(before["sensible"]),
            "total": mean(before["total"]),
            "n_samples": len(before["latent"]),
        },
        "after": {
            "latent": mean(after["latent"]),
            "sensible": mean(after["sensible"]),
            "total": mean(after["total"]),
            "n_samples": len(after["latent"]),
        },
        "change": {
            "latent": mean(after["latent"]) - mean(before["latent"]),
            "sensible": mean(after["sensible"]) - mean(before["sensible"]),
            "total": mean(after["total"]) - mean(before["total"]),
        },
    }

    label = (
        "Landfall flux transition"
        if has_actual_landfall
        else "Heat flux transition (no landfall)"
    )
    std_print("INFO", "hf", label)
    std_print(None, None, "Before Total HF", f"{results['before']['total']:.1f} W/m^2")
    std_print(None, None, "After  Total HF", f"{results['after']['total']:.1f} W/m^2")
    std_print(None, None, "Change Total HF", f"{results['change']['total']:.1f} W/m^2")
    return results


# Heat flux evolution (per track)
def analyze_heat_flux_evolution(flux_df, track_id):
    """Analyze heat flux statistics and ocean/land phase differences"""
    track_id = int(track_id)
    if flux_df is None or len(flux_df) < 5:
        std_print(
            "WARN",
            "hf",
            "Insufficient flux data (< 5)",
            len(flux_df) if flux_df is not None else 0,
        )
        return None

    results = {
        "track_id": track_id,
        "n_samples": len(flux_df),
        "mean_latent_hf": flux_df["latent_heat_flux"].mean(),
        "mean_sensible_hf": flux_df["sensible_heat_flux"].mean(),
        "mean_total_hf": flux_df["total_heat_flux"].mean(),
        "std_latent_hf": flux_df["latent_heat_flux"].std(),
        "std_sensible_hf": flux_df["sensible_heat_flux"].std(),
        "std_total_hf": flux_df["total_heat_flux"].std(),
    }

    if "is_land" in flux_df.columns:
        ocean_phase = flux_df[~flux_df["is_land"]]
        land_phase = flux_df[flux_df["is_land"]]
        if len(ocean_phase) > 0:
            results["ocean_mean_latent_hf"] = ocean_phase["latent_heat_flux"].mean()
            results["ocean_mean_sensible_hf"] = ocean_phase["sensible_heat_flux"].mean()
            results["ocean_mean_total_hf"] = ocean_phase["total_heat_flux"].mean()
            results["ocean_n_samples"] = len(ocean_phase)
        if len(land_phase) > 0:
            results["land_mean_latent_hf"] = land_phase["latent_heat_flux"].mean()
            results["land_mean_sensible_hf"] = land_phase["sensible_heat_flux"].mean()
            results["land_mean_total_hf"] = land_phase["total_heat_flux"].mean()
            results["land_n_samples"] = len(land_phase)
            if len(ocean_phase) > 0:
                results["latent_hf_change"] = (
                    results["land_mean_latent_hf"] - results["ocean_mean_latent_hf"]
                )
                results["sensible_hf_change"] = (
                    results["land_mean_sensible_hf"] - results["ocean_mean_sensible_hf"]
                )
                results["total_hf_change"] = (
                    results["land_mean_total_hf"] - results["ocean_mean_total_hf"]
                )
    return results


# BOE candidate classifier  (andersen2013)
def classify_boe_candidate(flux_df, lhf_threshold=70.0, min_hours=18):
    """Andersen & Shepherd (2013) BOE criterion:
    inland LHF >= 70 W/m^2 sustained for >= 18 consecutive hours.
    """
    if flux_df is None or len(flux_df) < 5:
        return {
            "is_boe_candidate": False,
            "max_consecutive_h": 0,
            "total_above_h": 0,
            "peak_lhf_land": np.nan,
            "mean_lhf_land": np.nan,
            "boe_start_time": None,
        }

    df = flux_df.copy().sort_values("time")
    if "is_land" not in df.columns:
        df["is_land"] = False

    land = df[df["is_land"]].reset_index(drop=True)
    if land.empty:
        return {
            "is_boe_candidate": False,
            "max_consecutive_h": 0,
            "total_above_h": 0,
            "peak_lhf_land": np.nan,
            "mean_lhf_land": np.nan,
            "boe_start_time": None,
        }

    above = land["latent_heat_flux"] >= lhf_threshold
    total_above = int(above.sum())
    peak_lhf = float(land["latent_heat_flux"].max())
    mean_lhf = float(land["latent_heat_flux"].mean())

    # Longest consecutive run above threshold
    max_run = 0
    boe_start = None
    run = 0
    run_start_idx = None
    for i, val in enumerate(above):
        if val:
            if run == 0:
                run_start_idx = i
            run += 1
            if run > max_run:
                max_run = run
                boe_start = pd.to_datetime(land.iloc[run_start_idx]["time"])
        else:
            run = 0

    is_boe = max_run >= min_hours

    result = {
        "is_boe_candidate": is_boe,
        "max_consecutive_h": max_run,
        "total_above_h": total_above,
        "peak_lhf_land": peak_lhf,
        "mean_lhf_land": mean_lhf,
        "boe_start_time": boe_start if is_boe else None,
    }

    label = "BOE CANDIDATE" if is_boe else "Not BOE candidate"
    std_print(
        "INFO",
        "hf",
        label,
        f"max consecutive {max_run}h ≥ {lhf_threshold} W/m^2  "
        f"(need ≥ {min_hours}h)",
    )
    return result


def compute_flux_tp_correlation(flux_df, track_df):
    if flux_df is None or len(flux_df) < 5:
        std_print(
            "WARN",
            "hf",
            "Insufficient flux data for correlation (< 5)",
            len(flux_df) if flux_df is not None else 0,
        )
        return None

    flux_times = pd.to_datetime(flux_df["time"])
    track_times = pd.to_datetime(track_df["time"])
    common_times = np.intersect1d(flux_times, track_times)
    if len(common_times) < 3:
        std_print(
            "WARN",
            "hf",
            "Insufficient common times for flux-tp correlation (< 3)",
            len(common_times),
        )
        return None

    flux_aligned = flux_df[flux_df["time"].isin(common_times)]["total_heat_flux"].values
    track_aligned = track_df[track_df["time"].isin(common_times)]
    if "tp" in track_aligned.columns:
        tp_vals = track_aligned["tp"].values * 1000  # m/hr -> mm/hr
        if len(flux_aligned) == len(tp_vals):
            try:
                r, p = pearsonr(flux_aligned, tp_vals)
                return {"r": r, "p": p, "n": len(flux_aligned)}
            except Exception as e:
                std_print("WARN", "hf", "Pearson not computed", str(e))
    return None


def export_heat_flux_results(all_flux_results, output_dir="output"):
    if not all_flux_results:
        std_print("WARN", "hf", "No flux results to export")
        return
    results_df = pd.DataFrame(all_flux_results)
    export_results_to_csv(results_df, "hf", output_dir, "heat_flux_analysis_summary")


def export_landfall_flux_results(all_landfall_results, output_dir):
    if not all_landfall_results:
        std_print("WARN", "hf", "No landfall flux results to export")
        return

    results_list = []
    seen_tracks = set()
    for result in all_landfall_results:
        if result is None:
            continue
        track_id = result["track_id"]
        if track_id in seen_tracks:
            continue
        seen_tracks.add(track_id)
        results_list.append(
            {
                "track_id": track_id,
                "reference_time": result["landfall_time"],
                "has_actual_landfall": result["has_actual_landfall"],
                "before_environment": result.get("before_environment", "Unknown"),
                "after_environment": result.get("after_environment", "Unknown"),
                "before_latent_hf": result["before"]["latent"],
                "before_sensible_hf": result["before"]["sensible"],
                "before_total_hf": result["before"]["total"],
                "before_n_samples": result["before"]["n_samples"],
                "after_latent_hf": result["after"]["latent"],
                "after_sensible_hf": result["after"]["sensible"],
                "after_total_hf": result["after"]["total"],
                "after_n_samples": result["after"]["n_samples"],
                "change_latent_hf": result["change"]["latent"],
                "change_sensible_hf": result["change"]["sensible"],
                "change_total_hf": result["change"]["total"],
            }
        )

    if results_list:
        results_df = pd.DataFrame(results_list)
        export_results_to_csv(results_df, "hf", output_dir, "heat_flux_transitions")
        std_print(None, None, "Using landfall", results_df["has_actual_landfall"].sum())
        std_print(
            None, None, "Using midpoint", (~results_df["has_actual_landfall"]).sum()
        )


def print_md_classification_summary(classifications, output_dir="output"):
    df = pd.DataFrame(classifications)
    short_mds = df[~df["is_long_lived"]]
    long_mds = df[df["is_long_lived"]]
    std_print("INFO", "hf", "Total MDs", len(df))
    std_print(
        None,
        None,
        "Short-lived (<96h)",
        f"{len(short_mds)} ({100*len(short_mds)/len(df):.1f}%)",
    )
    std_print(
        None,
        None,
        "Long-lived (≥96h)",
        f"{len(long_mds)} ({100*len(long_mds)/len(df):.1f}%)",
    )
    if len(short_mds) > 0:
        std_print(
            None,
            None,
            "Short-lived mean duration",
            f"{short_mds['duration_hours'].mean():.1f} ± {short_mds['duration_hours'].std():.1f} h",
        )
    if len(long_mds) > 0:
        std_print(
            None,
            None,
            "Long-lived mean duration",
            f"{long_mds['duration_hours'].mean():.1f} ± {long_mds['duration_hours'].std():.1f} h",
        )

    classification_summary_path = os.path.join(output_dir, "classification_summary.txt")
    std_print("SAVED", "hf", "Classification summary", classification_summary_path)


# Helper functions
def deaccumulate(ds, var, t_now, t_prev):
    """Convert accumulated J m-2 to W m-2.  ERA5 sign: negative downward."""
    dt = (pd.to_datetime(t_now) - pd.to_datetime(t_prev)).total_seconds()
    if dt <= 0:
        return np.nan
    v_now = float(ds[var].sel(time=t_now).values)
    v_prev = float(ds[var].sel(time=t_prev).values)
    return -(v_now - v_prev) / dt


def mean(lst):
    return float(np.nanmean(lst)) if lst else np.nan


def get_environment(is_land_list):
    if not is_land_list:
        return "Unknown"
    land_count = sum(is_land_list)
    ocean_count = len(is_land_list) - land_count
    if land_count > ocean_count:
        return "Land"
    if ocean_count > land_count:
        return "Ocean"
    if land_count == ocean_count == 0:
        return "Unknown"
    return "Mixed"
