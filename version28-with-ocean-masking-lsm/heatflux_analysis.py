"""
1. Surface Heat Flux Analysis (Latent, Sensible, Total)
2. Landfall Flux Transitions
3. Flux-Precipitation-Soil Moisture Correlations
4. Short vs Long-lived MD Classification
5. Duration-dependent analysis of MD behavior
"""

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


def analyze_flux_transition(ds, track_df, track_id, hours_before=24, hours_after=24):
    """
    ERA5 Heat Flux Sign Convention:
    - avg_slhtf, avg_ishf: W/m² (instantaneous, positive = away from surface)
    - slhf, ishf: J/m² (accumulated, ERA5 convention: negative downward)

    Conversion: W/m² = -dJ/m² / dt_seconds
    (Negative sign flips downward -> upward convention)
    """
    track_id = int(track_id)
    ds, time_coord = get_time_coord(ds)
    track_df = track_df.sort_values("time")
    # Find first landfall
    # Ensure is_land column exists
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

    # Time windows
    t_start = landfall_time - pd.Timedelta(hours=hours_before)
    t_end = landfall_time + pd.Timedelta(hours=hours_after)
    ds_times = pd.to_datetime(ds[time_coord].values)

    # Containers
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
        lat_buf, lon_buf = calculate_buffer_deg(row["latitude"], buffer_km=100)

        # Nearest ERA5 time
        idx = np.argmin(np.abs(ds_times - t))
        t_nearest = ds_times[idx]
        t_prev = t_nearest - pd.Timedelta(hours=1)
        idx_prev = np.argmin(np.abs(ds_times - t_prev))
        try:
            lat_slice = slice(lat_c + lat_buf, lat_c - lat_buf)
            lon_slice = slice(lon_c - lon_buf, lon_c + lon_buf)
            subset_now = ds.isel({time_coord: idx}).sel(
                latitude=lat_slice, longitude=lon_slice
            )
            subset_prev = ds.isel({time_coord: idx_prev}).sel(
                latitude=lat_slice, longitude=lon_slice
            )
            # Check which flux variables are available
            if "avg_slhtf" in ds.data_vars and "avg_ishf" in ds.data_vars:
                # Use average fluxes directly (already in W/m²)
                slhf_w = float(subset_now["avg_slhtf"].mean().values)
                ishf_w = float(subset_now["avg_ishf"].mean().values)
                # std_print('DEBUG', 'hf', 'Using avg_slhtf and avg_ishf (Accumulated)')
            elif "slhf" in ds.data_vars and "ishf" in ds.data_vars:
                # De-accumulate accumulated fluxes  J m-2  ->  W m-2  (ERA5 negative downward)
                dt_seconds = 3600.0
                subset_now = float(subset_now["slhf"].mean().values)
                subset_prev = float(subset_prev["slhf"].mean().values)
                slhf_w = -(subset_now - subset_prev) / dt_seconds
                ishf_w = -(subset_now - subset_prev) / dt_seconds
                # std_print('DEBUG', 'hf', 'Using avg_slhtf and avg_ishf (De-accumulated)')
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

            # Track land/ocean status
            status_bucket = after_is_land if t >= landfall_time else before_is_land
            status_bucket.append(is_land)
        except Exception as e:
            std_print("ERROR", "hf", f"Error at {t}", str(e))
            continue

    # Determine if this was an actual landfall transition
    has_actual_landfall = False
    if len(before["latent"]) > 0 and len(after["latent"]) > 0:
        # Check if we transitioned from ocean to land
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

    if has_actual_landfall:
        std_print("INFO", "hf", "Landfall flux transition")
    else:
        std_print("INFO", "hf", "Heat flux transition", "No landfall")
    std_print(None, None, "Before Total HF", f"{results['before']['total']:.1f} W/m²")
    std_print(None, None, "After Total HF", f"{results['after']['total']:.1f} W/m²")
    std_print(None, None, "Total HF Change", f"{results['change']['total']:.1f} W/m²")
    return results


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

    # Ocean vs Land phase analysis
    if "is_land" in flux_df.columns:
        # FIX:  Equalilty comparison
        ocean_phase = flux_df[flux_df["is_land"] == False]
        land_phase = flux_df[flux_df["is_land"] == True]
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
            # Calculate changes only if both phases exist
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


def compute_flux_tp_correlation(flux_df, track_df):
    if flux_df is None or len(flux_df) < 5:
        std_print(
            "WARN",
            "hf",
            "Insufficient flux data for correlation (< 5)",
            len(flux_df),
            # f"{len(flux_df) if flux_df is not None else 0}",
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
        tp_vals = track_aligned["tp"].values * 1000  # Convert to mm/hr
        if len(flux_aligned) == len(tp_vals):
            try:
                r, p = pearsonr(flux_aligned, tp_vals)
                return {"r": r, "p": p, "n": len(flux_aligned)}
            except Exception as e:
                std_print("WARN", "hf", "Pearson is not computed, set to None", str(e))
                return None
    return None


def classify_md_duration(track_df, short_threshold_hours=96):  # 96 hours = 4 days
    track_times = pd.to_datetime(track_df["time"])
    duration_hours = (track_times.max() - track_times.min()).total_seconds() / 3600
    classification = {
        "duration_hours": duration_hours,
        "is_long_lived": duration_hours >= short_threshold_hours,
        "category": (
            "Long-lived MD" if duration_hours >= short_threshold_hours else "Short MD"
        ),
        "start_time": track_times.min(),
        "end_time": track_times.max(),
        "num_points": len(track_df),
    }
    return classification


def export_landfall_flux_results(all_landfall_results, output_dir):
    """Export landfall flux transition results to CSV"""
    if not all_landfall_results:
        std_print("WARN", "hf", "No landfall flux results to export")
        return

    results_list = []
    seen_tracks = set()
    for result in all_landfall_results:
        if result is not None:
            track_id = result["track_id"]
            # Skip duplicate results for same track
            if track_id in seen_tracks:
                continue
            seen_tracks.add(track_id)
            results_list.append(
                {
                    "track_id": result["track_id"],
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
        std_print(None, None, "Using Landfall", results_df["has_actual_landfall"].sum())
        std_print(
            None, None, "Using midpoint", (~results_df["has_actual_landfall"]).sum()
        )


def print_md_classification_summary(classifications, output_dir="output"):
    df = pd.DataFrame(classifications)
    short_mds = df[~df["is_long_lived"]]
    long_mds = df[df["is_long_lived"]]
    std_print("INFO", "hf", "Total MDs detected tracks", len(df))
    std_print(
        None,
        None,
        "Short-lived MDs (< 96h)",
        f"{len(short_mds)} ({100 * len(short_mds) / len(df):.1f}%)",
    )
    std_print(
        None,
        None,
        "Long-lived MDs (≥ 96h)",
        f"{len(long_mds)} ({100 * len(long_mds) / len(df):.1f}%)",
    )
    if len(short_mds) > 0:
        std_print(
            None,
            None,
            "Short-lived MD mean duration",
            f"{short_mds['duration_hours'].mean():.1f} ± {short_mds['duration_hours'].std():.1f} hours",
        )
        std_print(
            None,
            None,
            "Short-lived MD range",
            f"{short_mds['duration_hours'].min():.1f} - {short_mds['duration_hours'].max():.1f} hours",
        )
    if len(long_mds) > 0:
        std_print(
            None,
            None,
            "Long-lived MD mean duration",
            f"{long_mds['duration_hours'].mean():.1f} ± {long_mds['duration_hours'].std():.1f} hours",
        )
        std_print(
            None,
            None,
            "Long-lived MD range",
            f"{long_mds['duration_hours'].min():.1f} - {long_mds['duration_hours'].max():.1f} hours",
        )

    classification_summary_path = os.path.join(output_dir, "classification_summary.txt")
    std_print("SAVED", "hf", "Classification Summary", classification_summary_path)


# TODO: trenberth2009earth, bowen ration \beta = SHF / LHF
# plot_bowen_ratio_evolution(flux_df, track_df, track_id, output_dir):

# TODO: andersen2013 : defines BOE as inland LHF >= 70 W/m² sustained for >= 18 h.
# classify_boe_candidate(flux_df, lhf_threshold=70, min_hours=18)


# Helper functions
def deaccumulate(ds, var, t_now, t_prev):
    """Convert accumulated J m-2 to W m-2."""
    dt = (pd.to_datetime(t_now) - pd.to_datetime(t_prev)).total_seconds()
    if dt <= 0:
        return np.nan
    # ERA5 sign convention: negative downward -> positive upward
    subset_now = float(ds[var].sel(time=t_now).values)
    subset_prev = float(ds[var].sel(time=t_prev).values)
    return -(subset_now - subset_prev) / dt


def mean(lst):
    return np.nanmean(lst) if lst else np.nan


def get_environment(is_land_list):
    """Determine dominant environment (ocean or land)"""
    if not is_land_list:
        return "Unknown"
    land_count = sum(is_land_list)
    ocean_count = len(is_land_list) - land_count
    if land_count == 0 and ocean_count == 0:
        return "Unknown"
    if land_count > ocean_count:
        return "Land"
    elif ocean_count > land_count:
        return "Ocean"
    else:
        return "Mixed"  # Equal land and ocean points


def export_heat_flux_results(all_flux_results, output_dir="output"):
    if not all_flux_results:
        std_print("WARN", "hf", "No flux results to export")
        return
    results_df = pd.DataFrame(all_flux_results)
    export_results_to_csv(results_df, "hf", output_dir, "heat_flux_analysis_summary")
