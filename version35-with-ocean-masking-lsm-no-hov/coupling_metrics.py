"""
1. Feedback Strength Index (FSI): Integrates TP-SWVL1 correlation with soil capacity
2. Diurnal Sensible Heat Flux Amplitude: Quantifies land surface diurnal heating
3. Coupling Efficiency: Relates precipitation-to-moisture feedback during passage
"""

# FIX: compling matrix summary report csv end before it's completion, summary CSV is being truncated, statistical analysis and statistics_analysis.py functions are operating on incomplete data. Likely df.to_csv() call inside a loop that exits on exception. Add try/except with an explicit finally: df.to_csv(...) pattern.
# TODO: added validate_coupling_data and validate_analysis_inputs, but not implemented. GAM analysis in statistics_analysis.py fails silently if coupling data has NaN-filled rows. The validation stub should at minimum check for null columns and minimum row count.

import os
import numpy as np
import pandas as pd

from utils import (
    std_print,
    extract_spatial_subset,
    get_time_coord,
)


# Feedback Strength Index (FSI)
def compute_soil_water_capacity_index(swvl1_values):
    # Typical SWVL1 range: 0.02 (dry) to 0.55 (wet)
    SWVL1_MIN = 0.02
    SWVL1_MAX = 0.55
    swvl1_clean = swvl1_values[~np.isnan(swvl1_values)]
    if len(swvl1_clean) == 0:
        return np.nan
    mean_swvl1 = np.mean(swvl1_clean)
    # Normalize
    capacity_index = (mean_swvl1 - SWVL1_MIN) / (SWVL1_MAX - SWVL1_MIN)
    capacity_index = np.clip(capacity_index, 0, 1)
    return capacity_index


def compute_feedback_strength_index(
    correlation_value, swvl1_values, pearson_p, significance_threshold=0.05
):
    """Feedback Strength Index (FSI) = |correlation(TP, SWVL1)| × Soil Water Capacity Index x Significance Factor
    - Correlation strength: How tightly linked is precipitation to soil moisture?
    - Capacity: How much can the soil respond (saturation potential)?
    - Significance: Is this feedback statistically robust?
    """
    if np.isnan(correlation_value):
        return {
            "FSI": np.nan,
            "correlation_strength": np.nan,
            "capacity_index": np.nan,
            "significance_factor": 0.0,
            "is_significant": False,
            "pearson_p": pearson_p,
        }
    # 1 Correlation strength
    corr_strength = np.abs(correlation_value)
    # 2 Soil water capacity
    capacity_idx = compute_soil_water_capacity_index(swvl1_values)
    # 3 Significance factor
    if pearson_p is not None:
        is_sig = pearson_p < significance_threshold
        sig_factor = 1.0 if is_sig else 0.5  # Reduce weight if not significant
    else:
        sig_factor = 1.0
        is_sig = True

    # Compute FSI
    fsi = corr_strength * capacity_idx * sig_factor
    return {
        "FSI": fsi,
        "correlation_strength": corr_strength,
        "capacity_index": capacity_idx,
        "significance_factor": sig_factor,
        "is_significant": is_sig,
        "pearson_p": pearson_p,
    }


def compute_fsi_all_tracks(ds, correlation_results_df, track_df_dict):
    """Compute FSI for all tracks"""
    fsi_df = []
    for _, row in correlation_results_df.iterrows():
        track_id = int(row["track_id"])
        pearson_r = row.get("pearson_r", np.nan)
        pearson_p = row.get("pearson_p", np.nan)

        if track_id not in track_df_dict:
            continue

        track_df = track_df_dict[track_id]
        swvl1_vals = []
        ds, time_coord = get_time_coord(ds)

        # Iterate the *track* DataFrame to extract SWVL1 at each track point
        for _, pt in track_df.iterrows():
            try:
                subset = extract_spatial_subset(
                    ds,
                    pt["latitude"],
                    pt["longitude"],
                    pd.to_datetime(pt["time"]),
                    buffer_km=100,
                )
                if subset is not None and "swvl1" in subset.data_vars:
                    sm = float(subset["swvl1"].mean().values)
                    if not np.isnan(sm):
                        swvl1_vals.append(sm)
            except Exception:
                continue

        if len(swvl1_vals) > 0:
            swvl1_array = np.array(swvl1_vals)
            fsi_dict = compute_feedback_strength_index(
                pearson_r, swvl1_array, pearson_p
            )
            fsi_df.append(
                {
                    "track_id": track_id,
                    "FSI": fsi_dict["FSI"],
                    "correlation_strength": fsi_dict["correlation_strength"],
                    "capacity_index": fsi_dict["capacity_index"],
                    "significance_factor": fsi_dict["significance_factor"],
                    "is_significant": fsi_dict["is_significant"],
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "mean_swvl1": float(np.mean(swvl1_array)),
                    "std_swvl1": float(np.std(swvl1_array)),
                }
            )
    return pd.DataFrame(fsi_df)


# 2: Diurnal heat flux amplitude
def compute_diurnal_shf_amplitude(flux_df, land_only=True):
    if flux_df is None or len(flux_df) < 4:
        return {
            "diurnal_amplitude": np.nan,
            "max_shf": np.nan,
            "min_shf": np.nan,
            "mean_shf": np.nan,
            "shf_range": np.nan,
            "n_valid_points": 0,
            "analysis_period": None,
        }

    flux_df = flux_df.copy()
    flux_df["time"] = pd.to_datetime(flux_df["time"])
    # Filter to land if requested
    if land_only and "is_land" in flux_df.columns:
        flux_df = flux_df[flux_df["is_land"] == True]
    if len(flux_df) < 4:
        return {
            "diurnal_amplitude": np.nan,
            "max_shf": np.nan,
            "min_shf": np.nan,
            "mean_shf": np.nan,
            "shf_range": np.nan,
            "n_valid_points": len(flux_df),
            "analysis_period": None,
        }

    shf_vals = flux_df["sensible_heat_flux"].values
    shf_clean = shf_vals[~np.isnan(shf_vals)]
    if len(shf_clean) < 2:
        return {
            "diurnal_amplitude": np.nan,
            "max_shf": np.nan,
            "min_shf": np.nan,
            "mean_shf": np.nan,
            "shf_range": np.nan,
            "n_valid_points": len(shf_clean),
            "analysis_period": None,
        }

    amp = np.max(shf_clean) - np.min(shf_clean)
    return {
        "diurnal_amplitude": amp,
        "max_shf": float(np.max(shf_clean)),
        "min_shf": float(np.min(shf_clean)),
        "mean_shf": float(np.mean(shf_clean)),
        "shf_range": (float(np.min(shf_clean)), float(np.max(shf_clean))),
        "n_valid_points": len(shf_clean),
        "analysis_period": (flux_df["time"].min(), flux_df["time"].max()),
    }


def compute_diurnal_amplitude_by_phase(flux_df, track_df):
    if flux_df is None or flux_df.empty:
        return {
            "ocean_phase": {"amplitude": np.nan, "n_points": 0},
            "land_phase": {"amplitude": np.nan, "n_points": 0},
        }

    flux_df = flux_df.copy()
    flux_df["time"] = pd.to_datetime(flux_df["time"])

    # Detect landfall
    landfall_time = None
    if "is_land" in track_df.columns:
        land_rows = track_df[track_df["is_land"] == True]
        if not land_rows.empty:
            landfall_time = pd.to_datetime(land_rows.iloc[0]["time"])
    results = {
        "ocean_phase": {
            "amplitude": np.nan,
            "max_shf": np.nan,
            "min_shf": np.nan,
            "mean_shf": np.nan,
            "n_points": 0,
        },
        "land_phase": {
            "amplitude": np.nan,
            "max_shf": np.nan,
            "min_shf": np.nan,
            "mean_shf": np.nan,
            "n_points": 0,
        },
    }

    # Ocean phase: before landfall
    if landfall_time is not None:
        ocean_flux = flux_df[flux_df["time"] < landfall_time]
        if len(ocean_flux) >= 2:
            shf_ocean = ocean_flux["sensible_heat_flux"].dropna().values
            if len(shf_ocean) >= 2:
                results["ocean_phase"]["amplitude"] = np.max(shf_ocean) - np.min(
                    shf_ocean
                )
                results["ocean_phase"]["max_shf"] = float(np.max(shf_ocean))
                results["ocean_phase"]["min_shf"] = float(np.min(shf_ocean))
                results["ocean_phase"]["mean_shf"] = float(np.mean(shf_ocean))
                results["ocean_phase"]["n_points"] = len(shf_ocean)

        # Land phase: after landfall
        land_flux = flux_df[flux_df["time"] >= landfall_time]
        if len(land_flux) >= 2:
            shf_land = land_flux["sensible_heat_flux"].dropna().values
            if len(shf_land) >= 2:
                results["land_phase"]["amplitude"] = np.max(shf_land) - np.min(shf_land)
                results["land_phase"]["max_shf"] = float(np.max(shf_land))
                results["land_phase"]["min_shf"] = float(np.min(shf_land))
                results["land_phase"]["mean_shf"] = float(np.mean(shf_land))
                results["land_phase"]["n_points"] = len(shf_land)
    else:
        # No landfall, compute for all data
        shf_all = flux_df["sensible_heat_flux"].dropna().values
        if len(shf_all) >= 2:
            results["ocean_phase"]["amplitude"] = np.max(shf_all) - np.min(shf_all)
            results["ocean_phase"]["max_shf"] = float(np.max(shf_all))
            results["ocean_phase"]["min_shf"] = float(np.min(shf_all))
            results["ocean_phase"]["mean_shf"] = float(np.mean(shf_all))
            results["ocean_phase"]["n_points"] = len(shf_all)
    return results


def compute_diurnal_amplitude_all_tracks(flux_results_dict, track_df_dict):
    amplitude_results = []
    for track_id, flux_df in flux_results_dict.items():
        if flux_df is None or flux_df.empty:
            continue

        track_df = track_df_dict.get(track_id)
        if track_df is None:
            continue
        # Overall amplitude (land-only)
        overall = compute_diurnal_shf_amplitude(flux_df, land_only=True)
        # Phase-specific amplitudes
        by_phase = compute_diurnal_amplitude_by_phase(flux_df, track_df)
        amplitude_results.append(
            {
                "track_id": track_id,
                "diurnal_amplitude_land": overall["diurnal_amplitude"],
                "max_shf_land": overall["max_shf"],
                "min_shf_land": overall["min_shf"],
                "mean_shf_land": overall["mean_shf"],
                "n_points_land": overall["n_valid_points"],
                "ocean_amplitude": by_phase["ocean_phase"]["amplitude"],
                "ocean_max_shf": by_phase["ocean_phase"]["max_shf"],
                "ocean_mean_shf": by_phase["ocean_phase"]["mean_shf"],
                "ocean_n_points": by_phase["ocean_phase"]["n_points"],
                "land_amplitude": by_phase["land_phase"]["amplitude"],
                "land_max_shf": by_phase["land_phase"]["max_shf"],
                "land_mean_shf": by_phase["land_phase"]["mean_shf"],
                "land_n_points": by_phase["land_phase"]["n_points"],
                "amplitude_change": (
                    (
                        by_phase["land_phase"]["amplitude"]
                        - by_phase["ocean_phase"]["amplitude"]
                    )
                    if (
                        not np.isnan(by_phase["land_phase"]["amplitude"])
                        and not np.isnan(by_phase["ocean_phase"]["amplitude"])
                    )
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(amplitude_results)


def compute_coupling_efficiency(fsi_df, correlation_df, amplitude_df, duration_df):
    """
    Coupling Efficiency = (FSI x Diurnal Amplitude) / (1 + Duration_Decay_Factor)
    - Stronger FSI & larger diurnal amplitude = tighter coupling
    - Longer-lived systems may show weaker coupling (system adjustment)
    """
    merged = fsi_df.merge(correlation_df, on="track_id", how="outer")
    merged = merged.merge(amplitude_df, on="track_id", how="outer")
    merged = merged.merge(duration_df, on="track_id", how="outer")
    merged["coupling_efficiency"] = merged.apply(compute_ce, axis=1)
    return merged


def validate_coupling_data(df):
    """Validate coupling data for analysis"""
    issues = []
    if df is None or df.empty:
        issues.append("Data is None or empty")
        return False, issues
    if df.empty or len(df) < 3:
        return False, "Insufficient data"
    missing_pct = df.isnull().mean() * 100
    if any(missing_pct > 20):
        return False, "Too many missing values"
    required_columns = ["track_id", "FSI", "correlation_strength"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    # Check for NaN values
    nan_counts = df[required_columns].isna().sum()
    if nan_counts.any():
        issues.append(f"NaN values: {nan_counts.to_dict()}")
    # Check data ranges
    if "FSI" in df.columns:
        fsi_range = (df["FSI"].min(), df["FSI"].max())
        if fsi_range[0] < 0 or fsi_range[1] > 1:
            issues.append(f"FSI out of expected range [0,1]: {fsi_range}")
    if "correlation_strength" in df.columns:
        corr_range = (
            df["correlation_strength"].min(),
            df["correlation_strength"].max(),
        )
        if corr_range[0] < -1 or corr_range[1] > 1:
            issues.append(f"Correlation out of range [-1,1]: {corr_range}")
    return len(issues) == 0, "; ".join(issues) if issues else "Valid"


def validate_analysis_inputs(track_df):
    """Validate all inputs for analysis"""
    validations = {}
    if len(track_df) < 24:  # Min for diurnal
        return False, "Track too short"
    for name, data in track_df.items():
        if isinstance(data, pd.DataFrame):
            validations[name] = {
                "not_empty": not data.empty,
                "shape": data.shape,
                "columns": list(data.columns),
                "nan_counts": data.isna().sum().to_dict() if not data.empty else {},
            }
        elif isinstance(data, list):
            validations[name] = {
                "length": len(data),
                "type": type(data[0]) if data else "empty",
            }
        elif data is None:
            validations[name] = {"status": "None"}
    return True, "valid", validations


def export_coupling_metrics(fsi_df, amplitude_df, coupling_efficiency_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # FSI Results
    fsi_path = os.path.join(output_dir, "coupling_metrics_FSI.csv")
    fsi_df.to_csv(fsi_path, index=False)
    std_print("SAVED", "cm", "FSI metrics", fsi_path)

    # Diurnal Amplitude
    amp_path = os.path.join(output_dir, "coupling_metrics_diurnal_amplitude.csv")
    amplitude_df.to_csv(amp_path, index=False)
    std_print("SAVED", "cm", "Diurnal amplitude metrics", amp_path)

    # Coupling Efficiency
    ce_path = os.path.join(output_dir, "coupling_metrics_efficiency.csv")
    coupling_efficiency_df.to_csv(ce_path, index=False)
    std_print("SAVED", "cm", "Coupling efficiency metrics", ce_path)
    return {"fsi": fsi_path, "amplitude": amp_path, "efficiency": ce_path}


def generate_coupling_metrics_report(
    fsi_df, amplitude_df, coupling_efficiency_df, output_dir
):
    report_path = os.path.join(output_dir, "coupling_metrics_summary_report.txt")
    with open(report_path, "w") as f:
        f.write("Monsoon Depression Coupling Metrics analysis\n\n")
        # FSI Summary
        f.write("1. Feedback Strength Index (FSI)\n")
        f.write(f"Tracks analyzed:                          {len(fsi_df)}\n")
        f.write(
            f"Mean FSI:                                 {fsi_df['FSI'].mean():.4f}\n"
        )
        f.write(
            f"Std Dev FSI:                              {fsi_df['FSI'].std():.4f}\n"
        )
        f.write(
            f"Max FSI:                                  {fsi_df['FSI'].max():.4f}\n"
        )
        f.write(
            f"Min FSI:                                  {fsi_df['FSI'].min():.4f}\n"
        )
        f.write("\nComponent Contributions:\n")
        f.write(
            f"  Correlation Strength (mean):            {fsi_df['correlation_strength'].mean():.4f}\n"
        )
        f.write(
            f"  Capacity Index (mean):                  {fsi_df['capacity_index'].mean():.4f}\n"
        )
        f.write(
            f"  Significant Results (p<0.05):           {fsi_df['is_significant'].sum()}/{len(fsi_df)}\n"
        )
        f.write("\n")
        # Diurnal Amplitude Summary
        f.write("2. Diurnal sensible heat flux amplitude\n")
        f.write(
            f"Land-phase amplitude (mean):              {amplitude_df['diurnal_amplitude_land'].mean():.2f} W/m²\n"
        )
        f.write(
            f"Ocean-phase amplitude (mean):             {amplitude_df['ocean_amplitude'].mean():.2f} W/m²\n"
        )
        f.write(
            f"Amplitude change (mean, Land-Ocean):      {amplitude_df['amplitude_change'].mean():.2f} W/m²\n"
        )
        f.write("\nMean SHF by Phase:\n")
        f.write(
            f"  Ocean phase:                            {amplitude_df['ocean_mean_shf'].mean():.2f} W/m²\n"
        )
        f.write(
            f"  Land phase:                             {amplitude_df['land_mean_shf'].mean():.2f} W/m²\n"
        )
        f.write("\n")

        # Coupling Efficiency Summary
        f.write("3. Coupling efficiency\n")
        f.write(
            f"Mean coupling efficiency:                 {coupling_efficiency_df['coupling_efficiency'].mean():.4f}\n"
        )
        f.write(
            f"Std Dev:                                  {coupling_efficiency_df['coupling_efficiency'].std():.4f}\n"
        )
        f.write(
            f"Tracks with strong coupling (CE > 0.5):   {(coupling_efficiency_df['coupling_efficiency'] > 0.5).sum()}\n"
        )
        f.write(
            f"Tracks with weak coupling (CE < 0.3):     {(coupling_efficiency_df['coupling_efficiency'] < 0.3).sum()}\n"
        )
        f.write("\n")
        # Interpretation
        # f.write("4. Interpretation & implications\n")
        # f.write("FSI captures the strength of precipitation-soil moisture coupling:\n")
        # f.write("  - High FSI: Strong feedback between rainfall and soil saturation\n")
        # f.write("  - Low FSI:  Weak coupling (dry soils, weak rainfall, low significance)\n\n")
        # f.write("Diurnal Amplitude measures land surface response:\n")
        # f.write("  - High amplitude: Sensitive land surface with strong diurnal heating\n")
        # f.write("  - Low amplitude:  Muted diurnal cycle (cloud cover, wet soils)\n\n")
        # f.write("Coupling Efficiency integrates both:\n")
        # f.write("  - Combined metric of feedback strength and surface response\n")
        # f.write("  - Normalized by system duration (longer systems show weaker signal)\n")
    std_print("SAVED", "cm", "Coupling metrics report", report_path)
    return report_path


# Helper functions
def compute_ce(row):
    fsi = row.get("FSI", np.nan)
    amp = row.get("diurnal_amplitude_land", np.nan)
    duration = row.get("duration_hours", np.nan)
    if np.isnan(fsi) or np.isnan(amp):
        return np.nan
    # Duration decay: systems > 120h show reduced coupling signal
    if np.isnan(duration):
        duration_factor = 1.0
    else:
        duration_factor = max(1.0, duration / 48)  # 48h as reference
        # duration_factor = 1.0 / (1.0 + np.exp((duration - 120) / 48))

    # Normalize
    amp_norm = np.tanh(amp / 100)  # Saturation curve
    # coupling_efficiency = (fsi * amp) / duration_factor
    coupling_efficiency = (fsi * amp_norm) / duration_factor
    return coupling_efficiency
