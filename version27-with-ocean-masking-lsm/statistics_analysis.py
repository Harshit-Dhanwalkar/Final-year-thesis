# FIX:
# 1. fix hardcoded columns names and coupling CSV paths
# 2. Fallback behavior Uses random perturbation if no capacity data, tsi = fsi * 0.8 + np.random.uniform(0, 0.2, len(fsi)) but actuall :
#     tsi = elevation_effect * slope_effect * aspect_effect
# TODO:
#   elevation_effect = min(1.0, elev_range / 3000.0)  # 3000m typical range
#   slope_effect = min(1.0, mean_slope / 30.0)        # 30° very steep
#   roughness_effect = min(1.0, elevation_std / 500.0) # 500m very rough

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from pygam import LinearGAM, s
from typing import Dict, Optional
import traceback

from utils import std_print

MAX_LAG = 1


def load_elevation_metrics(output_dir):
    """Load TSI metrics computed from HGT elevation data."""
    paths_to_try = [
        os.path.join(output_dir, "elevation_analysis", "elevation_metrics.csv"),
        "output/elevation_analysis/elevation_metrics.csv",
        "elevation_metrics.csv",
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                std_print("INFO", "sa", "Loaded elevation metrics", path)
                return df
            except Exception as e:
                std_print("WARN", "sa", f"Failed to load {path}", str(e)[:40])
    std_print(
        "WARN", "sa", "Elevation metrics not found", "TSI analysis will be skipped"
    )
    return None


def analyze_asm_fsi_relationship(coupling_metrics_dir, asm_df, output_dir):
    """Hypothesis: ASM fuels but FSI sustains (r≈0 expected)"""
    try:
        if asm_df is None:
            std_print("WARN", "sa", "ASM results None", "Cannot run analysis")
            return {}
        if isinstance(asm_df, pd.DataFrame) and asm_df.empty:
            std_print("WARN", "sa", "ASM results empty", "Cannot run analysis")
            return {}
        # Load FSI data
        fsi_path = os.path.join(coupling_metrics_dir, "coupling_metrics_FSI.csv")
        # Try alternative paths
        if not os.path.exists(fsi_path):
            alt_paths = [
                os.path.join(output_dir, "coupling_metrics_FSI.csv"),
                os.path.join(
                    output_dir, "coupling_metrics", "coupling_metrics_FSI.csv"
                ),
                "coupling_metrics_FSI.csv",
                "output_2022/coupling_metrics/coupling_metrics_FSI.csv",
                "output_2023/coupling_metrics/coupling_metrics_FSI.csv",
                "output_2024/coupling_metrics/coupling_metrics_FSI.csv",
                "output_2025/coupling_metrics/coupling_metrics_FSI.csv",
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    fsi_path = alt_path
                    break
        if not os.path.exists(fsi_path):
            std_print("WARN", "sa", "FSI file not found, Check", coupling_metrics_dir)
            return {}
        fsi_df = pd.read_csv(fsi_path)
        if fsi_df.empty:
            std_print("WARN", "sa", "FSI data empty", "Cannot run analysis")
            return {}
        # Validate and prepare data
        if "track_id" not in fsi_df.columns:
            fsi_df["track_id"] = fsi_df["track_id"].astype(int)
        if "track_id" not in asm_df.columns:
            asm_df = asm_df.copy()
            asm_df["track_id"] = range(len(asm_df))
            asm_df["track_id"] = asm_df["track_id"].astype(int)
        # Rename ASM column if needed
        if "ASM" not in asm_df.columns:
            if "SWVL1_pre24h_mean" in asm_df.columns:
                asm_df["ASM"] = asm_df["SWVL1_pre24h_mean"]
            elif "antecedent_sm" in asm_df.columns:
                asm_df["ASM"] = asm_df["antecedent_sm"]
            else:
                # Use 2nd column
                try:
                    asm_df["ASM"] = asm_df.iloc[:, 1]
                except Exception as e:
                    std_print("ERROR", "sa", "Could not extract ASM column. Check ASM file", str(e))
                    return {}

        # Check for FSI column in fsi_df
        if "FSI" not in fsi_df.columns:
            if "feedback_strength_index" in fsi_df.columns:
                fsi_df["FSI"] = fsi_df["feedback_strength_index"]
            else:
                std_print("ERROR", "sa", "FSI column not found", "Available: " + str(fsi_df.columns.tolist()))
                return {}

        # Merge data
        merged_df = pd.merge(
            fsi_df[["track_id", "FSI"]],
            asm_df[["track_id", "ASM"]],
            on="track_id",
            how="inner"
        )
        if merged_df.empty:
            std_print(
                "WARN",
                "sa",
                "No matching tracks after merge",
                "Check track_id alignment",
            )
            return {}
        merged_df = merged_df.dropna(subset=["ASM", "FSI"])
        if merged_df.empty:
            std_print(
                "WARN", "sa", "All data NaN after filtering", "Check source files"
            )
            return {}
        if "ASM" in merged_df.columns and "mean_swvl1" in merged_df.columns:
            asm_fsi_corr = merged_df["ASM"].corr(merged_df["mean_swvl1"])
            std_print(
                "DEBUG", "sa", "ASM vs mean_swvl1 correlation", f"{asm_fsi_corr:.3f}"
            )
            if abs(asm_fsi_corr) > 0.9:
                std_print(
                    "WARN",
                    "sa",
                    "ASM and mean_swvl1 highly correlated",
                    "They might be measuring similar things",
                )

        # Calculate correlation
        correlation = merged_df["ASM"].corr(merged_df["FSI"])
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        color_col = "duration_h" if "duration_h" in merged_df.columns else None
        if color_col and merged_df[color_col].notna().any():
            scatter = ax.scatter(
                merged_df["ASM"],
                merged_df["FSI"],
                c=merged_df[color_col],
                cmap="viridis",
                alpha=0.7,
                s=100,
                edgecolors="black",
                linewidth=1,
            )
            plt.colorbar(scatter, ax=ax, label="Duration (hours)")
        else:
            scatter = ax.scatter(
                merged_df["ASM"],
                merged_df["FSI"],
                c="steelblue",
                alpha=0.7,
                s=100,
                edgecolors="black",
                linewidth=1,
            )

        ax.set_xlabel("ASM$_{24h}$ (m³/m³)", fontsize=12, fontweight="bold")
        ax.set_ylabel("FSI (Feedback Strength Index)", fontsize=12, fontweight="bold")
        # Interpretation
        if np.isnan(correlation):
            title = "ASM vs. FSI (data issue)"
        else:
            # 'ASM Fuels but FSI Sustains (r = {correlation})
            title = f"ASM-FSI correlation (r = {correlation:.3f})"

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Save plot
        stats_dir = os.path.join(output_dir, "statistical_analysis")
        os.makedirs(stats_dir, exist_ok=True)
        plot_path = os.path.join(stats_dir, "h3_asm_fsi_scatter.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print(
            "INFO",
            "sa",
            "ASM-FSI correlation",
            f"{correlation:.3f}" if not np.isnan(correlation) else "NaN (data issue)",
        )
        std_print("SAVED", "sa", "ASM-FSI plot", plot_path)

        # Save results
        results_path = os.path.join(stats_dir, "h3_asm_fsi_results.csv")
        merged_df.to_csv(results_path, index=False)
        std_print("EXPORT", "sa", "results CSV", results_path)
        return {
            "correlation": correlation if not np.isnan(correlation) else None,
            "n_tracks": len(merged_df),
            "plot_path": plot_path,
            "results_path": results_path,
        }
    except Exception as e:
        std_print("ERROR", "sa", "ASM-FSI analysis failed", str(e))
        traceback.print_exc()
        return {}


def analyze_tsi(elevation_metrics_df, coupling_df, output_dir):
    """
    Hypothesis: Topographic effects on monsoon depression behavior
    Analyze TSI (computed from SRTM HGT elevation data) against:
      - Duration: TSI-duration correlation (expect r < -0.3)
      - FSI: Terrain modulation of feedback (optional)
      - ASM: Soil moisture interaction (optional)
    """
    if elevation_metrics_df is None or elevation_metrics_df.empty:
        std_print("ERROR", "sa", "Elevation metrics not available", "Skipping TSI analysis")
        # return {'status': 'skipped', 'reason': 'Elevation metrics not found (NoneType)'}
        return None
    try:
        # Check available data
        valid_tsi = elevation_metrics_df[elevation_metrics_df["TSI"].notna()]
        std_print("DEBUG", "sa", "Valid TSI records", f"{len(valid_tsi)}/{len(elevation_metrics_df)}")

        if "duration_h" in elevation_metrics_df.columns:
            valid_pairs = elevation_metrics_df.dropna(subset=["TSI", "duration_h"])
            if len(valid_pairs) >= 5:
                # Compute correlation
                tsi_duration_corr = valid_pairs["TSI"].corr(valid_pairs["duration_h"])
                # Plot
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                # Scatter plot
                ax = axes[0]
                scatter = ax.scatter(
                    valid_pairs["TSI"],
                    valid_pairs["duration_h"],
                    c=valid_pairs["elevation_range"]
                    if "elevation_range" in valid_pairs.columns
                    else "steelblue",
                    s=100,
                    alpha=0.6,
                    cmap="viridis",
                    edgecolors="black",
                    linewidth=1,
                )
                ax.set_xlabel(
                    "TSI (Topographic Sensitivity Index)",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_ylabel("Track Duration (hours)", fontsize=12, fontweight="bold")
                ax.set_title(
                    f"TSI vs Duration (r={tsi_duration_corr:.3f})",
                    fontsize=13,
                    fontweight="bold",
                )
                if "elevation_range" in valid_pairs.columns:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label("Elevation Range (m)", fontsize=10)
                ax.grid(True, alpha=0.3)

                # Component effects
                ax = axes[1]
                components = ["elevation_effect", "slope_effect", "roughness_effect"]
                component_values = []
                component_labels = [
                    "Elevation Effect",
                    "Slope Effect",
                    "Roughness Effect",
                ]
                for comp in components:
                    if comp in valid_pairs.columns:
                        component_values.append(valid_pairs[comp].mean())

                if component_values:
                    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                    # FIX: bars declared but never used
                    bars = ax.bar(
                        component_labels[: len(component_values)],
                        component_values,
                        color=colors[: len(component_values)],
                        edgecolor="black",
                        linewidth=1.5,
                        alpha=0.7,
                    )
                    ax.set_ylabel("Mean Component Value", fontsize=12, fontweight="bold")
                    ax.set_title("TSI Component Contributions", fontsize=13, fontweight="bold")
                    ax.set_ylim(0, 1.0)
                    ax.grid(True, alpha=0.3, axis="y")

                plt.tight_layout()
                stats_dir = os.path.join(output_dir, "statistical_analysis")
                os.makedirs(stats_dir, exist_ok=True)
                plot_path = os.path.join(stats_dir, "tsi_duration.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                std_print("SAVED", "sa", "TSI-duration plot", plot_path)

                # Interpretation
                interpretation = ""
                if tsi_duration_corr < -0.5:
                    interpretation = ("Strong negative correlation - higher TSI (terrain-sensitive) -> shorter duration")
                elif tsi_duration_corr < -0.2:
                    interpretation = ("Weak negative correlation - terrain sensitivity moderately reduces duration")
                elif tsi_duration_corr > 0.3:
                    interpretation = ("Positive correlation - terrain-sensitive areas sustain longer tracks")
                else:
                    interpretation = "Weak or no significant correlation"
                tsi_results = {
                    "status": "completed",
                    "tsi_duration_correlation": float(tsi_duration_corr),
                    "n_pairs": len(valid_pairs),
                    "interpretation": interpretation,
                    "mean_tsi": float(valid_pairs["TSI"].mean()),
                    "mean_duration_h": float(valid_pairs["duration_h"].mean()),
                    "plot_path": plot_path,
                }
                # Save results
                results_path = os.path.join(stats_dir, "tsi_results.csv")
                valid_pairs.to_csv(results_path, index=False)
                std_print("EXPORT", "sa", "results CSV", results_path)
                tsi_results["results_path"] = results_path
                return tsi_results
        return {
            "status": "incomplete",
            "reason": "Missing duration data",
            "note": "Cannot compute without duration information",
        }
    except Exception as e:
        std_print("ERROR", "sa", "TSI analysis failed", str(e))
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def analyze_gam_nonlinearity(coupling_metrics_dir, output_dir):
    """Hypothesis: FSI ~ s(SM) plateau at high SM"""
    try:
        # Try multiple paths for coupling metrics
        metrics_paths = [
            os.path.join(coupling_metrics_dir, "coupling_metrics_FSI.csv"),
            os.path.join(output_dir, "coupling_metrics_FSI.csv"),
            os.path.join(output_dir, "coupling_metrics", "coupling_metrics_FSI.csv"),
        ]
        metrics_path = None
        for path in metrics_paths:
            if os.path.exists(path):
                metrics_path = path
                break
        if not metrics_path:
            std_print("WARN", "sa", "Coupling metrics not found", "Checked 3 paths")
            return {}
        df = pd.read_csv(metrics_path)
        if df.empty:
            std_print("WARN", "sa", "Coupling data empty", "Skipping GAM")
            return {}

        # Check for FSI column
        if "FSI" not in df.columns:
            if "feedback_strength_index" in df.columns:
                df["FSI"] = df["feedback_strength_index"]
            else:
                std_print("WARN", "sa", "FSI column not found", f"Available {df.columns.tolist()}")
                return {}

        # Find soil moisture column
        sm_col = None
        for col in df.columns:
            if "swvl" in col.lower() or "soil" in col.lower() or "sm" in col.lower():
                sm_col = col
                break
        if not sm_col:
            std_print("WARN", "sa", "SM column not found", f"Available: {df.columns.tolist()}")
            return {}

        # Prepare data
        data = df[["FSI", sm_col]].dropna()
        if len(data) < 10:
            std_print("WARN", "sa", "Insufficient data for GAM (n < 10)", len(data))
            return {}

        # Fit GAM
        try:
            X = data[sm_col].values.reshape(-1, 1)
            y = data["FSI"].values
            gam = LinearGAM(s(0, n_splines=10))
            gam.fit(X, y)
            r_squared = gam.statistics_["GCV"]
            n_samples = len(data)
            std_print("INFO", "sa", "GAM plateau analysis", f"R²≈{r_squared:.3f}, n={n_samples}")
            return {
                "r_squared": float(r_squared),
                "n_samples": n_samples,
                "sm_column": sm_col,
            }
        except Exception as e:
            std_print("WARN", "sa", "GAM fitting error", str(e))
            return {}
    except Exception as e:
        std_print("ERROR", "sa", "GAM analysis failed", str(e))
        traceback.print_exc()
        return {}


# FIX: output_dir is not accessed
def test_granger_causality(coupling_df, output_dir):
    """TP -> SWVL1 causality"""
    try:
        tp_col = None
        sm_col = None
        # Try different column name patterns
        tp_candidates = ["mean_tp", "TP", "tp", "precipitation", "total_precipitation"]
        sm_candidates = ["mean_swvl1", "SWVL1", "swvl1", "soil_moisture", "soil_water"]
        for col in tp_candidates:
            if col in coupling_df.columns:
                tp_col = col
                break
        for col in sm_candidates:
            if col in coupling_df.columns:
                sm_col = col
                break

        if not tp_col or not sm_col:
            std_print("WARN", "sa", "Cannot find TP/SWVL1 columns", f"TP candidates: {tp_candidates}, SM candidates: {sm_candidates}")
            std_print("DEBUG", "sa", "Available columns", str(list(coupling_df.columns)))
            return {
                "status": "failed",
                "reason": "Missing TP or SWVL1 columns",
                "available_columns": list(coupling_df.columns),
            }
        std_print("INFO", "sa", "Using columns", f"{tp_col} -> {sm_col}")
        # Prepare data
        data = coupling_df[[tp_col, sm_col]].copy()
        data = data.apply(pd.to_numeric, errors="coerce").dropna()
        if len(data) < 10:
            return {
                "status": "failed",
                "reason": f"Insufficient data after cleaning: n={len(data)}",
                "tp_column": tp_col,
                "sm_column": sm_col,
                "n_samples": len(data)
            }
        # Run Granger test with maxlag=1 for small datasets
        maxlag = MAX_LAG if len(data) < 30 else 2
        try:
            gc_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            # Get p-value for MAX_LAG
            p_value = gc_result[1][0]["ssr_ftest"][1]
            result = {
                "p_value": p_value,
                "significant": p_value < 0.05,
                "n_samples": len(data),
                "tp_column": tp_col,
                "sm_column": sm_col,
                "maxlag": maxlag,
                "status": "success"
            }
            if p_value < 0.05:
                std_print("INFO", "sa", "Granger causality", f"p={p_value:.4f}")
            else:
                std_print("INFO", "sa", "No significant causality", f"p={p_value:.4f}")
            return result
        except Exception as e:
            std_print("WARN", "sa", "Granger test error", str(e))
            return {
                "status": "failed",
                "reason": f"Granger test error: {str(e)}",
                "tp_column": tp_col,
                "sm_column": sm_col,
            }
    except Exception as e:
        std_print("ERROR", "sa", "Granger causality failed", str(e))
        traceback.print_exc()
        return {"status": "failed", "reason": str(e)}


# FIX: output_dir is not accessed
def compute_proxy_tsi(coupling_df, output_dir):
    """Fallback: Proxy TSI when no elevation data"""
    std_print('WARN', 'sa', 'Using proxy TSI (no elevation data)', '')
    if 'FSI' in coupling_df.columns:
        fsi = coupling_df['FSI'].values
    else:
        return {'status': 'failed', 'reason': 'No FSI for proxy TSI'}
    # Simple proxy based on FSI and capacity
    tsi = fsi * 0.8  # Base on FSI
    if 'capacity_index' in coupling_df.columns:
        tsi = tsi * 0.7 + coupling_df['capacity_index'].values * 0.3
    results = {
        'TSI': tsi.tolist(),
        'TSI_mean': float(np.nanmean(tsi)),
        'TSI_std': float(np.nanstd(tsi)),
        'TSI_min': float(np.nanmin(tsi)),
        'TSI_max': float(np.nanmax(tsi)),
        'n_tracks': len(coupling_df),
        'source': 'proxy_fsi_based',
        'status': 'proxy',
        'note': 'Proxy TSI - elevation data required for accurate TSI'
    }
    return results


def compute_tsi_index(coupling_df, elevation_results, output_dir):
    """Topographic Sensitivity Index"""
    try:
        std_print("INFO", "sa", "Computing TSI index (using elevation data")
        # Check elevation results exists
        if elevation_results is None or elevation_results.empty:
            std_print("WARN", "sa", "No elevation data for TSI", "Using proxy TSI")
            return compute_proxy_tsi(coupling_df, output_dir)
        # Merge coupling data with elevation results
        if (
            "track_id" in coupling_df.columns
            and "track_id" in elevation_results.columns
        ):
            merged = pd.merge(coupling_df, elevation_results, on="track_id", how="inner")
        else:
            std_print("ERROR", "sa", "Cannot merge coupling and elevation data", "Missing track_id")
            return compute_proxy_tsi(coupling_df, output_dir)
        if merged.empty:
            std_print("WARN", "sa", "No matching tracks after merge", "Using proxy TSI")
            return compute_proxy_tsi(coupling_df, output_dir)
        # Extract TSI from elevation analysis if available
        if "TSI" in merged.columns:
            tsi = merged["TSI"].values
            results = {
                "TSI": tsi.tolist(),
                "TSI_mean": float(np.nanmean(tsi)),
                "TSI_std": float(np.nanstd(tsi)),
                "TSI_min": float(np.nanmin(tsi)),
                "TSI_max": float(np.nanmax(tsi)),
                "n_tracks": len(merged),
                "source": "elevation_data",
                "status": "success"
            }
        else:
            # Calculate TSI from components
            required_cols = ["terrain_effect", "soil_effect", "feedback_effect"]
            if all(col in merged.columns for col in required_cols):
                tsi = (
                    merged["terrain_effect"]
                    * merged["soil_effect"]
                    * merged["feedback_effect"]
                )
                results = {
                    "TSI": tsi.tolist(),
                    "TSI_mean": float(tsi.mean()),
                    "TSI_std": float(tsi.std()),
                    "TSI_min": float(tsi.min()),
                    "TSI_max": float(tsi.max()),
                    "n_tracks": len(merged),
                    "source": "calculated_from_components",
                    "status": "success"
                }
            else:
                std_print("WARN", "sa", "Missing TSI components", "Using proxy TSI")
                return compute_proxy_tsi(coupling_df, output_dir)
        # Correlation with duration
        if "duration_h" in merged.columns:
            valid_mask = ~np.isnan(tsi) & ~np.isnan(merged["duration_h"])
            if valid_mask.sum() > 2:
                corr = np.corrcoef(tsi[valid_mask], merged["duration_h"].values[valid_mask])[0, 1]
                results["tsi_duration_correlation"] = float(corr)
                results["tsi_duration_n"] = int(valid_mask.sum())
                # Interpretation
                if corr < -0.5:
                    results["interpretation"] = ("Strong negative: Higher TSI = Shorter duration")
                elif corr < -0.3:
                    results["interpretation"] = ("Moderate negative: Terrain sensitivity reduces duration")
                elif corr > 0.5:
                    results["interpretation"] = ("Strong positive: Higher TSI = Longer duration")
                elif corr > 0.3:
                    results["interpretation"] = "Moderate positive"
                else:
                    results["interpretation"] = "Weak/no relationship"
        std_print(None, None, "Mean" f"{results['TSI_mean']:.3f} ± {results['TSI_std']:.3f} (from {results['source']})")
        return results
    except Exception as e:
        std_print("ERROR", "sa", "TSI computation failed", str(e))
        traceback.print_exc()
        return {"status": "failed", "reason": str(e)[:100]}


def run_all_statistical_analyses(coupling_metrics_dir, asm_df, correlation_df, output_dir):
    std_print("INFO", "sa", "--- Starting Statistical Analyses ---", "")
    stats_output_dir = os.path.join(output_dir, "statistical_analysis")
    os.makedirs(stats_output_dir, exist_ok=True)

    all_results = {}
    # Try to load enhanced FSI data first
    metrics_paths = [
        os.path.join(coupling_metrics_dir, "coupling_metrics_FSI_enhanced.csv"),
        os.path.join(coupling_metrics_dir, "coupling_metrics_FSI.csv"),
        os.path.join(output_dir, "coupling_metrics", "coupling_metrics_FSI_enhanced.csv"),
        "coupling_metrics_FSI_enhanced.csv"
    ]
    coupling_df = None
    loaded_path = None
    for path in metrics_paths:
        if os.path.exists(path):
            try:
                coupling_df = pd.read_csv(path)
                if not coupling_df.empty:
                    std_print("INFO", "sa", "Loaded coupling data", path)
                    std_print("DEBUG", "sa", "Columns available", str(list(coupling_df.columns)))
                    loaded_path = path
                    break
            except Exception as e:
                std_print("WARN", "sa", f"Failed to load {path}", str(e)[:40])
    # If still no data, check if correlation_df exists
    if coupling_df is None or coupling_df.empty:
        if correlation_df is not None and not correlation_df.empty:
            std_print("INFO", "sa", "Using correlation data directly, Shape :", correlation_df.shape)
            coupling_df = correlation_df.copy()
            loaded_path = "correlation_df (direct)"
        else:
            std_print("ERROR", "sa", "No data found for analysis")
            return all_results

    # ASM-FSI relationship
    std_print("INFO", "sa", "--- Analyzing ASM-FSI relationship ---")
    asm_fsi_results = analyze_asm_fsi_relationship(coupling_metrics_dir, asm_df, output_dir)
    all_results["asm_fsi"] = asm_fsi_results

    # GAM nonlinearity (skip if < 10 tracks)
    std_print("INFO", "sa", "--- Running GAM nonlinear analysis ---")
    if len(coupling_df) >= 6:  # FIX: 6 is too less for GAM but I have only 6 trakcs
        gam_results = analyze_gam_nonlinearity(coupling_metrics_dir, output_dir)
        all_results["gam"] = gam_results
    else:
        std_print("WARN", "sa", "Skipping GAM", f"Need ≥10 tracks, have {len(coupling_df)}")
        all_results["gam"] = {
            "status": "skipped",
            "reason": f"Only {len(coupling_df)} tracks"
        }

    # Granger causality
    std_print("INFO", "sa", " --- Testing Granger causality ---")
    granger_results = test_granger_causality(coupling_df, output_dir)
    all_results["granger"] = granger_results

    # TSI from Elevation Data
    std_print("INFO", "sa", "--- Analyzing TSI with elevation data ---")
    elevation_metrics_df = load_elevation_metrics(output_dir)
    if elevation_metrics_df is not None and not elevation_metrics_df.empty:
        tsi_results = analyze_tsi(elevation_metrics_df, coupling_df, output_dir)
        all_results["tsi"] = tsi_results
    else:
        std_print("WARN", "sa", "Skipping TSI analysis", "Elevation metrics not available")
        all_results["tsi"] = {
            "status": "skipped",
            "reason": "Elevation metrics file not found",
            "note": "Run elevation_analysis.py first"
        }

    # Generate summary report
    summary_path = os.path.join(stats_output_dir, "statistical_summary.txt")
    try:
        with open(summary_path, "w") as f:
            f.write("Statistical analysis summary\n\n\n")
            f.write(f"   Total tracks analyzed:                   {len(coupling_df)}\n")
            f.write(f"   Data source:                             {loaded_path if loaded_path else 'Unknown'}\n\n")
            # Track list
            if "track_id" in coupling_df.columns:
                track_ids = sorted(coupling_df["track_id"].unique().tolist())
                f.write(f"   Track IDs:                               {track_ids}\n")
                f.write(f"   Number of tracks:                        {len(track_ids)}\n\n")

            f.write("ASM-FSI Relationship\n\n")
            if asm_fsi_results and "correlation" in asm_fsi_results:
                corr = asm_fsi_results["correlation"]
                f.write(f"Correlation (r): {corr:.3f}\n")
                if corr is not None:
                    if abs(corr) < 0.3:
                        f.write("Interpretation: Supports (ASM fuels but FSI sustains)\n")
                        f.write("                Weak correlation suggests independent mechanisms\n")
                    elif corr > 0.7:
                        f.write("Interpretation: Strong positive correlation\n")
                        f.write("                ASM and FSI may be directly linked\n")
                    else:
                        f.write("Interpretation: Moderate correlation\n")
                f.write(f"Tracks analyzed: {asm_fsi_results.get('n_tracks', 'N/A')}\n")
            else:
                f.write("Analysis incomplete or failed\n")
            f.write("\n")

            f.write("GAM Nonlinear Analysis\n\n")
            if "gam" in all_results:
                if all_results["gam"].get("status") == "skipped":
                    f.write(f"Status: SKIPPED - {all_results['gam'].get('reason')}\n")
                elif "r_squared" in all_results["gam"]:
                    f.write(f"R² score: {all_results['gam']['r_squared']:.3f}\n")
                    f.write("Expected: Plateau at high SM values\n")
                else:
                    f.write("Analysis failed or incomplete\n")
            f.write("\n")

            f.write("ranger Causality (TP -> SWVL1)\n\n")
            if granger_results and "significant" in granger_results:
                p_val = granger_results.get("p_value", "N/A")
                sig = granger_results.get("significant", False)
                f.write(f"p-value: {p_val}\n")
                f.write(f"Significant at α=0.05: {'YES' if sig else 'NO'}\n")
                if isinstance(p_val, float):
                    if p_val < 0.05:
                        f.write("Interpretation: Supports - TP Granger-causes SWVL1\n")
                    else:
                        f.write("Interpretation: No significant causality detected\n")
                f.write(
                    f"Data columns used: {granger_results.get('tp_column', '?')} -> {granger_results.get('sm_column', '?')}\n"
                )
            else:
                f.write("Analysis incomplete or failed\n")
            f.write("\n")

            f.write("Topographic Sensitivity Index (SRTM Data)\n\n")
            tsi_results = all_results.get("tsi", {})
            if tsi_results and "status" in tsi_results:
                if tsi_results["status"] == "completed":
                    f.write(
                        f"TSI-Duration Correlation: {tsi_results.get('tsi_duration_correlation', 'N/A'):.3f}\n"
                    )
                    f.write(f"Mean TSI: {tsi_results.get('mean_tsi', 'N/A'):.3f}\n")
                    f.write(f"Track count: {tsi_results.get('n_pairs', 'N/A')}\n")
                    f.write(f"Interpretation: {tsi_results.get('interpretation', '')}\n")
                elif tsi_results["status"] == "skipped":
                    f.write(f"Status: SKIPPED - {tsi_results.get('reason', 'Unknown')}\n")
                elif tsi_results["status"] == "incomplete":
                    f.write(f"Status: INCOMPLETE - {tsi_results.get('reason', 'Unknown')}\n")
                else:
                    f.write("Analysis incomplete or failed\n")
            f.write("\n")

            # Data Quality Note
            f.write("Data quality :\n\n")
            if len(coupling_df) < 10:
                f.write(f"  Only {len(coupling_df)} tracks - statistical power limited\n")
                f.write("   Consider relaxing detection thresholds to get more tracks\n")
            if "p_value" in coupling_df.columns:
                nan_pct = (coupling_df["p_value"].isna().sum() / len(coupling_df)) * 100
                if nan_pct > 20:
                    f.write(f"  {nan_pct:.1f}% of p-values are NaN\n")
        std_print("SAVED", "sa", "Statistical analysis summary", summary_path)
    except Exception as e:
        std_print("ERROR", "sa", "Failed to write summary", str(e))
        traceback.print_exc()
    return all_results
