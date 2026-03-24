# FIX:
# 1. fix hardcoded columns names and coupling CSV paths
# 2. Fallback behavior Uses random perturbation if no capacity data,
#     tsi = fsi * 0.8 + np.random.uniform(0, 0.2, len(fsi)) but actuall :
#     tsi = elevation_effect * slope_effect * aspect_effect

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
    """Load TSI metrics computed from HGT elevation data"""
    path = os.path.join(output_dir, "elevation_metrics.csv")
    try:
        df = pd.read_csv(path)
        std_print("DEBUG", "sa", "Loaded elevation metrics", path)
        return df
    except Exception as e:
        std_print("ERROR", "sa", f"Failed to load {path}", str(e))

    std_print(
        "WARN", "sa", "Elevation metrics not found", "TSI analysis will be skipped"
    )
    return None


def analyze_asm_fsi_relationship(fsi_df, asm_df, output_dir):
    """Hypothesis: ASM fuels but FSI sustains (r≈0 expected)"""
    try:
        if asm_df is None:
            std_print("WARN", "sa", "ASM results None", "Cannot run analysis")
            return {}
        if isinstance(asm_df, pd.DataFrame) and asm_df.empty:
            std_print("WARN", "sa", "ASM results empty", "Cannot run analysis")
            return {}
        if fsi_df is None or fsi_df.empty:
            std_print("WARN", "sa", "FSI data empty or None", "Cannot run analysis")
            return {}
        # Validate and prepare data
        if "track_id" not in fsi_df.columns:
            fsi_df["track_id"] = fsi_df["track_id"].astype(int)
        if "track_id" not in asm_df.columns:
            asm_df = asm_df.copy()
            asm_df["track_id"] = range(len(asm_df))
            asm_df["track_id"] = asm_df["track_id"].astype(int)

        if "ASM" not in asm_df.columns:
            asm_candidates = [
                "asmi_24h",  # primary: from antecedent_analysis.py
                "asmi_7d",  # secondary: 7-day window
                "SWVL1_pre24h_mean",  # legacy name
                "antecedent_sm",  # alternative name
            ]
            for cand in asm_candidates:
                if cand in asm_df.columns:
                    asm_df = asm_df.copy()
                    asm_df["ASM"] = pd.to_numeric(asm_df[cand], errors="coerce")
                    std_print("INFO", "sa", f"ASM column mapped from '{cand}'")
                    break
            else:
                # Fallback: first numeric column that isn't track_id
                numeric = asm_df.select_dtypes(include=[np.number]).columns.tolist()
                numeric = [c for c in numeric if c != "track_id"]
                if numeric:
                    asm_df = asm_df.copy()
                    asm_df["ASM"] = asm_df[numeric[0]]
                    std_print(
                        "WARN", "sa", f"ASM fallback to numeric column '{numeric[0]}'"
                    )
                else:
                    std_print(
                        "ERROR",
                        "sa",
                        "Could not find any numeric ASM column",
                        str(list(asm_df.columns)),
                    )
                    return {}

        # Check for FSI column in fsi_df
        if "FSI" not in fsi_df.columns:
            if "feedback_strength_index" in fsi_df.columns:
                fsi_df["FSI"] = fsi_df["feedback_strength_index"]
            else:
                std_print(
                    "ERROR",
                    "sa",
                    "FSI column not found",
                    "Available: " + str(fsi_df.columns.tolist()),
                )
                return {}

        # Merge data
        merged_df = pd.merge(
            fsi_df[["track_id", "FSI"]],
            asm_df[["track_id", "ASM"]],
            on="track_id",
            how="inner",
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

        # Figure
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
        plot_path = os.path.join(stats_dir, "asm_fsi_scatter.png")
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
        results_path = os.path.join(stats_dir, "asm_fsi_results.csv")
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
        std_print(
            "ERROR", "sa", "Elevation metrics not available", "Skipping TSI analysis"
        )
        # return {'status': 'skipped', 'reason': 'Elevation metrics not found (NoneType)'}
        return None
    try:
        # Check available data
        valid_tsi = elevation_metrics_df[elevation_metrics_df["TSI"].notna()]
        std_print(
            "DEBUG",
            "sa",
            "Valid TSI records",
            f"{len(valid_tsi)}/{len(elevation_metrics_df)}",
        )

        if "duration_h" in elevation_metrics_df.columns:
            valid_pairs = elevation_metrics_df.dropna(subset=["TSI", "duration_h"])
            if len(valid_pairs) >= 5:
                # Compute correlation
                tsi_duration_corr = valid_pairs["TSI"].corr(valid_pairs["duration_h"])

                # Figure
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Scatter plot
                ax = axes[0]
                scatter = ax.scatter(
                    valid_pairs["TSI"],
                    valid_pairs["duration_h"],
                    c=(
                        valid_pairs["elevation_range"]
                        if "elevation_range" in valid_pairs.columns
                        else "steelblue"
                    ),
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
                    ax.set_ylabel(
                        "Mean Component Value", fontsize=12, fontweight="bold"
                    )
                    ax.set_title(
                        "TSI Component Contributions", fontsize=13, fontweight="bold"
                    )
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
                    interpretation = "Strong negative correlation - higher TSI (terrain-sensitive) -> shorter duration"
                elif tsi_duration_corr < -0.2:
                    interpretation = "Weak negative correlation - terrain sensitivity moderately reduces duration"
                elif tsi_duration_corr > 0.3:
                    interpretation = "Positive correlation - terrain-sensitive areas sustain longer tracks"
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


def analyze_gam_nonlinearity(coupling_df, output_dir):
    """Hypothesis: FSI ~ s(SM) plateau at high SM"""
    try:
        if coupling_df is None or coupling_df.empty:
            std_print("WARN", "sa", "Coupling data empty", "Skipping GAM")
            return {}
        df = coupling_df.copy()

        # Check for FSI column
        if "FSI" not in df.columns:
            if "feedback_strength_index" in df.columns:
                df["FSI"] = df["feedback_strength_index"]
            else:
                std_print(
                    "WARN",
                    "sa",
                    "FSI column not found",
                    f"Available {df.columns.tolist()}",
                )
                return {}

        # Find soil moisture column
        sm_col = None
        for col in df.columns:
            if "swvl" in col.lower() or "soil" in col.lower() or "sm" in col.lower():
                sm_col = col
                break
        if not sm_col:
            std_print(
                "WARN", "sa", "SM column not found", f"Available: {df.columns.tolist()}"
            )
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
            r_squared = gam.statistics_["pseudo_r2"]
            n_samples = len(data)
            std_print(
                "INFO",
                "sa",
                "GAM plateau analysis",
                f"R²≈{r_squared:.3f}, n={n_samples}",
            )
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
            std_print(
                "WARN",
                "sa",
                "Cannot find TP/SWVL1 columns",
                f"TP candidates: {tp_candidates}, SM candidates: {sm_candidates}",
            )
            std_print(
                "DEBUG", "sa", "Available columns", str(list(coupling_df.columns))
            )
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
                "n_samples": len(data),
            }

        # Run Granger test with maxlag=1 for small datasets
        maxlag = MAX_LAG if len(data) < 30 else 2
        try:
            gc_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            # Get p-value for MAX_LAG
            pearson_p = gc_result[1][0]["ssr_ftest"][1]
            result = {
                "pearson_p": pearson_p,
                "significant": pearson_p < 0.05,
                "n_samples": len(data),
                "tp_column": tp_col,
                "sm_column": sm_col,
                "maxlag": maxlag,
                "status": "success",
            }
            if pearson_p < 0.05:
                std_print("INFO", "sa", "Granger causality", f"p={pearson_p:.4f}")
            else:
                std_print(
                    "INFO", "sa", "No significant causality", f"p={pearson_p:.4f}"
                )

            # Save
            stats_dir = os.path.join(output_dir, "statistical_analysis")
            os.makedirs(stats_dir, exist_ok=True)
            granger_causality_CSV = os.path.join(stats_dir, "granger_causality.csv")
            pd.DataFrame([result]).to_csv(
                os.path.join(stats_dir, "granger_causality.csv"), index=False
            )
            std_print(
                "EXPORT", "sa", "Granger causality results", granger_causality_CSV
            )
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
    std_print("WARN", "sa", "Using proxy TSI (no elevation data)", "")
    if "FSI" in coupling_df.columns:
        fsi = coupling_df["FSI"].values
    else:
        return {"status": "failed", "reason": "No FSI for proxy TSI"}

    # Simple proxy based on FSI and capacity
    tsi = fsi * 0.8  # Base on FSI
    if "capacity_index" in coupling_df.columns:
        tsi = tsi * 0.7 + coupling_df["capacity_index"].values * 0.3

    results = {
        "TSI": tsi.tolist(),
        "TSI_mean": float(np.nanmean(tsi)),
        "TSI_std": float(np.nanstd(tsi)),
        "TSI_min": float(np.nanmin(tsi)),
        "TSI_max": float(np.nanmax(tsi)),
        "n_tracks": len(coupling_df),
        "source": "proxy_fsi_based",
        "status": "proxy",
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
            merged = pd.merge(
                coupling_df, elevation_results, on="track_id", how="inner"
            )
        else:
            std_print(
                "ERROR",
                "sa",
                "Cannot merge coupling and elevation data",
                "Missing track_id",
            )
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
                "status": "success",
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
                    "status": "success",
                }
            else:
                std_print("WARN", "sa", "Missing TSI components", "Using proxy TSI")
                return compute_proxy_tsi(coupling_df, output_dir)

        # Correlation with duration
        if "duration_h" in merged.columns:
            valid_mask = ~np.isnan(tsi) & ~np.isnan(merged["duration_h"])
            if valid_mask.sum() > 2:
                corr = np.corrcoef(
                    tsi[valid_mask], merged["duration_h"].values[valid_mask]
                )[0, 1]
                results["tsi_duration_correlation"] = float(corr)
                results["tsi_duration_n"] = int(valid_mask.sum())

                # Interpretation
                if corr < -0.5:
                    results["interpretation"] = (
                        "Strong negative: Higher TSI = Shorter duration"
                    )
                elif corr < -0.3:
                    results["interpretation"] = (
                        "Moderate negative: Terrain sensitivity reduces duration"
                    )
                elif corr > 0.5:
                    results["interpretation"] = (
                        "Strong positive: Higher TSI = Longer duration"
                    )
                elif corr > 0.3:
                    results["interpretation"] = "Moderate positive"
                else:
                    results["interpretation"] = "Weak/no relationship"
        std_print(
            None,
            None,
            "Mean"
            f"{results['TSI_mean']:.3f} ± {results['TSI_std']:.3f} (from {results['source']})",
        )
        return results
    except Exception as e:
        std_print("ERROR", "sa", "TSI computation failed", str(e))
        traceback.print_exc()
        return {"status": "failed", "reason": str(e)}


def write_statistical_summary(all_results, coupling_df, output_dir):
    """
    all_results keys: "asm_fsi", "gam", "granger", "tsi"
    """
    stats_output_dir = os.path.join(output_dir, "statistical_analysis")
    os.makedirs(stats_output_dir, exist_ok=True)
    summary_path = os.path.join(stats_output_dir, "statistical_summary.txt")
    try:
        asm_fsi_results = all_results.get("asm_fsi", {})
        granger_results = all_results.get("granger", {})

        with open(summary_path, "w") as f:
            f.write("Statistical analysis summary\n\n\n")
            f.write(f"   Total tracks analyzed:                   {len(coupling_df)}\n")
            if "track_id" in coupling_df.columns:
                track_ids = sorted(coupling_df["track_id"].unique().tolist())
                f.write(f"   Track IDs:                               {track_ids}\n")
                f.write(
                    f"   Number of tracks:                        {len(track_ids)}\n\n"
                )

            f.write("ASM-FSI Relationship\n\n")
            if asm_fsi_results and "correlation" in asm_fsi_results:
                corr = asm_fsi_results["correlation"]
                f.write(f"Correlation (r): {corr:.3f}\n")
                if corr is not None:
                    if abs(corr) < 0.3:
                        f.write(
                            "Interpretation: Supports (ASM fuels but FSI sustains)\n"
                        )
                        f.write(
                            "                Weak correlation suggests independent mechanisms\n"
                        )
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
            gam = all_results.get("gam", {})
            if gam.get("status") == "skipped":
                f.write(f"Status: SKIPPED - {gam.get('reason')}\n")
            elif "r_squared" in gam:
                f.write(f"R² score: {gam['r_squared']:.3f}\n")
                f.write("Expected: Plateau at high SM values\n")
            else:
                f.write("Analysis failed or incomplete\n")
            f.write("\n")

            f.write("Granger Causality (TP -> SWVL1)\n\n")
            if granger_results and "significant" in granger_results:
                p_val = granger_results.get("pearson_p", "N/A")
                sig = granger_results.get("significant", False)
                f.write(f"p-value: {p_val}\n")
                f.write(f"Significant at α=0.05: {'YES' if sig else 'NO'}\n")
                if isinstance(p_val, float):
                    f.write(
                        "Interpretation: Supports - TP Granger-causes SWVL1\n"
                        if p_val < 0.05
                        else "Interpretation: No significant causality detected\n"
                    )
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
                    f.write(
                        f"Interpretation: {tsi_results.get('interpretation', '')}\n"
                    )
                elif tsi_results["status"] == "skipped":
                    f.write(
                        f"Status: SKIPPED - {tsi_results.get('reason', 'Unknown')}\n"
                    )
                elif tsi_results["status"] == "incomplete":
                    f.write(
                        f"Status: INCOMPLETE - {tsi_results.get('reason', 'Unknown')}\n"
                    )
                else:
                    f.write("Analysis incomplete or failed\n")
            f.write("\n")

            f.write("Data quality:\n\n")
            if len(coupling_df) < 10:
                f.write(
                    f"  Only {len(coupling_df)} tracks - statistical power limited\n"
                )
                f.write("  Consider relaxing detection thresholds to get more tracks\n")
            if "pearson_p" in coupling_df.columns:
                nan_pct = (
                    coupling_df["pearson_p"].isna().sum() / len(coupling_df)
                ) * 100
                if nan_pct > 20:
                    f.write(f"  {nan_pct:.1f}% of p-values are NaN\n")

        std_print("SAVED", "sa", "Statistical analysis summary", summary_path)
    except Exception as e:
        std_print("ERROR", "sa", "Failed to write summary", str(e))
        traceback.print_exc()
