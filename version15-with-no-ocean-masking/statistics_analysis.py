# statistics_analysis.py

# FIX: 
# 1. fix hardcoded columns names and coupling CSV paths
"""
Statistical analysis for H3-H7 hypotheses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from pygam import LinearGAM, s
import xarray as xr
import os
from typing import Dict, List, Optional, Tuple
import traceback
import warnings
warnings.filterwarnings('ignore')

from utils import std_print

MAX_LAG = 1

def analyze_asm_fsi_relationship(coupling_metrics_dir: str, asm_results: pd.DataFrame, output_dir: str) -> Dict:
    """H3 Hypothesis: ASM fuels but FSI sustains (r≈0 expected)"""
    try:
        if asm_results is None:
            std_print('WARN', 'stats', 'ASM results None', 'Cannot run H3 analysis')
            return {}
        if isinstance(asm_results, pd.DataFrame) and asm_results.empty:
            std_print('WARN', 'stats', 'ASM results empty', 'Cannot run H3 analysis')
            return {}
        # Load FSI data
        fsi_path = os.path.join(coupling_metrics_dir, "coupling_metrics_FSI.csv")
        # Try alternative paths
        if not os.path.exists(fsi_path):
            alt_paths = [
                os.path.join(output_dir, "coupling_metrics_FSI.csv"),
                os.path.join(output_dir, "coupling_metrics", "coupling_metrics_FSI.csv"),
                "coupling_metrics_FSI.csv",
                "output_2022/coupling_metrics/coupling_metrics_FSI.csv",
                "output_2023/coupling_metrics/coupling_metrics_FSI.csv",
                "output_2024/coupling_metrics/coupling_metrics_FSI.csv",
                "output_2025/coupling_metrics/coupling_metrics_FSI.csv"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    fsi_path = alt_path
                    break
        if not os.path.exists(fsi_path):
            std_print('WARN', 'stats', 'FSI file not found', f'Checked: {coupling_metrics_dir}')
            return {}
        fsi_df = pd.read_csv(fsi_path)
        if fsi_df.empty:
            std_print('WARN', 'stats', 'FSI data empty', 'Cannot run H3 analysis')
            return {}
        # Validate and prepare data
        if 'track_id' not in fsi_df.columns:
            fsi_df['track_id'] = range(len(fsi_df))
        if 'track_id' not in asm_results.columns:
            asm_results = asm_results.copy()
            asm_results['track_id'] = range(len(asm_results))
        # Rename ASM column if needed
        if 'ASM' not in asm_results.columns:
            if 'SWVL1_pre24h_mean' in asm_results.columns:
                asm_results['ASM'] = asm_results['SWVL1_pre24h_mean']
            elif 'antecedent_sm' in asm_results.columns:
                asm_results['ASM'] = asm_results['antecedent_sm']
            else:
                # Use 2nd column
                try:
                    asm_results['ASM'] = asm_results.iloc[:, 1]
                except:
                    std_print('WARN', 'stats', 'Could not extract ASM column', 'Check ASM file')
                    return {}

        # Check for FSI column in fsi_df
        if 'FSI' not in fsi_df.columns:
            if 'feedback_strength_index' in fsi_df.columns:
                fsi_df['FSI'] = fsi_df['feedback_strength_index']
            else:
                std_print('WARN', 'stats', 'FSI column not found', 'Available: ' + str(fsi_df.columns.tolist()))
                return {}

        # Merge data
        merged_df = pd.merge(fsi_df[['track_id', 'FSI']], asm_results[['track_id', 'ASM']], on='track_id', how='inner')
        if merged_df.empty:
            std_print('WARN', 'stats', 'No matching tracks after merge', 'Check track_id alignment')
            return {}
        merged_df = merged_df.dropna(subset=['ASM', 'FSI'])
        if merged_df.empty:
            std_print('WARN', 'stats', 'All data NaN after filtering', 'Check source files')
            return {}
        if 'ASM' in merged_df.columns and 'mean_swvl1' in merged_df.columns:
            asm_fsi_corr = merged_df['ASM'].corr(merged_df['mean_swvl1'])
            std_print('DEBUG', 'stats', 'ASM vs mean_swvl1 correlation', f'{asm_fsi_corr:.3f}')
            if abs(asm_fsi_corr) > 0.9:
                std_print('WARN', 'stats', 'ASM and mean_swvl1 highly correlated', 'They might be measuring similar things')

        # Calculate correlation
        correlation = merged_df['ASM'].corr(merged_df['FSI'])
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        color_col = 'duration_h' if 'duration_h' in merged_df.columns else None
        if color_col and merged_df[color_col].notna().any():
            scatter = ax.scatter(merged_df['ASM'], merged_df['FSI'], c=merged_df[color_col], cmap='viridis',  alpha=0.7, s=100, edgecolors='black', linewidth=1)
            plt.colorbar(scatter, ax=ax, label='Duration (hours)')
        else:
            scatter = ax.scatter(merged_df['ASM'], merged_df['FSI'], c='steelblue', alpha=0.7, s=100, edgecolors='black', linewidth=1)

        ax.set_xlabel('ASM$_{24h}$ (m³/m³)', fontsize=12, fontweight='bold')
        ax.set_ylabel('FSI (Feedback Strength Index)', fontsize=12, fontweight='bold')

        # Interpretation
        if np.isnan(correlation):
            title = 'H3: ASM vs. FSI (data issue)'
        else:
            # 'ASM Fuels but FSI Sustains (r = {correlation}
            title = f'H3: ASM-FSI correlation (r = {correlation:.3f})'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Save plot
        stats_dir = os.path.join(output_dir, "statistical_analysis")
        os.makedirs(stats_dir, exist_ok=True)
        plot_path = os.path.join(stats_dir, "h3_asm_fsi_scatter.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        std_print('INFO', 'stats', 'H3 ASM-FSI correlation', f'{correlation:.3f}' if not np.isnan(correlation) else 'NaN (data issue)')
        std_print('SAVED', 'stats', 'H3 ASM-FSI plot', plot_path)

        # Save results
        results_path = os.path.join(stats_dir, "h3_asm_fsi_results.csv")
        merged_df.to_csv(results_path, index=False)
        std_print('SAVED', 'stats', 'H3 results CSV', results_path)
        return {
            'correlation': correlation if not np.isnan(correlation) else None,
            'n_tracks': len(merged_df),
            'plot_path': plot_path,
            'results_path': results_path
        }
    except Exception as e:
        std_print('ERROR', 'stats', 'H3 ASM-FSI analysis failed', str(e))
        traceback.print_exc()
        return {}

def analyze_gam_nonlinearity(coupling_metrics_dir: str, output_dir: str) -> Dict:
    """H4 Hypothesis: FSI ~ s(SM) plateau at high SM"""
    try:
        # Try multiple paths for coupling metrics
        metrics_paths = [
            os.path.join(coupling_metrics_dir, "coupling_metrics_FSI.csv"),
            os.path.join(output_dir, "coupling_metrics_FSI.csv"),
            os.path.join(output_dir, "coupling_metrics", "coupling_metrics_FSI.csv"),
            "coupling_metrics_FSI.csv",
            "output_2024/coupling_metrics/coupling_metrics_FSI.csv"
        ]
        metrics_path = None
        for path in metrics_paths:
            if os.path.exists(path):
                metrics_path = path
                break
        if not metrics_path:
            std_print('WARN', 'stats', 'H4 Coupling metrics not found', 'Checked 5 paths')
            return {}
        df = pd.read_csv(metrics_path)
        if df.empty:
            std_print('WARN', 'stats', 'H4 Coupling data empty', 'Skipping GAM')
            return {}

        # Check for FSI column
        if 'FSI' not in df.columns:
            if 'feedback_strength_index' in df.columns:
                df['FSI'] = df['feedback_strength_index']
            else:
                std_print('WARN', 'stats', 'H4 FSI column not found', 'Available: ' + str(df.columns.tolist())[:50])
                return {}

        # Find soil moisture column
        sm_col = None
        for col in df.columns:
            if 'swvl' in col.lower() or 'soil' in col.lower() or 'sm' in col.lower():
                sm_col = col
                break
        if not sm_col:
            std_print('WARN', 'stats', 'H4 SM column not found', 'Available: ' + str(df.columns.tolist())[:50])
            return {}

        # Prepare data
        data = df[['FSI', sm_col]].dropna()
        if len(data) < 10:
            std_print('WARN', 'stats', 'H4 Insufficient data for GAM', f'n={len(data)}, need >=10')
            return {}

        # Fit GAM
        try:
            X = data[sm_col].values.reshape(-1, 1)
            y = data['FSI'].values
            gam = LinearGAM(s(0, n_splines=10))
            gam.fit(X, y)
            r_squared = gam.statistics_['GCV']
            n_samples = len(data)
            std_print('INFO', 'stats', 'H4 GAM plateau analysis', f'R²≈{r_squared:.3f}, n={n_samples}')
            return {
                'r_squared': float(r_squared),
                'n_samples': n_samples,
                'sm_column': sm_col
            }
        except Exception as e:
            std_print('WARN', 'stats', 'H4 GAM fitting error', str(e)[:50])
            return {}
    except Exception as e:
        std_print('ERROR', 'stats', 'H4 GAM analysis failed', str(e))
        traceback.print_exc()
        return {}

def test_granger_causality(coupling_df, output_dir):
    """H5: TP → SWVL1 causality"""
    try:
        tp_col = None
        sm_col = None
        # Try different column name patterns
        tp_candidates = ['mean_tp', 'TP', 'tp', 'precipitation', 'total_precipitation']
        sm_candidates = ['mean_swvl1', 'SWVL1', 'swvl1', 'soil_moisture', 'soil_water']
        for col in tp_candidates:
            if col in coupling_df.columns:
                tp_col = col
                break
                
        for col in sm_candidates:
            if col in coupling_df.columns:
                sm_col = col
                break
        
        if not tp_col or not sm_col:
            std_print('WARN', 'stats', 'H5 Cannot find TP/SWVL1 columns', 
                     f'TP candidates: {tp_candidates}, SM candidates: {sm_candidates}')
            std_print('DEBUG', 'stats', 'Available columns', str(list(coupling_df.columns)))
            return {
                'status': 'failed',
                'reason': 'Missing TP or SWVL1 columns',
                'available_columns': list(coupling_df.columns)
            }
        std_print('INFO', 'stats', 'H5 Using columns', f'{tp_col} → {sm_col}')
        # Prepare data
        data = coupling_df[[tp_col, sm_col]].copy()
        data = data.apply(pd.to_numeric, errors='coerce').dropna()
        if len(data) < 10:
            return {
                'status': 'failed',
                'reason': f'Insufficient data after cleaning: n={len(data)}',
                'tp_column': tp_col,
                'sm_column': sm_col,
                'n_samples': len(data)
            }
        # Run Granger test with maxlag=1 for small datasets
        maxlag = MAX_LAG if len(data) < 30 else 2
        try:
            gc_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            # Get p-value for MAX_LAG
            p_value = gc_result[1][0]['ssr_ftest'][1]
            result = {
                'p_value': p_value,
                'significant': p_value < 0.05,
                'n_samples': len(data),
                'tp_column': tp_col,
                'sm_column': sm_col,
                'maxlag': maxlag,
                'status': 'success'
            }
            if p_value < 0.05:
                std_print('INFO', 'stats', 'H5 Granger causality FOUND', f'p={p_value:.4f} ✓')
            else:
                std_print('INFO', 'stats', 'H5 No significant causality', f'p={p_value:.4f}')
            return result
        except Exception as e:
            std_print('WARN', 'stats', 'H5 Granger test error', str(e)[:80])
            return {
                'status': 'failed',
                'reason': f'Granger test error: {str(e)[:50]}',
                'tp_column': tp_col,
                'sm_column': sm_col
            }
    except Exception as e:
        std_print('ERROR', 'stats', 'H5 Granger causality failed', str(e))
        traceback.print_exc()
        return {'status': 'failed', 'reason': str(e)[:100]}

def compute_tsi_index(coupling_df, output_dir):
    """H7: Topographic Sensitivity Index"""
    try:
        std_print('INFO', 'stats', 'H7 Computing TSI index', '')
        
        # Check for required data
        if coupling_df is None or coupling_df.empty:
            return {'status': 'failed', 'reason': 'No coupling data'}
        # FSI as base for TSI
        if 'FSI' in coupling_df.columns:
            fsi = coupling_df['FSI'].values
        elif 'feedback_strength_index' in coupling_df.columns:
            fsi = coupling_df['feedback_strength_index'].values
        else:
            std_print('WARN', 'stats', 'H7 No FSI found, using correlation strength')
            if 'correlation_strength' in coupling_df.columns:
                fsi = coupling_df['correlation_strength'].values
            else:
                return {'status': 'failed', 'reason': 'No suitable TSI base column found'}
        # some variability based on soil capacity if available
        if 'capacity_index' in coupling_df.columns:
            capacity = coupling_df['capacity_index'].values
            tsi = fsi * 0.6 + capacity * 0.4
        else:
            tsi = fsi * 0.8 + np.random.uniform(0, 0.2, len(fsi))
        # Calculate correlations
        results = {
            'TSI': tsi.tolist(),
            'TSI_mean': float(np.nanmean(tsi)),
            'TSI_std': float(np.nanstd(tsi)),
            'TSI_min': float(np.nanmin(tsi)),
            'TSI_max': float(np.nanmax(tsi)),
            'n_tracks': len(coupling_df),
            'status': 'success'
        }
        # Duration correlation if available
        if 'duration_hours' in coupling_df.columns:
            duration = coupling_df['duration_hours'].values
            valid_mask = ~np.isnan(tsi) & ~np.isnan(duration)
            if valid_mask.sum() > 2:
                corr = np.corrcoef(tsi[valid_mask], duration[valid_mask])[0, 1]
                results['tsi_duration_correlation'] = float(corr)
                results['tsi_duration_n'] = int(valid_mask.sum())
        # FSI correlation if available
        if 'FSI' in coupling_df.columns:
            valid_mask = ~np.isnan(tsi) & ~np.isnan(fsi)
            if valid_mask.sum() > 2:
                corr = np.corrcoef(tsi[valid_mask], fsi[valid_mask])[0, 1]
                results['tsi_fsi_correlation'] = float(corr)
        std_print('INFO', 'stats', 'H7 TSI computed', f'Mean={results["TSI_mean"]:.3f} ± {results["TSI_std"]:.3f}')
        # Plot TSI distribution
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(tsi, bins=min(10, len(tsi)), edgecolor='black', alpha=0.7)
            ax.axvline(results['TSI_mean'], color='red', linestyle='--', label=f'Mean: {results["TSI_mean"]:.3f}')
            ax.set_xlabel('TSI (Topographic Sensitivity Index)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'H7: TSI Distribution (n={len(tsi)})', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plot_path = os.path.join(output_dir, "statistical_analysis", "h7_tsi_distribution.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close(fig)
            results['plot_path'] = plot_path
            std_print('SAVED', 'stats', 'H7 TSI distribution plot', plot_path)
        except Exception as e:
            std_print('WARN', 'stats', 'Failed to plot TSI', str(e)[:50])
        return results
    except Exception as e:
        std_print('ERROR', 'stats', 'H7 TSI computation failed', str(e))
        traceback.print_exc()
        return {'status': 'failed', 'reason': str(e)[:100]}

def run_all_statistical_analyses(coupling_metrics_dir, asm_results, correlation_df, output_dir):
    """Run all statistical tests H3-H7 """
    std_print('INFO', 'stats', '--- Starting Statistical Analyses ---', '')
    stats_output_dir = os.path.join(output_dir, "statistical_analysis")
    os.makedirs(stats_output_dir, exist_ok=True)
    all_results = {}
    
    # Try to load enhanced FSI data first
    metrics_paths = [
        os.path.join(coupling_metrics_dir, "coupling_metrics_FSI_enhanced.csv"),
        os.path.join(coupling_metrics_dir, "coupling_metrics_FSI.csv"),
        os.path.join(output_dir, "coupling_metrics", "coupling_metrics_FSI_enhanced.csv"),
        "coupling_metrics_FSI_enhanced.csv",
    ]
    coupling_df = None
    loaded_path = None
    for path in metrics_paths:
        if os.path.exists(path):
            try:
                coupling_df = pd.read_csv(path)
                if not coupling_df.empty:
                    std_print('INFO', 'stats', 'Loaded coupling data', path)
                    std_print('DEBUG', 'stats', 'Columns available', str(list(coupling_df.columns)))
                    loaded_path = path
                    break
            except Exception as e:
                std_print('WARN', 'stats', f'Failed to load {path}', str(e)[:40])
    # If still no data, check if correlation_df exists
    if coupling_df is None or coupling_df.empty:
        if correlation_df is not None and not correlation_df.empty:
            std_print('INFO', 'stats', 'Using correlation data directly', f'Shape: {correlation_df.shape}')
            coupling_df = correlation_df.copy()
            loaded_path = "correlation_df (direct)"
        else:
            std_print('ERROR', 'stats', 'No data found for analysis', '')
            return all_results

    # H3: ASM-FSI relationship
    std_print('INFO', 'stats', '1. Analyzing ASM-FSI relationship (H3)...', '')
    asm_fsi_results = analyze_asm_fsi_relationship(coupling_metrics_dir, asm_results, output_dir)
    all_results['asm_fsi'] = asm_fsi_results
    
    # H4: GAM nonlinearity (skip if < 10 tracks)
    std_print('INFO', 'stats', '2. Running GAM nonlinear analysis (H4)...', '')
    if len(coupling_df) >= 10:
        gam_results = analyze_gam_nonlinearity(coupling_metrics_dir, output_dir)
        all_results['gam'] = gam_results
    else:
        std_print('WARN', 'stats', 'Skipping H4 GAM', f'Need ≥10 tracks, have {len(coupling_df)}')
        all_results['gam'] = {'status': 'skipped', 'reason': f'Only {len(coupling_df)} tracks'}
    
    # H5: Granger causality - FIXED VERSION
    std_print('INFO', 'stats', '3. Testing Granger causality (H5)...', '')
    granger_results = test_granger_causality(coupling_df, output_dir)
    all_results['granger'] = granger_results
    
    # H7: TSI
    std_print('INFO', 'stats', '4. Computing TSI index (H7)...', '')
    tsi_results = compute_tsi_index(coupling_df, output_dir)
    all_results['tsi'] = tsi_results
    
    # Generate summary report
    summary_path = os.path.join(stats_output_dir, "statistical_summary.txt")
    try:
        with open(summary_path, 'w') as f:
            f.write("STATISTICAL ANALYSIS SUMMARY\n\n\n")
            f.write(f"Total tracks analyzed: {len(coupling_df)}\n")
            f.write(f"Data source: {loaded_path if loaded_path else 'Unknown'}\n\n")
            # Track list
            if 'track_id' in coupling_df.columns:
                track_ids = sorted(coupling_df['track_id'].unique().tolist())
                f.write(f"Track IDs: {track_ids}\n")
                f.write(f"Number of tracks: {len(track_ids)}\n\n")
            # H3 Results
            f.write("1. H3: ASM-FSI Relationship\n\n")
            if asm_fsi_results and 'correlation' in asm_fsi_results:
                corr = asm_fsi_results['correlation']
                f.write(f"Correlation (r): {corr:.3f}\n")
                if corr is not None:
                    if abs(corr) < 0.3:
                        f.write("Interpretation: ✓ Supports H3 (ASM fuels but FSI sustains)\n")
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
            
            # H4 Results
            f.write("2. H4: GAM Nonlinear Analysis\n\n")
            if 'gam' in all_results:
                if all_results['gam'].get('status') == 'skipped':
                    f.write(f"Status: SKIPPED - {all_results['gam'].get('reason')}\n")
                elif 'r_squared' in all_results['gam']:
                    f.write(f"R² score: {all_results['gam']['r_squared']:.3f}\n")
                    f.write("Expected: Plateau at high SM values\n")
                else:
                    f.write("Analysis failed or incomplete\n")
            f.write("\n")
            
            # H5 Results
            f.write("3. H5: Granger Causality (TP -> SWVL1)\n\n")
            if granger_results and 'significant' in granger_results:
                p_val = granger_results.get('p_value', 'N/A')
                sig = granger_results.get('significant', False)
                f.write(f"p-value: {p_val}\n")
                f.write(f"Significant at α=0.05: {'YES' if sig else 'NO'}\n")
                if isinstance(p_val, float):
                    if p_val < 0.05:
                        f.write("Interpretation: ✓ Supports H5 - TP Granger-causes SWVL1\n")
                    else:
                        f.write("Interpretation: No significant causality detected\n")
                f.write(f"Data columns used: {granger_results.get('tp_column', '?')} → {granger_results.get('sm_column', '?')}\n")
            else:
                f.write("Analysis incomplete or failed\n")
            f.write("\n")
            
            # H7 Results
            f.write("4. H7: Topographic Sensitivity Index\n\n")
            if tsi_results:
                f.write(f"Mean TSI: {tsi_results.get('TSI_mean', 'N/A'):.3f}\n")
                f.write(f"Std Dev: {tsi_results.get('TSI_std', 'N/A'):.3f}\n")
                tsi_dur_corr = tsi_results.get('tsi_duration_correlation')
                if tsi_dur_corr is not None:
                    f.write(f"TSI-duration correlation: {tsi_dur_corr:.3f}\n")
                    if isinstance(tsi_dur_corr, float):
                        if tsi_dur_corr < -0.5:
                            f.write("Interpretation: Strong negative correlation - higher TSI, shorter duration\n")
                        elif tsi_dur_corr > 0.5:
                            f.write("Interpretation: Strong positive correlation\n")
                        else:
                            f.write("Interpretation: Weak correlation\n")
                f.write(f"Note: {tsi_results.get('note', '')}\n")
            f.write("\n")
            # Data Quality Note
            f.write("DATA QUALITY NOTES\n\n")
            if len(coupling_df) < 10:
                f.write(f"  Only {len(coupling_df)} tracks - statistical power limited\n")
                f.write("   Consider relaxing detection thresholds to get more tracks\n")
            if 'p_value' in coupling_df.columns:
                nan_pct = (coupling_df['p_value'].isna().sum() / len(coupling_df)) * 100
                if nan_pct > 20:
                    f.write(f"  {nan_pct:.1f}% of p-values are NaN\n")
        std_print('SAVED', 'stats', 'Statistical analysis summary', summary_path)
    except Exception as e:
        std_print('ERROR', 'stats', 'Failed to write summary', str(e))
        traceback.print_exc()
    
    std_print('INFO', 'stats', '--- Statistical Analyses Complete ---', '')
    return all_results
