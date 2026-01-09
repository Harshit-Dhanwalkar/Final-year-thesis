# correlation_analysis.py - Functional Version
"""
Functional module for analyzing correlation between soil moisture (swvl1) 
and total precipitation (tp) for monsoon depression tracks.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

from speed_analysis import is_point_land


def extract_soil_moisture_precipitation_data(ds, track_df, buffer_km=100):
    """
    Extract swvl1 and tp values along the track with spatial buffer.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        ERA5 dataset with swvl1 and tp variables
    track_df : pandas.DataFrame
        Track data with time, latitude, longitude
    buffer_km : float
        Radius around track points to extract data (km)
    
    Returns:
    --------
    pandas.DataFrame with extracted data
    """
    print(f"Extracting soil moisture and precipitation data along track (buffer: {buffer_km}km)...")
    
    data_records = []
    
    for idx, row in track_df.iterrows():
        time = pd.to_datetime(row['time'])
        center_lat = row['latitude']
        center_lon = row['longitude']
        
        # Convert buffer to degrees (approximate)
        buffer_deg = buffer_km / 111.0  # ~1 degree = 111 km
        
        # Define spatial bounds
        lat_min = center_lat - buffer_deg
        lat_max = center_lat + buffer_deg
        lon_min = center_lon - buffer_deg
        lon_max = center_lon + buffer_deg
        
        # Ensure bounds within dataset
        lat_min = max(lat_min, ds.latitude.min().item())
        lat_max = min(lat_max, ds.latitude.max().item())
        lon_min = max(lon_min, ds.longitude.min().item())
        lon_max = min(lon_max, ds.longitude.max().item())
        
        try:
            # Extract data slice
            ds_slice = ds.sel(
                time=time,
                latitude=slice(lat_max, lat_min),  # Note: lat_max first (descending)
                longitude=slice(lon_min, lon_max)
            )
            
            # Calculate mean values in the buffer region
            swvl1_mean = float(ds_slice['swvl1'].mean().values)
            tp_mean = float(ds_slice['tp'].mean().values) * 1000  # Convert m/hr to mm/hr
            
            # Check if point is over land
            is_land_point = is_point_land(center_lat, center_lon)
            
            data_records.append({
                'time': time,
                'latitude': center_lat,
                'longitude': center_lon,
                'swvl1': swvl1_mean,
                'tp_mm_hr': tp_mean,
                'is_land': is_land_point,
                'stage': 'ocean' if not is_land_point else 'land',
                'buffer_km': buffer_km
            })
            
        except Exception as e:
            print(f"Warning: Could not extract data at time {time}: {e}")
            continue
    
    extracted_df = pd.DataFrame(data_records)
    print(f"Extracted {len(extracted_df)} data points")
    
    return extracted_df


def calculate_correlation_metrics(df):
    """
    Calculate various correlation metrics between swvl1 and tp.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with 'swvl1' and 'tp_mm_hr' columns
    
    Returns:
    --------
    Dictionary with correlation results
    """
    # Remove NaN values
    df_clean = df.dropna(subset=['swvl1', 'tp_mm_hr'])
    
    if len(df_clean) < 2:
        print("Insufficient data for correlation analysis")
        return None
    
    # Calculate Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(df_clean['swvl1'], df_clean['tp_mm_hr'])
    
    # Calculate Spearman correlation (non-parametric)
    spearman_r, spearman_p = stats.spearmanr(df_clean['swvl1'], df_clean['tp_mm_hr'])
    
    # Calculate Kendall's tau
    kendall_tau, kendall_p = stats.kendalltau(df_clean['swvl1'], df_clean['tp_mm_hr'])
    
    # Linear regression
    X = df_clean['swvl1'].values.reshape(-1, 1)
    y = df_clean['tp_mm_hr'].values
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r_squared = reg.score(X, y)
    
    results = {
        'n_samples': len(df_clean),
        'pearson': {
            'r': pearson_r,
            'p': pearson_p,
            'significant': pearson_p < 0.05
        },
        'spearman': {
            'r': spearman_r,
            'p': spearman_p,
            'significant': spearman_p < 0.05
        },
        'kendall': {
            'tau': kendall_tau,
            'p': kendall_p,
            'significant': kendall_p < 0.05
        },
        'regression': {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        },
        'statistics': {
            'swvl1_mean': df_clean['swvl1'].mean(),
            'swvl1_std': df_clean['swvl1'].std(),
            'tp_mean': df_clean['tp_mm_hr'].mean(),
            'tp_std': df_clean['tp_mm_hr'].std()
        }
    }
    
    # Print results
    print_correlation_results(results)
    
    return results


def print_correlation_results(results):
    """Print correlation results in a readable format."""
    if not results:
        print("No correlation results to display")
        return
    
    print("\n" + "="*60)
    print("SOIL MOISTURE (swvl1) vs PRECIPITATION (tp) CORRELATION")
    print("="*60)
    
    print(f"\nSample Size: {results['n_samples']}")
    
    print(f"\nSoil Moisture Statistics:")
    print(f"  Mean: {results['statistics']['swvl1_mean']:.4f} m³/m³")
    print(f"  Std:  {results['statistics']['swvl1_std']:.4f} m³/m³")
    
    print(f"\nPrecipitation Statistics:")
    print(f"  Mean: {results['statistics']['tp_mean']:.3f} mm/hr")
    print(f"  Std:  {results['statistics']['tp_std']:.3f} mm/hr")
    
    print(f"\nCorrelation Coefficients:")
    print(f"  Pearson r:  {results['pearson']['r']:.3f} "
          f"(p={results['pearson']['p']:.4f})")
    print(f"  Spearman ρ: {results['spearman']['r']:.3f} "
          f"(p={results['spearman']['p']:.4f})")
    print(f"  Kendall τ:  {results['kendall']['tau']:.3f} "
          f"(p={results['kendall']['p']:.4f})")
    
    print(f"\nLinear Regression:")
    print(f"  Slope: {results['regression']['slope']:.3f}")
    print(f"  Intercept: {results['regression']['intercept']:.3f}")
    print(f"  R²: {results['regression']['r_squared']:.3f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    r = results['pearson']['r']
    if abs(r) < 0.3:
        strength = "weak"
    elif abs(r) < 0.5:
        strength = "moderate"
    elif abs(r) < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if r > 0 else "negative"
    significance = "statistically significant" if results['pearson']['p'] < 0.05 else "not statistically significant"
    
    print(f"  There is a {strength} {direction} correlation between")
    print(f"  soil moisture and precipitation ({significance}).")


def plot_scatter_correlation(df, results, output_dir, track_id=None, save=True):
    """
    Create scatter plot with correlation line.
    """
    df_clean = df.dropna(subset=['swvl1', 'tp_mm_hr'])
    
    if len(df_clean) < 2:
        print("Insufficient data for plotting")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot with regression line
    ax = axes[0]
    scatter = ax.scatter(df_clean['swvl1'], df_clean['tp_mm_hr'], 
                       alpha=0.7, s=50, edgecolor='black', linewidth=0.5)
    
    # Add regression line
    if results and 'regression' in results:
        x_range = np.linspace(df_clean['swvl1'].min(), df_clean['swvl1'].max(), 100)
        y_pred = results['regression']['slope'] * x_range + results['regression']['intercept']
        ax.plot(x_range, y_pred, 'r-', linewidth=2, 
               label=f"y = {results['regression']['slope']:.3f}x + {results['regression']['intercept']:.3f}")
    
    ax.set_xlabel('Soil Moisture (swvl1, m³/m³)', fontsize=12)
    ax.set_ylabel('Precipitation Rate (tp, mm/hr)', fontsize=12)
    
    if track_id:
        ax.set_title(f'Track {track_id}: Soil Moisture vs Precipitation', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Soil Moisture vs Precipitation', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    
    # Add correlation info as text box
    if results and 'pearson' in results:
        textstr = '\n'.join([
            f"Pearson r = {results['pearson']['r']:.3f}",
            f"p = {results['pearson']['p']:.4f}",
            f"n = {results['n_samples']}",
            f"R² = {results['regression']['r_squared']:.3f}"
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
    
    # Time series of both variables
    ax2 = axes[1]
    time_series = df_clean.sort_values('time')
    
    # Plot soil moisture
    color1 = 'tab:blue'
    ax2.plot(time_series['time'], time_series['swvl1'], 
            color=color1, marker='o', linewidth=2, label='Soil Moisture')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Soil Moisture (m³/m³)', color=color1, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color1)
    
    # Plot precipitation on second y-axis
    ax3 = ax2.twinx()
    color2 = 'tab:red'
    ax3.plot(time_series['time'], time_series['tp_mm_hr'], 
            color=color2, marker='s', linewidth=2, linestyle='--', label='Precipitation')
    ax3.set_ylabel('Precipitation (mm/hr)', color=color2, fontsize=12)
    ax3.tick_params(axis='y', labelcolor=color2)
    
    ax2.set_title('Temporal Evolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    if save:
        if track_id:
            filename = os.path.join(output_dir, f"track_{int(track_id)}_correlation.png")
        else:
            filename = os.path.join(output_dir, "soil_moisture_precipitation_correlation.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved correlation plot: {filename}")
    
    # plt.show()
    return fig, axes


def plot_correlation_by_stage(df, output_dir, track_id=None, save=True):
    """
    Plot correlation separately for ocean and land stages.
    """
    df_clean = df.dropna(subset=['swvl1', 'tp_mm_hr'])
    
    if len(df_clean) < 2:
        print("Insufficient data for stage analysis")
        return None
    
    # Classify stages
    if 'stage' not in df_clean.columns:
        df_clean['stage'] = df_clean.apply(
            lambda row: 'ocean' if not row['is_land'] else 'land', 
            axis=1
        )
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Colors for stages
    stage_colors = {'ocean': 'blue', 'land': 'green'}
    
    # Plot 1: Combined with color-coded stages
    ax1 = axes[0]
    for stage, color in stage_colors.items():
        stage_data = df_clean[df_clean['stage'] == stage]
        if len(stage_data) > 0:
            ax1.scatter(stage_data['swvl1'], stage_data['tp_mm_hr'],
                      color=color, alpha=0.7, s=50, edgecolor='black',
                      linewidth=0.5, label=f'{stage.capitalize()} (n={len(stage_data)})')
    
    ax1.set_xlabel('Soil Moisture (m³/m³)', fontsize=12)
    ax1.set_ylabel('Precipitation (mm/hr)', fontsize=12)
    
    if track_id:
        ax1.set_title(f'Track {track_id}: Correlation by Stage', fontsize=14, fontweight='bold')
    else:
        ax1.set_title('Correlation by Stage', fontsize=14, fontweight='bold')
    
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    
    # Plot 2: Box plots comparing soil moisture distributions
    ax2 = axes[1]
    boxplot_data = []
    box_labels = []
    for stage in ['ocean', 'land']:
        stage_data = df_clean[df_clean['stage'] == stage]['swvl1']
        if len(stage_data) > 0:
            boxplot_data.append(stage_data.values)
            box_labels.append(stage.capitalize())
    
    if len(boxplot_data) > 0:
        bp = ax2.boxplot(boxplot_data, labels=box_labels, patch_artist=True)
        
        # Color boxes
        colors = [stage_colors.get(stage.lower(), 'gray') for stage in box_labels]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Soil Moisture (m³/m³)', fontsize=12)
        ax2.set_title('Soil Moisture Distribution by Stage', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Box plots comparing precipitation distributions
    ax3 = axes[2]
    boxplot_data_tp = []
    for stage in ['ocean', 'land']:
        stage_data = df_clean[df_clean['stage'] == stage]['tp_mm_hr']
        if len(stage_data) > 0:
            boxplot_data_tp.append(stage_data.values)
    
    if len(boxplot_data_tp) > 0:
        bp = ax3.boxplot(boxplot_data_tp, labels=box_labels, patch_artist=True)
        
        # Color boxes
        colors = [stage_colors.get(stage.lower(), 'gray') for stage in box_labels]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Precipitation (mm/hr)', fontsize=12)
        ax3.set_title('Precipitation Distribution by Stage', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save:
        if track_id:
            filename = os.path.join(output_dir, f"track_{int(track_id)}_correlation_by_stage.png")
        else:
            filename = os.path.join(output_dir, "correlation_by_stage.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved stage analysis plot: {filename}")

    # plt.show()
    return fig, axes


def plot_2d_heatmap(df, results, output_dir, track_id=None, save=True):
    """
    Create 2D histogram/heatmap of the relationship.
    """
    df_clean = df.dropna(subset=['swvl1', 'tp_mm_hr'])
    
    if len(df_clean) < 2:
        print("Insufficient data for heatmap")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create 2D histogram
    hb = ax.hexbin(df_clean['swvl1'], df_clean['tp_mm_hr'], 
                  gridsize=20, cmap='YlOrRd', bins='log')
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Count (log scale)', fontsize=12)
    
    # Add regression line
    if results and 'regression' in results:
        x_range = np.linspace(df_clean['swvl1'].min(), df_clean['swvl1'].max(), 100)
        y_pred = results['regression']['slope'] * x_range + results['regression']['intercept']
        ax.plot(x_range, y_pred, 'b-', linewidth=3, 
               label=f"Regression line (R²={results['regression']['r_squared']:.3f})")
    
    ax.set_xlabel('Soil Moisture (swvl1, m³/m³)', fontsize=14)
    ax.set_ylabel('Precipitation Rate (tp, mm/hr)', fontsize=14)
    
    if track_id:
        ax.set_title(f'Track {track_id}: 2D Density - Soil Moisture vs Precipitation', 
                    fontsize=16, fontweight='bold')
    else:
        ax.set_title('2D Density: Soil Moisture vs Precipitation', 
                    fontsize=16, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save:
        if track_id:
            filename = os.path.join(output_dir, f"track_{int(track_id)}_2d_heatmap.png")
        else:
            filename = os.path.join(output_dir, "correlation_heatmap_2d.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved 2D heatmap: {filename}")
    
    # plt.show()
    return fig, ax


def analyze_lag_correlation(df, max_lag_hours=24, output_dir=None, track_id=None, save=True):
    """
    Analyze correlation with time lags (soil moisture preceding precipitation).
    """
    df_clean = df.dropna(subset=['swvl1', 'tp_mm_hr']).sort_values('time')
    
    if len(df_clean) < max_lag_hours + 2:
        print(f"Insufficient data for lag analysis (need at least {max_lag_hours + 2} points)")
        return {}
    
    # Resample to regular intervals if needed
    df_time = df_clean.set_index('time')
    
    # Only resample the numeric columns needed for lag analysis
    df_numeric = df_time[['swvl1', 'tp_mm_hr']]  # Only keep numeric columns
    df_resampled = df_numeric.resample('1H').mean().interpolate()
    
    swvl1_series = df_resampled['swvl1'].values
    tp_series = df_resampled['tp_mm_hr'].values
    
    # Calculate lagged correlations
    lags = range(0, max_lag_hours + 1)
    correlations = []
    p_values = []
    
    for lag in lags:
        if lag == 0:
            swvl1_lagged = swvl1_series
            tp_original = tp_series
        else:
            swvl1_lagged = swvl1_series[:-lag]  # Soil moisture (lagged)
            tp_original = tp_series[lag:]       # Precipitation (later)
        
        # Ensure equal lengths
        min_len = min(len(swvl1_lagged), len(tp_original))
        swvl1_lagged = swvl1_lagged[:min_len]
        tp_original = tp_original[:min_len]
        
        if min_len > 1:
            r, p = stats.pearsonr(swvl1_lagged, tp_original)
            correlations.append(r)
            p_values.append(p)
        else:
            correlations.append(np.nan)
            p_values.append(np.nan)
    
    # Find maximum correlation lag
    valid_corrs = [c for c in correlations if not np.isnan(c)]
    if valid_corrs:
        max_corr = max(valid_corrs, key=abs)
        max_lag = lags[correlations.index(max_corr)]
    else:
        max_corr = np.nan
        max_lag = np.nan
    
    # Plot lag correlation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(lags, correlations, 'bo-', linewidth=2, markersize=8, label='Correlation')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=max_lag, color='red', linestyle='--', alpha=0.7, 
              label=f'Max correlation at lag {max_lag}h')
    
    # Add significance markers
    for lag, corr, p in zip(lags, correlations, p_values):
        if p < 0.05 and not np.isnan(corr):
            ax.plot(lag, corr, 'r*', markersize=12)
    
    ax.set_xlabel('Lag (hours)', fontsize=12)
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
    
    if track_id:
        ax.set_title(f'Track {track_id}: Lag Correlation (Soil Moisture → Precipitation)\n'
                    f'Max correlation: {max_corr:.3f} at lag {max_lag}h', 
                    fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Lag Correlation: Soil Moisture → Precipitation\n'
                    f'Max correlation: {max_corr:.3f} at lag {max_lag}h', 
                    fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save and output_dir:
        if track_id:
            filename = os.path.join(output_dir, f"track_{int(track_id)}_lag_correlation.png")
        else:
            filename = os.path.join(output_dir, "lag_correlation_analysis.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved lag correlation plot: {filename}")
    
    # plt.show()
    lag_results = {
        'lags': list(lags),
        'correlations': correlations,
        'p_values': p_values,
        'max_correlation': max_corr,
        'max_correlation_lag': max_lag
    }
    
    return lag_results


def plot_correlation_hovmoller(hov_precip, hov_soil, radii_km, track_times, outdir, track_id=None):
    """
    Create correlation Hovmöller plot comparing soil moisture and precipitation.
    """
    # Normalize data (Z-score) so they are on the same scale
    z_precip = (hov_precip - np.mean(hov_precip)) / np.std(hov_precip)
    z_soil = (hov_soil - np.mean(hov_soil)) / np.std(hov_soil)
    
    # The difference shows where one leads the other
    diff = z_soil - z_precip
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create contour plot
    im = ax.contourf(radii_km, np.arange(len(track_times)), diff, 
                    cmap='RdBu_r', levels=20, extend='both')
    
    plt.colorbar(im, label='Soil Moisture Lead (positive) / Precip Lead (negative) (Z-score)')
    plt.xlabel("Radius (km)", fontsize=12)
    plt.ylabel("Time Index", fontsize=12)
    
    if track_id:
        plt.title(f"Track {track_id}: Z-Score Difference - Soil Moisture vs Precipitation", 
                 fontsize=14, fontweight='bold')
    else:
        plt.title("Z-Score Difference: Soil Moisture vs Precipitation", 
                 fontsize=14, fontweight='bold')
    
    if track_id:
        filename = os.path.join(outdir, f"track_{int(track_id)}_hovmoller_correlation.png")
    else:
        filename = os.path.join(outdir, "hovmoller_correlation_spatial.png")
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved Hovmöller correlation plot: {filename}")
    
    plt.close()
    
    return fig, ax


def save_analysis_results(df, results, output_dir, track_id=None):
    """Save all extracted data and results to CSV files."""
    # Save extracted data
    if track_id:
        data_file = os.path.join(output_dir, f"track_{int(track_id)}_soil_moisture_precipitation.csv")
    else:
        data_file = os.path.join(output_dir, "extracted_soil_moisture_precipitation.csv")
    
    df.to_csv(data_file, index=False)
    print(f"Saved extracted data: {data_file}")
    
    # Save correlation results
    if results:
        if track_id:
            results_file = os.path.join(output_dir, f"track_{int(track_id)}_correlation_results.txt")
        else:
            results_file = os.path.join(output_dir, "correlation_results_summary.txt")
        
        with open(results_file, 'w') as f:
            f.write("SOIL MOISTURE vs PRECIPITATION CORRELATION ANALYSIS\n")
            f.write("="*60 + "\n\n")
            
            for key, value in results.items():
                if isinstance(value, dict):
                    f.write(f"\n{key.upper()}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"Saved results summary: {results_file}")


def analyze_single_track(ds, track_df, track_id, buffer_km=100, max_lag_hours=24):
    """
    Complete correlation analysis for a single track.
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING CORRELATION FOR TRACK {track_id}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = f"output/track_{int(track_id)}/correlation_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract data
    extracted_data = extract_soil_moisture_precipitation_data(ds, track_df, buffer_km)
    
    if len(extracted_data) < 2:
        print(f"Not enough data for Track {track_id}")
        return None, None
    
    # Step 2: Calculate correlations
    results = calculate_correlation_metrics(extracted_data)
    
    if not results:
        print(f"No valid correlation results for Track {track_id}")
        return extracted_data, None
    
    # Step 3: Create plots
    plot_scatter_correlation(extracted_data, results, output_dir, track_id, save=True)
    plot_correlation_by_stage(extracted_data, output_dir, track_id, save=True)
    plot_2d_heatmap(extracted_data, results, output_dir, track_id, save=True)
    
    # Step 4: Analyze lag correlation
    lag_results = analyze_lag_correlation(extracted_data, max_lag_hours, output_dir, track_id, save=True)
    
    # Step 5: Save results
    save_analysis_results(extracted_data, results, output_dir, track_id)
    
    # Combine all results
    if lag_results:
        results['lag_analysis'] = lag_results
    
    print(f"\nTrack {track_id} analysis complete!")
    
    return extracted_data, results


def analyze_all_tracks(ds, full_track_df, buffer_km=100, max_lag_hours=24):
    """
    Analyze correlations for all tracks and create summary.
    """
    print("\n" + "="*60)
    print("ANALYZING SOIL MOISTURE-PRECIPITATION CORRELATION FOR ALL TRACKS")
    print("="*60)
    
    all_results = {}
    all_extracted_data = {}
    
    track_ids = sorted(full_track_df['track_id'].unique())
    
    for track_id in track_ids:
        print(f"\nAnalyzing Track {track_id}...")
        print("-"*40)
        
        # Extract track data
        track_data = full_track_df[full_track_df['track_id'] == track_id].copy()
        
        try:
            # Run analysis for this track
            extracted_data, results = analyze_single_track(
                ds, track_data, track_id, buffer_km, max_lag_hours
            )
            
            if results:  # Only store if we got results
                all_results[track_id] = {
                    'pearson_r': results['pearson']['r'],
                    'pearson_p': results['pearson']['p'],
                    'n_samples': results['n_samples'],
                    'swvl1_mean': results['statistics']['swvl1_mean'],
                    'tp_mean': results['statistics']['tp_mean'],
                    'r_squared': results['regression']['r_squared']
                }
                
                all_extracted_data[track_id] = extracted_data
                
                print(f"Track {track_id}: r = {results['pearson']['r']:.3f} (n={results['n_samples']})")
        
        except Exception as e:
            print(f"Error analyzing track {track_id}: {e}")
            continue
    
    # Create summary plot across all tracks
    if all_results:
        create_summary_plot(all_results)
    
    return all_extracted_data, all_results


def create_summary_plot(all_results):
    """Create summary plot of correlations across all tracks."""
    if not all_results:
        return
    
    # Create output directory for summary
    summary_dir = "output/correlation_analysis_summary"
    os.makedirs(summary_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prepare data
    track_ids = list(all_results.keys())
    pearson_rs = [all_results[tid]['pearson_r'] for tid in track_ids]
    n_samples = [all_results[tid]['n_samples'] for tid in track_ids]
    swvl1_means = [all_results[tid]['swvl1_mean'] for tid in track_ids]
    tp_means = [all_results[tid]['tp_mean'] for tid in track_ids]
    r_squared = [all_results[tid]['r_squared'] for tid in track_ids]
    
    # Plot 1: Correlation coefficients
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(track_ids)), pearson_rs, color='steelblue', alpha=0.7)
    
    # Color bars by significance
    for i, tid in enumerate(track_ids):
        if all_results[tid]['pearson_p'] < 0.05:
            bars[i].set_color('green')
    
    ax1.set_xlabel('Track ID', fontsize=12)
    ax1.set_ylabel('Pearson Correlation Coefficient (r)', fontsize=12)
    ax1.set_title('Correlation by Track', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(track_ids)))
    ax1.set_xticklabels([f'Track {tid}' for tid in track_ids], rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add sample sizes
    for i, n in enumerate(n_samples):
        ax1.text(i, pearson_rs[i] + (0.02 if pearson_rs[i] >= 0 else -0.02), 
                f'n={n}', ha='center', va='bottom' if pearson_rs[i] >= 0 else 'top', fontsize=9)
    
    # Plot 2: Scatter of mean values
    ax2 = axes[0, 1]
    scatter = ax2.scatter(swvl1_means, tp_means, s=100, c=pearson_rs, 
                         cmap='RdYlBu', edgecolor='black', linewidth=1, alpha=0.8)
    
    # Add track labels
    for i, tid in enumerate(track_ids):
        ax2.annotate(f'{tid}', (swvl1_means[i], tp_means[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Mean Soil Moisture (m³/m³)', fontsize=12)
    ax2.set_ylabel('Mean Precipitation (mm/hr)', fontsize=12)
    ax2.set_title('Track Mean Values', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Correlation (r)', fontsize=12)
    
    # Plot 3: Sample size vs correlation
    ax3 = axes[1, 0]
    ax3.scatter(n_samples, pearson_rs, s=100, color='purple', alpha=0.7, edgecolor='black')
    
    for i, tid in enumerate(track_ids):
        ax3.annotate(f'{tid}', (n_samples[i], pearson_rs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Sample Size (n)', fontsize=12)
    ax3.set_ylabel('Correlation (r)', fontsize=12)
    ax3.set_title('Sample Size vs Correlation Strength', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of correlations
    ax4 = axes[1, 1]
    ax4.hist(pearson_rs, bins=10, color='orange', alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(pearson_rs), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(pearson_rs):.3f}')
    ax4.axvline(x=np.median(pearson_rs), color='blue', linestyle='--', linewidth=2,
               label=f'Median: {np.median(pearson_rs):.3f}')
    
    ax4.set_xlabel('Correlation Coefficient (r)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Correlation Coefficients', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=10)
    
    plt.suptitle('Summary: Soil Moisture-Precipitation Correlation Across All Tracks', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save summary plot
    filename = os.path.join(summary_dir, "all_tracks_summary.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved summary plot: {filename}")
    
    # plt.show()
    # Save summary data
    summary_df = pd.DataFrame([
        {
            'track_id': tid,
            'pearson_r': all_results[tid]['pearson_r'],
            'pearson_p': all_results[tid]['pearson_p'],
            'significant': all_results[tid]['pearson_p'] < 0.05,
            'n_samples': all_results[tid]['n_samples'],
            'swvl1_mean': all_results[tid]['swvl1_mean'],
            'tp_mean': all_results[tid]['tp_mean'],
            'r_squared': all_results[tid]['r_squared']
        }
        for tid in track_ids
    ])
    
    csv_file = os.path.join(summary_dir, "all_tracks_correlation_summary.csv")
    summary_df.to_csv(csv_file, index=False)
    print(f"Saved summary CSV: {csv_file}")
    
    return fig, axes


# Example usage function
def run_example_analysis():
    """Example of how to use the correlation analysis functions."""
    print("Example: Soil Moisture-Precipitation Correlation Analysis")
    print("="*60)
    
    # This would be replaced with your actual data loading
    # For example:
    # ds = xr.open_dataset('your_era5_data.nc')
    # track_df = pd.read_csv('your_track_data.csv')
    
    print("\nTo use this module, import the functions and call:")
    print("""
# For a single track:
extracted_data, results = analyze_single_track(ds, track_df, track_id=1)

# For all tracks:
all_data, all_results = analyze_all_tracks(ds, full_track_df)

# Individual functions:
data = extract_soil_moisture_precipitation_data(ds, track_df, buffer_km=100)
results = calculate_correlation_metrics(data)
plot_scatter_correlation(data, results, 'output_directory')
    """)
    
    print("\nSee function docstrings for detailed parameter information.")


if __name__ == "__main__":
    # Show example usage
    run_example_analysis()
