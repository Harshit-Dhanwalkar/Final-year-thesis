"""
Time-Averaged and Diagnostic Variable Maps with Ocean Contamination Detection
Special focus on soil moisture (swvl1) over ocean detection
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../')

# Configuration
DATA_DIR = "../Data/single-levels/ncfiles"
MONTH_DIRS = ["June-2024", "July-2024", "August-2024", "September-2024"]
OUTPUT_DIR = "variable_diagnostic_maps"

# Soil moisture thresholds (m³/m³)
SWVL1_VALID_MIN = 0.01   # Minimum valid soil moisture for land
SWVL1_VALID_MAX = 0.65   # Maximum valid soil moisture
OCEAN_THRESHOLD = 0.02   # Values above this over ocean are suspicious

def load_dataset():
    """Load and combine all monthly netCDF files."""
    print("Loading dataset...")
    file_list = []
    for month_dir in MONTH_DIRS:
        month_path = os.path.join(DATA_DIR, month_dir)
        if os.path.isdir(month_path):
            month_files = [os.path.join(month_path, f) 
                          for f in os.listdir(month_path) if f.endswith(".nc")]
            file_list.extend(month_files)
            print(f"  Found {len(month_files)} files in {month_dir}")
    if not file_list:
        raise FileNotFoundError("No netCDF files found!")
    print(f"Total files to load: {len(file_list)}")
    # Load with chunking to avoid memory issues
    ds = xr.open_mfdataset(
        file_list,
        combine="by_coords",
        chunks={'time': 24, 'latitude': 100, 'longitude': 100}
    )
    # Rename time coordinate if needed
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    ds = ds.sortby('time')
    print(f"Dataset loaded successfully!")
    print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
    print(f"Variables: {list(ds.data_vars.keys())}")
    print(f"Shape: {dict(ds.dims)}")
    return ds

def detect_ocean_contamination(ds):
    """Detect and analyze soil moisture data over ocean areas."""
    print("\n" + "="*70)
    print("DETECTING SOIL MOISTURE OVER OCEAN")
    print("="*70)
    
    if 'swvl1' not in ds.data_vars:
        print("Warning: 'swvl1' not found in dataset")
        return None
    
    # Create mask for ocean areas (approximate - you might want a better land/sea mask)
    # For now, we'll use a simple approach: where swvl1 should be zero or very low
    
    # Calculate statistics
    swvl1_data = ds['swvl1']
    
    # Maximum value at each grid point over time
    swvl1_max = swvl1_data.max(dim='time')
    
    # Count occurrences where swvl1 > threshold (indicating possible contamination)
    threshold = 0.02  # m³/m³ - anything above this over ocean is suspicious
    contamination_mask = swvl1_data > threshold
    contamination_count = contamination_mask.sum(dim='time')
    
    # Percentage of time with contamination
    total_times = len(ds.time)
    contamination_percent = (contamination_count / total_times) * 100
    
    # Create diagnostic plots
    os.makedirs(os.path.join(OUTPUT_DIR, "ocean_contamination"), exist_ok=True)
    
    # 1. Plot maximum swvl1 values
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([70, 95, 5, 30], crs=ccrs.PlateCarree())
    
    im = ax.pcolormesh(
        swvl1_max.longitude,
        swvl1_max.latitude,
        swvl1_max.values,
        cmap='YlOrRd',
        vmin=0,
        vmax=0.1,  # Focus on low values for ocean contamination
        transform=ccrs.PlateCarree()
    )
    
    ax.coastlines(resolution='50m', linewidth=1.0, color='black')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, 
                     color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Maximum swvl1 (m³/m³)', fontsize=12)
    
    ax.set_title('Maximum Soil Moisture Values (June-Sept 2024)\nValues > 0.02 over ocean indicate contamination',
                 fontsize=14, fontweight='bold', pad=15)
    
    output_path = os.path.join(OUTPUT_DIR, "ocean_contamination", "swvl1_maximum.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" >  [SAVED   ] Maximum swvl1 plot: {output_path}")
    
    # 2. Plot percentage of time with contamination
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([70, 95, 5, 30], crs=ccrs.PlateCarree())
    
    im = ax.pcolormesh(
        contamination_percent.longitude,
        contamination_percent.latitude,
        contamination_percent.values,
        cmap='hot_r',
        vmin=0,
        vmax=100,
        transform=ccrs.PlateCarree()
    )
    
    ax.coastlines(resolution='50m', linewidth=1.0, color='black')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, 
                     color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Percentage of time with swvl1 > 0.02 m³/m³', fontsize=12)
    
    ax.set_title('Soil Moisture Contamination Frequency (June-Sept 2024)\nHigher values indicate persistent ocean contamination',
                 fontsize=14, fontweight='bold', pad=15)
    
    output_path = os.path.join(OUTPUT_DIR, "ocean_contamination", "swvl1_contamination_frequency.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" >  [SAVED   ] Contamination frequency plot: {output_path}")
    
    # 3. Create histogram of swvl1 values
    # Focus on ocean areas - let's assume ocean is where maximum swvl1 < 0.1
    ocean_mask = swvl1_max < 0.1
    ocean_values = swvl1_data.where(ocean_mask).values.flatten()
    ocean_values = ocean_values[~np.isnan(ocean_values)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram for ocean areas
    ax1.hist(ocean_values[ocean_values > 0], bins=50, range=(0, 0.1), 
             color='blue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('swvl1 (m³/m³)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Soil Moisture Values over Ocean Areas\n(where max swvl1 < 0.1)')
    ax1.axvline(x=OCEAN_THRESHOLD, color='red', linestyle='--', 
                label=f'Threshold ({OCEAN_THRESHOLD} m³/m³)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot for different value ranges
    bins = [0, 0.01, 0.02, 0.05, 0.1, 1.0]
    labels = ['0-0.01', '0.01-0.02', '0.02-0.05', '0.05-0.1', '>0.1']
    binned_data = []
    for i in range(len(bins)-1):
        mask = (ocean_values >= bins[i]) & (ocean_values < bins[i+1])
        if np.any(mask):
            binned_data.append(ocean_values[mask])
        else:
            binned_data.append(np.array([]))
    
    ax2.boxplot(binned_data, labels=labels, showfliers=False)
    ax2.set_xlabel('swvl1 range (m³/m³)')
    ax2.set_ylabel('swvl1 value (m³/m³)')
    ax2.set_title('Distribution of Soil Moisture Values over Ocean')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.1)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "ocean_contamination", "swvl1_histograms.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" >  [SAVED   ] Histogram plots: {output_path}")
    
    # Print summary statistics
    print("\nSoil Moisture Contamination Summary:")
    print(f"  Threshold for contamination: > {OCEAN_THRESHOLD} m³/m³")
    print(f"  Grid points with any contamination: {np.sum(contamination_count.values > 0)}")
    print(f"  Maximum contamination value: {swvl1_max.max().values:.4f} m³/m³")
    print(f"  Mean contamination percentage: {contamination_percent.mean().values:.1f}%")
    print(f"  Max contamination percentage: {contamination_percent.max().values:.1f}%")
    
    return swvl1_max, contamination_percent

def create_raw_time_series_maps(ds):
    """Create maps showing raw time series characteristics (not averaged)."""
    print("\n" + "="*70)
    print("CREATING RAW TIME SERIES CHARACTERISTIC MAPS")
    print("="*70)
    
    variables = list(ds.data_vars.keys())
    print(f"Processing {len(variables)} variables")
    
    # Create output directory
    raw_dir = os.path.join(OUTPUT_DIR, "raw_characteristics")
    os.makedirs(raw_dir, exist_ok=True)
    
    for var_name in variables:
        print(f"\nProcessing: {var_name}")
        try:
            var_data = ds[var_name]
            
            # Calculate various characteristics (not averages!)
            stats = {
                'Maximum': var_data.max(dim='time'),
                'Minimum': var_data.min(dim='time'),
                'Range': var_data.max(dim='time') - var_data.min(dim='time'),
                'Std Dev': var_data.std(dim='time'),
                '90th Percentile': var_data.quantile(0.9, dim='time'),
                '10th Percentile': var_data.quantile(0.1, dim='time'),
            }
            
            # For precipitation, also calculate total
            if var_name == 'tp':
                stats['Total'] = var_data.sum(dim='time') * 1000  # Convert to mm
            
            # Create plots for each statistic
            for stat_name, stat_data in stats.items():
                fig = plt.figure(figsize=(14, 8))
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                ax.set_extent([70, 95, 5, 30], crs=ccrs.PlateCarree())
                
                # Choose appropriate colormap and range
                if var_name == 'swvl1':
                    cmap = 'YlOrRd'
                    vmin = stat_data.min().values if 'min' not in stat_name.lower() else None
                    vmax = stat_data.max().values if 'max' not in stat_name.lower() else None
                elif var_name == 'tp':
                    cmap = 'YlOrRd'
                elif var_name in ['t2m', 'stl1']:
                    cmap = 'RdBu_r'
                else:
                    cmap = 'viridis'
                
                im = ax.pcolormesh(
                    stat_data.longitude,
                    stat_data.latitude,
                    stat_data.values,
                    cmap=cmap,
                    transform=ccrs.PlateCarree()
                )
                
                ax.coastlines(resolution='50m', linewidth=1.0, color='black')
                ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
                ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
                
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                                 color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = False
                gl.right_labels = False
                gl.xformatter = LongitudeFormatter()
                gl.yformatter = LatitudeFormatter()
                
                cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
                
                # Get units
                units = ""
                if var_name == 'tp' and stat_name == 'Total':
                    units = "mm (total)"
                elif var_name == 'tp':
                    units = "m/hr"
                elif 'units' in ds[var_name].attrs:
                    units = str(ds[var_name].units)
                
                cbar.set_label(f'{var_name} {stat_name} ({units})', fontsize=12)
                
                title = f'{var_name.upper()} - {stat_name} (June-Sept 2024)'
                if var_name == 'tp' and stat_name == 'Total':
                    title = f'Total Precipitation - Sum over all time steps (June-Sept 2024)'
                
                ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
                
                # Save figure
                safe_stat_name = stat_name.replace(' ', '_').replace('/', '_')
                output_path = os.path.join(raw_dir, f'{var_name}_{safe_stat_name}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                print(f" >  [SAVED   ] {stat_name} plot: {output_path}")
                
        except Exception as e:
            print(f" >  [Error   ] Processing {var_name}: {e}")
            continue
    
    print(f"\nAll raw characteristic maps saved in: {raw_dir}")

def create_time_slice_plots(ds, variable='swvl1'):
    """Create plots for specific time slices to see raw values."""
    print(f"\n" + "="*70)
    print(f"CREATING TIME SLICE PLOTS FOR {variable.upper()}")
    print("="*70)
    
    if variable not in ds.data_vars:
        print(f"Warning: {variable} not found in dataset")
        return
    
    # Create output directory
    slice_dir = os.path.join(OUTPUT_DIR, "time_slices", variable)
    os.makedirs(slice_dir, exist_ok=True)
    
    # Get time information
    times = pd.to_datetime(ds.time.values)
    print(f"Total time steps: {len(times)}")
    
    # Convert to pandas Series for easier manipulation
    times_series = pd.Series(times)
    
    # Select specific time slices (e.g., one per month, and some random ones)
    time_indices = []
    
    # One from each month
    for month in [6, 7, 8, 9]:  # June to September
        month_mask = times_series.dt.month == month
        if np.any(month_mask):
            # Get indices for this month
            month_indices = np.where(month_mask)[0]
            # Take the middle index of the month
            mid_idx = len(month_indices) // 2
            time_indices.append(month_indices[mid_idx])
    
    # Add first and last time step
    time_indices.append(0)  # First
    time_indices.append(len(times) - 1)  # Last
    
    # Add some evenly spaced time slices
    n_slices = 6  # Number of evenly spaced slices
    step = max(1, len(times) // n_slices)
    even_indices = list(range(0, len(times), step))[:n_slices]
    time_indices.extend(even_indices)
    
    # Remove duplicates and sort
    time_indices = sorted(set(time_indices))
    
    print(f"Creating plots for {len(time_indices)} time slices")
    
    for i, idx in enumerate(time_indices):
        try:
            if idx >= len(times):
                print(f" >  [Warning ] Index {idx} out of range, skipping")
                continue
                
            time_slice = ds.isel(time=idx)
            time_str = str(times[idx])
            
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([70, 95, 5, 30], crs=ccrs.PlateCarree())
            
            data = time_slice[variable]
            
            # For swvl1, use a special colormap to highlight ocean contamination
            if variable == 'swvl1':
                cmap = plt.cm.YlOrRd
                # Set vmin to 0 to better see low values
                vmin = 0
                vmax = max(0.1, data.max().values * 1.1)
            else:
                cmap = 'viridis'
                vmin = None
                vmax = None
            
            im = ax.pcolormesh(
                data.longitude,
                data.latitude,
                data.values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree()
            )
            
            ax.coastlines(resolution='50m', linewidth=1.0, color='black')
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
            
            # Highlight grid points with potential ocean contamination
            if variable == 'swvl1':
                # Mark points with swvl1 > threshold
                contamination_mask = data.values > OCEAN_THRESHOLD
                contamination_points = np.where(contamination_mask)
                if len(contamination_points[0]) > 0:
                    # Get lat/lon coordinates
                    lons, lats = np.meshgrid(data.longitude.values, data.latitude.values)
                    ax.scatter(lons[contamination_points], lats[contamination_points],
                              s=15, color='red', alpha=0.7, transform=ccrs.PlateCarree(),
                              label=f'swvl1 > {OCEAN_THRESHOLD} m³/m³', 
                              marker='x', linewidths=1)
                    ax.legend(loc='upper right', fontsize=10)
            
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                             color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
            
            # Get units
            units = ""
            if variable == 'tp':
                units = "m/hr"
            elif 'units' in ds[variable].attrs:
                units = str(ds[variable].units)
            
            cbar.set_label(f'{variable} ({units})', fontsize=12)
            
            title = f'{variable.upper()} - Time Slice {i+1}\n{time_str}'
            if variable == 'swvl1':
                title += f'\nRed X: values > {OCEAN_THRESHOLD} m³/m³ (potential ocean contamination)'
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            
            # Save figure
            safe_time_str = time_str.replace(' ', '_').replace(':', '-').replace('.', '-')
            output_path = os.path.join(slice_dir, f'{variable}_slice_{i+1}_{safe_time_str}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f" >  [SAVED   ] Time slice {i+1}: {output_path}")
            
            # Print contamination statistics for this time slice
            if variable == 'swvl1':
                contamination_count = np.sum(contamination_mask)
                total_points = np.prod(data.shape)
                percent_contaminated = (contamination_count / total_points) * 100
                max_value = data.max().values
                mean_value = data.mean().values
                
                print(f"    Statistics: mean={mean_value:.4f}, max={max_value:.4f} m³/m³")
                if contamination_count > 0:
                    print(f"    Contamination: {contamination_count}/{total_points} points "
                          f"({percent_contaminated:.1f}%) > {OCEAN_THRESHOLD} m³/m³")
            
        except Exception as e:
            print(f" >  [Error   ] Processing time slice {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

def main():
    """Main function to run the analysis."""
    print("VARIABLE DIAGNOSTIC MAPS WITH OCEAN CONTAMINATION DETECTION")
    print("="*70)
    
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load dataset
        ds = load_dataset()
        
        # 1. Detect ocean contamination in soil moisture
        contamination_results = detect_ocean_contamination(ds)
        
        # 2. Create raw time series characteristic maps (not averages)
        create_raw_time_series_maps(ds)
        
        # 3. Create time slice plots for specific variables
        # Focus on swvl1 for contamination detection
        create_time_slice_plots(ds, variable='swvl1')
        
        # Optional: Create time slices for other variables too
        for var in ['tp', 't2m', 'msl']:
            if var in ds.data_vars:
                create_time_slice_plots(ds, variable=var)
        
        print(f"\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"All outputs saved in: {os.path.abspath(OUTPUT_DIR)}")
        print("\nSummary of outputs:")
        print("  1. ocean_contamination/ - Soil moisture contamination analysis")
        print("  2. raw_characteristics/ - Maps of raw statistics (max, min, std, etc.)")
        print("  3. time_slices/ - Individual time step plots")
        
    except Exception as e:
        print(f"\n Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
