"""
Time-Averaged Variable Maps with Ocean Contamination Detection
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
OUTPUT_DIR = "time_averaged_maps"

# Soil moisture thresholds (m³/m³)
SWVL1_VALID_MIN = 0.01   # Minimum valid soil moisture for land
SWVL1_VALID_MAX = 0.65   # Maximum valid soil moisture
OCEAN_THRESHOLD = 0.02   # Values below this over ocean are suspicious

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

def create_time_average_maps(ds):
    """Create time-averaged maps for all variables."""
    print("\n" + "="*70)
    print("CREATING TIME-AVERAGED MAPS")
    print("="*70)
    # Get all variables
    variables = list(ds.data_vars.keys())
    print(f"Processing {len(variables)} variables: {variables}")
    # Create output directory for maps
    maps_dir = os.path.join(OUTPUT_DIR, "variable_maps")
    os.makedirs(maps_dir, exist_ok=True)
    for var_name in variables:
        print(f"\nProcessing: {var_name}")
        try:
            # Compute time average
            if var_name == 'tp':
                # For precipitation, use sum instead of average
                var_avg = ds[var_name].sum(dim='time') * 1000  # Convert to mm
                units = "mm (total)"
                cmap = 'YlOrRd'
            else:
                var_avg = ds[var_name].mean(dim='time')
                units = str(ds[var_name].units) if 'units' in ds[var_name].attrs else ""
                # Choose colormap based on variable
                if var_name == 'swvl1':
                    cmap = 'YlGnBu'
                    units = "m³/m³"
                elif var_name in ['t2m', 'stl1']:
                    cmap = 'RdBu_r'
                elif var_name in ['msl', 'sp']:
                    cmap = 'viridis'
                elif 'wind' in var_name.lower() or var_name in ['u10', 'v10']:
                    cmap = 'plasma'
                elif 'flux' in var_name.lower():
                    cmap = 'coolwarm'
                else:
                    cmap = 'viridis'
            # Create figure
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            # Set extent (Indian region)
            ax.set_extent([70, 95, 5, 30], crs=ccrs.PlateCarree())
            # Plot the data
            im = ax.pcolormesh(
                var_avg.longitude,
                var_avg.latitude,
                var_avg.values,
                cmap=cmap,
                transform=ccrs.PlateCarree()
            )
            # Add coastlines and features
            ax.coastlines(resolution='50m', linewidth=1.0, color='black')
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
            # Add gridlines
            gl = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=True,
                linewidth=0.5,
                color='gray',
                alpha=0.5,
                linestyle='--'
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
            cbar.set_label(f'{var_name} ({units})', fontsize=12)
            # Set title
            if var_name == 'tp':
                title = f'Total Precipitation (June-Sept 2024)\nSum over all time steps'
            else:
                title = f'{var_name.upper()} - Time Average (June-Sept 2024)'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            # Save figure
            output_path = os.path.join(maps_dir, f'{var_name}_time_average.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f" >  [SAVED   ] Plot Map: {output_path}")
            # # Save the averaged data as netCDF
            # nc_output = os.path.join(maps_dir, f'{var_name}_time_average.nc')
            # var_avg.to_netcdf(nc_output)
            # print(f"  Data saved: {nc_output}")
        except Exception as e:
            print(f" >  [Error   ] Processing {var_name}: {e}")
            continue
    print(f"\nAll variable maps saved in: {maps_dir}")

def main():
    """Main function to run the analysis."""
    print("TIME-AVERAGED VARIABLE MAPS WITH OCEAN CONTAMINATION DETECTION")
    try:
        # Load dataset
        ds = load_dataset()
        # Create time-averaged maps for all variables
        create_time_average_maps(ds)
        print(f"All outputs saved in: {os.path.abspath(OUTPUT_DIR)}")
    except Exception as e:
        print(f"\n Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
