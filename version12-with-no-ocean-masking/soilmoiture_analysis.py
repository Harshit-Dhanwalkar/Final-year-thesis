"""
Analyzes the relationship between TP and SWVL1 during entire MD track passage.

Key Questions:
- Is there a correlation between TP and SWVL1?
- Does more precipitation lead to higher soil moisture?
- How strong is the linear relationship?
- Spatial patterns of both variables during passage?
"""

# TODO: add shade region for land (like in latent heat flux transiton and in speed plots)
# FIX: use buffer_km instaed of buffer_deg

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr, spearmanr, linregress
from statsmodels.tsa.stattools import acf

from utils import (
    std_print,
    extract_regional_data,
    extract_spatial_subset,
    setup_map_extent,
    clamp_to_domain,
    get_time_coord
)

def plot_spatial_tp_swvl1(ds, track_df, track_id, output_dir='output', buffer_deg=5.0):
# def plot_spatial_tp_swvl1(ds, track_df, track_id, output_dir='output', buffer_km=100):
    """2-panel spatial maps of TP and SWVL1 during track passage."""
    track_id = int(track_id)
    track = track_df.sort_values('time')

    min_lon, max_lon, min_lat, max_lat = setup_map_extent(track, buffer_deg=buffer_deg)
    min_lat, max_lat, min_lon, max_lon = clamp_to_domain(min_lat, max_lat, min_lon, max_lon)
    ds, time_coord = get_time_coord(ds)

    # Get averaged fields
    tp_data = ds['tp'].sel({
        time_coord: slice(track['time'].min(), track['time'].max()),
        'latitude': slice(max_lat, min_lat),
        'longitude': slice(min_lon, max_lon)
    }).sum(dim=time_coord) * 1000  # Total, converted to mm

    sm_data = ds['swvl1'].sel({
        time_coord: slice(track['time'].min(), track['time'].max()),
        'latitude': slice(max_lat, min_lat),
        'longitude': slice(min_lon, max_lon)
    }).mean(dim=time_coord)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7,), subplot_kw={'projection': ccrs.PlateCarree()})

    # Panel 1: Total Precipitation
    im1 = axes[0].contourf(tp_data.longitude, tp_data.latitude, tp_data.values, levels=20, cmap='YlOrRd', transform=ccrs.PlateCarree())
    contours1 = axes[0].contour(tp_data.longitude, tp_data.latitude, tp_data.values, levels=10, colors='black', linewidths=0.5, alpha=0.4, transform=ccrs.PlateCarree())
    axes[0].clabel(contours1, inline=True, fontsize=8, fmt='%1.0f')
    axes[0].plot(track['longitude'], track['latitude'], 'b-', linewidth=2, transform=ccrs.PlateCarree(), label='Track', zorder=5)
    axes[0].coastlines(resolution='50m', linewidth=1)
    axes[0].add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    axes[0].set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    axes[0].set_title('Total Precipitation (mm)', fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0], pad=0.02)
    cbar1.set_label('Total Precip (mm)', fontsize=10)
    axes[0].legend(loc='upper left')

    # Panel 2: Mean Soil Moisture
    im2 = axes[1].contourf(sm_data.longitude, sm_data.latitude, sm_data.values, levels=20, cmap='YlGnBu', transform=ccrs.PlateCarree())
    contours2 = axes[1].contour(sm_data.longitude, sm_data.latitude, sm_data.values, levels=10, colors='black', linewidths=0.5, alpha=0.4, transform=ccrs.PlateCarree())
    axes[1].clabel(contours2, inline=True, fontsize=8, fmt='%1.3f')
    axes[1].plot(track['longitude'], track['latitude'], 'r-', linewidth=2, transform=ccrs.PlateCarree(), label='Track', zorder=5)
    axes[1].coastlines(resolution='50m', linewidth=1)
    axes[1].add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    axes[1].set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    axes[1].set_title('Mean Soil Moisture (m³/m³)', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[1], pad=0.02)
    cbar2.set_label('Soil Moisture', fontsize=10)
    axes[1].legend(loc='upper left')

    fig.suptitle(f'Track {track_id} Spatial Distribution of TP and SWVL1', fontsize=13, fontweight='bold')
    plt.tight_layout()

    spactial_tp_sm_dir = os.path.join(output_dir, 'spatial_tp_sm')
    os.makedirs(spactial_tp_sm_dir, exist_ok=True)
    outpath = os.path.join(spactial_tp_sm_dir, f'track_{int(track_id)}_spatial_tp_sm.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'sm', f'Spatial TP-SWVL1 plot for Track {track_id}', outpath)

def compute_soil_moisture_memory(swvl1_series, lag_hours=24):
    """Compute autocorrelation of SWVL1 at specified lag.
    Higher autocorrelation = longer memory = more sustained feedback.
    """
    # Remove NaN values
    swvl1_clean = np.array(swvl1_series)
    swvl1_clean = swvl1_clean[~np.isnan(swvl1_clean)]
    if len(swvl1_clean) < lag_hours * 2:
        return np.nan
    try:
        # Compute autocorrelation function
        acf_vals = acf(swvl1_clean, nlags=lag_hours, fft=True)
        # Return autocorrelation at specified lag
        if lag_hours < len(acf_vals):
            memory_index = acf_vals[lag_hours]
            return memory_index
        else:
            return np.nan
    except:
        return np.nan

def analyze_soil_memory_by_track(track_df_dict, ds, output_dir):
    """Analyze soil moisture memory effect for each track"""
    memory_results = []

    for track_id, track_df in track_df_dict.items():
        swvl1_vals = []
        ds, time_coord = get_time_coord(ds)
        for _, row in track_df.iterrows():
            try:
                subset = extract_spatial_subset(ds, row['latitude'], row['longitude'], pd.to_datetime(row['time']), buffer_km=100, time_coord=time_coord)
                if subset is not None and 'swvl1' in subset.data_vars:
                    sm = float(subset['swvl1'].mean().values)
                    if not np.isnan(sm):
                        swvl1_vals.append(sm)
            except:
                continue
        if len(swvl1_vals) > 48:  # Need enough data for 24h lag
            memory_24h = compute_soil_moisture_memory(swvl1_vals, lag_hours=24)
            memory_results.append({
                'track_id': track_id,
                'memory_index_24h': memory_24h,
                'n_samples': len(swvl1_vals),
                'mean_swvl1': np.mean(swvl1_vals),
                'std_swvl1': np.std(swvl1_vals)
            })
    # Export results
    if memory_results:
        df = pd.DataFrame(memory_results)
        df.to_csv(os.path.join(output_dir, 'soil_memory_analysis.csv'), index=False)
        std_print('EXPORT', 'sm', 'Soil memory analysis to CSV', 'soil_memory_analysis.csv')
    return memory_results
