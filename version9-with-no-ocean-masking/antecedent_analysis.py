# This script computes soil moisture conditions in the 24 hours BEFORE landfall.
# Used to understand the moisture state of the land when the MD arrives.
# Is the antecedent soil moisture (wet vs dry) related to rainfall/flooding intensity during MD passage?

# Dry antecedent conditions -> Soil can absorb a lot of the incoming rain -> lower immediate flood risk (more infiltration).
# Wet antecedent conditions -> Soil is already near saturation → much more of the MD's heavy rainfall becomes surface runoff -> higher flood potential even if the total rainfall amount is not extreme.

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr
from datetime import datetime, timedelta

import traceback

from utils import (
    std_print,
    is_point_land,
    find_landfall_time,
    find_origin_type,
    extract_spatial_subset,
    get_time_coord,
    calculate_buffer_deg,
    clamp_to_domain,
    export_results_to_csv
)

# Helper function
def compute_antecedent_soil_moisture(ds, track_df, track_id, hours_before=24):
    track_id = int(track_id)
    # Get track data
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track.empty:
        std_print('ERROR', 'as', f'Track {track_id} not found')
        return np.nan

    # Get system start time (first point)
    start_time = pd.to_datetime(track.iloc[0]['time'])
    antecedent_start = start_time - pd.Timedelta(hours=hours_before)

    # Define spatial domain: 200km buffer around future track
    lat_min, lat_max = track['latitude'].min(), track['latitude'].max()
    lon_min, lon_max = track['longitude'].min(), track['longitude'].max()

    # buffer (approximately 2° = ~220km at equator)
    lat_buffer = 2.0
    lon_buffer = 2.0
    lat_min_buf = max(10.0, lat_min - lat_buffer)
    lat_max_buf = min(30.0, lat_max + lat_buffer)
    lon_min_buf = max(70.0, lon_min - lon_buffer)
    lon_max_buf = min(95.0, lon_max + lon_buffer)

    # Extract soil moisture data BEFORE system
    ds, time_coord = get_time_coord(ds)

    try:
        # Get temporal subset
        ds_antecedent = ds.sel({
            time_coord: slice(antecedent_start, start_time),
            'latitude': slice(lat_max_buf, lat_min_buf),  # ERA5: lat descending
            'longitude': slice(lon_min_buf, lon_max_buf)
        })

        if ds_antecedent[time_coord].size == 0:
            std_print('WARN', 'as', f'Track {track_id}', 'No temporal data in antecedent period')
            return np.nan

        if 'swvl1' not in ds_antecedent.data_vars:
            std_print('ERROR', 'as', f'Track {track_id}', 'SWVL1 not in dataset')
            return np.nan

        # Compute spatial and temporal average
        sm_mean = float(ds_antecedent['swvl1'].mean().values)

        std_print('INFO', 'as', f'Track {track_id} Antecedent SM ({hours_before}h before)')
        std_print(None, None, 'Period', f'{antecedent_start} to {start_time}')
        std_print(None, None, 'Mean SM', f'{sm_mean:.4f} m³/m³')
        std_print(None, None, 'Spatial domain', f'Lat: [{lat_min_buf:.1f}, {lat_max_buf:.1f}], Lon: [{lon_min_buf:.1f}, {lon_max_buf:.1f}]')
        return sm_mean
    except Exception as e:
        std_print('ERROR', 'as', f'Track {track_id} Antecedent computation failed', str(e))
        return np.nan

# Helper function
def classify_soil_moisture(sm_value):
    """Classify soil moisture level"""
    if np.isnan(sm_value):
        return 'No Data'
    elif sm_value < 0.1:
        return 'Very Dry'
    elif sm_value < 0.2:
        return 'Dry'
    elif sm_value < 0.3:
        return 'Moderate'
    elif sm_value < 0.4:
        return 'Wet'
    else:
        return 'Very Wet'


def plot_antecedent_spatial_map(ds, track_df, track_id, hours_before=24, output_dir='output'):
    track_id = int(track_id)
    # Get track data
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track.empty:
        std_print('ERROR', 'as', f'Track {track_id} not found')
        return False

    # Get system start time
    system_start = pd.to_datetime(track.iloc[0]['time'])
    antecedent_time = system_start - pd.Timedelta(hours=hours_before)

    # Define spatial extent with buffer
    lat_center = track['latitude'].mean()
    lon_center = track['longitude'].mean()
    lat_range = max(5.0, track['latitude'].max() - track['latitude'].min()) * 1.5
    lon_range = max(5.0, track['longitude'].max() - track['longitude'].min()) * 1.5

    min_lat = max(10.0, lat_center - lat_range/2)
    max_lat = min(30.0, lat_center + lat_range/2)
    min_lon = max(70.0, lon_center - lon_range/2)
    max_lon = min(95.0, lon_center + lon_range/2)

    ds, time_coord = get_time_coord(ds)
    try:
        # Get soil moisture at antecedent time
        ds_at_time = ds.sel({time_coord: antecedent_time}, method='nearest')
        sm_data = ds_at_time['swvl1'].sel(
            latitude=slice(max_lat, min_lat),
            longitude=slice(min_lon, max_lon)
        )
        if sm_data.size == 0:
            std_print('WARN', 'as', f'Track {track_id}', 'No SM data at antecedent time')
            return False

        # Figure
        fig = plt.figure(figsize=(13, 10))
        ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # type: ignore[assignment]

        ax.coastlines(resolution='50m', linewidth=1.0)
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.LAKES, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=6)
        ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, zorder=6)
        ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle='--')
        ax.add_feature(cfeature.RIVERS)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

        # Plot soil moisture
        im = ax.contourf(
            sm_data.longitude,
            sm_data.latitude,
            sm_data.values,
            levels=21,
            cmap='YlGnBu',
            transform=ccrs.PlateCarree(),
            extend='both'
        )

        # Contours
        contours = ax.contour(
            sm_data.longitude,
            sm_data.latitude,
            sm_data.values,
            levels=10,
            colors='black',
            linewidths=0.5,
            alpha=0.5,
            transform=ccrs.PlateCarree()
        )
        ax.clabel(contours, inline=True, fontsize=8, fmt='%1.2f')

        # Plot FUTURE track
        ax.plot(
            track['longitude'],
            track['latitude'],
            'r-',
            linewidth=3,
            transform=ccrs.PlateCarree(),
            label='Future MD Track',
            zorder=5
        )

        if len(track) > 0:
            # MD position at antecedent time (not yet formed), mark where it WILL form
            ax.scatter(
                track.iloc[0]['longitude'],
                track.iloc[0]['latitude'],
                s=300,
                c='gold',
                marker='s',
                edgecolors='black',
                linewidth=2,
                transform=ccrs.PlateCarree(),
                label='Future MD Genesis',
                zorder=6
            )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label('Soil Moisture (m³/m³)', fontsize=12, fontweight='bold')
        # Title
        title = f'Track {track_id}: Antecedent Soil Moisture\n{hours_before}h BEFORE System Formation\n({antecedent_time.strftime("%Y-%m-%d %H:%M")})'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        # Legend
        ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'track_{track_id}_antecedent_spatial_map.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        std_print('SAVED', 'as', 'Antecedent spatial map', plot_path)
        return True

    except Exception as e:
        std_print('ERROR', 'as', f'Track {track_id} Spatial map failed', str(e))
        traceback.print_exc()
        return False

def analyze_antecedent_for_all_tracks(ds, track_df, hours_before=24, output_dir='output'):
    std_print('INFO', 'as', 'Starting antecedent analysis for all tracks')
    unique_tracks = sorted(track_df['track_id'].unique())
    results = []
    for track_id in unique_tracks:
        std_print('INFO', 'as', f'Processing Track {track_id}')
        # Compute antecedent soil moisture
        asm = compute_antecedent_soil_moisture(ds, track_df, track_id, hours_before)
        # Create spatial map
        plot_antecedent_spatial_map(ds, track_df, track_id, hours_before, output_dir)
        # Get track characteristics
        track = track_df[track_df['track_id'] == track_id].sort_values('time')
        duration_hours = len(track)
        origin_type = find_origin_type(track)
        # Classify as wet/dry
        sm_class = classify_soil_moisture(asm)
        results.append({
            'track_id': track_id,
            'antecedent_sm': asm,
            'duration_hours': duration_hours,
            'origin_type': origin_type,
            'sm_class': sm_class,
            'system_start': pd.to_datetime(track.iloc[0]['time']),
            'n_points': len(track)
        })

        std_print('INFO', 'as', f'Track {track_id} Summary')
        std_print(None, None, 'Antecedent SM', f'{asm:.4f} m³/m³ ({sm_class})')
        std_print(None, None, 'Duration', f'{duration_hours} hours')
        std_print(None, None, 'Origin', origin_type)

    results_df = pd.DataFrame(results)

    # Analyze correlations
    analyze_antecedent_correlations(results_df, output_dir)
    create_antecedent_summary_report(results_df, output_dir)
    export_path = os.path.join(output_dir, 'antecedent_analysis_results.csv')
    results_df.to_csv(export_path, index=False)
    std_print('SAVED', 'as', 'Antecedent results CSV', export_path)
    return results_df

def analyze_antecedent_correlations(results_df, output_dir):
    """Analyze correlations between antecedent SM and track characteristics"""
    std_print('INFO', 'as', 'Analyzing antecedent correlations')

    # Filter tracks with valid antecedent data
    valid_df = results_df[results_df['antecedent_sm'].notna()].copy()
    if len(valid_df) < 3:
        std_print('WARN', 'as', 'Insufficient data for correlation analysis')
        return
    # 1. Correlation with duration
    if len(valid_df) >= 3:
        r_duration, p_duration = pearsonr(valid_df['antecedent_sm'], valid_df['duration_hours'])
        std_print('INFO', 'as', 'Antecedent SM vs Duration')
        std_print(None, None, 'Pearson r', f'{r_duration:.3f}')
        std_print(None, None, 'p-value', f'{p_duration:.4f}')
        if p_duration < 0.05:
            if r_duration > 0:
                interpretation = 'Wetter soils → Longer MD duration'
            else:
                interpretation = 'Drier soils → Longer MD duration'
        else:
            interpretation = 'No significant correlation'
        std_print(None, None, 'Interpretation', interpretation)

    # 2. Compare ocean vs land origin
    if 'origin_type' in valid_df.columns:
        ocean_tracks = valid_df[valid_df['origin_type'] == 'ocean_originated']
        land_tracks = valid_df[valid_df['origin_type'] == 'land_originated']
        std_print('INFO', 'as', 'Antecedent SM by Origin Type')
        if not ocean_tracks.empty:
            std_print(None, None, 'Ocean-origin mean SM', f'{ocean_tracks["antecedent_sm"].mean():.4f} m³/m³')
            std_print(None, None, 'Ocean-origin count', len(ocean_tracks))
        if not land_tracks.empty:
            std_print(None, None, 'Land-origin mean SM', f'{land_tracks["antecedent_sm"].mean():.4f} m³/m³')
            std_print(None, None, 'Land-origin count', len(land_tracks))

    # 3. Scatter plot
    create_antecedent_scatter_plot(valid_df, output_dir)

# Helper function
def create_antecedent_scatter_plot(results_df, output_dir):
    """Create scatter plot of antecedent SM vs duration"""
    if len(results_df) < 3:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by origin type
    colors = {'ocean_originated': 'blue', 'land_originated': 'green'}
    for origin, color in colors.items():
        subset = results_df[results_df['origin_type'] == origin]
        if not subset.empty:
            ax.scatter(
                subset['antecedent_sm'],
                subset['duration_hours'],
                c=color,
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5,
                label=origin.replace('_', ' ').title()
            )

    # regression line if significant correlation
    if len(results_df) >= 3:
        try:
            r, p = pearsonr(results_df['antecedent_sm'], results_df['duration_hours'])
            if p < 0.05:
                z = np.polyfit(results_df['antecedent_sm'], results_df['duration_hours'], 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(results_df['antecedent_sm'].min(), results_df['antecedent_sm'].max(), 100)
                ax.plot(x_line, p_fit(x_line), 'r--', linewidth=2, alpha=0.7,
                       label=f'r = {r:.3f}, p = {p:.4f}')
        except:
            pass

    ax.set_xlabel('Antecedent Soil Moisture (m³/m³)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Track Duration (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Antecedent Soil Moisture vs. MD Duration', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, framealpha=0.9)

    summary_text = f"Tracks analyzed: {len(results_df)}\n"
    summary_text += f"Mean antecedent SM: {results_df['antecedent_sm'].mean():.4f} m³/m³\n"
    summary_text += f"Mean duration: {results_df['duration_hours'].mean():.0f} hours"

    common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)
    ax.text(1.02, 0.98, summary_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=common_bbox
    )
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'antecedent_sm_vs_duration.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    std_print('SAVED', 'as', 'Antecedent scatter plot', plot_path)

def analyze_antecedent_trend(ds, track_df, track_id, output_dir='output'):
    """Analyze how soil moisture changes in the 48 hours BEFORE system formation"""
    track_id = int(track_id)
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track.empty:
        return np.nan

    system_start = pd.to_datetime(track.iloc[0]['time'])

    # Define time points: 48h, 36h, 24h, 12h, 0h before system
    time_points = [48, 36, 24, 12, 0]
    sm_values = []
    times = []
    ds, time_coord = get_time_coord(ds)

    for hours_before in time_points:
        antecedent_time = system_start - pd.Timedelta(hours=hours_before)
        # Spatial domain around future track
        lat_center = track['latitude'].mean()
        lon_center = track['longitude'].mean()
        try:
            ds_at_time = ds.sel({time_coord: antecedent_time}, method='nearest')
            # Extract region (1° buffer)
            lat_min = max(10.0, lat_center - 1.0)
            lat_max = min(30.0, lat_center + 1.0)
            lon_min = max(70.0, lon_center - 1.0)
            lon_max = min(95.0, lon_center + 1.0)
            sm_region = ds_at_time['swvl1'].sel(
                latitude=slice(lat_max, lat_min),
                longitude=slice(lon_min, lon_max)
            )
            if sm_region.size > 0:
                sm_mean = float(sm_region.mean().values)
                sm_values.append(sm_mean)
                times.append(hours_before)
        except Exception as e:
            std_print('WARN', 'as', f'Track {track_id} hour {hours_before}', str(e))
            continue

    if len(sm_values) >= 3:
        # Plot trend
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, sm_values, 'o-', linewidth=2.5, markersize=8, color='darkgreen')
        ax.fill_between(times, sm_values, alpha=0.2, color='lightgreen')
        ax.set_xlabel('Hours Before System Formation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Soil Water Volumetric Level-1 (m³/m³)', fontsize=12, fontweight='bold')
        ax.set_title(f'Track {track_id}: Antecedent Soil Moisture Trend\n48h Before MD Formation', fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Trend line
        if len(times) >= 3:
            z = np.polyfit(times, sm_values, 1)
            p_fit = np.poly1d(z)
            ax.plot(times, p_fit(times), 'r--', linewidth=2, alpha=0.7, label=f'Trend: {z[0]:.4f} m³/m³ per hour')

            # Interpret trend
            if z[0] > 0.001:
                trend_text = 'Increasing moisture before MD'
            elif z[0] < -0.001:
                trend_text = 'Decreasing moisture before MD'
            else:
                trend_text = 'Stable moisture before MD'

            common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)
            ax.text(1.02, 0.98, trend_text,
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   bbox=common_bbox
            )

        ax.legend(fontsize=10)
        ax.invert_xaxis()
        plt.tight_layout()

        plot_path = os.path.join(output_dir, f'track_{track_id}_antecedent_trend.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        std_print('SAVED', 'as', 'Antecedent trend plot', plot_path)
        return sm_values
    else:
        std_print('WARN', 'as', f'Track {track_id}', 'Insufficient data for trend analysis')
        return None

def create_antecedent_summary_report(results_df, output_dir):
    report_path = os.path.join(output_dir, 'antecedent_analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("Antecedent soil moisture analysis report\n")
        # Basic statistics
        f.write("1. Basic statistics\n")
        f.write(f"Total tracks analyzed: {len(results_df)}\n")

        valid_sm = results_df['antecedent_sm'].dropna()
        if len(valid_sm) > 0:
            f.write(f"Tracks with valid antecedent data: {len(valid_sm)}\n")
            f.write(f"Mean antecedent SM: {valid_sm.mean():.4f} m³/m³\n")
            f.write(f"Std deviation: {valid_sm.std():.4f} m³/m³\n")
            f.write(f"Minimum: {valid_sm.min():.4f} m³/m³\n")
            f.write(f"Maximum: {valid_sm.max():.4f} m³/m³\n\n")

        # Soil moisture classification
        f.write("2. Soil moisture classification\n")
        if 'sm_class' in results_df.columns:
            class_counts = results_df['sm_class'].value_counts()
            for sm_class, count in class_counts.items():
                f.write(f"{sm_class}: {count} tracks ({count/len(results_df)*100:.1f}%)\n")
            f.write("\n")

        # By origin type
        f.write("3. By origin type\n")
        if 'origin_type' in results_df.columns:
            origin_counts = results_df['origin_type'].value_counts()
            for origin, count in origin_counts.items():
                subset = results_df[results_df['origin_type'] == origin]
                sm_mean = subset['antecedent_sm'].mean()
                duration_mean = subset['duration_hours'].mean()
                f.write(f"{origin.replace('_', ' ').title()}: {count} tracks\n")
                f.write(f"  Mean antecedent SM: {sm_mean:.4f} m³/m³\n")
                f.write(f"  Mean duration: {duration_mean:.0f} hours\n\n")

        # Key findings
        f.write("4. Key findings\n")
        # Check for correlation with duration
        if len(valid_sm) >= 3:
            try:
                r, p = pearsonr(results_df['antecedent_sm'].dropna(), results_df['duration_hours'].dropna())
                f.write(f"Correlation with duration: r = {r:.3f}, p = {p:.4f}\n")
                if p < 0.05:
                    if r > 0.5:
                        f.write("  STRONG POSITIVE: Wetter soils support longer MDs\n")
                    elif r > 0.3:
                        f.write("  MODERATE POSITIVE: Wetter soils may extend MD life\n")
                    elif r < -0.3:
                        f.write("  NEGATIVE: Drier soils support longer MDs\n")
                    else:
                        f.write("  WEAK but significant relationship\n")
                else:
                    f.write("  No significant correlation with duration\n")
            except:
                f.write("  Correlation analysis failed\n")
    std_print('SAVED', 'as', 'Antecedent analysis report', report_path)
