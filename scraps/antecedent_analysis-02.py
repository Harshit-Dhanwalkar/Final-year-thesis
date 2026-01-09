# This script computes soil moisture conditions in the 24 hours BEFORE landfall.
# Used to understand the moisture state of the land when the MD arrives.
# Is the antecedent soil moisture (wet vs dry) related to rainfall/flooding intensity during MD passage?

# Dry antecedent conditions -> Soil can absorb a lot of the incoming rain -> lower immediate flood risk (more infiltration).
# Wet antecedent conditions -> Soil is already near saturation → much more of the MD's heavy rainfall becomes surface runoff -> higher flood potential even if the total rainfall amount is not extreme.

# FIX: Instead of recalculating is_point_land() repeatedly, use the is_land column already computed in the track dataframe.
# # In antecedent_analysis.py & correlation_analysis.py
# # Use the pre-computed column from the dataframe
#  landfall_row = track[track['is_land'] == True].iloc[0] if any(track['is_land']) else None
# if landfall_row is not None:
#     landfall_time = pd.to_datetime(landfall_row['time'])

# TODO: Better color bar
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05, map_projection=None)
# cbar = plt.colorbar(im, cax=cax)
# if i == 2:
#     label = 'Δ Soil Moisture (m³/m³)' if var_name == 'swvl1' else 'Δ Precipitation (mm)'
# else:
#     label = 'Soil Moisture (m³/m³)' if var_name == 'swvl1' else 'Total Precip (mm)'
# cbar.set_label(label, fontsize=9)

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import traceback

from utils.land_utils import is_point_land

def compute_antecedent_sm(
    ds,
    track_df,
    track_id,
    hours_before_landfall = 24,
    exclude_last_hours = 6,
    buffer_km = 100
):
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    if track_df is None:
            print(f" >  [ERROR   ] track_df is None for track {track_id}")
            return np.nan
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track.empty:
        return np.nan

    # Find first landfall point
    landfall_time = None
    ## Mannual is_point_land calculation
    for _, row in track.iterrows():
        if is_point_land(row['latitude'], row['longitude']):
            landfall_time = pd.to_datetime(row['time'])
            break
    ## Using pre-calculated is_point_land from landfall_row in track_df
    # landfall_row = track[track['is_land'] == True]
    # if landfall_row.empty:
    #     print(f" >  [INFO    ] Track {track_id}                      : No landfall detected")
    #     return np.nan
    # landfall_time = pd.to_datetime(landfall_row.iloc[0]['time'])

    # If no landfall, use end of track as reference
    # if landfall_time is None:
    #     landfall_time = pd.to_datetime(track['time'].max())
    # If no landfall, use start of track as reference
    if landfall_time is None:
        print(f" >  [INFO    ] Track {track_id}                      : No landfall. Using start point for analysis.")
        landfall_time = pd.to_datetime(track['time'].min())
        # return np.nan  # No landfall occurred

    sm_vals = []

    # Define time window: 24 hours before, but exclude last 6 hours
    t_start = landfall_time - pd.Timedelta(hours=hours_before_landfall)
    t_end = landfall_time - pd.Timedelta(hours=exclude_last_hours)

    # Convert dataset times to datetime
    ds_times = pd.to_datetime(ds[time_coord].values)

    # Adjust window to dataset bounds
    if t_start < ds_times.min():
        print(f" >  [WARNING ] Track {track_id}: t_start ({t_start}) < ds min ({ds_times.min()})")
        t_start = ds_times.min()
    if t_end > ds_times.max():
        print(f" >  [WARNING ] Track {track_id}: t_end ({t_end}) > ds max ({ds_times.max()})")
        t_end = ds_times.max()

    if t_start >= t_end:  # No valid window
        print(f" >  [WARNING ] Track {track_id}: No valid time window (t_start >= t_end)")
        return np.nan
    print(f" >  [INFO    ] Track {track_id} Time window          : {t_start} to {t_end}")

    # Filter track points in this window
    subset_track = track[
        (pd.to_datetime(track['time']) >= t_start) &
        (pd.to_datetime(track['time']) <= t_end)
    ]

    if subset_track.empty:
        print(f" >  [WARNING ] Track {track_id}:                       No track points in time window")
        return np.nan
    print(f" >  [INFO    ] Track {track_id} Points in time window: {len(subset_track)} ")

    # Extract soil moisture along track in the time window
    points_extracted = 0
    for _, row in subset_track.iterrows():
        try:
            lat, lon = row['latitude'], row['longitude']
            t = pd.to_datetime(row['time'])

            # Calculate buffer in degrees
            lat_buf = buffer_km / 111.0
            lon_buf = buffer_km / (111.0 * np.cos(np.radians(lat)))

            # Check if time exists in dataset
            if t not in ds_times:
                # Find nearest time
                nearest_idx = np.argmin(np.abs(ds_times - t))
                t = ds_times[nearest_idx]
                print(f" >> [INFO    ] Using nearest time: {t}")

            # Extract spatial average of soil moisture
            try:
                ds_at_time = ds.sel({time_coord: t}, method='nearest')
                lat_min = min(lat - lat_buf, lat + lat_buf)
                lat_max = max(lat - lat_buf, lat + lat_buf)
                lon_min = min(lon - lon_buf, lon + lon_buf)
                lon_max = max(lon - lon_buf, lon + lon_buf)

                subset = ds_at_time.sel({
                    'latitude': slice(lat_max, lat_min),
                    'longitude': slice(lon_min, lon_max)
                })

                if 'swvl1' in subset.data_vars:
                    sm = subset['swvl1'].mean().values
                    if not np.isnan(sm):
                        sm_vals.append(float(sm))
                        points_extracted += 1
                else:
                    print(f" >  [WARNING ] 'swvl1' not in subset at {t}")
            except Exception as e:
                print(f" >  [WARNING ] Extraction failed for point at {t}: {e}")
                continue
        except Exception as e:
            print(f" >  [WARNING ] Processing point failed: {e}")
            continue
    print(f" >  [INFO    ] Track {track_id} Extracted SM values  : {points_extracted}")

    # Return sm_mean or NaN if insufficient data
    if sm_vals:
        asm_mean = np.mean(sm_vals)
        print(f" >  [INFO    ] Track {track_id} ASM                  : {asm_mean:.4f} m³/m³")
        return asm_mean
    else:
        print(f" >  [WARNING ] Track {track_id}: No valid SM values extracted")
        return np.nan

def compute_antecedent_sm_all_tracks(
    ds,
    full_track_df,
    hours_before_landfall = 24,
    exclude_last_hours = 6,
    buffer_km = 100
):
    unique_tracks = sorted(full_track_df['track_id'].unique())
    results = []
    if full_track_df is None or (isinstance(full_track_df, pd.DataFrame) and full_track_df.empty):
        print(" >  [WARNING ] No track data available for ASM analysis.")
        return pd.DataFrame()
    for track_id in unique_tracks:
        print(f" >  [INFO    ] Processing track {track_id}")
        asm_24h = compute_antecedent_sm(
            ds=ds,
            track_df=full_track_df,
            track_id=track_id,
            hours_before_landfall=hours_before_landfall,
            exclude_last_hours=exclude_last_hours,
            buffer_km=buffer_km
        )
        has_landfall = not np.isnan(asm_24h)
        results.append({
            'track_id': track_id,
            'ASM_24h': asm_24h,
            'has_landfall': has_landfall
        })
        if has_landfall:
            print(f" >  [INFO    ] Track {track_id} ASM_24h              : {asm_24h:.4f} m³/m³")
        else:
            print(f" >  [WARNING ] Track {track_id}                      : No landfall or insufficient data")
    return pd.DataFrame(results)

def export_antecedent_results(
    asm_df: pd.DataFrame,
    output_dir: str = 'output'
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'antecedent_soil_moisture.csv')
    asm_df.to_csv(output_path, index=False)
    print(f" >> [SAVED   ] Antecedent SM results: {output_path}")


def plot_antecedent_sm_timeseries(
    ds: xr.Dataset,
    track_df: pd.DataFrame,
    track_id: int,
    buffer_km: float = 100,
    output_dir: str = 'output'
) -> None:
    """Plot soil moisture evolution before and after landfall."""
    print(f" >> [DEBUG   ] Starting timeseries plot for track {track_id}")
    # Filter for this track
    track_df = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track_df.empty:
        print(f" >  [ERROR   ] Track {track_id} not found in DataFrame")
        return
    print(f" >> [DEBUG   ] Track has {len(track_df)} points")

    # Extract SM along track
    sm_values = []
    times = []

    # Find landfall
    landfall_idx = -1
    landfall_time = times[landfall_idx]
    originates_over_land = False
    # Check if originates over land
    if len(track_df) > 0:
        first_row = track_df.iloc[0]
        originates_over_land = is_point_land(first_row['latitude'], first_row['longitude'])
    # Find first landfall
    for i, (_, row) in enumerate(track_df.iterrows()):
        if is_point_land(row['latitude'], row['longitude']):
            landfall_idx = i
            landfall_time = pd.to_datetime(row['time'])
            break
    # If manual loop fails, force a reference time to ensure the plot runs
    if landfall_time is None:
        landfall_idx = 0
        landfall_time = pd.to_datetime(track_df.iloc[0]['time'])
        print(f" >  [INFO    ] Track {track_id}: No manual landfall found. Using T0 for map.")

    # Approximation: 1 degree latitude ~ 111 km
    lat_buffer = buffer_km / 111.0
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'

    for _, row in track_df.iterrows():
        try:
            t = pd.to_datetime(row['time'])
            lat, lon = row['latitude'], row['longitude']
            lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat)))
            ds_at_time = ds.sel({time_coord: t}, method='nearest')
            lat_min = min(lat - lat_buffer, lat + lat_buffer)
            lat_max = max(lat - lat_buffer, lat + lat_buffer)
            lon_min = min(lon - lon_buffer, lon + lon_buffer)
            lon_max = max(lon - lon_buffer, lon + lon_buffer)
            subset = ds_at_time.sel({
                'latitude': slice(lat_max, lat_min),
                'longitude': slice(lon_min, lon_max)
            })
            if 'swvl1' in subset.data_vars:
                sm = subset['swvl1'].mean().values
                if not np.isnan(sm):
                    sm_values.append(sm)
                    times.append(t)
        except Exception as e:
            print(f" >  [WARNING ] Failed to extract SM at {t}: {e}")
            continue

    if len(sm_values) < 3:
        print(f" >  [WARNING ] Insufficient SM values ({len(sm_values)}) for timeseries plot")
        return

    # Determine status
    if originates_over_land:
        status_text = "Originates over land"
        status_color = 'blue'
    elif landfall_idx == -1:
        status_text = "No Landfall"
        status_color = 'gray'
    else:
        status_text = "Landfall"
        status_color = 'red'

    fig, ax = plt.subplots(figsize=(14, 7))
    # Plot soil moisture
    ax.plot(
        times,
        sm_values,
        'o-',
        linewidth=2.5,
        markersize=7,
        color='blue',
        label='Soil Moisture',
        alpha=0.8
    )
    ax.fill_between(times, sm_values, alpha=0.2, color='blue')
    # Landfall line and shading
    if landfall_idx >= 0 and landfall_idx < len(times):
        landfall_time = times[landfall_idx]
        ax.axvline(
            landfall_time,
            color=status_color,
            linestyle='--',
            linewidth=2.5,
            alpha=0.8,
            zorder=4
        )
        ax.plot(
            landfall_time,
            sm_values[landfall_idx],
            'D',
            color=status_color,
            markersize=10,
            zorder=5
        )
        ax.axvspan(times[0], landfall_time, alpha=0.1, color='blue')
        if landfall_idx < len(times) - 1:
            ax.axvspan(landfall_time, times[-1], alpha=0.1, color='red')

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Soil Water Content (m³/m³)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Track {track_id}: Antecedent Soil Moisture Time Series\nStatus: {status_text}',
        fontsize=13,
        fontweight='bold',
        pad=10
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    # Legend
    handles = [ax.get_lines()[0]]
    labels = ['Soil Moisture']
    if landfall_idx >= 0 and landfall_idx < len(times):
        handles.append(plt.Line2D(
            [0],
            [0],
            color=status_color,
            linestyle='--',
            linewidth=2.5
        ))
        labels.append(status_text)
    ax.legend(handles, labels, loc='best', fontsize=10, framealpha=0.9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    outdir = os.path.join(output_dir, 'antecedent_timeseries')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'track_{int(track_id)}_asm_timeseries.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" >> [SAVED   ] Antecedent SM timeseries: {outpath}")


def plot_sm_tp_map(
    ds,
    track_df,
    track_id,
    hours_before = 24,
    hours_after = 24,
    buffer_deg = 5.0,
    output_dir = 'output'
):
    """Comparison map before/after landfall"""
    track = track_df[track_df['track_id'] == track_id].sort_values('time')

    if len(track) < 3:
        print(f" >  [WARNING ] Track {track_id} has insufficient points: ({len(track)})")
        return

    # Find landfall and check origin
    landfall_idx = -1
    landfall_time = None
    has_landfall = landfall_idx != -1
    originates_over_land = False
    # Check if originates over land
    if len(track) > 0:
        first_row = track.iloc[0]
        originates_over_land = is_point_land(first_row['latitude'], first_row['longitude'])

    # Find first landfall
    for i, row in track.iterrows():
        if is_point_land(row['latitude'], row['longitude']):
            landfall_idx = i
            landfall_time = pd.to_datetime(row['time'])
            break
    if landfall_time is None:
        print(f" > [WARNING] No landfall detected for Track {track_id}. Using start of track for maps.")
        landfall_idx = 0
        landfall_time = pd.to_datetime(track.iloc[0]['time'])

    # Update AFTER finding landfall
    if landfall_time is not None:
        has_landfall = True
    if not has_landfall:
        if len(track) > 0:
            midpoint_idx = len(track) // 2
            if midpoint_idx < len(track):
                landfall_time = pd.to_datetime(track.iloc[midpoint_idx]['time'])
                landfall_idx = midpoint_idx
            else:
                landfall_time = pd.to_datetime(track.iloc[0]['time'])
                landfall_idx = 0
        else:
            print(f" >  [ERROR   ] Track {track_id} has no points")
            return

    # Get before/after times
    landfall_time = pd.Timestamp(landfall_time)
    before_start = landfall_time - pd.Timedelta(hours=hours_before)
    after_end = landfall_time + pd.Timedelta(hours=hours_after)
    track_before = track[pd.to_datetime(track['time']) < landfall_time]
    track_after = track[pd.to_datetime(track['time']) >= landfall_time]

    if len(track_before) == 0 and len(track) > 0:
        track_before = track.iloc[:1]
    if len(track_after) == 0 and len(track) > 0:
        track_after = track.iloc[-1:]

    # Get extent
    all_lats = pd.concat([track_before['latitude'], track_after['latitude']], ignore_index=True)
    all_lons = pd.concat([track_before['longitude'], track_after['longitude']], ignore_index=True)

    min_lat = all_lats.min() - buffer_deg
    max_lat = all_lats.max() + buffer_deg
    min_lon = all_lons.min() - buffer_deg
    max_lon = all_lons.max() + buffer_deg

    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    try:
        ds_times = pd.to_datetime(ds[time_coord].values)
        actual_before_start = pd.Timestamp(max(before_start, ds_times.min()))
        actual_after_end = pd.Timestamp(min(after_end, ds_times.max()))
        # Extract datasets
        ds_before = ds.sel({
            time_coord: slice(actual_before_start, landfall_time - pd.Timedelta(hours=1)),
            'latitude': slice(max_lat, min_lat),
            'longitude': slice(min_lon, max_lon)
        })
        ds_after = ds.sel({
            time_coord: slice(landfall_time, actual_after_end),
            'latitude': slice(max_lat, min_lat),
            'longitude': slice(min_lon, max_lon)
        })
        if ds_before[time_coord].size == 0 or ds_after[time_coord].size == 0:
            print(f" >  [WARNING ] Insufficient temporal data for track {track_id}")
            return False

        # Calculate fields
        sm_before_mean = ds_before['swvl1'].mean(dim=time_coord)
        sm_after_mean = ds_after['swvl1'].mean(dim=time_coord)
        sm_change = sm_after_mean - sm_before_mean
        tp_before_total = ds_before['tp'].sum(dim=time_coord) * 1000
        tp_after_total = ds_after['tp'].sum(dim=time_coord) * 1000
        tp_change = tp_after_total - tp_before_total 

        # Get antecedent SM
        asm_24h = compute_antecedent_sm(ds, track_df, track_id, 24, 6, 100)
        asm_display = f"{asm_24h:.4f} m³/m³" if not np.isnan(asm_24h) else "N/A"

        # Determine status text
        if originates_over_land:
            period_label = "Origin"
            status_text = "Originates over land"
        elif landfall_idx == -1:
            period_label = "Midpoint"
            status_text = "No Landfall"
        else:
            period_label = "Landfall"
            status_text = "Landfall"

        # Create 2x4 subplot layout
        fig, axes = plt.subplots(
            2,
            4,
            figsize=(20, 12),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # Calculate min/max for TP
        tp_all_values = np.concatenate([
            tp_before_total.values.flatten(),
            tp_after_total.values.flatten()
        ])
        tp_all_values = tp_all_values[~np.isnan(tp_all_values)]
        tp_vmin = np.nanmin(tp_all_values) if len(tp_all_values) > 0 else 0
        tp_vmax = np.nanmax(tp_all_values) if len(tp_all_values) > 0 else 1

        # Calculate symmetric limits for TP change
        tp_change_values = tp_change.values.flatten()
        tp_change_values = tp_change_values[~np.isnan(tp_change_values)]
        tp_change_vmax = max(abs(np.nanmin(tp_change_values)), abs(np.nanmax(tp_change_values))) if len(tp_change_values) > 0 else 1

        period_titles = [
            f'Before {period_label} ({hours_before}h)',
            f'After {period_label} ({hours_after}h)',
            'Change (After - Before)'
        ]

        # Row 0: Soil moisture maps (3 maps + stats box)
        for i, (data, title) in enumerate(zip(
            [sm_before_mean, sm_after_mean, sm_change],
            period_titles
        )):
            ax = axes[0, i]
            if data.size == 0 or np.all(np.isnan(data.values)):
                ax.text(
                    0.5,
                    0.5,
                    'No Data',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=12
                )
                ax.set_title(f'Soil Moisture\n{title}', fontsize=11, fontweight='bold')
                continue

            if i == 2:  # Change plot
                sm_change_vmax = max(abs(data.values.min()), abs(data.values.max()))
                im = ax.contourf(
                    data.longitude,
                    data.latitude,
                    data.values,
                    levels=20,
                    cmap='RdBu',
                    vmin=-sm_change_vmax,
                    vmax=sm_change_vmax,
                    transform=ccrs.PlateCarree(),
                    extend='both'
                )
            else:
                im = ax.contourf(
                    data.longitude,
                    data.latitude,
                    data.values,
                    levels=20,
                    cmap='YlGnBu',
                    transform=ccrs.PlateCarree(),
                    extend='both'
                )
            contours = ax.contour(
                data.longitude,
                data.latitude,
                data.values,
                levels=10,
                colors='black',
                linewidths=0.5,
                alpha=0.5,
                transform=ccrs.PlateCarree()
            )
            ax.clabel(contours, inline=True, fontsize=7, fmt='%1.3f')

            # Plot tracks
            if i == 0 and len(track_before) > 0:
                ax.plot(
                    track_before['longitude'],
                    track_before['latitude'],
                    'b-',
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                    label='Track',
                    zorder=5
                )
            elif i == 1 and len(track_after) > 0:
                ax.plot(
                    track_after['longitude'],
                    track_after['latitude'],
                    'r-',
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                    label='Track',
                    zorder=5
                )
            elif i == 2 and len(track) > 0:
                ax.plot(
                    track['longitude'],
                    track['latitude'],
                    'k-',
                    linewidth=1.5,
                    transform=ccrs.PlateCarree(),
                    alpha=0.7,
                    zorder=4
                )
                if 0 <= landfall_idx < len(track):
                    point = track.iloc[landfall_idx]
                    ax.plot(
                        point['longitude'],
                        point['latitude'],
                        'D',
                        color='yellow',
                        markersize=10,
                        transform=ccrs.PlateCarree(),
                        zorder=6
                    )

            ax.coastlines(resolution='50m', linewidth=0.6)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
            ax.set_title(f'Soil Moisture\n{title}', fontsize=11, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax, pad=0.02,shrink=0.6)
            if i == 2:
                cbar.set_label('Δ Soil Moisture (m³/m³)', fontsize=9)
            else:
                cbar.set_label('Soil Moisture (m³/m³)', fontsize=9)

        # SM Stats in 4th column of row 0
        ax_sm_stats = axes[0, 3]
        ax_sm_stats.axis('off')

        sm_before_val = sm_before_mean.mean().item() if sm_before_mean.size > 0 else np.nan
        sm_after_val = sm_after_mean.mean().item() if sm_after_mean.size > 0 else np.nan
        sm_change_val = sm_change.mean().item() if sm_change.size > 0 else np.nan

        sm_stats_text = (
            f"Soil moisture\n\n"
            f"Before:             {sm_before_val:.4f} m³/m³\n"
            f"After:              {sm_after_val:.4f} m³/m³\n"
            f"Change:             {sm_change_val:+.4f} m³/m³\n"
          f"\nAntecedent (24h):   {asm_display}"
        )
        ax_sm_stats.text(
            0.05,
            0.95,
            sm_stats_text,
            transform=ax_sm_stats.transAxes,
            fontsize=8,
            family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', alpha=0.8)
        )

        # Row 1: Precipitation maps (3 maps + stats box)
        for i, (data, title) in enumerate(zip(
            [tp_before_total, tp_after_total, tp_change],
            period_titles
        )):
            ax = axes[1, i]
            if data.size == 0 or np.all(np.isnan(data.values)):
                ax.text(
                    0.5,
                    0.5,
                    'No Data',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=12
                )
                ax.set_title(f'Total Precipitation\n{title}', fontsize=11, fontweight='bold')
                continue

            if i == 2:  # Change plot
                im = ax.contourf(
                    data.longitude,
                    data.latitude,
                    data.values,
                    levels=20,
                    cmap='RdBu',
                    vmin=-tp_change_vmax,
                    vmax=tp_change_vmax,
                    transform=ccrs.PlateCarree(),
                    extend='both'
                )
            else:  # Before/After plots
                im = ax.contourf(
                    data.longitude,
                    data.latitude,
                    data.values,
                    levels=20,
                    cmap='YlOrRd',
                    vmin=tp_vmin,
                    vmax=tp_vmax,
                    transform=ccrs.PlateCarree(),
                    extend='max'
                )

            contours = ax.contour(
                data.longitude,
                data.latitude,
                data.values,
                levels=10,
                colors='black',
                linewidths=0.5,
                alpha=0.5,
                transform=ccrs.PlateCarree()
            )
            ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')

            if i == 0 and len(track_before) > 0:
                ax.plot(
                    track_before['longitude'],
                    track_before['latitude'],
                    'b-',
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                    label='Track',
                    zorder=5
                )
            elif i == 1 and len(track_after) > 0:
                ax.plot(
                    track_after['longitude'],
                    track_after['latitude'],
                    'r-',
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                    label='Track',
                    zorder=5
                )
            elif i == 2 and len(track) > 0:
                ax.plot(
                    track['longitude'],
                    track['latitude'],
                    'k-',
                    linewidth=1.5,
                    transform=ccrs.PlateCarree(),
                    alpha=0.7,
                    zorder=4
                )
                if 0 <= landfall_idx < len(track):
                    point = track.iloc[landfall_idx]
                    ax.plot(
                        point['longitude'],
                        point['latitude'],
                        'D',
                        color='yellow',
                        markersize=10,
                        transform=ccrs.PlateCarree(),
                        zorder=6
                    )

            ax.coastlines(resolution='50m', linewidth=0.6)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
            ax.set_title(f'Total Precipitation\n{title}', fontsize=11, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax, pad=0.02,shrink=0.6)
            if i == 2:
                cbar.set_label('Δ Precipitation (mm)', fontsize=9)
            else:
                cbar.set_label('Total Precip (mm)', fontsize=9)

        # TP Stats in 4th column of row 1
        ax_tp_stats = axes[1, 3]
        ax_tp_stats.axis('off')

        tp_before_val = tp_before_total.mean().item() if tp_before_total.size > 0 else np.nan
        tp_after_val = tp_after_total.mean().item() if tp_after_total.size > 0 else np.nan
        tp_change_val = tp_change.mean().item() if tp_change.size > 0 else np.nan

        tp_stats_text = (
            f"Total Precipitation\n\n"
            f"Before:        {tp_before_val:.1f} mm\n"
            f"After:         {tp_after_val:.1f} mm\n"
            f"Change:        {tp_change_val:+.1f} mm\n"
          f"\nLandfall Time: {landfall_time.strftime('%d/%m')}"
        )
        ax_tp_stats.text(
            0.05,
            0.95,
            tp_stats_text,
            transform=ax_tp_stats.transAxes,
            fontsize=8,
            family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', alpha=0.8)
        )

        fig.suptitle(
            f'Track {track_id}: SM & TP Before/After {period_label}',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f'track_{int(track_id)}_sm_tp_before_after.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f" >  [SAVED   ] SM/TP before/after map: {outpath}")

    except Exception as e:
        print(f" >  [ERROR   ] Plotting before/after maps for track {track_id}: {e}")
        traceback.print_exc()
    return True  # Signal success


def analyze_landfall_fluxes(ds, track_df, track_id, hours_before=24, hours_after=24):
    # Find landfall time
    landfall_time = None
    for i, row in track_df.iterrows():
        if row.get('is_land', False) and i > 0 and not track_df.iloc[i-1].get('is_land', True):
            landfall_time = pd.to_datetime(row['time'])
            break
    if not landfall_time:
        return None
    # Extract fluxes before and after landfall
    before_start = landfall_time - pd.Timedelta(hours=hours_before)
    after_end = landfall_time + pd.Timedelta(hours=hours_after)
    # Calculate mean fluxes
    fluxes_before = {'latent': [], 'sensible': [], 'total': []}
    fluxes_after = {'latent': [], 'sensible': [], 'total': []}
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    for t in pd.date_range(before_start, landfall_time, freq='h'):
        if t in ds[time_coord].values:
            # Extract near landfall point
            point_data = ds.sel(**{time_coord: t}, method='nearest')
            # Simplified extraction - in practice, use spatial mean around track
            if 'avg_slhtf' in point_data and 'avg_ishf' in point_data:
                fluxes_before['latent'].append(float(point_data['avg_slhtf'].mean()))
                fluxes_before['sensible'].append(float(point_data['avg_ishf'].mean()))
                fluxes_before['total'].append(float(point_data['avg_slhtf'].mean() + point_data['avg_ishf'].mean()))
    for t in pd.date_range(landfall_time, after_end, freq='h'):
        if t in ds[time_coord].values:
            point_data = ds.sel(**{time_coord: t}, method='nearest')
            if 'avg_slhtf' in point_data and 'avg_ishf' in point_data:
                fluxes_after['latent'].append(float(point_data['avg_slhtf'].mean()))
                fluxes_after['sensible'].append(float(point_data['avg_ishf'].mean()))
                fluxes_after['total'].append(float(point_data['avg_slhtf'].mean() + point_data['avg_ishf'].mean()))
    # Calculate statistics
    results = {
        'track_id':      track_id,
        'landfall_time': landfall_time,
        'before':        {k: np.nanmean(v) if v else np.nan for k, v in fluxes_before.items()},
        'after':         {k: np.nanmean(v) if v else np.nan for k, v in fluxes_after.items()},
        'change': {
            'latent':    (np.nanmean(fluxes_after['latent']) - np.nanmean(fluxes_before['latent'])) 
                      if fluxes_after['latent'] and fluxes_before['latent'] else np.nan,
            'sensible':  (np.nanmean(fluxes_after['sensible']) - np.nanmean(fluxes_before['sensible']))
                       if fluxes_after['sensible'] and fluxes_before['sensible'] else np.nan,
            'total':     (np.nanmean(fluxes_after['total']) - np.nanmean(fluxes_before['total']))
                    if fluxes_after['total'] and fluxes_before['total'] else np.nan
        }
    }
    return results
