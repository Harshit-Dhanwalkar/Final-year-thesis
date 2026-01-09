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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import traceback

from utils.land_utils import is_point_land

def compute_antecedent_sm(
    ds: xr.Dataset,
    track_df: pd.DataFrame,
    track_id: int,
    hours_before_landfall: int = 24,
    exclude_last_hours: int = 6,
    buffer_km: float = 100
) -> float:
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track.empty:
        return np.nan

    sm_vals = []

    # Find first landfall point
    landfall_time = None
    for _, row in track.iterrows():
        if is_point_land(row['latitude'], row['longitude']):
            landfall_time = pd.to_datetime(row['time'])
            break

    # If no landfall, use end of track as reference
    # if landfall_time is None:
    #     landfall_time = pd.to_datetime(track['time'].max())
    if landfall_time is None:
        print(f" >  [INFO    ] Track {track_id}: No landfall detected")
        return np.nan  # No landfall occurred

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
    print(f" >  [INFO    ] Track {track_id}: Time window = {t_start} to {t_end}")

    # Filter track points in this window
    subset_track = track[
        (pd.to_datetime(track['time']) >= t_start) &
        (pd.to_datetime(track['time']) <= t_end)
    ]

    if subset_track.empty:
        print(f" >  [WARNING ] Track {track_id}: No track points in time window")
        return np.nan
    print(f" >  [INFO    ] Track {track_id}: {len(subset_track)} points in time window")

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
    print(f" >  [INFO    ] Track {track_id}: Extracted {points_extracted} SM values")

    # Return sm_mean or NaN if insufficient data
    if sm_vals:
        asm_mean = np.mean(sm_vals)
        print(f" >  [INFO    ] Track {track_id}: ASM = {asm_mean:.4f} m³/m³")
        return asm_mean
    else:
        print(f" >  [WARNING ] Track {track_id}: No valid SM values extracted")
        return np.nan

def compute_antecedent_sm_all_tracks(
    ds: xr.Dataset,
    full_track_df: pd.DataFrame,
    hours_before_landfall: int = 24,
    exclude_last_hours: int = 6,
    buffer_km: float = 100
) -> pd.DataFrame:
    unique_tracks = sorted(full_track_df['track_id'].unique())
    results = []
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
            print(f" >  [INFO    ] Track {track_id}: ASM_24h = {asm_24h:.4f} m³/m³")
        else:
            print(f" >  [WARNING ] Track {track_id}: No landfall or insufficient data")
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

    # Find landfall
    landfall_idx = -1
    originates_over_land = False
    # Check if originates over land
    if len(track_df) > 0:
        first_row = track_df.iloc[0]
        originates_over_land = is_point_land(first_row['latitude'], first_row['longitude'])
    # Find first landfall
    for i, (_, row) in enumerate(track_df.iterrows()):
        if is_point_land(row['latitude'], row['longitude']):
            landfall_idx = i
            break
    if landfall_idx == -1:
        landfall_idx = len(track_df) - 1

    # Extract SM along track
    sm_values = []
    times = []
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
    elif landfall_idx == -1 or landfall_idx >= len(times):
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


def plot_antecedent_sm_distribution(
    asm_results: pd.DataFrame,
    output_dir: str = 'output'
) -> None:
    """Histogram and boxplot of antecedent SM across all tracks."""
    valid_asm = asm_results[asm_results['has_landfall']]['ASM_24h'].dropna()

    if len(valid_asm) == 0:
        print(" >  [WARNING ] No valid ASM data for distribution plot")
        return

    median = valid_asm.median()
    mean = valid_asm.mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1 = axes[0]
    counts, bins, patches = ax1.hist(
        valid_asm,
        bins=15,
        color='steelblue',
        alpha=0.7,
        linewidth=1.2
    )
    ax1.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2.5,
        label=f'Median: {median:.4f}',
        zorder=4
    )
    ax1.axvline(
        mean,
        color='green',
        linestyle='--',
        linewidth=2.5,
        label=f'Mean: {mean:.4f}',
        zorder=4
    )

    ax1.set_xlabel('Antecedent SM (m³/m³)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency (Tracks)', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Antecedent Soil Moisture', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.5, axis='y')

    # Boxplot
    ax2 = axes[1]
    bp = ax2.boxplot(
        [valid_asm],
        vert=True,
        patch_artist=True,
        widths=0.5,
        boxprops=dict(facecolor='blue', alpha=0.7),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )
    ax2.scatter(
        [1]*len(valid_asm),
        valid_asm,
        alpha=0.4,
        s=50,
        color='blue',
        zorder=3
    )
    ax2.set_ylabel('Antecedent SM (m³/m³)', fontsize=11, fontweight='bold')
    ax2.set_title('Boxplot of Antecedent Soil Moisture', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(['All Tracks'])
    ax2.grid(True, alpha=0.5, axis='y')

    # Statistics
    stats_text = (
        f"N tracks: {len(valid_asm)}\n"
        f"Mean: {mean:.4f}\n"
        f"Median: {median:.4f}\n"
        f"Std Dev: {valid_asm.std():.4f}\n"
        f"Q1: {valid_asm.quantile(0.25):.4f}\n"
        f"Q3: {valid_asm.quantile(0.75):.4f}\n"
        f"Min: {valid_asm.min():.4f}\n"
        f"Max: {valid_asm.max():.4f}"
    )
    ax2.text(
        1.5,
        valid_asm.min(),
        stats_text,
        fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8)
    )

    plt.tight_layout()
    outdir = os.path.join(output_dir, 'antecedent_analysis')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'asm_distribution.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" >> [SAVED   ] ASM distribution: {outpath}")


def plot_soil_moisture_precipitation_map(
    ds: xr.Dataset,
    track_df: pd.DataFrame,
    track_id: int,
    hours_before: int = 24,
    hours_after: int = 24,
    buffer_deg: float = 5.0,
    output_dir: str = 'output'
) -> None:
    """comparison map before/after landfall."""
    track = track_df[track_df['track_id'] == track_id].sort_values('time')

    if len(track) < 3:
        print(f" >> [WARNING ] Track {track_id} has insufficient points ({len(track)})")
        return

    # Find landfall and check origin
    landfall_idx = -1
    landfall_time = None
    originates_over_land = False

    if len(track) > 0:
        first_row = track.iloc[0]
        originates_over_land = is_point_land(first_row['latitude'], first_row['longitude'])

    for i, row in track.iterrows():
        if is_point_land(row['latitude'], row['longitude']):
            landfall_idx = i
            landfall_time = pd.to_datetime(row['time'])
            break

    has_landfall = landfall_idx != -1

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
    before_start = landfall_time - pd.Timedelta(hours=hours_before)
    after_end = landfall_time + pd.Timedelta(hours=hours_after)

    track_before = track[pd.to_datetime(track['time']) < landfall_time]
    track_after = track[pd.to_datetime(track['time']) >= landfall_time]

    if len(track_before) == 0 and len(track) > 0:
        track_before = track.iloc[:1]
    if len(track_after) == 0 and len(track) > 0:
        track_after = track.iloc[-1:]

    # Get extent
    all_lats = pd.concat([track_before['latitude'], track_after['latitude']])
    all_lons = pd.concat([track_before['longitude'], track_after['longitude']])

    min_lat = all_lats.min() - buffer_deg
    max_lat = all_lats.max() + buffer_deg
    min_lon = all_lons.min() - buffer_deg
    max_lon = all_lons.max() + buffer_deg

    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    try:
        ds_times = pd.to_datetime(ds[time_coord].values)
        actual_before_start = max(before_start, ds_times.min())
        actual_after_end = min(after_end, ds_times.max())

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
            return

        # Calculate fields
        sm_before_mean = ds_before['swvl1'].mean(dim=time_coord)
        sm_after_mean = ds_after['swvl1'].mean(dim=time_coord)
        tp_before_total = ds_before['tp'].sum(dim=time_coord) * 1000
        tp_after_total = ds_after['tp'].sum(dim=time_coord) * 1000
        sm_change = sm_after_mean - sm_before_mean

        # Get antecedent SM
        asm_24h = compute_antecedent_sm(ds, track_df, track_id, 24, 6, 100)
        asm_display = f"{asm_24h:.4f} m³/m³" if not np.isnan(asm_24h) else "N/A"

        # Determine status text
        if originates_over_land:
            period_label = "Origin"
            status_text = "Originates over land"
        elif not has_landfall:
            period_label = "Midpoint"
            status_text = "No Landfall"
        else:
            period_label = "Landfall"
            status_text = "Landfall"

        # Create 6-panel figure
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(18, 12),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        period_titles = [
            f'Before {period_label} ({hours_before}h)',
            f'After {period_label} ({hours_after}h)',
            'Change (After - Before)'
        ]

        # Soil moisture maps
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
            else:
                if len(track) > 0:
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

            ax.coastlines(resolution='50m', linewidth=1)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
            ax.set_title(f'Soil Moisture\n{title}', fontsize=11, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax, pad=0.02)
            if i == 2:
                cbar.set_label('Δ Soil Moisture (m³/m³)', fontsize=9)
            else:
                cbar.set_label('Soil Moisture (m³/m³)', fontsize=9)

        # Precipitation maps
        for i, (data, title) in enumerate(zip([tp_before_total, tp_after_total], period_titles[:2])):
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

            im = ax.contourf(
                data.longitude,
                data.latitude,
                data.values,
                levels=20,
                cmap='YlOrRd',
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
                ax.plot(track_before['longitude'], track_before['latitude'], 'b-',
                       linewidth=2, transform=ccrs.PlateCarree(), label='Track', zorder=5)
            else:
                if len(track_after) > 0:
                    ax.plot(track_after['longitude'], track_after['latitude'], 'r-',
                           linewidth=2, transform=ccrs.PlateCarree(), label='Track', zorder=5)

            ax.coastlines(resolution='50m', linewidth=1)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
            ax.set_title(f'Total Precipitation\n{title}', fontsize=11, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label('Total Precip (mm)', fontsize=9)

        # Statistics panel
        ax_stats = axes[1, 2]
        ax_stats.axis('off')

        sm_before_val = sm_before_mean.mean().item() if sm_before_mean.size > 0 else np.nan
        sm_after_val = sm_after_mean.mean().item() if sm_after_mean.size > 0 else np.nan
        sm_change_val = sm_change.mean().item() if sm_change.size > 0 else np.nan
        tp_before_val = tp_before_total.mean().item() if tp_before_total.size > 0 else np.nan
        tp_after_val = tp_after_total.mean().item() if tp_after_total.size > 0 else np.nan

        stats_text = (
            f"Track {track_id} Analysis\n"
            f"Status: {status_text}\n"
            f"Time: {landfall_time.strftime('%Y-%m-%d %H:%M')}\n"
            f"\nANTECEDENT CONDITIONS:\n"
            f"  • 24h Mean SM: {asm_display}\n"
            f"\nSOIL MOISTURE:\n"
            f"  • Before: {sm_before_val:.4f} m³/m³\n"
            f"  • After:  {sm_after_val:.4f} m³/m³\n"
            f"  • Change: {sm_change_val:.4f} m³/m³\n"
            f"\nPRECIPITATION:\n"
            f"  • Before: {tp_before_val:.1f} mm\n"
            f"  • After:  {tp_after_val:.1f} mm\n"
            f"\nTRACK INFO:\n"
            f"  • Duration: {len(track)} points\n"
            f"  • Before: {len(track_before)} points\n"
            f"  • After:  {len(track_after)} points\n"
        )
        ax_stats.text(
            0.1,
            0.8,
            stats_text,
            transform=ax_stats.transAxes,
            fontsize=9,
            family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        )
        fig.suptitle(
            f'Track {track_id}: SM & Precipitation Before/After {period_label}',
            fontsize=14,
            fontweight='bold',
            y=0.98
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
