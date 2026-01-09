# This script computes soil moisture conditions in the 24 hours BEFORE landfall.
# Used to understand the moisture state of the land when the MD arrives.
# Is the antecedent soil moisture (wet vs dry) related to rainfall/flooding intensity during MD passage?

# Dry antecedent conditions -> Soil can absorb a lot of the incoming rain -> lower immediate flood risk (more infiltration).
# Wet antecedent conditions -> Soil is already near saturation → much more of the MD's heavy rainfall becomes surface runoff -> higher flood potential even if the total rainfall amount is not extreme.

# FIX: before after 24 landfall plots (plot_sm_tp_map) shows full track rather than just clipped track for that duration

# FIX: even tho clipped spacial domain to 10N70E to 30N95E yet before/after plots uses full extend

# FIX: Instead of recalculating is_point_land() repeatedly, use the is_land column already computed in the track dataframe.
# # In antecedent_analysis.py & correlation_analysis.py
# # Use the pre-computed column from the dataframe
#  landfall_row = track[track['is_land'] == True].iloc[0] if any(track['is_land']) else None
# if landfall_row is not None:
#     landfall_time = pd.to_datetime(landfall_row['time'])

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr, spearmanr, ttest_ind
import seaborn as sns
import traceback

from utils.land_utils import is_point_land

# Use clipped domain instead of dataset bounds (10N 70E - 30N 95E)
clipped_min_lat = 10.0  # N
clipped_max_lat = 30.0  # N
clipped_min_lon = 70.0  # E
clipped_max_lon = 95.0  # E

def compute_antecedent_sm(
    ds,
    track_df,
    track_id,
    hours_before_landfall = 24,
    exclude_last_hours = 6,
    buffer_km = 100,
    land_only=True
):
    track_id = int(track_id)
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    if track_df is None:
            print(f" >  [ERROR an  ] track_df is None for track {track_id}")
            return np.nan
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track.empty:
        return np.nan

    # Find first landfall point
    landfall_time = None
    ## Using pre-calculated is_point_land from landfall_row in track_df
    if 'is_land' in track.columns:
        landfall_row = track[track['is_land'] == True]
        print(" >  [INFO an   ] Using is_land check from landfall_row")
        if not landfall_row.empty:
            landfall_time = pd.to_datetime(landfall_row.iloc[0]['time'])
        if landfall_row.empty:
            print(f" >  [INFO an   ] Track {track_id}                      : No landfall detected")
            return np.nan
    # If still no landfall, manual check: is_point_land calculation
    if landfall_time is None:
        print(" >  [WARNING ] Mannual landfall check rather than using is_point_land")
        for _, row in track.iterrows():
            if is_point_land(row['latitude'], row['longitude']):
                landfall_time = pd.to_datetime(row['time'])
                break

    # If no landfall, use end of track as reference
    # if landfall_time is None:
    #     print(f" >  [WARNING ] Track {track_id}  No landfall. Using end point for analysis.")
    #     landfall_time = pd.to_datetime(track['time'].max())
    # If no landfall, use start of track as reference
    if landfall_time is None:
        print(f" >  [WARNING ] Track {track_id}  No landfall. Using start point for analysis.")
        landfall_time = pd.to_datetime(track['time'].min())

    sm_vals = []
    ocean_cells_excluded = 0
    land_cells_included = 0

    # Convert dataset times to datetime
    ds_times = pd.to_datetime(ds[time_coord].values)
    # Define time window: 24 hours before, but exclude last 6 hours
    t_start = landfall_time - pd.Timedelta(hours=hours_before_landfall)
    t_end = landfall_time - pd.Timedelta(hours=exclude_last_hours)

    print(f" >> [DEBUG] Landfall time found: {landfall_time}")
    print(f" >> [DEBUG] Time window: {t_start} to {t_end}")
    print(f" >> [DEBUG] Dataset time range: {ds_times.min()} to {ds_times.max()}")
    if t_start >= t_end:
        print(f" >> [DEBUG] Invalid window: t_start >= t_end")
        return np.nan

    # Adjust window to dataset bounds
    if t_start < ds_times.min():
        print(f" >  [WARNING ] Track {track_id}                           :          t_start ({t_start}) < ds min ({ds_times.min()})")
        t_start = ds_times.min()
    if t_end > ds_times.max():
        print(f" >  [WARNING ] Track {track_id}                           :          t_end ({t_end}) > ds max ({ds_times.max()})")
        t_end = ds_times.max()

    if t_start >= t_end:  # No valid window
        print(f" >  [WARNING ] Track {track_id}                           :          No valid time window (t_start >= t_end)")
        return np.nan
    print(f" >  [INFO an   ] Track {track_id} Time window               :          {t_start} to {t_end}")

    # Filter track points in this window
    subset_track = track[
        (pd.to_datetime(track['time']) >= t_start) &
        (pd.to_datetime(track['time']) <= t_end)
    ]

    if subset_track.empty:
        print(f" >  [WARNING ] Track {track_id}, No track points in time window")
        return np.nan
    print(f" >  [INFO an   ] Track {track_id} Points in time window     :          {len(subset_track)} ")

    # Extract soil moisture along track in the time window
    points_extracted = 0
    for _, row in subset_track.iterrows():
        try:
            lat, lon = row['latitude'], row['longitude']
            t = pd.to_datetime(row['time'])
            # Only process if point is over land
            if land_only and not is_point_land(lat, lon):
                ocean_cells_excluded += 1
                continue

            # Calculate buffer in degrees
            lat_buf = buffer_km / 111.0
            lon_buf = buffer_km / (111.0 * np.cos(np.radians(lat)))

            # Check if time exists in dataset
            if t not in ds_times:
                # Find nearest time
                nearest_idx = np.argmin(np.abs(ds_times - t))
                t = ds_times[nearest_idx]
                print(f" >> [INFO an   ] Using nearest time                :          {t}")

            # Extract spatial average of soil moisture
            try:
                ds_at_time = ds.sel({time_coord: t}, method='nearest')
                # Spatial bounds
                lat_min = min(lat - lat_buf, lat + lat_buf)
                lat_max = max(lat - lat_buf, lat + lat_buf)
                lon_min = min(lon - lon_buf, lon + lon_buf)
                lon_max = max(lon - lon_buf, lon + lon_buf)
                # Clamp to dataset bounds
                lat_min = max(lat_min, float(ds_at_time.latitude.min()))
                lat_max = min(lat_max, float(ds_at_time.latitude.max()))
                lon_min = max(lon_min, float(ds_at_time.longitude.min()))
                lon_max = min(lon_max, float(ds_at_time.longitude.max()))
                # Check if bounds are valid
                if lat_min >= lat_max or lon_min >= lon_max:
                    print(f" >  [WARNING ] Invalid bounds at {t}            : lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")
                    continue

                subset = ds_at_time.sel({
                    'latitude': slice(lat_max, lat_min),
                    'longitude': slice(lon_min, lon_max)
                })
                if subset.sizes.get('latitude', 0) == 0 or subset.sizes.get('longitude', 0) == 0:
                    ocean_cells_excluded += 1
                    print(f" >  [WARNING ] Empty subset at {t}")
                    continue

                if 'swvl1' in subset.data_vars:
                    sm = subset['swvl1'].mean().values
                    # Contamination check,  Reasonable soil moisture: 0.01 - 0.70 m³/m³
                    if not np.isnan(sm) and 0.01 < sm < 0.70:
                        sm_vals.append(float(sm))
                        land_cells_included += 1
                    else:
                        ocean_cells_excluded += 1
                else:
                    print(f" >  [WARNING ] 'swvl1' not in subset at {t}")
            except Exception as e:
                ocean_cells_excluded += 1
                print(f" >  [WARNING ] Extraction failed for point at {t}: {e}")
                continue
        except Exception as e:
            ocean_cells_excluded += 1
            print(f" >  [WARNING ] Processing point failed: {e}")
            continue

    # Diagnostic
    print(f" >  [DEBUG   ] Track {track_id} Extraction statistics     :")
    print(f"                Land cells included              :          {land_cells_included}")
    print(f"                Ocean/invalid cells excluded     :          {ocean_cells_excluded}")
    print(f"                Inclusion rate                   :          {100*land_cells_included/(land_cells_included+ocean_cells_excluded if ocean_cells_excluded+land_cells_included > 0 else 1):.1f}%")

    print(f" >  [INFO an   ] Track {track_id} Extracted SM values       :          {points_extracted}")

    # Return sm_mean or NaN if insufficient data
    if sm_vals:
        asm_mean = np.mean(sm_vals)
        asm_std = np.std(sm_vals)
        print(f" >  [INFO an   ] Track {track_id} ASM                       :          {asm_mean:.4f} ± {asm_std:.4f} m³/m³")
        return asm_mean
    else:
        print(f" >  [WARNING ] Track {track_id}: No valid land-based SM values extracted")
        return np.nan


def compute_antecedent_sm_multiwindow(
    ds,
    track_df,
    track_id,
    buffer_km=100,
    land_only=True
):
    """ Compute soil moisture at multiple windows for transition analysis
    "How do TP and SWVL1 affect MD ocean -> land transition?"
    """
    track_id = int(track_id)
    windows = {
        'antecedent_24h': {
            'hours_before': 24,
            'hours_after': 6,
            'description': 'Background conditions (18h window)',
            'purpose': 'Baseline soil state before system approach'
        },
        'approach_12h': {
            'hours_before': 12,
            'hours_after': 0,
            'description': 'Entire approach phase',
            'purpose': 'System development and soil interaction'
        },
        'landfall_6h': {
            'hours_before': 6,
            'hours_after': 0,
            'description': 'Immediate pre-landfall',
            'purpose': 'Direct transition conditions'
        },
        'post_landfall_6h': {
            'is_post_landfall': True,
            'hours_before': 0,
            'hours_after': 6,
            'description': 'First 6h after landfall',
            'purpose': 'Land surface response'
        }
    }
    results = {}
    for window_name, window_config in windows.items():
        hours_before = window_config['hours_before']
        hours_after = window_config['hours_after']
        print(f" >  [INFO an   ] Computing {window_name}:")
        print(f"                {window_config['description']}")
        print(f"                Purpose: {window_config['purpose']}")

        is_post = window_config.get('is_post_landfall', False)
        if is_post:
            asm = compute_antecedent_sm_post_landfall(
                ds=ds,
                track_df=track_df,
                track_id=track_id,
                hours_after_landfall=window_config['hours_after'],
                buffer_km=buffer_km,
                land_only=land_only
            )
        else:
            asm = compute_antecedent_sm(
                ds=ds,
                track_df=track_df,
                track_id=track_id,
                hours_before_landfall=window_config['hours_before'],
                exclude_last_hours=window_config['hours_after'],
                buffer_km=buffer_km,
                land_only=land_only
            )
        results[window_name] = {
            'value': asm,
            'hours_before': window_config['hours_before'],
            'hours_after': window_config['hours_after'],
            'description': window_config['description']
        }
    return results

def compute_antecedent_sm_post_landfall(
    ds,
    track_df,
    track_id,
    hours_after_landfall=6,
    buffer_km=100,
    land_only=True
):
    """
    Compute soil moisture AFTER landfall (land phase).
    hours_after_landfall: How many hours AFTER landfall to analyze
    """
    track_id = int(track_id)
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'
    if track_df is None:
        print(f" >  [ERROR an  ] track_df is None for track {track_id}")
        return np.nan
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track.empty:
        return np.nan
    # Find first landfall point
    landfall_time = None
    for _, row in track.iterrows():
        if is_point_land(row['latitude'], row['longitude']):
            landfall_time = pd.to_datetime(row['time'])
            break
    if landfall_time is None:
        print(f" >  [WARNING ] Track {track_id}: No landfall detected for post-landfall analysis")
        return np.nan

    sm_vals = []
    ocean_cells_excluded = 0
    land_cells_included = 0

    # Convert dataset times to datetime
    ds_times = pd.to_datetime(ds[time_coord].values)
    # Define time window: FROM landfall TO landfall + hours_after
    t_start = landfall_time
    t_end = landfall_time + pd.Timedelta(hours=hours_after_landfall)
    # Adjust window to dataset bounds
    if t_start < ds_times.min():
        print(f" >  [WARNING ] Track {track_id}                           :          t_start ({t_start}) < ds min ({ds_times.min()})")
        t_start = ds_times.min()
    if t_end > ds_times.max():
        print(f" >  [WARNING ] Track {track_id}                           :          t_end ({t_end}) > ds max ({ds_times.max()})")
        t_end = ds_times.max()
    if t_start >= t_end:
        print(f" >  [WARNING ] After adjustment                  :           t_start ({t_start}) >= t_end ({t_end})")
        print(f"      Landfall time                              :           {landfall_time}")
        print(f"      Dataset range                              :           {ds_times.min()} to {ds_times.max()}")
        return np.nan
    print(f" >  [INFO an   ] Track {track_id} Post-Landfall Window      :          {t_start} to {t_end}")

    # Filter track points in this window AND over land
    subset_track = track[
        (pd.to_datetime(track['time']) >= t_start) &
        (pd.to_datetime(track['time']) <= t_end)
    ]
    if subset_track.empty:
        print(f" >  [WARNING ] Track {track_id}                           : No points in post-landfall time window")
        return np.nan
    print(f" >  [INFO an   ] Track {track_id} Points in post-landfall window:      {len(subset_track)}")

    # Extract soil moisture along track in the time window
    for _, row in subset_track.iterrows():
        try:
            lat, lon = row['latitude'], row['longitude']
            t = pd.to_datetime(row['time'])
            if land_only and not is_point_land(lat, lon):
                ocean_cells_excluded += 1
                continue
            # Calculate buffer in degrees
            lat_buf = buffer_km / 111.0
            lon_buf = buffer_km / (111.0 * np.cos(np.radians(lat)))
            # Find nearest time
            if t not in ds_times:
                nearest_idx = np.argmin(np.abs(ds_times - t))
                t = ds_times[nearest_idx]
            try:
                # Extract spatial average of soil moisture
                ds_at_time = ds.sel({time_coord: t}, method='nearest')
                # Spatial bounds
                lat_min = min(lat - lat_buf, lat + lat_buf)
                lat_max = max(lat - lat_buf, lat + lat_buf)
                lon_min = min(lon - lon_buf, lon + lon_buf)
                lon_max = max(lon - lon_buf, lon + lon_buf)
                # Clamp to dataset bounds
                lat_min = max(lat_min, float(ds_at_time.latitude.min()))
                lat_max = min(lat_max, float(ds_at_time.latitude.max()))
                lon_min = max(lon_min, float(ds_at_time.longitude.min()))
                lon_max = min(lon_max, float(ds_at_time.longitude.max()))
                # Check if bounds are valid
                if lat_min >= lat_max or lon_min >= lon_max:
                    ocean_cells_excluded += 1
                    continue
                subset = ds_at_time.sel({
                    'latitude': slice(lat_max, lat_min),
                    'longitude': slice(lon_min, lon_max)
                })
                if subset.sizes.get('latitude', 0) == 0 or subset.sizes.get('longitude', 0) == 0:
                    ocean_cells_excluded += 1
                    continue
                if 'swvl1' in subset.data_vars:
                    sm = subset['swvl1'].mean().values
                    if not np.isnan(sm) and 0.01 < sm < 0.70:
                        sm_vals.append(float(sm))
                        land_cells_included += 1
                    else:
                        ocean_cells_excluded += 1
            except Exception as e:
                ocean_cells_excluded += 1
                print(f" ERROR : {e}")
                continue
        except Exception as e:
            ocean_cells_excluded += 1
            print(f" ERROR : {e}")
            continue

    print(f" >  [DEBUG   ] Track {track_id} Post-landfall extraction  :")
    print(f"      Land cells included                        :          {land_cells_included}")
    print(f"      Ocean/invalid cells excluded               :          {ocean_cells_excluded}")

    # Return sm_mean or NaN if insufficient data
    if sm_vals:
        asm_mean = np.mean(sm_vals)
        print(f" >  [INFO an   ] Track {track_id} Post-landfall SM (6h)     :          {asm_mean:.4f} m³/m³")
        return asm_mean
    else:
        print(f" >  [WARNING ] Track {track_id}                           :          No valid post-landfall SM values extracted")
        return np.nan

def export_multiwindow_results(track_id, multiwindow_results, output_dir):
    """Export multi-window ASM results for correlation analysis"""
    track_id = int(track_id)
    export_data = {
        'track_id': track_id,
    }
    for window_name, window_data in multiwindow_results.items():
        export_data[f'{window_name}_asm'] = window_data['value']
        export_data[f'{window_name}_hours_before'] = window_data['hours_before']
    df = pd.DataFrame([export_data])
    output_path = os.path.join(output_dir, 'asm_multiwindow_results.csv')
    df.to_csv(output_path, index=False)
    print(f" >> [SAVED   ] Multi-window ASM results          :           {output_path}")
    print(f"                Summary for Track {track_id}              :")
    for window_name, window_data in multiwindow_results.items():
        print(f"                      {window_name:20s}       :         {window_data['value']:7.4f} m³/m³")
    return df

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
        print(f" >  [INFO an   ] Processing track {track_id}")
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
            print(f" >  [INFO an   ] Track {track_id} ASM_24h                   :          {asm_24h:.4f} m³/m³")
        else:
            print(f" >  [WARNING ] Track {track_id}                            :          No landfall or insufficient data")
    return pd.DataFrame(results)

def compute_asm_tp_correlation(ds, full_track_df, asm_results_df, buffer_km=100, output_dir='output'):
    """
    Correlate antecedent soil moisture with precipitation during track passage.
    Research question: Does low ASM lead to different rainfall patterns?
    """
    print(" -- ASM - PRECIPITATION CORRELATION ANALYSIS --")
    correlation_data = []
    unique_tracks = asm_results_df['track_id'].unique()
    for track_id in unique_tracks:
        track_df = full_track_df[full_track_df['track_id'] == track_id].copy()
        asm_value = asm_results_df[asm_results_df['track_id'] == track_id]['ASM_24h'].values
        if len(asm_value) == 0 or np.isnan(asm_value[0]):
            continue
        asm_24h = asm_value[0]
        # Extract precipitation time series along track
        tp_vals = []
        for _, row in track_df.iterrows():
            try:
                t = pd.to_datetime(row['time'])
                lat, lon = row['latitude'], row['longitude']
                lat_buf = buffer_km / 111.0
                lon_buf = buffer_km / (111.0 * np.cos(np.radians(lat)))
                time_coord = 'time' if 'time' in ds.coords else 'valid_time'
                subset = ds.sel({
                    time_coord: t,
                    'latitude': slice(lat + lat_buf, lat - lat_buf),
                    'longitude': slice(lon - lon_buf, lon + lon_buf)
                }, method='nearest')
                if 'tp' in subset.data_vars:
                    tp = float(subset['tp'].mean().values) * 1000  # m to mm/hr
                    if not np.isnan(tp) and tp >= 0:
                        tp_vals.append(tp)
            except:
                continue
        if len(tp_vals) < 3:
            continue
        tp_mean = np.mean(tp_vals)
        tp_max = np.max(tp_vals)
        tp_std = np.std(tp_vals)
        tp_total = np.sum(tp_vals)
        correlation_data.append({
            'track_id': track_id,
            'ASM_24h': asm_24h,
            'mean_tp': tp_mean,
            'max_tp': tp_max,
            'std_tp': tp_std,
            'total_tp': tp_total,
            'n_tp_samples': len(tp_vals)
        })
        print(f" >  [INFO   ] Track {track_id}                            : \n"
              f"                ASM={asm_24h:.4f},\n"
              f"                Mean TP={tp_mean:.3f} mm/hr,\n"
              f"                Max TP={tp_max:.3f} mm/hr")
    if not correlation_data:
        print(" >  [WARNING ] No correlation data computed")
        return None
    corr_df = pd.DataFrame(correlation_data)

    # Compute correlations
    print(" -- CORRELATION STATISTICS --")
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(corr_df['ASM_24h'], corr_df['mean_tp'])
    print(f"      ASM vs Mean TP                             :")
    print(f"       Pearson r = {pearson_r:.3f}, p-value = {pearson_p:.4f}")
    spearman_r, spearman_p = spearmanr(corr_df['ASM_24h'], corr_df['mean_tp'])
    print(f"       Spearman ρ = {spearman_r:.3f}, p-value = {spearman_p:.4f}")
    # Max precipitation
    pearson_r_max, p_max = pearsonr(corr_df['ASM_24h'], corr_df['max_tp'])
    print(f"      ASM vs Max TP                              :")
    print(f"       Pearson r = {pearson_r_max:.3f}, p-value = {p_max:.4f}")
    # Total precipitation
    pearson_r_total, p_total = pearsonr(corr_df['ASM_24h'], corr_df['total_tp'])
    print(f"      ASM vs Total TP                            :")
    print(f"       Pearson r = {pearson_r_total:.3f}, p-value = {p_total:.4f}")

    # Interpretation
    print(" -- INTERPRETATION --")
    if pearson_r > 0.5 and pearson_p < 0.05:
        print(" >    STRONG POSITIVE correlation: High ASM → Higher rainfall")
        print("     (Wet soils accumulate more moisture, produce more rain)")
    elif pearson_r > 0.3 and pearson_p < 0.05:
        print(" >    MODERATE POSITIVE correlation: Higher ASM tends to produce more rain")
    elif pearson_r < -0.3 and pearson_p < 0.05:
        print(" >    NEGATIVE correlation: Dry soils produce more rain (paradox?)")
        print("       (May indicate system intensity is independent of soil moisture)")
    else:
        print(" >    NO SIGNIFICANT correlation: ASM does not explain rainfall variation")

    # Export results
    export_path = os.path.join(output_dir, 'asm_tp_correlation.csv')
    os.makedirs(output_dir, exist_ok=True)
    corr_df.to_csv(export_path, index=False)
    print(f" >> [SAVED   ] ASM-TP correlation results: {export_path}")
    return corr_df


def compute_asm_duration_correlation(asm_results_df, duration_df, output_dir='output'):
    """
    Correlate antecedent soil moisture with MD duration.
    Research question: Does drier soil cause faster weakening?
    """
    print(" -- ASM - DURATION CORRELATION ANALYSIS --")
    # Merge datasets
    merged_df = asm_results_df.merge(
        duration_df[['track_id', 'duration_hours', 'is_long_lived']],
        on='track_id',
        how='inner'
    )
    if len(merged_df) < 3:
        print(" >   [WARNING ] Insufficient data for correlation (< 3 tracks)")
        return None
    print(f"       Tracks with both ASM and duration data     :          {len(merged_df)}")
    print(f"        Track ID | ASM_24h | Duration (h) | Category")
    for _, row in merged_df.iterrows():
        category = "Long-lived" if row['is_long_lived'] else "Short-lived"
        print(f"      {int(row['track_id']):8d} | {row['ASM_24h']:7.4f} | {row['duration_hours']:12.1f} | {category}")

    # Compute correlations
    print(" -- CORRELATION STATISTICS --")
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(merged_df['ASM_24h'], merged_df['duration_hours'])
    print(f"     ASM vs Duration:")
    print(f"      Pearson r = {pearson_r:.3f}, p-value = {pearson_p:.4f}")
    spearman_r, spearman_p = spearmanr(merged_df['ASM_24h'], merged_df['duration_hours'])
    print(f"      Spearman ρ = {spearman_r:.3f}, p-value = {spearman_p:.4f}")

    # Interpretation
    print(" -- INTERPRETATION --")
    if pearson_r > 0.5 and pearson_p < 0.05:
        print("    STRONG POSITIVE: High ASM → Longer track duration")
        print("     (Wet soils support longer system persistence)")
    elif pearson_r > 0.3 and pearson_p < 0.05:
        print("    MODERATE POSITIVE: Higher ASM tends to extend duration")
    elif pearson_r < -0.3 and pearson_p < 0.05:
        print("    NEGATIVE: Dry soils actually support longer tracks (paradox!)")
    else:
        print("    NO SIGNIFICANT: ASM does not predict duration")

    # By category
    print(" -- COMPARISON: Long-lived vs Short-lived MDs --")
    long_lived = merged_df[merged_df['is_long_lived']]
    short_lived = merged_df[~merged_df['is_long_lived']]
    if len(long_lived) > 0:
        print(f"      Long-lived MDs (≥96h)                      :")
        print(f"        Mean ASM                                 : {long_lived['ASM_24h'].mean():.4f} ± {long_lived['ASM_24h'].std():.4f} m³/m³")
        print(f"        Mean Duration                            : {long_lived['duration_hours'].mean():.1f} ± {long_lived['duration_hours'].std():.1f} hours")
        print(f"        Count                                    :          {len(long_lived)}")
    if len(short_lived) > 0:
        print(f"      Short-lived MDs (<96h)                     :")
        print(f"        Mean ASM                                 : {short_lived['ASM_24h'].mean():.4f} ± {short_lived['ASM_24h'].std():.4f} m³/m³")
        print(f"        Mean Duration                            : {short_lived['duration_hours'].mean():.1f} ± {short_lived['duration_hours'].std():.1f} hours")
        print(f"        Count                                    : {len(short_lived)}")
    # Statistical test
    if len(long_lived) > 0 and len(short_lived) > 0:
        t_stat, t_pval = ttest_ind(long_lived['ASM_24h'], short_lived['ASM_24h'])
        print(f"      T-test (Long vs Short ASM):")
        print(f"        t-statistic = {t_stat:.3f}, p-value = {t_pval:.4f}")
        if t_pval < 0.05:
            print(f"        Significant difference in ASM between categories")
        else:
            print(f"        No significant ASM difference between categories")

    # Export
    export_path = os.path.join(output_dir, 'asm_duration_correlation.csv')
    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(export_path, index=False)
    print(f" >> [SAVED   ] ASM-Duration correlation results  :          {export_path}")
    return merged_df

def create_asm_summary_report(
    asm_results_df,
    corr_tp_df,
    corr_duration_df,
    output_dir='output'
):
    if corr_tp_df is None:
        print(" >  [WARNING ] Skipping TP correlation section: No data available.")
        return

    report_path = os.path.join(output_dir, 'ASM_Analysis_Report.txt')
    with open(report_path, 'w') as f:
        f.write("ANTECEDENT SOIL MOISTURE ANALYSIS - REPORT\n")
        # Section 1: ASM Statistics
        f.write("1. ANTECEDENT SOIL MOISTURE STATISTICS\n")
        f.write(f"  Number of tracks analyzed:                     {len(asm_results_df)}\n")
        f.write(f"  Mean ASM (24h before landfall):                {asm_results_df['ASM_24h'].mean():.4f} m³/m³\n")
        f.write(f"  Std Dev:                                       {asm_results_df['ASM_24h'].std():.4f} m³/m³\n")
        f.write(f"  Min:                                           {asm_results_df['ASM_24h'].min():.4f} m³/m³\n")
        f.write(f"  Max:                                           {asm_results_df['ASM_24h'].max():.4f} m³/m³\n")
        f.write(f"  Range:                                         {asm_results_df['ASM_24h'].max() - asm_results_df['ASM_24h'].min():.4f} m³/m³\n\n")
        # Section 2: ASM-Precipitation Correlation
        f.write("2. ASM - PRECIPITATION CORRELATION\n")
        if corr_tp_df is not None and not corr_tp_df.empty:
            r_mean, p_mean = pearsonr(corr_tp_df['ASM_24h'], corr_tp_df['mean_tp'])
            r_max, p_max = pearsonr(corr_tp_df['ASM_24h'], corr_tp_df['max_tp'])
            r_total, p_total = pearsonr(corr_tp_df['ASM_24h'], corr_tp_df['total_tp'])
            f.write(f"  ASM vs Mean TP:\n")
            f.write(f"    Pearson r: {r_mean:.3f}, p-value: {p_mean:.3f}\n")
            f.write(f"    Significance: {'YES' if p_mean < 0.05 else 'NO'}\n\n")
            f.write(f"  ASM vs Max TP:\n")
            f.write(f"    Pearson r = {r_max:.3f}, p-value = {p_max:.4f}\n")
            f.write(f"    Significance: {'YES' if p_max < 0.05 else 'NO'}\n\n")
            f.write(f"  ASM vs Total TP:\n")
            f.write(f"    Pearson r = {r_total:.3f}, p-value = {p_total:.4f}\n")
            f.write(f"    Significance: {'YES' if p_total < 0.05 else 'NO'}\n\n")
        else:
            f.write("  [STATUS] No valid correlation data (insufficient land points).\n")
        # Section 3: ASM-Duration Correlation
        f.write("3. ASM - DURATION CORRELATION\n")
        if corr_duration_df is not None:
            r_dur, p_dur = pearsonr(corr_duration_df['ASM_24h'], corr_duration_df['duration_hours'])
            f.write(f"  Pearson r = {r_dur:.3f}, p-value = {p_dur:.4f}\n")
            f.write(f"  Significance: {'YES' if p_dur < 0.05 else 'NO'}\n\n")
            long_lived = corr_duration_df[corr_duration_df['is_long_lived']]
            short_lived = corr_duration_df[~corr_duration_df['is_long_lived']]
            f.write(f"  Long-lived MDs (≥96h):\n")
            f.write(f"    Mean ASM = {long_lived['ASM_24h'].mean():.4f} ± {long_lived['ASM_24h'].std():.4f} m³/m³\n")
            f.write(f"    Count = {len(long_lived)}\n\n")
            f.write(f"  Short-lived MDs (<96h):\n")
            f.write(f"    Mean ASM = {short_lived['ASM_24h'].mean():.4f} ± {short_lived['ASM_24h'].std():.4f} m³/m³\n")
            f.write(f"    Count = {len(short_lived)}\n\n")
        else:
            f.write("  [STATUS] No valid correlation data (insufficient land points).\n")
        # Key Findings
        # if abs(r_mean) < 0.3:
        #     f.write("  • ASM shows WEAK linear relationship with precipitation\n")
        # elif r_mean > 0.3 and p_mean < 0.05:
        #     f.write("  • ASM shows POSITIVE relationship with precipitation\n")
        #     f.write("    (Wet soils → Higher rainfall intensity)\n")
        # elif r_mean < -0.3 and p_mean < 0.05:
        #     f.write("  • ASM shows NEGATIVE relationship with precipitation\n")
        #     f.write("    (Dry soils → Higher rainfall intensity)\n")
        # else:
        #     f.write("  • ASM does NOT significantly predict rainfall intensity\n")
        # if abs(r_dur) < 0.3:
        #     f.write("  • ASM shows WEAK relationship with track duration\n")
        # elif r_dur > 0.3 and p_dur < 0.05:
        #     f.write("  • ASM shows POSITIVE relationship with duration\n")
        #     f.write("    (Wetter soils support longer-lived systems)\n")
        # elif r_dur < -0.3 and p_dur < 0.05:
        #     f.write("  • ASM shows NEGATIVE relationship with duration\n")
        #     f.write("    (Drier soils support longer tracks)\n")
        # else:
        #     f.write("  • ASM does NOT significantly predict system duration\n")
    print(f" >> [SAVED   ] Summary report                    :          {report_path}")

def export_antecedent_results(
    asm_df,
    output_dir = 'output'
):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'antecedent_soil_moisture.csv')
    asm_df.to_csv(output_path, index=False)
    print(f" >> [SAVED   ] Antecedent SM results             :          {output_path}")



def plot_antecedent_sm_timeseries(ds, track_df, track_id, buffer_km, output_dir = 'output'):
    """Plot soil moisture evolution before and after landfall."""
    # print(f" >> [DEBUG   ] Starting timeseries plot for track {track_id}")
    # Filter for this track
    track_df = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track_df.empty:
        print(f" >  [ERROR an  ] Track {track_id} not found in DataFrame")
        return
    print(f" >> [INFO an   ] Number of Track points            :          {len(track_df)}")

    # Extract SM along track
    sm_values = []
    times = []
    originates_over_land = False
    # Check if originates over land
    if len(track_df) > 0:
        first_row = track_df.iloc[0]
        originates_over_land = is_point_land(first_row['latitude'], first_row['longitude'])

    # Find landfall
    landfall_idx = -1
    # landfall_time = times[landfall_idx]
    landfall_time = None
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
        print(f" >  [INFO an   ] Track {track_id}                  :          No landfall found. Using track start.")

    # Approximation: 1 degree latitude ~ 111 km
    for _, row in track_df.iterrows():
        try:
            t = pd.to_datetime(row['time'])
            time_coord = 'time' if 'time' in ds.coords else 'valid_time'
            lat_buffer = buffer_km / 111.0
            lat, lon = row['latitude'], row['longitude']
            lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat)))
            ds_at_time = ds.sel({time_coord: t}, method='nearest')
            # Dataset bound
            lat_min = min(lat - lat_buffer, lat + lat_buffer)
            lat_max = max(lat - lat_buffer, lat + lat_buffer)
            lon_min = min(lon - lon_buffer, lon + lon_buffer)
            lon_max = max(lon - lon_buffer, lon + lon_buffer)
            # Clamp to dataset bounds
            lat_min = max(lat_min, float(ds_at_time.latitude.min()))
            lat_max = min(lat_max, float(ds_at_time.latitude.max()))
            lon_min = max(lon_min, float(ds_at_time.longitude.min()))
            lon_max = min(lon_max, float(ds_at_time.longitude.max()))
            if lat_min >= lat_max or lon_min >= lon_max:
                continue

            subset = ds_at_time.sel({
                'latitude': slice(lat_max, lat_min),
                'longitude': slice(lon_min, lon_max)
            })
            if subset.sizes.get('latitude', 0) == 0 or subset.sizes.get('longitude', 0) == 0:
                continue

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

    landfall_idx_in_times = -1
    if times:  # Check times is not empty
        for idx, t in enumerate(times):
            if t >= landfall_time:
                landfall_idx_in_times = idx
                break

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
        alpha=0.9
    )
    ax.fill_between(times, sm_values, alpha=0.2, color='blue')
    # Landfall line and shading
    if landfall_idx_in_times >= 0 and landfall_idx_in_times < len(times):
        landfall_time_plot = times[landfall_idx_in_times]
        ax.axvline(
            landfall_time_plot,
            color=status_color,
            linestyle='--',
            linewidth=2.5,
            alpha=0.8,
            zorder=4
        )
        ax.plot(
            landfall_time_plot,
            sm_values[landfall_idx_in_times],
            'D',
            color=status_color,
            markersize=10,
            zorder=5
        )
        # Shade regions
        ax.axvspan(times[0], landfall_time_plot, alpha=0.1, color='blue')
        if landfall_idx_in_times < len(times) - 1:
            ax.axvspan(landfall_time_plot, times[-1], alpha=0.1, color='red')

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Soil Water Content (m³/m³)', fontsize=12, fontweight='bold')
    ax.set_title(f'Track {track_id}: Antecedent Soil Moisture Time Series\nStatus: {status_text}', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.5, linestyle='--')
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
    print(f" >> [SAVED   ] Antecedent SM timeseries          :          {outpath}")

def plot_sm_tp_map(
    ds,
    track_df,
    track_id,
    hours_before = 24,
    hours_after = 24,
    buffer_km = 300,
    min_extent_deg = 10.0,
    output_dir = 'output',
):
    """Comparison map before/after landfall"""
    print(f" >> [DEBUG   ] Starting plot_sm_tp_map for track {track_id}")
    # Get ALL points for track
    track = track_df[track_df['track_id'] == track_id].sort_values('time')

    track_id = int(track_id)
    if len(track) < 3:
        print(f" >  [WARNING ] Track {track_id} has insufficient points   :          {len(track)}")
        return

    # Find landfall and check origin
    landfall_time = None
    # Find landfall using pre-computed column
    if 'is_land' in track.columns:
        landfall_rows = track[track['is_land'] == True]
        if not landfall_rows.empty:
            landfall_time = pd.to_datetime(landfall_rows.iloc[0]['time'])
    # If not found, use manual check
    if landfall_time is None:
        for i, row in track.iterrows():
            if is_point_land(row['latitude'], row['longitude']):
                landfall_time = pd.to_datetime(row['time'])
                break
    # If still no landfall, warn
    if landfall_time is None:
        print(f" >  [WARNING ] No landfall detected for Track {track_id}. Using start of track for maps.")
        landfall_time = pd.to_datetime(track.iloc[0]['time'])

    # # Check if originates over land
    originates_over_land = False
    if len(track) > 0:
        first_row = track.iloc[0]
        originates_over_land = is_point_land(first_row['latitude'], first_row['longitude'])

    # # Find first landfall
    # landfall_idx = -1
    # for i, row in track.iterrows():
    #     if is_point_land(row['latitude'], row['longitude']):
    #         landfall_idx = i
    #         landfall_time = pd.to_datetime(row['time'])
    #         break
    # if landfall_time is None:
    #     print(f" > [WARNING] No landfall detected for Track {track_id}. Using start of track for maps.")
    #     landfall_idx = 0
    #     landfall_time = pd.to_datetime(track.iloc[0]['time'])

    # # Update AFTER finding landfall
    # has_landfall = landfall_idx != -1
    # if landfall_time is not None:
    #     has_landfall = True
    # if not has_landfall:
    #     if len(track) > 0:
    #         midpoint_idx = len(track) // 2
    #         if midpoint_idx < len(track):
    #             landfall_time = pd.to_datetime(track.iloc[midpoint_idx]['time'])
    #             landfall_idx = midpoint_idx
    #         else:
    #             landfall_time = pd.to_datetime(track.iloc[0]['time'])
    #             landfall_idx = 0
    #     else:
    #         print(f" >  [ERROR an  ] Track {track_id} has no points")
    #         return
    #
    # Determine status text for period label
    if originates_over_land:
        period_label = "Origin"
        status_text = "Originates over land"
    elif landfall_time is None:
        period_label = "Midpoint"
        status_text = "No Landfall"
        # Use midpoint as reference
        midpoint_idx = len(track) // 2
        landfall_time = pd.to_datetime(track.iloc[midpoint_idx]['time'])
    else:
        period_label = "Landfall"
        status_text = "Landfall"

    # Get before/after times
    landfall_time = pd.Timestamp(landfall_time)
    before_start = landfall_time - pd.Timedelta(hours=hours_before)
    after_end = landfall_time + pd.Timedelta(hours=hours_after)

    track['time_dt'] = pd.to_datetime(track['time'])
    # Get points before and after landfall for plotting
    track_window = track[
        (track['time_dt'] >= before_start) &
        (track['time_dt'] <= after_end)
    ].copy()
    # Split into before/after
    track_before = track_window[pd.to_datetime(track_window['time']) < landfall_time]
    track_after = track_window[pd.to_datetime(track_window['time']) >= landfall_time]
    # print(f" >> [INFO an   ] Track window filtering:")
    print(f" >> [DEBUG]      Landfall time: {landfall_time}")
    print(f" >> [DEBUG]      Window: {before_start} to {after_end}")
    print(f" >> [DEBUG]      Track points total: {len(track)}")
    print(f" >> [DEBUG]      Points in window: {len(track_window)}")
    print(f" >> [DEBUG]      Points before landfall: {len(track_before)}")
    print(f" >> [DEBUG]      Points after landfall: {len(track_after)}")
    if len(track_window) > 0:
        print(f" >> [DEBUG] First point in window: {track_window.iloc[0]['time']}")
        print(f" >> [DEBUG] Last point in window: {track_window.iloc[-1]['time']}")
        print(f" >> [DEBUG] Time span in window: {(pd.to_datetime(track_window.iloc[-1]['time']) - pd.to_datetime(track_window.iloc[0]['time'])).total_seconds()/3600:.1f} hours")
    elif len(track_window) < 0 :
        print(f" >> [DEBUG] No points in time window!")
        print(f" >> [DEBUG] Track times: {track['time'].iloc[0]} to {track['time'].iloc[-1]}")
        print(f" >> [DEBUG] Landfall time: {landfall_time}")
        print(f" >> [DEBUG] Before start: {before_start}")
        print(f" >> [DEBUG] After end: {after_end}")
    else :
        print(f" >  [WARNING ] No track points in time window for Track {track_id}")
        return False

    # if len(track_before) == 0 and len(track) > 0:
    #     # track_before = track.iloc[:1]
    #     track_before = track.iloc[:max(1, len(track)//3)]
    # if len(track_after) == 0 and len(track) > 0:
    #     # track_after = track.iloc[-1:]
    #     track_after = track.iloc[-max(1, len(track)//3):]

    # Get extent
    # all_lats = pd.concat([track_before['latitude'], track_after['latitude']], ignore_index=True)
    # all_lons = pd.concat([track_before['longitude'], track_after['longitude']], ignore_index=True)
    all_lats = track['latitude']
    all_lons = track['longitude']
    if all_lats.empty or all_lons.empty:
        print(f" >  [WARNING ] No valid track points for extent calculation")
        return False

    # Calculate track centroid
    centroid_lat = all_lats.mean()
    centroid_lon = all_lons.mean()
    # Calculate initial extent based on track
    lat_range = max(5.0, all_lats.max() - all_lats.min()) * 2.0  # double the lat range
    lon_range = max(5.0, all_lons.max() - all_lons.min()) * 2.0  # double the lon range
    # Add extra padding (20% of range)
    lat_range = lat_range * 1.2
    lon_range = lon_range * 1.2
    # Apply minimum extent
    lat_range = max(lat_range, min_extent_deg)
    lon_range = max(lon_range, min_extent_deg)

    # Calculate final bounds
    min_lat = centroid_lat - lat_range/2
    max_lat = centroid_lat + lat_range/2
    min_lon = centroid_lon - lon_range/2
    max_lon = centroid_lon + lon_range/2

    # Clamp to clipped domain bounds
    min_lat = max(min_lat, clipped_min_lat)
    max_lat = min(max_lat, clipped_max_lat)
    min_lon = max(min_lon, clipped_min_lon)
    max_lon = min(max_lon, clipped_max_lon)

    # Buffer in degrees (for data extraction,)
    buffer_deg_lat = buffer_km / 111.0
    buffer_deg_lon = buffer_km / (111.0 * np.cos(np.radians(centroid_lat)))
    # mean_lat = track_df['latitude'].mean()
    # buffer_deg_lon = buffer_km / (111.0 * np.cos(np.radians(mean_lat)))
    time_coord = 'time' if 'time' in ds.coords else 'valid_time'

    print(f" >> [INFO an   ] Map extent for track {track_id}   :")
    print(f"                Latitude                         :          [{min_lat:.2f}, {max_lat:.2f}] deg (range: {lat_range:.1f}°)")
    print(f"                Longitude                        :          [{min_lon:.2f}, {max_lon:.2f}] deg (range: {lon_range:.1f}°)")
    print(f"                Centroid                         :          ({centroid_lat:.2f}, {centroid_lon:.2f})")
    print(f"                Track points                     :          {len(track)}")
    print(f"                Before landfall                  :          {len(track_before)} points")
    print(f"                After landfall                   :          {len(track_after)} points")
    print(f"                Clipped to domain                :          {clipped_min_lat}N-{clipped_max_lat}N, {clipped_min_lon}E-{clipped_max_lon}E")

    # Check if extent is reasonable
    if (max_lat - min_lat) < 2.0 or (max_lon - min_lon) < 2.0:
        print(f" >  [WARNING ] Map extent is too small! Forcing larger extent.")
        # Force minimum size
        min_lat = centroid_lat - 5.0
        max_lat = centroid_lat + 5.0
        min_lon = centroid_lon - 5.0
        max_lon = centroid_lon + 5.0
        # Still clamp to clipped domain
        min_lat = max(min_lat, clipped_min_lat)
        max_lat = min(max_lat, clipped_max_lat)
        min_lon = max(min_lon, clipped_min_lon)
        max_lon = min(max_lon, clipped_max_lon)

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
        tp_before_total = ds_before['tp'].sum(dim=time_coord) * 1000  # m to mm
        tp_after_total = ds_after['tp'].sum(dim=time_coord) * 1000   # m to mm
        tp_change = tp_after_total - tp_before_total 

        # Get antecedent SM
        asm_24h = compute_antecedent_sm(ds, track_df, track_id, 24, 6, 100)
        asm_display = f"{asm_24h:.4f} m³/m³" if not np.isnan(asm_24h) else "N/A"

        # Create 2x4 subplot layout
        fig, axes = plt.subplots(
            2,
            4,
            figsize=(22, 14),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        # For soil moisture
        sm_all_values = np.concatenate([
            sm_before_mean.values.flatten(),
            sm_after_mean.values.flatten()
        ])
        sm_all_values = sm_all_values[~np.isnan(sm_all_values)]
        sm_vmin = np.percentile(sm_all_values, 5) if len(sm_all_values) > 0 else 0
        sm_vmax = np.percentile(sm_all_values, 95) if len(sm_all_values) > 0 else 1
        # For Total Precipitation
        tp_all_values = np.concatenate([
            tp_before_total.values.flatten(),
            tp_after_total.values.flatten()
        ])
        tp_all_values = tp_all_values[~np.isnan(tp_all_values)]
        tp_vmin = 0  # Always start at 0 for precipitation
        tp_vmax = np.percentile(tp_all_values, 95) if len(tp_all_values) > 0 else 10
        # For change plots
        sm_change_values = sm_change.values.flatten()
        sm_change_values = sm_change_values[~np.isnan(sm_change_values)]
        sm_change_vmax = max(abs(np.nanpercentile(sm_change_values, 5)),
                           abs(np.nanpercentile(sm_change_values, 95))) if len(sm_change_values) > 0 else 0.1
        tp_change_values = tp_change.values.flatten()
        tp_change_values = tp_change_values[~np.isnan(tp_change_values)]
        tp_change_vmax = max(abs(np.nanpercentile(tp_change_values, 5)),
                           abs(np.nanpercentile(tp_change_values, 95))) if len(tp_change_values) > 0 else 10

        period_titles = [
            f'Before {period_label} ({hours_before}h)',
            f'After {period_label} ({hours_after}h)',
            'Change (After - Before)'
        ]

        # Row 0: Soil moisture maps
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
                ax.set_title(f'Soil Moisture\n{title}', fontsize=12, fontweight='bold')
                continue

            if i == 2:  # Change plot for soil moisture
                sm_change_vmax = max(abs(data.values.min()), abs(data.values.max()))
                im = ax.contourf(
                    data.longitude,
                    data.latitude,
                    data.values,
                    levels=21,
                    cmap='RdBu',
                    vmin=-sm_change_vmax,
                    vmax=sm_change_vmax,
                    transform=ccrs.PlateCarree(),
                    extend='both'
                )
            else:  # Before/After plots
                im = ax.contourf(
                    data.longitude,
                    data.latitude,
                    data.values,
                    levels=21,
                    cmap='YlGnBu',
                    vmin=sm_vmin,
                    vmax=sm_vmax,
                    transform=ccrs.PlateCarree(),
                    extend='both'
                )
            # Contour lines
            if not np.all(np.isnan(data.values)):
                contours = ax.contour(
                    data.longitude,
                    data.latitude,
                    data.values,
                    levels=8,
                    colors='black',
                    linewidths=0.5,
                    alpha=0.5,
                    transform=ccrs.PlateCarree()
                )
                ax.clabel(contours, inline=True, fontsize=7, fmt='%1.3f')

            # Plot track segments
            if i == 0 and len(track_before) > 0:
                ax.plot(
                    track_before['longitude'],
                    track_before['latitude'],
                    'b-',
                    linewidth=2.5,
                    transform=ccrs.PlateCarree(),
                    zorder=5
                )
            elif i == 1 and len(track_after) > 0:
                ax.plot(
                    track_after['longitude'],
                    track_after['latitude'],
                    'r-',
                    linewidth=2.5,
                    transform=ccrs.PlateCarree(),
                    zorder=5
                )
            elif i == 2 and len(track_window) > 0:
                # Plot windowed track
                ax.plot(
                    track_window['longitude'],
                    track_window['latitude'],
                    'k-',
                    linewidth=2.5,
                    alpha=0.9,
                    transform=ccrs.PlateCarree(),
                    zorder=4
                )
                # Mark landfall point
                # if 0 <= landfall_idx < len(track):
                    # point = track.iloc[landfall_idx]
                if len(track_before) > 0:
                    point = track_before.iloc[-1]
                    ax.plot(
                        point['longitude'],
                        point['latitude'],
                        'D',
                        color='yellow',
                        markersize=10,
                        markeredgecolor='black',
                        markeredgewidth=1,
                        transform=ccrs.PlateCarree(),
                        zorder=6,
                        label='Landfall'
                    )

            ax.coastlines(resolution='50m', linewidth=0.8)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.1)
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            ax.set_title(f'Soil Moisture\n{title}', fontsize=12, fontweight='bold')
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, pad=0.02,shrink=0.7)
            if i == 2:
                cbar.set_label('Δ Soil Moisture (m³/m³)', fontsize=10)
            else:
                cbar.set_label('Soil Moisture (m³/m³)', fontsize=10)

        # SM Stats in 4th column of row 0
        ax_sm_stats = axes[0, 3]
        ax_sm_stats.axis('off')

        sm_before_val = sm_before_mean.mean().item() if sm_before_mean.size > 0 else np.nan
        sm_after_val = sm_after_mean.mean().item() if sm_after_mean.size > 0 else np.nan
        sm_change_val = sm_change.mean().item() if sm_change.size > 0 else np.nan

        sm_stats_text = (
            f"Soil moisture statistics\n\n"
            f"Mean before:        {sm_before_val:.4f} m³/m³\n"
            f"Mean after:         {sm_after_val:.4f} m³/m³\n"
            f"Mean change:        {sm_change_val:+.4f} m³/m³\n"
            f"Antecedent (24h):   {asm_display}\n\n"
            f"Track status:       {status_text}\n"
            f"Landfall time:      {landfall_time.strftime('%Y-%m-%d %H:%M')}"
            # f"\nMap extent:         {lon_range:.1f}° × {lat_range:.1f}°"
        )
        ax_sm_stats.text(
            0.05,
            0.95,
            sm_stats_text,
            transform=ax_sm_stats.transAxes,
            fontsize=11,
            family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', alpha=0.3, facecolor='lightblue')
        )

        # Row 1: Total Precipitation maps
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
                ax.set_title(f'Total Precipitation\n{title}', fontsize=12, fontweight='bold')
                continue

            if i == 2:  # Change plot for Total Precipitation
                im = ax.contourf(
                    data.longitude,
                    data.latitude,
                    data.values,
                    levels=21,
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
                    levels=21,
                    cmap='YlOrRd',
                    vmin=tp_vmin,
                    vmax=tp_vmax,
                    transform=ccrs.PlateCarree(),
                    extend='max'
                )
            # Contour lines
            if not np.all(np.isnan(data.values)):
                contours = ax.contour(
                    data.longitude,
                    data.latitude,
                    data.values,
                    levels=8,
                    colors='black',
                    linewidths=0.5,
                    alpha=0.5,
                    transform=ccrs.PlateCarree()
                )
                ax.clabel(contours, inline=True, fontsize=7, fmt='%1.0f')
            # Plot track segments
            if i == 0 and len(track_before) > 0:
                ax.plot(
                    track_before['longitude'],
                    track_before['latitude'],
                    'b-',
                    linewidth=2.5,
                    transform=ccrs.PlateCarree(),
                    zorder=5
                )
            elif i == 1 and len(track_after) > 0:
                ax.plot(
                    track_after['longitude'],
                    track_after['latitude'],
                    'r-',
                    linewidth=2.5,
                    transform=ccrs.PlateCarree(),
                    zorder=5
                )
            elif i == 2 and len(track_window) > 0:
                # Plot windowed track
                ax.plot(
                    track_window['longitude'],
                    track_window['latitude'],
                    'k-',
                    linewidth=2.5,
                    alpha=0.9,
                    transform=ccrs.PlateCarree(),
                    zorder=4
                )
                # Mark landfall point
                # if 0 <= landfall_idx < len(track):
                    # point = track.iloc[landfall_idx]
                if len(track_before) > 0:
                    point = track_before.iloc[-1]
                    ax.plot(
                        point['longitude'],
                        point['latitude'],
                        'D',
                        color='yellow',
                        markersize=10,
                        markeredgecolor='black',
                        markeredgewidth=1,
                        transform=ccrs.PlateCarree(),
                        zorder=6,
                        label='Landfall'
                    )

            ax.coastlines(resolution='50m', linewidth=0.8)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.2)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', alpha=0.5)
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.1)
            ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            ax.set_title(f'Total Precipitation\n{title}', fontsize=12, fontweight='bold')
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.7)
            if i == 2:
                cbar.set_label('Δ Precipitation (mm)', fontsize=10)
            else:
                cbar.set_label('Total Precip (mm)', fontsize=10)

        # TP Stats in 4th column of row 1
        ax_tp_stats = axes[1, 3]
        ax_tp_stats.axis('off')

        tp_before_val = tp_before_total.mean().item() if tp_before_total.size > 0 else np.nan
        tp_after_val = tp_after_total.mean().item() if tp_after_total.size > 0 else np.nan
        tp_change_val = tp_change.mean().item() if tp_change.size > 0 else np.nan

        tp_stats_text = (
            f"Total Precipitation statistics\n\n"
            f"Mean before:      {tp_before_val:.1f} mm\n"
            f"Mean after:       {tp_after_val:.1f} mm\n"
            f"Mean change:      {tp_change_val:+.1f} mm\n"
            f"Max before:       {tp_before_total.max().item():.1f} mm\n"
            f"Max after:        {tp_after_total.max().item():.1f} mm\n"
            f"Total before:     {tp_before_total.sum().item():.1f} mm\n"
            f"Total after:      {tp_after_total.sum().item():.1f} mm"
        )
        ax_tp_stats.text(
            0.05,
            0.95,
            tp_stats_text,
            transform=ax_tp_stats.transAxes,
            fontsize=11,
            family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', alpha=0.3, facecolor='lightblue')
        )
        # Main title
        fig.suptitle(
            f'Track {track_id}: Soil Moisture & Precipitation Evolution\n'
            f'Before/After {period_label} ({hours_before}h before → {hours_after}h after)',
            fontsize=16,
            fontweight='bold',
            y=0.8,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)

        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f'track_{int(track_id)}_sm_tp_before_after.png')
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f" >  [SAVED   ] SM/TP before/after map            :          {outpath}")
        print(f" >  [DEBUG   ] Extent                            :          {min_lon:.1f}E to {max_lon:.1f}E, {min_lat:.1f}N to {max_lat:.1f}N")
        return True  # Signal success
    except Exception as e:
        print(f" >  [ERROR an  ] Plotting before/after maps for track {track_id}: {e}")
        traceback.print_exc()
        return False   # Signal failure


def analyze_landfall_fluxes(ds, track_df, track_id, hours_before=24, hours_after=24):
    track_id = int(track_id)
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
