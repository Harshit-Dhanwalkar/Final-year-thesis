# This script computes soil moisture conditions in the 24 hours BEFORE landfall.
# Used to understand the moisture state of the land when the MD arrives.
# Is the antecedent soil moisture (wet vs dry) related to rainfall/flooding intensity during MD passage?

# Dry antecedent conditions -> Soil can absorb a lot of the incoming rain -> lower immediate flood risk (more infiltration).
# Wet antecedent conditions -> Soil is already near saturation → much more of the MD's heavy rainfall becomes surface runoff -> higher flood potential even if the total rainfall amount is not extreme.

# FIX: before after 24 landfall plots (plot_sm_tp_map) shows full track rather than just clipped track for that duration
# FIX: even tho clipped spacial domain to 10N70E to 30N95E yet before/after plots uses full extend
# TODO: runs multiple t-tests without correction
# from scipy.stats import pearsonr
# from statsmodels.stats.multitest import multipletests
# # Collect all p-values
# pvalues = [p_mean, p_max, p_total]
# tests = ['ASM vs Mean TP', 'ASM vs Max TP', 'ASM vs Total TP']
# # Apply Benjamini-Hochberg correction
# reject, p_corrected, _, _ = multipletests(pvalues, method='fdr_bh', alpha=0.05)
# # Report corrected values
# for test, p_orig, p_corr, sig in zip(tests, pvalues, p_corrected, reject):
#     std_print('INFO', 'an', f'{test}', f'p_uncorrected={p_orig:.4f}, p_corrected={p_corr:.4f}, sig={sig}')

import os
import numpy as np
import pandas as pd
from pandas.core import common
import xarray as xr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr, spearmanr, ttest_ind
from statsmodels.stats.multitest import multipletests
import seaborn as sns
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

def compute_antecedent_swvl1(ds, track_df, track_id, hours_before_landfall, exclude_last_hours, buffer_km = 100):
    track_id = int(track_id)
    ds, time_coord = get_time_coord(ds)
    # Convert dataset times to datetime
    ds_times = pd.to_datetime(ds[time_coord].values)
    if track_df is None:
        std_print('ERROR', 'an', f'Track {track_id}', 'track_df is None')
        return np.nan
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track.empty:
        return np.nan

    # Find first landfall point
    origin_type = find_origin_type(track)
    if origin_type == 'land_originated':
        std_print('INFO', 'an', f'Track {track_id} : Land-originated', 'Skipping ASM (no ocean→land transition)')
        return {'value': np.nan, 'reason': 'land_originated'}

    # Ensure is_land column exists
    if 'is_land' not in track.columns:
        track['is_land'] = track.apply(
            lambda row: is_point_land(row['latitude'], row['longitude']),
            axis=1
        )

    # Now find landfall
    landfall_time = find_landfall_time(track)
    if landfall_time is None:
        std_print('WARN', 'an', f'Track {track_id}', 'No landfall detected')
        return {'value': np.nan, 'reason': 'no_landfall'}

    std_print('INFO', 'an', f'Track {track_id} Landfall detected', landfall_time.strftime('%Y-%m-%d %H:%M'))
    # std_print('DEBUG', 'an', f'Dataset time range : {ds_times.min()} to {ds_times.max()}{landfall_time}')
    # std_print('DEBUG', 'an', f'Time window : {t_start} to {t_end}')

    swvl1_vals = []
    ocean_cells_excluded = 0
    land_cells_included = 0

    # Define time window: 24 hours before, but exclude last 6 hours
    t_start = landfall_time - pd.Timedelta(hours=hours_before_landfall)
    t_end = landfall_time - pd.Timedelta(hours=exclude_last_hours)

    # Adjust window to dataset bounds
    if t_start >= t_end:  # No valid window
        std_print('WARN', 'an', f'Track {track_id}', 'No valid time window (t_start >= t_end)')
        return np.nan

    if t_start < ds_times.min():
        t_start = ds_times.min()
    if t_end > ds_times.max():
        t_end = ds_times.max()
    if t_start >= t_end:
        std_print('WARN', 'an', f'Track {track_id}','No valid time window')
        return np.nan
    std_print('INFO', 'an', f'Track {track_id} Time window', f'{t_start} to {t_end}')

    # Filter track points in this window
    subset_track = track[(pd.to_datetime(track['time']) >= t_start) & (pd.to_datetime(track['time']) <= t_end)]

    if subset_track.empty:
        std_print('WARN', 'an', f'Track {track_id}', 'No track points in time window')
        return np.nan
    std_print('INFO', 'an', f'Track {track_id} Points in time window', f'{len(subset_track)} points')

    # Extract soil moisture along track in the time window
    for _, row in subset_track.iterrows():
        try:
            lat, lon = row['latitude'], row['longitude']
            t = pd.to_datetime(row['time'])
            # Check if time exists in dataset
            if t not in ds_times:
                # Find nearest time
                nearest_idx = np.argmin(np.abs(ds_times - t))
                t = ds_times[nearest_idx]
                std_print('INFO', 'an', 'Using nearest time', t)

            # Extract spatial average of soil moisture
            subset = extract_spatial_subset(ds, lat, lon, t, buffer_km=buffer_km, time_coord=time_coord)
            if 'swvl1' in subset.data_vars:
                # Check spatial coverage
                valid_cells = (~np.isnan(subset['swvl1'].values)).sum()
                total_cells = subset['swvl1'].size
                coverage = valid_cells / total_cells if total_cells > 0 else 0
                if coverage < 0.5:  # Require >50% valid data
                    ocean_cells_excluded += 1
                    continue
                swvl1 = subset['swvl1'].mean().values
                if not np.isnan(swvl1) and 0.01 < swvl1 < 0.70:
                    swvl1_vals.append(float(swvl1))

            if subset is None:
                ocean_cells_excluded += 1
                std_print('WARN', 'an', f'Track {track_id}', 'Subset is None')
                continue

            # if 'swvl1' in subset.data_vars:
            #     swvl1 = subset['swvl1'].mean().values
            #     if not np.isnan(swvl1) and 0.01 < swvl1 < 0.70:
            #         swvl1_vals.append(float(swvl1))
            #         land_cells_included += 1
            #     else:
            #         ocean_cells_excluded += 1
            # else:
            #     ocean_cells_excluded += 1

        except Exception as e:
            ocean_cells_excluded += 1
            std_print('ERROR', 'an', f'Track {track_id} Processing point failed', str(e))
            continue

    # Diagnostic
    # std_print('INFO', 'an', f'Track {track_id} Extraction statistics')
    # std_print('INFO', 'an', 'Land cells included', land_cells_included)
    # std_print('INFO', 'an', 'Ocean/invalid cells excluded', ocean_cells_excluded)
    # inclusion_rate = 100*land_cells_included/(land_cells_included+ocean_cells_excluded) if (ocean_cells_excluded+land_cells_included) > 0 else 0
    # std_print('INFO', 'an', 'Inclusion rate', f'{inclusion_rate:.1f}', '%')

    # Return swvl1_mean or NaN if insufficient data
    if swvl1_vals:
        aswvl1_mean = np.mean(swvl1_vals)
        aswvl1_std = np.std(swvl1_vals)
        std_print('INFO', 'an', f'Track {track_id} Aswvl1', f'{aswvl1_mean:.4f} ± {aswvl1_std:.4f} m³/m³')
        return aswvl1_mean
    else:
        std_print('WARN', 'an', f'Track {track_id}', 'No valid land-based SWVL1 Values extracted')
        return np.nan

def compute_antecedent_swvl1_multiwindow(ds, track_df, track_id, buffer_km=100, land_only=False):
    """ Compute soil moisture at multiple windows for transition analysis"""
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
        std_print('INFO', 'an', f'Computing {window_name}', window_config['description'])
        std_print('INFO', 'an', 'Purpose', window_config['purpose'])
        # hours_before = window_config['hours_before']
        # hours_after = window_config['hours_after']
        is_post = window_config.get('is_post_landfall', False)
        if is_post:
            aswvl1 = compute_antecedent_swvl1_post_landfall(
                ds=ds,
                track_df=track_df,
                track_id=track_id,
                hours_after_landfall=window_config['hours_after'],
                buffer_km=buffer_km,
            )
        else:
            aswvl1 = compute_antecedent_swvl1(
                ds=ds,
                track_df=track_df,
                track_id=track_id,
                hours_before_landfall=window_config['hours_before'],
                exclude_last_hours=window_config['hours_after'],
                buffer_km=buffer_km,
            )
        results[window_name] = {
            'value': aswvl1,
            'hours_before': window_config['hours_before'],
            'hours_after': window_config['hours_after'],
            'description': window_config['description']
        }
    return results

def compute_antecedent_swvl1_post_landfall(ds, track_df, track_id, hours_after_landfall, buffer_km=100, land_only=False):
    """
    Compute soil moisture AFTER landfall (land phase).
    hours_after_landfall: How many hours AFTER landfall to analyze
    """
    track_id = int(track_id)
    ds, time_coord = get_time_coord(ds)
    if track_df is None:
        std_print('ERROR', 'an', f'track_df is None for track {track_id}')
        return np.nan
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track.empty:
        return np.nan

    # Find first landfall point
    origin_type = find_origin_type(track)
    if origin_type == 'land_originated':
        std_print('INFO', 'an', f'Track {track_id} : Land-originated', 'Skipping ASM (no ocean→land transition)')
        return np.nan

    # Ensure is_land column exists
    # if 'is_land' not in track.columns:
    #     track['is_land'] = track.apply(lambda row: is_point_land(row['latitude'], row['longitude']), axis=1)
    landfall_time = find_landfall_time(track)
    if landfall_time is None and 'is_land' in track.columns:
        landfall_rows = track[track['is_land'] == True]
        if not landfall_rows.empty:
            landfall_time = pd.to_datetime(landfall_rows.iloc[0]['time'])
    if landfall_time is None:
        std_print('WARN', 'an', f'Track {track_id} : No landfall detected', 'Returning NaN')
        return np.nan

    std_print('INFO', 'an', f'Track {track_id} : Landfall detected', landfall_time.strftime('%Y-%m-%d %H:%M'))

    swvl1_vals = []
    ocean_cells_excluded = 0
    land_cells_included = 0

    # Convert dataset times to datetime
    ds_times = pd.to_datetime(ds[time_coord].values)
    # Define time window: FROM landfall TO landfall + hours_after
    t_start = landfall_time
    t_end = landfall_time + pd.Timedelta(hours=hours_after_landfall)

    # Adjust window to dataset bounds
    if t_start < ds_times.min():
        std_print('WARN', 'an', f'Track {track_id} t_start < ds min', f'{t_start} < {ds_times.min()}')
        t_start = ds_times.min()
    if t_end > ds_times.max():
        std_print('WARN', 'an', f'Track {track_id} t_end > ds max', f'{t_end} > {ds_times.max()}')
        t_end = ds_times.max()
    if t_start >= t_end:
        std_print('WARN', 'an', f'Track {track_id}', 'No valid post-landfall window (t_start >= t_end)')
        std_print('INFO', 'an', f'Track {track_id} Landfall detected', landfall_time)
        std_print('INFO', 'an', f'Track {track_id} Dataset range {ds_times.min()} to {ds_times.max()}')
        return np.nan
    std_print('INFO', 'an', f'Track {track_id} Time window', f'{t_start} to {t_end}')

    # Filter track points in this window AND over land
    subset_track = track[(pd.to_datetime(track['time']) >= t_start) & (pd.to_datetime(track['time']) <= t_end)]
    if subset_track.empty:
        std_print('WARN', 'an', f'Track {track_id}', 'No points in post-landfall tim e window')
        return np.nan

    std_print('INFO', 'an', f'Track {track_id} Points in post-landfall window', len(subset_track))

    # Extract soil moisture along track in the time window
    for _, row in subset_track.iterrows():
        try:
            lat, lon = row['latitude'], row['longitude']
            t = pd.to_datetime(row['time'])
            # Check land-only constraint
            if land_only and not is_point_land(lat, lon):
                ocean_cells_excluded += 1
                continue
            lat_buf, lon_buf = calculate_buffer_deg(lat, buffer_km)
            # Find nearest time in dataset
            if t not in ds_times:
                nearest_idx = np.argmin(np.abs(ds_times - t))
                t = ds_times[nearest_idx]
            ds_at_time = ds.sel({time_coord: t}, method='nearest')
            try:
                # Extract spatial average of soil moisture
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
                subset = ds_at_time.sel({'latitude': slice(lat_max, lat_min), 'longitude': slice(lon_min, lon_max)})
                if subset.sizes.get('latitude', 0) == 0 or subset.sizes.get('longitude', 0) == 0:
                    ocean_cells_excluded += 1
                    continue

                if 'swvl1' in subset.data_vars:
                    swvl1 = subset['swvl1'].mean().values
                    if not np.isnan(swvl1) and 0.01 < swvl1 < 0.70:
                        swvl1_vals.append(float(swvl1))
                        land_cells_included += 1
                    else:
                        ocean_cells_excluded += 1
            except Exception as e:
                ocean_cells_excluded += 1
                std_print('ERROR', 'an', f'Track {track_id}', str(e))
                continue
        except Exception as e:
            ocean_cells_excluded += 1
            std_print('ERROR', 'an', f'Track {track_id}', str(e))
            continue

    std_print('DEBUG', 'an', f'Track {track_id} Post-landfall extraction')
    std_print('DEBUG', 'an', 'Land cells included', land_cells_included)
    std_print('DEBUG', 'an', 'Ocean/invalid cells excluded', ocean_cells_excluded)

    # Return swvl1_mean or NaN if insufficient data
    if swvl1_vals:
        aswvl1_mean = np.mean(swvl1_vals)
        std_print('INFO', 'an', f'Track {track_id} Post-landfall swvl1 (6h)', f'{aswvl1_mean:.4f}  m³/m³')
        return aswvl1_mean
    else:
        std_print('WARN', 'an', f'Track {track_id}', 'No valid post-landfall SWVL1 values extracted')
        return np.nan

def export_multiwindow_results(track_id, multiwindow_results, output_dir):
    """Export multi-window ASM results for correlation analysis"""
    track_id = int(track_id)
    export_data = {'track_id': track_id}

    for window_name, window_data in multiwindow_results.items():
       value = window_data['value']
       if isinstance(value, dict):
            value = value.get('value', np.nan)
       export_data[f'{window_name}_asm'] = value
       export_data[f'{window_name}_hours_before'] = window_data['hours_before']

    asm_multiwindow_df = pd.DataFrame([export_data])
    output_path = os.path.join(output_dir, f'asm_multiwindow_results_track_{track_id}.csv')
    asm_multiwindow_df.to_csv(output_path, index=False)
    export_results_to_csv(asm_multiwindow_df, output_dir, f'antecedent_soil_moisture_track_{track_id}', 'an')
    std_print('SAVED', 'an', 'Multi-window ASM CSV', output_path)
    std_print('INFO', 'an', f'Summary for Track {track_id}')
    for window_name, window_data in multiwindow_results.items():
        val = window_data.get('value', np.nan)
        val_str = f"{val:7.4f}" if isinstance(val, (int, float)) else "N/A"
        std_print(None, None, f'   {window_name}', f'{val_str}  m³/m³')
    return asm_multiwindow_df

def compute_antecedent_swvl1_all_tracks(ds, full_track_df, hours_before_landfall, exclude_last_hours, buffer_km = 100):
    unique_tracks = sorted(full_track_df['track_id'].unique())
    results = []

    if full_track_df is None or (isinstance(full_track_df, pd.DataFrame) and full_track_df.empty):
        std_print('WARN', 'an', 'No track data available for ASM analysis')
        return pd.DataFrame()

    for track_id in unique_tracks:
        std_print('INFO', 'an', f'Processing track {track_id}')

        # Get track data
        track = full_track_df[full_track_df['track_id'] == track_id].sort_values('time')
        # Check if track has landfall
        origin_type = find_origin_type(track)
        asm_results = {}
        if origin_type == 'land_originated':
            # FIX:
            # For land-originated tracks, won't be "antecedent" in the traditional sense
            std_print('INFO', 'an', f'Track {track_id} Land-originated - computing soil moisture at start')
            # Compute soil moisture at track start
            asm_24h = compute_swvl1_at_start(ds, track, buffer_km)
            has_landfall = False
        else:
            for exclusion_hours in [0, 3, 6, 12]:
                asm_24h = compute_antecedent_swvl1(ds, full_track_df, track_id, hours_before_landfall=24, exclude_last_hours=exclusion_hours, buffer_km=100)
                asm_results[f'ASM_excl_{exclusion_hours}h'] = asm_24h
                has_landfall = pd.notna(asm_24h) and not isinstance(asm_24h, dict)
        results.append({
            'track_id': track_id,
            'ASM_24h_excl_0h': asm_results.get('ASM_excl_0h'),
            'ASM_24h_excl_3h': asm_results.get('ASM_excl_3h'),
            'ASM_24h_excl_6h': asm_results.get('ASM_excl_6h'),
            'ASM_24h_excl_12h': asm_results.get('ASM_excl_12h'),
            'has_landfall': has_landfall,
            'origin_type': origin_type
        })
        if not pd.isna(asm_24h):
            std_print('INFO', 'an', f'Track {track_id} ASM_24h', f'{asm_24h:.4f} m³/m³')
        else:
            std_print('WARN', 'an', f'Track {track_id}', 'No landfall or insufficient data')
    return pd.DataFrame(results)

# Helper for compute_antecedent_swvl1_all_tracks function
def compute_swvl1_at_start(ds, track, buffer_km=100):
    """Compute soil moisture at the start of a land-originated track"""
    if track.empty:
        return np.nan
    # Use the first 3 points of the track
    start_points = track.head(3)
    swvl1_vals = []
    for _, row in start_points.iterrows():
        try:
            t = pd.to_datetime(row['time'])
            lat, lon = row['latitude'], row['longitude']
            # Check if point is over land
            if not is_point_land(lat, lon):
                continue
            subset = extract_spatial_subset(ds, lat, lon, t, buffer_km=buffer_km)
            if subset is None or 'swvl1' not in subset.data_vars:
                continue
            swvl1 = float(subset['swvl1'].mean().values)
            if not np.isnan(swvl1) and 0.01 < swvl1 < 0.70:
                swvl1_vals.append(swvl1)
        except Exception:
            continue
    return np.mean(swvl1_vals) if swvl1_vals else np.nan

def compute_asm_tp_correlation(ds, full_track_df, asm_results_df, buffer_km=100, output_dir='output'):
    """
    Correlate antecedent soil moisture with precipitation during track passage.
    Question: Does low ASM lead to different rainfall patterns?
    """
    std_print('INFO', 'an', 'ASM - Precipitation correlation analysis')
    correlation_data = []

    unique_tracks = asm_results_df['track_id'].unique()
    for track_id in unique_tracks:
        track_df = full_track_df[full_track_df['track_id'] == track_id].copy()
        asm_row = asm_results_df[asm_results_df['track_id'] == track_id]
        if asm_row.empty:
            continue
        asm_col = 'ASM_24h_excl_6h'
        # Check if the column exists, if not try other available columns
        if asm_col not in asm_row.columns:
            # Find any available ASM_24h column
            available_cols = [col for col in asm_row.columns if 'ASM_24h' in col]
            if not available_cols:
                continue  # Skip this track if no ASM column found
            asm_col = available_cols[0]  # Use the first available
        asm_value = asm_row[asm_col].iloc[0]
        if isinstance(asm_value, dict):
            asm_value = asm_value.get('value', np.nan)
        if pd.isna(asm_value):
            continue
        asm_24h = float(asm_value)
        ####
        # Extract precipitation time series along track
        tp_vals = []
        ds, time_coord = get_time_coord(ds)
        ds_times = pd.to_datetime(ds[time_coord].values)
        for _, row in track_df.iterrows():
            try:
                lat, lon = row['latitude'], row['longitude']
                t = pd.to_datetime(row['time'])
                lat_buf, lon_buf = calculate_buffer_deg(lat, buffer_km)
                # Find nearest time
                if t not in ds_times:
                    nearest_idx = np.argmin(np.abs(ds_times - t))
                    t = ds_times[nearest_idx]
                # Get dataset at this time
                ds_at_time = ds.sel({time_coord: t}, method='nearest')
                # Extract spatial average of soil moisture
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
                    continue
                subset = ds_at_time.sel({
                    'latitude': slice(lat_max, lat_min),
                    'longitude': slice(lon_min, lon_max)
                })
                if subset.sizes.get('latitude', 0) == 0 or subset.sizes.get('longitude', 0) == 0:
                    continue

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
        std_print(None, 'an', f'Track {track_id} ASM analysis')
        std_print(None, 'an', 'ASM value', f'{asm_24h:.4f}')
        std_print(None, 'an', 'Mean TP', f'{tp_mean:.3f} mm/hr')
        std_print(None, 'an', 'Max TP', f'{tp_max:.3f} mm/hr')
    if not correlation_data:
        std_print('WARN', 'an', 'No correlation data computed')
        return None
    corr_df = pd.DataFrame(correlation_data)

    # Compute correlations
    std_print('INFO', 'an', 'Correlation : ASM-TP')
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(corr_df['ASM_24h'], corr_df['mean_tp'])
    spearman_r, spearman_p = spearmanr(corr_df['ASM_24h'], corr_df['mean_tp'])
    std_print(None, None, 'ASM vs Mean TP - Pearson', f'r={pearson_r:.3f}, p={pearson_p:.4f}')
    std_print(None, None, 'ASM vs Mean TP - Spearman', f'ρ={spearman_r:.3f}, p={spearman_p:.4f}')
    # Max precipitation
    pearson_r_max, p_max = pearsonr(corr_df['ASM_24h'], corr_df['max_tp'])
    std_print(None, None, 'ASM vs Max TP - Pearson', f'r={pearson_r_max:.3f}, p={p_max:.4f}')
    # Total precipitation
    pearson_r_total, p_total = pearsonr(corr_df['ASM_24h'], corr_df['total_tp'])
    std_print(None, None, 'ASM vs Total TP - Pearson', f'r={pearson_r_total:.3f}, p={p_total:.4f}')

    # Interpretation
    std_print('INFO', 'an', 'Interpretation : ASM-TP Correlation')
    if pearson_r > 0.5 and pearson_p < 0.05:
        std_print(None, None, 'Result: STRONG POSITIVE correlation (High ASM → Higher rainfall)')
    elif pearson_r > 0.3 and pearson_p < 0.05:
        std_print(None, None, 'Result: MODERATE POSITIVE correlation (Higher ASM tends to produce more rain)')
    elif pearson_r < -0.3 and pearson_p < 0.05:
        std_print(None, None, 'Result: NEGATIVE correlation (Dry soils produce more rain)')
    else:
        std_print(None, None, 'Result: NO SIGNIFICANT correlation (ASM does not explain rainfall variation)')

    # Export results
    export_path = os.path.join(output_dir, 'asm_tp_correlation.csv')
    os.makedirs(output_dir, exist_ok=True)
    corr_df.to_csv(export_path, index=False)
    std_print('SAVED', 'an', 'ASM-TP correlation CSV', export_path)
    return corr_df


def compute_asm_duration_correlation(asm_results_df, duration_df, output_dir='output'):
    """
    Correlate antecedent soil moisture with MD duration.
    Question: Does drier soil cause faster weakening?
    """
    std_print('INFO', 'an', 'ASM - Duration correlation analysis')
    # Merge datasets
    merged_df = asm_results_df.merge(duration_df[['track_id', 'duration_hours', 'is_long_lived']], on='track_id', how='inner')
    if len(merged_df) < 3:
        std_print('WARN', 'an', 'Insufficient data for correlation (< 3 tracks)')
        return None

    asm_col = None
    available_asm_cols = [col for col in merged_df.columns if 'ASM_24h' in col]
    if available_asm_cols:
        asm_col = available_asm_cols[0]  # Use first available ASM column
        std_print('INFO', 'an', 'Using ASM column', asm_col)
    else:
        std_print('ERROR', 'an', 'No ASM_24h columns found in dataframe')
        std_print('ERROR', 'an', 'Available columns', f'{merged_df.columns.tolist()}')
        return None

    asm_values = []
    for idx, row in merged_df.iterrows():
        asm_val = row[asm_col]
        if isinstance(asm_val, dict):
            asm_val = asm_val.get('value', np.nan)
        asm_values.append(asm_val)
    merged_df['ASM_24h_clean'] = asm_values
    std_print('INFO', 'an', 'Tracks with both ASM and duration data', len(merged_df))

    std_print(None, None, 'Track ID | ASM_24h | Duration (h) | Category')
    for _, row in merged_df.iterrows():
        category = "Long-lived" if row['is_long_lived'] else "Short-lived"
        asm_val = row['ASM_24h_clean']
        # Handle NaN values
        if pd.isna(asm_val):
            asm_str = "   N/A"
        else:
            asm_str = f"{asm_val:7.4f}"
        std_print(None, None, f"{int(row['track_id']):8d} | {asm_str} | {row['duration_hours']:12.1f} | {category}")

    # Compute correlations
    std_print('INFO', 'an', 'Correlation statistics')
    # Filter out NaN values for correlation
    valid_data = merged_df.dropna(subset=['ASM_24h_clean', 'duration_hours'])
    if len(valid_data) < 3:
        std_print('WARN', 'an', f'Insufficient valid data for correlation (after removing NaN)', len(valid_data))
        return None
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(valid_data['ASM_24h_clean'], valid_data['duration_hours'])
    std_print('INFO', 'an', 'ASM vs Duration :')
    std_print(None, None, 'ASM vs Duration - Pearson', f'r={pearson_r:.3f}, p={pearson_p:.4f}')
    try:
        spearman_r, spearman_p = spearmanr(valid_data['ASM_24h_clean'], valid_data['duration_hours'])
        std_print(None, None, 'ASM vs Duration - Spearman', f'ρ={spearman_r:.3f}, p={spearman_p:.4f}')
    except:
        std_print('WARN', 'an', 'Spearman correlation failed')

    # Interpretation
    std_print('INFO', 'an', 'Interpretation: ASM-Duration Correlation')
    if pearson_r > 0.5 and pearson_p < 0.05:
        std_print(None, None, 'Result: STRONG POSITIVE (High ASM → Longer track duration)')
    elif pearson_r > 0.3 and pearson_p < 0.05:
        std_print(None, None, 'Result: MODERATE POSITIVE (Higher ASM extends duration)')
    elif pearson_r < -0.3 and pearson_p < 0.05:
        std_print(None, None, 'Result: NEGATIVE (Dry soils support longer tracks)')
    else:
        std_print(None, None, 'Result: NO SIGNIFICANT (ASM does not predict duration)')

    # By category
    std_print('INFO', 'an', 'Comparison: Long-lived vs Short-lived MDs')
    long_lived = merged_df[merged_df['is_long_lived']]
    short_lived = merged_df[~merged_df['is_long_lived']]
    if len(long_lived) > 0:
        std_print(None, None, 'Long-lived MDs count', len(long_lived))
        long_lived_asm = long_lived['ASM_24h_clean'].dropna()
        if len(long_lived_asm) > 0:
            std_print(None, None, 'Long-lived MDs (≥96h) - Mean ASM', f"{long_lived_asm.mean():.4f} ± {long_lived_asm.std():.4f} m³/m³")
            std_print(None, None, 'Long-lived MDs (≥96h) - Mean Duration', f"{long_lived['duration_hours'].mean():.1f} ± {long_lived['duration_hours'].std():.1f} hours")
    if len(short_lived) > 0:
        std_print(None, None, 'Short-lived MDs count', len(short_lived))
        short_lived_asm = short_lived['ASM_24h_clean'].dropna()
        if len(short_lived_asm) > 0:
            std_print(None, None, 'Short-lived MDs (<96h) - Mean ASM', f"{short_lived_asm.mean():.4f} ± {short_lived_asm.std():.4f} m³/m³")
            std_print(None, None, 'Short-lived MDs (<96h) - Mean Duration', f"{short_lived['duration_hours'].mean():.1f} ± {short_lived['duration_hours'].std():.1f} hours")

    # Statistical test
    if len(long_lived) > 0 and len(short_lived) > 0:
        try:
            t_stat, t_pval = ttest_ind(long_lived['ASM_24h_clean'].dropna(), short_lived['ASM_24h_clean'].dropna())
            std_print('INFO', 'an', 'T-test (Long vs Short ASM)', f't={t_stat:.3f}, p={t_pval:.4f}')
            if t_pval < 0.05:
                std_print(None, None, 'Result: Significant difference in ASM between categories')
            else:
                std_print(None, None, 'Result: No significant ASM difference between categories')
        except:
            std_print('WARN', 'an', 'T-test failed')

    # Export
    export_path = os.path.join(output_dir, 'asm_duration_correlation.csv')
    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(export_path, index=False)
    std_print('SAVED', 'an', 'ASM-Duration correlation CSV', export_path)
    return merged_df

def create_asm_summary_report(asm_results_df, corr_tp_df, corr_duration_df, output_dir='output'):
    if corr_tp_df is None:
        std_print('WARN', 'an', 'Skipping TP correlation section', 'No data available')
        return

    report_path = os.path.join(output_dir, 'ASM_Analysis_Report_global.txt')
    tests = []
    pvalues = []
    with open(report_path, 'w') as f:
        f.write("Antecedent soil moisture analysis - report\n")
        # Section 1: ASM Statistics
        f.write("1. Antecedent soil moisture statistics\n")
        f.write(f"  Number of tracks analyzed:                     {len(asm_results_df)}\n")
        # Find available ASM column
        asm_col = 'ASM_24h_excl_6h'
        if asm_col not in asm_results_df.columns:
            available_cols = [col for col in asm_results_df.columns if 'ASM_24h' in col]
            if available_cols:
                asm_col = available_cols[0]
            else:
                asm_col = None
        if asm_col and asm_col in asm_results_df.columns:
            valid_asm = asm_results_df[asm_col].dropna()
            if len(valid_asm) > 0:
                f.write(f"  Mean ASM (24h before landfall):                {valid_asm.mean():.4f} m³/m³\n")
                f.write(f"  Std Dev:                                       {valid_asm.std():.4f} m³/m³\n")
                f.write(f"  Min:                                           {valid_asm.min():.4f} m³/m³\n")
                f.write(f"  Max:                                           {valid_asm.max():.4f} m³/m³\n")
                f.write(f"  Range:                                         {valid_asm.max() - valid_asm.min():.4f} mÂ³/mÂ³\n\n")
        else:
            f.write(f"  No ASM data available\n\n")
        # Section 2: ASM-Precipitation Correlation
        f.write("2. ASM - Precipitation correlation\n")
        if corr_tp_df is not None and 'mean_tp' in corr_tp_df.columns:
            # Find the ASM column (should be 'ASM_24h' from compute_asm_tp_correlation)
            asm_col = 'ASM_24h'
            std_print('DEBUG', 'an', f'r_tp_df columns are: {corr_tp_df.columns.tolist()}')
            if asm_col in corr_tp_df.columns:
                valid_data = corr_tp_df.dropna(subset=[asm_col, 'mean_tp'])
                if len(valid_data) > 1:  # Pearsonr needs at least 2 points
                    r_mean, p_mean = pearsonr(valid_data[asm_col], valid_data['mean_tp'])
                    r_max, p_max = pearsonr(corr_tp_df[asm_col], corr_tp_df['max_tp'])
                    r_total, p_total = pearsonr(corr_tp_df[asm_col], corr_tp_df['total_tp'])

                    tests.append('ASM vs Mean TP')
                    pvalues.append(p_mean)
                    tests.append('ASM vs Max TP')
                    pvalues.append(p_max)
                    tests.append('ASM vs Total TP')
                    pvalues.append(p_total)

                reject, p_corrected, _, _ = multipletests(
                    pvalues,
                    method='fdr_bh',  # Benjamini-Hochberg FDR correction
                    alpha=0.05
                )
                for test_name, p_orig, p_corr, is_sig in zip(tests, pvalues, p_corrected, reject):
                    std_print('INFO', 'an', f"{test_name} :")
                    std_print(None, None, 'p_uncorrected', f'{p_orig:.4f}')
                    std_print(None, None, 'p_corrected',   f'{p_corr:.4f}')
                    std_print(None, None, 'Significant',   f'{is_sig}')

                f.write(f"  ASM vs Mean TP:\n")
                f.write(f"    Pearson r: {r_mean:.3f}, p-value: {p_mean:.4f}\n")
                f.write(f"    Significance: {'YES' if p_mean < 0.05 else 'NO'}\n\n")
                f.write(f"  ASM vs Max TP:\n")
                f.write(f"    Pearson r = {r_max:.3f}, p-value = {p_max:.4f}\n")
                f.write(f"    Significance: {'YES' if p_max < 0.05 else 'NO'}\n\n")
                f.write(f"  ASM vs Total TP:\n")
                f.write(f"    Pearson r = {r_total:.3f}, p-value = {p_total:.4f}\n")
                f.write(f"    Significance: {'YES' if p_total < 0.05 else 'NO'}\n\n")
            else:
                std_print('WARN', 'an', 'Insufficient data for ASM correlation')
                r_mean, p_mean = np.nan, np.nan
        else:
            std_print('WARN', 'an', 'ASM_24h column missing (likely due to land-originated tracks)')
        # else:
        #     f.write("  [STATUS] No valid correlation data (insufficient land points).\n")
        # # Section 3: ASM-Duration Correlation
        # f.write("3. ASM - DURATION CORRELATION\n")
        # if corr_duration_df is not None:
        #     r_dur, p_dur = pearsonr(corr_duration_df['ASM_24h'], corr_duration_df['duration_hours'])
        #     f.write(f"  Pearson r = {r_dur:.3f}, p-value = {p_dur:.4f}\n")
        #     f.write(f"  Significance: {'YES' if p_dur < 0.05 else 'NO'}\n\n")
        #     long_lived = corr_duration_df[corr_duration_df['is_long_lived']]
        #     short_lived = corr_duration_df[~corr_duration_df['is_long_lived']]
        #     f.write(f"  Long-lived MDs (≥96h):\n")
        #     f.write(f"    Mean ASM = {long_lived['ASM_24h'].mean():.4f} ± {long_lived['ASM_24h'].std():.4f} m³/m³\n")
        #     f.write(f"    Count = {len(long_lived)}\n\n")
        #     f.write(f"  Short-lived MDs (<96h):\n")
        #     f.write(f"    Mean ASM = {short_lived['ASM_24h'].mean():.4f} ± {short_lived['ASM_24h'].std():.4f} m³/m³\n")
        #     f.write(f"    Count = {len(short_lived)}\n\n")
        # else:
        #     f.write("  [STATUS] No valid correlation data (insufficient land points).\n")

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
    std_print('SAVED', 'an', 'Summary report', report_path)

def export_antecedent_results(asm_df, output_dir = 'output'):
    std_print('DEBUG', 'an', 'ASM results EXPORTING')
    export_results_to_csv(asm_df, output_dir, 'asm_results', 'an')

def plot_antecedent_swvl1_timeseries(ds, track_df, track_id, buffer_km, output_dir = 'output'):
    """Plot only the 24-hour before landfall period (antecedent conditions)"""
    # Filter track
    track_df = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track_df.empty:
        std_print('ERROR', 'an', f'Track {track_id} not found in DataFrame')
        return
    std_print('INFO', 'an', f'Track {track_id} Number of points', len(track_df))

    # Find landfall
    landfall_time = find_landfall_time(track_df)
    if landfall_time is None:
        std_print('WARN', 'an', f'Track {track_id}', 'No landfall detected - skipping antecedent plot')
        # # If it's land-originated, use the start of the track as the "event" time
        # landfall_time = pd.to_datetime(track_df['time'].iloc[0])
        # std_print(None, None, f'Track {track_id}', 'Using start time as reference for land-originated MD')
        return

    # 24-hour window before landfall
    window_start = landfall_time - pd.Timedelta(hours=24)
    window_end = landfall_time

    # Clip track to only the antecedent window
    mask = (pd.to_datetime(track_df['time']) >= window_start) & (pd.to_datetime(track_df['time']) <= window_end)
    antecedent_track = track_df[mask].copy()
    if antecedent_track.empty:
        std_print('WARN', 'an', f'Track {track_id}', 'No points in antecedent window')
        return

    # Extract SWVL1 values along the antecedent track
    swvl1_values = []
    times = []
    ds, time_coord = get_time_coord(ds)
    for _, row in track_df.iterrows():
        try:
            t = pd.to_datetime(row['time'])
            lat, lon = row['latitude'], row['longitude']

            subset = extract_spatial_subset(ds, lat, lon, t, buffer_km=buffer_km, time_coord=time_coord)
            if subset is None or 'swvl1' not in subset.data_vars:
                continue
            swvl1 = float(subset['swvl1'].mean().values)
            if not np.isnan(swvl1) and 0.01 < swvl1 < 0.70:
                swvl1_values.append(swvl1)
                times.append(t)
        except Exception as e:
            std_print('ERROR', 'an', f'Track {track_id} Failed to extract swvl1', str(e))
            continue

    if len(swvl1_values) < 2:
        std_print('WARN', 'an', f'Track {track_id} Insufficient swvl1 data for antecedent plot', f'{len(swvl1_values)} points')
        return

    times = pd.to_datetime(times)
    swvl1_values = np.array(swvl1_values)

    fig, ax = plt.subplots(figsize=(14, 7))
    # Plot
    color_swvl1 = 'tab:brown'
    ax.set_ylabel('Soil Water Volumatric Layer 1 (swvl1) (m³/m³)', color=color_swvl1, fontsize=11, fontweight='bold')
    ax.plot(times, swvl1_values, 's-', color=color_swvl1, linewidth=2.5, markersize=5,  label='Soil Water Volumatric Layer 1', alpha=0.8, zorder=3)
    ax.tick_params(axis='y', labelcolor=color_swvl1)
    # Mark the landfall point (at the right edge)
    ax.axvline(landfall_time, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='Landfall', zorder=4)
    ax.plot(landfall_time, swvl1_values[-1], 'D', color='red', markersize=5, markeredgecolor='darkred', markeredgewidth=1, zorder=5)

    # Shade the antecedent region
    # ax.axvspan(times[0], landfall_time, alpha=0.1, color='blue', label='Antecedent Period (24h pre-landfall)')

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Soil Water Volumatric Level-1 (m³/m³)', fontsize=12, fontweight='bold')
    ax.set_title(f'Track {track_id}: Antecedent Soil Moisture\n24 Hours before Landfall', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.5, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    antecedent_dir = os.path.join(output_dir, 'antecedent_soil_moisture')
    os.makedirs(antecedent_dir, exist_ok=True)
    plot_path = os.path.join(antecedent_dir, f'track_{int(track_id)}_antecedent_SWVL1.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    std_print('SAVED', 'an', f'Antecedent swvl1 24h pre-landfall plot', plot_path)

    std_print('INFO', 'an', f'Track {track_id} Antecedent swvl1 Statistics')
    std_print(None, None, 'Mean Antecedent SWVL1 (24h pre-landfall)', f'{np.mean(swvl1_values):.4f} m³/m³')
    std_print(None, None, 'Std Dev', f'{np.std(swvl1_values):.4f} m³/m³')
    std_print(None, None, 'Initial (earliest)', f'{swvl1_values[0]:.4f} m³/m³')
    std_print(None, None, 'At landfall', f'{swvl1_values[-1]:.4f} m³/m³')
    std_print(None, None, 'Trend', f'{swvl1_values[-1] - swvl1_values[0]:+.4f} m³/m³')
    return np.mean(swvl1_values)

def plot_antecedent_vs_response_comparison(ds, track_df, track_id, buffer_km=100, output_dir='output'):
    """
    LEFT: Antecedent period (24h before landfall) - Initial conditions
    RIGHT: Response period (24h after landfall) - System response
    """
    track_df = track_df[track_df['track_id'] == track_id].sort_values('time')
    if track_df.empty:
        std_print('ERROR', 'an', f'Track {track_id} not found')
        return

    landfall_time = find_landfall_time(track_df)
    if landfall_time is None:
        std_print('WARN', 'an', f'Track {track_id}', 'No landfall - skipping comparison')
        return

    # Define windows
    antecedent_start = landfall_time - pd.Timedelta(hours=24)
    antecedent_end = landfall_time
    response_start = landfall_time
    response_end = landfall_time + pd.Timedelta(hours=24)

    ds, time_coord = get_time_coord(ds)

    # Extract antecedent data
    antecedent_mask = (pd.to_datetime(track_df['time']) >= antecedent_start) & (pd.to_datetime(track_df['time']) <= antecedent_end)
    antecedent_track = track_df[antecedent_mask].copy()

    # Extract response data
    response_mask = (pd.to_datetime(track_df['time']) >= response_start) & (pd.to_datetime(track_df['time']) <= response_end)
    response_track = track_df[response_mask].copy()

    # Extract swvl1 for both periods
    def extract_swvl1_from_track(track_subset):
        swvl1_values = []
        times = []
        for _, row in track_subset.iterrows():
            try:
                t = pd.to_datetime(row['time'])
                lat, lon = row['latitude'], row['longitude']
                subset = extract_spatial_subset(ds, lat, lon, t, buffer_km, time_coord)
                if subset and 'swvl1' in subset.data_vars:
                    swvl1 = float(subset['swvl1'].mean().values)
                    if not np.isnan(swvl1) and 0.01 < swvl1 < 0.70:
                        swvl1_values.append(swvl1)
                        times.append(t)
            except:
                continue
        return np.array(times), np.array(swvl1_values)

    antecedent_times, antecedent_swvl1 = extract_swvl1_from_track(antecedent_track)
    response_times, response_swvl1 = extract_swvl1_from_track(response_track)

    if len(antecedent_swvl1) < 2 or len(response_swvl1) < 2:
        std_print('WARN', 'an', f'Track {track_id}', 'Insufficient data for comparison')
        return

    # 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Antecedent (Pre-landfall)
    ax1.plot(antecedent_times, antecedent_swvl1, 'o-', linewidth=2.5, markersize=7, color='navy', label='Antecedent swvl1', alpha=0.9)
    ax1.fill_between(antecedent_times, antecedent_swvl1, alpha=0.2, color='blue')
    ax1.axvline(landfall_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Soil Water Content (m³/m³)', fontsize=10, fontweight='bold')
    ax1.set_title('Antecedent Period (24h BEFORE Landfall)\nInitial Condition', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.4)

    common_bbox = dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.7, edgecolor="none", linewidth=0.8)

    aswvl1_mean = np.mean(antecedent_swvl1)
    aswvl1_std = np.std(antecedent_swvl1)
    textbox1 = f'Mean: {aswvl1_mean:.4f} m³/m³\nStd: {aswvl1_std:.4f}\nTrend: {antecedent_swvl1[-1]-antecedent_swvl1[0]:+.4f}'
    ax1.text(1.02, 0.85, textbox1, transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', family='monospace', bbox=common_bbox)

    # Panel 2: Response (Post-landfall)
    ax2.plot(response_times, response_swvl1, 's-', linewidth=2.5, markersize=7, color='darkred', label='Response swvl1', alpha=0.9)
    ax2.fill_between(response_times, response_swvl1, alpha=0.2, color='red')
    ax2.axvline(landfall_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Soil Water Volumatric Level-1 (m³/m³)', fontsize=10, fontweight='bold')
    ax2.set_title('Response Period (24h AFTER Landfall)\nSystem Interaction', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.4)

    # Add text box for response stats
    response_mean = np.mean(response_swvl1)
    response_std = np.std(response_swvl1)
    textbox2 = f'Mean: {response_mean:.4f} m³/m³\nStd: {response_std:.4f}\nTrend: {response_swvl1[-1]-response_swvl1[0]:+.4f}'
    ax2.text(1.02, 0.85, textbox2, transform=ax2.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left', family='monospace', bbox=common_bbox)

    # Overall title with comparison
    swvl1_change = response_mean - aswvl1_mean
    fig.suptitle(
        f'Track {track_id}: Antecedent vs Response Comparison\n'
        f'swvl1 Change at Landfall: {swvl1_change:+.4f} m³/m³ '
        f'({100*swvl1_change/aswvl1_mean:+.1f}%)',
        fontsize=13, fontweight='bold', y=1.00
    )
    plt.tight_layout()

    antecedent_dir = os.path.join(output_dir, 'antecedent_comparison')
    os.makedirs(antecedent_dir, exist_ok=True)
    plot_path = os.path.join(antecedent_dir, f'track_{int(track_id)}_antecedent_vs_response.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    std_print('SAVED', 'an', 'Antecedent vs Response comparison', plot_path)
    std_print('INFO', 'an', f'Track {track_id} swvl1 change at landfall')
    std_print(None, None, 'Antecedent Mean', f'{aswvl1_mean:.4f} m³/m³')
    std_print(None, None, 'Response Mean', f'{response_mean:.4f} m³/m³')
    std_print(None, None, 'Change', f'{swvl1_change:+.4f} m³/m³ ({100*swvl1_change/aswvl1_mean:+.1f}%)')

def plot_swvl1_tp_map(ds, track_df, track_id, hours_before = 24, hours_after = 24, buffer_km = 300, min_extent_deg = 10.0, output_dir = 'output'):
    """Comparison map before/after landfall"""
    # Get ALL points for track
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    ds, time_coord = get_time_coord(ds)
    ds_times = pd.to_datetime(ds[time_coord].values)
    track_id = int(track_id)

    if len(track) < 3:
        std_print('WARN', 'an', f'Track {track_id} has insufficient points', len(track))
        return False

    # Find landfall and check origin
    landfall_time = None
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
    # If still no landfall use start of track
    if landfall_time is None:
        std_print('WARN', 'an', f'Track {track_id}', 'No landfall detected, Using start of track')
        landfall_time = pd.to_datetime(track.iloc[0]['time'])

    # # Check if originates over land
    origin_type = find_origin_type(track)

    # Determine status text for period label
    if origin_type == 'land_originated':
        period_label = "Origin"
        status_text = "Originates over land"
    elif landfall_time is None:
        period_label = "Midpoint"
        status_text = "No Landfall"
        # use start point
        landfall_time = pd.to_datetime(track.iloc[0]['time'])
        # # Use midpoint as reference
        # midpoint_idx = len(track) // 2
        # landfall_time = pd.to_datetime(track.iloc[midpoint_idx]['time'])
    else:
        period_label = "Landfall"
        status_text = "Landfall"

    # Get before/after times
    landfall_time = pd.Timestamp(landfall_time)
    before_start = landfall_time - pd.Timedelta(hours=hours_before)
    after_end = landfall_time + pd.Timedelta(hours=hours_after)

    track['time_dt'] = pd.to_datetime(track['time'])
    # Get points before and after landfall for plotting
    track_window = track[(track['time_dt'] >= before_start) &  (track['time_dt'] <= after_end)].copy()
    # Split into before/after
    track_before = track_window[pd.to_datetime(track_window['time']) < landfall_time]
    track_after = track_window[pd.to_datetime(track_window['time']) >= landfall_time]
    std_print('DEBUG', 'an', 'Landfall time :', f'{landfall_time}')
    std_print('DEBUG', 'an', 'Window :', f'{before_start} to {after_end}')
    std_print('DEBUG', 'an', 'Track points total :', f'{len(track)}')
    std_print('DEBUG', 'an', 'Points in window :', f'{len(track_window)}')
    std_print('DEBUG', 'an', 'Points before landfall :', f'{len(track_before)}')
    std_print('DEBUG', 'an', 'Points after landfall :', f'{len(track_after)}')
    if len(track_window) > 0:
        std_print('DEBUG', 'an', 'First point in window :', f"{track_window.iloc[0]['time']}")
        std_print('DEBUG', 'an', 'Last point in window :', f"{track_window.iloc[-1]['time']}")
        std_print('DEBUG', 'an', 'Time span in window :', f"{(pd.to_datetime(track_window.iloc[-1]['time']) - pd.to_datetime(track_window.iloc[0]['time'])).total_seconds()/3600:.1f} hours")
    elif len(track_window) < 0 :
        std_print('DEBUG', 'an', f'No points in time window!')
        std_print('DEBUG', 'an', 'Track times :', f"{track['time'].iloc[0]} to {track['time'].iloc[-1]}")
        std_print('DEBUG', 'an', 'Landfall time :', landfall_time)
        std_print('DEBUG', 'an', 'Before start :', before_start)
        std_print('DEBUG', 'an', 'After end :', after_end)
    else :
        std_print('WARN', 'an', f'Track {track_id} : No points in time window')

    # Get extent
    all_lats = track['latitude']
    all_lons = track['longitude']
    if all_lats.empty or all_lons.empty:
        std_print('WARN', 'an', f'Track {track_id}', 'No valid track points for extent')
        return False

    # Calculate track centroid
    centroid_lat = all_lats.mean()
    centroid_lon = all_lons.mean()
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

    # Clamp to study domain
    min_lat, max_lat, min_lon, max_lon = clamp_to_domain(min_lat, max_lat, min_lon, max_lon)

    # std_print('INFO', 'an', f'Track {track_id} Map extent (clipped)', f'{min_lon:.1f}E-{max_lon:.1f}E, {min_lat:.1f}N-{max_lat:.1f}N')
    std_print('INFO', 'an', f' --- Map extent for track {track_id} ---')
    std_print(None, None, 'Latitude', f'[{min_lat:.2f}, {max_lat:.2f}] deg (range: {lat_range:.1f}°)')
    std_print(None, None, 'Longitude', f'[{min_lon:.2f}, {max_lon:.2f}] deg (range: {lon_range:.1f}°)')
    std_print(None, None, 'Centroid', f'({centroid_lat:.2f}, {centroid_lon:.2f})')
    std_print(None, None, 'Track points', f'{len(track)}')
    std_print(None, None, 'Before landfall', f'{len(track_before)} points')
    std_print(None, None, 'After landfall', f'{len(track_after)} points')

    try:
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
            std_print('WARN', 'an', f'Track {track_id} Insufficient temporal data')
            return False

        sm_before_mean = ds_before['swvl1'].mean(dim=time_coord)
        sm_after_mean = ds_after['swvl1'].mean(dim=time_coord)
        sm_change = sm_after_mean - sm_before_mean
        tp_before_total = ds_before['tp'].sum(dim=time_coord) * 1000  # m to mm
        tp_after_total = ds_after['tp'].sum(dim=time_coord) * 1000   # m to mm
        tp_change = tp_after_total - tp_before_total

        # Get antecedent SM
        for exclusion_hours in [0, 3, 6, 12]:
            asm_24h = compute_antecedent_swvl1(ds, track_df, track_id, hours_before_landfall=24, exclude_last_hours=exclusion_hours, buffer_km=100)
            if isinstance(asm_24h, dict):
                asm_24h = asm_24h.get('value', np.nan)
            if np.isnan(asm_24h):
                asm_display = "N/A"
            else:
                asm_display = f"{asm_24h:.4f} m³/m³"

            # Create 2x4 subplot layout
            fig, axes = plt.subplots(2, 4, figsize=(22, 14,), subplot_kw={'projection': ccrs.PlateCarree()})

            # For soil moisture
            sm_all_values = np.concatenate([sm_before_mean.values.flatten(), sm_after_mean.values.flatten()])
            sm_all_values = sm_all_values[~np.isnan(sm_all_values)]
            sm_vmin = np.percentile(sm_all_values, 5) if len(sm_all_values) > 0 else 0
            sm_vmax = np.percentile(sm_all_values, 95) if len(sm_all_values) > 0 else 1

            # For Total Precipitation
            tp_all_values = np.concatenate([tp_before_total.values.flatten(), tp_after_total.values.flatten()])
            tp_all_values = tp_all_values[~np.isnan(tp_all_values)]
            tp_vmin = 0  # Always start at 0 for precipitation
            tp_vmax = np.percentile(tp_all_values, 95) if len(tp_all_values) > 0 else 10

            # For change plots
            sm_change_values = sm_change.values.flatten()
            sm_change_values = sm_change_values[~np.isnan(sm_change_values)]
            sm_change_vmax = max(abs(np.nanpercentile(sm_change_values, 5)), abs(np.nanpercentile(sm_change_values, 95))) if len(sm_change_values) > 0 else 0.1
            tp_change_values = tp_change.values.flatten()
            tp_change_values = tp_change_values[~np.isnan(tp_change_values)]
            tp_change_vmax = max(abs(np.nanpercentile(tp_change_values, 5)), abs(np.nanpercentile(tp_change_values, 95))) if len(tp_change_values) > 0 else 10

            period_titles = [f'Before {period_label} ({hours_before}h)', f'After {period_label} ({hours_after}h)', 'Change (After - Before)']

            # Row 0: Soil moisture maps
            for i, (data, title) in enumerate(zip([sm_before_mean, sm_after_mean, sm_change], period_titles)):
                ax = axes[0, i]
                if data.size == 0 or np.all(np.isnan(data.values)):
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', fontsize=12)
                    ax.set_title(f'Soil Moisture\n{title}', fontsize=12, fontweight='bold')
                    continue
                if i == 2:  # Change plot for soil moisture
                    sm_change_vmax = max(abs(data.values.min()), abs(data.values.max()))
                    im = ax.contourf(data.longitude, data.latitude, data.values, levels=21, cmap='RdBu', vmin=-sm_change_vmax, vmax=sm_change_vmax, transform=ccrs.PlateCarree(), extend='both')
                else:  # Before/After plots
                    im = ax.contourf(data.longitude, data.latitude, data.values, levels=21, cmap='YlGnBu', vmin=sm_vmin, vmax=sm_vmax, transform=ccrs.PlateCarree(), extend='both')
                # Contour lines
                if not np.all(np.isnan(data.values)):
                    contours = ax.contour(data.longitude, data.latitude, data.values, levels=8, colors='black', linewidths=0.5, alpha=0.5, transform=ccrs.PlateCarree())
                    ax.clabel(contours, inline=True, fontsize=7, fmt='%1.3f')
                # Plot track segments
                if i == 0 and len(track_before) > 0:
                    ax.plot(track_before['longitude'], track_before['latitude'], 'b-', linewidth=2.5, transform=ccrs.PlateCarree(), zorder=5)
                elif i == 1 and len(track_after) > 0:
                    ax.plot(track_after['longitude'], track_after['latitude'], 'r-', linewidth=2.5, transform=ccrs.PlateCarree(), zorder=5)
                elif i == 2 and len(track_window) > 0:
                    # Plot windowed track
                    ax.plot(track_window['longitude'], track_window['latitude'], 'k-', linewidth=2.5, alpha=0.9, transform=ccrs.PlateCarree(), zorder=4)
                    # Mark landfall point
                    landfall_idx = -1
                    if landfall_time is not None and not track_window.empty:
                        for idx, row in track_window.iterrows():
                            t = pd.to_datetime(row['time'])
                            if t >= landfall_time:
                                landfall_idx = idx
                                break
                    # if len(track_before) > 0:
                    #     point = track_before.iloc[-1]
                    if 0 <= landfall_idx < len(track_window):
                        point = track.iloc[landfall_idx]
                        ax.plot(point['longitude'], point['latitude'], 'D', color='yellow', markersize=10, markeredgecolor='black', markeredgewidth=1, transform=ccrs.PlateCarree(), zorder=6, label='Landfall')

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
                    landfall_idx = -1
                    if landfall_time is not None and not track_window.empty:
                        for idx, row in track_window.iterrows():
                            t = pd.to_datetime(row['time'])
                            if t >= landfall_time:
                                landfall_idx = idx
                                break
                    # if len(track_before) > 0:
                    #     point = track_before.iloc[-1]
                    if 0 <= landfall_idx < len(track_window):
                        point = track.iloc[landfall_idx]
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
                f'Track {track_id}: Soil Moisture & Precipitation Evolution\n'  f'Before/After {period_label} ({hours_before}h before → {hours_after}h after)',
                fontsize=16,
                fontweight='bold',
                y=0.8
            )
            plt.tight_layout()
            plt.subplots_adjust(top=0.8)

            os.makedirs(output_dir, exist_ok=True)
            sm_tp_before_after_map_path = os.path.join(output_dir, f'swvl1_tp_before_after_track_{track_id}.png')
            plt.savefig(sm_tp_before_after_map_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            std_print('SAVED', 'an', 'SWVL1/TP brefore/after map', sm_tp_before_after_map_path)
            std_print('DEBUG', 'an', 'Extent', f'{min_lon:.1f}E to {max_lon:.1f}E, {min_lat:.1f}N to {max_lat:.1f}N')
            return True  # Signal success
    except Exception as e:
        std_print('ERROR', 'an', f'Track {track_id} Plotting brefore/after maps:', str(e))
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

    ds, time_coord = get_time_coord(ds)
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
