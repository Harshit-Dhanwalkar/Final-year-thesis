"""
Ocean Masking Utilities for ERA5 Soil Moisture (swvl1) Data

ERA5 provides invalid/unrealistic SWVL1 values over ocean:
- Ocean pixels: SWVL1 approx 0 or NaN or unrealistic values
- Valid land range: 0.01 < SWVL1 < 0.70 m³/m³
"""

import numpy as np
import xarray as xr
import pandas as pd

from utils.land_utils import is_point_land

# For debug
import traceback

# Valid soil moisture range (m³/m³)
SWVL1_MIN = 0.005    # Absolute minimum (dry sand)
SWVL1_MAX = 0.70     # Absolute maximum (saturated clay)
SWVL1_VALID_MIN = 0.01  # Practical minimum for land
SWVL1_VALID_MAX = 0.65  # Practical maximum for land

# Ocean detection thresholds
OCEAN_SWVL1_THRESHOLD_LOW = 0.005   # Below this = likely ocean
OCEAN_SWVL1_THRESHOLD_HIGH = 0.02   # Above this = likely land


def is_valid_land_swvl1(swvl1_value):
    if np.isnan(swvl1_value):
        return False
    if swvl1_value < SWVL1_VALID_MIN or swvl1_value > SWVL1_VALID_MAX:
        return False
    return True

def mask_swvl1_over_ocean(
    swvl1_value,
    lat,
    lon,
    strict_mode=True
):
    """Mask SWVL1 over ocean using combined criteria"""
    # 1. Geographic check (most reliable)
    if not is_point_land(lat, lon):
        print(f"    [MASKED] Ocean point ({lat:.2f}N, {lon:.2f}E): SWVL1={swvl1_value:.4f}")
        return np.nan
    # 2. Physical value check
    if not is_valid_land_swvl1(swvl1_value):
        print(f"    [MASKED] Invalid value ({lat:.2f}N, {lon:.2f}E): SWVL1={swvl1_value:.4f} outside [{SWVL1_VALID_MIN}, {SWVL1_VALID_MAX}]")
        return np.nan
    # Additional checks
    # 3. Strict mode:
    if strict_mode:
        # Suspicious low values over "land" (likely ocean pixel in land mask)
        if swvl1_value < SWVL1_VALID_MIN + 0.005:
            print(f"    [MASKED] Suspiciously low ({lat:.2f}N, {lon:.2f}E): SWVL1={swvl1_value:.4f} (strict mode)")
            return np.nan
    return swvl1_value

def extract_land_only_swvl1(
    ds,
    var_name='swvl1',
    time_coord='time',
    strict=True
):
    """Extract and mask SWVL1, returning only valid land values"""
    if var_name not in ds.data_vars:
        print(f" > [WARNING] {var_name} not in dataset")
        return ds

    ds_masked = ds.copy()
    swvl1_data = ds[var_name].values
    mask = np.ones_like(swvl1_data, dtype=bool)
    # Get lat/lon coordinates
    lats = ds.latitude.values
    lons = ds.longitude.values
    # Mask
    masked_count = 0
    total_count = 0
    for t_idx in range(swvl1_data.shape[0]):
        for lat_idx in range(swvl1_data.shape[1]):
            for lon_idx in range(swvl1_data.shape[2]):
                value = swvl1_data[t_idx, lat_idx, lon_idx]
                lat = float(lats[lat_idx])
                lon = float(lons[lon_idx])
                total_count += 1
                # Apply masking
                if np.isnan(value) or not is_point_land(lat, lon):
                    mask[t_idx, lat_idx, lon_idx] = False
                    masked_count += 1
                elif not is_valid_land_swvl1(value):
                    mask[t_idx, lat_idx, lon_idx] = False
                    masked_count += 1
                elif strict and value < SWVL1_VALID_MIN + 0.005:
                    mask[t_idx, lat_idx, lon_idx] = False
                    masked_count += 1
    # Apply mask
    ds_masked[var_name] = ds[var_name].where(mask, np.nan)
    print(f" > [MASKING] {var_name}: masked {masked_count}/{total_count} points ({100*masked_count/total_count:.1f}%)")
    return ds_masked


def extract_spatial_mean_land_only(
    ds,
    var_name,
    t,
    lat,
    lon,
    buffer_km=100,
    time_coord='time',
    strict_mode=True
):
    """Extract spatial mean, masking ocean pixels"""
    lat = float(lat)
    lon = float(lon)
    lat_buffer = buffer_km / 111.0
    lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat)))
    try:
        lat_min = min(lat - lat_buffer, lat + lat_buffer)
        lat_max = max(lat - lat_buffer, lat + lat_buffer)
        lon_min = min(lon - lon_buffer, lon + lon_buffer)
        lon_max = max(lon - lon_buffer, lon + lon_buffer)

        # Select time first
        ds_at_time = ds.sel({time_coord: t}, method='nearest')
        subset = ds_at_time.sel({
            'latitude': slice(lat_max, lat_min),  # north to south
            'longitude': slice(lon_min, lon_max)  # west to east
        })
        if var_name not in subset.data_vars:
            print(f" > [ERROR] {var_name} not in subset at {t}")
            return np.nan
        # Get data
        data = subset[var_name].values
        lats = subset.latitude.values
        lons = subset.longitude.values
        # Mask ocean pixels
        valid_values = []
        masked_count = 0
        for i in range(len(lats)):
            for j in range(len(lons)):
                value = data[i, j]
                # Skip invalid values
                if np.isnan(value):
                    continue
                # Check if land
                if not is_point_land(lats[i], lons[j]):
                    masked_count += 1
                    if masked_count % 10 == 0:
                        print(f"    [OCEAN MASKED] ({float(lats[i]):.2f}N, {float(lons[j]):.2f}E): SWVL1={value:.4f}")
                    continue
                # Check physical validity
                if var_name == 'swvl1':
                    if not is_valid_land_swvl1(value):
                        masked_count += 1
                        if masked_count % 10 == 0:
                            print(f"    [INVALID MASKED] ({float(lats[i]):.2f}N, {float(lons[j]):.2f}E): SWVL1={value:.4f} outside range")
                        continue
                    if strict_mode and value < SWVL1_VALID_MIN + 0.005:
                        masked_count += 1
                        if masked_count % 10 == 0:
                            print(f"    [STRICT MASKED] ({float(lats[i]):.2f}N, {float(lons[j]):.2f}E): SWVL1={value:.4f} too low")
                        continue
                valid_values.append(value)
        if len(valid_values) < 2:  # Need at least 2 valid points
            print(f" >  [WARN    ] insuffceient {var_name} at {t} only {len(valid_values)} valid pixels (masked {masked_count})")
            return np.nan
        return float(np.mean(valid_values))
    except Exception as e:
        print(f" >  [ERROR   ] extract_spatial_mean_land_only({var_name}, {t}): {e}")
        traceback.print_exc()
        return np.nan


def diagnose_ocean_contamination(
    ds,
    sample_size=1000,
    var_name='swvl1'
):
    """Diagnose ocean contamination in dataset.
    Reports:
    - % of ocean pixels in geographic bounds
    - SWVL1 distribution over ocean vs land
    - Suggested thresholds
    """
    print(f" OCEAN CONTAMINATION DIAGNOSIS for {var_name.upper()}")
    if var_name not in ds.data_vars:
        print(f" >  [ERROR   ] {var_name} not in dataset")
        return

    # Sample random pixels
    swvl1_data = ds[var_name].values
    lats = ds.latitude.values
    lons = ds.longitude.values

    np.random.seed(42)
    n_total = len(lats) * len(lons)
    sample_indices = np.random.choice(n_total, min(sample_size, n_total), replace=False)

    ocean_values = []
    land_values = []
    ocean_count = 0
    land_count = 0
    invalid_count = 0
    # Sample pixels
    for idx in sample_indices:
        lat_idx, lon_idx = np.unravel_index(idx, (len(lats), len(lons)))
        lat = lats[lat_idx]
        lon = lons[lon_idx]
        # Get mean value over time
        values = swvl1_data[:, lat_idx, lon_idx]
        valid_vals = values[~np.isnan(values)]
        if len(valid_vals) == 0:
            invalid_count += 1
            continue

        mean_val = np.mean(valid_vals)
        is_land = is_point_land(lat, lon)
        if is_land:
            land_values.append(mean_val)
            land_count += 1
        else:
            ocean_values.append(mean_val)
            ocean_count += 1
    print(f"  SAMPLING RESULTS (n={len(sample_indices)} pixels):")
    print(f"   Ocean pixels:     {ocean_count:4d} ({100*ocean_count/len(sample_indices):5.1f}%)")
    print(f"   Land pixels:      {land_count:4d} ({100*land_count/len(sample_indices):5.1f}%)")
    print(f"   Invalid (NaN):    {invalid_count:4d} ({100*invalid_count/len(sample_indices):5.1f}%)")

    if ocean_values:
        ocean_values = np.array(ocean_values)
        print(f"\n  OCEAN SWVL1 DISTRIBUTION:")
        print(f"   Mean:             {np.mean(ocean_values):.4f} m³/m³")
        print(f"   Median:           {np.median(ocean_values):.4f} m³/m³")
        print(f"   Std Dev:          {np.std(ocean_values):.4f} m³/m³")
        print(f"   Min/Max:          {np.min(ocean_values):.4f} / {np.max(ocean_values):.4f}")
        print(f"   % < 0.01:         {100*np.sum(ocean_values < 0.01)/len(ocean_values):.1f}%")
        print(f"   % < 0.02:         {100*np.sum(ocean_values < 0.02)/len(ocean_values):.1f}%")

    if land_values:
        land_values = np.array(land_values)
        print(f"\n  LAND SWVL1 DISTRIBUTION:")
        print(f"   Mean:             {np.mean(land_values):.4f} m³/m³")
        print(f"   Median:           {np.median(land_values):.4f} m³/m³")
        print(f"   Std Dev:          {np.std(land_values):.4f} m³/m³")
        print(f"   Min/Max:          {np.min(land_values):.4f} / {np.max(land_values):.4f}")
        print(f"   % > 0.10:         {100*np.sum(land_values > 0.10)/len(land_values):.1f}%")
