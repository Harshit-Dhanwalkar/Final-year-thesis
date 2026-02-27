import numpy as np
import pandas as pd
from utils.io_utils import std_print
from utils.temporal_utils import get_time_coord

import traceback

# Clipped domain for Indian monsoon region
CLIPPED_MIN_LAT = 10.0
CLIPPED_MAX_LAT = 30.0
CLIPPED_MIN_LON = 70.0
CLIPPED_MAX_LON = 95.0

EARTH_R = 6371.0 # km

def clamp_to_domain(min_lat, max_lat, min_lon, max_lon):
    """Clamp spatial bounds to clipped monsoon domain"""
    min_lat = max(min_lat, CLIPPED_MIN_LAT)
    max_lat = min(max_lat, CLIPPED_MAX_LAT)
    min_lon = max(min_lon, CLIPPED_MIN_LON)
    max_lon = min(max_lon, CLIPPED_MAX_LON)

    return min_lat, max_lat, min_lon, max_lon

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate great-circle distance between two points (km).
    Args:
        lon1, lat1: Point 1 (degrees)
        lon2, lat2: Point 2 (degrees)
    Returns:
        float: Distance in km
    """
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_R * c

def get_default_buffer_km(variable):
    """Get recommended buffer for each variable"""
    # default_buffers = {
    #     'tp': 150,          # Total precipitation
    #     'swvl1': 100,       # Soil water volume layer 1
    #     'speed': 50,        # Track speed (tighter)
    #     'slhf': 150,        # Surface latent heat flux
    #     'avg_slhf': 150,    # Alternative name
    #     'avg_ishf': 150,    # Alternative name
    #     't2m': 100,         # 2m temperature
    #     'u10': 150,         # Wind U component
    #     'v10': 150,         # Wind V component
    #     'msl': 100,         # Mean sea level pressure
    # }
    default_buffers = {
        'tp': 350,          # Total precipitation
        'swvl1': 350,       # Soil water volume layer 1
        'speed': 50,        # Track speed (tighter)
        'avg_slhf': 350,    # Alternative name
        'avg_ishf': 350,    # Alternative name
        't2m': 100,         # 2m temperature
        'u10': 150,         # Wind U component
        'v10': 150,         # Wind V component
        'msl': 100,         # Mean sea level pressure
    }
    return default_buffers.get(variable.lower(), 100)

def calculate_buffer_deg(lat, buffer_km):
    """Calculate lat/lon buffer in degrees from distance in km"""
    lat_buffer = buffer_km / 111.0
    lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat)))
    return lat_buffer, lon_buffer

def extract_spatial_subset(ds, lat, lon, time, buffer_km=100, time_coord=None, verbose=False):
    """Extract regional data around a point with latitude-dependent buffering"""
    try:
        # time coordinate if not provided
        if time_coord is None:
            ds, time_coord = get_time_coord(ds)
            if verbose:
                std_print('DEBUG', f'Auto-detected time_coord: {time_coord}')
        if verbose:
            std_print('DEBUG', f'Calculating buffer: {buffer_km} km')
        lat_buffer, lon_buffer = calculate_buffer_deg(lat, buffer_km)
        # Create bounds
        lat_min, lat_max = lat - lat_buffer, lat + lat_buffer
        lon_min, lon_max = lon - lon_buffer, lon + lon_buffer
        # Clamp to dataset bounds
        lat_min = max(lat_min, float(ds.latitude.min()))
        lat_max = min(lat_max, float(ds.latitude.max()))
        lon_min = max(lon_min, float(ds.longitude.min()))
        lon_max = min(lon_max, float(ds.longitude.max()))
        if verbose:
            std_print('DEBUG', f'Bounds - Lat: [{lat_min:.2f}, {lat_max:.2f}], Lon: [{lon_min:.2f}, {lon_max:.2f}]')
        # Select
        subset = ds.sel({
            time_coord: time,
            'latitude': (ds.latitude >= lat_min) & (ds.latitude <= lat_max),
            'longitude': (ds.longitude >= lon_min) & (ds.longitude <= lon_max)
        }, method='nearest')
        if verbose:
            std_print('DEBUG', 'su', f'Subset shape : {subset.sizes}')
        return subset if subset.sizes['latitude'] > 0 else None
    except Exception as e:
        if verbose:
            std_print('ERROR', 'su', 'extract_spatial_subset failed', str(e))
            traceback.print_exc()
        return None

def extract_regional_data(ds, track_df, track_id, variable, buffer_km=100):
    """Extract time series of a variable along a track"""
    track_id = int(track_id)
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    values = []
    valid_times = []

    for _, row in track.iterrows():
        ds, time_coord = get_time_coord(ds)
        lat_c = row['latitude']
        lon_c = row['longitude']
        t = pd.to_datetime(row['time'])

        subset = extract_spatial_subset(
            ds, lat_c, lon_c, t, 
            buffer_km=buffer_km, 
            time_coord=time_coord
        )
        if subset is not None and variable in subset.data_vars:
            val = float(subset[variable].mean().values)
            if not np.isnan(val):
                values.append(val)
                valid_times.append(t)
    return np.array(valid_times), np.array(values)

def extract_and_average_variable(ds, lat, lon, time, var_name, buffer_km=None, time_coord=None, aggregation='mean', verbose=False):
    """Extract variable at point with spatial averaging"""
    try:
        # time coordinate if not provided
        if time_coord is None:
            ds, time_coord = get_time_coord(ds)
            if verbose:
                print(f"Auto-detected time_coord: '{time_coord}'")
        if buffer_km is None:
            buffer_km = get_default_buffer_km(var_name)
            if verbose:
                std_print('DEBUG', 'su' f"Auto-selected buffer: {buffer_km} km for '{var_name}'")
        if verbose:
            std_print('DEBUG', 'su', f"Extracting {var_name} at ({lat:.2f}°N, {lon:.2f}°E)")
        # Extract subset
        subset = extract_spatial_subset(ds, lat, lon, time, buffer_km=buffer_km, time_coord=time_coord, verbose=verbose)
        if subset is None:
            if verbose:
                std_print('ERROR', 'su', 'Spatial subset returned None')
            return np.nan
        if subset.sizes.get('latitude', 0) == 0:
            if verbose:
                std_print('ERROR', 'su', 'Subset has no latitude points!')
            return np.nan
        if var_name not in subset.data_vars:
            if verbose:
                std_print('ERROR', 'su', f"Variable '{var_name}' not in subset")
                std_print('DEBUG', 'su', 'Available variables: {list(subset.data_vars)}')
            return np.nan
        var_data = subset[var_name]
        if verbose:
            std_print('DEBUG', 'su', f'Variable shape : {var_data.shape}')
        # Apply aggregation
        if aggregation == 'mean':
            result = float(var_data.mean().values)
        elif aggregation == 'sum':
            result = float(var_data.sum().values)
        elif aggregation == 'max':
            result = float(var_data.max().values)
        elif aggregation == 'min':
            result = float(var_data.min().values)
        elif aggregation == 'std':
            result = float(var_data.std().values)
        else:
            result = float(var_data.mean().values)
        if verbose:
            std_print('DEBUG', 'su', f'Result: {result:.6f}')
        return result if not np.isnan(result) else np.nan
    except Exception as e:
        if verbose:
            std_print('ERROR', 'su', 'Extraction failed', str(e))
            traceback.print_exc()
        return np.nan
