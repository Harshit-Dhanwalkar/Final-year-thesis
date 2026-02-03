import numpy as np
import pandas as pd
from utils.temporal_utils import get_time_coord

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

def calculate_buffer_deg(lat, buffer_km):
    """
    Calculate lat/lon buffer in degrees from distance in km.
    Returns:
        tuple: (lat_buffer_deg, lon_buffer_deg)
    """
    lat_buffer = buffer_km / 111.0
    lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat)))
    return lat_buffer, lon_buffer

# def extract_spatial_subset(ds, lat, lon, time, buffer_km=100, time_coord='time'):
#     try:
#         lat_buffer, lon_buffer = calculate_buffer_deg(lat, buffer_km)
#         # order for slice (north to south)
#         lat_max = lat + lat_buffer
#         lat_min = lat - lat_buffer
#
#         # For latitude, slice from higher to lower values
#         if lat_max < lat_min:
#             lat_max, lat_min = lat_min, lat_max
#
#         # For longitude,wrap-around if needed
#         lon_min = lon - lon_buffer
#         lon_max = lon + lon_buffer
#
#         subset = ds.sel({
#             time_coord: time,
#             'latitude': slice(lat_max, lat_min),  # North to South
#             'longitude': slice(lon_min, lon_max)   # West to East
#         }, method='nearest')
#
#         return subset
#     except Exception as e:
#         print(f"Error in extract_spatial_subset: {e}")
#         return None

def extract_spatial_subset(ds, lat, lon, time, buffer_km=100, time_coord='time'):
    """Extract regional data around a point with latitude-dependent buffering"""
    try:
        lat_buffer, lon_buffer = calculate_buffer_deg(lat, buffer_km)
        # Create bounds
        lat_min, lat_max = lat - lat_buffer, lat + lat_buffer
        lon_min, lon_max = lon - lon_buffer, lon + lon_buffer

        # Select using where/isin
        lat_min = max(lat_min, float(ds.latitude.min()))
        lat_max = min(lat_max, float(ds.latitude.max()))
        lon_min = max(lon_min, float(ds.longitude.min()))
        lon_max = min(lon_max, float(ds.longitude.max()))

        subset = ds.sel({
            time_coord: time,
            'latitude': (ds.latitude >= lat_min) & (ds.latitude <= lat_max),
            'longitude': (ds.longitude >= lon_min) & (ds.longitude <= lon_max)
        }, method='nearest')

        return subset if subset.sizes['latitude'] > 0 else None
    except Exception as e:
        print(f"Error in extract_spatial_subset: {e}")
        return None

def extract_regional_data(ds, track_df, track_id, variable, buffer_km = 100):
    track_id = int(track_id)
    track = track_df[track_df['track_id'] == track_id].sort_values('time')
    values = []
    valid_times = []

    for _, row in track.iterrows():
        # time_coord = "time" if "time" in ds.coords else "valid_time"
        ds, time_coord = get_time_coord(ds)
        lat_c = row['latitude']
        lon_c = row['longitude']
        t = pd.to_datetime(row['time'])

        subset = extract_spatial_subset(ds, lat_c, lon_c, t, buffer_km=buffer_km, time_coord=time_coord)
        if variable in subset.data_vars:
            val = float(subset[variable].mean().values)
            if not np.isnan(val):
                values.append(val)
                valid_times.append(t)
    return np.array(valid_times), np.array(values)
