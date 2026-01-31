import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader

from utils.spatial_utils import haversine_distance

def is_point_land(lat: float, lon: float) -> bool:
    """Check if a point is over land."""
    try:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False
        point = sgeom.Point(lon, lat)
        # '110m'
        # land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
        # '50m'
        land_shp = shpreader.natural_earth(resolution='50m', category='physical', name='land')
        # '10m' if slow
        # land_shp = shpreader.natural_earth(resolution='10m', category='physical', name='land')
        reader = shpreader.Reader(land_shp)
        for geometry in reader.geometries():
            if geometry.contains(point):
                return True
        return False
    except Exception:
        return False

def find_landfall_time(track_df):
    """Find the first ocean→land transition time (landfall)"""
    if track_df.empty:
        return None
    # Use pre-computed is_land column if available
    if 'is_land' in track_df.columns:
        landfall_rows = track_df[track_df['is_land'] == True]
        if not landfall_rows.empty:
            return pd.to_datetime(landfall_rows.iloc[0]['time'])
    return None

def get_landfall_status_array(track_df):
    """
    Returns:
        np.array of bool (True=land, False=ocean)
    """
    if 'is_land' not in track_df.columns:
        track_df['is_land'] = track_df.apply(
            lambda row: is_point_land(row['latitude'], row['longitude']),
            axis=1
        )
    return track_df['is_land'].values

def get_landfall_index(track_df):
    """
    landfall_idx, origin_type, is_valid_transition
    - Returns -1 if no ocean -> land transition
    """
    if 'is_land' not in track_df.columns:
        track_df['is_land'] = track_df.apply(
            lambda r: is_point_land(r['latitude'], r['longitude']), axis=1)

    origin = track_df['is_land'].iloc[0]
    ###
    ocean_start_required = 3
    ocean_points_at_start = 0
    # Count consecutive ocean points at start
    for i in range(min(len(track_df), ocean_start_required)):
        if not track_df['is_land'].iloc[i]:
            ocean_points_at_start += 1
    ####

    # Find first ocean -> land transition
    for i in range(1, len(track_df)):
        if not track_df['is_land'].iloc[i-1] and track_df['is_land'].iloc[i]:
            return i, ('land_originated' if origin else 'ocean_originated'), True
    return -1, ('land_originated' if origin else 'ocean_originated'), False
# def get_landfall_index(track_df):
#     """
#     landfall_idx, origin_type, is_valid_transition
#     - Returns -1 if no ocean→land transition
#     - Check: Must have significant ocean portion before landfall
#     """
#     if 'is_land' not in track_df.columns:
#         track_df['is_land'] = track_df.apply(
#             lambda r: is_point_land(r['latitude'], r['longitude']), axis=1)
#     ocean_start_required = 3
#     origin = track_df['is_land'].iloc[0]
#     ocean_points_at_start = 0
#     # Count consecutive ocean points at start
#     for i in range(min(len(track_df), ocean_start_required)):
#         if not track_df['is_land'].iloc[i]:
#             ocean_points_at_start += 1
#     # Strict requirement: MD should start over ocean
#     if ocean_points_at_start < ocean_start_required:
#         return -1, 'invalid_origin', False
#     # Find first ocean -> land transition
#     for i in range(1, len(track_df)):
#         if not track_df['is_land'].iloc[i-1] and track_df['is_land'].iloc[i]:
#             # Verify this is a true landfall (not just a coastal crossing)
#             # Check next few points to confirm land persistence
#             land_persist_check = 3
#             if i + land_persist_check < len(track_df):
#                 next_points_land = all(track_df['is_land'].iloc[i:i+land_persist_check])
#                 if next_points_land:
#                     return i, 'ocean_originated', True
#     return -1, 'ocean_originated', False

def find_origin_type(track_df):
    """Classify track origin as ocean-originated or land-originated"""
    if track_df.empty:
        return 'unknown'

    # # check first 3 points
    # n_check = min(3, len(track_df))
    # first_points = track_df.head(n_check)
    # land_count = 0

    # for _, row in first_points.iterrows():
    #     lat, lon = row['latitude'], row['longitude']
    #     if 'is_land' in track_df.columns:
    #         is_land = row['is_land']
    #     else:
    #         is_land = is_point_land(lat, lon)
    #     land_count += is_land
    #
    # # Check if origin is over land
    # if land_count > n_check / 2:
    #     return 'land_originated'
    # else:
    #     return 'ocean_originated'

    # check first point
    first_point = track_df.iloc[0]
    lat, lon = first_point['latitude'], first_point['longitude']

    # Check if origin is over land
    if 'is_land' in track_df.columns:
        is_origin_land = first_point['is_land']
    else:
        is_origin_land = is_point_land(lat, lon)
    return 'land_originated' if is_origin_land else 'ocean_originated'

def calculate_track_curvature(track_df):
    """Calculate track curvature to detect stationary/recurving systems"""
    if len(track_df) < 10:
        return 0
    lons = track_df['longitude'].values
    lats = track_df['latitude'].values
    # Calculate total path length
    total_distance = 0
    for i in range(1, len(track_df)):
        total_distance += haversine_distance(lons[i-1], lats[i-1], lons[i], lats[i])
    # Net displacement
    net_distance = haversine_distance(lons[0], lats[0], lons[-1], lats[-1])
    # Curvature ratio (1 = straight line, >1 = curved)
    if net_distance > 0:
        return total_distance / net_distance
    return 1

def classify_track_status(track_df):
    """
    Returns:
            - origin: 'ocean' or 'land'
            - has_landfall: bool
            - landfall_time: pd.Timestamp or None
            - duration_hours: float
            - origin_time: pd.Timestamp
            - dissipation_time: pd.Timestamp
    """
    track_times = pd.to_datetime(track_df['time'])
    result = {
        'origin': find_origin_type(track_df),
        'has_landfall': False,
        'landfall_time': None,
        'origin_time': track_times.min(),
        'dissipation_time': track_times.max(),
        'duration_hours': (track_times.max() - track_times.min()).total_seconds() / 3600,
        'num_points': len(track_df)
    }
    landfall_time = find_landfall_time(track_df)
    if landfall_time is not None:
        result['has_landfall'] = True
        result['landfall_time'] = landfall_time

    return result
