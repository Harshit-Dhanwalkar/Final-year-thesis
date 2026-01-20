import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader

def is_point_land(lat: float, lon: float) -> bool:
    """Check if a point is over land."""
    try:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False
        point = sgeom.Point(lon, lat)
        # '110m'
        # land_shp = shpreader.natural_earth(
        #     resolution='110m',
        #     category='physical',
        #     name='land'
        # )
        land_shp = shpreader.natural_earth(
            resolution='50m',
            category='physical',
            name='land'
        )
        # '10m' if slow
        # land_shp = shpreader.natural_earth(
        #     resolution='10m',
        #     category='physical',
        #     name='land'
        # )
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
    - Returns -1 if no ocean→land transition
    """
    if 'is_land' not in track_df.columns:
        track_df['is_land'] = track_df.apply(
            lambda r: is_point_land(r['latitude'], r['longitude']), axis=1)

    origin = track_df['is_land'].iloc[0]

    # Find first ocean -> land transition
    for i in range(1, len(track_df)):
        if not track_df['is_land'].iloc[i-1] and track_df['is_land'].iloc[i]:
            return i, ('land_originated' if origin else 'ocean_originated'), True
    return -1, ('land_originated' if origin else 'ocean_originated'), False

def find_origin_type(track_df):
    """Classify track origin as ocean-originated or land-originated"""
    if track_df.empty:
        return 'unknown'

    first_point = track_df.iloc[0]
    lat, lon = first_point['latitude'], first_point['longitude']

    # Check if origin is over land
    if 'is_land' in track_df.columns:
        is_origin_land = first_point['is_land']
    else:
        is_origin_land = is_point_land(lat, lon)

    return 'land_originated' if is_origin_land else 'ocean_originated'

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
