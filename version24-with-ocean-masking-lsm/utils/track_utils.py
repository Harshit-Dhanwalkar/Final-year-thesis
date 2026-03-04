import functools
import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import shapely.strtree as strtree_mod
import cartopy.io.shapereader as shpreader

from utils.io_utils import std_print
from utils.spatial_utils import haversine_distance

LAND_TREE: strtree_mod.STRtree | None = None
LAND_GEOMS: list = []


def build_land_index() -> strtree_mod.STRtree:
    """Build (or return cached) STRtree from Natural Earth land polygons"""
    global LAND_TREE, LAND_GEOMS
    if LAND_TREE is not None:
        return LAND_TREE
    land_shp = shpreader.natural_earth(
        resolution="50m",
        category="physical",
        name="land",
    )
    reader = shpreader.Reader(land_shp)
    LAND_GEOMS = list(reader.geometries())
    LAND_TREE = strtree_mod.STRtree(LAND_GEOMS)
    return LAND_TREE


def is_point_land(lat, lon):
    """Check if a point is over land."""
    try:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False
        tree = build_land_index()
        point = sgeom.Point(lon, lat)
        # query() returns indices of geometries whose bounding box intersects point
        candidates = tree.query(point)
        for idx in candidates:
            if LAND_GEOMS[idx].contains(point):
                return True
        return False
    except Exception as e:
        std_print("ERROR", "tu", "Failed determining point is land", str(e))
        return False


def is_land(lats, lons):
    """Classify an entire array of (lat, lon) pairs in one call"""
    tree = build_land_index()
    result = np.zeros(len(lats), dtype=bool)
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        point = sgeom.Point(lon, lat)
        candidates = tree.query(point)
        for idx in candidates:
            if LAND_GEOMS[idx].contains(point):
                result[i] = True
                break
    return result


def find_landfall_time(track_df):
    """Find the first ocean→land transition time (landfall)"""
    if track_df.empty:
        return None
    # Use pre-computed is_land column if available
    if "is_land" in track_df.columns:
        landfall_rows = track_df[track_df["is_land"] == True]
        if not landfall_rows.empty:
            return pd.to_datetime(landfall_rows.iloc[0]["time"])
    return None


def get_landfall_status_array(track_df):
    """
    Returns:
        np.array of bool (True=land, False=ocean)
    """
    if "is_land" not in track_df.columns:
        track_df["is_land"] = is_land(
            track_df["latitude"].values, track_df["longitude"].values
        )
    return track_df["is_land"].values


def get_landfall_index(track_df):
    """
    landfall_idx, origin_type, is_valid_transition
    Returns:
        -1 if no ocean -> land transition
    """
    if "is_land" not in track_df.columns:
        track_df["is_land"] = is_land(
            track_df["latitude"].values, track_df["longitude"].values
        )

    origin = track_df["is_land"].iloc[0]
    ###
    ocean_start_required = 3
    ocean_points_at_start = 0
    # Count consecutive ocean points at start
    for i in range(min(len(track_df), ocean_start_required)):
        if not track_df["is_land"].iloc[i]:
            ocean_points_at_start += 1
    ####

    # Find first ocean -> land transition
    for i in range(1, len(track_df)):
        if not track_df["is_land"].iloc[i - 1] and track_df["is_land"].iloc[i]:
            return i, ("land_originated" if origin else "ocean_originated"), True
    return -1, ("land_originated" if origin else "ocean_originated"), False


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
        return "unknown"

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
    lat, lon = first_point["latitude"], first_point["longitude"]

    # Check if origin is over land
    if "is_land" in track_df.columns:
        is_origin_land = first_point["is_land"]
    else:
        is_origin_land = is_point_land(lat, lon)
    return "land_originated" if is_origin_land else "ocean_originated"


def calculate_track_curvature(track_df):
    """Calculate track curvature to detect stationary/recurving systems"""
    if len(track_df) < 10:
        return 0.0
    lons = track_df["longitude"].values
    lats = track_df["latitude"].values
    # Vectorised pairwise distances along the track
    segment_dists = haversine_distance(lons[:-1], lats[:-1], lons[1:], lats[1:])
    total_distance = float(np.sum(segment_dists))
    net_distance = float(haversine_distance(lons[0], lats[0], lons[-1], lats[-1]))
    return total_distance / net_distance if net_distance > 0 else 1.0


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
    track_times = pd.to_datetime(track_df["time"])
    result = {
        "origin": find_origin_type(track_df),
        "has_landfall": False,
        "landfall_time": None,
        "origin_time": track_times.min(),
        "dissipation_time": track_times.max(),
        "duration_hours": (track_times.max() - track_times.min()).total_seconds()
        / 3600,
        "num_points": len(track_df),
    }
    landfall_time = find_landfall_time(track_df)
    if landfall_time is not None:
        result["has_landfall"] = True
        result["landfall_time"] = landfall_time
    return result
