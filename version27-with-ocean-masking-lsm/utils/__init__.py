import os
import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader

from .track_utils import (
    is_point_land,
    is_land,
    find_landfall_time,
    get_landfall_index,
    find_origin_type,
    calculate_track_curvature,
    classify_track_status,
    get_landfall_status_array,
)

from .spatial_utils import (
    haversine_distance,
    clamp_to_domain,
    get_default_buffer_km,
    calculate_buffer_deg,
    extract_spatial_subset,
    extract_regional_data,
    extract_track_variable,
    extract_and_average_variable,
    pearsonr_corrected,
    effective_sample_size,
)

from .temporal_utils import get_time_coord, get_time_window, validate_time_bounds

from .io_utils import std_print, export_results_to_csv

from .plot_utils import (
    setup_map_extent,
    add_land_ocean_shading,
    get_standardized_colors,
    common_bbox,
    legend_style,
)

__all__ = [
    "is_point_land",
    "is_land",
    "find_landfall_time",
    "get_landfall_index",
    "find_origin_type",
    "calculate_track_curvature",
    "classify_track_status",
    "get_landfall_status_array",
    "haversine_distance",
    "calculate_track_curvature",
    "extract_spatial_subset",
    "extract_regional_data",
    "extract_track_variable",
    "get_default_buffer_km",
    "calculate_buffer_deg",
    "extract_and_average_variable",
    "pearsonr_corrected",
    "effective_sample_size",
    "clamp_to_domain",
    "get_time_coord",
    "get_time_window",
    "validate_time_bounds",
    "std_print",
    "export_results_to_csv",
    "setup_map_extent",
    "add_land_ocean_shading",
    "get_standardized_colors",
    "common_bbox",
    "legend_style",
]
