import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Month colors
MONTH_COLORS = {
    "June": "#FFA500",
    "July": "#FF4500",
    "August": "#4169E1",
    "September": "#9370DB",
}

# Color scheme for consistency
COLOR_SCHEME = {
    "ocean": "#87CEEB",
    "land": "#D2B48C",
    "positive": "#1f77b4",
    "negative": "#ff7f0e",
    "track": "#000000",
    "grid": "#CCCCCC",
}


def get_standardized_colors(color_type="month", month=None):
    """
    color_type: 'month', 'ocean_land', 'correlation'
    month: Month name (if color_type='month')
    """
    if color_type == "month" and month:
        return MONTH_COLORS.get(month, "#000000")
    elif color_type == "ocean_land":
        return {"ocean": COLOR_SCHEME["ocean"], "land": COLOR_SCHEME["land"]}
    elif color_type == "correlation":
        return {
            "positive": COLOR_SCHEME["positive"],
            "negative": COLOR_SCHEME["negative"],
        }
    else:
        return COLOR_SCHEME


def setup_map_extent(track_df, buffer_deg=5.0):
    """Calculate map extent with buffer around track.
    Args:
        track_df: DataFrame with 'latitude', 'longitude'
        buffer_deg: Buffer in degrees
    Returns:
        tuple: (min_lon, max_lon, min_lat, max_lat)
    """
    min_lat = track_df["latitude"].min() - buffer_deg
    max_lat = track_df["latitude"].max() + buffer_deg
    min_lon = track_df["longitude"].min() - buffer_deg
    max_lon = track_df["longitude"].max() + buffer_deg

    return min_lon, max_lon, min_lat, max_lat


def add_land_ocean_shading(
    ax, time_data, is_land_data, land_color="bisque", ocean_color="azure"
):
    # Shade land regions
    ax.fill_between(
        time_data,
        0,
        1,
        where=is_land_data,
        color=land_color,
        alpha=0.5,
        transform=ax.get_xaxis_transform(),
        label="Land",
        zorder=0,
    )

    # Shade ocean regions
    ax.fill_between(
        time_data,
        0,
        1,
        where=~is_land_data,
        color=ocean_color,
        alpha=0.5,
        transform=ax.get_xaxis_transform(),
        label="Ocean",
        zorder=0,
    )


def common_bbox(custom_params=None):
    """Plot Text annotations"""
    bbox_default = {
        "boxstyle": "round,pad=0.05",
        "facecolor": "white",
        "alpha": 0.8,
        "edgecolor": "none",
        "linewidth": 0.8,
    }
    if custom_params:
        bbox_default.update(custom_params)
    return bbox_default


def legend_style(custom_params=None):
    legend_default = {
        "fontsize": 8,
        "frameon": True,
        "facecolor": "white",
        "edgecolor": "none",
        "framealpha": 0.8,
        "fancybox": True,
        "shadow": True,
    }
    if custom_params:
        legend_default.update(custom_params)
    return legend_default
