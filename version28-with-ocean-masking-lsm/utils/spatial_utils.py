import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t as t_dist, pearsonr
from statsmodels.tsa.stattools import acf

from utils.io_utils import std_print
from utils.temporal_utils import get_time_coord

import traceback

# Clipped domain for Indian monsoon region
CLIPPED_MIN_LAT = 10.0
CLIPPED_MAX_LAT = 30.0
CLIPPED_MIN_LON = 70.0
CLIPPED_MAX_LON = 95.0

EARTH_R = 6371.0  # km


def clamp_to_domain(min_lat, max_lat, min_lon, max_lon):
    """Clamp spatial bounds to clipped monsoon domain"""
    min_lat = max(min_lat, CLIPPED_MIN_LAT)
    max_lat = min(max_lat, CLIPPED_MAX_LAT)
    min_lon = max(min_lon, CLIPPED_MIN_LON)
    max_lon = min(max_lon, CLIPPED_MAX_LON)
    return min_lat, max_lat, min_lon, max_lon


def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate great-circle distance (km). Accepts scalars OR numpy arrays."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def get_default_buffer_km(variable):
    """Get recommended buffer for each variable"""
    default_buffers = {
        "u10": 150,
        "v10": 150,
        "msl": 100,
        "tp": 350,
        "swvl1": 350,
        "speed": 50,
        "avg_slhf": 350,
        "avg_ishf": 350,
        "t2m": 100,
    }
    return default_buffers.get(variable.lower(), 100)


def calculate_buffer_deg(lat, buffer_km):
    """Calculate lat/lon buffer in degrees from distance in km"""
    lat_buffer = buffer_km / 111.0
    lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat)))
    return lat_buffer, lon_buffer


def extract_spatial_subset(ds, lat, lon, time_step, buffer_km=100):
    """Extract regional data around a point with latitude-dependent buffering.

    Note: time_coord is auto-detected internally; do NOT pass it as a keyword
    argument (the old signature had `time_coord` — that parameter is removed).
    """
    try:
        ds, time_coord = get_time_coord(ds)
        lat_buffer, lon_buffer = calculate_buffer_deg(lat, buffer_km)
        lat_min = lat - lat_buffer
        lat_max = lat + lat_buffer
        lon_min = lon - lon_buffer
        lon_max = lon + lon_buffer
        # Clamp to dataset bounds
        lat_min = max(lat_min, float(ds.latitude.min()))
        lat_max = min(lat_max, float(ds.latitude.max()))
        lon_min = max(lon_min, float(ds.longitude.min()))
        lon_max = min(lon_max, float(ds.longitude.max()))
        time_is_indexed = time_coord in ds.dims and time_coord in ds.indexes
        sel_dict = {
            "latitude": (ds.latitude >= lat_min) & (ds.latitude <= lat_max),
            "longitude": (ds.longitude >= lon_min) & (ds.longitude <= lon_max),
        }
        if time_is_indexed:
            sel_dict[time_coord] = time_step
        subset = ds.sel(sel_dict, method="nearest")
        return subset if subset.sizes["latitude"] > 0 else None
    except Exception as e:
        std_print("ERROR", "su", "extract_spatial_subset failed", str(e))
        traceback.print_exc()
        return None


def extract_regional_data(ds, track_df, track_id, variable, buffer_km=100):
    """Extract time series of a variable along a track"""
    track_id = int(track_id)
    track = track_df[track_df["track_id"] == track_id].sort_values("time")
    values = []
    valid_times = []
    for _, row in track.iterrows():
        lat_c = row["latitude"]
        lon_c = row["longitude"]
        t = pd.to_datetime(row["time"])
        subset = extract_spatial_subset(ds, lat_c, lon_c, t, buffer_km=buffer_km)
        if subset is not None and variable in subset.data_vars:
            val = float(subset[variable].mean().values)
            if not np.isnan(val):
                values.append(val)
                valid_times.append(t)
    return np.array(valid_times), np.array(values)


def extract_track_variable(
    ds, track_df, track_id, time_coord, var_name, buffer_km=350.0
):
    """Vectorised track-variable extraction (used in soil moisture memory analysis)."""
    track_id = int(track_id)
    track = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(track["time"])

    lat_buf, lon_buf = calculate_buffer_deg(track["latitude"].mean(), buffer_km)
    lat_min = max(track["latitude"].min() - lat_buf, float(ds.latitude.min()))
    lat_max = min(track["latitude"].max() + lat_buf, float(ds.latitude.max()))
    lon_min = max(track["longitude"].min() - lon_buf, float(ds.longitude.min()))
    lon_max = min(track["longitude"].max() + lon_buf, float(ds.longitude.max()))

    t_start = times.min() - pd.Timedelta(hours=1)
    t_end = times.max() + pd.Timedelta(hours=1)

    ds_sub = ds[var_name].sel(
        {
            time_coord: slice(t_start, t_end),
            "latitude": ds.latitude[
                (ds.latitude >= lat_min) & (ds.latitude <= lat_max)
            ],
            "longitude": ds.longitude[
                (ds.longitude >= lon_min) & (ds.longitude <= lon_max)
            ],
        }
    )
    track_lats = xr.DataArray(track["latitude"].values, dims="points")
    track_lons = xr.DataArray(track["longitude"].values, dims="points")
    track_times = xr.DataArray(times.values, dims="points")
    interpolated = ds_sub.interp(
        {time_coord: track_times, "latitude": track_lats, "longitude": track_lons},
        method="nearest",
    )
    values = interpolated.values
    valid = ~np.isnan(values)
    return times.values[valid], values[valid]


def extract_and_average_variable(
    ds, lat, lon, time_step, time_coord, var_name, buffer_km, aggregation="mean"
):
    """Extract variable at point with spatial averaging."""
    try:
        lat = float(lat) if hasattr(lat, "__float__") else lat
        lon = float(lon) if hasattr(lon, "__float__") else lon
        if buffer_km is None:
            buffer_km = get_default_buffer_km(var_name)
        subset = extract_spatial_subset(ds, lat, lon, time_step, buffer_km)
        if subset is None or subset.sizes.get("latitude", 0) == 0:
            return np.nan
        if var_name not in subset.data_vars:
            return np.nan
        var_data = subset[var_name]
        agg_map = {
            "mean": var_data.mean,
            "sum": var_data.sum,
            "max": var_data.max,
            "min": var_data.min,
            "std": var_data.std,
        }
        result = float(agg_map.get(aggregation, var_data.mean)().values)
        return result if not np.isnan(result) else np.nan
    except Exception as e:
        std_print("ERROR", "su", "Extraction failed", str(e))
        traceback.print_exc()
        return np.nan


def effective_sample_size(x, y, max_lag=10):
    """
    Effective sample size (n_eff) corrected for autocorrelation.

    ERA5 hourly time series are strongly autocorrelated, making raw Pearson
    p-values far too optimistic.  Uses the first-order AR approximation:

        n_eff = n * (1 - rho1_x * rho1_y) / (1 + rho1_x * rho1_y)

    where rho1 is the lag-1 ACF of each series.

    Reference: Bretherton et al. (1999), J. Climate 12, 1990-2009.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = np.asarray(x)[mask]
    y_clean = np.asarray(y)[mask]
    n = len(x_clean)
    if n < 20:
        return n
    try:
        acf_x = acf(x_clean, nlags=max_lag, fft=True)
        acf_y = acf(y_clean, nlags=max_lag, fft=True)
        rho1_x = acf_x[1] if len(acf_x) > 1 else 0.0
        rho1_y = acf_y[1] if len(acf_y) > 1 else 0.0
        denom = 1.0 + rho1_x * rho1_y
        if denom <= 0:
            return n
        n_eff = n * (1.0 - rho1_x * rho1_y) / denom
        return int(max(min(n_eff, n), max(n // 10, 5)))
    except Exception as e:
        std_print("WARN", "su", "effective_sample_size failed, returning n", str(e))
        return n


def pearsonr_corrected(x, y):
    """
    Pearson r with autocorrelation-corrected p-value.

    Parameters
    ----------
    x, y : array-like (may contain NaN)

    Returns
    -------
    r      : Pearson correlation coefficient
    p_raw  : Standard (uncorrected) p-value — unreliable for autocorrelated data
    p_corr : Corrected p-value using n_eff  — use this for significance testing
    n_eff  : Effective sample size
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_c = np.asarray(x)[mask]
    y_c = np.asarray(y)[mask]
    if len(x_c) < 5:
        return np.nan, np.nan, np.nan, len(x_c)
    r, p_raw = pearsonr(x_c, y_c)
    n_eff = effective_sample_size(x_c, y_c)
    if n_eff < 3:
        return float(r), float(p_raw), np.nan, n_eff
    t_stat = r * np.sqrt((n_eff - 2) / max(1.0 - r**2, 1e-10))
    p_corr = float(2 * t_dist.sf(abs(t_stat), df=n_eff - 2))
    return float(r), float(p_raw), float(p_corr), int(n_eff)
