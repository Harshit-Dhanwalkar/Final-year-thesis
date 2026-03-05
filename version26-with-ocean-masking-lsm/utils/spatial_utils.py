import numpy as np
import pandas as pd
import xarray as xr
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
    """Calculate great-circle distance (km). Accepts scalars OR numpy arrays.
    Pure NumPy: no compilation cost, handles vectorised array calls natively.
    """
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
    # default_buffers = {
    #     'u10': 150,         # Wind U component
    #     'v10': 150,         # Wind V component
    #     'msl': 100,         # Mean sea level pressure
    #     'tp': 150,          # Total precipitation
    #     'swvl1': 100,       # Soil water volume layer 1
    #     'speed': 50,        # Track speed (tighter)
    #     'slhf': 150,        # Surface latent heat flux
    #     "avg_slhf": 150,    # Surface latent heat flux
    #     "avg_ishf": 150,    # Surface Sensible heat flux
    #     't2m': 100,         # 2m temperature
    # }
    default_buffers = {
        "u10": 150,  # Wind U component
        "v10": 150,  # Wind V component
        "msl": 100,  # Mean sea level pressure
        "tp": 350,  # Total precipitation
        "swvl1": 350,  # Soil water volume layer 1
        "speed": 50,  # Track speed
        "avg_slhf": 350,  # Surface latent heat flux
        "avg_ishf": 350,  # Surface Sensible heat flux
        "t2m": 100,  # 2m temperature
    }
    return default_buffers.get(variable.lower(), 100)


def calculate_buffer_deg(lat, buffer_km):
    """Calculate lat/lon buffer in degrees from distance in km"""
    lat_buffer = buffer_km / 111.0
    lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat)))
    return lat_buffer, lon_buffer


def extract_spatial_subset(ds, lat, lon, time_step, time_coord, buffer_km=100):
    """Extract regional data around a point with latitude-dependent buffering"""
    try:
        # time coordinate if not provided
        ds, time_coord = get_time_coord(ds)
        # std_print("DEBUG", f"Calculating buffer: {buffer_km} km")
        lat_buffer, lon_buffer = calculate_buffer_deg(lat, buffer_km)
        # Spatial Bounds
        lat_min = lat - lat_buffer
        lat_max = lat + lat_buffer
        lon_min = lon - lon_buffer
        lon_max = lon + lon_buffer
        # Clamp to dataset bounds
        lat_min = max(lat_min, float(ds.latitude.min()))
        lat_max = min(lat_max, float(ds.latitude.max()))
        lon_min = max(lon_min, float(ds.longitude.min()))
        lon_max = min(lon_max, float(ds.longitude.max()))
        # std_print("DEBUG", f"Bounds - Lat: [{lat_min:.2f}, {lat_max:.2f}], Lon: [{lon_min:.2f}, {lon_max:.2f}]")
        time_is_indexed = time_coord in ds.dims and time_coord in ds.indexes
        sel_dict = {
            "latitude": (ds.latitude >= lat_min) & (ds.latitude <= lat_max),
            "longitude": (ds.longitude >= lon_min) & (ds.longitude <= lon_max),
        }
        if time_is_indexed:
            sel_dict[time_coord] = time_step
        subset = ds.sel(sel_dict, method="nearest")
        # std_print("DEBUG", "su", f"Subset shape : {subset.sizes}")
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
        ds, time_coord = get_time_coord(ds)
        lat_c = row["latitude"]
        lon_c = row["longitude"]
        t = pd.to_datetime(row["time"])

        subset = extract_spatial_subset(
            ds, lat_c, lon_c, t, buffer_km=buffer_km, time_coord=time_coord
        )
        if subset is not None and variable in subset.data_vars:
            val = float(subset[variable].mean().values)
            if not np.isnan(val):
                values.append(val)
                valid_times.append(t)
    return np.array(valid_times), np.array(values)


# Used in soil_moisture_analysis
def extract_track_variable(
    ds, track_df, track_id, time_coord, var_name, buffer_km=350.0
):
    """For soil_memory_by_track variable extracking"""
    track_id = int(track_id)
    track = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(track["time"])

    # spatial bounding box for the whole track (with buffer)
    lat_buf, lon_buf = calculate_buffer_deg(track["latitude"].mean(), buffer_km)
    lat_min = track["latitude"].min() - lat_buf
    lat_max = track["latitude"].max() + lat_buf
    lon_min = track["longitude"].min() - lon_buf
    lon_max = track["longitude"].max() + lon_buf

    # Clamp to dataset extent
    lat_min = max(lat_min, float(ds.latitude.min()))
    lat_max = min(lat_max, float(ds.latitude.max()))
    lon_min = max(lon_min, float(ds.longitude.min()))
    lon_max = min(lon_max, float(ds.longitude.max()))

    # Spatial subset
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
    # vectorised nearest-time + spatial interpolation along track
    track_lats = xr.DataArray(track["latitude"].values, dims="points")
    track_lons = xr.DataArray(track["longitude"].values, dims="points")
    track_times_xr = xr.DataArray(times.values, dims="points")
    interpolated = ds_sub.interp(
        {time_coord: track_times_xr, "latitude": track_lats, "longitude": track_lons},
        method="nearest",
    )
    values = interpolated.values  # shape (N_points,)
    valid = ~np.isnan(values)
    return times.values[valid], values[valid]


def extract_and_average_variable(
    ds, lat, lon, time_step, time_coord, var_name, buffer_km, aggregation="mean"
):
    """Extract variable at point with spatial averaging"""
    try:
        lat = float(lat) if hasattr(lat, "__float__") else lat
        lon = float(lon) if hasattr(lon, "__float__") else lon
        # time coordinate if not provided
        # if time_coord is None:
        #     ds, time_coord = get_time_coord(ds)
        #     std_print("DEBUG", "su", f"Auto-detected time_coord {time_coord}")
        if buffer_km is None:
            buffer_km = get_default_buffer_km(var_name)
            std_print(
                "DEBUG", "su" f"Auto-selected buffer {buffer_km} km for {var_name}"
            )
            std_print(
                "DEBUG", "su", f"Extracting {var_name} at ({lat:.2f}°N, {lon:.2f}°E)"
            )
        # Extract subset
        subset = extract_spatial_subset(ds, lat, lon, time_step, time_coord, buffer_km)
        if subset is None:
            std_print("ERROR", "su", "Spatial subset returned None")
            return np.nan
        if subset.sizes.get("latitude", 0) == 0:
            std_print("ERROR", "su", "Subset has no latitude points!")
            return np.nan
        if var_name not in subset.data_vars:
            std_print("ERROR", "su", f"Variable '{var_name}' not in subset")
            std_print("DEBUG", "su", f"Available variables {list(subset.data_vars)}")
            return np.nan
        var_data = subset[var_name]
        # std_print("DEBUG", "su", f"Variable shape : {var_data.shape}")
        # Apply aggregation
        if aggregation == "mean":
            result = float(var_data.mean().values)
        elif aggregation == "sum":
            result = float(var_data.sum().values)
        elif aggregation == "max":
            result = float(var_data.max().values)
        elif aggregation == "min":
            result = float(var_data.min().values)
        elif aggregation == "std":
            result = float(var_data.std().values)
        else:
            result = float(var_data.mean().values)
        # std_print("DEBUG", "su", f"Result {result:.6f}")
        return result if not np.isnan(result) else np.nan
    except Exception as e:
        std_print("ERROR", "su", "Extraction failed", str(e))
        traceback.print_exc()
        return np.nan


# def extract_spatial_mean(ds, var_name, lat, lon, time_step, buffer_km):
#     """Extract spatial mean of variable at point"""
#     ds, time_coord = get_time_coord(ds)
#     subset = extract_spatial_subset(ds, lat, lon, time_step, time_coord, buffer_km)
#     if subset is None or subset.sizes.get("latitude", 0) == 0:
#         return np.nan
#     if var_name in subset:
#         try:
#             val = float(subset[var_name].mean().values)
#             return val if not np.isnan(val) else np.nan
#         except:
#             return np.nan
#     else:
#         return np.nan


# def effective_sample_size(x, y, max_lag=10):
#     """Calculate effective sample size accounting for autocorrelation"""
#     mask = ~np.isnan(x) & ~np.isnan(y)
#     x_clean = x[mask]
#     y_clean = y[mask]
#     if len(x_clean) < 20:
#         return len(x_clean)
#     try:
#         acf_x = acf(x_clean, nlags=max_lag, fft=True)
#         acf_y = acf(y_clean, nlags=max_lag, fft=True)
#         rho1 = acf_x[1] if len(acf_x) > 1 else 0
#         rho2 = acf_y[1] if len(acf_y) > 1 else 0
#         n = len(x_clean)
#         n_eff = n * (1 - rho1 * rho2) / (1 + rho1 * rho2)
#         n_eff = max(min(n_eff, n), n / 10)
#         return n_eff
#     except:
#         return len(x_clean)
