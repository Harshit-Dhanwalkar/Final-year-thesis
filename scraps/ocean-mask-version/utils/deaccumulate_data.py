import numpy as np
import pandas as pd

def deaccumulate_tp(ds, t, lat, lon, buffer_km=100, time_coord='time'):
    lat_buffer = buffer_km / 111.0
    lon_buffer = buffer_km / (111.0 * np.cos(np.radians(lat)))
    try:
        # Handle lon wrapping
        ds_lon = ds.longitude.values
        if ds_lon.max() > 180 and lon < 0:
            lon_adj = lon + 360
        else:
            lon_adj = lon
        # Current time step
        ds_times = pd.to_datetime(ds[time_coord].values)
        current_idx = np.argmin(np.abs(ds_times - pd.to_datetime(t)))
        current_time = ds_times[current_idx]
        subset_current = ds.sel(
            **{time_coord: current_time},
            latitude=slice(lat + lat_buffer, lat - lat_buffer),
            longitude=slice(lon_adj - lon_buffer, lon_adj + lon_buffer)
        )
        tp_current = float(subset_current['tp'].mean().values)
        # Previous time step (for de-accumulation)
        if current_idx > 0:
            prev_time = ds_times[current_idx - 1]
            subset_prev = ds.sel(
                **{time_coord: prev_time},
                latitude=slice(lat + lat_buffer, lat - lat_buffer),
                longitude=slice(lon_adj - lon_buffer, lon_adj + lon_buffer)
            )
            tp_prev = float(subset_prev['tp'].mean().values)
            # De-accumulate (hourly)
            tp_hourly = (tp_current - tp_prev) * 1000  # m → mm
            return max(0, tp_hourly)  # Avoid negative from numerical noise
        else:
            # First time step of day: use raw value
            tp_hourly = tp_current * 1000  # m → mm
            return tp_hourly
    except Exception as e:
        print(f"Error de-accumulating TP at {t}: {e}")
        return np.nan
