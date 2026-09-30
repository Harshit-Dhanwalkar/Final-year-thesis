"""
Soil Moisture–Atmosphere Feedback (SAF) coupling index
Chen et al. (2026), Nature Communications.

I(w, q) = r(w, q) * sigma(q)
computed for each calendar month, then averaged over JJAS.
"""

import os
import numpy as np
import xarray as xr

from utils import std_print, get_time_coord, calculate_buffer_deg


def compute_specific_humidity(d2m, sp):
    """
    d2m : 2m dew point temperature [K]
    sp  : surface pressure [Pa]
    returns q [kg/kg]
    """
    Td_C = d2m - 273.15
    e_hPa = 6.112 * np.exp(17.67 * Td_C / (Td_C + 243.5))
    sp_hPa = sp / 100.0
    q = 0.622 * e_hPa / (sp_hPa - e_hPa)
    return q


def compute_saf_index_from_file(monthly_nc, lsm_nc=None, months=(6, 7, 8, 9)):
    """
    Compute SAF coupling index from the monthly ERA5 file.

    Parameters
    ----------
    monthly_nc : str
        Path to saf_monthly_JJAS_*.nc (swvl1, d2m, sp)
    lsm_nc : str, optional
        Path to land-sea mask file (lsm variable). If None, tries to use
        lsm from monthly file if present.
    months : tuple
        Calendar months (default JJAS)

    Returns
    -------
    xarray.DataArray
        2D SAF coupling index (latitude, longitude)
    """
    ds = xr.open_dataset(monthly_nc)
    ds, time_coord = get_time_coord(ds)

    # Compute specific humidity
    ds = ds.assign(q=compute_specific_humidity(ds.d2m, ds.sp))

    # Select months
    ds_m = ds.sel(time=ds[time_coord].dt.month.isin(months))

    idx_list = []
    for m in months:
        ds_mm = ds_m.sel(time=ds_m[time_coord].dt.month == m)
        if ds_mm[time_coord].size < 5:
            std_print("WARN", "saf", f"Insufficient years for month {m}",
                      f"{ds_mm[time_coord].size} years")
            continue

        w = ds_mm.swvl1
        q = ds_mm.q

        # Anomalies across years
        w_anom = w - w.mean(dim=time_coord)
        q_anom = q - q.mean(dim=time_coord)

        # Correlation across years
        r = xr.corr(w_anom, q_anom, dim=time_coord)
        # Standard deviation of q
        std_q = q.std(dim=time_coord)

        I_m = r * std_q
        I_m.name = f"saf_month{m}"
        idx_list.append(I_m)

    if not idx_list:
        std_print("ERROR", "saf", "No valid months for SAF index")
        return None

    I = xr.concat(idx_list, dim="month").mean(dim="month")
    I.name = "saf_coupling_index"

    # Mask ocean
    if lsm_nc and os.path.exists(lsm_nc):
        lsm_ds = xr.open_dataset(lsm_nc)
        lsm_var = "lsm" if "lsm" in lsm_ds else list(lsm_ds.data_vars)[0]
        lsm = lsm_ds[lsm_var]
        # Drop non-spatial dims
        for d in lsm.dims:
            if d not in ("latitude", "longitude"):
                lsm = lsm.isel({d: 0}, drop=True)
        # Regrid to I's grid
        lsm = lsm.interp(latitude=I.latitude, longitude=I.longitude, method="nearest")
        I = I.where(lsm >= 0.5)
    elif "lsm" in ds:
        lsm = ds.lsm.isel(time=0, drop=True) if "time" in ds.lsm.dims else ds.lsm
        lsm = lsm.interp(latitude=I.latitude, longitude=I.longitude, method="nearest")
        I = I.where(lsm >= 0.5)

    ds.close()
    return I


def sample_saf_along_track(track_df, saf_da, buffer_km=100):
    """
    Sample the SAF index at each track point.

    Parameters
    ----------
    track_df : pd.DataFrame
        Must have 'latitude', 'longitude'
    saf_da : xarray.DataArray
        2D SAF index (lat, lon)
    buffer_km : float
        Spatial averaging radius

    Returns
    -------
    pd.DataFrame
        track_df with added 'saf_index' column
    """
    track_df = track_df.copy()
    saf_vals = []

    for _, row in track_df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        lat_buf, lon_buf = calculate_buffer_deg(lat, buffer_km)

        try:
            sub = saf_da.sel(
                latitude=slice(lat + lat_buf, lat - lat_buf),  # ERA5 descending
                longitude=slice(lon - lon_buf, lon + lon_buf),
            )
            if sub.size > 0 and not np.all(np.isnan(sub.values)):
                saf_vals.append(float(sub.mean().values))
            else:
                saf_vals.append(np.nan)
        except Exception:
            saf_vals.append(np.nan)

    track_df["saf_index"] = saf_vals
    return track_df
