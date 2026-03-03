# INFO: Method:
# 1. Get MD center. For r = 0 (distance from MD center), linearly interpolating
# to get the grid point value at the exact center location, take that as it is
# with no averaging (since it is a single interpolated point), and plot it.
# 2. Take an annulus (disk) of radius 25km with tolerance of +/-10km, so all grid
# points from 15-35km distance are included. Within this annulus, divided the area
# into 16 equal sectors of 22.5 degree each (N, NNE, NE, ENE, E, ESE, SE, SSE, S,
# SSW, SW, WSW, W, WNW, NW, NNW). For each directional sector, find all grid points
# that fall in that sector-annulus segment (the trapezium-like region) and average
# those points to get one value per sector. Then average these 16 sector values
# together to get a single azimuthally-averaged value for the 25km radius, and then
# plot it.
# 3. An annulus of radius 50km with tolerance of +/-10km (all points from 40-60km)
# and repeat the same sectoring and averaging process. Continuing this for all radii:
# [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400] km

import os
from numba import njit, prange
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from utils import std_print
import traceback
import warnings

from utils import legend_style

warnings.filterwarnings("ignore", message="Mean of empty slice")

EARTH_R = 6371000.0
EARTH_OMEGA = 7.2921159e-5

VARIABLE_UNITS = {
    "u10": "m/s",
    "v10": "m/s",
    "wind10": "m/s",
    "vr10": "m/s",
    "vt10": "m/s",
    "vor_rel": "x10⁻⁵ s⁻¹",
    "vor_abs": "x10⁻⁵ s⁻¹",
    "vor_rel_polar": "x10⁻⁵ s⁻¹",
    "vor_abs_polar": "x10⁻⁵ s⁻¹",
    "speed_kmh": "km/h",
    "speed_ms": "m/s",
    "msl": "hPa",
    "sp": "hPa",
    "tp": "mm/hr",
    "t2m": "K",
    "d2m": "K",
    "stl1": "K",
    "swvl1": "m³/m³",
    "avg_slhtf": "W/m²",
    "avg_ishf": "W/m²",
    "total_hf": "W/m²",
    "w": "g/kg",
    "dswvl1_dt": "(m³/m³/hr)"
}

VARIABLE_NAMES = {
    "u10": "Zonal Wind (10m)",
    "v10": "Meridional Wind (10m)",
    "wind10": "Wind Speed (10m)",
    "vr10": "Radial Wind (Inflow/Outflow)",
    "vt10": "Tangential Wind (Swirl)",
    "vor_rel": "Relative Vorticity",
    "vor_abs": "Absolute Vorticity",
    "vor_rel_polar": "Relative Vorticity",
    "vor_abs_polar": "Absolute Vorticity",
    "speed_kmh": "Translation Speed",
    "speed_ms": "Translation Speed",
    "msl": "Mean Sea Level Pressure",
    "sp": "Surface Pressure",
    "tp": "Total Precipitation",
    "t2m": "2m Air Temperature",
    "d2m": "2m Dew Point Temperature",
    "stl1": "Soil Temperature (0-7cm)",
    "swvl1": "Soil Water Volumatric Layer-1 (0-7cm)",
    "avg_slhtf": "Surface Latent Heat Flux",
    "avg_ishf": "Surface Sensible Heat Flux",
    "total_hf": "Total Surface Heat Flux",
    "w": "Water Vapour Mixing Ratio",
    "dswvl1_dt": "Soil Moisture Tendency  ∂SWVL1/∂t"
}


# Helper functions
def precompute_sector_masks(
    lonc,
    latc,
    lon_grid,
    lat_grid,
    radii_km,
    directions_deg,
    radius_tolerance_percent=10,  # 10% of radius
    min_radius_tolerance_km=10.0,  # Minimum 10km
    direction_tolerance_deg=11.25,  # \pm 11.25 deg (half of 22.5 deg sector)
):
    """Compute all sector masks for one track centre position"""
    # Compute distance & direction for every  grid point
    dlat_km = (lat_grid - latc) * 111.0
    dlon_km = (lon_grid - lonc) * 111.0 * np.cos(np.radians(latc))
    distances = np.sqrt(dlat_km**2 + dlon_km**2)  # (nlat, nlon)
    directions = (np.degrees(np.arctan2(dlon_km, dlat_km)) + 360) % 360
    masks = {}
    for j, r in enumerate(radii_km):
        if r == 0:
            continue  # handled separately in the centre-point branch
        tol = max(min_radius_tolerance_km, r * (radius_tolerance_percent / 100.0))
        radius_mask = np.abs(distances - r) <= tol  # annulus
        for direction_index, dir_deg in enumerate(directions_deg):
            dir_diff = np.abs(directions - dir_deg)
            dir_diff = np.minimum(dir_diff, 360 - dir_diff)
            sector_mask = radius_mask & (dir_diff <= direction_tolerance_deg)
            masks[(j, direction_index)] = sector_mask
    return masks


@njit(cache=True, parallel=True)
def masked_nanmean_2d(arr, mask):
    """Mean of arr where mask is True"""
    total = 0.0
    count = 0
    rows, cols = arr.shape
    for r in prange(rows):
        for c in range(cols):
            if mask[r, c]:
                v = arr[r, c]
                if not np.isnan(v):
                    total += v
                    count += 1
    if count == 0:
        return np.nan
    return total / count


@njit(cache=True, parallel=True)
def masked_nanmean_wind_components(u, v, mask, dir_rad, component):
    """
    Compute radial (component=0) or tangential (component=1) wind mean
        Vr = u*sin(θ) + v*cos(θ)
        Vt = -u*cos(θ) + v*sin(θ)
    """
    sin_d = np.sin(dir_rad)
    cos_d = np.cos(dir_rad)
    total = 0.0
    count = 0
    rows, cols = u.shape
    for r in prange(rows):
        for c in range(cols):
            if mask[r, c]:
                ui = u[r, c]
                vi = v[r, c]
                if not (np.isnan(ui) or np.isnan(vi)):
                    if component == 0:
                        total += ui * sin_d + vi * cos_d  # Vr
                    else:
                        total += -ui * cos_d + vi * sin_d  # Vt
                    count += 1
    if count == 0:
        return np.nan
    return total / count


def compute_mixing_ratio(d2m_K, sp_Pa):
    """
    Compute water vapour mixing ratio (g/kg) from numpy arrays.
        d2m_K : dew point temperature in Kelvin
        sp_Pa : surface pressure in Pascals
    e_s = 6.112 * exp(17.67 * T_dp / (d2mc + 243.5))  [hPa]   d2m in degree C
    w   = 0.622 * e_s / (P - e_s)                     [kg/kg] converted to g/kg
    """
    d2mc = d2m_K - 273.15
    e_s = 6.112 * np.exp(17.67 * d2mc / (d2mc + 243.5))
    atmos_press = sp_Pa / 100.0
    denom = atmos_press - e_s
    denom = np.where(denom <= 0, np.nan, denom)
    w = 0.622 * (e_s / denom) * 1000
    return w


def compute_vorticitycartesian(u10, v10, lat_grid_2d, lat_vals, lon_vals):
    """
    Compute cartesian vorticity (zeta = dv/dx - du/dy) on ERA5 2D grid [s^-1].
    Uses central finite differences with dx/dy properly in metres.
    """
    dlon_deg = abs(lon_vals[1] - lon_vals[0])
    dlat_deg = abs(lat_vals[1] - lat_vals[0])

    # ERA5 latitudes descending (90->-90)
    lat_dir = 1.0 if lat_vals[1] > lat_vals[0] else -1.0
    dy_m = lat_dir * dlat_deg * 111000.0  # metres
    dx_2d = dlon_deg * 111000.0 * np.cos(np.radians(lat_grid_2d))  # (nlat, nlon)

    # dv/dx - central differences along longitude axis (axis=1)
    dvdx = np.full_like(v10, np.nan)
    dvdx[:, 1:-1] = (v10[:, 2:] - v10[:, :-2]) / (2.0 * dx_2d[:, 1:-1])
    dvdx[:, 0] = (v10[:, 1] - v10[:, 0]) / dx_2d[:, 0]
    dvdx[:, -1] = (v10[:, -1] - v10[:, -2]) / dx_2d[:, -1]

    # du/dy - central differences along latitude axis (axis=0)
    dudy = np.full_like(u10, np.nan)
    dudy[1:-1, :] = (u10[2:, :] - u10[:-2, :]) / (2.0 * dy_m)
    dudy[0, :] = (u10[1, :] - u10[0, :]) / dy_m
    dudy[-1, :] = (u10[-1, :] - u10[-2, :]) / dy_m
    return dvdx - dudy  # (nlat, nlon) [s^-1]


def compute_vorticity_polar(vt_avg, radii_km):
    """
    Compute polar azimuthal-mean vorticity (zeta = V_t/r  +  dV_t/dr)  [s^-1]
                                 vorticity with L'Hopital's rule at r=0

    For Northern Hemisphere cyclones, Vt is positive (counter-clockwise),
    The shear term dV_t/dr is positive inside, negative outside.
    """
    r_m = np.asarray(radii_km, dtype=float) * 1000.0
    N_steps, N_radii = vt_avg.shape

    # Initialization
    zeta = np.full_like(vt_avg, np.nan)

    # Handle r=0 with L'Hopital's rule
    if N_radii > 1 and r_m[0] == 0:
        dr = r_m[1] - r_m[0]
        if dr > 0:
            # Forward difference for gradient at r=0
            dvt_dr_0 = (vt_avg[:, 1] - vt_avg[:, 0]) / dr
            zeta[:, 0] = 2 * dvt_dr_0

    # Handle all r > 0
    r_pos_mask = r_m > 0
    if np.any(r_pos_mask):
        r_pos = r_m[r_pos_mask]
        vt_pos = vt_avg[:, r_pos_mask]
        # V_t/r term
        vt_over_r = vt_pos / r_pos[np.newaxis, :]
        # dV_t/dr term
        dvt_dr = np.gradient(vt_avg, r_m, axis=1, edge_order=2)
        dvt_dr_pos = dvt_dr[:, r_pos_mask]
        # zeta = V_t/r + dV_t/dr
        zeta[:, r_pos_mask] = vt_over_r + dvt_dr_pos
    return zeta


def compute_swvl1_tendency(swvl1_avg, track_times):
    """Compute temporal derivative of azimuthally-averaged SWVL1:
        (\\partial swvl1/ \\partial t)  [m³/m³/hr]

    With ERA5 hourly resolution, the central difference at index i is:
        (swvl1[i+1] - swvl1[i-1]) / (2 * 1h)
    Forward/backward differences are used only at the two endpoints.
    """
    times_pd = pd.to_datetime(track_times)
    dt_hours = np.diff(times_pd.asi8) / 1e9 / 3600.0  # nanoseconds -> hours
    dt_hours = dt_hours.copy().astype(float)
    bad_dt = dt_hours <= 0
    if np.any(bad_dt):
        std_print("WARN", "az", "Zero/negative dt found in track times", "replacing with median dt")
        median_dt = float(np.median(dt_hours[~bad_dt])) if np.any(~bad_dt) else 1.0
        dt_hours[bad_dt] = median_dt

    t_coords = np.concatenate([[0.0], np.cumsum(dt_hours)])  # shape (N_steps)
    # Check if uniform (all dt equal within 1%)
    if np.allclose(dt_hours, dt_hours[0], rtol=0.01):
        dswvl1_dt = np.gradient(swvl1_avg, float(dt_hours[0]), axis=0)
    else:
        # Non-uniform: passing full coordinate array
        dswvl1_dt = np.gradient(swvl1_avg, t_coords, axis=0)
    return dswvl1_dt  # [m³/m³/hr]


def get_grid_points_in_sector(
    lonc,
    latc,
    radius_km,
    direction_deg,
    lon_grid,
    lat_grid,
    max_radius_tolerance_km=10.0,  # Minimum 10km
    direction_tolerance_deg=11.25,  # \pm 11.25 deg (half of 22.5 deg sector)
):
    """
        radius_tolerance_km: Controls thickness of the annulus
        direction_tolerance_deg: Controls width of each directional sector

    Returns:
        sector_mask: Boolean mask of points in the sector
        distances: Distances from center (km)
        directions: Directions from center (degrees)
    """
    # Calculate distance and direction from center to ALL grid points
    dlat_km = (lat_grid - latc) * 111.0  # latitude to km
    dlon_km = (lon_grid - lonc) * 111.0 * np.cos(np.radians(latc))
    distances = np.sqrt(dlat_km**2 + dlon_km**2)  # Distance in km
    # Direction from center: 0°=North, 90°=East, 180°=South, 270°=West
    directions = np.degrees(np.arctan2(dlon_km, dlat_km))
    directions = (directions + 360) % 360  # Ensure 0-360 range
    # Points within radius tolerance
    radius_mask = np.abs(distances - radius_km) <= max_radius_tolerance_km
    # Points within direction tolerance
    dir_diff = np.abs(directions - direction_deg)
    dir_diff = np.minimum(dir_diff, 360 - dir_diff)  # Handle 0/360 wraparound
    direction_mask = dir_diff <= direction_tolerance_deg
    # Combine masks
    sector_mask = radius_mask & direction_mask
    return sector_mask, distances, directions


def compute_azimuthal_hovmoller_sectors(
    ds,
    track_df,
    radii_km,
    variables,
    n_theta=16,
    radius_tolerance_percent=10,  # 10% of radius
    min_radius_tolerance_km=10.0,  # Minimum 10km
    direction_tolerance_deg=11.25,  # \pm 11.25 deg (half of 22.5 deg sector)
    lsm_2d=None,
):
    """
    For each radius > 0:
      1. Divide circle into n_theta directional sectors
      2. For each sector, find all grid points within radius and direction tolerance
      3. Average values within each sector
      4. Store directional results (time, radius, direction)
    """
    track_times = track_df["time"].values
    N_steps = len(track_times)
    N_radii = len(radii_km)

    # Initialize results dictionary for directional data
    hov_results = {v: np.full((N_steps, N_radii, n_theta), np.nan) for v in variables}

    # Get the actual dimensions of the data
    lat_values = ds["latitude"].values
    lon_values = ds["longitude"].values
    # Pre-compute 2D grid of coordinates with proper indexing
    lon_grid_2d, lat_grid_2d = np.meshgrid(lon_values, lat_values, indexing="xy")

    # Crop LSM to match ERA5 regional grid (once, not per timestep).
    land_mask_global = None
    if lsm_2d is not None:
        try:
            import xarray as xr
            if hasattr(lsm_2d, "sel"):
                # ERA5 latitudes can be descending (90→-90); sel with method="nearest" handles both
                lsm_cropped = lsm_2d.sel(
                    latitude=lat_values,
                    longitude=lon_values,
                    method="nearest",
                )
                land_mask_global = lsm_cropped.values.squeeze() >= 0.5
            else:
                land_mask_global = np.asarray(lsm_2d, dtype=bool)
            std_print("INFO", "az", "LSM cropped to ERA5 grid", f"{land_mask_global.shape}")
        except Exception as _lsm_e:
            std_print("WARN", "az", "LSM crop failed — ocean masking disabled", str(_lsm_e))
            land_mask_global = None

    # Directions based on n_theta
    if n_theta == 16:
        directions_deg = np.array(
            [
              0,     # "N"
              22.5,  # "NNE"
              45,    # "NE"
              67.5,  # "ENE"
              90,    # "E" 
              112.5, # "ESE"
              135,   # "SE"
              157.5, # "SSE"
              180,   # "S"
              202.5, # "SSW"
              225,   # "SW"
              247.5, # "WSW"
              270,   # "W" 
              292.5, # "WNW"
              315,   # "NW"
              337.5, # "NNW"
            ]
        )
    else:
        directions_deg = np.linspace(0, 360, n_theta, endpoint=False)

    landfall_index = -1
    per_radius_landfall_index = np.full(N_radii, -1, dtype=int)
    for i, t in enumerate(track_times):
        row = track_df.iloc[i]
        lonc, latc = row["longitude"], row["latitude"]

        # Track landfall
        if "is_land" in track_df.columns:
            is_land_now = track_df.iloc[i]["is_land"]
            if landfall_index == -1 and is_land_now:
                landfall_index = i
        try:
            ds_t = ds.sel(time=t, method="nearest")
        except Exception as e:
            std_print("ERROR", "az", f"Failed to select time {t}", str(e))
            continue

        # std_print("DEBUG", "az", f"Processing time {pd.to_datetime(t).strftime('%d/%m %H:%M')} at ({latc:.2f}°, {lonc:.2f}°)")
        masks = precompute_sector_masks(lonc, latc, lon_grid_2d, lat_grid_2d, radii_km, directions_deg, radius_tolerance_percent, min_radius_tolerance_km, direction_tolerance_deg)
        for j, r in enumerate(radii_km):
            try:
                # Case: r = 0 km (center point only)
                if r == 0:
                    try:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            interpretcenter = ds_t.interp(longitude=lonc, latitude=latc, method="linear")
                    except Exception as e:
                        std_print("ERROR", "az", f"Interpolation failed at time {t}", str(e))
                        # Fallback to nearest neighbor
                        interpretcenter = ds_t.sel(longitude=lonc, latitude=latc, method="nearest")
                    for var in variables:
                        if var == "msl":
                            val = float(interpretcenter["msl"].values) / 100.0 # pa -> hPa
                        elif var == "tp":
                            if "tp" in interpretcenter:
                                val = float(interpretcenter["tp"].values) * 1000.0  # m/hr to mm/hr
                            else:
                                val = np.nan
                        elif var == "wind10":
                            u = float(interpretcenter["u10"].values)
                            v = float(interpretcenter["v10"].values)
                            val = np.sqrt(u**2 + v**2)
                        elif var == "speed_ms":
                            # Wind speed in m/s from u10, v10
                            if "u10" in interpretcenter and "v10" in interpretcenter:
                                u = float(interpretcenter["u10"].values)
                                v = float(interpretcenter["v10"].values)
                                val = np.sqrt(u**2 + v**2)
                            else:
                                val = np.nan
                        elif var == "vor_rel":
                            # Cartesian vorticity - bilinear interpret to center
                            if "u10" in ds_t and "v10" in ds_t:
                                vor2dc = compute_vorticitycartesian(
                                    ds_t["u10"].values,
                                    ds_t["v10"].values,
                                    lat_grid_2d,
                                    ds_t["latitude"].values,
                                    ds_t["longitude"].values,
                                )
                                vor_da = xr.DataArray(
                                    vor2dc,
                                    dims=["latitude", "longitude"],
                                    coords={
                                        "latitude": ds_t["latitude"],
                                        "longitude": ds_t["longitude"],
                                    },
                                )
                                val = float(vor_da.interp(longitude=lonc, latitude=latc, method="linear").values)
                            else:
                                val = np.nan
                        elif var == "vor_abs":
                            # Absolute vorticity = Cartesian relative + Coriolis f
                            if "u10" in ds_t and "v10" in ds_t:
                                vor2dc = compute_vorticitycartesian(
                                    ds_t["u10"].values,
                                    ds_t["v10"].values,
                                    lat_grid_2d,
                                    ds_t["latitude"].values,
                                    ds_t["longitude"].values,
                                )
                                vor_da = xr.DataArray(
                                    vor2dc,
                                    dims=["latitude", "longitude"],
                                    coords={
                                        "latitude": ds_t["latitude"],
                                        "longitude": ds_t["longitude"],
                                    },
                                )
                                vor_rel_val = float(vor_da.interp(longitude=lonc, latitude=latc, method="linear").values)
                                f = 2 * EARTH_OMEGA * np.sin(np.radians(latc))
                                val = vor_rel_val + f
                            else:
                                val = np.nan
                        elif var in ("vor_rel_polar", "vor_abs_polar"):
                            # Polar vorticity is computed after azimuthal averaging and r=0 to NaN
                            val = np.nan
                        elif var == "w":
                            # Mixing ratio: w = 0.622 * e_s / (P - e_s)
                            if "d2m" in interpretcenter and "sp" in interpretcenter:
                                d2mc = float(interpretcenter["d2m"].values)
                                spc = float(interpretcenter["sp"].values)
                                val = float(compute_mixing_ratio(np.array([d2mc]), np.array([spc]))[0])
                            else:
                                val = np.nan
                        elif var == "vr10":
                            # Radial wind at center is mathematically zero (singular point)
                            val = 0.0
                        elif var == "vt10":
                            # Tangential wind at center is mathematically zero (singular point)
                            val = 0.0
                        elif var == "total_hf":
                            # Total heat flux = Latent + Sensible
                            if (
                                "avg_slhtf" in interpretcenter
                                and "avg_ishf" in interpretcenter
                            ):
                                lhf = float(interpretcenter["avg_slhtf"].values)
                                shf = float(interpretcenter["avg_ishf"].values)
                                val = lhf + shf
                            else:
                                val = np.nan
                        else:
                            val = (float(interpretcenter[var].values) if var in interpretcenter else np.nan)
                        hov_results[var][i, j, :] = val

                    # Per-radius landfall detection for r=0:
                    if per_radius_landfall_index[j] == -1 and land_mask_global is not None:
                        dlat_km = (lat_grid_2d - latc) * 111.0
                        dlon_km = (lon_grid_2d - lonc) * 111.0 * np.cos(np.radians(latc))
                        distances = np.sqrt(dlat_km ** 2 + dlon_km ** 2)
                        centre_mask = distances <= max(min_radius_tolerance_km, 12.5)
                        if np.any(centre_mask & land_mask_global):
                            per_radius_landfall_index[j] = i
                    # std_print("DEBUG", "az", f"  {r:3d}km: Center point only")
                    continue

                # For r > 0: Calculate radius tolerance based on percentage or minimum
                valid_directions = []
                for direction_index, dir_deg in enumerate(directions_deg):
                    sector_mask = masks.get((j, direction_index))

                    # Guard: key missing from masks dict
                    if sector_mask is None:
                        continue

                    # Shape sanity check (without overwriting sector_mask)
                    u10_arr = ds_t["u10"].values if "u10" in ds_t else None
                    if u10_arr is not None and sector_mask.shape != u10_arr.shape:
                        std_print("WARN", "az", "Shape mismatch: mask vs data", f"{sector_mask.shape} vs {u10_arr.shape}")
                        continue

                    n_points = np.sum(sector_mask)
                    if n_points == 0:
                        continue

                    # FIX: Cannot access attribute cache_t for class FunctionType
                    # FIX: Cannot assign attribute _lsm_warned for class FunctionType
                    if (
                        not hasattr(compute_azimuthal_hovmoller_sectors, "cache_t")
                        or compute_azimuthal_hovmoller_sectors.cache_t != t
                    ):
                        c = {}
                        for v in [
                            "u10",
                            "v10",
                            "msl",
                            "sp",
                            "tp",
                            "t2m",
                            "d2m",
                            "stl1",
                            "swvl1",
                            "avg_slhtf",
                            "avg_ishf",
                        ]:
                            if v in ds_t:
                                c[v] = ds_t[v].values

                        if "u10" in c and "v10" in c:
                            c["vor2dcart"] = compute_vorticitycartesian(
                                c["u10"],
                                c["v10"],
                                lat_grid_2d,
                                ds_t["latitude"].values,
                                ds_t["longitude"].values,
                            )
                            c["wind10"] = np.sqrt(c["u10"] ** 2 + c["v10"] ** 2)
                        if "d2m" in c and "sp" in c:
                            c["w"] = compute_mixing_ratio(c["d2m"], c["sp"])
                        if "avg_slhtf" in c and "avg_ishf" in c:
                            c["total_hf"] = c["avg_slhtf"] + c["avg_ishf"]
                        # Land-sea mask: use pre-cropped array
                        if land_mask_global is not None:
                            c["land_mask_2d"] = land_mask_global
                        elif "lsm" in ds_t:
                            c["land_mask_2d"] = ds_t["lsm"].values.squeeze() >= 0.5
                        else:
                            if not hasattr(compute_azimuthal_hovmoller_sectors, "_lsm_warned"):
                                std_print("WARN", "az", "No lsm provided — SWVL1/STL1 will include ocean fill values")
                                compute_azimuthal_hovmoller_sectors._lsm_warned = True
                            c["land_mask_2d"] = None

                        compute_azimuthal_hovmoller_sectors.cache_t = t
                        compute_azimuthal_hovmoller_sectors.cache = c
                    c = compute_azimuthal_hovmoller_sectors.cache

                    # Per-radius landfall detection
                    if per_radius_landfall_index[j] == -1 and c.get("land_mask_2d") is not None:
                        # Build a full annulus mask: union of all sector masks at current radius
                        dlat_km = (lat_grid_2d - latc) * 111.0
                        dlon_km = (lon_grid_2d - lonc) * 111.0 * np.cos(np.radians(latc))
                        distances = np.sqrt(dlat_km ** 2 + dlon_km ** 2)
                        radius_tol = max(min_radius_tolerance_km, radii_km[j] * (radius_tolerance_percent / 100.0))
                        if radii_km[j] == 0:
                            # r=0: check if centre point itself is land
                            annulus_mask = distances <= radius_tol
                        else:
                            annulus_mask = np.abs(distances - radii_km[j]) <= radius_tol
                        if np.any(annulus_mask & c["land_mask_2d"]):
                            per_radius_landfall_index[j] = i

                    fcor = 2 * EARTH_OMEGA * np.sin(np.radians(latc))
                    dir_rad = np.radians(dir_deg)

                    # Process each variable using pre-cached arrays
                    for var in variables:
                        arr = c.get(var)
                        if arr is not None:
                            # Raw ERA5 variable - unit conversions where needed
                            # sv = arr[sector_mask]
                            # sv = sv[~np.isnan(sv)]
                            # if len(sv) > 0:
                            #   avg_val = float(np.nanmean(sv))
                            val = masked_nanmean_2d(arr, sector_mask)
                            if not np.isnan(val):
                                if var == "msl":
                                    val /= 100.0  # Pa -> hPa
                                elif var == "tp":
                                    val *= 1000.0  # m/hr -> mm/hr
                                hov_results[var][i, j, direction_index] = val
                        elif var in ("wind10", "speed_ms") and "wind10" in c:
                            sv = c["wind10"][sector_mask]
                            sv = sv[~np.isnan(sv)]
                            if len(sv) > 0:
                                hov_results[var][i, j, direction_index] = float(np.nanmean(sv))
                        elif var == "vor_rel" and "vor2dcart" in c:
                            # Cartesian
                            sv = c["vor2dcart"][sector_mask]
                            sv = sv[~np.isnan(sv)]
                            if len(sv) > 0:
                                hov_results[var][i, j, direction_index] = float(np.nanmean(sv))
                        elif var == "vor_abs" and "vor2dcart" in c:
                            sv = c["vor2dcart"][sector_mask]
                            sv = sv[~np.isnan(sv)]
                            if len(sv) > 0:
                                hov_results[var][i, j, direction_index] = (float(np.nanmean(sv)) + fcor)
                        elif var in ("vor_rel_polar", "vor_abs_polar"):
                            # Polar vorticity derived AFTER azimuthal averaging
                            pass
                        elif var == "w" and "w" in c:
                            sv = c["w"][sector_mask]
                            sv = sv[~np.isnan(sv)]
                            if len(sv) > 0:
                                hov_results[var][i, j, direction_index] = float(np.nanmean(sv))
                        elif var == "vr10" and "u10" in c and "v10" in c:
                            u_s = c["u10"][sector_mask]
                            v_s = c["v10"][sector_mask]
                            ok = ~(np.isnan(u_s) | np.isnan(v_s))
                            if np.sum(ok) > 0:
                                # vr = u_s[ok] * np.sin(dir_rad) + v_s[ok] * np.cos(dir_rad)
                                vr = masked_nanmean_wind_components(c["u10"], c["v10"], sector_mask, dir_rad, 0)
                                hov_results[var][i, j, direction_index] = float(np.nanmean(vr))
                        elif var == "vt10" and "u10" in c and "v10" in c:
                            u_s = c["u10"][sector_mask]
                            v_s = c["v10"][sector_mask]
                            ok = ~(np.isnan(u_s) | np.isnan(v_s))
                            if np.sum(ok) > 0:
                                # vt = -u_s[ok] * np.cos(dir_rad) + v_s[ok] * np.sin(dir_rad)
                                vt = masked_nanmean_wind_components(c["u10"], c["v10"], sector_mask, dir_rad, 1)
                                hov_results[var][i, j, direction_index] = float(np.nanmean(vt))
                        elif var == "total_hf" and "total_hf" in c:
                            sv = c["total_hf"][sector_mask]
                            sv = sv[~np.isnan(sv)]
                            if len(sv) > 0:
                                hov_results[var][i, j, direction_index] = float(np.nanmean(sv))
                if len(valid_directions) > 0:
                    dir_summary = ", ".join(valid_directions)
                    dir_summary = dir_summary[:-1]
                    # std_print("DEBUG", "az", f"  {r:3d}km: {dir_summary} ({len(valid_directions)}/{n_theta})")
            except Exception as e:
                std_print("ERROR", "az", f"Failed at time {t}, radius {r}km", str(e))
                traceback.print_exc()
                continue
    return hov_results, track_times, landfall_index, per_radius_landfall_index


def average_azimuthally(hov_results, variables, radii_km=None):
    """
    Azimuthally average the (N_steps, N_radii, N_theta) hov_results.
    If "vt10" is available and radii_km is provided, also computes:
      - "vor_rel_polar"  = <Vt>/r + d<Vt>/dr
      - "vor_abs_polar"  = vor_rel_polar + f   (f averaged over all timesteps)
    """
    averaged = {}
    for var in variables:
        if var in hov_results:
            with np.errstate(invalid="ignore"):
                averaged[var] = np.nanmean(hov_results[var], axis=2)

    # Polar vorticity - derived post-averaging from the V_t radial profile
    need_polar = any(v in variables for v in ("vor_rel_polar", "vor_abs_polar"))
    if need_polar and "vt10" in averaged and radii_km is not None:
        vt_avg = averaged["vt10"]  # (N_steps, N_radii)
        r_arr = np.asarray(radii_km, dtype=float)
        vor_p = compute_vorticity_polar(vt_avg, r_arr)  # (N_steps, N_radii)
        if "vor_rel_polar" in variables:
            averaged["vor_rel_polar"] = vor_p
        if "vor_abs_polar" in variables:
            averaged["vor_abs_polar"] = vor_p  # f added at time of plotting
    return averaged


def plot_hovmoller_azimuthal_average(
    hov_arr,
    track_times,
    radii_km,
    var_name,
    track_df,
    track_id,
    output_dir="hov_plots",
    landfall_index=-1,
    per_radius_landfall_index=None,
    levels=36
):
    track_id = int(track_id)
    plot_dir = os.path.join(output_dir, f"track_{track_id}", "hovmoller")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    var_label = VARIABLE_NAMES.get(var_name, var_name)
    var_unit = VARIABLE_UNITS.get(var_name, "")

    # Colormap based on variable
    if var_name in ["tp", "wind10", "avg_slhtf", "avg_ishf", "total_hf"]:
        cmap = "plasma"
    elif var_name in ["t2m", "stl1"]:
        cmap = "coolwarm"
    elif var_name in ["msl", "sp"]:
        cmap = "viridis_r"  # Reversed for pressure (lower pressure = darker)
    elif var_name in ["swvl1", "speed_ms"]:
        cmap = "YlGnBu"
    elif var_name == "w":
        cmap = "Blues"
    elif var_name in ["vor_rel", "vor_abs", "vor_rel_polar", "vor_abs_polar"]:
        hov_arr = hov_arr * 1e5 # Scale vorticity to x10^-5
        cmap = "RdBu_r"  # diverging: cyclonic +ve, anticyclonic -ve
    elif var_name == "dswvl1_dt":
        cmap = "RdBu_r"
    else:
        cmap = "viridis"

    # Colour limits: diverging colormaps need symmetric (-vmax, +ymax) centred at 0
    DIVERGING_VARS = {"vor_rel", "vor_abs", "vor_rel_polar", "vor_abs_polar", "dswvl1_dt", "vr10"}
    valid_vals = hov_arr[~np.isnan(hov_arr)]
    nan_frac = 1.0 - len(valid_vals) / hov_arr.size
    if nan_frac > 0.0:
        std_print("WARN", "az", f"{var_name} has {nan_frac*100:.1f}% NaN values")
    if len(valid_vals) == 0:
        std_print("ERROR", "az", f"All-NaN for {var_name}", "skipping plot")
        return None

    if var_name in DIVERGING_VARS:
        vmax = float(np.nanpercentile(np.abs(hov_arr), 97))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = float(np.nanmax(np.abs(valid_vals))) if len(valid_vals) > 0 else 1e-6
        vmin = -vmax
    else:
        vmin = float(np.nanpercentile(hov_arr, 2))
        vmax = float(np.nanpercentile(hov_arr, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(np.nanmin(valid_vals)), float(np.nanmax(valid_vals))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = -1e-6, 1e-6
    elif vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    fig, ax = plt.subplots(figsize=(10, 8))
    N_steps = len(track_times)
    X, Y = np.meshgrid(radii_km, np.arange(N_steps))

    im = ax.contourf(X, Y, hov_arr, levels=np.linspace(vmin, vmax, levels), cmap=cmap, extend="both")
    auto_levels = np.linspace(vmin, vmax, int(levels / 3))
    auto_levels = auto_levels[np.abs(auto_levels) > 1e-9] # Remove zero
    contours = ax.contour(X, Y, hov_arr, levels=auto_levels, colors="black", linewidths=0.5, alpha=0.45)
    ax.clabel(contours, inline=True, fontsize=7, fmt="%1.2f")

    cbar = plt.colorbar(im, pad=0.02, ax=ax)
    cbar.set_label(f"{var_label} ({var_unit})", fontsize=12)

    ax.set_xlabel("Radius from center (km)", fontsize=11)
    ax.set_ylabel("Time", fontsize=11)
    ax.set_title(
        f"Track {track_id}: {var_label}",
        fontsize=13,
        fontweight="bold",
    )

    # Format y-axis with time labels
    if N_steps <= 20:
        ytick_positions = np.arange(N_steps)
        ytick_labels = [
            pd.to_datetime(t).strftime("%d/%m/%Y\n%H:%M:%S")
            for t in track_times
        ]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, rotation=45, fontsize=8)
    else:
        step = max(1, N_steps // 10)
        ytick_positions = np.arange(0, N_steps, step)
        ytick_labels = [
            pd.to_datetime(track_times[i]).strftime("%d/%m/%Y\n%H:%M:%S")
            for i in range(0, N_steps, step)
        ]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=8)

    # Format x-axis with radius labels
    ax.set_xticks(radii_km)
    ax.set_xticklabels([f"{int(r)}" for r in radii_km], fontsize=9, rotation=45, ha="right")

    # # Landfall boundary line
    # has_diagonal = (per_radius_landfall_index is not None and np.any(per_radius_landfall_index >= 0))
    # if has_diagonal:
    #     lf_x = [radii_km[j] for j in range(len(radii_km)) if per_radius_landfall_index[j] >= 0]
    #     lf_y = [per_radius_landfall_index[j] for j in range(len(radii_km)) if per_radius_landfall_index[j] >= 0]
    #     if landfall_index != -1 and landfall_index < N_steps:
    #         landfall_time_str = pd.to_datetime(track_times[landfall_index]).strftime("%d/%m/%Y %H:%M")
    #         lf_label = f"Landfall (centre): {landfall_time_str}"
    #     else:
    #         lf_label = None
    #     ax.plot(lf_x, lf_y, color="black", linestyle="--", linewidth=2, label=lf_label, alpha=0.9, zorder=5)
    #     # Small dot at r=0 (Landfall centre point)
    #     if landfall_index != -1 and landfall_index < N_steps:
    #         ax.scatter([radii_km[0]], [landfall_index], color="black", s=50, zorder=6, clip_on=False)
    # elif landfall_index != -1 and landfall_index < N_steps:
    #     # Fallback: horizontal line when no lsm available
    #     landfall_time_str = pd.to_datetime(track_times[landfall_index]).strftime("%d/%m/%Y %H:%M")
    #     ax.axhline(y=landfall_index, color="black", linestyle="--", linewidth=2,label=f"Landfall (centre): {landfall_time_str}", alpha=0.9)
    #
    # legend_handles, labels = ax.get_legend_handles_labels()
    # if legend_handles:
    #     ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.83, 1.05), **legend_style())

    # # Landfall boundary line
    has_diagonal = (per_radius_landfall_index is not None and np.any(per_radius_landfall_index >= 0))
    if has_diagonal:
        lf_x = [radii_km[j] for j in range(len(radii_km)) if per_radius_landfall_index[j] >= 0]
        lf_y = [per_radius_landfall_index[j] for j in range(len(radii_km)) if per_radius_landfall_index[j] >= 0]

        # for r = 0 to 25km
        if len(lf_x) > 0 and lf_x[0] != radii_km[0] and landfall_index != -1 and landfall_index < N_steps:
            lf_x = [radii_km[0]] + lf_x
            lf_y = [landfall_index] + lf_y

        if landfall_index != -1 and landfall_index < N_steps:
            landfall_time_str = pd.to_datetime(track_times[landfall_index]).strftime("%d/%m/%Y %H:%M")
            lf_label = f"Landfall: {landfall_time_str}"
        else:
            lf_label = None
        ax.plot(lf_x, lf_y, color="black", linestyle="--", linewidth=2, label=lf_label, alpha=0.9, zorder=5)

        legend_handles, labels = ax.get_legend_handles_labels()
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.83, 1.05), **legend_style())

    # Save plot
    plot_path = os.path.join(plot_dir, f"hov_{var_name}_azimuthal_avg.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "az", f"Hovmöller plot for {var_name}", plot_path)
