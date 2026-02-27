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
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from utils import std_print
import traceback
import warnings

from utils import common_bbox, legend_style

warnings.filterwarnings("ignore", message="Mean of empty slice")

LEVELS = 36
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
}


# Helper functions
def compute_mixing_ratio(d2m_K, sp_Pa):
    """
    Compute water vapour mixing ratio (g/kg) from numpy arrays.
    Args:
        d2m_K : dew point temperature in Kelvin
        sp_Pa : surface pressure in Pascals
    e_s = 6.112 * exp(17.67 * T_dp / (d2m_C + 243.5))  [hPa], d2m in degree C
    w   = 0.622 * e_s / (P - e_s)                      [kg/kg] converted to g/kg
    """
    d2m_C = d2m_K - 273.15
    e_s = 6.112 * np.exp(17.67 * d2m_C / (d2m_C + 243.5))
    atmos_press = sp_Pa / 100.0
    denom = atmos_press - e_s
    denom = np.where(denom <= 0, np.nan, denom)
    w = 0.622 * (e_s / denom) * 1000
    return w


def compute_vorticity_cartesian(u10, v10, lat_grid_2d, lat_vals, lon_vals):
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

    # Initialize with NaNs
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


def get_grid_points_in_sector(
    lon_c,
    lat_c,
    radius_km,
    direction_deg,
    lon_grid,
    lat_grid,
    radius_tolerance_km=10.0,
    direction_tolerance_deg=11.25,
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
    dlat_km = (lat_grid - lat_c) * 111.0  # latitude to km
    dlon_km = (lon_grid - lon_c) * 111.0 * np.cos(np.radians(lat_c))
    distances = np.sqrt(dlat_km**2 + dlon_km**2)  # Distance in km

    # Direction from center: 0°=North, 90°=East, 180°=South, 270°=West
    directions = np.degrees(np.arctan2(dlon_km, dlat_km))
    directions = (directions + 360) % 360  # Ensure 0-360 range

    # Points within radius tolerance
    radius_mask = np.abs(distances - radius_km) <= radius_tolerance_km
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

    # Directions based on n_theta
    if n_theta == 16:
        directions_deg = np.array(
            [
                0,
                22.5,
                45,
                67.5,
                90,
                112.5,
                135,
                157.5,
                180,
                202.5,
                225,
                247.5,
                270,
                292.5,
                315,
                337.5,
            ]
        )
        direction_names = [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ]
    else:
        directions_deg = np.linspace(0, 360, n_theta, endpoint=False)
        direction_names = [f"{d:.1f}°" for d in directions_deg]

    landfall_index = -1
    for i, t in enumerate(track_times):
        row = track_df.iloc[i]
        lon_c, lat_c = row["longitude"], row["latitude"]

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

        # std_print(
        #     "DEBUG",
        #     "az",
        #     f"Processing time {pd.to_datetime(t).strftime('%d/%m %H:%M')} at ({lat_c:.2f}°, {lon_c:.2f}°)",
        # )
        for j, r in enumerate(radii_km):
            try:
                # Case: r = 0 km (center point only)
                if r == 0:
                    try:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            interp_center = ds_t.interp(
                                longitude=lon_c, latitude=lat_c, method="linear"
                            )
                    except Exception as e:
                        std_print(
                            "WARN", "az", f"Interpolation failed at time {t}", str(e)
                        )
                        # Fallback to nearest neighbor
                        interp_center = ds_t.sel(
                            longitude=lon_c, latitude=lat_c, method="nearest"
                        )
                    for var in variables:
                        if var == "msl":
                            val = float(interp_center["msl"].values) / 100.0
                        elif var == "tp":
                            if "tp" in interp_center:
                                val = (
                                    float(interp_center["tp"].values) * 1000.0
                                )  # m/hr to mm/hr
                            else:
                                val = np.nan
                        elif var == "wind10":
                            u = float(interp_center["u10"].values)
                            v = float(interp_center["v10"].values)
                            val = np.sqrt(u**2 + v**2)
                        elif var == "speed_ms":
                            # Wind speed in m/s from u10, v10
                            if "u10" in interp_center and "v10" in interp_center:
                                u = float(interp_center["u10"].values)
                                v = float(interp_center["v10"].values)
                                val = np.sqrt(u**2 + v**2)
                            else:
                                val = np.nan
                        elif var == "vor_rel":
                            # Cartesian vorticity — bilinear interp to center
                            if "u10" in ds_t and "v10" in ds_t:
                                vor2d_c = compute_vorticity_cartesian(
                                    ds_t["u10"].values,
                                    ds_t["v10"].values,
                                    lat_grid_2d,
                                    ds_t["latitude"].values,
                                    ds_t["longitude"].values,
                                )
                                vor_da = xr.DataArray(
                                    vor2d_c,
                                    dims=["latitude", "longitude"],
                                    coords={
                                        "latitude": ds_t["latitude"],
                                        "longitude": ds_t["longitude"],
                                    },
                                )
                                val = float(
                                    vor_da.interp(
                                        longitude=lon_c, latitude=lat_c, method="linear"
                                    ).values
                                )
                            else:
                                val = np.nan
                        elif var == "vor_abs":
                            # Absolute vorticity = Cartesian relative + Coriolis f
                            if "u10" in ds_t and "v10" in ds_t:
                                vor2d_c = compute_vorticity_cartesian(
                                    ds_t["u10"].values,
                                    ds_t["v10"].values,
                                    lat_grid_2d,
                                    ds_t["latitude"].values,
                                    ds_t["longitude"].values,
                                )
                                vor_da = xr.DataArray(
                                    vor2d_c,
                                    dims=["latitude", "longitude"],
                                    coords={
                                        "latitude": ds_t["latitude"],
                                        "longitude": ds_t["longitude"],
                                    },
                                )
                                vor_rel_val = float(
                                    vor_da.interp(
                                        longitude=lon_c, latitude=lat_c, method="linear"
                                    ).values
                                )
                                f = 2 * EARTH_OMEGA * np.sin(np.radians(lat_c))
                                val = vor_rel_val + f
                            else:
                                val = np.nan
                        elif var in ("vor_rel_polar", "vor_abs_polar"):
                            # Polar vorticity is computed AFTER azimuthal averaging and r=0 to NaN
                            val = np.nan
                        elif var == "w":
                            # Formula: w = 0.622 * e_s / (P - e_s)
                            if "d2m" in interp_center and "sp" in interp_center:
                                d2m_c = float(interp_center["d2m"].values)
                                sp_c = float(interp_center["sp"].values)
                                val = float(
                                    compute_mixing_ratio(
                                        np.array([d2m_c]), np.array([sp_c])
                                    )[0]
                                )
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
                                "avg_slhtf" in interp_center
                                and "avg_ishf" in interp_center
                            ):
                                lhf = float(interp_center["avg_slhtf"].values)
                                shf = float(interp_center["avg_ishf"].values)
                                val = lhf + shf
                            else:
                                val = np.nan
                        else:
                            # Direct variables from dataset
                            val = (
                                float(interp_center[var].values)
                                if var in interp_center
                                else np.nan
                            )
                        hov_results[var][i, j, :] = val
                    # std_print("DEBUG", "az", f"  {r:3d}km: Center point only")
                    continue

                # For r > 0: Calculate radius tolerance based on percentage or minimum
                radius_tolerance = max(
                    min_radius_tolerance_km, r * (radius_tolerance_percent / 100.0)
                )

                valid_directions = []
                for d_idx, (dir_deg, dir_name) in enumerate(
                    zip(directions_deg, direction_names)
                ):
                    sector_mask, _, _ = get_grid_points_in_sector(
                        lon_c,
                        lat_c,
                        r,
                        dir_deg,
                        lon_grid_2d,
                        lat_grid_2d,
                        radius_tolerance_km=radius_tolerance,
                        direction_tolerance_deg=direction_tolerance_deg,
                    )

                    # Verify mask shape against pre-cached u10 shape
                    u10_arr = ds_t["u10"].values if "u10" in ds_t else None
                    if u10_arr is not None and sector_mask.shape != u10_arr.shape:
                        std_print(
                            "WARN",
                            "az",
                            f"Shape mismatch: mask {sector_mask.shape} vs data {u10_arr.shape}",
                        )
                        continue

                    n_points = np.sum(sector_mask)
                    if n_points == 0:
                        continue

                    valid_directions.append(dir_name)

                    if (
                        not hasattr(compute_azimuthal_hovmoller_sectors, "_cache_t")
                        or compute_azimuthal_hovmoller_sectors._cache_t != t
                    ):
                        _c = {}
                        for _v in [
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
                            if _v in ds_t:
                                _c[_v] = ds_t[_v].values
                        if "u10" in _c and "v10" in _c:
                            _c["vor2d_cart"] = compute_vorticity_cartesian(
                                _c["u10"],
                                _c["v10"],
                                lat_grid_2d,
                                ds_t["latitude"].values,
                                ds_t["longitude"].values,
                            )
                            _c["wind10"] = np.sqrt(_c["u10"] ** 2 + _c["v10"] ** 2)
                        if "d2m" in _c and "sp" in _c:
                            _c["w"] = compute_mixing_ratio(_c["d2m"], _c["sp"])
                        if "avg_slhtf" in _c and "avg_ishf" in _c:
                            _c["total_hf"] = _c["avg_slhtf"] + _c["avg_ishf"]
                        compute_azimuthal_hovmoller_sectors._cache_t = t
                        compute_azimuthal_hovmoller_sectors._cache = _c
                    _c = compute_azimuthal_hovmoller_sectors._cache

                    f_cor = 2 * EARTH_OMEGA * np.sin(np.radians(lat_c))
                    dir_rad = np.radians(dir_deg)

                    # Process each variable using pre-cached arrays
                    for var in variables:
                        arr = _c.get(var)
                        if arr is not None:
                            # Raw ERA5 variable — unit conversions where needed
                            sv = arr[sector_mask]
                            sv = sv[~np.isnan(sv)]
                            if len(sv) > 0:
                                avg_val = float(np.nanmean(sv))
                                if var == "msl":
                                    avg_val /= 100.0  # Pa -> hPa
                                elif var == "tp":
                                    avg_val *= 1000.0  # m/hr -> mm/hr
                                hov_results[var][i, j, d_idx] = avg_val
                        elif var in ("wind10", "speed_ms") and "wind10" in _c:
                            sv = _c["wind10"][sector_mask]
                            sv = sv[~np.isnan(sv)]
                            if len(sv) > 0:
                                hov_results[var][i, j, d_idx] = float(np.nanmean(sv))
                        elif var == "vor_rel" and "vor2d_cart" in _c:
                            # Cartesian: sample pre-computed 2D field
                            sv = _c["vor2d_cart"][sector_mask]
                            sv = sv[~np.isnan(sv)]
                            if len(sv) > 0:
                                hov_results[var][i, j, d_idx] = float(np.nanmean(sv))
                        elif var == "vor_abs" and "vor2d_cart" in _c:
                            sv = _c["vor2d_cart"][sector_mask]
                            sv = sv[~np.isnan(sv)]
                            if len(sv) > 0:
                                hov_results[var][i, j, d_idx] = (
                                    float(np.nanmean(sv)) + f_cor
                                )
                        elif var in ("vor_rel_polar", "vor_abs_polar"):
                            # Polar vorticity derived AFTER azimuthal averaging — skip here
                            pass
                        elif var == "w" and "w" in _c:
                            sv = _c["w"][sector_mask]
                            sv = sv[~np.isnan(sv)]
                            if len(sv) > 0:
                                hov_results[var][i, j, d_idx] = float(np.nanmean(sv))
                        elif var == "vr10" and "u10" in _c and "v10" in _c:
                            u_s = _c["u10"][sector_mask]
                            v_s = _c["v10"][sector_mask]
                            ok = ~(np.isnan(u_s) | np.isnan(v_s))
                            if np.sum(ok) > 0:
                                vr = u_s[ok] * np.sin(dir_rad) + v_s[ok] * np.cos(
                                    dir_rad
                                )
                                hov_results[var][i, j, d_idx] = float(np.nanmean(vr))
                        elif var == "vt10" and "u10" in _c and "v10" in _c:
                            u_s = _c["u10"][sector_mask]
                            v_s = _c["v10"][sector_mask]
                            ok = ~(np.isnan(u_s) | np.isnan(v_s))
                            if np.sum(ok) > 0:
                                vt = -u_s[ok] * np.cos(dir_rad) + v_s[ok] * np.sin(
                                    dir_rad
                                )
                                hov_results[var][i, j, d_idx] = float(np.nanmean(vt))
                        elif var == "total_hf" and "total_hf" in _c:
                            sv = _c["total_hf"][sector_mask]
                            sv = sv[~np.isnan(sv)]
                            if len(sv) > 0:
                                hov_results[var][i, j, d_idx] = float(np.nanmean(sv))
                if len(valid_directions) > 0:
                    dir_summary = ", ".join(valid_directions)
                    dir_summary = dir_summary[:-1]
                    # std_print("DEBUG", "az", f"  {r:3d}km: {dir_summary} ({len(valid_directions)}/{n_theta})",)
            except Exception as e:
                std_print("ERROR", "az", f"Failed at time {t}, radius {r}km", str(e))
                traceback.print_exc()
                continue
    return hov_results, track_times, landfall_index


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
):
    plot_dir = os.path.join(output_dir, f"track_{int(track_id)}", "hovmoller")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    var_label = VARIABLE_NAMES.get(var_name, var_name)
    var_unit = VARIABLE_UNITS.get(var_name, "")

    # Scale vorticity to x10^-5
    VOR_VARS = ("vor_rel", "vor_abs", "vor_rel_polar", "vor_abs_polar")
    if var_name in VOR_VARS:
        hov_arr = hov_arr * 1e5

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
    elif var_name in VOR_VARS:
        cmap = "RdBu_r"  # diverging: cyclonic +ve, anticyclonic -ve
    else:
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(10, 8))
    N_steps = len(track_times)
    X, Y = np.meshgrid(radii_km, np.arange(N_steps))

    im = ax.contourf(X, Y, hov_arr, levels=LEVELS, cmap=cmap, extend="both")

    # Contour lines
    contours = ax.contour(
        X, Y, hov_arr, levels=int(LEVELS / 3), colors="black", linewidths=0.5, alpha=0.5
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt="%1.2f")

    cbar = plt.colorbar(im, pad=0.02, ax=ax)
    cbar.set_label(f"{var_label} ({var_unit})", fontsize=12)

    ax.set_xlabel("Radius from center (km)", fontsize=11)
    ax.set_ylabel("Time", fontsize=11)
    ax.set_title(
        f"Track {int(track_id)}: {var_label}",
        fontsize=13,
        fontweight="bold",
    )

    # Format y-axis with time labels
    if N_steps <= 20:
        ytick_positions = np.arange(N_steps)
        ytick_labels = [
            pd.to_datetime(t).strftime("%d/%m/%Y\n%H:%M") for t in track_times
        ]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, rotation=45, fontsize=8)
    else:
        step = max(1, N_steps // 10)
        ytick_positions = np.arange(0, N_steps, step)
        ytick_labels = [
            pd.to_datetime(track_times[i]).strftime("%d/%m/%Y\n%H:%M")
            for i in range(0, N_steps, step)
        ]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=8)

    # Format x-axis with radius labels
    ax.set_xticks(radii_km)
    ax.set_xticklabels(
        [f"{int(r)}" for r in radii_km], fontsize=9, rotation=45, ha="right"
    )
    # if len(radii_km) > 12:
    #     ax.set_xticklabels(
    #         [f"{int(r)}" for r in radii_km], fontsize=9, rotation=45, ha="right"
    #     )
    # else:
    #   ax.set_xticklabels([f"{int(r)}" for r in radii_km], fontsize=10)

    # Landfall mark
    if landfall_index != -1 and landfall_index < N_steps:
        ax.axhline(
            y=landfall_index,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Landfall",
            alpha=0.9,
        )
    bbox_style = common_bbox()
    # Landfall annotation at top right
    if landfall_index != -1 and landfall_index < N_steps:
        landfall_time_str = pd.to_datetime(track_times[landfall_index]).strftime(
            "%d/%m %H:%M"
        )
        landfall_text = f"Landfall: {landfall_time_str}"
        ax.text(
            0.77,
            1.011,
            landfall_text,
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox=bbox_style,
        )
    # Legend if landfall line is plotted
    if landfall_index != -1 and landfall_index < N_steps:
        legend_handles, labels = ax.get_legend_handles_labels()
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right", **legend_style())

    # Save plot
    plot_path = os.path.join(plot_dir, f"hov_{var_name}_azimuthal_avg.png")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "az", f"Hovmöller plot for {var_name}", plot_path)
