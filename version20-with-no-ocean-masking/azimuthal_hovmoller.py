# INFO: Method:
# 1. Get MD center. For r = 0 (distance from MD center), linearly interpolating
# to get the grid point value at the exact center location, take that as it is
# with no averaging (since it is a single interpolated point), and plot it.
# 2. Take an annulus (disk) of radius 25km with tolerance of +/-10km, so all grid
# points from 15-35km distance are included. Within this annulus, divided the area
# into 16 equal sectors of 22.5 degree each (N, NNE, NE, ENE, E, ESE, SE, SSE, S,
# SSW, SW, WSW, W, WNW, NW, NNW). For each directional sector, find all grid
# points that fall in that sector-annulus segment (the trapezium-like region)
# and average those points to get one value per sector. Then average these 16
# sector values together to get a single azimuthally-averaged value for the 25km
# radius, and then plot it.
# 3. Now take an annulus of radius 50km with tolerance of +/-10km (all points
# from 40-60km) and repeat the same sectoring and averaging process. Continuing
# this for all radii: [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300,
# 325, 350, 375, 400] km

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
def interp_center_np(arr2d):
    """Bilinear interp of 2D cached array to (lon_c, lat_c)."""
    lat_vals = ds_t["latitude"].values
    lon_vals = ds_t["longitude"].values
    # nearest indices
    i_lat = np.argmin(np.abs(lat_vals - lat_c))
    i_lon = np.argmin(np.abs(lon_vals - lon_c))
    # clamp to valid range
    i_lat = np.clip(i_lat, 0, arr2d.shape[0] - 1)
    i_lon = np.clip(i_lon, 0, arr2d.shape[1] - 1)
    return float(arr2d[i_lat, i_lon])


def compute_mixing_ratio(d2m_K, sp_Pa):
    """
    Compute water vapour mixing ratio (g/kg) from numpy arrays.
    Args:
        d2m_K : dew point temperature in Kelvin  (numpy array, ERA5 d2m)
        sp_Pa : surface pressure in Pascals       (numpy array, ERA5 sp)
    Returns:
        mixing ratio in g/kg (numpy array, same shape)

    e_s = 6.112 * exp(17.67 * T_dp / (T_dp + 243.5))   [hPa], T_dp in degree C
    w   = 0.622 * e_s / (P - e_s)                       [kg/kg] converted to g/kg
    """
    d2m_C = d2m_K - 273.15  # K -> deg C
    e_s = 6.112 * np.exp(17.67 * d2m_C / (d2m_C + 243.5))  # hPa
    atmos_press = sp_Pa / 100.0  # Pa -> hPa
    denom = atmos_press - e_s
    denom = np.where(denom <= 0, np.nan, denom)
    w = 0.622 * (e_s / denom) * 1000  # g/kg
    return w


def compute_vorticity_2d(ds_t, lat_grid_2d, lon_grid_2d):
    """
    Compute relative vorticity (zeta = dv/dx - du/dy) on ERA5 2D grid [s^-1].
    Uses central finite differences with dx/dy properly in metres.
    """
    u2d = ds_t["u10"].values  # (nlat, nlon)
    v2d = ds_t["v10"].values

    lat_vals = ds_t["latitude"].values  # 1D
    lon_vals = ds_t["longitude"].values  # 1D
    dlon_deg = abs(lon_vals[1] - lon_vals[0])
    dlat_deg = abs(lat_vals[1] - lat_vals[0])

    lat_dir = 1.0 if lat_vals[1] > lat_vals[0] else -1.0  # +1 ascending, -1 descending
    dy_m = lat_dir * dlat_deg * 111000.0  # positive -> increasing lat going down rows
    dx_2d = dlon_deg * 111000.0 * np.cos(np.radians(lat_grid_2d))  # (nlat, nlon)

    # \partial v/ \partial x  (central differences along lon axis=1)
    dvdx = np.full_like(v2d, np.nan)
    dvdx[:, 1:-1] = (v2d[:, 2:] - v2d[:, :-2]) / (2.0 * dx_2d[:, 1:-1])
    dvdx[:, 0] = (v2d[:, 1] - v2d[:, 0]) / dx_2d[:, 0]
    dvdx[:, -1] = (v2d[:, -1] - v2d[:, -2]) / dx_2d[:, -1]

    # \partial u/ \partial y  (central differences along lat axis=0)
    dudy = np.full_like(u2d, np.nan)
    dudy[1:-1, :] = (u2d[2:, :] - u2d[:-2, :]) / (2.0 * dy_m)
    dudy[0, :] = (u2d[1, :] - u2d[0, :]) / dy_m
    dudy[-1, :] = (u2d[-1, :] - u2d[-2, :]) / dy_m

    vor_rel = dvdx - dudy  # 2D
    return vor_rel


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

        cached = {}
        for _var in [
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
            if _var in ds_t:
                cached[_var] = ds_t[_var].values  # one disk read per variable

        if "u10" in cached and "v10" in cached:
            cached["vor2d"] = compute_vorticity_2d(ds_t, lat_grid_2d, lon_grid_2d)
            cached["wind10"] = np.sqrt(cached["u10"] ** 2 + cached["v10"] ** 2)  # m/s

        if "d2m" in cached and "sp" in cached:
            cached["w"] = compute_mixing_ratio(cached["d2m"], cached["sp"])  # g/kg

        if "avg_slhtf" in cached and "avg_ishf" in cached:
            cached["total_hf"] = cached["avg_slhtf"] + cached["avg_ishf"]

        _ref_shape = cached.get("u10", cached.get("t2m", None))
        cached_shape = _ref_shape.shape if _ref_shape is not None else None

        # std_print(
        #     "DEBUG",
        #     "az",
        #     f"Processing time {pd.to_datetime(t).strftime('%d/%m %H:%M')} at ({lat_c:.2f}°, {lon_c:.2f}°)",
        # )
        for j, r in enumerate(radii_km):
            try:
                # Case: r = 0 km (center point only)
                if r == 0:

                    f_cor = 2 * EARTH_OMEGA * np.sin(np.radians(lat_c))

                    for var in variables:
                        if var == "msl" and "msl" in cached:
                            val = interp_center_np(cached["msl"]) / 100.0
                        elif var == "tp" and "tp" in cached:
                            val = interp_center_np(cached["tp"]) * 1000.0
                        elif var == "total_hf" and "total_hf" in cached:
                            val = interp_center_np(cached["total_hf"])
                        elif var == "wind10" and "wind10" in cached:
                            val = interp_center_np(cached["wind10"])
                        elif var == "speed_ms" and "wind10" in cached:
                            val = interp_center_np(cached["wind10"])
                        elif var == "vor_rel" and "vor2d" in cached:
                            val = interp_center_np(cached["vor2d"])
                        elif var == "vor_abs" and "vor2d" in cached:
                            val = interp_center_np(cached["vor2d"]) + f_cor
                        elif var == "w" and "w" in cached:
                            val = interp_center_np(cached["w"])
                        elif var == "vr10":
                            val = 0.0  # radial wind undefined at center
                        elif var == "vt10":
                            val = 0.0  # tangential wind undefined at center
                        elif var in cached:
                            val = interp_center_np(cached[var])
                        else:
                            val = np.nan
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

                    # Verify mask shape using pre-cached shape (no .values call)
                    if cached_shape is not None and sector_mask.shape != cached_shape:
                        std_print(
                            "WARN",
                            "az",
                            f"Shape mismatch: mask {sector_mask.shape} vs data {cached_shape}",
                        )
                        continue

                    n_points = np.sum(sector_mask)
                    if n_points == 0:
                        continue

                    valid_directions.append(dir_name)

                    f_cor = 2 * EARTH_OMEGA * np.sin(np.radians(lat_c))
                    dir_rad = np.radians(dir_deg)

                    # Process each variable using pre-cached numpy arrays (no .values calls)
                    for var in variables:
                        arr = cached.get(var)  # direct lookup in cached raw arrays
                        if arr is not None:
                            # Raw ERA5 variable — apply unit conversions
                            sector_vals = arr[sector_mask]
                            clean = sector_vals[~np.isnan(sector_vals)]
                            if len(clean) > 0:
                                avg_val = np.nanmean(clean)
                                if var == "msl":
                                    avg_val = avg_val / 100.0  # Pa -> hPa
                                elif var == "tp":
                                    avg_val = avg_val * 1000.0  # m/hr -> mm/hr
                                hov_results[var][i, j, d_idx] = avg_val
                        elif var == "total_hf" and "total_hf" in cached:
                            vals = cached["total_hf"][sector_mask]
                            clean = vals[~np.isnan(vals)]
                            if len(clean) > 0:
                                hov_results[var][i, j, d_idx] = np.nanmean(clean)
                        elif var == "wind10" and "wind10" in cached:
                            vals = cached["wind10"][sector_mask]
                            clean = vals[~np.isnan(vals)]
                            if len(clean) > 0:
                                hov_results[var][i, j, d_idx] = np.nanmean(clean)
                        elif var == "speed_ms" and "wind10" in cached:
                            vals = cached["wind10"][sector_mask]
                            clean = vals[~np.isnan(vals)]
                            if len(clean) > 0:
                                hov_results[var][i, j, d_idx] = np.nanmean(clean)
                        elif var == "vor_rel" and "vor2d" in cached:
                            vals = cached["vor2d"][sector_mask]
                            clean = vals[~np.isnan(vals)]
                            if len(clean) > 0:
                                hov_results[var][i, j, d_idx] = np.nanmean(clean)
                        elif var == "vor_abs" and "vor2d" in cached:
                            vals = cached["vor2d"][sector_mask]
                            clean = vals[~np.isnan(vals)]
                            if len(clean) > 0:
                                hov_results[var][i, j, d_idx] = (
                                    np.nanmean(clean) + f_cor
                                )
                        elif var == "w" and "w" in cached:
                            vals = cached["w"][sector_mask]
                            clean = vals[~np.isnan(vals)]
                            if len(clean) > 0:
                                hov_results[var][i, j, d_idx] = np.nanmean(clean)
                        elif var == "vr10" and "u10" in cached and "v10" in cached:
                            u_s = cached["u10"][sector_mask]
                            v_s = cached["v10"][sector_mask]
                            ok = ~(np.isnan(u_s) | np.isnan(v_s))
                            if np.sum(ok) > 0:
                                vr = u_s[ok] * np.sin(dir_rad) + v_s[ok] * np.cos(
                                    dir_rad
                                )
                                hov_results[var][i, j, d_idx] = np.nanmean(vr)
                        elif var == "vt10" and "u10" in cached and "v10" in cached:
                            u_s = cached["u10"][sector_mask]
                            v_s = cached["v10"][sector_mask]
                            ok = ~(np.isnan(u_s) | np.isnan(v_s))
                            if np.sum(ok) > 0:
                                vt = -u_s[ok] * np.cos(dir_rad) + v_s[ok] * np.sin(
                                    dir_rad
                                )
                                hov_results[var][i, j, d_idx] = np.nanmean(vt)
                if len(valid_directions) > 0:
                    dir_summary = ", ".join(valid_directions)
                    dir_summary = dir_summary[:-1]
                    # std_print("DEBUG", "az", f"  {r:3d}km: {dir_summary} ({len(valid_directions)}/{n_theta})",)
            except Exception as e:
                std_print("ERROR", "az", f"Failed at time {t}, radius {r}km", str(e))
                traceback.print_exc()
                continue
    return hov_results, track_times, landfall_index


def average_azimuthally(hov_results, variables):
    averaged = {}
    for var in variables:
        if var in hov_results:
            with np.errstate(invalid="ignore"):
                averaged[var] = np.nanmean(hov_results[var], axis=2)
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
    if var_name in ("vor_rel", "vor_abs"):
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
    elif var_name in ["vor_rel", "vor_abs"]:
        cmap = "RdBu_r"
    elif var_name == "w":
        cmap = "BuPu"
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
