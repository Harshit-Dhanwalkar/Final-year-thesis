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
import matplotlib.pyplot as plt
from utils import std_print
import traceback
import warnings

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
    "vor_rel": "s⁻¹",
    "vor_abs": "s⁻¹",
    "speed_kmh": "km/h",
    "speed_ms": "m/s",
    "msl": "hPa",
    "sp": "hPa",
    "tp": "mm/hr",
    "t2m": "K",
    "stl1": "K",
    "swvl1": "m³/m³",
    "avg_slhtf": "W/m²",
    "avg_ishf": "W/m²",
    "total_hf": "W/m²",
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
    "stl1": "Soil Temperature (0-7cm)",
    "swvl1": "Soil Water Volumatric Layer-1 (0-7cm)",
    "avg_slhtf": "Surface Latent Heat Flux",
    "avg_ishf": "Surface Sensible Heat Flux",
    "total_hf": "Total Surface Heat Flux",
}


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

    # Define directions based on n_theta
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

        std_print(
            "DEBUG",
            "az",
            f"Processing time {pd.to_datetime(t).strftime('%d/%m %H:%M')} at ({lat_c:.2f}°, {lon_c:.2f}°)",
        )
        for j, r in enumerate(radii_km):
            try:
                # Special case: r = 0 km (center point only)
                if r == 0:
                    interp_center = ds_t.interp(
                        longitude=lon_c, latitude=lat_c, method="linear"
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
                            # Relative vorticity computed from wind components
                            if "u10" in interp_center and "v10" in interp_center:
                                u = float(interp_center["u10"].values)
                                v = float(interp_center["v10"].values)
                                val = np.sqrt(u**2 + v**2)  # Wind speed proxy
                            else:
                                val = np.nan
                        elif var == "vor_abs":
                            # Absolute vorticity = relative vorticity + f (Coriolis)
                            if "u10" in interp_center and "v10" in interp_center:
                                u = float(interp_center["u10"].values)
                                v = float(interp_center["v10"].values)
                                f = 2 * EARTH_OMEGA * np.sin(np.radians(lat_c))
                                wind_speed = np.sqrt(u**2 + v**2)
                                val = wind_speed + abs(f)
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

                    # Verify mask shape matches data
                    if sector_mask.shape != ds_t["u10"].values.shape:
                        std_print(
                            "WARN",
                            "az",
                            f"Shape mismatch: mask {sector_mask.shape} vs data {ds_t['u10'].values.shape}",
                        )
                        continue

                    n_points = np.sum(sector_mask)
                    if n_points == 0:
                        continue

                    valid_directions.append(dir_name)

                    # Process each variable for this sector
                    for var in variables:
                        if var in ds_t:
                            var_data = ds_t[var].values
                            sector_values = var_data[sector_mask]
                            sector_values_clean = sector_values[
                                ~np.isnan(sector_values)
                            ]
                            if len(sector_values_clean) > 0:
                                avg_val = np.nanmean(sector_values_clean)
                                if var == "msl":
                                    avg_val = avg_val / 100.0  # Pa to hPa
                                elif var == "tp":
                                    avg_val = avg_val * 1000.0  # m/hr to mm/hr
                                hov_results[var][i, j, d_idx] = avg_val
                        elif var == "wind10":
                            if "u10" in ds_t and "v10" in ds_t:
                                u_data = ds_t["u10"].values[sector_mask]
                                v_data = ds_t["v10"].values[sector_mask]
                                valid_pair_mask = ~(np.isnan(u_data) | np.isnan(v_data))
                                if np.sum(valid_pair_mask) > 0:
                                    u_clean = u_data[valid_pair_mask]
                                    v_clean = v_data[valid_pair_mask]
                                    w10_data = np.sqrt(u_clean**2 + v_clean**2)
                                    hov_results[var][i, j, d_idx] = np.nanmean(w10_data)
                        elif var == "speed_ms":
                            if "u10" in ds_t and "v10" in ds_t:
                                u_data = ds_t["u10"].values[sector_mask]
                                v_data = ds_t["v10"].values[sector_mask]
                                valid_pair_mask = ~(np.isnan(u_data) | np.isnan(v_data))
                                if np.sum(valid_pair_mask) > 0:
                                    u_clean = u_data[valid_pair_mask]
                                    v_clean = v_data[valid_pair_mask]
                                    speed_data = np.sqrt(u_clean**2 + v_clean**2)
                                    hov_results[var][i, j, d_idx] = np.nanmean(
                                        speed_data
                                    )
                        elif var == "vor_rel":
                            if "u10" in ds_t and "v10" in ds_t:
                                u_data = ds_t["u10"].values[sector_mask]
                                v_data = ds_t["v10"].values[sector_mask]
                                valid_pair_mask = ~(np.isnan(u_data) | np.isnan(v_data))
                                if np.sum(valid_pair_mask) > 0:
                                    u_clean = u_data[valid_pair_mask]
                                    v_clean = v_data[valid_pair_mask]
                                    wind_speed = np.sqrt(u_clean**2 + v_clean**2)
                                    hov_results[var][i, j, d_idx] = np.nanmean(
                                        wind_speed
                                    )
                        elif var == "vor_abs":
                            if "u10" in ds_t and "v10" in ds_t:
                                u_data = ds_t["u10"].values[sector_mask]
                                v_data = ds_t["v10"].values[sector_mask]
                                valid_pair_mask = ~(np.isnan(u_data) | np.isnan(v_data))
                                if np.sum(valid_pair_mask) > 0:
                                    u_clean = u_data[valid_pair_mask]
                                    v_clean = v_data[valid_pair_mask]
                                    f = 2 * EARTH_OMEGA * np.sin(np.radians(lat_c))
                                    wind_speed = np.sqrt(u_clean**2 + v_clean**2)
                                    abs_vort = wind_speed + abs(f)
                                    hov_results[var][i, j, d_idx] = np.nanmean(abs_vort)
                        elif var == "vr10":
                            if "u10" in ds_t and "v10" in ds_t:
                                u_data = ds_t["u10"].values[sector_mask]
                                v_data = ds_t["v10"].values[sector_mask]
                                valid_pair_mask = ~(np.isnan(u_data) | np.isnan(v_data))
                                if np.sum(valid_pair_mask) > 0:
                                    u_clean = u_data[valid_pair_mask]
                                    v_clean = v_data[valid_pair_mask]
                                    dir_rad = np.radians(dir_deg)
                                    vr = u_clean * np.sin(dir_rad) + v_clean * np.cos(
                                        dir_rad
                                    )
                                    hov_results[var][i, j, d_idx] = np.nanmean(vr)
                        elif var == "vt10":
                            if "u10" in ds_t and "v10" in ds_t:
                                u_data = ds_t["u10"].values[sector_mask]
                                v_data = ds_t["v10"].values[sector_mask]
                                valid_pair_mask = ~(np.isnan(u_data) | np.isnan(v_data))
                                if np.sum(valid_pair_mask) > 0:
                                    u_clean = u_data[valid_pair_mask]
                                    v_clean = v_data[valid_pair_mask]
                                    dir_rad = np.radians(dir_deg)
                                    vt = -u_clean * np.cos(dir_rad) + v_clean * np.sin(
                                        dir_rad
                                    )
                                    hov_results[var][i, j, d_idx] = np.nanmean(vt)
                        elif var == "total_hf":
                            if "avg_slhtf" in ds_t and "avg_ishf" in ds_t:
                                lhf_data = ds_t["avg_slhtf"].values[sector_mask]
                                shf_data = ds_t["avg_ishf"].values[sector_mask]
                                valid_mask = ~(np.isnan(lhf_data) | np.isnan(shf_data))
                                if np.sum(valid_mask) > 0:
                                    total_hf = (
                                        lhf_data[valid_mask] + shf_data[valid_mask]
                                    )
                                    hov_results[var][i, j, d_idx] = np.nanmean(total_hf)
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

    # Choose colormap based on variable
    if var_name in ["tp", "wind10", "avg_slhtf", "avg_ishf", "total_hf"]:
        cmap = "plasma"
    elif var_name in ["t2m", "stl1"]:
        cmap = "coolwarm"
    elif var_name in ["msl", "sp"]:
        cmap = "viridis_r"  # Reversed for pressure (lower pressure = darker)
    elif var_name in ["swvl1", "speed_ms"]:
        cmap = "YlGnBu"
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
    common_bbox = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3)
    # Landfall annotation at top right
    if landfall_index != -1 and landfall_index < N_steps:
        landfall_time_str = pd.to_datetime(track_times[landfall_index]).strftime(
            "%m/%d %H:%M"
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
            bbox=common_bbox,
        )
    # Legend if landfall line is plotted
    if landfall_index != -1 and landfall_index < N_steps:
        ax.legend(
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
            edgecolor="none",
            fancybox=True,
            shadow=True,
        )

    # Save plot
    plot_path = os.path.join(plot_dir, f"hov_{var_name}_azimuthal_avg.png")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "az", f"Hovmöller plot for {var_name}", plot_path)
