import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import RegularGridInterpolator

from utils import (
    std_print,
    is_point_land,
    find_landfall_time,
    setup_output_dir
)

LEVELS = 36 # 20

EARTH_R = 6371000.0 # m
EARTH_OMEGA = 7.2921159e-5  # angular velocity (rad/s)

VARIABLE_UNITS = {
    'u10': 'm/s',
    'v10': 'm/s',
    'wind10': 'm/s',    # wind10 = sqrt(u10^2 + v10^2)
    'vr10': 'm/s',
    'vt10': 'm/s',
    'vor_rel': 's⁻¹',
    'vor_abs': 's⁻¹',
    'speed_kmh': 'km/h',
    'speed_ms': 'm/s',
    'msl': 'hPa',
    'sp': 'hPa',
    'tp': 'cm/hr',
    't2m': 'K',
    'stl1': 'K',
    'swvl1': 'm³/m³',
    'avg_slhtf': 'W/m²',
    'avg_ishf': 'W/m²',
    'total_hf': 'W/m²'   # Total surface heat flux = avg_ishf + avg_slhtf
}

# list of variable for plotting
VARIABLE_NAMES = {
    'u10': 'Zonal Wind (10m)',
    'v10': 'Meridional Wind (10m)',
    'wind10': 'Wind Speed (10m)',
    'vr10': 'Radial Wind (Inflow/Outflow)',
    'vt10': 'Tangential Wind (Swirl)',
    'vor_rel': 'Relative Vorticity',
    'vor_abs': 'Absolute Vorticity',
    'speed_kmh': 'Translation Speed',
    'msl': 'Mean Sea Level Pressure',
    'sp': 'Surface Pressure',
    'tp': 'Total Precipitation',
    't2m': '2m Air Temperature',
    'stl1': 'Soil Temperature (0-7cm)',
    'swvl1': 'Soil Water Volumatric Layer-1 (0-7cm)',
    'avg_slhtf': 'Surface Latent Heat Flux',
    'avg_ishf': 'Surface Sensible Heat Flux',
    'total_hf': 'Total Surface Heat Flux'
}

# Helper function
def get_polar_points(lon_c, lat_c, radius_km, n_theta=72):
    angles = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    d_lat = radius_km / 111.0
    d_lon = radius_km / (111.0 * np.cos(np.radians(lat_c)))

    lons = lon_c + d_lon * np.cos(angles)
    lats = lat_c + d_lat * np.sin(angles)
    return lons, lats, angles

# Helper function
def compute_vorticity(u, v, lat, lon):
    """Calculate relative vorticity : zeta = (partial v/ partial x) - (partial u/ partial y)"""
    # Convert lat/lon to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Create 2D grids
    lat_grid, lon_grid = np.meshgrid(lat_rad, lon_rad, indexing='ij')
    # Calculate grid spacing in meters
    dlat = np.gradient(lat_rad)
    dlon = np.gradient(lon_rad)
    # Calculate derivatives with proper metric terms
    # \partial u/ \partial y = (1/R) * \partial u/\partial \theta
    dudy = np.gradient(u, axis=0) / (EARTH_R * dlat[:, np.newaxis])

    # \partial v/\partial x = (1/(R cosθ)) * (\partial v)/(\partial \lambda)
    cos_lat = np.cos(lat_grid)
    # Avoid division by zero near poles
    cos_lat = np.where(cos_lat < 1e-6, 1e-6, cos_lat)
    dvdx = np.gradient(v, axis=1) / (EARTH_R * cos_lat * dlon[np.newaxis, :])
    # Relative vorticity: \zeta = \partial v/\partial x - \partial u/\partial y
    vorticity = dvdx - dudy
    return vorticity

def compute_azimuthal_hovmoller(ds, track_df, radii_km, variables, n_theta=72):
    """
    Compute azimuthal Hovmöller diagrams for a given track.
        DataFrame containing track coordinates with columns:
        'time', 'latitude', 'longitude', 'track_id'
        Radii (in kilometers) to extract data
        n_theta : Number of azimuthal angles
    """
    track_times = track_df['time'].values
    N_steps = len(track_times)
    N_radii = len(radii_km)

    # Initialize results
    hov_results = {v: np.full((N_steps, N_radii), np.nan) for v in variables}
    derived_winds = {'vr10', 'vt10', 'vor_rel', 'vor_abs'}
    needs_wind = any(v in derived_winds for v in variables)

    if any(v in ['vr10', 'vt10', 'vor_rel', 'vor_abs'] for v in variables):
        wind_vars_needed = ['u10', 'v10']
        hov_results['vr10'] = np.zeros((N_steps, N_radii))
        hov_results['vt10'] = np.zeros((N_steps, N_radii))
        hov_results['vor_rel'] = np.zeros((N_steps, N_radii))
        hov_results['vor_abs'] = np.zeros((N_steps, N_radii))

    landfall_index = -1
    for i, t in enumerate(track_times):
        row = track_df.iloc[i]
        lon_c, lat_c = row['longitude'], row['latitude']
        # Use pre-computed is_land if available
        if 'is_land' in track_df.columns:
            is_land_now = track_df.iloc[i]['is_land']
            if landfall_index == -1 and is_land_now:
                landfall_index = i

        # track_df actually contains a speed_ms column before calling the Hovmöller function.
        # If it doesn't, calculate it using speed_analysis.py
        if 'speed_ms' not in track_df.columns:
            std_print('DEBUG', 'az', 'speed_ms not found in `track_df`, Computing : speed_ms')
            dist = np.sqrt(track_df['longitude'].diff()**2 + track_df['latitude'].diff()**2) * 111.0
            dt = track_df['time'].diff().dt.total_seconds() / 3600.0
            track_df['speed_ms'] = (dist * 1000.0) / (dt * 3600.0)
        # Select time slice
        try:
            ds_t = ds.sel(time=t, method='nearest')
        except Exception as e:
            std_print('ERROR', 'ho', f'Failed to select time {t}', str(e))
            continue

        # Pre-calculate vorticity for this time if needed
        vorticity_map = None
        vorticity_abs_map = None
        if 'vor_rel' in variables or 'vor_abs' in variables:
            if 'u10' in ds_t and 'v10' in ds_t:
                # Get lat/lon arrays
                lat = ds_t['latitude'].values
                lon = ds_t['longitude'].values
                # Calculate vorticity over entire domain
                vorticity_map = compute_vorticity(ds_t['u10'].values, ds_t['v10'].values, lat, lon)
                # Absolute vorticity: \zeta + f (where f = 2Ω sin(\phi))
                omega = EARTH_OMEGA
                lat_rad = np.deg2rad(lat[:, np.newaxis])
                coriolis = 2 * omega * np.sin(lat_rad)
                vorticity_abs_map = vorticity_map + coriolis

        for j, r in enumerate(radii_km):
            try:
                # 1. Generate Polar Grid Points for radius r
                lons_p, lats_p, phis = get_polar_points(lon_c, lat_c, r, n_theta)
                # Check if polar points are within dataset bounds
                lon_min, lon_max = float(ds['longitude'].min()), float(ds['longitude'].max())
                lat_min, lat_max = float(ds['latitude'].min()), float(ds['latitude'].max())
                # Filter out-of-bounds points
                valid_mask = ((lons_p >= lon_min) & (lons_p <= lon_max) & (lats_p >= lat_min) & (lats_p <= lat_max))
                if np.sum(valid_mask) < n_theta // 2:
                    # Skip if too many points out of bounds
                    for var in variables:
                        hov_results[var][i, j] = np.nan
                    continue
                # Filter to valid points only
                lons_p_valid = lons_p[valid_mask]
                lats_p_valid = lats_p[valid_mask]
                phis_valid = phis[valid_mask]

                # 2. Interpolate ERA5 data onto VALID polar points only
                interp_ds = ds_t.interp(
                    longitude=xr.DataArray(lons_p_valid),
                    latitude=xr.DataArray(lats_p_valid),
                    method='linear',
                    # kwargs={'fill_value': np.nan}
                )

                # 3. Calculate Base Variables
                for var in variables:
                    if var in interp_ds:
                        val = interp_ds[var].mean().values
                        # Convert MSL from Pa to hPa
                        if var == 'msl':
                            val = val / 100.00
                        # # Convert TP from mm/hr to m/hr
                        # if var == 'tp':
                        #     val = val * 1000.00
                        hov_results[var][i, j] = val if not np.isnan(val) else np.nan
                    elif var == 'wind10':
                        if 'u10' in interp_ds and 'v10' in interp_ds:
                            w10 = np.sqrt(interp_ds['u10']**2 + interp_ds['v10']**2)
                            val = w10.mean().values
                            hov_results[var][i, j] = val if not np.isnan(val) else np.nan
                    elif var == 'total_hf':
                        if 'avg_slhtf' in interp_ds and 'avg_ishf' in interp_ds:
                            hf = interp_ds['avg_slhtf'] + interp_ds['avg_ishf']
                            val = hf.mean().values
                            hov_results[var][i, j] = val if not np.isnan(val) else np.nan
                    elif var == 'speed_ms':
                            if 'speed_ms' in track_df.columns:
                                hov_results[var][i, j] = track_df.iloc[i]['speed_ms']
                            else:
                                hov_results[var][i, j] = np.nan

                # 4. Calculate wind components (radial, tangential, vorticity)
                if 'u10' in interp_ds and 'v10' in interp_ds:
                    u = interp_ds['u10'].values
                    v = interp_ds['v10'].values
                    valid_wind_mask = ~(np.isnan(u) | np.isnan(v))
                    if np.sum(valid_wind_mask) > 0:
                        u_valid = u[valid_wind_mask]
                        v_valid = v[valid_wind_mask]
                        phis_wind_valid = phis_valid[valid_wind_mask]
                        # Radial and tangential wind
                        vr = u_valid * np.cos(phis_wind_valid) + v_valid * np.sin(phis_wind_valid)
                        vt = -u_valid * np.sin(phis_wind_valid) + v_valid * np.cos(phis_wind_valid)
                        if 'vr10' in hov_results:
                            hov_results['vr10'][i, j] = np.nanmean(vr)
                        if 'vt10' in hov_results:
                            hov_results['vt10'][i, j] = np.nanmean(vt)
                        # Calculate vorticity at polar points
                        if vorticity_map is not None:
                            # Create interpolator for vorticity
                            lat_grid = ds_t['latitude'].values
                            lon_grid = ds_t['longitude'].values
                            # Handle longitude wrapping if needed
                            lon_grid_adj = lon_grid.copy()
                            if np.any(lons_p_valid < lon_grid.min()):
                                lon_grid_adj[lon_grid_adj > 180] -= 360
                                lons_p_adj = lons_p_valid.copy()
                                lons_p_adj[lons_p_adj > 180] -= 360
                            else:
                                lons_p_adj = lons_p_valid
                            # Create interpolator
                            interp_vort = RegularGridInterpolator(
                                (lat_grid, lon_grid_adj),
                                vorticity_map,
                                method='linear',
                                bounds_error=False,
                                fill_value=np.nan
                            )
                            # Interpolate vorticity at polar points
                            points = np.column_stack([lats_p_valid, lons_p_adj])
                            vort_at_points = interp_vort(points)
                            # Average vorticity around the circle
                            valid_vort_mask = ~np.isnan(vort_at_points)
                            if np.sum(valid_vort_mask) > 0:
                                if 'vor_rel' in hov_results:
                                    hov_results['vor_rel'][i, j] = np.nanmean(vort_at_points[valid_vort_mask])
                                if 'vor_abs' in hov_results and vorticity_abs_map is not None:
                                    interp_vort_abs = RegularGridInterpolator(
                                        (lat_grid, lon_grid_adj),
                                        vorticity_abs_map,
                                        method='linear',
                                        bounds_error=False,
                                        fill_value=np.nan
                                    )
                                    vort_abs_at_points = interp_vort_abs(points)
                                    hov_results['vor_abs'][i, j] = np.nanmean(vort_abs_at_points[valid_vort_mask])
                    else:
                        # No valid wind data
                        if 'vr10' in hov_results:
                            hov_results['vr10'][i, j] = np.nan
                        if 'vt10' in hov_results:
                            hov_results['vt10'][i, j] = np.nan
                        if 'vor_rel' in hov_results:
                            hov_results['vor_rel'][i, j] = np.nan
                        if 'vor_abs' in hov_results:
                            hov_results['vor_abs'][i, j] = np.nan
            except Exception as e:
                std_print('ERROR', 'ho', f'Failed at time {t}, radius {r} km', str(e))
                continue
    return hov_results, track_times, landfall_index

def plot_hovmoller(hov_arr, track_times, radii_km, var_name, track_df, output_dir="hov_plots", landfall_index=-1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    var_label = VARIABLE_NAMES.get(var_name, var_name)
    var_unit = VARIABLE_UNITS.get(var_name, '')
    track_df = track_df.set_index('time').reindex(track_times).reset_index()

    display_arr = hov_arr.copy()
    fmt = '%1.2f'

    if 'vor_rel' in var_name and 'vor_abs' in var_name:
        display_arr = hov_arr * 1e5  # Scale to readable units
        var_unit = r'$\times 10^{-5}$ s$^{-1}$'
        fmt = '%1.2f'
    elif var_name == ['tp', 'swvl1']:
        fmt = '%1.4f'

    if var_name in ['tp', 'wind10', 'avg_slhtf', 'avg_ishf', 'total_hf']:
        cmap = 'plasma'
    elif var_name in ['t2m', 'stl1']:
        cmap = 'coolwarm'
    elif var_name in ['msl', 'sp']:
        cmap = 'viridis'
    elif var_name == ['swvl1', 'speed_ms']:
        cmap = 'YlGnBu'
    else:
        cmap = 'viridis'

    if 'is_land' not in track_df.columns:
         track_df['is_land'] = track_df.apply(lambda row: is_point_land(row['latitude'], row['longitude']), axis=1)
    land_status = track_df['is_land'].fillna(False).values
    landfall_indices = np.where(np.diff(land_status.astype(int)) == 1)[0] + 1
    # landfall_index = -1
    # if len(landfall_indices) > 0:
    #     # Get the first landfall instance
    #     landfall_index = landfall_indices[0]
    # elif land_status[0] and len(land_status) > 1 and not land_status[1]:
    #     # Handle case where the track starts on land and immediately moves to ocean
    #     landfall_index = 0
    # elif land_status[0] and all(land_status):
    #     # Case where the entire tracked period is over land
    #     landfall_index = 0
    landfall_index = landfall_indices[0] if len(landfall_indices) > 0 else landfall_index

    # Determine landfall status
    originates_over_land = False
    if len(track_df) > 0:
        first_row = track_df.iloc[0]
        originates_over_land = is_point_land(first_row['latitude'], first_row['longitude'])

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    # Use index positions for y-axis
    N_steps = len(track_times)
    # Contours
    X, Y = np.meshgrid(radii_km, np.arange(N_steps))
    im = ax.contourf(X, Y, hov_arr, levels=LEVELS, cmap=cmap, extend='both')
    contours = ax.contour(X, Y, hov_arr, levels=int(LEVELS/3), colors='black', linewidths=0.5, alpha=0.5)
    # contours = ax.contour(X, Y, display_arr, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    ax.clabel(contours, inline=True, fontsize=8, fmt=fmt)
    # Landfall line
    if landfall_index != -1 and landfall_index < N_steps:
        ax.axhline(y=landfall_index, color='black', linestyle='--', linewidth=2, label='Landfall', alpha=0.9)
    cbar = plt.colorbar(im, pad=0.02)
    # label = f"{var_label} ({var_unit})" if var_unit else var_label
    # cbar.set_label(label, fontsize=12)
    cbar.set_label(f"{var_label} ({var_unit})", fontsize=12)

    ax.set_xlabel("Radius from center (km)", fontsize=11)
    ax.set_ylabel("Time (hours)", fontsize=11)
    ax.set_title(f'{var_label}', fontsize=13, fontweight='bold')

    # plt.xlabel("Radius from center (km)", fontsize=11)
    # plt.ylabel("Time (hr)", fontsize=11)
    # plt.grid(True, alpha=0.3, linestyle='--')
    # plt.title(f"{var_label}", fontsize=13, fontweight='bold')

    # Set y-ticks to show real datetime
    if N_steps <= 20:
        # Show all time steps
        ytick_positions = np.arange(N_steps)
        ytick_labels = [pd.to_datetime(t).strftime('%m/%d\n%H:%M') for t in track_times]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, rotation=45, fontsize=8)
    else:
        # Show subset of time points
        step = max(1, N_steps // 10)
        ytick_positions = np.arange(0, N_steps, step)
        # ytick_labels = [pd.to_datetime(track_times[i]).strftime('%m/%d\n%H:%M')
        #                    for i in np.arange(0, N_steps, step)]
        ytick_labels = [pd.to_datetime(track_times[int(i)]).strftime('%d/%m\n%H:%M')
                           if int(i) < len(track_times) else ''
                           for i in np.arange(0, N_steps, step)]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=8)

    # Set x-ticks
    ax.set_xticks(radii_km[::2])
    ax.set_xticklabels(radii_km[::2], fontsize=10)
    common_bbox =  dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3)
    # Landfall annotation at top right
    if landfall_index != -1 and landfall_index < N_steps:
        landfall_time_str = pd.to_datetime(track_times[landfall_index]).strftime("%m/%d %H:%M")
        landfall_text = f"Landfall: {landfall_time_str}"
        ax.text(0.77, 1.011, landfall_text, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', fontsize=10, fontweight='bold', color='black', bbox=common_bbox)

    # Legend if landfall line is plotted
    if landfall_index != -1 and landfall_index < N_steps:
        ax.legend(loc='upper right', fontsize=9)

    azimuthal_plot_path = os.path.join(output_dir, f"hov_{var_name}.png")
    plt.tight_layout()
    plt.savefig(azimuthal_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print('SAVED', 'az', 'Hovmoller plot', azimuthal_plot_path)
