import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import (
    std_print,
    is_point_land,
    find_landfall_time,
    setup_output_dir
)

VARIABLE_UNITS = {
    'u10': 'm/s',
    'v10': 'm/s',
    'wind10': 'm/s',  # wind10 = sqrt(u10^2 + v10^2)
    'vr10': 'm/s',    # Radial Wind
    'vt10': 'm/s',    # Tangential Wind
    'speed_kmh': 'km/h',
    'speed_ms': 'm/s',
    'msl': 'Pa',
    'tp': 'mm/hr',
    't2m': '°C',
    'sp': 'hPa',
    'stl1': '°C',
    'swvl1': 'm³/m³',
    'avg_slhtf': 'W/m²',
    'avg_ishf': 'W/m²',
    'total_hf': 'W/m²' # Total surface heat flux = avg_ishf + avg_slhtf
}

# list of variable for plotting
VARIABLE_NAMES = {
    'u10': 'Zonal Wind (10m)',
    'v10': 'Meridional Wind (10m)',
    'wind10': 'Wind Speed (10m)',
    'vr10': 'Radial Wind (Inflow/Outflow)',
    'vt10': 'Tangential Wind (Swirl)',
    'speed_kmh': 'Translation Speed',
    'msl': 'Mean Sea Level Pressure',
    'tp': 'Total Precipitation',
    't2m': '2m Temperature',
    # 'sp': 'Surface Pressure',
    'stl1': 'Soil Temperature (0-7cm)',
    'swvl1': 'Soil Water Content (0-7cm)',
    'avg_slhtf': 'Surface Latent Heat Flux',
    'avg_ishf': 'Surface Sensible Heat Flux',
    'total_hf': 'Total Surface Heat Flux'
}

EARTH_R = 6371.0 # km

# Helper function
def get_polar_points(
    lon_c,
    lat_c,
    radius_km,
    n_theta=72,
):
    angles = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    d_lat = radius_km / 111.0
    d_lon = radius_km / (111.0 * np.cos(np.radians(lat_c)))

    lons = lon_c + d_lon * np.cos(angles)
    lats = lat_c + d_lat * np.sin(angles)
    return lons, lats, angles

def compute_azimuthal_hovmoller(ds, track_df, radii_km, variables, n_theta=72):
    """
    Compute azimuthal Hovmöller diagrams for a given track.
        DataFrame containing track coordinates with columns:
        'time', 'latitude', 'longitude', 'track_id'
        Radii (in kilometers) to extract data
        n_theta : Number of azimuthal angles (default: 72)
    """
    track_times = track_df['time'].values
    N_steps = len(track_times)
    N_radii = len(radii_km)

    # Initialize results
    hov_results = {v: np.zeros((N_steps, N_radii)) for v in variables}
    if 'vr10' in variables or 'vt10' in variables:
        hov_results['vr10'] = np.zeros((N_steps, N_radii))
        hov_results['vt10'] = np.zeros((N_steps, N_radii))

    landfall_index = -1
    for i, t in enumerate(track_times):
        row = track_df.iloc[i]
        lon_c, lat_c = row['longitude'], row['latitude']

        # Use pre-computed is_land if available
        if 'is_land' in track_df.columns:
            is_land_now = track_df.iloc[i]['is_land']
            if landfall_index == -1 and is_land_now:
                landfall_index = i

        # Select time slice
        try:
            ds_t = ds.sel(time=t, method='nearest')
        except Exception as e:
            std_print('WARN', 'ho', f'Failed to select time {t} :', e)
            continue

        for j, r in enumerate(radii_km):
            try:
                # 1. Generate Polar Grid Points for radius r
                lons_p, lats_p, phis = get_polar_points(lon_c, lat_c, r, n_theta)
                # Check if polar points are within dataset bounds
                lon_min, lon_max = float(ds['longitude'].min()), float(ds['longitude'].max())
                lat_min, lat_max = float(ds['latitude'].min()), float(ds['latitude'].max())
                # Filter out-of-bounds points
                valid_mask = (
                    (lons_p >= lon_min) & (lons_p <= lon_max) &
                    (lats_p >= lat_min) & (lats_p <= lat_max)
                )
                if np.sum(valid_mask) < n_theta // 2:
                    # Less than half the points are valid - skip this radius
                    for var in variables:
                        hov_results[var][i, j] = np.nan
                    if 'vr10' in hov_results:
                        hov_results['vr10'][i, j] = np.nan
                    if 'vt10' in hov_results:
                        hov_results['vt10'][i, j] = np.nan
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
                    kwargs={'fill_value': np.nan}
                )
                # 3. Calculate Base Variables
                for var in variables:
                    if var in interp_ds:
                        val = interp_ds[var].mean().values
                        # Check if valid (not NaN)
                        if not np.isnan(val):
                            hov_results[var][i, j] = val
                        else:
                            hov_results[var][i, j] = np.nan
                    elif var == 'wind10':
                        if 'u10' in interp_ds and 'v10' in interp_ds:
                            w10 = np.sqrt(interp_ds['u10']**2 + interp_ds['v10']**2)
                            val = w10.mean().values
                            hov_results[var][i, j] = val if not np.isnan(val) else np.nan
                        else:
                            hov_results[var][i, j] = np.nan
                    elif var == 'total_hf':
                        if 'avg_slhtf' in interp_ds and 'avg_ishf' in interp_ds:
                            hf = interp_ds['avg_slhtf'] + interp_ds['avg_ishf']
                            val = hf.mean().values
                            hov_results[var][i, j] = val if not np.isnan(val) else np.nan
                        else:
                            hov_results[var][i, j] = np.nan
                # 4. Calculate Radial and Tangential Wind Rotation
                if 'u10' in interp_ds and 'v10' in interp_ds:
                    u = interp_ds['u10'].values
                    v = interp_ds['v10'].values
                    valid_wind_mask = ~(np.isnan(u) | np.isnan(v))
                    if np.sum(valid_wind_mask) > 0:
                        u_valid = u[valid_wind_mask]
                        v_valid = v[valid_wind_mask]
                        phis_wind_valid = phis_valid[valid_wind_mask]
                        ## Vr = u*cos(phi) + v*sin(phi) [positive = outward]
                        ## Vt = -u*sin(phi) + v*cos(phi) [positive = Cyclonic]
                        vr = u_valid * np.cos(phis_wind_valid) + v_valid * np.sin(phis_wind_valid)
                        vt = -u_valid * np.sin(phis_wind_valid) + v_valid * np.cos(phis_wind_valid)
                        if 'vr10' in hov_results:
                            hov_results['vr10'][i, j] = np.nanmean(vr)
                        if 'vt10' in hov_results:
                            hov_results['vt10'][i, j] = np.nanmean(vt)
                    else:
                        if 'vr10' in hov_results:
                            hov_results['vr10'][i, j] = np.nan
                        if 'vt10' in hov_results:
                            hov_results['vt10'][i, j] = np.nan
            except Exception as e:
                std_print('WARN', 'ho', f'Failed to select time {t}, radius {r} km :', e)
                # Mark this cell as NaN
                for var in variables:
                    hov_results[var][i, j] = np.nan
                if 'vr10' in hov_results:
                    hov_results['vr10'][i, j] = np.nan
                if 'vt10' in hov_results:
                    hov_results['vt10'][i, j] = np.nan
                continue

    return hov_results, track_times, landfall_index


def plot_hovmoller(hov_arr, track_times, radii_km, var_name, track_df, output_dir="hov_plots", landfall_index=-1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if var_name in ['tp', 'wind10', 'avg_slhtf', 'avg_ishf', 'total_hf']:
        cmap = 'plasma'
    elif var_name in ['t2m', 'stl1']:
        cmap = 'coolwarm'
    elif var_name in ['msl', 'sp']:
        cmap = 'viridis'
    elif var_name == 'swvl1':
        cmap = 'YlGnBu'
    else:
        cmap = 'viridis'

    var_label = VARIABLE_NAMES.get(var_name, var_name)
    var_unit = VARIABLE_UNITS.get(var_name, '')
    track_df = track_df.set_index('time').reindex(track_times).reset_index()

    if 'is_land' not in track_df.columns:
         # Attempting to use the utility you imported in Step 1
         track_df['is_land'] = track_df.apply(
            lambda row: is_point_land(row['latitude'], row['longitude']),
            axis=1
        )
    land_status = track_df['is_land'].fillna(False).values
    landfall_indices = np.where(np.diff(land_status.astype(int)) == 1)[0] + 1
    landfall_index = -1
    if len(landfall_indices) > 0:
        # Get the first landfall instance
        landfall_index = landfall_indices[0]
    elif land_status[0] and len(land_status) > 1 and not land_status[1]:
        # Handle case where the track starts on land and immediately moves to ocean
        landfall_index = 0
    elif land_status[0] and all(land_status):
        # Case where the entire tracked period is over land
        landfall_index = 0
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
    im = ax.contourf(X, Y, hov_arr, levels=20, cmap=cmap, extend='both')
    contours = ax.contour(X, Y, hov_arr, levels=10, colors='black', linewidths=0.5, alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')
    # Landfall line
    if landfall_index != -1 and landfall_index < N_steps:
        ax.axhline(
            y=landfall_index,
            color='black',
            linestyle='--',
            linewidth=2,
            label='Landfall',
            alpha=0.9,
        )
    cbar = plt.colorbar(im, pad=0.02)
    label = f"{var_label} ({var_unit})" if var_unit else var_label
    cbar.set_label(label, fontsize=12)

    plt.xlabel("Radius from center (km)", fontsize=11)
    plt.ylabel("Time (hr)", fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.title(f"{var_label}", fontsize=13, fontweight='bold')

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
        ytick_labels = [pd.to_datetime(track_times[int(i)]).strftime('%m/%d\n%H:%M') 
                           if int(i) < len(track_times) else ''
                           for i in np.arange(0, N_steps, step)]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=8)

    # Set x-ticks
    ax.set_xticks(radii_km[::2])
    ax.set_xticklabels(radii_km[::2], fontsize=10)

    # Landfall annotation at top right
    if landfall_index != -1 and landfall_index < N_steps:
        landfall_time_str = pd.to_datetime(track_times[landfall_index]).strftime("%m/%d %H:%M")
        landfall_text = f"Landfall: {landfall_time_str}"
        ax.text(
            0.75,   # x
            1.015,  # y
            landfall_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='left',
            fontsize=10,
            fontweight='bold',
            color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3)
        )

    # Legend if landfall line is plotted
    if landfall_index != -1 and landfall_index < N_steps:
        ax.legend(loc='upper right', fontsize=9)

    azimuthal_plot_path = os.path.join(output_dir, f"hov_{var_name}.png")
    plt.tight_layout()
    plt.savefig(azimuthal_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print('SAVED', 'az', 'Hovmoller plot', azimuthal_plot_path)
