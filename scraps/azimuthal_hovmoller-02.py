import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import numba
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from utils.land_utils import is_point_land

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
    't2m': 'Â°C',
    'sp': 'hPa',
    'stl1': 'Â°C',
    'swvl1': 'mÂ³/mÂ³',
    'avg_slhtf': 'W/mÂ²',
    'avg_ishf': 'W/mÂ²'
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
    'avg_ishf': 'Surface Sensible Heat Flux'
}

# Constants
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

def compute_azimuthal_hovmoller(
    ds,
    track_df,
    radii_km,
    variables,
    n_theta=72,
):
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

        # Check for Landfall
        if landfall_index == -1 and is_point_land(lon_c, lat_c):
            landfall_index = i

        # Select time slice
        ds_t = ds.sel(time=t, method='nearest')

        for j, r in enumerate(radii_km):
            # 1. Generate Polar Grid Points for radius r
            lons_p, lats_p, phis = get_polar_points(lon_c, lat_c, r, n_theta)

            # 2. Interpolate ERA5 data onto polar points
            interp_ds = ds_t.interp(
                longitude=xr.DataArray(lons_p),
                latitude=xr.DataArray(lats_p),
                method='linear'
            )

            # 3. Calculate Base Variables
            for var in variables:
                if var in interp_ds:
                    hov_results[var][i, j] = interp_ds[var].mean().values
                elif var == 'wind10':
                    w10 = np.sqrt(interp_ds['u10']**2 + interp_ds['v10']**2)
                    hov_results[var][i, j] = w10.mean().values

            # 4. Calculate Radial and Tangential Wind Rotation
            if 'u10' in interp_ds and 'v10' in interp_ds:
                u = interp_ds['u10'].values
                v = interp_ds['v10'].values

                ## Vr = u*cos(phi) + v*sin(phi) [positive = outward]
                ## Vt = -u*sin(phi) + v*cos(phi) [positive = Cyclonic]
                vr = u * np.cos(phis) + v * np.sin(phis)
                vt = -u * np.sin(phis) + v * np.cos(phis)

                if 'vr10' in hov_results: hov_results['vr10'][i, j] = np.mean(vr)
                if 'vt10' in hov_results: hov_results['vt10'][i, j] = np.mean(vt)

    return hov_results, track_times, landfall_index


def plot_hovmoller(
    hov_arr,
    track_times,
    radii_km,
    var_name,
    track_df,
    outdir="hov_plots",
    landfall_index=-1,
):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if var_name in ['tp', 'wind10', 'avg_slhtf', 'avg_ishf']:
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
        print(f"Warning: 'is_land' column not found in track_df for variable {var_name}")
        # Try to compute land status if not available
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
    im = ax.contourf(
         X, Y, hov_arr,
         levels=20,
         cmap=cmap,
         extend='both'
    )
    contours = ax.contour(
         X, Y, hov_arr,
         levels=10,
         colors='black',
         linewidths=0.5,
         alpha=0.5
    )
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

    plt.xlabel("Radius from center (km)", fontsize=12)
    plt.ylabel("Time (hr)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.title(f"HovmÃ¶ller Plot for {var_label}", fontsize=14, fontweight='bold')

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
        ytick_labels = [pd.to_datetime(track_times[i]).strftime('%m/%d\n%H:%M') 
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
            0.98, 1.02,
            landfall_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            fontsize=10,
            fontweight='bold',
            color='black',
        )

    # Add legend if landfall line is plotted
    if landfall_index != -1 and landfall_index < N_steps:
        ax.legend(loc='upper right', fontsize=9)

    outfile = os.path.join(outdir, f"hov_{var_name}.png")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    # print(f" >> [Saved   ] HovmÃ¶ller plot: {outfile}")
