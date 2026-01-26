import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import sys
import time

import traceback

from utils.land_utils import is_point_land
from utils.ocean_masking import mask_swvl1_over_ocean

VARIABLE_UNITS = {
    'u10': 'm/s',
    'v10': 'm/s',
    'wind10': 'm/s',     # wind10 = sqrt(u10^2 + v10^2)
    'vr10': 'm/s',       # Radial Wind
    'vt10': 'm/s',       # Tangential Wind
    'speed_kmh': 'km/h',
    'speed_ms': 'm/s',
    'msl': 'Pa',
    'tp': 'mm/hr',
    't2m': 'Â°C',
    'sp': 'hPa',
    'stl1': 'Â°C',
    'swvl1': 'mÂ³/mÂ³',
    'avg_slhtf': 'W/mÂ²',
    'avg_ishf': 'W/mÂ²',
    'total_hf': 'W/mÂ²'   # Total surface heat flux = avg_ishf + avg_slhtf
}

# List of variable for plotting
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
    """
    Compute azimuthal HovmÃ¶ller diagrams for a given track.
        DataFrame containing track coordinates with columns:
        'time', 'latitude', 'longitude', 'track_id'
        Radii (in kilometers) to extract data
        n_theta : Number of azimuthal angles (default: 72)
    """
    track_times = track_df['time'].values
    track_times = pd.to_datetime(track_times)
    N_steps = len(track_times)
    N_radii = len(radii_km)

    # Initialize results
    hov_results = {}
    for v in variables:
        hov_results[v] = np.zeros((N_steps, N_radii))

    derived_vars = ['vr10', 'vt10', 'wind10', 'total_hf']
    for dv in derived_vars:
        if dv not in hov_results:
            hov_results[dv] = np.zeros((N_steps, N_radii))

    landfall_index = -1
    total_steps = N_steps * len(radii_km)
    current_step = 0

    spinner = ["â—œ", "â—", "â—ž", "â—Ÿ"]
    dots_sequence = [" ", ". ", ".. ", "..."]
    spinner_index = 0
    for i, t in enumerate(track_times):
        if i % 5 == 0:
            current_spinner = spinner[spinner_index % len(spinner)]
            current_dots = dots_sequence[spinner_index % len(dots_sequence)]
            sys.stdout.write(f"\r{current_spinner} Processing time step {i+1}/{N_steps}{current_dots}")
            sys.stdout.flush()
            time.sleep(0.1)
            spinner_index += 1

        row = track_df.iloc[i]
        lon_c, lat_c = row['longitude'], row['latitude']

        # Check for Landfall
        if landfall_index == -1 and is_point_land(lon_c, lat_c):
            landfall_index = i
        # Select time slice
        try:
            ds_t = ds.sel(time=t, method='nearest')
        except Exception as e:
            t_str = pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')
            print(f" >  [WARNING ] 1 Failed at time {t_str} : {e}")
            continue

        for j, r in enumerate(radii_km):
            try:
                # 1. Generate Polar Grid Points for radius r
                lons_p, lats_p, phis = get_polar_points(lon_c, lat_c, r, n_theta)
                # Check if polar points are within dataset bounds
                lon_min = float(ds['longitude'].min())
                lon_max = float(ds['longitude'].max())
                lat_min = float(ds['latitude'].min())
                lat_max = float(ds['latitude'].max())
                # Filter out-of-bounds points
                valid_mask = (
                    (lons_p >= lon_min) & (lons_p <= lon_max) &
                    (lats_p >= lat_min) & (lats_p <= lat_max)
                )

                if not np.any(valid_mask):
                    continue

                # 2. Interpolate Dataset to Polar Points
                # Filter to valid points only
                lons_p_valid = lons_p[valid_mask]
                lats_p_valid = lats_p[valid_mask]
                phis_valid = phis[valid_mask]
                n_valid_points = len(lons_p_valid)

                interp_ds = ds_t.interp(
                    coords={'longitude': lons_p_valid, 'latitude': lats_p_valid},
                    method='nearest'
                )
                
                # Ensure all variables have exactly n_valid_points
                # Sometimes xarray can add extra dimensions, so we need to handle this
                for var in list(interp_ds.data_vars):
                    if interp_ds[var].ndim > 1:
                        # Multi-dimensional - select first time slice if present
                        interp_ds[var] = interp_ds[var].isel({d: 0 for d in interp_ds[var].dims if d != 'longitude' and d != 'latitude'}, drop=False)
                    # Ensure correct size
                    try:
                        data_check = interp_ds[var].values.flatten()
                        if len(data_check) != n_valid_points:
                            # This shouldn't happen, but if it does, we skip this radius
                            raise ValueError(f"Size mismatch for {var}: expected {n_valid_points}, got {len(data_check)}")
                    except:
                        pass

                # 3. Calculate base Variables
                for var in variables:
                    if var in ['vr10', 'vt10', 'wind10', 'total_hf']:
                        # Skip derived variables here, handle separately
                        continue
                    if var not in interp_ds.data_vars:
                        continue

                    if var == 'swvl1':
                        # Extract only land pixels
                        swvl1_data = interp_ds['swvl1'].values
                        # Ensure it's 1D
                        swvl1_data = np.asarray(swvl1_data).flatten()
                        # Trim to n_valid_points to match grid
                        swvl1_data = swvl1_data[:n_valid_points]
                        
                        valid_land_data = []  # Collect only masked-valid land values
                        for k in range(min(len(swvl1_data), len(lats_p_valid), len(lons_p_valid))):
                            value = float(swvl1_data[k])  # Convert to Python scalar
                            lat_val = float(lats_p_valid[k])
                            lon_val = float(lons_p_valid[k])
                            # Apply full ocean masking
                            masked_value = mask_swvl1_over_ocean(value, lat_val, lon_val, strict_mode=True)
                            if not np.isnan(masked_value):
                                valid_land_data.append(masked_value)
                            else:
                                print(f" [MASKED ] Polar point ({lat_val:.2f}N, {lon_val:.2f}E) at r={r}km: Original SWVL1={value:.4f}")
                        # Require at least half the points to be valid land for a reliable mean
                        if len(valid_land_data) >= n_theta // 2:
                            hov_results['swvl1'][i, j] = float(np.nanmean(valid_land_data))
                        else:
                            t_str = pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')
                            hov_results['swvl1'][i, j] = np.nan
                            print(f" >  [WARNING ] Insufficient valid land points at t={t_str}, r={r}km: {len(valid_land_data)}/{n_theta}")
                    else:
                        # Non-swvl1 vars - extract and flatten properly
                        data_values = interp_ds[var].values
                        # Flatten to ensure 1D array
                        data_values = np.asarray(data_values).flatten()
                        if len(data_values) > 0:
                            val = float(np.nanmean(data_values))
                            # Use np.isnan on scalar result
                            if not np.isnan(val):
                                hov_results[var][i, j] = val

                # 4. Handle derived variables - ONLY if requested
                if 'wind10' in variables:
                    if 'u10' in interp_ds.data_vars and 'v10' in interp_ds.data_vars:
                        u_vals = interp_ds['u10'].values
                        v_vals = interp_ds['v10'].values
                        # Ensure 1D arrays
                        u_vals = np.asarray(u_vals).flatten()
                        v_vals = np.asarray(v_vals).flatten()
                        # Trim to n_valid_points if needed
                        u_vals = u_vals[:n_valid_points]
                        v_vals = v_vals[:n_valid_points]
                        
                        # Check length alignment
                        if len(u_vals) > 0 and len(v_vals) > 0 and len(u_vals) == len(phis_valid) == len(v_vals):
                            w10 = np.sqrt(u_vals**2 + v_vals**2)
                            val = float(np.nanmean(w10))
                            # Check for NaN
                            if not np.isnan(val):
                                hov_results['wind10'][i, j] = val
                        else:
                            hov_results['wind10'][i, j] = np.nan
                    else:
                        hov_results['wind10'][i, j] = np.nan

                if 'total_hf' in variables:
                    if 'avg_slhtf' in interp_ds.data_vars and 'avg_ishf' in interp_ds.data_vars:
                        slhtf_vals = interp_ds['avg_slhtf'].values
                        ishf_vals = interp_ds['avg_ishf'].values
                        # Ensure 1D arrays
                        slhtf_vals = np.asarray(slhtf_vals).flatten()
                        ishf_vals = np.asarray(ishf_vals).flatten()
                        # Trim to n_valid_points if needed
                        slhtf_vals = slhtf_vals[:n_valid_points]
                        ishf_vals = ishf_vals[:n_valid_points]
                        
                        # Check length alignment
                        if len(slhtf_vals) > 0 and len(ishf_vals) > 0 and len(slhtf_vals) == len(phis_valid) == len(ishf_vals):
                            hf_total = slhtf_vals + ishf_vals
                            val = float(np.nanmean(hf_total))
                            if not np.isnan(val):
                                hov_results['total_hf'][i, j] = val
                        else:
                            hov_results['total_hf'][i, j] = np.nan
                    else:
                        hov_results['total_hf'][i, j] = np.nan

                # 5. Calculate Radial and Tangential Wind - ONLY if requested
                if 'vr10' in variables or 'vt10' in variables:
                    if 'u10' in interp_ds.data_vars and 'v10' in interp_ds.data_vars:
                        u = interp_ds['u10'].values
                        v = interp_ds['v10'].values
                        # Ensure 1D arrays
                        u = np.asarray(u).flatten()
                        v = np.asarray(v).flatten()
                        # Trim to n_valid_points if needed
                        u = u[:n_valid_points]
                        v = v[:n_valid_points]
                        
                        # Check length alignment - must match phis_valid length
                        if (len(u) > 0 and len(v) > 0 and 
                            len(u) == len(phis_valid) and len(v) == len(phis_valid)):
                            # Filter out NaN values
                            valid_wind_mask = ~(np.isnan(u) | np.isnan(v))
                            if np.any(valid_wind_mask):
                                # Apply mask to all arrays
                                u_valid = u[valid_wind_mask]
                                v_valid = v[valid_wind_mask]
                                phis_wind_valid = phis_valid[valid_wind_mask]
                                
                                # Vr = u*cos(phi) + v*sin(phi) [positive = outward]
                                # Vt = -u*sin(phi) + v*cos(phi) [positive = Cyclonic]
                                vr = u_valid * np.cos(phis_wind_valid) + v_valid * np.sin(phis_wind_valid)
                                vt = -u_valid * np.sin(phis_wind_valid) + v_valid * np.cos(phis_wind_valid)
                                
                                if 'vr10' in variables:
                                    vr_mean = float(np.nanmean(vr))
                                    if not np.isnan(vr_mean):
                                        hov_results['vr10'][i, j] = vr_mean
                                    else:
                                        hov_results['vr10'][i, j] = np.nan
                                if 'vt10' in variables:
                                    vt_mean = float(np.nanmean(vt))
                                    if not np.isnan(vt_mean):
                                        hov_results['vt10'][i, j] = vt_mean
                                    else:
                                        hov_results['vt10'][i, j] = np.nan
                            else:
                                if 'vr10' in variables:
                                    hov_results['vr10'][i, j] = np.nan
                                if 'vt10' in variables:
                                    hov_results['vt10'][i, j] = np.nan
                        else:
                            if 'vr10' in variables:
                                hov_results['vr10'][i, j] = np.nan
                            if 'vt10' in variables:
                                hov_results['vt10'][i, j] = np.nan
                    else:
                        if 'vr10' in variables:
                            hov_results['vr10'][i, j] = np.nan
                        if 'vt10' in variables:
                            hov_results['vt10'][i, j] = np.nan

            except Exception as e:
                t_str = pd.Timestamp(t).strftime('%Y-%m-%d %H:%M:%S')
                print(f" >  [WARNING ] Failed at time {t_str}, radius {r}: {e}")
                # Mark this cell as NaN
                for v in hov_results.keys():
                    hov_results[v][i, j] = np.nan
                continue

            current_step += 1
            if current_step % 10 == 0:  # Every 10 radius-time
                print(f" >  [PROGRESS ] HovmÃ¶ller: {current_step}/{total_steps} steps completed")
    # Clear the spinner
    sys.stdout.write("\r" + " " * 50 + "\r")
    sys.stdout.flush()

    return hov_results, track_times, landfall_index


def plot_hovmoller(
    hov_results,
    track_df,
    track_id,
    track_times,
    radii_km,
    var_name,
    outdir="hov_plots",
):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Type check
    if not isinstance(hov_results, dict):
        print(f" >  [ERROR   ] hov_results is not a dictionary  :   {type(hov_results)}")
        return
    # Variable check
    if var_name not in hov_results:
        print(f" >  [WARNING ] Variable '{var_name}' not in hov_results")
        print(f"             Available variables                :   {list(hov_results.keys())}")
        return
    # Select colormap
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

    track_times = pd.to_datetime(track_times)
    hov_arr = hov_results[var_name]
    var_label = VARIABLE_NAMES.get(var_name, var_name.upper())
    var_unit = VARIABLE_UNITS.get(var_name, '')
    # Reindex track_df to match track_times
    try:
        track_df = track_df.set_index('time').reindex(track_times).reset_index()
    except Exception as e:
        print(f" >  [WARNING ] Could not reindex track_df: {e}")
    # Find landfall
    land_status = [is_point_land(row['latitude'], row['longitude']) for _, row in track_df.iterrows()]
    landfall_index = -1
    for idx, status in enumerate(land_status):
        if status and landfall_index == -1:
            landfall_index = idx
            break
    if landfall_index == -1 and any(land_status):
        landfall_index = land_status.index(True)
    elif not land_status[0] and all(not s for s in land_status):
        landfall_index = len(land_status) - 1
    elif land_status[0] and not all(land_status):
        landfall_index = 0
    elif land_status[0] and all(land_status):
        landfall_index = 0

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

    plt.xlabel("Radial distance from center of MD (km)", fontsize=11)
    plt.ylabel("Time (hr)", fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.title(f"{var_label}", fontsize=13, fontweight='bold')

    # Set y-ticks to show real datetime
    if N_steps <= 20:
        # Show all time steps
        ytick_positions = np.arange(N_steps)
        ytick_labels = [pd.Timestamp(t).strftime('%m/%d\n%H:%M') for t in track_times]
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, rotation=45, fontsize=8)
    else:
        # Show subset of time points
        step = max(1, N_steps // 10)
        ytick_positions = np.arange(0, N_steps, step)
        ytick_labels = []
        for i in np.arange(0, N_steps, step):
            idx = int(i)
            if idx < len(track_times):
                label = pd.Timestamp(track_times[idx]).strftime('%m/%d\n%H:%M')
                ytick_labels.append(label)
            else:
                ytick_labels.append('')
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
            1.014,  # y
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

    outfile = os.path.join(outdir, f"hov_{var_name}.png")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" >> [SAVED   ] Hovmoller plot                    :         {outfile}")
