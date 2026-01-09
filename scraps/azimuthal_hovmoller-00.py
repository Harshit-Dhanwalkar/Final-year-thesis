import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import numba
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from speed_analysis import is_point_land

VARIABLE_UNITS = {
    'u10': 'm/s',
    'v10': 'm/s',
    'wind10': 'm/s',  # wind10 = sqrt(u10^2 + v10^2)
    'speed_kmh': 'km/h',
    'speed_ms': 'm/s',
    'msl': 'Pa',
    'tp': 'mm/hr',
    't2m': '°C',
    'sp': 'hPa',
    'stl1': '°C',
    'swvl1': 'm³/m³',
    'avg_slhtf': 'W/m²',
    'avg_ishf': 'W/m²'
}

# list of variable for plotting
VARIABLE_NAMES = {
    'u10': 'Zonal Wind (10m)',
    'v10': 'Meridional Wind (10m)',
    'wind10': 'Wind Speed (10m)',
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

# Compute azimuthal average for one time & one radius
@numba.jit(nopython=True, parallel=True)
def calculate_distances(lat_grid, lon_grid, center_lat, center_lon):
    lat1 = np.deg2rad(center_lat)
    lon1 = np.deg2rad(center_lon)

    n_lat, n_lon = lat_grid.shape
    dists = np.zeros((n_lat, n_lon))

    for i in numba.prange(n_lat):
        for j in numba.prange(n_lon):
            lat2 = np.deg2rad(lat_grid[i, j])
            lon2 = np.deg2rad(lon_grid[i, j])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            dists[i, j] = 2 * EARTH_R * np.arcsin(np.sqrt(a))
    return dists

def circle_mean(ds, varname, center_lat, center_lon, radius_km, lat_grid, lon_grid):
    dists = calculate_distances(lat_grid, lon_grid, center_lat, center_lon)
    dr = max(10, radius_km * 0.15)  # band
    mask = (dists >= radius_km - dr) & (dists <= radius_km + dr)

    if not np.any(mask):
        return np.nan

    if varname == "wind10":
        # wind = sqrt(u10^2 + v10^2)
        if "u10" not in ds.data_vars or "v10" not in ds.data_vars:
            return np.nan
        u = ds["u10"].values
        v = ds["v10"].values
        field = np.sqrt(u**2 + v**2)
    else:
        if varname not in ds.data_vars:
            return np.nan
        field = ds[varname].values

    values = field[mask]
    valid_values = values[~np.isnan(values)]

    if len(valid_values) == 0:
        return np.nan
    return float(np.nanmean(valid_values))


def process_time_step(args):
    ds_t, latc, lonc, t_idx, radii_km, variables, lat_grid, lon_grid = args
    time_results = {}

    for ri, rk in enumerate(radii_km):
        for var in variables:
            value = circle_mean(ds_t, var, latc, lonc, rk, lat_grid, lon_grid)

            # Sign reversed of latent heat flux and sensible heat flux (surface to the atmosphere positive)
            if var in ['avg_slhtf', 'avg_ishf']:
                value = -value if not np.isnan(value) else np.nan

            time_results[(var, ri, t_idx)] = value
    return time_results


def compute_azimuthal_hovmoller(ds, track_df, radii_km, variables, max_workers=None):
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count() - 1, 4)

    track_df = track_df.sort_values("time").reset_index(drop=True)
    track_times = pd.to_datetime(track_df["time"]).values

   # Check which variables are available
    available_vars = []
    for var in variables:
        if var == "wind10":
            if "u10" in ds.data_vars and "v10" in ds.data_vars:
                available_vars.append(var)
        elif var in ds.data_vars:
            available_vars.append(var)

    if not available_vars:
        print("Warning: None of the requested variables are available in the dataset")
        return {}, track_times

    print(f"Computing Hovmöller for available variables: {available_vars}")

    lat_vals = ds.latitude.values
    lon_vals = ds.longitude.values
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    # Prepare data for each time step
    data_slices = []
    for t in range(len(track_df)):
        ds_t = ds.sel(time=track_times[t], method="nearest")
        latc = track_df["latitude"].iloc[t]
        lonc = track_df["longitude"].iloc[t]
        data_slices.append((ds_t, latc, lonc, t, radii_km, variables, lat_grid, lon_grid))

    # Results dictionary
    results = {var: np.zeros((len(radii_km), len(track_df))) * np.nan
               for var in variables}
    print(f"Computing Hovmöller with {max_workers} parallel workers...")

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_time_step, data): i
                  for i, data in enumerate(data_slices)}

        completed = 0
        for future in as_completed(futures):
            try:
                time_results = future.result()
                for (var, ri, t_idx), value in time_results.items():
                    results[var][ri, t_idx] = value
                completed += 1
                if completed % 5 == 0 or completed == len(data_slices):
                    print(f"  Progress: {completed}/{len(data_slices)} time steps")
            except Exception as e:
                print(f"Error processing time step: {e}")
                continue

    print(f"Completed Hovmöller computation for {len(available_vars)} variables")
    return results, track_times


def plot_hovmoller(hov_arr, track_times, radii_km, varname, track_df, outdir="hov_plots"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if varname in ['tp', 'wind10', 'avg_slhtf', 'avg_ishf']:
        cmap = 'plasma'
    elif varname in ['t2m', 'stl1']:
        cmap = 'coolwarm'
    elif varname in ['msl', 'sp']:
        cmap = 'viridis'
    elif varname == 'swvl1':
        cmap = 'YlGnBu'
    else:
        cmap = 'viridis'

    var_label = VARIABLE_NAMES.get(varname, varname)
    var_unit = VARIABLE_UNITS.get(varname, '')

    track_df = track_df.set_index('time').reindex(track_times).reset_index()

    if 'is_land' not in track_df.columns:
        print(f"Warning: 'is_land' column not found in track_df for variable {varname}")
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

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Use index positions for y-axis
    N_steps = len(track_times)
    y_min = 0
    y_max = N_steps - 0.5 if N_steps > 1 else 0.5

    # im = plt.imshow(
    #     hov_arr.T,
    #     aspect="auto",
    #     origin="lower",
    #     interpolation="nearest",
    #     extent=[radii_km[0], radii_km[-1], -0.5, y_max],
    #     cmap=cmap
    # )

    ### with contours
    X, Y = np.meshgrid(radii_km, np.arange(N_steps))
    im = ax.contourf(
         X, Y, hov_arr.T,
         levels=20,
         cmap=cmap,
         extend='both'
    )

    # contour lines
    contours = ax.contour(
         X, Y, hov_arr.T,
         levels=10,
         colors='black',
         linewidths=0.5,
         alpha=0.5
    )
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')
    ###

    # --- Landfall Line ---
    if landfall_index != -1 and landfall_index < N_steps:
        ax.axhline(
            y=landfall_index,
            color='black',
            linestyle='--',
            linewidth=2.5,
            alpha=0.9,
        )

    cbar = plt.colorbar(im, pad=0.02)
    label = f"{var_label} ({var_unit})" if var_unit else var_label
    cbar.set_label(label, fontsize=12)

    plt.xlabel("Radius from center (km)", fontsize=12)
    plt.ylabel("Time (hr)", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.title(f"Hovmöller Plot for {var_label}", fontsize=14, fontweight='bold')

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

    # Add landfall annotation
    if landfall_index != -1:
        landfall_time_str = pd.to_datetime(track_times[landfall_index]).strftime("%m/%d %H:%M")
        landfall_text = f"Landfall: {landfall_time_str}"
        ax.text(
            0.98, 1,
            landfall_text,
            transform=plt.gca().transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            fontsize=8,
            fontweight='bold',
            color='black',
        )

    outfile = os.path.join(outdir, f"hov_{varname}.png")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Hovmöller plot: {outfile}")
