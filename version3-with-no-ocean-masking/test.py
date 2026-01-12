# test_heatflux.py
from heatflux_analysis import extract_heat_flux_timeseries, plot_heat_flux_timeseries
import xarray as xr
import numpy as np
import pandas as pd

# Create mock dataset with heat flux variables
track_id = 1
n_points = 48
times = pd.date_range('2024-08-01', periods=n_points, freq='h')
lats = np.linspace(10, 30, 21)
lons = np.linspace(65, 95, 31)

# Add heat flux variables to dataset
ds = xr.Dataset({
    'avg_slhtf': xr.DataArray(np.random.randn(n_points, len(lats), len(lons)), 
                              dims=['time', 'latitude', 'longitude']),
    'avg_ishf': xr.DataArray(np.random.randn(n_points, len(lats), len(lons)), 
                             dims=['time', 'latitude', 'longitude']),
    'latitude': lats,
    'longitude': lons,
    'time': times
})

track_df = pd.DataFrame({
    'time': times,
    'latitude': np.linspace(15, 25, n_points),
    'longitude': np.linspace(75, 85, n_points),
    'is_land': [False] * 24 + [True] * 24,
    'track_id': [1] * n_points
})

print("Testing heat flux extraction...")
flux_df = extract_heat_flux_timeseries(ds, track_df, track_id, buffer_km=100)

if flux_df is not None:
    print(f"Heat flux data extracted: {len(flux_df)} points")
    print(flux_df.head())
    
    # Test plotting
    plot_heat_flux_timeseries(flux_df, track_df, track_id, './test_output')
    print("Heat flux plots should be generated in ./test_output")
else:
    print("Failed to extract heat flux data")
