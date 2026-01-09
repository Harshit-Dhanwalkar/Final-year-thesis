import os

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = "Data/single-levels/ncfiles/"
print("--- Checking Available Months ---")
months = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
print("Available months:", ", ".join(sorted(months)))

for month in sorted(months):
    DATA_DIR = os.path.join(BASE_DIR, month)
    print(f"\n=== Processing {month} ===")
    if not os.path.exists(DATA_DIR):
        print(
            f"Error: Directory '{DATA_DIR}' not found. Please ensure it exists and contains NetCDF files."
        )
    else:
        ncfiles = [f for f in os.listdir(DATA_DIR) if f.endswith(".nc")]

        if not ncfiles:
            print(f"No NetCDF files found in '{DATA_DIR}'.")

        for filename in sorted(ncfiles):
            file_path = os.path.join(DATA_DIR, filename)
            try:
                ds = xr.open_dataset(file_path)

                print(f"--- File: {file_path} ---")

                # Check for 'valid_time' or 'time'
                time_coord = "valid_time" if "valid_time" in ds.coords else "time"
                if time_coord not in ds.coords:
                    print("Error: Time coordinate ('valid_time' or 'time') not found.")
                    continue

                start_time_stamp = ds[time_coord].min().item()
                end_time_stamp = ds[time_coord].max().item()

                start_time_readable = pd.to_datetime(start_time_stamp)
                end_time_readable = pd.to_datetime(end_time_stamp)

                print(f"  Time period: {start_time_readable} to {end_time_readable}")

                # --- Check Temporal Resolution ---
                if len(ds[time_coord]) >= 2:
                    time_diff = ds[time_coord].diff(dim=time_coord).mean().item()
                    time_resolution = pd.to_timedelta(time_diff).total_seconds() / 3600
                    print(f"  Temporal Resolution: {time_resolution:.1f} hours")
                else:
                    print("  Temporal Resolution: Not enough time steps to calculate difference.")

                # --- Check Spacial Resolution ---
                print(
                    f"  Latitude range: {ds['latitude'].min().item()} to {ds['latitude'].max().item()}"
                )
                print(
                    f"  Longitude range: {ds['longitude'].min().item()} to {ds['longitude'].max().item()}"
                )

                # --- Data Variables ---
                print("  --- Data Variables ---")
                if ds.data_vars:
                    for var_name, var_data in ds.data_vars.items():
                        print(f"  Variable: {var_name}")
                        # print(f"    Dimensions: {var_data.dims}")

                        # try:
                        #     data_array = var_data.values
                        #
                        #     if data_array.ndim == 0:
                        #         print(f"    First Value: {data_array.item():.4f}")
                        #     elif data_array.size > 0:
                        #         first_slice = data_array.flat[:5].tolist()
                        #         formatted_slice = [f"{x:.4f}" for x in first_slice]
                        #         print(
                        #             f"    Sample Values (first 5): {formatted_slice}..."
                        #         )
                        #     else:
                        #         print("    Sample Values: [Empty data array]")
                        # except Exception as e:
                        #     print(f"    Error fetching sample values: {e}")
                else:
                    print("  No data variables found in this file.")
            except FileNotFoundError:
                print(f"Error: File not found at {file_path}. Skipping.")
            except KeyError as e:
                print(
                    f"Error: Missing expected coordinate ('latitude', 'longitude', or time coord) in {file_path}: {e}. Skipping."
                )
            except Exception as e:
                print(
                    f"An unexpected error occurred while processing {file_path}: {e}. Skipping."
                )
