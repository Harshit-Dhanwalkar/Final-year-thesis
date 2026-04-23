import os
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = "../Data/single-levels/ncfiles/"

print("--- Checking Available Years ---")
years = sorted(
    [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
)
print("Available years:", ", ".join(years))
print()

lsm_soiltype_z_path = os.path.join(BASE_DIR, "lsm-soiltype-z.nc")
if os.path.exists(lsm_soiltype_z_path):
    print(f"--- Land-Sea, Soil Type and Geopotential Mask: {lsm_soiltype_z_path} ---")
    try:
        ds_lsm = xr.open_dataset(lsm_soiltype_z_path)
        print(f"  Variables : {list(ds_lsm.data_vars)}")
        print(
            f"  Latitude  : {float(ds_lsm['latitude'].min()):.2f}° – {float(ds_lsm['latitude'].max()):.2f}°"
        )
        print(
            f"  Longitude : {float(ds_lsm['longitude'].min()):.2f}° – {float(ds_lsm['longitude'].max()):.2f}°"
        )
        ds_lsm.close()
    except Exception as e:
        print(f"  [ERROR] {e}")
print()

for year in years:
    YEAR_DIR = os.path.join(BASE_DIR, year)
    print(f"  Year: {year}")

    months = sorted(
        [d for d in os.listdir(YEAR_DIR) if os.path.isdir(os.path.join(YEAR_DIR, d))]
    )
    if not months:
        print("  [WARN] No month subdirectories found")
        continue
    print(f"  Available months: {', '.join(months)}")

    for month in months:
        DATA_DIR = os.path.join(YEAR_DIR, month)
        print(f"--- Month: {month} ---")

        ncfiles = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".nc")])
        if not ncfiles:
            print(f"  [WARN] No NetCDF files found in {DATA_DIR}")
            continue

        for filename in ncfiles:
            file_path = os.path.join(DATA_DIR, filename)
            print(f"  ---  File: {filename} ---")
            try:
                ds = xr.open_dataset(file_path)

                time_coord = "valid_time" if "valid_time" in ds.coords else "time"
                if time_coord not in ds.coords:
                    print("    [ERROR] No time coordinate found")
                    continue

                t_min = pd.to_datetime(ds[time_coord].min().item())
                t_max = pd.to_datetime(ds[time_coord].max().item())
                n_steps = len(ds[time_coord])
                print(f"    Time period  : {t_min}  ->  {t_max}")
                print(f"    Time steps   : {n_steps}")

                if n_steps >= 2:
                    dt_h = (
                        pd.to_timedelta(
                            ds[time_coord].diff(dim=time_coord).mean().item()
                        ).total_seconds()
                        / 3600
                    )
                    print(f"    Resolution   : {dt_h:.1f} h")

                print(
                    f"    Latitude     : {ds['latitude'].min().item():.2f}° – {ds['latitude'].max().item():.2f}°"
                )
                print(
                    f"    Longitude    : {ds['longitude'].min().item():.2f}° – {ds['longitude'].max().item():.2f}°"
                )

                print(f"    Variables ({len(ds.data_vars)}):")
                for var_name, var_data in ds.data_vars.items():
                    units = var_data.attrs.get("units", "?")
                    ln = var_data.attrs.get(
                        "long_name", var_data.attrs.get("standard_name", "")
                    )
                    print(
                        f"      {var_name:<10} [{units}]" + (f"  - {ln}" if ln else "")
                    )

                ds.close()

            except Exception as e:
                print(f"    [ERROR] {e}")
