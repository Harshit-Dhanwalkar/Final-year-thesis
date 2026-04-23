import os
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = "../Data/pressure-levels/nc-files/"
EXPECTED_LEVELS = [200, 300, 400, 500, 600, 700, 850]

months = sorted(
    [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
)
print("Available months:", ", ".join(months))

for month in months:
    DATA_DIR = os.path.join(BASE_DIR, month)
    print(f"\n--- Month: {month} ---")

    ncfiles = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".nc")])
    if not ncfiles:
        print(f"  [WARN] No NetCDF files found in {DATA_DIR}")
        continue

    # Check which pressure levels are present
    found_levels = []
    for f in ncfiles:
        for lvl in EXPECTED_LEVELS:
            if f"{lvl}hPa" in f:
                found_levels.append(lvl)
    missing = sorted(set(EXPECTED_LEVELS) - set(found_levels))
    if missing:
        print(f"  [WARN] Missing levels: {missing}")
    else:
        print(f"  All expected levels present: {sorted(found_levels)}")

    for filename in ncfiles:
        file_path = os.path.join(DATA_DIR, filename)
        print(f"  --- File: {filename} ---")
        try:
            ds = xr.open_dataset(file_path)

            # Time coordinate
            time_coord = "valid_time" if "valid_time" in ds.coords else "time"
            if time_coord not in ds.coords:
                print("  [ERROR] No time coordinate found")
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

            # Spatial
            lat_min = float(ds["latitude"].min())
            lat_max = float(ds["latitude"].max())
            lon_min = float(ds["longitude"].min())
            lon_max = float(ds["longitude"].max())
            lat_res = float(abs(ds["latitude"].diff("latitude").mean()))
            lon_res = float(abs(ds["longitude"].diff("longitude").mean()))
            print(
                f"    Latitude     : {lat_min:.2f}° – {lat_max:.2f}°  (Δ{lat_res:.2f}°)"
            )
            print(
                f"    Longitude    : {lon_min:.2f}° – {lon_max:.2f}°  (Δ{lon_res:.2f}°)"
            )

            # Pressure level coordinate
            for coord_name in ["pressure_level", "level", "plev", "isobaricInhPa"]:
                if coord_name in ds.coords or coord_name in ds.dims:
                    lvl_vals = ds[coord_name].values
                    print(f"    Pressure lvl : {lvl_vals} hPa  (coord='{coord_name}')")
                    break
            else:
                # Try to infer from filename
                for lvl in EXPECTED_LEVELS:
                    if f"{lvl}hPa" in filename:
                        print(f"    Pressure lvl : {lvl} hPa  (inferred from filename)")
                        break

            # Variables
            print(f"    Variables ({len(ds.data_vars)}):")
            for var_name, var_data in ds.data_vars.items():
                attrs = var_data.attrs
                units = attrs.get("units", "?")
                ln = attrs.get("long_name", attrs.get("standard_name", ""))
                has_nan = bool(np.isnan(var_data.values).any())
                vmin = float(np.nanmin(var_data.values))
                vmax = float(np.nanmax(var_data.values))
                print(
                    f"      {var_name:<6}  [{units:<12}]  "
                    f"range [{vmin:>12.4f}, {vmax:>12.4f}]  "
                    f"NaN={has_nan}" + (f"  - {ln}" if ln else "")
                )
            ds.close()
        except Exception as e:
            print(f"  [ERROR] {e}")
