"""
Download ERA5 single-level and pressure-level data for monsoon analysis.
~/.cdsapirc CDS credentials:
url: https://cds.climate.copernicus.eu/api
key: <api-key>
"""

import argparse
import calendar
import os
import sys
import time

import cdsapi

#  Domain
AREA = [30, 70, 7, 95]  # [North, West, South, East]

YEAR = 2025
#  Monsoon months
MONSOON_MONTHS = {
    "June": "06",
    "July": "07",
    "August": "08",
    "September": "09",
}

# Single-level variables organized by step type
# Instantaneous variables (time = 00:00 to 23:00, step = 0)
INSTANT_VARS = [
    "10m_u_component_of_wind",  # u10
    "10m_v_component_of_wind",  # v10
    "2m_temperature",  # t2m
    "2m_dewpoint_temperature",  # d2m
    "mean_sea_level_pressure",  # msl
    "surface_pressure",  # sp
    "soil_temperature_level_1",  # stl1
    "volumetric_soil_water_layer_1",  # swvl1
]

# Average variables (step = 0-1, average over previous hour)
AVG_VARS = [
    "mean_surface_latent_heat_flux",  # avg_slhf
    "mean_surface_sensible_heat_flux",  # avg_ishf
]

# Accumulated variables (step = 0-12, 12-24 for 12-hour accumulations)
ACCUM_VARS = [
    "total_precipitation",  # tp
]

#  Constant single-level fields (no time dimension)
CONSTANT_VARS = [
    "land_sea_mask",  # lsm
    "geopotential",  # z  (orography)
    "soil_type",  # slt
]

#  Pressure levels
PRESSURE_LEVELS = ["200", "300", "400", "500", "600", "700", "850"]

PRESSURE_LEVEL_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",  # omega (Pa/s)
    "vorticity",  # relative vorticity
    "relative_humidity",
    "specific_humidity",
    "temperature",
    "geopotential",
]

#  Hourly time steps
ALL_HOURS = [f"{h:02d}:00" for h in range(24)]  # 00:00, 01:00 ... 23:00

#  SAF monthly variables (for coupling index)
SAF_VARS = [
    "volumetric_soil_water_layer_1",  # swvl1
    "2m_dewpoint_temperature",         # d2m
    "surface_pressure",                # sp
]
SAF_MONTHS = ["06", "07", "08", "09"]  # JJAS

#  Output root (relative to where the script lives)
BASE_DIR = os.path.join("..", "Data")
SL_DIR = os.path.join(BASE_DIR, "single-levels", "ncfiles")
PL_DIR = os.path.join(BASE_DIR, "pressure-levels", "nc-files")
CONSTANTS_NC = os.path.join(SL_DIR, "lsm-soiltype-z.nc")


# Helper functions
def days_in_month(year: int, month_num: int) -> list[str]:
    """Return zero-padded day strings for a given year/month."""
    n = calendar.monthrange(year, month_num)[1]
    return [f"{d:02d}" for d in range(1, n + 1)]


def should_skip(path: str, skip_existing: bool) -> bool:
    if skip_existing and os.path.exists(path) and os.path.getsize(path) > 1024:
        print(f"[SKIP] already exists: {os.path.basename(path)}")
        return True
    return False


def download_with_retry(client, dataset, request, target, retries=3, wait=60):
    """Attempt download up to `retries` times with exponential back-off."""
    for attempt in range(1, retries + 1):
        try:
            client.retrieve(dataset, request, target)
            if os.path.exists(target):
                size_mb = os.path.getsize(target) / 1024**2
                print(f"[INFO]  {os.path.basename(target)}  ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            print(f"[ERROR]  Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                sleep_s = wait * attempt
                print(f"         Retrying in {sleep_s}s …")
                time.sleep(sleep_s)
    print(f"[FAIL] Gave up on {os.path.basename(target)}")
    return False


def download_single_level_step_type(
    client: cdsapi.Client,
    year: int,
    month_name: str,
    month_num: str,
    step_type: str,
    variables: list,
    skip_existing: bool,
):
    """Download single-level data for a specific step type (instant/avg/accum)."""
    month_dir = os.path.join(SL_DIR, str(year), f"{month_name}-{year}")
    os.makedirs(month_dir, exist_ok=True)

    # File naming matching your 2024 structure
    target = os.path.join(month_dir, f"data_stream-oper_stepType-{step_type}.nc")
    if should_skip(target, skip_existing):
        return

    print(f"[INFO] Downloading single-levels {month_name} {year} ({step_type})")

    # Build request based on step type
    request = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": str(year),
        "month": month_num,
        "day": days_in_month(year, int(month_num)),
        "time": ALL_HOURS,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    # Add step-specific parameters
    if step_type == "accum":
        # Accumulated fields: 12-hour accumulations (00-12 and 12-24)
        request["step"] = "12"
    elif step_type == "avg":
        # Average fields: average over the previous hour
        request["step"] = "0-1"
    # Instantaneous fields: no step parameter needed

    download_with_retry(
        client,
        "reanalysis-era5-single-levels",
        request,
        target,
    )


# Constants
def download_constants(client: cdsapi.Client, skip_existing: bool = True):
    """Download LSM, geopotential, and soil type - time-invariant fields."""
    os.makedirs(SL_DIR, exist_ok=True)
    if should_skip(CONSTANTS_NC, skip_existing):
        return

    print("\n[INFO] Downloading constants (lsm, z, slt) ")
    request = {
        "product_type": "reanalysis",
        "variable": CONSTANT_VARS,
        "year": YEAR,
        "month": "01",
        "day": "01",
        "time": "00:00",
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    download_with_retry(
        client,
        "reanalysis-era5-single-levels",
        request,
        CONSTANTS_NC,
    )


# Single-level monthly downloads
def download_single_level_month(
    client: cdsapi.Client,
    year: int,
    month_name: str,
    month_num: str,
    skip_existing: bool,
):
    """Download single-level data for a month, separated by step type."""
    # instantaneous variables
    if INSTANT_VARS:
        download_single_level_step_type(
            client, year, month_name, month_num, "instant", INSTANT_VARS, skip_existing
        )
    # average variables
    if AVG_VARS:
        download_single_level_step_type(
            client, year, month_name, month_num, "avg", AVG_VARS, skip_existing
        )
    # accumulated variables
    if ACCUM_VARS:
        download_single_level_step_type(
            client, year, month_name, month_num, "accum", ACCUM_VARS, skip_existing
        )


# Pressure-level monthly downloads
def download_pressure_level_month(
    client: cdsapi.Client,
    year: int,
    month_name: str,
    month_num: str,
    level_hpa: str,
    skip_existing: bool,
):
    month_dir = os.path.join(PL_DIR, str(year), f"{month_name}-{year}")
    os.makedirs(month_dir, exist_ok=True)

    # File naming: August-2024-850hPa.nc  - matches vortex_analysis.py expectation
    target = os.path.join(month_dir, f"{month_name}-{year}-{level_hpa}hPa.nc")
    if should_skip(target, skip_existing):
        return

    print(f"[INFO] Downloading pressure-levels {month_name} {year}  {level_hpa} hPa")
    request = {
        "product_type": "reanalysis",
        "variable": PRESSURE_LEVEL_VARS,
        "pressure_level": level_hpa,
        "year": str(year),
        "month": month_num,
        "day": days_in_month(year, int(month_num)),
        "time": ALL_HOURS,
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    download_with_retry(
        client,
        "reanalysis-era5-pressure-levels",
        request,
        target,
    )


# SAF monthly averaged data download
def download_saf_monthly_data(
    client: cdsapi.Client,
    start_year: int,
    end_year: int,
    skip_existing: bool = True,
):
    """
    Download monthly averaged ERA5 single-level data for the SAF coupling index.
    Variables: swvl1, d2m, sp.
    Months: JJAS.
    Saves one NetCDF file per year-range: saf_monthly_JJAS_{start}_{end}.nc
    """
    os.makedirs(SL_DIR, exist_ok=True)
    target = os.path.join(
        SL_DIR, f"saf_monthly_JJAS_{start_year}_{end_year}.nc"
    )
    if should_skip(target, skip_existing):
        return

    print(f"\n[INFO] Downloading SAF monthly averaged data ({start_year}–{end_year})")
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": SAF_VARS,
        "year": [str(y) for y in range(start_year, end_year + 1)],
        "month": SAF_MONTHS,
        "time": "00:00",
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    download_with_retry(
        client,
        "reanalysis-era5-single-levels-monthly-means",
        request,
        target,
    )


# Per-year orchestration
def download_year(client: cdsapi.Client, year: int, skip_existing: bool):
    print(f"\n[INFO] --- Processing Year {year} ---")
    for month_name, month_num in MONSOON_MONTHS.items():
        print(f"\n  --- {month_name} {year} ---")

        # Single levels - now downloads 3 separate files (instant, avg, accum)
        download_single_level_month(client, year, month_name, month_num, skip_existing)

        # Pressure levels - one request per level
        print(f"[INFO] Pressure-levels for {month_name} {year}:")
        for level in PRESSURE_LEVELS:
            download_pressure_level_month(
                client, year, month_name, month_num, level, skip_existing
            )


def parse_args():
    p = argparse.ArgumentParser(
        description="Download ERA5 monsoon data (single + pressure levels) and SAF monthly data."
    )
    p.add_argument(
        "--year",
        nargs="+",
        type=int,
        help="Year(s) to download hourly data, e.g. --year 2022 2023 2025",
    )
    p.add_argument(
        "--constants-only",
        action="store_true",
        help="Only download the time-invariant constants file (lsm, z, slt) then exit.",
    )
    p.add_argument(
        "--saf",
        action="store_true",
        help="Download monthly averaged data for SAF coupling index (swvl1, d2m, sp) for JJAS.",
    )
    p.add_argument(
        "--saf-years",
        nargs=2,
        type=int,
        default=[1991, 2020],
        metavar=("START", "END"),
        help="Year range for SAF monthly data (default: 1991 2020).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist and are non-empty (default: True).",
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    skip = args.skip_existing and not args.no_skip

    # If only SAF is requested, do that and exit.
    if args.saf:
        client = cdsapi.Client()
        download_saf_monthly_data(
            client, args.saf_years[0], args.saf_years[1], skip_existing=skip
        )
        print("\n[INFO] SAF monthly data download complete. Exiting.")
        sys.exit(0)

    if not args.constants_only and not args.year:
        print("[ERROR] provide --year, --constants-only, or --saf")
        sys.exit(1)

    t0 = time.time()
    client = cdsapi.Client()

    # Constants
    download_constants(client, skip_existing=skip)

    if args.constants_only:
        print("\n[INFO] Constants downloaded. Exiting.")
        sys.exit(0)

    for year in sorted(args.year):
        download_year(client, year, skip_existing=skip)

    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\nAll downloads finished in {h}h {m}m {s}s")
