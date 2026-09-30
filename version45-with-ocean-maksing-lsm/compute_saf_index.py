import os

from saf_coupling import compute_saf_index_from_file
from utils import std_print

MONTHLY_NC = os.path.join(
    "..", "Data", "single-levels", "ncfiles", "saf_monthly_JJAS_1991_2020.nc"
)
LSM_NC = os.path.join(
    "..", "Data", "single-levels", "ncfiles", "lsm-soiltype-z.nc"
)
OUT_NC = os.path.join(
    "..", "Data", "single-levels", "ncfiles", "saf_index_JJAS_1991_2020.nc"
)

def main():
    if not os.path.exists(MONTHLY_NC):
        std_print("ERROR", "saf", "Monthly file not found", MONTHLY_NC)
        return

    std_print("INFO", "saf", "Computing SAF coupling index …")
    saf = compute_saf_index_from_file(MONTHLY_NC, lsm_nc=LSM_NC, months=(6, 7, 8, 9))
    if saf is None:
        std_print("ERROR", "saf", "SAF index computation failed")
        return

    saf.to_netcdf(OUT_NC)
    std_print("SAVED", "saf", "SAF index NetCDF", OUT_NC)

if __name__ == "__main__":
    main()
