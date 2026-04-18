# NOTE:
# Soil moisture analysis - tp, swvl1, sm deficit, lead-lag, spatial maps
# - Analyzes the relationship between Total Precipitation (TP) and
#   Volumetric Soil Water Layer-1 (SWVL1, 0–7 cm) along and around MD tracks.
# 1. plot_spatial_tp_swvl1  [plot_spatial_tp_swvl1]
#     Two separate spatial maps over the MD track domain:
#     a) Total TP accumulated during passage (mm), with white/cyan track overlay.
#     b) Mean SWVL1 with a near-saturation contour overlay (red dashed = 90% of SWVL1_SAT).
#     - Low deficit -> soil near saturation -> LHF peaks -> Brown Ocean.
#     - High deficit -> soil absorbs rain -> no BOE
#     Reference: seneviratne2010investigating:  deficit = SWVL1_sat - SWVL1
# 2. SM Deficit Timeseries  [plot_sm_deficit_timeseries]
#     Single-panel SWVL1 timeseries with:
#       - Saturation line (SWVL1_SAT = 0.45 m^3/m^3) - saturation = deficit
#       - Dry threshold (0.20 m^3/m^3) - sienna fill below = definitely no BOE
#       - Land/ocean shading + landfall marker
#     Physical basis (Seneviratne et al. 2010):
#       - Low deficit (<0.05 m^3/m^3) -> near-saturated -> LHF peaks -> BOE conditions
#       - High deficit (>0.15 m^3/m^3) -> soil absorbs rain -> no BOE, faster decay
#     SWVL1_SAT = 0.45 m^3/m^3 (ERA5 approximate saturation for loamy/clay soils over India)
# 3. Soil Moisture Memory  [analyze_soil_memory_by_track, compute_soil_moisture_memory]
#     Autocorrelation of SWVL1 at 24-hour lag using statsmodels ACF.
#     Higher autocorrelation = longer memory = more sustained land-atmosphere feedback.
#     Minimum 48 data points required (= 48 hours of hourly ERA5).
#     Exports per-track memory index to soil_memory_analysis.csv.
# 4. SM Deficit Helper  [compute_sm_deficit]
#     deficit = clip(SWVL1_sat − SWVL1, 0)  [m^3/m^3]
#     Used internally for diagnostics and threshold checking.
# 5. SWVL1–TP Lead-Lag Cross-Correlation  [compute_swvl1_tp_lead_lag]
#     Cross-correlation (CCF) between SWVL1 and TP at lags (−max_lag, +max_lag) hours,
#     stratified by elevation band (plains <200m, rolling 200–600m, elevated >600m).
#     Positive lag CCF peak at 6–24 h = SWVL1 leads TP = BOE pre-moistening feedback.
#     Disrupted feedback over elevated terrain = orographic drainage effect.
#     Requires 'elevation_m' column in track_df.
#     Global plot: plot_swvl1_tp_lead_lag_all_tracks
#
# References:
#  - Seneviratne et al. (2010) - soil moisture-climate feedback, deficit definition
#  - Eltahir & Bras (1996)     - precipitation recycling, BOE land-atmosphere coupling
#  - Clapp & Hornberger (1978) - soil hydraulic parameters (beta, Ksat)

# TODO:
# 1. add shade region for land (like in latent heat flux transiton and in speed plots)
# 2. The studies used “regional SM and ST fields at four soil layers (0-10 cm, 10-40cm, 40-100 cm and 100-200 cm)” and incorporated “surface static fields such as land use and soil type”, but did not analyze specific hydraulic parameters or regional pedological characteristics.
# + Clapp-Hornberger Parameters and Soil Moisture Memory
#   - Clapp-Hornberger soil hydraulic parameters significantly control soil moisture memory timescales through their influence on water retention and conductivity:
# - Beta Exponent (β): This parameter controls the shape of the soil water retention curve and hydraulic conductivity function. Higher β values (typical of clay-rich soils) create:
#     - Steeper retention curves with more gradual drainage
#     - Longer soil moisture memory timescales
#     - Greater sensitivity to initial moisture conditions
# - Saturated Hydraulic Conductivity (Ksat): This controls the maximum rate of water movement through soil. Lower Ksat values (typical of fine-textured soils) result in:
#     - Slower drainage and longer moisture persistence
#     - Extended memory timescales
#     - Greater land-atmosphere coupling potential
# - Expected Regional Variations:
#   - Gangetic Plains Clay Loam:
#     - Moderate β values (~7-10)
#     - Moderate Ksat (10^(-6) to 10^(-5) m/s)
#     - Soil moisture memory: 2-4 weeks
#     - Balanced drainage and retention characteristics
#   - Deccan Plateau Sandy Clay:
#     - Intermediate β values (~8-12)
#     - Variable Ksat depending on clay content (10^(-7) to 10^(-5) m/s)
#     - Soil moisture memory: 1-3 weeks
#     - Complex behavior due to mixed texture
#   - Coastal Alluvium:
#     - Lower β values (~5-8) for sandy components
#     - Higher Ksat (10^(-5) to 10^(-4) m/s)
#     - Soil moisture memory: 3-10 days
#     - Rapid drainage with shorter memory

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from statsmodels.tsa.stattools import acf

from utils import (
    std_print,
    get_default_buffer_km,
    calculate_buffer_deg,
    extract_track_variable,
    clamp_to_domain,
    get_time_coord,
    legend_style,
)

# Constants
SWVL1_SAT = 0.45  # m^3/m^3  ERA5 approximate saturation

# # Map domain
# FULL_EXTENT = [70.0, 95.0, 7.0, 30.0]  # lon_min, lon_max, lat_min, lat_max
# JJAS_MONTHS = [6, 7, 8, 9]

# Font sizes
FS_TITLE = 14
FS_LABEL = 12
FS_TICK = 10
FS_LEGEND = 10
FS_ANNOT = 9


def plot_spatial_tp_swvl1(
    ds, track_df, track_id, output_dir, buffer_km=500, lsm_ds=None
):
    """Spatial maps of TP and SWVL1 during track passage, with SM deficit overlay"""
    track_id = int(track_id)
    track = track_df.sort_values("time")

    lat_center = track["latitude"].mean()
    lat_buf, lon_buf = calculate_buffer_deg(lat_center, buffer_km)
    min_lon = track["longitude"].min() - lon_buf
    max_lon = track["longitude"].max() + lon_buf
    min_lat = track["latitude"].min() - lat_buf
    max_lat = track["latitude"].max() + lat_buf

    min_lat, max_lat, min_lon, max_lon = clamp_to_domain(
        min_lat, max_lat, min_lon, max_lon
    )
    ds, time_coord = get_time_coord(ds)

    time_slice = slice(track["time"].min(), track["time"].max())
    lat_slice = slice(max_lat, min_lat)  # ERA5 descending
    lon_slice = slice(min_lon, max_lon)

    # Get averaged fields
    ds_tp = (
        ds["tp"]
        .sel({time_coord: time_slice, "latitude": lat_slice, "longitude": lon_slice})
        .sum(dim=time_coord)
        * 1000
    )  # Total, converted to mm

    ds_swlv1 = (
        ds["swvl1"]
        .sel({time_coord: time_slice, "latitude": lat_slice, "longitude": lon_slice})
        .mean(dim=time_coord)
    )

    # SM deficit = SWVL1_sat - SWVL1  (Seneviratne 2010)
    # Low deficit -> near saturation -> BOE conditions likely
    deficit_data = (SWVL1_SAT - ds_swlv1).clip(min=0)

    # ERA5 ocean SWVL1 masking
    # Apply land-sea mask so SWVL1 and deficit maps show land-only.
    if lsm_ds is not None:
        try:
            lsm_var = "lsm" if "lsm" in lsm_ds.data_vars else list(lsm_ds.data_vars)[0]
            lsm_raw = lsm_ds[lsm_var]
            # Drop every dimension that is not latitude / longitude
            non_spatial = [
                d for d in lsm_raw.dims if d not in ("latitude", "longitude")
            ]
            for dim in non_spatial:
                lsm_raw = lsm_raw.isel({dim: 0}, drop=True)  # now strictly 2-D

            lsm_grid = lsm_raw.sel(latitude=lat_slice, longitude=lon_slice).interp(
                latitude=ds_swlv1.latitude,
                longitude=ds_swlv1.longitude,
                method="nearest",
            )
            ocean_mask = lsm_grid.values < 0.5  # True = ocean cell
            ds_swlv1 = ds_swlv1.where(~ocean_mask)
            deficit_data = deficit_data.where(~ocean_mask)
            std_print("INFO", "sm", "LSM ocean mask applied to SWVL1 and deficit maps")
        except Exception as e:
            std_print(
                "WARN", "sm", "LSM masking failed, showing raw ERA5 values", str(e)
            )

    # Land/ocean mask for track coloring
    is_land_track = (
        track["is_land"].values
        if "is_land" in track.columns
        else np.ones(len(track), dtype=bool)
    )

    # Figure
    # Plot 1: Total Precipitation
    fig1, ax1 = plt.subplots(
        figsize=(11, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    im1 = ax1.contourf(
        ds_tp.longitude,
        ds_tp.latitude,
        ds_tp.values,
        levels=20,
        cmap="YlOrRd",
        transform=ccrs.PlateCarree(),
    )
    contours1 = ax1.contour(
        ds_tp.longitude,
        ds_tp.latitude,
        ds_tp.values,
        levels=10,
        colors="black",
        linewidths=0.5,
        alpha=0.5,
        transform=ccrs.PlateCarree(),
    )
    ax1.clabel(contours1, inline=True, fontsize=FS_LABEL, fmt="%1.0f")

    # Land/Ocean phase
    lons_arr = track["longitude"].values
    lats_arr = track["latitude"].values
    for i in range(len(track) - 1):
        seg_color = "saddlebrown" if is_land_track[i] else "dodgerblue"
        ax1.plot(
            [lons_arr[i], lons_arr[i + 1]],
            [lats_arr[i], lats_arr[i + 1]],
            color=seg_color,
            linewidth=2.0,
            transform=ccrs.PlateCarree(),
            zorder=5,
            solid_capstyle="round",
        )
    ax1.plot([], [], color="saddlebrown", linewidth=2, label="Track (land)")
    ax1.plot([], [], color="dodgerblue", linewidth=2, label="Track (ocean)")

    ax1.coastlines(resolution="50m", linewidth=1)
    ax1.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax1.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.4)
    ax1.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label("Total Precipitation (mm)", fontsize=FS_LABEL)

    legend_handles, labels = ax1.get_legend_handles_labels()
    if legend_handles:
        ax1.legend(handles=legend_handles, loc="upper left", **legend_style())

    ax1.set_title(
        f"Track {track_id}: Total Precipitation During Passage",
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    ax1.set_xlabel("Longitude (°E)", fontsize=FS_LABEL, fontweight="bold")
    ax1.set_ylabel("Latitude (°N)", fontsize=FS_LABEL, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tp_path = os.path.join(output_dir, f"track_{track_id}_spatial_tp.png")
    plt.savefig(tp_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    std_print("SAVED", "sm", "Spatial TP plot", tp_path)

    # Figure
    # Plot 2: SWVL1 + SM Deficit
    fig2, axes2 = plt.subplots(
        1, 2, figsize=(18, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    for ax2, data, cmap, fmt, title_sfx, cbar_lbl in [
        (
            axes2[0],
            ds_swlv1,
            "YlGnBu",
            "%1.3f",
            "Mean SWVL1 (0-7cm)",
            "SWVL1 (m³/m³)",
        ),
        (
            axes2[1],
            deficit_data,
            "YlOrBr_r",
            "%1.3f",
            "SM Deficit (sat − obs)",
            f"Deficit (m³/m³)  [sat={SWVL1_SAT}]",
        ),
    ]:
        im2 = ax2.contourf(
            data.longitude,
            data.latitude,
            data.values,
            levels=20,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
        )
        ctrs = ax2.contour(
            data.longitude,
            data.latitude,
            data.values,
            levels=10,
            colors="black",
            linewidths=0.5,
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )
        ax2.clabel(ctrs, inline=True, fontsize=FS_LABEL, fmt=fmt)

        for i in range(len(track) - 1):
            seg_color = "saddlebrown" if is_land_track[i] else "dodgerblue"
            ax2.plot(
                [lons_arr[i], lons_arr[i + 1]],
                [lats_arr[i], lats_arr[i + 1]],
                color=seg_color,
                linewidth=2.0,
                transform=ccrs.PlateCarree(),
                zorder=5,
                solid_capstyle="round",
            )

        ax2.coastlines(resolution="50m", linewidth=1)
        ax2.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        ax2.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.4)
        ax2.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
        cbar2.set_label(cbar_lbl, fontsize=FS_LABEL)
        ax2.set_title(
            f"Track {track_id}: {title_sfx}", fontsize=FS_TITLE, fontweight="bold"
        )
        ax2.set_xlabel("Longitude (°E)", fontsize=FS_LABEL, fontweight="bold")
        ax2.set_ylabel("Latitude (°N)", fontsize=FS_LABEL, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save
    sm_path = os.path.join(output_dir, f"track_{track_id}_spatial_swvl1.png")
    plt.savefig(sm_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    std_print("SAVED", "sm", "Spatial SWVL1 + deficit plot", sm_path)


def analyze_soil_memory_by_track(ds, track_df_dict, output_dir):
    """Analyze soil moisture memory effect for each track"""
    memory_results = []
    buffer_km = get_default_buffer_km("swvl1")
    for track_id, track_df in track_df_dict.items():
        swvl1_vals = []
        ds, time_coord = get_time_coord(ds)
        try:
            time_vals_arr, swvl1_vals_arr = extract_track_variable(
                ds, track_df, track_id, time_coord, "swvl1", buffer_km
            )
        except Exception as e:
            std_print("ERROR", "sm", "extract_track_variable failed", str(e))
            continue
        swvl1_vals = swvl1_vals_arr.tolist()
        if len(swvl1_vals) > 48:  # Need enough data for 24h lag
            memory_24h = compute_soil_moisture_memory(swvl1_vals, lag_hours=24)
            memory_results.append(
                {
                    "track_id": track_id,
                    "memory_index_24h": memory_24h,
                    "n_samples": len(swvl1_vals),
                    "mean_swvl1": np.mean(swvl1_vals),
                    "std_swvl1": np.std(swvl1_vals),
                    "buffer_km": buffer_km,
                }
            )

    # Export
    if memory_results:
        memory_results_df = pd.DataFrame(memory_results)
        soil_memory_CSV = os.path.join(output_dir, "soil_memory_analysis.csv")
        memory_results_df.to_csv(soil_memory_CSV, index=False)
        std_print("EXPORT", "sm", "Soil memory analysis to CSV", soil_memory_CSV)
    else:
        std_print("ERROR", "sm", "No memory_results to export")
    return memory_results


# Helper functions
def compute_sm_deficit(swvl1_values, swvl1_sat=0.45):
    """
    Soil moisture deficit: deficit = SWVL1_sat - SWVL1  (Seneviratne et al. 2010)

    Low deficit  -> soil near saturation -> LHF peaks -> Brown Ocean Effect possible
    High deficit -> soil absorbs incoming rain -> no BOE, faster storm decay

    swvl1_sat defaults to 0.45 m^3/m^3 (ERA5 approximate saturation for loamy/clay
    soils over the Indian subcontinent)
    """
    deficit = swvl1_sat - np.asarray(swvl1_values, dtype=float)
    return np.clip(deficit, 0, None)  # deficit cannot be negative


def compute_soil_moisture_memory(swvl1_series, lag_hours=24):
    """Compute autocorrelation of SWVL1 at specified lag.
    Higher autocorrelation = longer memory = more sustained feedback

    ACF requires at least lag_hours * 2 points to be meaningful
    (24h lag needs ≥48 points, corresponds to ≥48h of hourly ERA5 data)
    """
    # std_print("DEBUG", "sm", " --- Computing Soil moisture Memory ---")
    # Remove NaN values
    swvl1_clean = np.array(swvl1_series)
    swvl1_clean = swvl1_clean[~np.isnan(swvl1_clean)]

    # Need at least 2 times lag length + buffer
    min_required = lag_hours * 2 + 4
    if len(swvl1_clean) < min_required:
        std_print(
            "WARN",
            "sm",
            f"ACF lag={lag_hours}h needs ≥{min_required} pts",
            f"have {len(swvl1_clean)} - returning NaN",
        )
        return np.nan

    try:
        # Compute Auto-Correlation Function
        acf_vals = acf(swvl1_clean, nlags=lag_hours, fft=True)
        if lag_hours < len(acf_vals):
            memory_index = acf_vals[lag_hours]
            return memory_index
        else:
            return np.nan
    except Exception as e:
        std_print("ERROR", "sm", "Returing NaN for memory_index", str(e))
        return np.nan
