# NOTE:
# 1. TP + SWVL1  [tp_swvl1_timeseries]
#    - Total Precipitation (tp)         : mm/hr  (ERA5 raw: m/hr, *1000)
#    - Volumetric Soil Water Layer-1    : m³/m³  (0-7 cm depth, ERA5: swvl1)
#      SWVL1 is masked to 0 for ocean points to highlight the land-phase  saturation.
#      The precipitation-soil moisture feedback: rising TP over land should drive
#      rising SWVL1 unless runoff dominates.
# 2. Surface Heat Fluxes  [hf_timeseries]
#    - Latent Heat Flux  (LHF)  : W/m²  (ERA5: slhf J/m² accumulated, or avg_slhtf W/m² instantaneous avg)
#    - Sensible Heat Flux (SHF) : W/m²  (ERA5: sshf J/m² accumulated, or avg_ishf W/m² instantaneous avg)
#    Accumulated fields are de-accumulated as:
#        Flux [W/m²] = -(value_now − value_prev) / dt_seconds
#    The negative sign follows ERA5 sign convention (negative = away from the surface).
# 3. Equivalent Potential Temperature  [theta_e_timeseries]
#    - θ_e (Bolton 1980): Conserved under adiabatic ascent; tracks "convective fuel."
#    Derived quantity: θ_e (Bolton 1980 approximation):
#      1. Saturation vapour pressure at dew point (Magnus formula)
#             e_s = 611.2 * exp(17.67 * (T_d − 273.15) / (T_d − 29.65)) [Pa]
#      2. Specific humidity from e_s and surface pressure
#              q   = 0.622 * e_s / (P − e_s)                             [kg/kg]
#      3. Equivalent potential temperature
#              θ_e = T * (P0/P)^(Rd/cp) * exp(Lv * q / (cp * T))         [K]
#      Constants: Lv = 2.501*10^6 J/kg, Rd = 287.05 J/kg/K, cp = 1005.7 J/kg/K, P0=100000 Pa
#      Pressure used: SP (surface pressure, Pa) preferred; MSL used as fallback
#      if SP is absent from the dataset (see extract_theta_e_timeseries).
#    - θ_e is conserved under both dry and moist adiabatic ascent, making it
#      the best single variable to track the thermodynamic "fuel" available
#      for deep convection along the MD track.
#    - θ_e > ~340 K over land  ->  ample convective instability
#    - Rising θ_e approaching or after landfall ->  BOE conditions possible;
#      the land surface is sustaining moisture and convective potential
#    - The gap (T2m − D2m), i.e. dew-point depression, the spread between the two lower lines,
#      quantifies how far the boundary layer is from saturation:
#        * Small gap (<2 K)  ->  near-saturated boundary layer, high CAPE.
#        * Large gap (>5 K)  ->  dry air entrainment, suppressed deep convection
#    - D2m tracks the actual moisture content independently of temperature;
#      a sustained high D2m over land after landfall is the clue for Brown Ocean Effect.
# 4. Precipitation Recycling Ratio  [recycling_ratio_timeseries]
#     recycling_ratio = (local_evaporation) / (total_rainfall)
#     ρ = E / P  where E = LHF/(Lv x ρ_w)  [mm hr⁻¹] and P = TP  [mm hr⁻¹]
#     ρ -> 1 = rainfall mostly from local evaporation (BOE signature).
#     ρ -> 0 = rainfall from advected oceanic moisture.
#     Simplified Budyko single-column formulation (not the full 2-D Eltahir & Bras formulation).
#     This threshold is applied in md_analyse.py before calling this function.
#     Note: over the Indian subcontinent during monsoon, regional recycling fractions
#     of 10-40% are typical (Eltahir & Bras 1996); track-level ρ values above this range
#     during the land phase constitute strong BOE evidence.
# 5. Dew-Point Depression Windward/Leeward  [plot_dpd_windward_leeward]
#     Isolates the orographic Foehn-drying signature of the Eastern Ghats.
#     DPD = T2m − D2m  [K]
#     - Windward  (BoB side, lon > ghats_lon): expect DPD decrease as ascending air cools
#       moist-adiabatically and boundary layer approaches saturation.
#     - Leeward   (Deccan side, lon < ghats_lon): expect DPD spike as descending air
#       warms dry-adiabatically (Foehn effect) cutting off convective fuel.
#     - The DPD jump across the Ghats crest is the observable signature of
#       orographic moisture depletion driving the "deception" hypothesis.
#     + On the windward slope DPD drops (moistening from forced orographic ascent).
#     + On the leeward side DPD spikes (Foehn-like descent warms and dries air).
#    - Result: The MD "rains out" on the Eastern Ghats and enters the leeward side
#      thermodynamically depleted, causing premature dissipation.
#     Reference: Medina & Houze (2003); Barros & Lang (2003).
# 6. Bowen Ratio (β) [bowen_ratio_timeseries]
#    - β = SHF / LHF (Dimensionless)
#    - Ocean Signature: β << 1.0 (typically 0.1–0.2), as energy drives evaporation.
#    - BOE Signature: β remains low (< 0.5) over land, indicating the "wet"  surface is
#      mimicking the ocean's energy partitioning.
#    - Dry Land: β > 1.0 (SHF dominates), leading to rapid MD decay.
#    Over ocean tracks show \beta << 1.0, he energy at the ocean surface is primarily used for
#    evaporation (Latent Heat Flux, LHF) rather than heating the air (Sensible Heat Flux, SHF).
# 7. Diurnal Heat Flux change [diurnal_hf_amplitude_timeseries]
#    - Rolling Δ(HF) = HF_max − HF_min over a sliding window (default 24 h)
#    - Computed separately for LHF, SHF, and total HF
#    - Medina & Houze (2003, Quarterly Journal) for orographic precipitation structures over barriers comparable to the Eastern Ghats
#    - Barros & Lang (2003) for the specific rain-shadow effect
#    On the leeward side, it should spike as the dry Foehn-like descent warms and dries the descending air
#    This T2m−D2m jump on the leeward side is a direct observable signature of orographic deception hypothesis
#
# References:
#  - Eltahir & Bras (1996)      - precipitation recycling ratio
#  - Bolton (1980)              - θ_e formulation
#  - Medina & Houze (2003)      - orographic precipitation, windward/leeward asymmetry
#  - Barros & Lang (2003)       - rain-shadow effects in monsoon terrain
#  - Andersen & Shepherd (2013) - BOE LHF ≥ 70 W/m^2 threshold
#  - Trenberth et al. (2009)    - Earth energy budget, Bowen ratio, diurnal partitioning

# TODO:
# 1. Currently uses a simplified single-column Budyko formulation ρ = E/P. This is fine as a track-level diagnostic but this is not the Eltahir & Bras 2-D regional formulation. It will overestimate recycling when local TP is small (denominator near zero).

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import traceback

from utils import (
    std_print,
    calculate_buffer_deg,
    get_time_coord,
    common_bbox,
    legend_style,
    is_point_land,
    add_land_ocean_shading,
)


def tp_swvl1_timeseries(
    track_df,
    track_id,
    tp,
    swvl1,
    landfall_index,
    output_dir,
    originates_over_land=True,
):
    track_id = int(track_id)
    try:
        if not isinstance(tp, np.ndarray):
            tp = np.array(tp)
        if not isinstance(swvl1, np.ndarray):
            swvl1 = np.array(swvl1)

        times = pd.to_datetime(track_df["time"])
        is_land_arr = track_df["is_land"].values.astype(bool)

        # Create a copy of swvl1 and mask ocean points to NaN.
        swvl1_land = swvl1.copy().astype(float)
        # Apply ocean mask
        if len(swvl1_land) == len(is_land_arr):
            swvl1_land[~is_land_arr] = np.nan

        # Detect landfall
        landfall_time = None
        for i in range(1, len(track_df)):
            if not is_land_arr[i - 1] and is_land_arr[i]:
                landfall_time = times.iloc[i]
                break

        valid_data_mask = ~(np.isnan(tp) | np.isnan(swvl1_land))
        valid_count = np.sum(valid_data_mask)
        if valid_count < 2:
            std_print("WARN", "ts", "Insufficient valid data (< 2)", valid_count)
            return False

        # std_print("DEBUG", "ts", "Extracted TP-SWVL1", len(valid_times))

        # Figure
        fig, ax = plt.subplots(figsize=(18, 6))

        # Plot TP on primary axis
        color_tp = "tab:blue"
        ax.set_xlabel("Time", fontsize=12, fontweight="bold")
        ax.set_ylabel(
            "Total Precipitation (tp) (mm/hr)",
            color=color_tp,
            fontsize=10,
            fontweight="bold",
        )
        line1 = ax.plot(
            times,
            tp,
            "-o",
            color=color_tp,
            linewidth=2.5,
            markersize=5,
            label="Total Precipitation",
            alpha=0.8,
            zorder=4,
        )
        ax.tick_params(axis="y", labelcolor=color_tp)
        ax.grid(True, alpha=0.5, linestyle="--", zorder=0)

        # Land / ocean background shading
        add_land_ocean_shading(ax, times, is_land_arr, land_color="wheat", ocean_color="lightblue")
        # Landfall line
        if landfall_time is not None:
            ax.axvline(
                landfall_time,
                color="red",
                linestyle="--",
                linewidth=1.8,
                label="Landfall",
                zorder=5,
            )

        # Plot SWVL1 on secondary axis
        ax2 = ax.twinx()
        color_swvl1 = "tab:brown"
        ax2.set_ylabel(
            "Soil Water Volumetric Layer-1 (swvl1) (m³/m³)",
            color=color_swvl1,
            fontsize=12,
            fontweight="bold",
        )
        line2 = ax2.plot(
            times,
            swvl1_land,
            "s-",
            color=color_swvl1,
            linewidth=2.5,
            markersize=5,
            label="Volumetric Soil Water Level-1",
            alpha=0.8,
            zorder=4,
        )
        ax2.tick_params(axis="y", labelcolor=color_swvl1)

        tp_clean = tp[~np.isnan(tp)]
        swvl1_clean = swvl1_land[~np.isnan(swvl1_land)]
        tp_max = np.max(tp_clean) if len(tp_clean) > 0 else 1
        swvl1_max = np.max(swvl1_clean) if len(swvl1_clean) > 0 else 0.1
        tp_pad = (tp_max) * 0.1 if tp_max > 0 else 0.5
        swvl1_pad = (swvl1_max) * 0.1 if swvl1_max > 0 else 0.05

        ax.set_ylim(bottom=0, top=tp_max + tp_pad)
        ax2.set_ylim(bottom=0, top=swvl1_max + swvl1_pad)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.3, zorder=1)
        ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.3, zorder=1)

        # Legend
        handles = []
        labels = []
        handles.extend(line1 + line2)
        labels.extend([l.get_label() for l in line1 + line2])
        handles.append(Patch(facecolor="lightblue", alpha=0.5))
        labels.append("Ocean Phase")
        handles.append(Patch(facecolor="wheat", alpha=0.5))
        labels.append("Land Phase")

        if handles:
            ax.legend(handles, labels, loc="upper right", **legend_style())

        title = f"Track {track_id}: Precipitation & Soil Water Content Evolution"
        if originates_over_land:
            title += "\n(Land-Originated MD)"
        elif landfall_index != -1:
            title += "\n(Ocean-Originated, Landfall Detected)"
        else:
            title += "\n(Ocean-Originated, No Landfall)"
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m/%Y %H:%M"))

        # Save
        os.makedirs(output_dir, exist_ok=True)
        tp_swvl1_timeseries_plot = f"track_{track_id}_TP-SWVL1_time_series.png"
        tp_swvl1_timeseries_plot_path = os.path.join(output_dir, tp_swvl1_timeseries_plot)
        plt.savefig(tp_swvl1_timeseries_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "ts", "TP-SWVL1 time series", tp_swvl1_timeseries_plot_path)
        return True
    except Exception as e:
        std_print("ERROR", "ts", "TP-SWVL1 plot failed", str(e))
        traceback.print_exc()
        return False


def hf_timeseries(flux_df, track_id, output_dir):
    try:
        track_id = int(track_id)
        if flux_df is None or len(flux_df) < 3:
            std_print(
                "WARN",
                "ts",
                "Insufficient data points (< 3)",
                len(flux_df) if flux_df is not None else 0,
            )
            return False

        # Ensure flag exists
        if "is_interpolated" not in flux_df.columns:
            flux_df["is_interpolated"] = False

        # std_print("DEBUG", "ts", "Extracted heat-flux points", len(flux_df))
        # required_cols = [
        #     "time",
        #     "latent_heat_flux",
        #     "sensible_heat_flux",
        #     "total_heat_flux",
        # ]
        # missing_cols = [col for col in required_cols if col not in flux_df.columns]
        # if missing_cols:
        #     std_print("ERROR", "ts", "Missing columns", missing_cols)
        #     return False
        #
        # if "is_land" not in flux_df.columns:
        #     std_print("DEBUG", "ts", "Creating is_land column", "Default to False")
        #     flux_df = flux_df.copy()
        #     flux_df["is_land"] = False

        times = pd.to_datetime(flux_df["time"])
        is_land_arr = flux_df["is_land"].values.astype(bool)
        is_interp = flux_df["is_interpolated"].values

        # Detect landfall
        landfall_time = None
        for i in range(1, len(flux_df)):
            if not is_land_arr[i - 1] and is_land_arr[i]:
                landfall_time = times.iloc[i]
                break

        # try:
        #     times = pd.to_datetime(flux_df["time"])
        # except Exception as e:
        #     std_print("ERROR", "ts", "Time conversion failed", str(e))
        #     return False

        # Figure
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        # # Panel 1: Latent Heat Flux
        # ax1 = axes[0]
        # ax1.plot(
        #     times,
        #     flux_df["latent_heat_flux"],
        #     color="green",
        #     linewidth=2.5,
        #     marker="o",
        #     markersize=6,
        #     alpha=0.8,
        #     zorder=3,
        # )
        # # Land / ocean background shading
        # add_land_ocean_shading(ax1, times, is_land_arr, land_color="wheat", ocean_color="lightblue")
        # # Landfall line
        # if landfall_time is not None:
        #     ax1.axvline(
        #         landfall_time,
        #         color="red",
        #         linestyle="--",
        #         linewidth=1.8,
        #         label="Landfall",
        #         zorder=5,
        #     )
        #
        # ax1.set_ylabel("Latent Heat Flux (W/m²)", fontsize=12, fontweight="bold")
        # ax1.grid(True, alpha=0.4, linestyle="--", zorder=0)
        # ax1.set_title(
        #     f"Track {track_id}: Surface Heat Flux Evolution",
        #     fontsize=13,
        #     fontweight="bold",
        #     pad=12,
        # )
        #
        # # Panel 2: Sensible Heat Flux
        # ax2 = axes[1]
        # ax2.plot(
        #     times,
        #     flux_df["sensible_heat_flux"],
        #     color="orange",
        #     linewidth=2.5,
        #     marker="s",
        #     markersize=6,
        #     alpha=0.8,
        #     zorder=3,
        # )
        # # Land / ocean background shading
        # add_land_ocean_shading(ax2, times, is_land_arr, land_color="wheat", ocean_color="lightblue")
        # # Landfall line
        # if landfall_time is not None:
        #     ax2.axvline(
        #         landfall_time,
        #         color="red",
        #         linestyle="--",
        #         linewidth=1.8,
        #         label="Landfall",
        #         zorder=5,
        #     )
        #
        # ax2.set_ylabel("Sensible Heat Flux (W/m²)", fontsize=12, fontweight="bold")
        # ax2.grid(True, alpha=0.4, linestyle="--", zorder=0)
        #
        # for ax in axes:
        #     plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        #     ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m/%Y %H:%M"))
        # plt.tight_layout()

        for ax, var, color, label in [
            (axes[0], "latent_heat_flux",   "green",  "Latent Heat Flux (LHF)"),
            (axes[1], "sensible_heat_flux", "orange", "Sensible Heat Flux (SHF)"),
        ]:
            # Real data points
            real_mask = ~is_interp
            ax.plot(times[real_mask], flux_df.loc[real_mask, var],
                    "-o", color=color, linewidth=2.5, markersize=5, alpha=0.9, label=label, zorder=4)

            # Interpolated points
            interp_mask = is_interp
            if interp_mask.any():
                ax.plot(
                    times[interp_mask],
                    flux_df.loc[interp_mask, var],
                    "x--",
                    color="red",
                    linewidth=1.8,
                    markersize=7,
                    alpha=0.85,
                    label="Interpolated (gap-filled)",
                    zorder=5
                )

                # Land / ocean background shading
                add_land_ocean_shading(ax, times, is_land_arr, land_color="wheat", ocean_color="lightblue")
                # Landfall line
                if landfall_time is not None:
                    ax.axvline(
                        landfall_time,
                        color="red",
                        linestyle="--",
                        linewidth=1.8,
                        label="Landfall",
                        zorder=5,
                    )

            ax.set_ylabel(f"{label} (W/m²)", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.4, linestyle="--")

        axes[0].set_title(f"Track {track_id}: Surface Heat Flux Evolution", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Time", fontsize=12, fontweight="bold")

        for ax in axes:
            ax.legend(loc="upper right", **legend_style())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m %H:%M"))

        plt.tight_layout()

        # Save
        os.makedirs(output_dir, exist_ok=True)
        hf_timeseries_plot = f"track_{track_id}_HF_time_series.png"
        hf_timeseries_plot_path = os.path.join(output_dir, hf_timeseries_plot)
        plt.savefig(hf_timeseries_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "ts", "Heat flux timeseries", hf_timeseries_plot_path)
        return True
    except Exception as e:
        std_print("ERROR", "ts", "Heat flux timeseries plot failed", str(e))
        traceback.print_exc()
        return False


def theta_e_timeseries(theta_df, track_id, output_dir):
    """
    Plot equivalent potential temperature (theta_e) along the MD track over time.

    All three variables are plotted in Kelvin on a single panel:
      - θ_e (K)  ~330-370 K  : stability/moisture fuel signal
      - T2m (K)  ~295-310 K  : temperature driver of θ_e
      - D2m (K)  ~285-305 K  : moisture driver; always <= T2m

    The vertical spread between T2m and D2m is the dew-point depression:
      - Small spread (<2 K)  ->  near-saturated BL, high CAPE
      - Large spread (>5 K)  ->  dry air entrainment, suppressed convection
    Rising θ_e after landfall with sustained high D2m = BOE fingerprint.
    """
    try:
        track_id = int(track_id)
        if theta_df is None or len(theta_df) < 3:
            std_print("WARN", "ts", "Insufficient theta_e data (< 3)")
            return False

        # std_print("DEBUG", "ts", "Extracted theta_e points", len(theta_df))

        required = {"time", "theta_e_K", "t2m_K", "d2m_K", "is_land"}
        missing = required - set(theta_df.columns)
        if missing:
            std_print("ERROR", "ts", "theta_e_timeseries: missing columns", str(missing))
            return False

        times = pd.to_datetime(theta_df["time"])
        theta_e = theta_df["theta_e_K"].values  # in K
        t2m = theta_df["t2m_K"].values  # in K
        d2m = theta_df["d2m_K"].values  # in K
        is_land_arr = theta_df["is_land"].values.astype(bool)

        # Detect landfall
        landfall_time = None
        for i in range(1, len(theta_df)):
            if not is_land_arr[i - 1] and is_land_arr[i]:
                landfall_time = times.iloc[i]
                break

        # Figure
        fig, ax = plt.subplots(figsize=(16, 7))

        ax.plot(
            times,
            theta_e,
            "-o",
            color="darkorange",
            linewidth=2.5,
            markersize=4,
            alpha=0.88,
            label="θ_e  (K)",
            zorder=4,
        )
        ax.plot(
            times,
            t2m,
            "-^",
            color="firebrick",
            linewidth=2.2,
            markersize=4,
            alpha=0.82,
            label="T2m (K)",
            zorder=4,
        )
        ax.plot(
            times,
            d2m,
            "-s",
            color="steelblue",
            linewidth=2.2,
            markersize=4,
            alpha=0.82,
            label="D2m (K)",
            zorder=4,
        )
        # Land / ocean background shading
        add_land_ocean_shading(ax, times, is_land_arr, land_color="wheat", ocean_color="lightblue")
        # Landfall line
        if landfall_time is not None:
            ax.axvline(
                landfall_time,
                color="red",
                linestyle="--",
                linewidth=1.8,
                label="Landfall",
                zorder=5,
            )

        # Annotate peak θ_e during land phase
        land_idx = np.where(is_land_arr & ~np.isnan(theta_e))[0]
        if len(land_idx) > 0:
            peak_i = land_idx[np.argmax(theta_e[land_idx])]
            ax.annotate(
                f"Peak θ_e: {theta_e[peak_i]:.1f} K",
                xy=(times.iloc[peak_i], theta_e[peak_i]),
                xytext=(12, 6),
                textcoords="offset points",
                fontsize=9,
                color="darkorange",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.0),
                bbox=common_bbox(),
            )
            # Mean dew-point depression over land phase
            mean_depression = np.nanmean(t2m[land_idx] - d2m[land_idx])
            ax.annotate(
                f"Mean D.P. depression (land): {mean_depression:.1f} K",
                xy=(0.9, 0.5),
                xycoords="axes fraction",
                fontsize=9,
                color="purple",
                bbox=common_bbox(),
            )

        # Labels and formatting
        ax.set_ylabel("Temperature (K)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Track {track_id}: θ_e, T2m and D2m Along Track",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.grid(True, alpha=0.4, linestyle="--", zorder=0)
        ax.legend(loc="upper left", **legend_style())

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m/%Y %H:%M"))
        plt.tight_layout()

        # Save
        os.makedirs(output_dir, exist_ok=True)
        theta_e_plot_path = os.path.join(
            output_dir, f"track_{track_id}_theta_e_timeseries.png"
        )
        plt.savefig(theta_e_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "ts", "theta_e timeseries", theta_e_plot_path)
        return True
    except Exception as e:
        std_print("ERROR", "ts", "theta_e timeseries plot failed", str(e))
        traceback.print_exc()
        return False


def recycling_ratio_timeseries(result, track_id, output_dir):
    """Plot recycling ratio using precomputed arrays"""
    times = pd.Series(pd.to_datetime(result["time"])).reset_index(drop=True)
    is_land_arr = np.array(result["is_land"])
    rho_df = result["rho"]
    E = result["E_mm_h"]
    P = result["P_mm_h"]

    # Detect landfall
    landfall_time = None
    for i in range(1, len(rho_df)):
        if not is_land_arr[i - 1] and is_land_arr[i]:
            landfall_time = times.iloc[i]
            break

    # Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for ax in (ax1, ax2):
        # Land / ocean background shading
        add_land_ocean_shading(ax, times, is_land_arr, land_color="wheat", ocean_color="lightblue")
        # Landfall line
        if landfall_time is not None:
            ax.axvline(
                landfall_time,
                color="red",
                linestyle="--",
                linewidth=1.8,
                label="Landfall",
                zorder=5,
            )

    ax1.plot(
      times,
      rho_df,
      "-o",
      color="teal",
      linewidth=2,
      markersize=4,
      label="Recycling ratio ρ = E / P"
    )
    ax1.axhline(
      0.25,
      color="darkorange",
      linestyle="--",
      label="ρ = 0.25 (significant local recycling)"
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Recycling ratio ρ", fontsize=12)
    ax1.set_title(f"Track {track_id}: Precipitation Recycling Ratio", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", **legend_style())

    ax2.plot(
        times,
        P,
        "-o",
        color="royalblue",
        label="Precipitation P (mm h⁻¹)"
    )
    ax2.plot(
        times,
        E,
        "-o",
        color="tomato",
        label="Evaporation E (mm h⁻¹)"
    )
    ax2.set_ylabel("Flux (mm h⁻¹)", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", **legend_style())

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m/%Y %H:%M"))
    plt.tight_layout()

    # Save
    rr_path = os.path.join(output_dir, f"track_{track_id}_recycling_ratio.png")
    plt.savefig(rr_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ts", "Recycling ratio timeseries", rr_path)


# FIX: currently only analysis eartern ghats (82 lon)
# Dew-Point Depression: Windward vs Leeward (orographic deception signature)
def dpd_windward_leeward_timeseries(
    theta_df,
    track_df,
    track_id,
    output_dir,
    ghats_lon=82,   # approx Eastern Ghats main crest longitude
):
    """
    Visualise the Dew-Point Depression (DPD = T2m − D2m) to isolate the
    orographic Foehn-drying signature as the MD crosses the Eastern Ghats.

    Windward (BoB-facing) slope -> forced moist ascent -> DPD drops -> near-saturation
    Leeward  (Deccan-facing) side -> dry adiabatic descent (Foehn) -> DPD spikes

    The DPD jump across the Ghats crest is the observable signature of orographic
    moisture depletion that drives the "deception" hypothesis (Medina & Houze 2003;
    Barros & Lang 2003).
    """
    try:
        track_id = int(track_id)
        if theta_df is None or len(theta_df) < 3:
            std_print("WARN", "ts", f"Track {track_id}: insufficient theta_e data for DPD plot")
            return False

        required = {"time", "t2m_K", "d2m_K", "is_land"}
        missing = required - set(theta_df.columns)
        if missing:
            std_print("WARN", "ts", f"Track {track_id}: DPD plot skipped - missing cols", str(missing))
            return False

        df = theta_df.copy().sort_values("time").reset_index(drop=True)
        df["dpd_K"] = df["t2m_K"] - df["d2m_K"]   # always >= 0 physically

        times = pd.to_datetime(df["time"])
        is_land_arr = df["is_land"].values.astype(bool)

        # Landfall time
        landfall_time = None
        for i in range(1, len(df)):
            if not is_land_arr[i - 1] and is_land_arr[i]:
                landfall_time = times.iloc[i]
                break

        # Classify each land-point as windward / leeward / ocean
        # Windward = land AND lon > ghats_lon  (BoB side, approaching crest)
        # Leeward  = land AND lon <= ghats_lon (Deccan side, past crest)
        if track_df['longitude'].max() < ghats_lon: # If track never reaches the East coast/Ghats
            std_print("INFO", "ts", f"Track {track_id} skipped: West of Ghats range")
            return
        if "longitude" in df.columns:
            lons = df["longitude"].values
        else:
            # Fall back to track_df
            merged = df.merge(track_df[["time", "longitude"]].drop_duplicates("time"), on="time", how="left")
            lons = merged["longitude"].values

        is_windward = is_land_arr & (lons >  ghats_lon)
        is_leeward  = is_land_arr & (lons <= ghats_lon)

        # Terrain crossing time (first point where lon crosses ghats_lon)
        crest_time = None
        for i in range(1, len(df)):
            if is_land_arr[i]:
                if lons[i - 1] > ghats_lon >= lons[i]:    # moving westward (typical MD)
                    crest_time = times.iloc[i]
                    break
                elif lons[i - 1] < ghats_lon <= lons[i]:  # moving eastward
                    crest_time = times.iloc[i]
                    break

        # DPD values
        dpd           = df["dpd_K"].values
        ocean_vals    = dpd[~is_land_arr]
        windward_vals = dpd[is_windward]
        leeward_vals  = dpd[is_leeward]

        # Figure
        fig, ax = plt.subplots(figsize=(16, 7))

        # Land / ocean background shading
        add_land_ocean_shading(ax, times, is_land_arr, land_color="wheat", ocean_color="lightblue")
        # Landfall line
        if landfall_time is not None:
            ax.axvline(
                landfall_time,
                color="red",
                linestyle="--",
                linewidth=1.8,
                label="Landfall",
                zorder=5,
            )
        # Ocean segment
        ax.scatter(
            times[~is_land_arr],
            dpd[~is_land_arr],
            c="steelblue",
            s=30,
            alpha=0.6,
            label="Ocean",
            zorder=3,
        )
        # Windward segment
        ax.scatter(
            times[is_windward],
            dpd[is_windward],
            c="seagreen",
            s=50,
            marker="^",
            alpha=0.9,
            label=f"Windward (lon > {ghats_lon}°E)",
            zorder=4,
        )
        # Leeward segment
        ax.scatter(
            times[is_leeward],
            dpd[is_leeward],
            c="firebrick",
            s=50,
            marker="v",
            alpha=0.9,
            label=f"Leeward (lon ≤ {ghats_lon}°E)",
            zorder=4,
        )
        # Connect all points
        ax.plot(
            times,
            dpd,
            "-",
            color="gray",
            linewidth=1.0,
            alpha=0.4,
            zorder=2
        )

        # Reference thresholds
        ax.axhline(
            2.0,
            color="seagreen",
            linestyle="--",
            linewidth=1.2,
            label="DPD = 2 K (near-saturated)",
            zorder=3
        )
        ax.axhline(
            5.0,
            color="firebrick",
            linestyle="--",
            linewidth=1.2,
            label="DPD = 5 K (dry air entrained)",
            zorder=3
        )

        # Marker lines
        if crest_time is not None:
            ax.axvline(
                crest_time,
                color="darkorange",
                linestyle="-.",
                linewidth=1.8,
                label=f"Ghats crest (~{ghats_lon}°E)",
                zorder=5
            )

        ax.set_xlabel("Time", fontsize=12, fontweight="bold")
        ax.set_ylabel("Dew-Point Depression  T2m − D2m  (K)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Track {track_id}: Dew-Point Depression - Orographic Foehn Signal\n"
            "↓ Windward DPD = moistening (forced ascent)  |  "
            "↑ Leeward DPD = Foehn drying (orographic deception)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper left", **legend_style())
        ax.grid(True, alpha=0.35, linestyle="--", zorder=0)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m %H:%M"))

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        dpd_path = os.path.join(output_dir, f"track_{track_id}_dpd_windward_leeward.png")
        plt.savefig(dpd_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "ts", "DPD windward/leeward plot", dpd_path)

        return {
            "track_id":          track_id,
            "mean_dpd_ocean":    float(np.nanmean(ocean_vals))    if len(ocean_vals) > 0    else np.nan,
            "mean_dpd_windward": float(np.nanmean(windward_vals)) if len(windward_vals) > 0 else np.nan,
            "mean_dpd_leeward":  float(np.nanmean(leeward_vals))  if len(leeward_vals) > 0  else np.nan,
            "dpd_jump_K":        (float(np.nanmean(leeward_vals)) - float(np.nanmean(windward_vals)))
                                 if len(windward_vals) > 0 and len(leeward_vals) > 0 else np.nan,
            "ghats_crest_time":  str(crest_time) if crest_time else None,
        }
    except Exception as e:
        std_print("ERROR", "ts", f"Track {track_id}: DPD windward/leeward plot failed", str(e))
        traceback.print_exc()
        return None


# Bowen ratio (LHF/SHF) evolution
def bowen_ratio_timeseries(flux_df, track_id, output_dir):
    """
    Bowen ratio  β = SHF / LHF  (Trenberth et al. 2009)

    Interpretation:
      β ~ 0.10        : open ocean  (LHF dominates)
      β < 0.30        : BOE-active land surface mimicking ocean
      0.30 < β < 1.0  : transitional – partially moist land
      β > 1.0         : dry land (SHF dominates, no BOE)

    Reference:
    - Trenberth et al. 2009
    - Andersen & Shepherd 2013
    """
    track_id = int(track_id)
    if flux_df is None or len(flux_df) < 5:
        std_print(
            "WARN",
            "ts",
            f"Track {track_id}: Insufficient flux for Bowen ratio",
            len(flux_df) if flux_df is not None else 0,
        )
        return None

    df = flux_df.copy()
    # Ensure no division by zero if LHF is near 0
    df["bowen"] = df["sensible_heat_flux"] / df["latent_heat_flux"].replace(0, np.nan)
    df["bowen"] = df["bowen"].clip(-2, 5)  # clip outliers for display

    times = pd.to_datetime(df["time"])
    is_land_arr = (
        df["is_land"].values.astype(bool)
        if "is_land" in df.columns
        else np.zeros(len(df), bool)
    )

    # Detect landfall
    landfall_time = None
    for i in range(1, len(df)):
        if not is_land_arr[i - 1] and is_land_arr[i]:
            landfall_time = times.iloc[i]
            break

    # Figure
    fig, ax = plt.subplots(figsize=(16, 7))

    # Land / ocean background shading
    add_land_ocean_shading(ax, times, is_land_arr, land_color="wheat", ocean_color="lightblue")
    # Landfall line
    if landfall_time is not None:
        ax.axvline(
            landfall_time,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label="Landfall",
            zorder=5,
        )

    # Plot \beta
    ax.plot(
        times,
        df["bowen"],
        "-o",
        color="darkred",
        linewidth=2.5,
        markersize=5,
        alpha=0.85,
        label="Bowen ratio β = SHF / LHF",
        zorder=4,
    )

    # Reference thresholds (Trenberth et al. 2009; Andersen & Shepherd 2013)
    ax.axhline(
        0.10,
        color="steelblue",
        linestyle="--",
        linewidth=1.2,
        label="β = 0.10 (open ocean)",
        zorder=3,
    )
    ax.axhline(
        0.30,
        color="seagreen",
        linestyle="--",
        linewidth=1.5,
        label="β = 0.30 (BOE threshold)",
        zorder=3,
    )
    ax.axhline(
        1.00,
        color="firebrick",
        linestyle=":",
        linewidth=1.2,
        label="β = 1.00 (dry land)",
        zorder=3,
    )
    # Shade BOE-active region on land: β < 0.30
    ax.fill_between(
        times,
        df["bowen"],
        0.30,
        where=(df["bowen"] < 0.30) & is_land_arr,
        alpha=0.15,
        color="seagreen",
        label="BOE-active (β < 0.30 on land)",
        zorder=2,
    )

    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bowen ratio β", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Track {track_id}: Bowen Ratio (β = SHF / LHF)",
        # "β < 0.30 sustained inland -> Brown Ocean Effect"
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", **legend_style())
    ax.grid(True, alpha=0.4, linestyle="--", zorder=0)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m %H:%M"))
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    bowen_ratio_path = os.path.join(output_dir, f"track_{track_id}_bowen_ratio.png")
    plt.savefig(bowen_ratio_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ts", "Bowen ratio plot", bowen_ratio_path)

    # Return summary stats for global table
    land_bowen = df.loc[is_land_arr, "bowen"].dropna()
    return {
        "track_id": track_id,
        "mean_bowen_land": float(land_bowen.mean()) if len(land_bowen) > 0 else np.nan,
        "min_bowen_land": float(land_bowen.min()) if len(land_bowen) > 0 else np.nan,
        "boe_active_hours": int((land_bowen < 0.30).sum()),
    }


def diurnal_hf_amplitude_timeseries(flux_df, track_id, output_dir, window_hours=24):
    """
    Rolling Δ(HF) = HF_max − HF_min over a sliding window (default 24 h).
    Computed separately for LHF, SHF, and total HF.
 
    Interpretation:
      Over open ocean : Δ(LHF) ~ 10–30 W/m²  (weak diurnal cycle, SST-driven)
      Over dry land   : Δ(LHF) ~ 80–150 W/m² (strong solar forcing)
      BOE signature   : Δ(LHF) stays ocean-like (< 40 W/m²) over land -> the wet surface damps the diurnal cycle, mimicking the ocean.
 
    A rising Δ(SHF) after landfall combined with a flat/falling Δ(LHF) indicates the Bowen ratio is increasing -> soil is drying -> BOE breakdown.
    Conversely, both staying low signals sustained BOE-active surface conditions.

    Reference: Trenberth et al. (2009) — Earth energy budget, diurnal partitioning.
    """
    try:
        track_id = int(track_id)
        if flux_df is None or len(flux_df) < window_hours // 2:
            std_print("WARN", "ts", f"Track {track_id}: insufficient flux for diurnal amplitude",
                      len(flux_df) if flux_df is not None else 0)
            return None
 
        required = {"time", "latent_heat_flux", "sensible_heat_flux", "total_heat_flux"}
        if not required.issubset(flux_df.columns):
            std_print("WARN", "ts", f"Track {track_id}: diurnal HF skipped – missing columns",
                      str(required - set(flux_df.columns)))
            return None
 
        df = flux_df.copy().sort_values("time")
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
 
        # Rolling max − min  window
        window = f"{window_hours}h"
        min_pts = max(window_hours // 4, 3)    # at least 25% of window filled
        roll = df[["latent_heat_flux", "sensible_heat_flux", "total_heat_flux"]].rolling(
            window, min_periods=min_pts
        )
        amp = (roll.max() - roll.min()).reset_index()
        amp.columns = ["time", "delta_lhf", "delta_shf", "delta_thf"]
 
        is_land_col = df["is_land"].values if "is_land" in df.columns else np.zeros(len(df), bool)
        amp["is_land"] = is_land_col
        times = pd.to_datetime(amp["time"])
        is_land_arr = amp["is_land"].values.astype(bool)
 
        # Detect landfall
        landfall_time = None
        for i in range(1, len(amp)):
            if not is_land_arr[i - 1] and is_land_arr[i]:
                landfall_time = times.iloc[i]
                break
 
        # Reference thresholds
        OCEAN_THRESH = 30.0    # W/m² — weak diurnal cycle (ocean-like)
        DRY_THRESH   = 80.0    # W/m² — strong diurnal cycle (dry land)
 
        # Figure
        fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
 
        panels = [
            (axes[0], amp["delta_lhf"],  "green",      f"Δ(LHF)  (W/m²)"),
            (axes[1], amp["delta_shf"],  "darkorange",  f"Δ(SHF)  (W/m²)"),
            (axes[2], amp["delta_thf"],  "steelblue",   f"Δ(Total HF)  (W/m²)"),
        ]
 
        for ax, series, color, ylabel in panels:
            add_land_ocean_shading(ax, times, is_land_arr,
                                   land_color="wheat", ocean_color="lightblue")
            ax.plot(times, series, "-o", color=color, linewidth=2.2,
                    markersize=4, alpha=0.85, zorder=4)
 
            # Reference thresholds only on LHF panel
            if ylabel.startswith("Δ(LHF"):
                ax.axhline(OCEAN_THRESH, color="steelblue", linestyle="--",
                           linewidth=1.2, label=f"Δ = {OCEAN_THRESH:.0f} W/m²  (ocean-like)", zorder=3)
                ax.axhline(DRY_THRESH,  color="firebrick",  linestyle=":",
                           linewidth=1.2, label=f"Δ = {DRY_THRESH:.0f} W/m²  (dry land)",   zorder=3)
                # Shade BOE-active region (ocean-like diurnal amplitude on land)
                ax.fill_between(
                    times, series, OCEAN_THRESH,
                    where=(series < OCEAN_THRESH) & is_land_arr,
                    alpha=0.15, color="seagreen", label="BOE-active (Δ(LHF) < 30 on land)", zorder=2,
                )
 
            if landfall_time is not None:
                ax.axvline(landfall_time, color="red", linestyle="--",
                           linewidth=1.6, label="Landfall", zorder=5)
 
            ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.35, linestyle="--", zorder=0)
            if ylabel.startswith("Δ(LHF"):
                ax.legend(loc="upper right", **legend_style())
 
        axes[0].set_title(
            f"Track {track_id}: Diurnal Heat Flux Amplitude  change Δ(HF) (rolling {window_hours} h)\n"
            "Low Δ(LHF) over land -> ocean-like surface -> BOE active",
            fontsize=13, fontweight="bold",
        )
        axes[-1].set_xlabel("Time", fontsize=12, fontweight="bold")
        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d/%m %H:%M"))
        plt.tight_layout()
 
        os.makedirs(output_dir, exist_ok=True)
        dhf_path = os.path.join(output_dir, f"track_{track_id}_diurnal_HF_amplitude.png")
        plt.savefig(dhf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "ts", "Diurnal HF amplitude plot", dhf_path)
 
        # Per-track summary dict
        land_dlhf = amp.loc[is_land_arr, "delta_lhf"].dropna()
        return {
            "track_id":          track_id,
            "mean_delta_lhf_land": float(land_dlhf.mean()) if len(land_dlhf) > 0 else np.nan,
            "boe_diurnal_hours":   int((land_dlhf < OCEAN_THRESH).sum()),  # hours with ocean-like amplitude
            "window_hours":        window_hours,
        }
    except Exception as e:
        std_print("ERROR", "ts", f"Track {track_id}: diurnal HF amplitude plot failed", str(e))
        traceback.print_exc()
        return None


# Helper / extraction functions
def extract_heat_flux_timeseries(ds, track_df, track_id, buffer_km=100):
    """Handles two ERA5 conventions:
    1. Accumulated: slhf, sshf (J m⁻²) - negative = away from surface
       De-accumulated as: Flux [W/m²] = -(val_now − val_prev) / dt_seconds
    2. Instantaneous Average: avg_slhtf, avg_ishf (W m⁻²)
    """
    track_id = int(track_id)
    ds, time_coord = get_time_coord(ds)
    ds_times = pd.to_datetime(ds.time.values)

    # lat_values = ds.latitude.values
    # lat_descending = lat_values[0] > lat_values[-1] if len(lat_values) > 1 else True

    track = track_df.sort_values("time").reset_index(drop=True)
    if "is_land" not in track.columns:
        track["is_land"] = track.apply(
            lambda r: is_point_land(r["latitude"], r["longitude"]), axis=1
        )

    has_accumulated = "slhf" in ds.data_vars and "sshf" in ds.data_vars
    has_average = "avg_slhtf" in ds.data_vars and "avg_ishf" in ds.data_vars
    if not (has_accumulated or has_average):
        std_print("ERROR", "ts", "No valid heat flux variables found")
        std_print(None, None, "Check variables :", list(ds.data_vars))
        return None

    flux_records = []
    for i in range(1, len(track)):
        row = track.iloc[i]
        t_now = pd.to_datetime(row["time"])
        lat_c = row["latitude"]
        lon_c = row["longitude"]

    flux_records = []
    prev_slhf = prev_sshf = None
    prev_t = None

    for i in range(len(track)):
        row = track.iloc[i]
        t_now = pd.to_datetime(row["time"])
        lat_c = row["latitude"]
        lon_c = row["longitude"]
        is_land_pt = row["is_land"]

        # Find nearest ERA5 timestep
        idx_now = np.argmin(np.abs(ds_times - t_now))
        t_era = ds_times[idx_now]
        is_interpolated = False

        # Filling 2-hour gap with average of prev + next
        if prev_t is not None and (t_era - prev_t).total_seconds() == 7200:
            t_mid = prev_t + pd.Timedelta(hours=1)
            is_interpolated = True

            # Linear interpolation (avg) for the missing hour
            if has_accumulated:
                slhf_now = float(ds.isel({time_coord: idx_now})["slhf"].mean().values)
                sshf_now = float(ds.isel({time_coord: idx_now})["sshf"].mean().values)
                slhf_mid = (prev_slhf + slhf_now) / 2
                sshf_mid = (prev_sshf + sshf_now) / 2
            else:
                # Instantaneous averages
                slhf_mid = prev_slhf
                sshf_mid = prev_sshf

            flux_records.append({
                "time": t_mid,
                "latent_heat_flux": slhf_mid,
                "sensible_heat_flux": sshf_mid,
                "total_heat_flux": slhf_mid + sshf_mid,
                "latitude": lat_c,
                "longitude": lon_c,
                "is_land": is_land_pt,
                "is_interpolated": True,
            })

        try:
            lat_buf, lon_buf = calculate_buffer_deg(lat_c, buffer_km)
            lat_slice = slice(lat_c + lat_buf, lat_c - lat_buf) if ds.latitude[0] > ds.latitude[-1] else slice(lat_c - lat_buf, lat_c + lat_buf)
            lon_slice = slice(lon_c - lon_buf, lon_c + lon_buf)

            sub = ds.isel({time_coord: idx_now}).sel(latitude=lat_slice, longitude=lon_slice)

            if has_average:
                slhf_w = float(sub["avg_slhtf"].mean().values)
                sshf_w = float(sub["avg_ishf"].mean().values)
            else:
                # De-accumulation (accumulated fields)
                if prev_slhf is not None:
                    dt_sec = (t_era - prev_t).total_seconds()
                    slhf_now = float(sub["slhf"].mean().values)
                    sshf_now = float(sub["sshf"].mean().values)
                    slhf_w = -(slhf_now - prev_slhf) / dt_sec if dt_sec > 0 else np.nan
                    sshf_w = -(sshf_now - prev_sshf) / dt_sec if dt_sec > 0 else np.nan
                else:
                    slhf_w = sshf_w = np.nan

            if np.isnan(slhf_w) or np.isnan(sshf_w):
                continue

            flux_records.append({
                "time": t_era,
                "latent_heat_flux": slhf_w,
                "sensible_heat_flux": sshf_w,
                "total_heat_flux": slhf_w + sshf_w,
                "latitude": lat_c,
                "longitude": lon_c,
                "is_land": is_land_pt,
                "is_interpolated": is_interpolated,
            })

            # Update previous for next gap-filling
            prev_slhf = float(sub["slhf"].mean().values) if has_accumulated else slhf_w
            prev_sshf = float(sub["sshf"].mean().values) if has_accumulated else sshf_w
            prev_t = t_era
        except Exception as e:
            std_print("WARN", "ts", f"Heat flux extraction failed at {t_now}", str(e))
            continue
    if not flux_records:
        std_print("ERROR", "ts", "No heat-flux data extracted")
        return None

    df = pd.DataFrame(flux_records)
    n_interp = df["is_interpolated"].sum()
    std_print("INFO", "ts", f"Extracted {len(df)} points ({n_interp} interpolated)")
    return df


# Equivalent Potential Temperature (θ_e)
def compute_theta_e(t2m_K, d2m_K, sp_Pa):
    """
    Bolton (1980) approximation:
      e_s  = 611.2 * exp( 17.67 * (T_d−273.15) / (T_d−29.65) )   [Pa]
      q    = 0.622 * e_s / (P − e_s)                             [kg/kg]
      θ_e  = T * (P0/P)^(Rd/cp) * exp(Lv * q / (cp * T))         [K]
    SP (surface pressure) is preferred over MSL for the pressure term because
    θ_e is a near-surface parcel property
    """
    Lv = 2.501e6  # J/kg
    Rd = 287.05  # J/kg/K
    cp = 1005.7  # J/kg/K
    P0 = 100000.0  # Pa
    e_s = 611.2 * np.exp(17.67 * (d2m_K - 273.15) / (d2m_K - 29.65))
    q = 0.622 * e_s / (sp_Pa - e_s)
    theta_e = t2m_K * (P0 / sp_Pa) ** (Rd / cp) * np.exp(Lv * q / (cp * t2m_K))
    return theta_e


def extract_theta_e_timeseries(ds, track_df, track_id, buffer_km=100):
    """Extract θ_e, T2m, D2m, and SP along an MD track"""
    track_id = int(track_id)
    ds, time_coord = get_time_coord(ds)
    ds_times = pd.to_datetime(ds.time.values)

    has_sp = "sp" in ds.data_vars
    has_msl = "msl" in ds.data_vars
    if "t2m" not in ds.data_vars or "d2m" not in ds.data_vars:
        std_print("WARN", "ts", "t2m or d2m not in dataset - skipping theta_e")
        return None
    if not has_sp and not has_msl:
        std_print("WARN", "ts", "sp and msl both absent - skipping theta_e")
        return None
    pressure_var = "sp" if has_sp else "msl"

    lat_values = ds.latitude.values
    lat_descending = lat_values[0] > lat_values[-1] if len(lat_values) > 1 else True

    track = track_df.sort_values("time").reset_index(drop=True)
    if "is_land" not in track.columns:
        track["is_land"] = track.apply(
            lambda r: is_point_land(r["latitude"], r["longitude"]), axis=1
        )

    theta_e_records = []
    for i in range(len(track)):
        row = track.iloc[i]
        t_now = pd.to_datetime(row["time"])
        lat_c = row["latitude"]
        lon_c = row["longitude"]
        is_land_pt = row["is_land"]

        lat_buf, lon_buf = calculate_buffer_deg(lat_c, buffer_km)

        idx_now = np.argmin(np.abs(ds_times - t_now))
        ds_now = ds.isel({time_coord: idx_now})

        lat_min = max(lat_c - lat_buf, float(ds.latitude.min()))
        lat_max = min(lat_c + lat_buf, float(ds.latitude.max()))
        lon_min = max(lon_c - lon_buf, float(ds.longitude.min()))
        lon_max = min(lon_c + lon_buf, float(ds.longitude.max()))
        if lat_min >= lat_max or lon_min >= lon_max:
            continue

        lat_slice = (
            slice(lat_max, lat_min) if lat_descending else slice(lat_min, lat_max)
        )
        lon_slice = slice(lon_min, lon_max)

        try:
            sub = ds_now.sel(latitude=lat_slice, longitude=lon_slice)
            if sub.sizes.get("latitude", 0) == 0 or sub.sizes.get("longitude", 0) == 0:
                sub = ds_now.sel(
                    latitude=lat_c, longitude=lon_c, method="nearest"
                ).expand_dims(["latitude", "longitude"])
        except Exception:
            sub = ds_now.sel(
                latitude=lat_c, longitude=lon_c, method="nearest"
            ).expand_dims(["latitude", "longitude"])

        try:
            t2m_val = float(sub["t2m"].mean().values)
            d2m_val = float(sub["d2m"].mean().values)
            sp_val = float(sub[pressure_var].mean().values)
            if any(np.isnan(v) for v in (t2m_val, d2m_val, sp_val)):
                continue
            theta_val = float(compute_theta_e(t2m_val, d2m_val, sp_val))
            theta_e_records.append(
                {
                    "time": t_now,
                    "theta_e_K": theta_val,
                    "t2m_K": t2m_val,
                    "d2m_K": d2m_val,
                    "sp_Pa": sp_val,
                    "latitude": lat_c,
                    "longitude": lon_c,
                    "is_land": is_land_pt,
                }
            )
        except Exception as e:
            std_print("WARN", "ts", f"theta_e failed at {t_now}", str(e))
            continue

    if not theta_e_records:
        std_print("ERROR", "ts", "No theta_e data extracted")
        return None
    return pd.DataFrame(theta_e_records)
