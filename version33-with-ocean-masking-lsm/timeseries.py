# NOTE:
# 1. TP + SWVL1  [tp_swvl1_timeseries]
#    - Total Precipitation (tp)         : mm/hr  (ERA5 raw: m/hr, *1000)
#    - Volumetric Soil Water Layer-1    : m³/m³  (0-7 cm depth, ERA5: swvl1)
#      SWVL1 is zeroed-out for ocean points since it has no physical meaning over the ocean. The dual
#      -axis lets both variables be read simultaneously.
#      Together they reveal the precipitation-soil moisture feedback: rising TP over land should drive
#      rising SWVL1 unless runoff dominates.
# 2. Surface Heat Fluxes  [hf_timeseries]
#    - Latent Heat Flux  (LHF)  : W/m²  (ERA5: slhf J/m² accumulated, or avg_slhtf W/m² instantaneous avg)
#    - Sensible Heat Flux (SHF) : W/m²  (ERA5: sshf J/m² accumulated, or
#                                          avg_ishf W/m² instantaneous avg)
#    Accumulated fields are de-accumulated as:
#        Flux [W/m²] = -(value_now − value_prev) / dt_seconds
#    The negative sign follows ERA5 sign convention (negative = away from the surface). The ratio
#    LHF/SHF (Bowen ratio) is the key BOE diagnostic: a low Bowen ratio (LHF >> SHF) over land signals
#    ocean-like surface energy partitioning - the Brown Ocean Effect.
# 3. Equivalent Potential Temperature  [theta_e_timeseries]
#    Plots θ_e, T2m, and D2m (in K)
#    Derived quantity: θ_e (Bolton 1980 approximation):
#      Step 1: Saturation vapour pressure at dew point (Magnus formula)
#              e_s = 611.2 * exp( 17.67 * (T_d − 273.15) / (T_d − 29.65) )   [Pa]
#      Step 2: Specific humidity from e_s and surface pressure
#              q   = 0.622 * e_s / (P − e_s)                                  [kg/kg]
#      Step 3: Equivalent potential temperature
#              θ_e = T * (P0/P)^(Rd/cp) * exp(Lv * q / (cp * T))             [K]
#      Constants: Lv=2.501*10⁶ J/kg, Rd=287.05 J/kg/K, cp=1005.7 J/kg/K,
#                 P0=100000 Pa
#      Pressure used: SP (surface pressure, Pa) preferred; MSL used as fallback
#      if SP is absent from the dataset (see extract_theta_e_timeseries).
#
#    - θ_e is conserved under both dry and moist adiabatic ascent, making it
#      the best single variable to track the thermodynamic "fuel" available
#      for deep convection along the MD track.
#    - θ_e > ~340 K over land  ->  ample convective instability
#    - Rising θ_e approaching or after landfall  ->  BOE conditions possible;
#      the land surface is sustaining moisture and convective potential
#    - The gap (T2m − D2m), i.e. dew-point depression, visible in the plot as
#      the spread between the two lower lines, quantifies how far the boundary
#      layer is from saturation:
#        * Small gap (<2 K)  ->  near-saturated boundary layer, high CAPE
#        * Large gap (>5 K)  ->  dry air entrainment, suppressed deep convection
#    - D2m tracks the actual moisture content independently of temperature;
#      a sustained high D2m over land after landfall is the atmospheric
#      fingerprint of the Brown Ocean Effect.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import traceback

from utils import (
    std_print,
    is_point_land,
    add_land_ocean_shading,
    calculate_buffer_deg,
    get_time_coord,
    common_bbox,
    legend_style,
)


def tp_swvl1_timeseries(
    track_df,
    track_id,
    tp,
    swvl1,
    valid_times,
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
        if not isinstance(valid_times, (list, np.ndarray, pd.DatetimeIndex)):
            raise ValueError("valid_times must be datetime-like")

        # Create a copy of swvl1
        swvl1_land = swvl1.copy()

        # Get land array for shading and for zeroing ocean values
        if track_df is not None and "is_land" in track_df.columns:
            is_land_array = track_df["is_land"].values

            # Set SWVL1 to 0 for ocean points (not land)
            swvl1_land[~is_land_array] = 0
        else:
            is_land_array = np.zeros(len(valid_times), dtype=bool)
            if originates_over_land:
                is_land_array[:] = True
            elif landfall_index > 0:
                is_land_array[landfall_index:] = True
                swvl1_land[:landfall_index] = 0

        valid_data_mask = ~(np.isnan(tp) | np.isnan(swvl1_land))
        valid_count = np.sum(valid_data_mask)
        if valid_count < 2:
            std_print(
                "WARN", "ts", "Insufficient valid data", f"{valid_count} < 2 points"
            )
            return False
        if len(tp) != len(swvl1_land) or len(tp) != len(valid_times):
            std_print("WARN", "ts", "Data length mismatch", "swvl1_land != tp != times")
            return False

        # std_print("DEBUG", "ts", "Extracted TP-SWVL1", len(valid_times))

        # Figure
        fig, ax1 = plt.subplots(figsize=(18, 6))

        # Determine landfall status
        if originates_over_land:
            status_text = "Originates over land"
            status_color = "blue"
        elif landfall_index == -1:
            status_text = "No Landfall"
            status_color = "gray"
        else:
            status_text = "Landfall"
            status_color = "red"

        # Plot TP on primary axis
        color_tp = "tab:blue"
        ax1.set_xlabel("Time", fontsize=12, fontweight="bold")
        ax1.set_ylabel(
            "Total Precipitation (tp) (mm/hr)",
            color=color_tp,
            fontsize=10,
            fontweight="bold",
        )
        line1 = ax1.plot(
            valid_times,
            tp,
            "-o",
            color=color_tp,
            linewidth=2.5,
            markersize=5,
            label="Total Precipitation",
            alpha=0.8,
            zorder=4,
        )
        ax1.tick_params(axis="y", labelcolor=color_tp)
        ax1.grid(True, alpha=0.5, linestyle="--", zorder=0)
        add_land_ocean_shading(
            ax1,
            pd.to_datetime(valid_times),
            is_land_array,
            land_color="wheat",
            ocean_color="lightblue",
        )
        if (
            landfall_index != -1
            and landfall_index < len(valid_times)
            and not originates_over_land
        ):
            landfall_time = valid_times[landfall_index]
            ax1.axvline(
                x=landfall_time,
                color=status_color,
                linestyle="--",
                linewidth=2.5,
                alpha=0.8,
                label=status_text,
                zorder=5,
            )

        # Plot SWVL1 on secondary axis (using swvl1_land with ocean=0)
        ax2 = ax1.twinx()
        color_swvl1 = "tab:brown"
        ax2.set_ylabel(
            "Soil Water Volumetric Layer-1 (swvl1) (m³/m³)",
            color=color_swvl1,
            fontsize=12,
            fontweight="bold",
        )
        line2 = ax2.plot(
            valid_times,
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

        # Set y-limits to include zero on both axes
        tp_clean = tp[~np.isnan(tp)]
        swvl1_clean = swvl1_land[~np.isnan(swvl1_land)]
        tp_max = np.max(tp_clean) if len(tp_clean) > 0 else 1
        swvl1_max = np.max(swvl1_clean) if len(swvl1_clean) > 0 else 0.1
        tp_pad = (tp_max) * 0.1 if tp_max > 0 else 0.5
        swvl1_pad = (swvl1_max) * 0.1 if swvl1_max > 0 else 0.05

        ax1.set_ylim(bottom=0, top=tp_max + tp_pad)
        ax2.set_ylim(bottom=0, top=swvl1_max + swvl1_pad)
        ax1.axhline(
            y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.3, zorder=1
        )
        ax2.axhline(
            y=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.3, zorder=1
        )

        # Legend
        handles = []
        labels = []
        handles.extend(line1 + line2)
        labels.extend([l.get_label() for l in line1 + line2])

        if (
            landfall_index != -1
            and landfall_index < len(valid_times)
            and not originates_over_land
        ):
            handles.append(
                plt.Line2D([0], [0], color=status_color, linestyle="--", linewidth=2.5)
            )
            labels.append(status_text)

        handles.append(Patch(facecolor="lightblue", alpha=0.5))
        labels.append("Ocean Phase")
        handles.append(Patch(facecolor="wheat", alpha=0.5))
        labels.append("Land Phase")

        if handles:
            ax1.legend(handles, labels, loc="upper right", **legend_style())

        title = f"Track {track_id}: Precipitation & Soil Water Content Evolution"
        if originates_over_land:
            title += "\n(Land-Originated MD)"
        elif landfall_index != -1:
            title += "\n(Ocean-Originated, Landfall Detected)"
        else:
            title += "\n(Ocean-Originated, No Landfall)"
        ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)

        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax1.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%d/%m/%Y %H:%M")
        )

        # Save
        os.makedirs(output_dir, exist_ok=True)
        tp_swvl1_timeseries_plot = f"track_{track_id}_TP-SWVL1_time_series.png"
        tp_swvl1_timeseries_plot_path = os.path.join(
            output_dir, tp_swvl1_timeseries_plot
        )
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
                "Insufficient data points",
                f"{len(flux_df) if flux_df is not None else 0} < 3",
            )
            return False

        # std_print("DEBUG", "ts", "Extracted heat-flux points", len(flux_df))

        required_cols = [
            "time",
            "latent_heat_flux",
            "sensible_heat_flux",
            "total_heat_flux",
        ]
        missing_cols = [col for col in required_cols if col not in flux_df.columns]
        if missing_cols:
            std_print("ERROR", "ts", "Missing columns", missing_cols)
            return False

        if "is_land" not in flux_df.columns:
            std_print("DEBUG", "ts", "Creating is_land column", "Default to False")
            flux_df = flux_df.copy()
            flux_df["is_land"] = False

        times = pd.to_datetime(flux_df["time"])
        is_land_arr = flux_df["is_land"].values.astype(bool)

        # Detect landfall
        landfall_time = None
        for i in range(1, len(flux_df)):
            if not is_land_arr[i - 1] and is_land_arr[i]:
                landfall_time = times.iloc[i]
                break

        try:
            times = pd.to_datetime(flux_df["time"])
        except Exception as e:
            std_print("ERROR", "ts", "Time conversion failed", str(e))
            return False

        is_land_array = flux_df["is_land"].values

        # Figure
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        # Panel 1: Latent Heat Flux
        ax1 = axes[0]
        ax1.plot(
            times,
            flux_df["latent_heat_flux"],
            color="green",
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.8,
            zorder=3,
        )
        add_land_ocean_shading(
            ax1, times, is_land_array, land_color="wheat", ocean_color="lightblue"
        )

        # Landfall line
        if landfall_time is not None:
            ax1.axvline(
                landfall_time,
                color="red",
                linestyle="--",
                linewidth=1.8,
                label="Landfall",
                zorder=5,
            )

        ax1.set_ylabel("Latent Heat Flux (W/m²)", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.4, linestyle="--", zorder=0)
        ax1.set_title(
            f"Track {track_id}: Surface Heat Flux Evolution",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        # Panel 2: Sensible Heat Flux
        ax2 = axes[1]
        ax2.plot(
            times,
            flux_df["sensible_heat_flux"],
            color="orange",
            linewidth=2.5,
            marker="s",
            markersize=6,
            alpha=0.8,
            zorder=3,
        )
        add_land_ocean_shading(
            ax2, times, is_land_array, land_color="wheat", ocean_color="lightblue"
        )

        # Landfall line
        if landfall_time is not None:
            ax2.axvline(
                landfall_time,
                color="red",
                linestyle="--",
                linewidth=1.8,
                label="Landfall",
                zorder=5,
            )

        ax2.set_ylabel("Sensible Heat Flux (W/m²)", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.4, linestyle="--", zorder=0)

        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
            ax.xaxis.set_major_formatter(
                plt.matplotlib.dates.DateFormatter("%d/%m/%Y %H:%M")
            )
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
            std_print("WARN", "ts", "Insufficient theta_e data")
            return False

        # std_print("DEBUG", "ts", "Extracted theta_e points", len(theta_df))

        required = {"time", "theta_e_K", "t2m_K", "d2m_K", "is_land"}
        missing = required - set(theta_df.columns)
        if missing:
            std_print(
                "ERROR", "ts", "theta_e_timeseries: missing columns", str(missing)
            )
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
        add_land_ocean_shading(
            ax, times, is_land_arr, land_color="wheat", ocean_color="lightblue"
        )

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
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter("%d/%m/%Y %H:%M")
        )
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

    lat_values = ds.latitude.values
    lat_descending = lat_values[0] > lat_values[-1] if len(lat_values) > 1 else True

    track = track_df.sort_values("time").reset_index(drop=True)
    if "is_land" not in track.columns:
        track["is_land"] = track.apply(
            lambda r: is_point_land(r["latitude"], r["longitude"]), axis=1
        )

    has_accumulated = "slhf" in ds.data_vars and "sshf" in ds.data_vars
    has_average = "avg_slhtf" in ds.data_vars and "avg_ishf" in ds.data_vars
    if not (has_accumulated or has_average):
        std_print("ERROR", "ts", "No valid heat flux variables found")
        std_print(None, None, "Check variables", list(ds.data_vars))
        return None

    flux_records = []
    for i in range(1, len(track)):
        row_now = track.iloc[i]
        row_prev = track.iloc[i - 1]

        t_now = pd.to_datetime(row_now["time"])
        t_prev = pd.to_datetime(row_prev["time"])

        dt_seconds = int((t_now - t_prev).total_seconds())
        if dt_seconds != 3600:
            std_print("WARN", "ts", f"Computed `dt_seconds` is {dt_seconds} (!= 3600)")
        if dt_seconds == 0:
            std_print("ERROR", "ts", "Time dt_seconds = 0", "Continuing...")
            continue
        elif dt_seconds < 0:
            std_print("ERROR", "ts", "Time dt_seconds < 0", "Continuing...")
            continue

        lat_c = row_now["latitude"]
        lon_c = row_now["longitude"]
        is_land = row_now["is_land"]
        lat_buf, lon_buf = calculate_buffer_deg(row_now["latitude"], buffer_km)

        idx_now = np.argmin(np.abs(ds_times - t_now))
        idx_prev = np.argmin(np.abs(ds_times - t_prev))
        try:
            lat_min = max(lat_c - lat_buf, float(ds.latitude.min()))
            lat_max = min(lat_c + lat_buf, float(ds.latitude.max()))
            lon_min = max(lon_c - lon_buf, float(ds.longitude.min()))
            lon_max = min(lon_c + lon_buf, float(ds.longitude.max()))
            if lat_min >= lat_max or lon_min >= lon_max:
                continue

            ds_now = ds.isel({time_coord: idx_now})
            ds_prev = ds.isel({time_coord: idx_prev})

            if lat_descending:
                lat_slice = slice(lat_max, lat_min)
            else:
                lat_slice = slice(lat_min, lat_max)
            lon_slice = slice(lon_min, lon_max)

            try:
                sub_now = ds_now.sel(latitude=lat_slice, longitude=lon_slice)
                sub_prev = ds_prev.sel(latitude=lat_slice, longitude=lon_slice)
            except Exception as e:
                std_print("ERROR", "ts", f"Selection failed for {t_now}", {str(e)})
                std_print("DEBUG", "ts", "Alternative approach", "Using `where()`")
                lat_mask = (ds_now.latitude >= lat_min) & (ds_now.latitude <= lat_max)
                lon_mask = (ds_now.longitude >= lon_min) & (ds_now.longitude <= lon_max)
                if lat_mask.any() and lon_mask.any():
                    sub_now = ds_now.where(lat_mask & lon_mask, drop=True)
                    sub_prev = ds_prev.where(lat_mask & lon_mask, drop=True)
                else:
                    std_print(
                        "WARN",
                        "ts",
                        f"No data within buffer at {t_now}. Using nearest point",
                    )
                    sub_now = ds_now.sel(
                        latitude=lat_c, longitude=lon_c, method="nearest"
                    ).expand_dims(["latitude", "longitude"])
                    sub_prev = ds_prev.sel(
                        latitude=lat_c, longitude=lon_c, method="nearest"
                    ).expand_dims(["latitude", "longitude"])

            if (
                sub_now.sizes.get("latitude", 0) == 0
                or sub_now.sizes.get("longitude", 0) == 0
            ):
                std_print("WARN", "ts", f"Empty subset at {t_now}. Using nearest point")
                sub_now = ds_now.sel(
                    latitude=lat_c, longitude=lon_c, method="nearest"
                ).expand_dims(["latitude", "longitude"])
                sub_prev = ds_prev.sel(
                    latitude=lat_c, longitude=lon_c, method="nearest"
                ).expand_dims(["latitude", "longitude"])

            if "avg_slhtf" in sub_now.data_vars and "avg_ishf" in sub_now.data_vars:
                slhf_w = float(sub_now["avg_slhtf"].mean().values)
                sshf_w = float(sub_now["avg_ishf"].mean().values)
            elif "slhf" in sub_now.data_vars and "sshf" in sub_now.data_vars:
                if "slhf" in sub_now.data_vars and "slhf" in sub_prev.data_vars:
                    slhf_now = float(sub_now["slhf"].mean().values)
                    slhf_prev = float(sub_prev["slhf"].mean().values)
                    slhf_w = -(slhf_now - slhf_prev) / dt_seconds
                else:
                    slhf_w = np.nan
                if "sshf" in sub_now.data_vars and "sshf" in sub_prev.data_vars:
                    sshf_now = float(sub_now["sshf"].mean().values)
                    sshf_prev = float(sub_prev["sshf"].mean().values)
                    sshf_w = -(sshf_now - sshf_prev) / dt_seconds
                else:
                    sshf_w = np.nan
            else:
                std_print("WARN", "ts", "No valid heat flux variables found at", t_now)
                continue

            if np.isnan(slhf_w) or np.isnan(sshf_w):
                std_print(
                    "WARN",
                    "ts",
                    f"NaN values at {t_now}",
                    f"SLHF={slhf_w:.4f}, SSHF={sshf_w:.4f}",
                )
                continue

            flux_records.append(
                {
                    "time": t_now,
                    "latent_heat_flux": slhf_w,
                    "sensible_heat_flux": sshf_w,
                    "total_heat_flux": slhf_w + sshf_w,
                    "latitude": lat_c,
                    "longitude": lon_c,
                    "is_land": is_land,
                }
            )
        except Exception as e:
            std_print("ERROR", "ts", f"Processing time step at {t_now}", str(e))
            traceback.print_exc()
            continue

    if not flux_records:
        std_print("ERROR", "ts", "No heat-flux data extracted")
        return None
    return pd.DataFrame(flux_records)


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
