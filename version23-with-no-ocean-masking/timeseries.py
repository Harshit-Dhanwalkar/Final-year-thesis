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


def extract_heat_flux_timeseries(ds, track_df, track_id, buffer_km=100):
    """
    Handles two ERA5 conventions:
    1. Accumulated: slhf, sshf (J m⁻²) - negative = away from surface
    2. Instantaneous Average: avg_slhtf, avg_ishf (W m⁻²)
    """
    track_id = int(track_id)
    # std_print("DEBUG", "ts", "Dataset variables available", list(ds.data_vars.keys()))
    ds, time_coord = get_time_coord(ds)
    ds_times = pd.to_datetime(ds.time.values)

    # DEBUG: Check latitude order
    lat_values = ds.latitude.values
    lat_descending = lat_values[0] > lat_values[-1] if len(lat_values) > 1 else True
    std_print(
        "DEBUG",
        "ts",
        f"Latitude order : {'descending' if lat_descending else 'ascending'}",
    )

    track = track_df.sort_values("time").reset_index(drop=True)
    if "is_land" not in track.columns:
        track["is_land"] = track.apply(
            lambda r: is_point_land(r["latitude"], r["longitude"]), axis=1
        )

    # Detect which flux type is available
    has_accumulated = "slhf" in ds.data_vars and "sshf" in ds.data_vars
    has_average = "avg_slhtf" in ds.data_vars and "avg_ishf" in ds.data_vars
    if not (has_accumulated or has_average):
        std_print(
            "ERROR",
            "ts",
            "No valid heat flux variables found. Check dataset variables",
            list(ds.data_vars),
        )
        return None
    flux_type = "average" if has_average else "accumulated"
    std_print("DEBUG", "ts", f"Using flux type : {flux_type}")

    # Container
    flux_records = []
    for i in range(1, len(track)):
        row_now = track.iloc[i]
        row_prev = track.iloc[i - 1]

        t_now = pd.to_datetime(row_now["time"])
        t_prev = pd.to_datetime(row_prev["time"])

        dt_seconds = (t_now - t_prev).total_seconds()
        if dt_seconds == 0:
            std_print(
                "ERROR", "ts", f"Track {track_id} Time dt_seconds = 0", "Continuing..."
            )
            continue
        elif dt_seconds < 0:
            std_print(
                "ERROR", "ts", f"Track {track_id} Time dt_seconds < 0", "Continuing..."
            )
            continue

        lat_c = row_now["latitude"]
        lon_c = row_now["longitude"]
        is_land = row_now["is_land"]
        lat_buf, lon_buf = calculate_buffer_deg(row_now["latitude"], buffer_km)
        # std_print('DEBUG', 'hf', f'Dataset structure: lat range={float(ds.latitude.min())} to {float(ds.latitude.max())}, lon range={float(ds.longitude.min())} to {float(ds.longitude.max())}')

        # Find nearest time indices
        idx_now = np.argmin(np.abs(ds_times - t_now))
        idx_prev = np.argmin(np.abs(ds_times - t_prev))
        try:
            # Create bounds for spatial selection
            lat_min = lat_c - lat_buf
            lat_max = lat_c + lat_buf
            lon_min = lon_c - lon_buf
            lon_max = lon_c + lon_buf
            # std_print('DEBUG', 'hf', f'Requested bounds: lat={float(lat_min):.4f} to {float(lat_max):.4f}, lon={float(lon_min):.4f} to {float(lon_max):.4f}')

            # Clamp to dataset bounds
            lat_min = max(lat_min, float(ds.latitude.min()))
            lat_max = min(lat_max, float(ds.latitude.max()))
            lon_min = max(lon_min, float(ds.longitude.min()))
            lon_max = min(lon_max, float(ds.longitude.max()))
            # Check validity
            if lat_min >= lat_max or lon_min >= lon_max:
                continue
            # std_print('DEBUG', 'hf', f'Bounds after clipping: lat={float(lat_min):.4f} to {float(lat_max):.4f}, lon={float(lon_min):.4f} to {float(lon_max):.4f}')

            #####
            # sub_now = ds.isel({time_coord: idx_now}).sel(
            #     latitude=slice(lat_max, lat_min),  # North to South
            #     longitude=slice(lon_min, lon_max)   # West to East
            # )
            # # Check valid data
            # if sub_now.sizes.get('latitude', 0) == 0 or sub_now.sizes.get('longitude', 0) == 0:
            #     std_print('WARN', 'hf', f'No data within buffer at', t_now)
            #     continue
            #####
            # sub_now = ds_now.sel(
            #     latitude=slice(lat_max, lat_min),  # North to South
            #     longitude=slice(lon_min, lon_max)  # West to East
            # )
            # sub_prev = ds_prev.sel(
            #     latitude=slice(lat_max, lat_min),  # North to South
            #     longitude=slice(lon_min, lon_max)  # West to East
            # )
            # # Check if subset is empty
            # if sub_now.sizes.get('latitude', 0) == 0 or sub_now.sizes.get('longitude', 0) == 0:
            #     std_print('WARN', 'hf', f'No data within buffer at', t_now)
            #     continue
            #####
            # Select time slices
            ds_now = ds.isel({time_coord: idx_now})
            ds_prev = ds.isel({time_coord: idx_prev})

            # Handle latitude order for slicing
            if lat_descending:
                # ERA5 order: north to south (descending)
                lat_slice = slice(lat_max, lat_min)
            else:
                # Ascending order: south to north
                lat_slice = slice(lat_min, lat_max)
            lon_slice = slice(lon_min, lon_max)
            # std_print('DEBUG', 'hf', f'Using slice: lat=slice({float(lat_slice.start):.4f}, {float(lat_slice.stop):.4f}), lon=slice({float(lon_slice.start):.4f}, {float(lon_slice.stop):.4f})')

            try:
                sub_now = ds_now.sel(latitude=lat_slice, longitude=lon_slice)
                sub_prev = ds_prev.sel(latitude=lat_slice, longitude=lon_slice)
            except Exception as e:
                std_print(
                    "WARN",
                    "ts",
                    f"Selection failed for {t_now}:,  {str(e)}. Using alternative approach.",
                )
                # Alternative approach using where()
                lat_mask = (ds_now.latitude >= lat_min) & (ds_now.latitude <= lat_max)
                lon_mask = (ds_now.longitude >= lon_min) & (ds_now.longitude <= lon_max)
                if lat_mask.any() and lon_mask.any():
                    sub_now = ds_now.where(lat_mask & lon_mask, drop=True)
                    sub_prev = ds_prev.where(lat_mask & lon_mask, drop=True)
                else:
                    std_print(
                        "WARN",
                        "ts",
                        f"No data within buffer at {t_now}. Using nearest point.",
                    )
                    sub_now = ds_now.sel(
                        latitude=lat_c, longitude=lon_c, method="nearest"
                    ).expand_dims(["latitude", "longitude"])
                    sub_prev = ds_prev.sel(
                        latitude=lat_c, longitude=lon_c, method="nearest"
                    ).expand_dims(["latitude", "longitude"])

            # Check if subset is valid
            if (
                sub_now.sizes.get("latitude", 0) == 0
                or sub_now.sizes.get("longitude", 0) == 0
            ):
                std_print(
                    "WARN", "ts", f"Empty subset at {t_now}. Using nearest point."
                )
                sub_now = ds_now.sel(
                    latitude=lat_c, longitude=lon_c, method="nearest"
                ).expand_dims(["latitude", "longitude"])
                sub_prev = ds_prev.sel(
                    latitude=lat_c, longitude=lon_c, method="nearest"
                ).expand_dims(["latitude", "longitude"])
            # Check for heat flux variables and Compute fluxes
            if "avg_slhtf" in sub_now.data_vars and "avg_ishf" in sub_now.data_vars:
                slhf_w = float(sub_now["avg_slhtf"].mean().values)
                sshf_w = float(sub_now["avg_ishf"].mean().values)
                # std_print('DEBUG', 'hf', f'Using avg_slhtf/avg_ishf: LHF={slhf_w:.4f}, SHF={sshf_w:.4f}')
            elif "slhf" in sub_now.data_vars and "sshf" in sub_now.data_vars:
                # De-accumulate accumulated fluxes J m-2 -> W m-2
                dt_seconds = 3600.0  # 1-hour difference
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
                std_print(
                    "DEBUG",
                    "ts",
                    f"Using slhf/sshf (deaccumulated): LHF={slhf_w:.4f}, SHF={sshf_w:.4f}",
                )
            else:
                std_print("WARN", "ts", "No valid heat flux variables found at", t_now)
                continue
            if np.isnan(slhf_w) or np.isnan(sshf_w):
                std_print(
                    "WARN",
                    "ts",
                    f"Track {track_id} NaN values encountered at {t_now}",
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
            # std_print('DEBUG', 'hf', f'Successfully extracted fluxes at {t_now}: LHF={slhf_w:.1f}, SHF={sshf_w:.1f}')
        except Exception as e:
            std_print("ERROR", "ts", f"Error processing time step at {t_now}", str(e))
            traceback.print_exc()
            continue

    if not flux_records:
        std_print("WARN", "ts", f"Track {track_id}", "No heat-flux data extracted")
        return None

    std_print(
        "INFO",
        "ts",
        f"Extracted heat flux records for Track {track_id}",
        f"{len(flux_records)} records",
    )
    return pd.DataFrame(flux_records)


def tp_swvl1_timeseries(
    tp,
    swvl1,
    valid_times,
    landfall_index,
    track_id,
    output_dir,
    originates_over_land=True,
    track_df=None,
):
    track_id = int(track_id)
    std_print("INFO", "ts", f" --- Track {track_id} TP-SWVL1 Time series plot ---")
    try:
        if not isinstance(tp, np.ndarray):
            tp = np.array(tp)
        if not isinstance(swvl1, np.ndarray):
            swvl1 = np.array(swvl1)
        if not isinstance(valid_times, (list, np.ndarray, pd.DatetimeIndex)):
            raise ValueError("valid_times must be datetime-like")

        # Create a copy of swvl1 to modify
        swvl1_modified = swvl1.copy()

        # Get land array for shading and for zeroing ocean values
        if track_df is not None and "is_land" in track_df.columns:
            is_land_array = track_df["is_land"].values

            # Set SWVL1 to 0 for ocean points (not land)
            swvl1_modified[~is_land_array] = 0
            std_print(
                "DEBUG",
                "ts",
                f"Track {track_id}",
                f"Set SWVL1=0 for {np.sum(~is_land_array)} ocean points",
            )
        else:
            is_land_array = np.zeros(len(valid_times), dtype=bool)
            if originates_over_land:
                is_land_array[:] = True
                # All points are land, no modification needed
            elif landfall_index > 0:
                is_land_array[landfall_index:] = True
                # Set pre-landfall points (ocean) to 0
                swvl1_modified[:landfall_index] = 0
                std_print(
                    "DEBUG",
                    "ts",
                    f"Track {track_id}",
                    f"Set SWVL1=0 for {landfall_index} pre-landfall ocean points",
                )

        valid_data_mask = ~(np.isnan(tp) | np.isnan(swvl1_modified))
        valid_count = np.sum(valid_data_mask)
        if valid_count < 2:
            std_print(
                "WARN",
                "ts",
                f"Track {track_id} Insufficient valid data",
                f"{valid_count} points",
            )
            return False
        if len(tp) != len(swvl1_modified) or len(tp) != len(valid_times):
            std_print("WARN", "ts", f"Track {track_id}", "Data length mismatch")
            return False

        # Figure
        fig, ax1 = plt.subplots(figsize=(18, 6))
        bbox_style = common_bbox()

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
        ax1.set_xlabel("Time", fontsize=11, fontweight="bold")
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

        # Plot SWVL1 on secondary axis (using modified version with ocean=0)
        ax2 = ax1.twinx()
        color_swvl1 = "tab:brown"
        ax2.set_ylabel(
            "Soil Water Volumetric Layer 1 (swvl1) (m³/m³)",
            color=color_swvl1,
            fontsize=11,
            fontweight="bold",
        )
        line2 = ax2.plot(
            valid_times,
            swvl1_modified,
            "s-",
            color=color_swvl1,
            linewidth=2.5,
            markersize=5,
            label="Soil Water Volumetric Level-1 (Ocean=0)",
            alpha=0.8,
            zorder=4,
        )
        ax2.tick_params(axis="y", labelcolor=color_swvl1)

        # Set y-limits to include zero on both axes
        tp_clean = tp[~np.isnan(tp)]
        swvl1_clean = swvl1_modified[~np.isnan(swvl1_modified)]
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
        labels.append("Ocean Phase (SWVL1=0)")
        handles.append(Patch(facecolor="wheat", alpha=0.5))
        labels.append("Land Phase")

        legend_handles, labels = ax1.get_legend_handles_labels()
        if legend_handles:
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

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(
            output_dir, f"track_{track_id}_TP-SWVL1_time_series.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "ts", "TP-SWVL1 time series (ocean=0)", plot_path)
        return True
    except Exception as e:
        std_print("ERROR", "ts", f"Track {track_id} TP-SWVL1 plot failed", str(e))
        traceback.print_exc()
        return False


def hf_timeseries(flux_df, track_id, output_dir):
    try:
        track_id = int(track_id)
        # Validate input
        if flux_df is None or len(flux_df) < 3:
            std_print(
                "WARN",
                "ts",
                f"Track {track_id} Insufficient data points",
                f"{len(flux_df) if flux_df is not None else 0} < 3",
            )
            return False

        # Check required columns
        required_cols = [
            "time",
            "latent_heat_flux",
            "sensible_heat_flux",
            "total_heat_flux",
        ]
        missing_cols = [col for col in required_cols if col not in flux_df.columns]
        if missing_cols:
            std_print("ERROR", "ts", f"Track {track_id} Missing columns", missing_cols)
            return False

        # Ensure is_land column exists
        if "is_land" not in flux_df.columns:
            std_print(
                "DEBUG",
                "ts",
                f"Track {track_id}",
                "Creating is_land column (defaulting to False)",
            )
            flux_df = flux_df.copy()
            flux_df["is_land"] = False

        # Convert time to datetime
        try:
            times = pd.to_datetime(flux_df["time"])
        except Exception as e:
            std_print("ERROR", "ts", f"Track {track_id} Time conversion failed", str(e))
            return False

        is_land_array = flux_df["is_land"].values

        # Figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        # plot 1: latent heat flux
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
        # ocean/land shading
        add_land_ocean_shading(
            ax1, times, is_land_array, land_color="wheat", ocean_color="lightblue"
        )
        ax1.set_ylabel("Latent Heat Flux (W/m²)", fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.4, linestyle="--", zorder=0)
        ax1.set_title(
            f"Track {track_id}: Surface Heat Flux Evolution",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        # subplot 2: sensible heat flux
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
        # ocean/land shading
        add_land_ocean_shading(
            ax2, times, is_land_array, land_color="wheat", ocean_color="lightblue"
        )
        ax2.set_ylabel("Sensible Heat Flux (W/m²)", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.4, linestyle="--", zorder=0)

        # subplot 3: total heat flux
        ax3 = axes[2]
        ax3.plot(
            times,
            flux_df["total_heat_flux"],
            color="purple",
            linewidth=2.5,
            marker="D",
            markersize=6,
            alpha=0.8,
            zorder=3,
        )
        # ocean/land shading
        add_land_ocean_shading(
            ax3, times, is_land_array, land_color="wheat", ocean_color="lightblue"
        )
        ax3.set_ylabel("Total Heat Flux (W/m²)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Time", fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.4, linestyle="--", zorder=0)

        legend_handles = [
            Patch(facecolor="lightblue", alpha=0.5, label="Ocean Phase"),
            Patch(facecolor="wheat", alpha=0.5, label="Land Phase"),
        ]
        ax3.legend(handles=legend_handles, loc="lower left", **legend_style())

        # Format x-axis for all subplots
        for ax in axes:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
            ax.xaxis.set_major_formatter(
                plt.matplotlib.dates.DateFormatter("%d/%m/%Y %H:%M")
            )
        plt.tight_layout()

        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(
            output_dir, f"track_{int(track_id)}_heat_flux_time_series.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "ts", "Heat flux timeseries", plot_path)
        return True
    except Exception as e:
        std_print(
            "ERROR", "ts", f"Track {track_id} Heat flux timeseries plot failed", str(e)
        )
        traceback.print_exc()
        return False
