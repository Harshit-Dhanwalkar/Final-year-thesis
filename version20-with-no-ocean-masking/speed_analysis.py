"""
The average translational velocity of a monsoon depression is approx 2-6 m/s, to roughly 7-22 km/h.
"""

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
    legend_style,
)

input_dir = "output_*"
EARTH_R = 6371.0  # km


def compute_track_speeds(track_df):
    """Calculate translational speeds between consecutive track points"""
    track_df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(track_df["time"]).values
    coords = np.radians(track_df[["latitude", "longitude"]].values)
    lat, lon = coords[:, 0], coords[:, 1]
    dlat = np.diff(lat, prepend=lat[0])
    dlon = np.diff(lon, prepend=lon[0])

    # n = len(track_df)
    # speeds_kmh = np.full(n, np.nan)
    # distances_km = np.full(n, np.nan)
    #
    # # Calculate speed between consecutive points
    # for i in range(n - 1):
    #     time_gap = (times[i+1] - times[i]).astype('timedelta64[s]').astype(float) / 3600.0
    #     # Accept gaps between 0.5 and 12 hours
    #     if 0.5 <= time_gap <= 12:
    #         dist = geodesic((lats[i], lons[i]), (lats[i+1], lons[i+1])).km
    #         if time_gap <= 0:
    #             speeds_kmh[i] = np.nan
    #             continue
    #         if np.isnan(lats[i]) or np.isnan(lons[i]) or np.isnan(lats[i+1]) or np.isnan(lons[i+1]):
    #             speeds_kmh[i] = np.nan
    #             distances_km[i] = np.nan
    #             continue
    #         speed = dist / time_gap
    #         speeds_kmh[i] = speed
    #         distances_km[i] = dist
    #
    # # Forward fill last point
    # if n > 1 and not np.isnan(speeds_kmh[-2]):
    #     speeds_kmh[-1] = speeds_kmh[-2]
    #     distances_km[-1] = distances_km[-2]
    #
    # # Smooth with 3-point rolling mean
    # speeds_series = pd.Series(speeds_kmh)
    # speeds_smoothed = speeds_series.rolling(window=3, center=True, min_periods=1).mean().values
    #
    # track_df['speed_kmh'] = speeds_smoothed
    # track_df['speed_ms'] = speeds_smoothed / 3.6
    # track_df['distance_km'] = distances_km

    # Haversine-like approximation for distance
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    track_df["distance_km"] = EARTH_R * c  # Earth Radius in km

    # Calculate time delta in hours
    dt_hours = track_df["time"].diff().dt.total_seconds() / 3600.0
    dt_hours.iloc[0] = np.nan  # First point has no speed
    # Speeds
    track_df["speed_kmh"] = track_df["distance_km"] / dt_hours
    track_df["speed_ms"] = track_df["speed_kmh"] / 3.6
    # Fill first row NaN with second row's speed to avoid plotting gaps
    track_df["speed_kmh"] = track_df["speed_kmh"].ffill().bfill()
    track_df["speed_ms"] = track_df["speed_ms"].ffill().bfill()
    return track_df


def plot_speed_analysis_kmh(track_df, track_id, output_dir):
    try:
        track_id = int(track_id)
        # Validate input
        if track_df is None or len(track_df) < 2:
            std_print(
                "WARN",
                "sp",
                f"Track {track_id} Insufficient data",
                f"{len(track_df) if track_df is not None else 0} points",
            )
            return False
        # Check required columns
        if "time" not in track_df.columns:
            std_print("ERROR", "sp", f"Track {track_id}", "Missing time column")
            return False
        if "speed_kmh" not in track_df.columns:
            std_print(
                "WARN",
                "sp",
                f"Track {track_id}",
                "Missing speed_kmh column, computing speeds",
            )
            track_df = compute_track_speeds(track_df)
        # Ensure is_land column exists
        if "is_land" not in track_df.columns:
            std_print("DEBUG", "sp", f"Track {track_id}", "Creating is_land column")
            track_df = track_df.copy()
            track_df["is_land"] = [
                is_point_land(lat, lon)
                for lat, lon in zip(track_df["latitude"], track_df["longitude"])
            ]

        # Convert time to datetime
        times = pd.to_datetime(track_df["time"])
        is_land_array = track_df["is_land"].values

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        # Plot speed line
        ax.plot(
            times,
            track_df["speed_kmh"],
            "-o",
            color="blue",
            linewidth=2.5,
            markersize=6,
            label="Translation Speed",
            alpha=0.85,
            zorder=3,
        )
        # ocean/land shading
        add_land_ocean_shading(
            ax, times, is_land_array, land_color="wheat", ocean_color="lightblue"
        )

        # Labels and title
        ax.set_xlabel("Time", fontsize=12, fontweight="bold")
        ax.set_ylabel("Translation Speed (km/h)", fontsize=12, fontweight="bold")

        # Calculate statistics
        # land_pct = np.mean(is_land_array) * 100 if len(is_land_array) > 0 else 0
        # land_pts = np.sum(is_land_array)
        # ocean_pts = len(is_land_array) - land_pts

        # Legend
        lines = [ax.get_lines()[0]]
        lines.append(Patch(facecolor="lightblue", alpha=0.5, label="Ocean Phase"))
        lines.append(Patch(facecolor="wheat", alpha=0.5, label="Land Phase"))

        legend_handles, labels = ax.get_legend_handles_labels()
        if legend_handles:
            ax.legend(
                legend_handles + lines,
                ["Translation Speed", "Ocean Phase", "Land Phase"],
                loc="upper right",
                **legend_style(),
            )

        title = f"Track {track_id}: Translation Speed (km/h)"
        # title += f'\nLand: {land_pts} points ({land_pct:.1f}%) | Ocean: {ocean_pts} points ({100-land_pct:.1f}%)'
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

        # Grid and formatting
        ax.grid(True, alpha=0.4, linestyle="--", zorder=0)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        # Save
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f"track_{int(track_id)}_speed_kmh.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        std_print("SAVED", "sp", "Speed plot (km/h) with ocean/land shading", outpath)
        return True
    except Exception as e:
        std_print("ERROR", "sp", f"Track {track_id} Speed plot (km/h) failed", str(e))
        traceback.print_exc()
        return False


def plot_speed_analysis_ms(track_df, track_id, output_dir):
    try:
        track_id = int(track_id)
        # Validate input
        if track_df is None or len(track_df) < 2:
            std_print(
                "WARN",
                "sp",
                f"Track {track_id} Insufficient data",
                f"{len(track_df) if track_df is not None else 0} points",
            )
            return False
        # Check required columns
        if "time" not in track_df.columns:
            std_print("ERROR", "sp", f"Track {track_id}", "Missing time column")
            return False
        if "speed_ms" not in track_df.columns:
            std_print(
                "WARN",
                "sp",
                f"Track {track_id}",
                "Missing speed_ms column, computing speeds",
            )
            track_df = compute_track_speeds(track_df)
        # Ensure is_land column exists
        if "is_land" not in track_df.columns:
            std_print("DEBUG", "sp", f"Track {track_id}", "Creating is_land column")
            track_df = track_df.copy()
            track_df["is_land"] = [
                is_point_land(lat, lon)
                for lat, lon in zip(track_df["latitude"], track_df["longitude"])
            ]

        # Convert time to datetime
        times = pd.to_datetime(track_df["time"])
        is_land_array = track_df["is_land"].values

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        # Plot speed line
        ax.plot(
            times,
            track_df["speed_ms"],
            "-o",
            color="blue",
            linewidth=2.5,
            markersize=6,
            label="Translation Speed",
            alpha=0.85,
            zorder=3,
        )
        # ocean/land shading
        add_land_ocean_shading(
            ax, times, is_land_array, land_color="wheat", ocean_color="lightblue"
        )

        # Labels and title
        ax.set_xlabel("Time", fontsize=12, fontweight="bold")
        ax.set_ylabel("Translation Speed (m/s)", fontsize=12, fontweight="bold")

        # Calculate statistics
        # land_pct = np.mean(is_land_array) * 100 if len(is_land_array) > 0 else 0
        # land_pts = np.sum(is_land_array)
        # ocean_pts = len(is_land_array) - land_pts

        title = f"Track {track_id}: Translation Speed (m/s)"
        # title += f'\nLand: {land_pts} points ({land_pct:.1f}%) | Ocean: {ocean_pts} points ({100-land_pct:.1f}%)'
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

        # Legend
        lines = [ax.get_lines()[0]]
        lines.append(Patch(facecolor="lightblue", alpha=0.5, label="Ocean Phase"))
        lines.append(Patch(facecolor="wheat", alpha=0.5, label="Land Phase"))

        legend_handles, labels = ax.get_legend_handles_labels()
        if legend_handles:
            ax.legend(
                legend_handles + lines,
                ["Translation Speed", "Ocean Phase", "Land Phase"],
                loc="upper right",
                **legend_style(),
            )

        # Grid and formatting
        ax.grid(True, alpha=0.4, linestyle="--", zorder=0)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.tight_layout()

        # Save
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f"track_{int(track_id)}_speed_ms.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        std_print("SAVED", "sp", "Speed plot (m/s) with ocean/land shading", outpath)
        return True
    except Exception as e:
        std_print("ERROR", "sp", f"Track {track_id} Speed plot (m/s) failed", str(e))
        traceback.print_exc()
        return False


def main():
    track_files = []
    for file in os.listdir(input_dir):
        if (
            file.startswith("track_")
            and file.endswith(".csv")
            and "speed" not in file
            and "detections" not in file
        ):
            track_files.append(os.path.join(input_dir, file))
    if not track_files:
        std_print("ERROR", "sp", "No Track CSV files found in", input_dir)
        return
    std_print("INFO", "sp", "Track files found for speed analysis", len(track_files))

    all_tracks_data = {}
    success_count = 0
    # Process each track
    for fpath in sorted(track_files):
        try:
            df = pd.read_csv(fpath)
            # Extract track ID from filename
            fname = os.path.basename(fpath)
            track_id = int(fname.split("_")[1].split(".")[0])

            # Land/ocean classification
            df["is_land"] = [
                is_point_land(lat, lon)
                for lat, lon in zip(df["latitude"], df["longitude"])
            ]

            # Compute speeds
            df = compute_track_speeds(df)
            all_tracks_data[track_id] = {
                "df": df,
                "speed_kmh": df["speed_kmh"].values,
                "speed_ms": df["speed_ms"].values,
            }

            # Plot track speeds (both km/h and m/s)
            success_kmh = plot_speed_analysis_kmh(df, track_id, input_dir)
            success_ms = plot_speed_analysis_ms(df, track_id, input_dir)
            if success_kmh and success_ms:
                success_count += 1

            # Print statistics
            mean_spd = df["speed_kmh"].mean()
            std_spd = df["speed_kmh"].std()
            std_print(
                "INFO",
                "sp",
                f"Track {track_id}",
                f"Mean: {mean_spd:.2f}±{std_spd:.2f} km/h ({mean_spd / 3.6:.2f}±{std_spd / 3.6:.2f} m/s)",
            )
        except Exception as e:
            std_print("ERROR", "sp", f"Processing {fpath}", str(e))
            traceback.print_exc()
            continue
    if not all_tracks_data:
        std_print("ERROR", "sp", "No tracks processed successfully")
        return

    std_print(
        "INFO",
        "sp",
        f"Successfully plotted {success_count}/{len(all_tracks_data)} tracks",
    )
    std_print("SAVED", "sp", "Translation speed plots saved to", input_dir)


if __name__ == "__main__":
    main()
