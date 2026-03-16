"""
Elevation Analysis for MD Tracks using SRTM HGT Data

  1. Orographic speed reduction      - terrain slows MDs (barrier effect)
  2. Orographic precipitation ratio  - windward vs flat TP amplification
  3. Elevation vs penetration scatter - terrain controls inland reach
  4. Landfall elevation classification - low-coast vs terrain-facing MDs
  5. Soil texture map (Clapp & Hornberger 1978) - SM memory interpretation

TSI combines three terrain-dependent factors:
  TSI = elevation_effect x slope_effect x roughness_effect
  - elevation_effect: normalized elevation contrast
  - slope_effect: mean slope magnitude
  - roughness_effect: terrain variability (std of elevation)
  Each component normalised to [0, 1].

Dependencies: sudo apt install gdal-bin

Reference: Kirshbaum (2018) JGR-Atmos.
"""

# TODO:
# 1. Add TSI vs. inland penetration distance (km from landfall to dissipation)
# 2. Mean elevation along track vs. speed anomaly (speed relative to pre-landfall baseline)
# FIX:
# 1. Need lag plot seperate and line plot instead of bar chart.
# 2. full_track_df is a flat DataFrame (all tracks concatenated). compute_tsi_for_all_tracks calls tracks_df.items() which only works on a dict. You need to convert before the call
# # Convert full_track_df → dict before calling elevation analysis
# track_dict_for_elev = {
#     tid: full_track_df[full_track_df["track_id"] == tid].copy()
#     for tid in full_track_df["track_id"].unique()
# }
# elevation_metrics = run_elevation_analysis(
#     tracks_df=track_dict_for_elev, coupling_df=fsi_df,
#     hgt_dir=HGT_DIR, output_dir=elevation_dir
# )

import os
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import sobel
from scipy.stats import spearmanr
import traceback

from utils import (
    std_print,
    common_bbox,
    legend_style,
    haversine_distance,
    get_landfall_index,
    pearsonr_corrected,
    is_land,
)

# Soil texture constants (Clapp & Hornberger 1978, Table 2)
# Approximate centre-of-mass coordinates for the main pedological zones over the Indian subcontinent.
# \beta = Clapp-Hornberger exponent; higher \beta -> faster drainage -> shorter SM memory
# Hydraulic conductivity k_sat (m/s) for rough scale.
SOIL_TEXTURE_ZONES = {
    "Gangetic Plains (Clay Loam)": {
        "lat_range": (25, 28),
        "lon_range": (77, 88),
        "beta": 7.12,
        "k_sat": 3.8e-6,
        "color": "#8B4513",
    },
    "Deccan Plateau (Sandy Clay)": {
        "lat_range": (15, 22),
        "lon_range": (74, 80),
        "beta": 5.39,
        "k_sat": 6.3e-6,
        "color": "#D2691E",
    },
    "Indo-Gangetic (Silty Clay)": {
        "lat_range": (22, 25),
        "lon_range": (75, 85),
        "beta": 8.52,
        "k_sat": 2.0e-6,
        "color": "#A0522D",
    },
    "Coastal Alluvium (Loamy Sand)": {
        "lat_range": (10, 18),
        "lon_range": (79, 86),
        "beta": 4.38,
        "k_sat": 1.6e-5,
        "color": "#DEB887",
    },
    "Eastern Ghats (Sandy Loam)": {
        "lat_range": (11, 20),
        "lon_range": (77, 82),
        "beta": 4.90,
        "k_sat": 7.2e-6,
        "color": "#F4A460",
    },
}
# SM memory timescale (days) from Seneviratne et al. (2010):
#   \tau \aprrox 1 / (k_sat x \beta)   (qualitative order-of-magnitude)


def plot_soil_texture_interpretation(results_df, output_dir):
    """
    Bar chart of inferred soil texture zones along track corridors and their implied SM memory timescales.
    """
    os.makedirs(output_dir, exist_ok=True)
    if "lat_landfall" not in results_df.columns:
        std_print(
            "WARN", "ea", "plot_soil_texture_interpretation: no lat_landfall column"
        )
        return

    zone_counts = {z: 0 for z in SOIL_TEXTURE_ZONES}
    for _, row in results_df.iterrows():
        lat = row.get("lat_landfall", np.nan)
        lon = row.get("lon_landfall", np.nan)
        if np.isnan(lat) or np.isnan(lon):
            continue
        for zone, info in SOIL_TEXTURE_ZONES.items():
            if (
                info["lat_range"][0] <= lat <= info["lat_range"][1]
                and info["lon_range"][0] <= lon <= info["lon_range"][1]
            ):
                zone_counts[zone] += 1
                break

    zones = list(zone_counts.keys())
    counts = [zone_counts[z] for z in zones]
    colors = [SOIL_TEXTURE_ZONES[z]["color"] for z in zones]
    betas = [SOIL_TEXTURE_ZONES[z]["beta"] for z in zones]
    k_sats = [SOIL_TEXTURE_ZONES[z]["k_sat"] for z in zones]
    taus = [soil_memory_timescale(b, k) for b, k in zip(betas, k_sats)]

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: track count per zone
    ax = axes[0]
    bars = ax.barh(zones, counts, color=colors, edgecolor="black", linewidth=1)
    ax.set_xlabel("Track count", fontsize=11, fontweight="bold")
    ax.set_title("Landfall Zone - Soil Texture Distribution", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    for i, v in enumerate(counts):
        ax.text(v + 0.05, i, str(v), va="center", fontsize=10)

    # Panel 2: SM memory timescale per zone
    ax = axes[1]
    bars = ax.barh(zones, taus, color=colors, edgecolor="black", linewidth=1, alpha=0.8)
    ax.set_xlabel("SM memory timescale (days)", fontsize=11, fontweight="bold")
    ax.set_title("Inferred SM Memory by Soil Texture", fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    for i, (v, b) in enumerate(zip(taus, betas)):
        ax.text(v + 0.05, i, f"β={b:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(output_dir, "soil_texture_memory.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Soil texture / memory plot", outpath)


class TerrainAnalyzer:
    """
    Loads and analyzes SRTM HGT elevation tiles.
    HGT file naming: N/SxxE/Wyyy.hgt
    """

    def __init__(self, hgt_dir):
        self.hgt_dir = hgt_dir
        self.tiles_cache = {}  # Cache loaded tiles
        self.tile_index = {}
        if os.path.exists(hgt_dir):
            self.build_tile_index()
            std_print(
                "INFO",
                "ea",
                "Initialized TerrainAnalyzer",
                f"HGT dir: {hgt_dir} ({len(self.tile_index)} tiles indexed)"
            )
        else:
            std_print("WARN", "ea", "HGT directory not found", hgt_dir)

    # HGT I/O helpers
    def build_tile_index(self):
        pattern = re.compile(r'[NS]\d{2}[EW]\d{3}\.hgt', re.IGNORECASE)
        for root, dirs, files in os.walk(self.hgt_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                # Direct match: file is exactly "N24E087.hgt"
                if pattern.fullmatch(fname):
                    self.tile_index[fname.upper()] = full_path
                else:
                    # Embedded match: file is something like "H44N24E087.hgt"
                    m = pattern.search(fname)
                    if m:
                        self.tile_index[m.group().upper()] = full_path

    def get_tile_filename(self, lat, lon):
        """Returns full path to HGT tile, or None if not available"""
        lat_int = int(np.floor(lat))
        lon_int = int(np.floor(lon))
        lat_str = f"{'N' if lat_int >= 0 else 'S'}{abs(lat_int):02d}"
        lon_str = f"{'E' if lon_int >= 0 else 'W'}{abs(lon_int):03d}"
        pattern = f"{lat_str}{lon_str}.HGT"   # index keys are upper-cased
        return self.tile_index.get(pattern)   # None if tile not available

    def load_hgt_tile(self, tile_path):
        # Check cache first
        if tile_path in self.tiles_cache:
            return self.tiles_cache[tile_path]
        if not os.path.exists(tile_path):
            std_print("ERROR", "ea", "HGT tile not found", tile_path)
            return None

        try:
            with rasterio.open(tile_path) as src:
                # Read elevation data (SRTM uses 1 band)
                elevation = src.read(1).astype(np.float32)
                # Handle nodata values (SRTM uses -32768)
                nodata = src.nodata
                if nodata is not None:
                    elevation[elevation == nodata] = np.nan
                tile_data = {
                    "data": elevation,
                    "transform": src.transform,
                    # 'crs': src.crs,
                    # 'bounds': src.bounds,
                    "shape": elevation.shape,
                }
                # Cache for future use
                self.tiles_cache[tile_path] = tile_data
                elev_min = np.nanmin(elevation)
                elev_max = np.nanmax(elevation)
                valid_pct = 100 * (~np.isnan(elevation)).sum() / elevation.size
                std_print("DEBUG", "ea", f"Loaded {os.path.basename(tile_path)}")
                std_print("DEBUG", "ea", f"Elev: [{elev_min:.0f}, {elev_max:.0f}] m")
                std_print("DEBUG", "ea", f"Valid: {valid_pct:.1f}%")
                return tile_data
        except Exception as e:
            std_print("ERROR", "ea", f"Failed to load {tile_path}", str(e))
            return None

    def get_elevation_at_point(self, lat, lon):
        """Return elevation (m) at a single lat/lon point, or None if tile unavailable."""
        tile_name = self.get_tile_filename(lat, lon)
        if tile_name is None: # tile not available
            return None
        tile_data = self.load_hgt_tile(tile_name)
        if tile_data is None: # tile found but failed to read
            return None
        try:
            row, col = rasterio.transform.rowcol(tile_data["transform"], lon, lat)
            row = int(round(row))
            col = int(round(col))
            h, w = tile_data["shape"]
            if 0 <= row < h and 0 <= col < w:
                v = tile_data["data"][row, col]
                return float(v) if not np.isnan(v) else None
        except Exception:
            pass
        return None

    # def get_elevation_profile(self, track_df):
    #     """Extract elevation profile along MD track, batched per tile."""
    #     track_df = track_df.copy().reset_index(drop=True)
    #     lats = track_df["latitude"].values
    #     lons = track_df["longitude"].values
    #     elevations = np.full(len(track_df), np.nan)
    #
    #     for track_id, row in track_df.iterrows():
    #         lat = row["latitude"]
    #         lon = row["longitude"]
    #         elev = self.get_elevation_at_point(lat, lon)
    #         elevations.append(elev if elev is not None else np.nan)
    #     track_df["elevation"] = elevations
    #
    #     # Log statistics
    #     valid_count = track_df["elevation"].notna().sum()
    #     # valid_pct = 100 * valid_count / len(track_df)
    #     if valid_count > 0:
    #         elev_stats = track_df["elevation"].describe()
    #         # std_print("DEBUG", "ea", "Elevation profile extracted", f"{valid_pct:.1f}% valid, range: [{elev_stats['min']:.0f}, {elev_stats['max']:.0f}] m")
    #     else:
    #         std_print("ERROR", "ea", "Elevation profile", "No valid elevation data for track")
    #     return track_df
    def get_elevation_profile(self, track_df):
        """Extract elevation profile along MD track, batched per tile."""
        track_df = track_df.copy().reset_index(drop=True)
        lats = track_df["latitude"].values
        lons = track_df["longitude"].values
        elevations = np.full(len(track_df), np.nan)

        # Classify each point as land or ocean once
        if "is_land" in track_df.columns:
            land_mask = track_df["is_land"].values.astype(bool)
        else:
            land_mask = is_land(lats, lons)

        # Group land points row indices by (lat_int, lon_int) tile key
        tile_groups = {}
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            if not land_mask[i]:
                continue # ocean point, skip
            key = (int(np.floor(lat)), int(np.floor(lon)))
            tile_groups.setdefault(key, []).append(i)

        missing_land_tiles = []
        for (lat_int, lon_int), indices in tile_groups.items():
            tile_path = self.get_tile_filename(lat_int + 0.5, lon_int + 0.5)
            if tile_path is None:
                lat_str = f"{'N' if lat_int >= 0 else 'S'}{abs(lat_int):02d}"
                lon_str = f"{'E' if lon_int >= 0 else 'W'}{abs(lon_int):03d}"
                missing_land_tiles.append(f"{lat_str}{lon_str}")
                continue
            tile_data = self.load_hgt_tile(tile_path)
            if tile_data is None:
                continue
            for i in indices:
                try:
                    row, col = rasterio.transform.rowcol(tile_data["transform"], lons[i], lats[i])
                    row, col = int(round(row)), int(round(col))
                    h, w = tile_data["shape"]
                    if 0 <= row < h and 0 <= col < w:
                        v = tile_data["data"][row, col]
                        if not np.isnan(v):
                            elevations[i] = float(v)
                except Exception:
                    pass

        if missing_land_tiles:
            std_print("WARN", "ea",f"Missing land HGT tiles ({len(missing_land_tiles)})",", ".join(sorted(missing_land_tiles)))

        track_df["elevation"] = elevations
        land_count   = int(np.sum(land_mask))
        valid_count  = int(np.sum(~np.isnan(elevations)))
        ocean_count  = len(track_df) - land_count

        std_print("DEBUG", "ea", "Elevation profile",
                  f"{valid_count}/{land_count} land points valid | "
                  f"{ocean_count} ocean points (NaN, expected)")

        if valid_count == 0 and land_count > 0:
            std_print("WARN", "ea","No elevation data for land points - missing tiles")
        elif valid_count == 0 and land_count == 0:
            std_print("DEBUG", "ea","Track is entirely over ocean - no elevation data expected")
        return track_df

    def calculate_slope(self, tile_name, cell_size=30.0):
        """Slope magnitude (degrees) via Sobel operator."""
        tile_data = self.load_hgt_tile(tile_name)
        if tile_data is None:
            return None
        elevation = tile_data["data"].copy()

        # Interpolate NaN values for gradient calculation
        nan_mask = np.isnan(elevation)
        if np.all(nan_mask):
            return None

        # Simple interpolation: replace NaN with local mean
        if nan_mask.any():
            valid = elevation[~nan_mask]
            elevation[nan_mask] = np.nanmean(valid)

        # Sobel operator for gradients
        dz_dx = sobel(elevation, axis=1) / (8 * cell_size)  # East-West
        dz_dy = sobel(elevation, axis=0) / (8 * cell_size)  # North-South

        # Slope magnitude in degrees
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope = np.degrees(slope_rad)

        # Cap unrealistic values
        # slope = np.clip(slope, 0, 90)
        return slope

    def get_terrain_metrics_at_point(self, lat, lon, window_size=3):
        """Get comprehensive terrain metrics at a point with local neighborhood"""
        tile_name = self.get_tile_filename(lat, lon)
        if tile_name is None:
            return None
        tile_data = self.load_hgt_tile(tile_name)
        if tile_data is None:
            return None
        # Get pixel coordinates
        transform = tile_data["transform"]
        col, row = ~transform * (lon, lat)
        row, col = int(round(row)), int(round(col))
        # Check bounds
        shape = tile_data["shape"]
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            return None
        elevation = tile_data["data"]
        point_elev = elevation[row, col]
        if np.isnan(point_elev):
            return None
        # Local neighborhood
        half_win = window_size // 2
        r_min = max(0, row - half_win)
        r_max = min(shape[0], row + half_win + 1)
        c_min = max(0, col - half_win)
        c_max = min(shape[1], col + half_win + 1)
        neighborhood = elevation[r_min:r_max, c_min:c_max]
        valid_neighborhood = neighborhood[~np.isnan(neighborhood)]

        if len(valid_neighborhood) == 0:
            # No valid neighbors, use point elevation only
            return {
                "elevation": float(point_elev),
                "slope": 0.0,
                "aspect": -1.0,
                "roughness": 0.0,
                "elevation_std": 0.0,
                "n_valid_neighbors": 0,
            }
        # Metrics
        metrics = {
            "elevation": float(point_elev),
            "slope": float(np.nanmean(np.abs(np.diff(valid_neighborhood)))),
            "aspect": -1.0,  # Aspect needs bivariate gradient
            "roughness": float(np.nanstd(valid_neighborhood)),
            "elevation_std": float(np.nanstd(valid_neighborhood)),
            "n_valid_neighbors": len(valid_neighborhood),
        }
        return metrics

    # TSI computation
    def compute_tsi(self, track_df, track_coupling=None):
        """Compute Topographic Sensitivity Index (TSI) for a track from land-phase points only.
        TSI = elevation_effect x slope_effect x roughness_effect
        """
        # Use land points if the column is available
        if "is_land" in track_df.columns:
            land_df = track_df[track_df["is_land"] == True]
        else:
            land_df = track_df  # NaN filtering below

        # Extract valid elevations
        valid_elev = track_df["elevation"].dropna()
        tsi_result = {
            "TSI": np.nan,
            "TSI_class": "insufficient_data",
            "elevation_effect": np.nan,
            "slope_effect": np.nan,
            "roughness_effect": np.nan,
            "mean_elevation": np.nan,
            "elevation_range": np.nan,
            "mean_slope": np.nan,
            "elevation_std": float(valid_elev.std()),  
            "n_valid_points": len(valid_elev),
            "n_land_points": len(land_df), 
        }
        if len(valid_elev) < 3:
            return tsi_result

        # 1. Elevation effect
        elev_range = valid_elev.max() - valid_elev.min()
        elev_eff = min(1.0, elev_range / 3000.0)

        # 2. Slope effect
        mean_slope = np.abs(valid_elev.diff().dropna()).mean()
        slope_eff = min(1.0, mean_slope / 30.0)

        # 3. Roughness effect
        roughness_eff = min(1.0, valid_elev.std() / 500.0)

        # Compute TSI
        tsi = elev_eff * slope_eff * roughness_eff
        # Classify TSI
        if tsi >= 0.05:
            tsi_class = "High (terrain-sensitive)"
        elif tsi >= 0.02:
            tsi_class = "Moderate"
        elif tsi >= 0.005:
            tsi_class = "Low (terrain-insensitive)"
        else:
            tsi_class = "Very low (flat)"

        result = {
            "TSI": float(tsi),
            "TSI_class": tsi_class,
            "elevation_effect": float(elev_eff),
            "slope_effect": float(slope_eff),
            "roughness_effect": float(roughness_eff),
            "mean_elevation": float(valid_elev.mean()),
            "elevation_range": float(elev_range),
            "mean_slope": float(mean_slope),
            "n_valid_points": len(valid_elev),
            "n_land_points": len(land_df),
        }
        # Coupling metrics if available
        if track_coupling is not None:
            if "FSI" in track_coupling:
                result["FSI"] = track_coupling["FSI"]
            if "duration_h" in track_coupling:
                result["duration_h"] = track_coupling["duration_h"]
        return result

    def compute_tsi_for_all_tracks(self, tracks_df, coupling_df):
        """Compute TSI for all tracks"""
        if isinstance(tracks_df, pd.DataFrame):
            if "track_id" not in tracks_df.columns:
                std_print("ERROR", "ea", "tracks_df has no track_id column")
                return pd.DataFrame(), {}
            tracks_df = {
                int(tid): tracks_df[tracks_df["track_id"] == tid].copy()
                for tid in tracks_df["track_id"].unique()
            }

        all_results = []
        elevation_enriched_tracks_df = {}
        for track_id, track_df in tracks_df.items():
            # Ensure track_df
            if isinstance(track_df, pd.Series):
                continue
            try:
                # Get elevation profile
                track_with_elev = self.get_elevation_profile(track_df)

                # Coupling metrics if available
                track_coupling = None
                if coupling_df is not None and not coupling_df.empty:
                    if "track_id" in coupling_df.columns:
                        mask = coupling_df["track_id"] == track_id
                        if mask.any():
                            track_coupling = coupling_df[mask].iloc[0].to_dict()
                    elif track_id in coupling_df.index:
                        track_coupling = coupling_df.loc[track_id].to_dict()

                # Compute TSI
                tsi_result = self.compute_tsi(track_with_elev, track_coupling)
                tsi_result["track_id"] = track_id

                # Calculate duration if available
                if "time" in track_df.columns:
                    times = pd.to_datetime(track_df["time"])
                    duration_h = (times.max() - times.min()).total_seconds() / 3600
                    tsi_result["duration_h"] = duration_h
                all_results.append(tsi_result)
                std_print(
                    "INFO",
                    "ea",
                    "TSI",
                    f"{tsi_result['TSI']:.3f} ({tsi_result['TSI_class']})",
                )
            except Exception as e:
                std_print("ERROR", "ea", "Computing TSI Failed", str(e))
                traceback.print_exc()
                continue

            try:
                landfall_idx, _, is_valid = get_landfall_index(track_with_elev)
                if is_valid and landfall_idx >= 0:
                    lf  = track_with_elev.iloc[landfall_idx]
                    end = track_with_elev.iloc[-1]
                    tsi_result["lat_landfall"]          = float(lf["latitude"])
                    tsi_result["lon_landfall"]          = float(lf["longitude"])
                    tsi_result["elevation_at_landfall"] = (
                        float(lf["elevation"]) if pd.notna(lf.get("elevation", np.nan)) else np.nan
                    )
                    tsi_result["penetration_km"] = float(haversine_distance(
                        lf["longitude"],
                        lf["latitude"],
                        end["longitude"],
                        end["latitude"]
                    ))
                else:
                    for col in ("lat_landfall", "lon_landfall","elevation_at_landfall", "penetration_km"):
                        tsi_result[col] = np.nan
            except Exception as lf_err:
                std_print("ERROR", "ea", "landfall columns failed", str(lf_err))
                for col in ("lat_landfall", "lon_landfall","elevation_at_landfall", "penetration_km"):
                    tsi_result[col] = np.nan
            elevation_enriched_tracks_df[track_id] = track_with_elev  
        return (pd.DataFrame(all_results) if all_results else pd.DataFrame()), elevation_enriched_tracks_df

    # Export /
    def export_elevation_metrics(self, tsi_df, output_dir):
        if tsi_df.empty:
            std_print("WARN", "ea", "No TSI results to export")
            return
        os.makedirs(output_dir, exist_ok=True)
        # Save CSV
        # FIX: use export_to_csv()
        csv_path = os.path.join(output_dir, "elevation_metrics.csv")
        tsi_df.to_csv(csv_path, index=False)
        std_print("SAVED", "ea", "Elevation metrics", csv_path)
        # Save summary report
        report_path = os.path.join(output_dir, "elevation_summary.txt")
        with open(report_path, "w") as f:
            f.write("Elevation & Terrain analysis summary\n\n")
            f.write(f"Total tracks analyzed: {len(tsi_df)}\n")
            f.write(f"Valid elevation data: {(tsi_df['n_valid_points'] > 0).sum()}\n\n")

            # valid_tsi = tsi_df[tsi_df["TSI"].notna()]["TSI"]
            valid_tsi = tsi_df["TSI"].dropna()
            if len(valid_tsi):
                f.write("Topographic Sensitivity Index (TSI) statistics\n\n")
                f.write(f"  Mean TSI: {valid_tsi.mean():.3f}\n")
                f.write(f"  Std Dev:  {valid_tsi.std():.3f}\n")
                # f.write(f"  Min TSI:  {valid_tsi.min():.3f}\n")
                # f.write(f"  Max TSI:  {valid_tsi.max():.3f}\n\n")
                f.write(f"  Range: [{valid_tsi.min():.3f}, {valid_tsi.max():.3f}]\n\n")

                f.write("TSI Classification Distribution:\n")
                class_dist = tsi_df["TSI_class"].value_counts()
                for cls, count in class_dist.items():
                    pct = 100 * count / len(tsi_df)
                    f.write(f"  {cls}: {count} tracks ({pct:.1f}%)\n")
                f.write("\n")

            # Component effects
            f.write("Terrain component effects\n\n")
            if "elevation_effect" in tsi_df.columns:
                f.write(
                    f"  Elevation effect (mean): {tsi_df['elevation_effect'].mean():.3f}\n"
                )
            if "slope_effect" in tsi_df.columns:
                f.write(
                    f"  Slope effect (mean):     {tsi_df['slope_effect'].mean():.3f}\n"
                )
            if "roughness_effect" in tsi_df.columns:
                f.write(
                    f"  Roughness effect (mean): {tsi_df['roughness_effect'].mean():.3f}\n"
                )
            f.write("\n")

            # Elevation statistics
            f.write("Elevation statistics\n\n")
            if "mean_elevation" in tsi_df.columns:
                f.write(
                    f"  Mean track elevation: {tsi_df['mean_elevation'].mean():.1f} m\n"
                )
            if "max_elevation" in tsi_df.columns:
                f.write(
                    f"  Max track elevation:  {tsi_df['max_elevation'].max():.1f} m\n"
                )
            if "elevation_range" in tsi_df.columns:
                f.write(
                    f"  Mean elevation range: {tsi_df['elevation_range'].mean():.1f} m\n"
                )
            if "elevation_std" in tsi_df.columns:
                f.write(
                    f"  Mean elevation std:   {tsi_df['elevation_std'].mean():.1f} m\n"
                )
            f.write("\n")

            # TSI vs duration correlation
            if "duration_h" in tsi_df.columns:
                sub = tsi_df.dropna(subset=["TSI", "duration_h"])
                if len(sub) >= 5:
                    r, p_raw, p_corr, n_eff = pearsonr_corrected(
                        sub["TSI"].values, sub["duration_h"].values
                    )
                    sig = (
                        "significant"
                        if (not np.isnan(p_corr) and p_corr < 0.05)
                        else "not significant"
                    )
                    f.write("\n")

                    f.write("TSI vs Duration\n")
                    f.write(
                        f"  r = {r:.3f}, p_corr = {p_corr:.4e} [{sig}], n_eff = {n_eff}"
                    )
        std_print("SAVED", "ea", "Elevation summary report", report_path)

    def plot_elevation_metrics(self, tsi_df, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        bbox_style = common_bbox()

        # TODO: do i really need histogram?
        # 1. TSI histogram
        ax = axes[0, 0]
        valid_tsi = tsi_df[tsi_df["TSI"].notna()]["TSI"]
        if len(valid_tsi) > 0:
            ax.hist(valid_tsi, bins=15, edgecolor="black", alpha=0.7, color="steelblue")
            ax.axvline(
                valid_tsi.mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {valid_tsi.mean():.3f}",
            )
            ax.set_xlabel(
                "Topographic Sensitivity Index (TSI)", fontsize=11, fontweight="bold"
            )
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title("TSI Distribution", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            # Legend
            legend_handles, labels = ax.get_legend_handles_labels()
            if legend_handles:
                ax.legend(handles=legend_handles, loc="best", **legend_style())

        # 2. TSI vs Duration
        ax = axes[0, 1]
        if "duration_h" in tsi_df.columns:
            valid_subset = tsi_df.dropna(subset=["TSI", "duration_h"])
            if len(valid_subset) > 0:
                c_vals = valid_subset["elevation_range"] if "elevation_range" in valid_subset.columns else "steelblue"
                sc1 = ax.scatter(
                    valid_subset["TSI"],
                    valid_subset["duration_h"],
                    c=c_vals,
                    cmap="viridis",
                    s=90,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.8,
                )
                ax.set_xlabel("TSI", fontsize=11, fontweight="bold")
                ax.set_ylabel("Duration (hours)", fontsize=11, fontweight="bold")
                ax.set_title("TSI vs Track Duration", fontsize=13, fontweight="bold")
                ax.grid(True, alpha=0.3)

                if "elevation_range" in valid_subset.columns:
                    cbar = plt.colorbar(sc1, ax=ax)
                    cbar.set_label("Elevation Range (m)", fontsize=10)

                # Correlation
                # corr = valid_subset["TSI"].corr(valid_subset["duration_h"])
                r, _, p_corr, n_eff = pearsonr_corrected(valid_subset["TSI"].values, valid_subset["duration_h"].values)
                sig = "**" if (not np.isnan(p_corr) and p_corr < 0.05) else "ns"
                ax.text(
                    0.05,
                    0.95,
                    f"r = {r:.3f}\np_corr = {p_corr:.4e} {sig}\nn_eff = {n_eff}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    bbox=bbox_style,
                    family="monospace"
                )

        # 3. Component effects
        ax = axes[1, 0]
        components = []
        means = []
        for col, lbl in [
            ("elevation_effect", "Elevation"),
            ("slope_effect", "Slope"),
            ("roughness_effect", "Roughness")
        ]:
            if col in tsi_df.columns:
                components.append(lbl)
                means.append(tsi_df[col].mean())

        if components:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            bars = ax.bar(
                components,
                means,
                color=colors[: len(components)],
                edgecolor="black",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Mean Effect Value", fontsize=11, fontweight="bold")
            ax.set_title("TSI Component Effects", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

            # Value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    bbox=bbox_style,
                )

        # 4. Elevation range vs TSI
        ax = axes[1, 1]
        if "elevation_range" in tsi_df.columns:
            valid_subset = tsi_df.dropna(subset=["TSI", "elevation_range"])
            if len(valid_subset) > 0:
                c_vals = valid_subset["duration_h"] if "duration_h" in valid_subset.columns else "steelblue"
                sc2 = ax.scatter(
                    valid_subset["elevation_range"],
                    valid_subset["TSI"],
                    c=c_vals,
                    cmap="plasma",
                    s=90,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.8,
                )
                if "duration_h" in valid_subset.columns:
                    plt.colorbar(sc2, ax=ax).set_label("Duration (h)", fontsize=9)

                r, _, p_corr, n_eff = pearsonr_corrected(valid_subset["elevation_range"].values, valid_subset["TSI"].values)
                ax.text(
                    0.05,
                    0.95,
                    f"r = {r:.3f}\np_corr = {p_corr:.4e}",
                    transform=ax.transAxes,
                    fontsize=10,
                    va="top",
                    bbox=bbox_style,
                    family="monospace"
                )
                ax.set_xlabel("Elevation Range (m)", fontsize=11, fontweight="bold")
                ax.set_ylabel("TSI", fontsize=11, fontweight="bold")
                ax.set_title("Elevation Range vs TSI", fontsize=13, fontweight="bold")
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "elevation_metrics_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        # std_print("SAVED", "ea", "Elevation metric plots", plot_path)

# Orographic speed reduction
def compute_orographic_speed_reduction(tracks_df_dict, output_dir):
    """
    Check whether Track translation speed drops as elevation rises.
 
    For each track that has both 'elevation' and 'speed_ms' columns, computes:
      - Pearson r between elevation gradient and speed change
      - Mean speed in low-elevation (<400 m) vs high-elevation (>500 m) segments
    """
    os.makedirs(output_dir, exist_ok=True)
    all_elev = []
    all_speed = []
    all_dElev = []
    all_dSpeed = []
 
    for track_id, track_df in tracks_df_dict.items():
        if "elevation" not in track_df.columns or "speed_ms" not in track_df.columns:
            continue
        df = track_df.dropna(subset=["elevation", "speed_ms"]).copy()
        if len(df) < 5:
            continue
        # Elevation gradient: change in elev per track step (m per ~step)
        d_elev = df["elevation"].diff().fillna(0).values
        d_speed = df["speed_ms"].diff().fillna(0).values
        all_elev.extend(df["elevation"].values.tolist())
        all_speed.extend(df["speed_ms"].values.tolist())
        all_dElev.extend(d_elev.tolist())
        all_dSpeed.extend(d_speed.tolist())
 
    if len(all_elev) < 10:
        std_print("WARN", "ea", "Insufficient data for orographic speed reduction")
        return
 
    all_elev = np.array(all_elev)
    all_speed = np.array(all_speed)
    all_dElev = np.array(all_dElev)
    all_dSpeed = np.array(all_dSpeed)
 
    # Plot 1: elevation vs speed scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(all_elev, all_speed, s=20, alpha=0.4, c="steelblue", edgecolors="none")
    r, p_raw, p_corr, n_eff = pearsonr_corrected(all_elev, all_speed)
    z = np.polyfit(all_elev, all_speed, 1)
    x_line = np.linspace(all_elev.min(), all_elev.max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), "r--", lw=1.5)
    sig = "**" if (not np.isnan(p_corr) and p_corr < 0.05) else "ns"
    ax.text(
        0.05,
        0.95,
        f"r = {r:.3f}\np_corr = {p_corr:.4e} {sig}\nn_eff = {n_eff}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=common_bbox(),
        family="monospace"
    )
    ax.set_xlabel("Elevation (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Translation Speed (m/s)", fontsize=11, fontweight="bold")
    ax.set_title("Orographic Speed Reduction\n(elevation vs translation speed)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
 
    # Plot 2: elevation change vs speed change
    ax = axes[1]
    # Filter to upslope points only (dElev > 0 = storm is ascending terrain)
    mask = all_dElev > 0
    if mask.sum() >= 5:
        ax.scatter(
            all_dElev[mask],
            all_dSpeed[mask],
            s=20,
            alpha=0.4,
            c="darkorange",
            edgecolors="none",
            label="Ascending terrain"
        )
        r2, _, p2, n2 = pearsonr_corrected(all_dElev[mask], all_dSpeed[mask])
        z2 = np.polyfit(all_dElev[mask], all_dSpeed[mask], 1)
        x2 = np.linspace(all_dElev[mask].min(), all_dElev[mask].max(), 50)
        ax.plot(x2, np.poly1d(z2)(x2), "r--", lw=1.5)
        sig2 = "**" if (not np.isnan(p2) and p2 < 0.05) else "ns"
        ax.text(
            0.05,
            0.95,
            f"r = {r2:.3f}\np_corr = {p2:.4e} {sig2}\nn_eff = {n2}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=common_bbox(),
            family="monospace"
        )
    ax.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Elevation change per step (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Speed change per step (km/h)", fontsize=11, fontweight="bold")
    ax.set_title("Terrain Ascent vs Speed Change\n(positive dElev = moving uphill)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
 
    plt.tight_layout()
    outpath = os.path.join(output_dir, "orographic_speed_reduction.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Orographic speed reduction plot", outpath)


# Orographic precipitation enhancement ratio
def compute_orographic_precip_ratio(tracks_df_dict, ds, output_dir):
    """
    Orographic enhancement ratio:
      OER = mean_TP_upslope / mean_TP_flat
 
    Upslope = track points where elevation gradient > threshold (ascending).
    Flat     = track points where |elevation gradient| < threshold.
    """
    os.makedirs(output_dir, exist_ok=True)
    GRAD_THRESHOLD = 10   # m per track step
 
    oer_results = []
    for track_id, df in tracks_df_dict.items():
        if "elevation" not in df.columns or "tp_mmhr" not in df.columns:
            continue
        df = df.dropna(subset=["elevation", "tp_mmhr"]).copy()
        if len(df) < 5:
            continue
        d_elev  = df["elevation"].diff().fillna(0).values
        tp_vals = df["tp_mmhr"].values
 
        upslope_tp = tp_vals[d_elev >  GRAD_THRESHOLD]
        flat_tp    = tp_vals[np.abs(d_elev) <= GRAD_THRESHOLD]
 
        if len(upslope_tp) < 2 or len(flat_tp) < 2 or np.mean(flat_tp) < 1e-6:
            continue
        oer = float(np.mean(upslope_tp) / np.mean(flat_tp))
        oer_results.append({
            "track_id":         track_id,
            "oer":              oer,
            "mean_tp_upslope":  float(np.mean(upslope_tp)),
            "mean_tp_flat":     float(np.mean(flat_tp)),
            "n_upslope":        len(upslope_tp),
            "n_flat":           len(flat_tp),
        })
        std_print("INFO", "ea", f"Track {track_id} OER", f"{oer:.3f}")
 
    if not oer_results:
        std_print("WARN", "ea", "No OER results - add tp_mmhr column to track DataFrames")
        return pd.DataFrame()
 
    oer_df = pd.DataFrame(oer_results)
    # Export
    csv_path = os.path.join(output_dir, "orographic_precip_ratio.csv")
    oer_df.to_csv(csv_path, index=False)
    std_print("EXPORT", "ea", "OER results", csv_path)
 
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = oer_df["oer"].dropna()
    valid_mean = valid.mean()
    ax.hist(valid, bins=15, edgecolor="black", alpha=0.75, color="teal")
    ax.axvline(1.0, color="black", linestyle="--", lw=1.5, label="OER = 1.0 (no enhancement)")
    ax.axvline(valid_mean, color="red",  linestyle="--", lw=1.5, label=f"Mean OER = {valid_mean:.2f}")
    ax.set_xlabel("Orographic Enhancement Ratio (OER)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Track count", fontsize=11, fontweight="bold")
    ax.set_title("Orographic Precipitation Enhancement\n(upslope TP / flat TP)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    legend_handles, _ = ax.get_legend_handles_labels()
    if legend_handles:
        ax.legend(handles=legend_handles, **legend_style())
    plt.tight_layout()
    outpath = os.path.join(output_dir, "orographic_precip_ratio.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "OER plot", outpath)
    return oer_df

# Elevation vs inland penetration scatter
def plot_elevation_vs_penetration(tsi_df, output_dir):
    """
      1. Mean elevation  vs penetration distance
      2. TSI             vs penetration distance
      3. Elevation range vs penetration distance
    """
    if "penetration_km" not in tsi_df.columns:
        std_print("WARN", "ea", "plot_elevation_vs_penetration: missing penetration_km column")
        return
 
    pairs = [
        ("mean_elevation",  "Mean elevation (m)",    "steelblue"),
        ("TSI",             "TSI",                   "darkorange"),
        ("elevation_range", "Elevation range (m)",   "purple"),
    ]
    pairs = [(col, lbl, c) for col, lbl, c in pairs if col in tsi_df.columns]
    n = len(pairs)
    if n == 0:
        return
 
    # Figure
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]
 
    for ax, (col, lbl, color) in zip(axes, pairs):
        sub = tsi_df.dropna(subset=[col, "penetration_km"])
        if len(sub) < 3:
            continue
        x, y = sub[col].values, sub["penetration_km"].values
        ax.scatter(x, y, s=80, alpha=0.75, c=color, edgecolors="black", lw=0.7, zorder=3)
        r, _, p_corr, n_eff = pearsonr_corrected(x, y)
        rho, _ = spearmanr(x, y)
        z = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(x_line, np.poly1d(z)(x_line), "r--", lw=1.5, alpha=0.7)
        sig = "**" if (not np.isnan(p_corr) and p_corr < 0.05) else "ns"
        p_str = f"{p_corr:.4e}" if not np.isnan(p_corr) else "N/A"
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}\nρ = {rho:.3f}\np_corr = {p_str} {sig}\nn_eff = {n_eff}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox=common_bbox(),
            family="monospace"
        )
        ax.set_xlabel(lbl, fontsize=11, fontweight="bold")
        ax.set_title(f"{lbl} vs\nPenetration", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
 
    axes[0].set_ylabel("Inland penetration distance (km)", fontsize=11, fontweight="bold")
    fig.suptitle("Terrain Metrics vs Inland Penetration Distance",fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "elevation_vs_penetration.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "Elevation vs penetration scatter", outpath)


# Landfall elevation classification
def classify_landfall_elevation(tsi_df, output_dir):
    # FIX: verify 50-400 criterion for flat or elevated region
    """
    Classify MDs by landfall-point elevation:
      - Low coast  : elevation at landfall < 50 m (flat)
      - Elevated   : elevation at landfall 50–400 m
      - Terrain-facing: elevation > 400 m  (Western / Eastern Ghats exposure)
 
    Compares penetration and duration between classes.
    """
    if "elevation_at_landfall" not in tsi_df.columns:
        std_print("WARN", "ea", "classify_landfall_elevation: missing elevation_at_landfall column")
        return None
 
    def lf_class(elev):
        if np.isnan(elev):
            return "Unknown"
        if elev < 50:
            return "Low coast (<50 m)"
        if elev < 400:
            return "Elevated (50-400 m)"
        return "Terrain-facing (>400 m)"
 
    tsi_df = tsi_df.copy()
    tsi_df["lf_class"] = tsi_df["elevation_at_landfall"].apply(lf_class)
    std_print("INFO", "ea", "Landfall class distribution", tsi_df["lf_class"].value_counts().to_dict())
 
    # Export
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "landfall_elevation_class.csv")
    tsi_df[["track_id", "lf_class", "elevation_at_landfall","penetration_km", "duration_h"]].to_csv(csv_path, index=False)
    std_print("EXPORT", "ea", "Landfall elevation classes", csv_path)
 
    # Box plot: penetration by class
    classes = [c for c in ["Low coast (<50 m)", "Elevated (50-400 m)", "Terrain-facing (>400 m)"]
               if c in tsi_df["lf_class"].values]
    if "penetration_km" in tsi_df.columns and len(classes) >= 2:
        # Figure
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [tsi_df[tsi_df["lf_class"] == c]["penetration_km"].dropna().values for c in classes]
        bp  = ax.boxplot(
            data,
            label=classes,
            patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2)
        )
        ax.set_ylabel("Inland penetration (km)", fontsize=11, fontweight="bold")
        ax.set_title("Inland Penetration by Landfall Elevation Class", fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        for i, d in enumerate(data):
            ax.text(
                i + 1,
                max(d) if len(d) else 0,
                f"n={len(d)}",
                ha="center",
                va="bottom",
                fontsize=9
            )
        plt.tight_layout()
        outpath = os.path.join(output_dir, "landfall_class_penetration.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        std_print("SAVED", "ea", "Landfall class penetration plot", outpath)
    return tsi_df


def plot_track_elevation_profile(track_df, track_id, output_dir):
    """
    Elevation along track (left) vs translation speed (right).
    Highlights where terrain rise coincides with speed drop.
    """
    track_id = int(track_id)
    required = {"time", "elevation", "speed_ms", "is_land"}
    missing = required - set(track_df.columns)
    if missing:
        std_print("WARN", "ea", "plot_track_elevation_profile missing cols", str(missing))
        return

    df = track_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(df["time"])
    elevation  = df["elevation"].values.astype(float)
    speed = df["speed_ms"].values.astype(float)
    is_land_arr = df["is_land"].values.astype(bool)

    # Landfall time (first ocean→land transition)
    landfall_time = None
    for i in range(1, len(df)):
        if not is_land_arr[i - 1] and is_land_arr[i]:
            landfall_time = times.iloc[i]
            break

    # Figure
    fig, ax1 = plt.subplots(figsize=(16, 7))

    ax1.fill_between(times, 0, 1,where=is_land_arr, transform=ax1.get_xaxis_transform(),color="wheat", alpha=0.45, label="Land phase", zorder=0)
    ax1.fill_between(times, 0, 1,where=~is_land_arr, transform=ax1.get_xaxis_transform(),color="lightblue", alpha=0.45, label="Ocean phase", zorder=0)

    # Elevation (left axis)
    elev_land = np.where(is_land_arr, elevation, np.nan)
    ax1.fill_between(times, 0, elev_land,color="#8B6914", alpha=0.55, label="Elevation (land)", zorder=2)
    ax1.plot(times, elev_land, color="#5C3D0A", linewidth=1.2, zorder=3)

    # Reference elevations
    for ref_m, ls, label in [(400, "--", "400 m"), (500, "-.", "500 m")]:
        ax1.axhline(ref_m, color="gray", linestyle=ls, linewidth=0.8, alpha=0.6, label=label)

    ax1.set_ylabel("Elevation (m)", fontsize=12, fontweight="bold", color="#5C3D0A")
    ax1.tick_params(axis="y", labelcolor="#5C3D0A")
    elev_max = np.nanmax(elev_land) if not np.all(np.isnan(elev_land)) else 600
    ax1.set_ylim(0, max(elev_max * 1.25, 600))

    # Speed (right axis)
    ax2 = ax1.twinx()
    ax2.plot(times, speed, "-o", color="royalblue", linewidth=2.2,markersize=4, alpha=0.85, label="Speed (m/s)", zorder=4)
    ax2.set_ylabel("Translation Speed (m/s)", fontsize=12, fontweight="bold", color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")

    # Landfall line
    if landfall_time is not None:
        ax1.axvline(landfall_time, color="red", linestyle="--", linewidth=1.8,label="Landfall", zorder=5)

    # Annotate peak elevation on land
    valid_land = np.where(is_land_arr & ~np.isnan(elevation))[0]
    if len(valid_land) > 0:
        peak_idx = valid_land[np.argmax(elevation[valid_land])]
        peak_elev = elevation[peak_idx]
        peak_speed = speed[peak_idx]
        ax1.annotate(
            f"Peak: {peak_elev:.0f} m\nSpeed: {peak_speed:.1f} m/s",
            xy=(times.iloc[peak_idx], peak_elev),
            xytext=(15, 10), textcoords="offset points",
            fontsize=9, color="#5C3D0A",
            arrowprops=dict(arrowstyle="->", color="#5C3D0A", lw=1.0),
            bbox=common_bbox(),
        )
        # Minimum speed on land
        min_speed_idx = valid_land[np.argmin(speed[valid_land])]
        ax2.annotate(
            f"Min: {speed[min_speed_idx]:.1f} m/s",
            xy=(times.iloc[min_speed_idx], speed[min_speed_idx]),
            xytext=(15, -20), textcoords="offset points",
            fontsize=9, color="royalblue",
            arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.0),
            bbox=common_bbox(),
        )

    # Pearson r: elevation vs speed (land phase only)
    land_elev  = elevation[is_land_arr & ~np.isnan(elevation) & ~np.isnan(speed)]
    land_speed = speed[is_land_arr & ~np.isnan(elevation) & ~np.isnan(speed)]
    if len(land_elev) >= 5:
        r, _, p_corr, n_eff = pearsonr_corrected(land_elev, land_speed)
        sig = "**" if (not np.isnan(p_corr) and p_corr < 0.05) else "ns"
        p_str = f"{p_corr:.3e}" if not np.isnan(p_corr) else "N/A"
        stats_text = (
            f"Land phase (n_eff={n_eff})\n"
            f"r(elev, speed) = {r:.3f}\n"
            f"p_corr = {p_str} {sig}"
        )
        ax1.text(0.02, 0.97, stats_text,transform=ax1.transAxes, fontsize=9, va="top",family="monospace", bbox=common_bbox())

    # Lag-correlation inset: r(elevation[t], speed[t+k]) for k in -12..+12 h
    # INFO: lag = +3h means the MD speed drops about 3 hours after encountering high terrain 
    lags_h  = list(range(-12, 13, 1))
    lag_rs  = []
    for lag in lags_h:
        if lag < 0:
            xe, xs = land_elev[:lag],  land_speed[-lag:]
        elif lag > 0:
            xe, xs = land_elev[lag:],  land_speed[:-lag]
        else:
            xe, xs = land_elev, land_speed
        if len(xe) < 5 or np.std(xe) < 1e-6 or np.std(xs) < 1e-6:
            lag_rs.append(np.nan)
        else:
            r_lag, _ = pearsonr_corrected(xe, xs)[:2]
            lag_rs.append(float(r_lag) if not np.isnan(r_lag) else np.nan)

    # Inset axes
    ax_inset = ax1.inset_axes([0.60, 0.65, 0.37, 0.30])
    ax_inset.bar(lags_h, lag_rs, color=["royalblue" if v < 0 else "tomato"for v in lag_rs], alpha=0.75, width=0.8)
    ax_inset.axhline(0, color="black", linewidth=0.6)
    ax_inset.axvline(0, color="gray",  linewidth=0.6, linestyle="--")
    best_lag = lags_h[np.nanargmin(lag_rs)]   # most negative r
    ax_inset.axvline(best_lag, color="navy", linewidth=1.2, linestyle=":")
    ax_inset.set_xlabel("Lag (hours)", fontsize=7)
    ax_inset.set_ylabel("r", fontsize=7)
    ax_inset.set_title(f"Speed lag r  (min at {best_lag:+d}h)", fontsize=7)
    ax_inset.tick_params(labelsize=6)
    ax_inset.set_ylim(-1, 1)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,loc="upper right", **legend_style())

    ax1.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax1.set_title(f"Track {track_id}: Elevation Profile vs Translation Speed",fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--", zorder=0)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"track_{track_id}_elevation_profile.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "elevation profile", outpath)


def plot_track_elevation_map(track_df, track_id, terrain_analyzer, output_dir):
    """
    DEM tiles , track line colored by translation speed.
    Tracks ran into high terrain vs passed through low-elevation corridors.
    """
    track_id = int(track_id)
    required = {"latitude", "longitude", "speed_ms", "is_land"}
    if not required.issubset(track_df.columns):
        std_print("WARN", "ea", "plot_track_elevation_map missing cols",
                  str(required - set(track_df.columns)))
        return

    df = track_df.sort_values("time").reset_index(drop=True)
    lats = df["latitude"].values
    lons = df["longitude"].values
    speeds = df["speed_ms"].values.astype(float)

    # Bounding box with buffer
    buf = 1.5
    lat_min  = lats.min() - buf
    lat_max  = lats.max() + buf
    lon_min  = lons.min() - buf
    lon_max  = lons.max() + buf

    # Figure
    fig, ax = plt.subplots(figsize=(13, 11),subplot_kw={"projection": ccrs.PlateCarree()})

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # DEM tiles
    tile_patches = []
    for lat_int in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        for lon_int in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            tile_path = terrain_analyzer.get_tile_filename(lat_int + 0.5, lon_int + 0.5)
            if tile_path is None:
                continue
            tile_data = terrain_analyzer.load_hgt_tile(tile_path)
            if tile_data is None:
                continue
            elev = tile_data["data"].copy()
            elev[elev < 0] = 0     # treat void/ocean pixels as 0
            # Downsample for display speed (step ≈ 30 arc-seconds)
            step = max(1, elev.shape[0] // 120)
            elev_ds = elev[::step, ::step]
            tile_patches.append((elev_ds, lon_int, lat_int))

    # Consistent elevation scale across all tiles
    if tile_patches:
        all_vals = [np.nanpercentile(p[0], 98) for p in tile_patches]
        vmax_elev = max(float(np.nanmax(all_vals)), 400.0)
    else:
        vmax_elev = 1000.0
        ax.add_feature(cfeature.LAND, facecolor="#E8D5B0", alpha=0.6)

    for (elev_ds, lon_int, lat_int) in tile_patches:
        img_extent = (lon_int, lon_int + 1, lat_int, lat_int + 1)
        ax.imshow(elev_ds, extent=img_extent, origin="upper",cmap="terrain", vmin=0, vmax=vmax_elev,transform=ccrs.PlateCarree(), alpha=0.85, zorder=1)

    # proxy for the elevation colorbar
    sm_elev = plt.cm.ScalarMappable(cmap="terrain",norm=plt.Normalize(0, vmax_elev))
    sm_elev.set_array([])
    cbar_elev = plt.colorbar(sm_elev, ax=ax, pad=0.01, shrink=0.55, aspect=20)
    cbar_elev.set_label("Elevation (m)", fontsize=10)

    # Elevation contours from track elevation column
    if "elevation" in df.columns:
        elev_vals = df["elevation"].values
        land_mask = df["is_land"].values.astype(bool)
        # Contour lines only make sense on a gridded field; here annotate key thresholds
        for thresh, color, ls in [(400, "saddlebrown", "--"), (500, "sienna", "-.")]:
            crossings = np.where(np.diff((elev_vals > thresh).astype(int)))[0]
            for ci in crossings:
                ax.plot(lons[ci], lats[ci], "^", color=color,markersize=7, transform=ccrs.PlateCarree(), zorder=5,label=f"{thresh} m crossing")

    # Track line colored by speed
    valid_speed = np.where(np.isnan(speeds), np.nanmedian(speeds), speeds)
    speed_min = max(0, np.nanpercentile(valid_speed, 5))
    speed_max = np.nanpercentile(valid_speed, 95)

    points   = np.array([lons, lats]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm_speed = plt.Normalize(speed_min, speed_max)
    lc = LineCollection(segments, cmap="RdYlGn",norm=norm_speed, linewidth=3.0,transform=ccrs.PlateCarree(), zorder=6)
    lc.set_array(valid_speed[:-1])
    ax.add_collection(lc)

    cbar_speed = plt.colorbar(lc, ax=ax, pad=0.08, shrink=0.55, aspect=20)
    cbar_speed.set_label("Translation Speed (m/s)", fontsize=10)

    # # Start / End / Landfall markers
    # ax.plot(lons[0], lats[0], "g*", markersize=10,transform=ccrs.PlateCarree(), zorder=8,label="Origin", markeredgecolor="black", markeredgewidth=0.5)
    # ax.plot(lons[-1], lats[-1], "rx", markersize=10, markeredgewidth=2,transform=ccrs.PlateCarree(), zorder=8, label="Dissipation")

    is_land_arr = df["is_land"].values.astype(bool)
    for i in range(1, len(df)):
        if not is_land_arr[i - 1] and is_land_arr[i]:
            ax.plot(lons[i], lats[i], "D", color="darkorange",markersize=8, transform=ccrs.PlateCarree(), zorder=8,label="Landfall", markeredgecolor="black", markeredgewidth=0.5)
            break

    ax.coastlines(resolution="50m", linewidth=1.0, color="black", zorder=7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=":", color="gray", zorder=7)
    ocean_feature = cfeature.NaturalEarthFeature("physical", "ocean", "50m", facecolor="lightcyan", edgecolor="none")
    ax.add_feature(ocean_feature, facecolor="lightcyan", alpha=0.4, zorder=2)
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4, linewidth=0.5)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="lower left", **legend_style())

    ax.set_title(f"Track {track_id}: Track over Terrain",fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"track_{track_id}_elevation_map.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "elevation map", outpath)


def plot_all_tracks_elevation_comparison(tracks_dict, terrain_analyzer, output_dir):
    """
    All tracks overlaid on DEM
    Line width proportional speed (thin = slow = terrain-braked, thick = fast = open corridor).
    """
    if not tracks_dict:
        return

    all_lats = np.concatenate([df["latitude"].values for df in tracks_dict.values()])
    all_lons = np.concatenate([df["longitude"].values for df in tracks_dict.values()])
    buf = 2.0
    lat_min, lat_max = all_lats.min() - buf, all_lats.max() + buf
    lon_min, lon_max = all_lons.min() - buf, all_lons.max() + buf

    fig, ax = plt.subplots(figsize=(16, 13),subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # DEM background
    tile_patches = []
    for lat_int in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        for lon_int in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            tp = terrain_analyzer.get_tile_filename(lat_int + 0.5, lon_int + 0.5)
            if tp is None:
                continue
            td = terrain_analyzer.load_hgt_tile(tp)
            if td is None:
                continue
            elev = td["data"].copy()
            elev[elev < 0] = 0
            step = max(1, elev.shape[0] // 120)
            tile_patches.append((elev[::step, ::step], lon_int, lat_int))

    vmax_elev = max(
        (float(np.nanpercentile(p[0], 98)) for p in tile_patches),
        default=1000.0
    )
    for elev_ds, lon_int, lat_int in tile_patches:
        ax.imshow(
            elev_ds,
            extent=(lon_int, lon_int+1, lat_int, lat_int+1),
            origin="upper",
            cmap="terrain",
            vmin=0,
            vmax=vmax_elev,
            transform=ccrs.PlateCarree(),
            alpha=0.75,
            zorder=1
        )

    sm_elev = plt.cm.ScalarMappable(cmap="terrain", norm=plt.Normalize(0, vmax_elev))
    sm_elev.set_array([])
    plt.colorbar(sm_elev, ax=ax, pad=0.01, shrink=0.5, aspect=18,label="Elevation (m)")

    # Per-track overlays
    track_colors = plt.cm.tab10(np.linspace(0, 1, len(tracks_dict)))
    summary_rows = []
    legend_handles = []

    for (track_id, track_df), color in zip(
            sorted(tracks_dict.items()), track_colors):

        df = track_df.sort_values("time").reset_index(drop=True)
        lats = df["latitude"].values
        lons = df["longitude"].values
        is_land_arr = df["is_land"].values.astype(bool) if "is_land" in df.columns else np.ones(len(df), bool)

        # Speed column
        speed_col = (
            "speed_ms"  if "speed_ms"  in df.columns else
            "speed_kmh" if "speed_kmh" in df.columns else
            None
        )
        speeds = df[speed_col].values.astype(float) if speed_col else np.full(len(df), 15.0)
        speed_unit = "m/s" if speed_col == "speed_ms" else ("km/h" if speed_col == "speed_kmh" else "km/h")

        # Track line — width mapped to speed (3–6 pt range)
        speed_norm = np.clip((speeds - np.nanpercentile(speeds, 5)) / (np.nanpercentile(speeds, 95) - np.nanpercentile(speeds, 5) + 1e-6), 0, 1)
        lw_arr = 2.0 + speed_norm * 4.0           # thin = slow, thick = fast

        points   = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lws      = lw_arr[:-1]
        # Draw in segments so width varies
        for seg, lw in zip(segments, lws):
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                color=color,
                linewidth=lw,
                transform=ccrs.PlateCarree(),
                solid_capstyle="round",
                zorder=6 + 0.01
            )

        # Origin / dissipation
        ax.plot(
            lons[0],
            lats[0],
            "*",
            color=color,
            markersize=13,
            markeredgecolor="black",
            markeredgewidth=0.6,
            transform=ccrs.PlateCarree(),
            zorder=9
        )
        ax.plot(
            lons[-1],
            lats[-1],
            "X",
            color=color,
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.6,
            transform=ccrs.PlateCarree(),
            zorder=9
        )

        # Landfall marker
        for i in range(1, len(df)):
            if not is_land_arr[i-1] and is_land_arr[i]:
                ax.plot(
                    lons[i],
                    lats[i],
                    "D",
                    color=color,
                    markersize=9,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    transform=ccrs.PlateCarree(),
                    zorder=9
                )
                break

        # Terrain barrier label: max elevation after landfall
        if "elevation" in df.columns:
            land_elev = df.loc[is_land_arr, "elevation"].dropna()
            max_elev  = float(land_elev.max()) if len(land_elev) else np.nan
            mean_elev = float(land_elev.mean()) if len(land_elev) else np.nan
        else:
            max_elev = mean_elev = np.nan

        land_speed = speeds[is_land_arr & ~np.isnan(speeds)]
        mean_speed_land = float(np.mean(land_speed)) if len(land_speed) else np.nan

        # Penetration km (landfall point → dissipation)
        pen_km = np.nan
        for i in range(1, len(df)):
            if not is_land_arr[i-1] and is_land_arr[i]:
                pen_km = haversine_distance(
                    lons[i], lats[i], lons[-1], lats[-1]
                )
                break

        summary_rows.append({
            "track_id":       track_id,
            "max_land_elev_m": round(max_elev, 0),
            "mean_land_elev_m": round(mean_elev, 0),
            "mean_land_speed": round(mean_speed_land, 1),
            "penetration_km":  round(pen_km, 0),
        })

        # Legend handle
        std_print("DEBUG", "ea", "Using speed unit", speed_unit)
        legend_handles.append(Line2D(
            [0], [0], color=color, linewidth=3,
            label=f"Track {track_id} max:{max_elev:.0f}m  "
                  f"penetration:{pen_km:.0f}km  "
                  f"speed:{mean_speed_land:.1f}{speed_unit}"
        ))

    # Threshold contour lines on DEM (400 m, 500 m, 700 m)
    for thresh_m, ls, lc in [(400,"--","saddlebrown"), (700,"-.", "sienna")]:
        ax.contour
        ax.plot([], [], linestyle=ls, color=lc, linewidth=1.5,label=f"{thresh_m} m threshold")

    ax.coastlines(resolution="50m", linewidth=1.0, color="black", zorder=7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=":", color="gray", zorder=7)
    ocean_feat = cfeature.NaturalEarthFeature("physical", "ocean", "50m", facecolor="lightcyan", edgecolor="none")
    ax.add_feature(ocean_feat, alpha=0.4, zorder=2)
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.35, linewidth=0.5)

    ax.legend(handles=legend_handles, loc="lower left",**legend_style({"fontsize": 9}))
    ax.set_title("All Tracks: Path over Terrain", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "all_tracks_elevation_comparison.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    std_print("SAVED", "ea", "All-tracks elevation comparison", outpath)

    # Export summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "track_terrain_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        std_print("EXPORT", "ea", "Track terrain summary", csv_path)
    return summary_rows


def run_single_track_elevation_analysis(track_df, track_id, output_dir, terrain_analyzer=None):
    """
    Per-track elevation analysis:
      1. Extract elevation profile
      2. Profile plot: elevation + speed vs time (orographic braking signal)
      3. Map plot: DEM background + track colored by speed (corridor vs barrier)

    Called inside the per-track loop in md_detection_tracking.py.
    Returns track_df enriched with 'elevation' column.
    """
    track_id = int(track_id)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract elevation
    if "speed_ms" not in track_df.columns:
        std_print("WARN", "ea","speed_ms missing - run compute_track_speeds first")

    try:
        track_df = terrain_analyzer.get_elevation_profile(track_df)
    except Exception as e:
        std_print("ERROR", "ea", "get_elevation_profile failed", str(e))
        return track_df

    valid = track_df["elevation"].notna().sum()
    if valid == 0:
        std_print("WARN", "ea","no elevation data - skipping per-track plots")
        return track_df

    # 2. Dual-axis profile: elevation + speed over time
    try:
        plot_track_elevation_profile(track_df, track_id, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "elevation profile plot failed", str(e))
        traceback.print_exc()

    # 3. Spatial map: DEM background + track colored by speed
    try:
        plot_track_elevation_map(track_df, track_id, terrain_analyzer, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "elevation map plot failed", str(e))
        traceback.print_exc()
    return track_df

def run_elevation_analysis(tracks_df, coupling_df, hgt_dir, output_dir, terrain_analyzer=None):
    """
    1. Load SRTM HGT tiles
    2. Extract elevation profiles for all tracks
    3. Compute TSI
    4. Orographic speed reduction (if speed_ms available)
    5. Elevation vs penetration scatter (if penetration_km available)
    6. Landfall elevation classification (if elevation_at_landfall available)
    7. Soil texture interpretation (if lat_landfall available)
    8. Export elevation_metrics.csv
    """
    # std_print("DEBUG", "ea", " --- Elevation analysis ---", f"{len(tracks_df)} tracks")
    # Initialize terrain analyzer
    terrain = terrain_analyzer if terrain_analyzer is not None else TerrainAnalyzer(hgt_dir=hgt_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Compute TSI for all tracks
    tsi_df, elevation_enriched_tracks_df = terrain.compute_tsi_for_all_tracks(tracks_df, coupling_df)
    if tsi_df.empty:
        std_print("ERROR", "ea", "No TSI results computed", "No elevation data")
        return pd.DataFrame()

    # Export results
    terrain.export_elevation_metrics(tsi_df, output_dir)
    terrain.plot_elevation_metrics(tsi_df, output_dir)

    # Orographic speed reduction
    try:
        compute_orographic_speed_reduction(elevation_enriched_tracks_df, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Orographic speed reduction failed", str(e))

    # Elevation vs penetration (only if column available)
    try:
        plot_elevation_vs_penetration(tsi_df, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Elevation vs penetration plot failed", str(e))

    # Landfall elevation classification
    try:
        classify_landfall_elevation(tsi_df, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Landfall elevation classification failed", str(e))

    # Soil texture
    try:
        plot_soil_texture_interpretation(tsi_df, output_dir)
    except Exception as e:
        std_print("ERROR", "ea", "Soil texture plot failed", str(e))

    std_print("INFO", "ea", "Elevation analysis complete", output_dir)
    return tsi_df


# Helper functions
def soil_memory_timescale(beta, k_sat_ms):
    """
    Approximate SM memory timescale (days).
    Higher beta and higher k_sat -> water drains faster -> shorter memory.
    """
    k_sat_day = k_sat_ms * 86400  # m/day
    tau = 1.0 / (k_sat_day * beta + 1e-10)
    return round(tau, 1)


def load_elevation_metrics(output_dir):
    """Load TSI metrics computed from HGT elevation data"""
    for path in [
        os.path.join(output_dir, "elevation_analysis", "elevation_metrics.csv"),
        "output/elevation_analysis/elevation_metrics.csv",
        "elevation_metrics.csv",
    ]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                std_print("INFO", "ea", "Loaded elevation metrics", path)
                return df
            except Exception as e:
                std_print("ERROR", "ea", f"Failed to load {path}", str(e))

    # paths_to_try = [
    #     os.path.join(output_dir, "elevation_analysis", "elevation_metrics.csv"),
    #     "output/elevation_analysis/elevation_metrics.csv",
    #     "elevation_metrics.csv",
    # ]
    # for path in paths_to_try:
    #     if os.path.exists(path):
    #         try:
    #             df = pd.read_csv(path)
    #             std_print("INFO", "ea", "Loaded elevation metrics", path)
    #             return df
    #         except Exception as e:
    #             std_print("ERROR", "ea", f"Failed to load {path}", str(e))
    std_print("WARN", "ea", "Elevation metrics not found", "TSI analysis will be skipped")
    return None

