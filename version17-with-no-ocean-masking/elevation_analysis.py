"""
Elevation Analysis for MD Tracks using SRTM HGT Data
TSI combines three terrain-dependent factors:
  TSI = elevation_effect × slope_effect × roughness_effect
  - elevation_effect: normalized elevation contrast
  - slope_effect: mean slope magnitude
  - roughness_effect: terrain variability (std of elevation)
"""

"""
deps: sudo apt install gdal-bin
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from typing import Dict, List, Tuple, Optional
import traceback
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from rasterio.vrt import WarpedVRT
import subprocess
import shutil

from utils import std_print

def load_elevation_metrics(output_dir: str) -> Optional[pd.DataFrame]:
    """Load TSI metrics computed from HGT elevation data."""
    paths_to_try = [
        os.path.join(output_dir, 'elevation_analysis', 'elevation_metrics.csv'),
        'output/elevation_analysis/elevation_metrics.csv',
        'elevation_metrics.csv'
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                std_print('INFO', 'stats', 'Loaded elevation metrics', path)
                return df
            except Exception as e:
                std_print('WARN', 'stats', f'Failed to load {path}', str(e)[:40])
    std_print('WARN', 'ea', 'Elevation metrics not found', 'TSI analysis will be skipped')
    return None

class TerrainAnalyzer:
    """
    Loads and analyzes SRTM HGT elevation tiles for track TSI computation.
    HGT file naming: N/SxxE/Wyyy.hgt
      N/S = hemisphere (North/South)
      xx = latitude (00-89)
      E/W = hemisphere (East/West)
      yyy = longitude (000-179)
    """
    def __init__(self, hgt_dir: str = '../Data/Elevation/HGT-files/'):
        self.hgt_dir = hgt_dir
        self.tiles_cache = {}  # Cache loaded tiles for performance
        if os.path.exists(hgt_dir):
            std_print('INFO', 'ea', f'Initialized TerrainAnalyzer', f'HGT dir: {hgt_dir}')
        else:
            std_print('WARN', 'ea', f'HGT directory not found', hgt_dir)

    def _find_hgt_file(self, tile_name):
        """Search recursively for the HGT file in the directory structure."""
        # tile_name is expected to be something like 'N28E078.hgt'
        for root, dirs, files in os.walk(self.hgt_dir):
            # Check if any file matches the pattern, ignoring prefixes like 'H44'
            for f in files:
                if tile_name in f:
                    return os.path.join(root, f)
        return None

    # def get_tile_filename(self, lat: float, lon: float) -> str:
    #     """Returns the SRTM naming convention string"""
    #     lat_int = int(np.floor(lat))
    #     lon_int = int(np.floor(lon))
    #     lat_str = f"{'N' if lat_int >= 0 else 'S'}{abs(lat_int):02d}"
    #     lon_str = f"{'E' if lon_int >= 0 else 'W'}{abs(lon_int):03d}"
    #     return f"{lat_str}{lon_str}"
    def get_tile_filename(self, lat, lon):
        """Returns the SRTM naming convention string"""
        lat_int = int(np.floor(lat))
        lon_int = int(np.floor(lon))
        lat_str = f"{'N' if lat_int >= 0 else 'S'}{abs(lat_int):02d}"
        lon_str = f"{'E' if lon_int >= 0 else 'W'}{abs(lon_int):03d}"
        expected_pattern = f"{lat_str}{lon_str}.hgt"
        # Check if the file exists anywhere in the HGT directory tree
        for root, dirs, files in os.walk(self.hgt_dir):
            for f in files:
                if expected_pattern in f: # Match "N28E078.hgt" or "H44N28E078.hgt"
                    return os.path.join(root, f)
        return expected_pattern

    def load_hgt_tile(self, tile_name):
        # Check cache first
        if tile_name in self.tiles_cache:
            return self.tiles_cache[tile_name]
        # Try standard path: hgt_dir/tile_name
        tile_path = os.path.join(self.hgt_dir, tile_name)
        path = self._find_hgt_file(f"{tile_name}.hgt")
        if not tile_path:
            std_print('ERROR', 'ea', f'Tile not found in {self.hgt_dir}', tile_name)
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
                    'data': elevation,
                    'transform': src.transform,
                    # 'crs': src.crs,
                    # 'bounds': src.bounds,
                    'shape': elevation.shape
                }
                # Cache for future use
                self.tiles_cache[tile_name] = tile_data
                elev_min = np.nanmin(elevation)
                elev_max = np.nanmax(elevation)
                valid_pct = 100 * (~np.isnan(elevation)).sum() / elevation.size
                std_print('DEBUG', 'ea', f'Loaded {tile_name}')
                std_print('DEBUG', 'ea', f'Elev: [{elev_min:.0f}) {elev_max:.0f}] m')
                std_print('DEBUG', 'ea', f'Valid: {valid_pct:.1f}%')
                return tile_data
        except Exception as e:
            std_print('ERROR', 'ea', f'Failed to load {tile_name}', str(e))
            return None

    def get_elevation_at_point(self, lat, lon):
        tile_name = self.get_tile_filename(lat, lon)
        tile_data = self.load_hgt_tile(tile_name)
        if tile_data is None:
            return None
        try:
            # Convert lat/lon to pixel coordinates
            row, col = rasterio.transform.rowcol(tile_data['transform'], lon, lat)
            # Round to nearest pixel
            row = int(round(row))
            col = int(round(col))
            # Check bounds
            shape = tile_data['shape']
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                elevation = tile_data['data'][row, col]
                return float(elevation) if not np.isnan(elevation) else None
        except:
            pass
        return None

    def get_elevation_profile(self, track_df):
        """Extract elevation profile along MD track"""
        track_df = track_df.copy()
        elevations = []
        for idx, row in track_df.iterrows():
            lat, lon = row['latitude'], row['longitude']
            elev = self.get_elevation_at_point(lat, lon)
            elevations.append(elev if elev is not None else np.nan)
        track_df['elevation'] = elevations
        # Log statistics
        valid_count = track_df['elevation'].notna().sum()
        valid_pct = 100 * valid_count / len(track_df)
        if valid_count > 0:
            elev_stats = track_df['elevation'].describe()
            std_print('DEBUG', 'ea', 'Elevation profile extracted', f'{valid_pct:.1f}% valid, range: [{elev_stats["min"]:.0f}, {elev_stats["max"]:.0f}] m')
        else:
            std_print('ERROR', 'ea', 'Elevation profile', f'No valid elevation data for track')
        return track_df
 
    def calculate_slope(self, tile_name, cell_size = 30.0):
        """Calculate slope from elevation tile using Sobel operator"""
        tile_data = self.load_hgt_tile(tile_name)
        if tile_data is None:
            return None
        elevation = tile_data['data'].copy()
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
        slope = np.clip(slope, 0, 90)
        return slope

    def get_terrain_metrics_at_point(self, lat: float, lon: float, window_size: int = 3) -> Optional[Dict]:
        """Get comprehensive terrain metrics at a point with local neighborhood"""
        tile_name = self.get_tile_filename(lat, lon)
        tile_data = self.load_hgt_tile(tile_name)
        if tile_data is None:
            return None
        # Get pixel coordinates
        transform = tile_data['transform']
        col, row = ~transform * (lon, lat)
        row, col = int(round(row)), int(round(col))
        # Check bounds
        shape = tile_data['shape']
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            return None
        elevation = tile_data['data']
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
                'elevation': float(point_elev),
                'slope': 0.0,
                'aspect': -1.0,
                'roughness': 0.0,
                'elevation_std': 0.0,
                'n_valid_neighbors': 0
            }
        # Metrics
        metrics = {
            'elevation': float(point_elev),
            'slope': float(np.nanmean(np.abs(np.diff(valid_neighborhood)))),
            'aspect': -1.0,  # Aspect needs bivariate gradient
            'roughness': float(np.nanstd(valid_neighborhood)),
            'elevation_std': float(np.nanstd(valid_neighborhood)),
            'n_valid_neighbors': len(valid_neighborhood)
        }
        return metrics

    def compute_tsi(self, track_df: pd.DataFrame, track_coupling: Optional[Dict] = None) -> Dict:
        """Compute Topographic Sensitivity Index (TSI) for a track.
        TSI = elevation_effect × slope_effect × roughness_effect
        Where each component is normalized to [0, 1]:
          - elevation_effect: (max_elev - min_elev) / 3000m
          - slope_effect: mean_slope / 30°
          - roughness_effect: std(elevation) / 500m
        """
        # Extract valid elevations
        valid_elev = track_df['elevation'].dropna()
        tsi_result =  {
            'TSI': np.nan,
            'TSI_class': 'insufficient_data',
            'elevation_effect': np.nan,
            'slope_effect': np.nan,
            'roughness_effect': np.nan,
            'mean_elevation': np.nan,
            'elevation_range': np.nan,
            'mean_slope': np.nan,
            'elevation_std': np.nan,
            'n_valid_points': len(valid_elev),
            'notes': f'Only {len(valid_elev)} valid elevation points'
        }
        if len(valid_elev) < 3:
            return tsi_result

        # 1. ELEVATION EFFECT
        elev_range = valid_elev.max() - valid_elev.min()
        elev_eff = min(1.0, elev_range / 3000.0)

        # 2. SLOPE EFFECT
        mean_slope = np.abs(valid_elev.diff().dropna()).mean()
        slope_eff = min(1.0, mean_slope / 30.0)

        # 3. ROUGHNESS EFFECT
        roughness_eff = min(1.0, valid_elev.std() / 500.0)

        # Compute TSI
        tsi = elev_eff * slope_eff * roughness_eff
        # Classify TSI
        if tsi >= 0.5: tsi_class = 'High (terrain-sensitive)'
        elif tsi >= 0.3: tsi_class = 'Moderate'
        elif tsi >= 0.1: tsi_class = 'Low (terrain-insensitive)'
        else: tsi_class = 'Very low (flat)'
        result = {
            'TSI': float(tsi),
            'TSI_class': tsi_class,
            'elevation_effect': float(elev_eff),
            'slope_effect': float(slope_eff),
            'roughness_effect': float(roughness_eff),
            'mean_elevation': float(valid_elev.mean()),
            'elevation_range': float(elev_range),
            'mean_slope': float(mean_slope),
            'n_valid_points': len(valid_elev),
            'notes': f'{len(valid_elev)}/{len(track_df)} points valid'
        }
        # Add coupling metrics if available
        if track_coupling is not None:
            if 'FSI' in track_coupling:
                result['FSI'] = track_coupling['FSI']
            if 'duration_h' in track_coupling:
                result['duration_h'] = track_coupling['duration_h']
        return result

    def compute_tsi_for_all_tracks(self, tracks_dict, coupling_df) :
        """Compute TSI for all tracks"""
        all_results = []
        for track_id, track_df in tracks_dict.items():
            try:
                # Ensure track_df is a DataFrame, not a Serie
                if isinstance(track_df, pd.Series):
                    continue
                # Get elevation profile
                track_with_elev = self.get_elevation_profile(track_df)
                # Get coupling metrics if available
                track_coupling = None
                if coupling_df is not None and not coupling_df.empty:
                    if 'track_id' in coupling_df.columns:
                        mask = coupling_df['track_id'] == track_id
                        if mask.any():
                            track_coupling = coupling_df[mask].iloc[0].to_dict()
                    elif track_id in coupling_df.index:
                        track_coupling = coupling_df.loc[track_id].to_dict()
                # Compute TSI
                tsi_result = self.compute_tsi(track_with_elev, track_coupling)
                tsi_result['track_id'] = track_id
                # Calculate duration if available
                if 'time' in track_df.columns:
                    times = pd.to_datetime(track_df['time'])
                    duration_h = (times.max() - times.min()).total_seconds() / 3600
                    tsi_result['duration_h'] = duration_h
                all_results.append(tsi_result)
                std_print('INFO', 'ea', f'Track {track_id} TSI computed', f'TSI={tsi_result["TSI"]:.3f} ({tsi_result["TSI_class"]})')
            except Exception as e:
                std_print('ERROR', 'ea', f'Track {track_id} failed', str(e)[:60])
                traceback.print_exc()
                continue
        if all_results:
            tsi_df = pd.DataFrame(all_results)
            return tsi_df
        else:
            return pd.DataFrame()

    def export_elevation_metrics(self, tsi_df, output_dir):
        if tsi_df.empty:
            std_print('WARN', 'ea', 'No TSI results to export')
            return
        os.makedirs(output_dir, exist_ok=True)
        # Save CSV
        csv_path = os.path.join(output_dir, 'elevation_metrics.csv')
        tsi_df.to_csv(csv_path, index=False)
        std_print('SAVED', 'ea', 'Elevation metrics', csv_path)
        # Save summary report
        report_path = os.path.join(output_dir, 'elevation_summary.txt')
        with open(report_path, 'w') as f:
            f.write("ELEVATION & TERRAIN ANALYSIS SUMMARY\n\n")
            f.write(f"Total tracks analyzed: {len(tsi_df)}\n")
            f.write(f"Tracks with valid elevation data: {(tsi_df['n_valid_points'] > 0).sum()}\n\n")
            # TSI statistics
            f.write("TOPOGRAPHIC SENSITIVITY INDEX (TSI)\n\n")
            valid_tsi = tsi_df[tsi_df['TSI'].notna()]['TSI']
            if len(valid_tsi) > 0:
                f.write(f"Mean TSI: {valid_tsi.mean():.3f}\n")
                f.write(f"Std Dev:  {valid_tsi.std():.3f}\n")
                f.write(f"Min TSI:  {valid_tsi.min():.3f}\n")
                f.write(f"Max TSI:  {valid_tsi.max():.3f}\n\n")
                # TSI classification
                f.write("TSI Classification Distribution:\n")
                class_dist = tsi_df['TSI_class'].value_counts()
                for cls, count in class_dist.items():
                    pct = 100 * count / len(tsi_df)
                    f.write(f"  {cls}: {count} tracks ({pct:.1f}%)\n")
                f.write("\n")
            # Component effects
            f.write("TERRAIN COMPONENT EFFECTS\n\n")
            if 'elevation_effect' in tsi_df.columns:
                f.write(f"Elevation effect (mean): {tsi_df['elevation_effect'].mean():.3f}\n")
            if 'slope_effect' in tsi_df.columns:
                f.write(f"Slope effect (mean):     {tsi_df['slope_effect'].mean():.3f}\n")
            if 'roughness_effect' in tsi_df.columns:
                f.write(f"Roughness effect (mean): {tsi_df['roughness_effect'].mean():.3f}\n")
            f.write("\n")
            # Elevation statistics
            f.write("ELEVATION STATISTICS\n\n")
            if 'mean_elevation' in tsi_df.columns:
                f.write(f"Mean track elevation: {tsi_df['mean_elevation'].mean():.1f} m\n")
            if 'max_elevation' in tsi_df.columns:
                f.write(f"Max track elevation:  {tsi_df['max_elevation'].max():.1f} m\n")
            if 'elevation_range' in tsi_df.columns:
                f.write(f"Mean elevation range: {tsi_df['elevation_range'].mean():.1f} m\n")
            if 'elevation_std' in tsi_df.columns:
                f.write(f"Mean elevation std:   {tsi_df['elevation_std'].mean():.1f} m\n")
            f.write("\n")
            # TSI-duration relationship
            if 'duration_h' in tsi_df.columns:
                f.write("TSI vs DURATION RELATIONSHIP\n\n")
                valid_pairs = tsi_df.dropna(subset=['TSI', 'duration_h'])
                if len(valid_pairs) > 2:
                    corr = valid_pairs['TSI'].corr(valid_pairs['duration_h'])
                    f.write(f"Correlation (TSI vs duration): {corr:.3f}\n")
                    if corr < -0.5:
                        f.write("Interpretation: Higher TSI → Shorter duration (terrain-sensitive tracks shorter)\n")
                    elif corr > 0.5:
                        f.write("Interpretation: Higher TSI → Longer duration\n")
                    else:
                        f.write("Interpretation: Weak correlation\n")
                f.write("\n")
        std_print('SAVED', 'ea', 'Summary report', report_path)

    def plot_elevation_metrics(self, tsi_df: pd.DataFrame, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # 1. TSI histogram
        ax = axes[0, 0]
        valid_tsi = tsi_df[tsi_df['TSI'].notna()]['TSI']
        if len(valid_tsi) > 0:
            ax.hist(valid_tsi, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(valid_tsi.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {valid_tsi.mean():.3f}')
            ax.set_xlabel('TSI (Topographic Sensitivity Index)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('(a) TSI Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        # 2. TSI vs Duration
        ax = axes[0, 1]
        if 'duration_h' in tsi_df.columns:
            valid = tsi_df.dropna(subset=['TSI', 'duration_h'])
            if len(valid) > 0:
                scatter = ax.scatter(valid['TSI'], valid['duration_h'], c=valid['elevation_range'] if 'elevation_range' in valid.columns else 'steelblue', cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=1)
                ax.set_xlabel('TSI', fontsize=11, fontweight='bold')
                ax.set_ylabel('Duration (hours)', fontsize=11, fontweight='bold')
                ax.set_title('(b) TSI vs Track Duration', fontsize=12, fontweight='bold')
                if 'elevation_range' in valid.columns:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Elevation Range (m)', fontsize=10)
                ax.grid(True, alpha=0.3)
                # Add correlation
                corr = valid['TSI'].corr(valid['duration_h'])
                ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        # 3. Component effects
        ax = axes[1, 0]
        components = []
        means = []
        if 'elevation_effect' in tsi_df.columns:
            components.append('Elevation')
            means.append(tsi_df['elevation_effect'].mean())
        if 'slope_effect' in tsi_df.columns:
            components.append('Slope')
            means.append(tsi_df['slope_effect'].mean())
        if 'roughness_effect' in tsi_df.columns:
            components.append('Roughness')
            means.append(tsi_df['roughness_effect'].mean())
        if components:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            bars = ax.bar(components, means, color=colors[:len(components)], edgecolor='black', linewidth=1.5, alpha=0.7)
            ax.set_ylabel('Mean Effect Value', fontsize=11, fontweight='bold')
            ax.set_title('(c) TSI Component Effects', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        # 4. Elevation range vs TSI
        ax = axes[1, 1]
        if 'elevation_range' in tsi_df.columns:
            valid = tsi_df.dropna(subset=['TSI', 'elevation_range'])
            if len(valid) > 0:
                scatter = ax.scatter(valid['elevation_range'], valid['TSI'], c=valid['duration_h'] if 'duration_h' in valid.columns else 'steelblue', cmap='plasma', s=100, alpha=0.6, edgecolors='black', linewidth=1)
                ax.set_xlabel('Elevation Range (m)', fontsize=11, fontweight='bold')
                ax.set_ylabel('TSI', fontsize=11, fontweight='bold')
                ax.set_title('(d) Elevation Range vs TSI', fontsize=12, fontweight='bold')
                if 'duration_h' in valid.columns:
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label('Duration (h)', fontsize=10)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'elevation_metrics_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        std_print('SAVED', 'ea', 'Elevation plots', plot_path)


def run_elevation_analysis(tracks_dict, coupling_df, output_dir = 'output', hgt_dir = '../Data/Elevation/HGT-files/') :
    """
    1. Loads SRTM HGT elevation data
    2. Extracts elevation profiles for each track
    3. Computes Topographic Sensitivity Index (TSI)
    4. Exports elevation_metrics.csv for statistical analysis
    """
    std_print('INFO', 'ea', '--- Elevation Analysis ---')
    std_print('INFO', 'ea', 'Starting elevation analysis', f'{len(tracks_dict)} tracks')
    # Initialize terrain analyzer
    terrain = TerrainAnalyzer(hgt_dir=hgt_dir)

    # Compute TSI for all tracks
    tsi_df = terrain.compute_tsi_for_all_tracks(tracks_dict, coupling_df)
    if tsi_df.empty:
        std_print('WARN', 'ea', 'No TSI results computed', 'Check elevation data')
        return pd.DataFrame()
    # Export results
    terrain_output_dir = os.path.join(output_dir, 'elevation_analysis')
    os.makedirs(terrain_output_dir, exist_ok=True)
    terrain.export_elevation_metrics(tsi_df, terrain_output_dir)
    terrain.plot_elevation_metrics(tsi_df, terrain_output_dir)
    std_print('EXPORT', 'ea', 'Elevation metrics', os.path.join(terrain_output_dir, 'elevation_metrics.csv'))
    return tsi_df
