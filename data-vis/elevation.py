import rasterio
import numpy as np
from rasterio.enums import Resampling

def load_hgt_tile(tile_path, resample_factor=None):
    """
    Load SRTM HGT tile with proper nodata handling

    Parameters:
    -----------
    tile_path : str
        Path to .hgt file
    resample_factor : float, optional
        Factor to resample (e.g., 0.5 for half resolution)

    Returns:
    --------
    elevation : np.ndarray (float32)
        Elevation data with NaN for nodata
    transform : rasterio.transform
        Geo-transform
    crs : rasterio.crs.CRS
        Coordinate reference system
    """
    with rasterio.open(tile_path) as src:
        # Get nodata value (usually -32768 for SRTM)
        nodata = src.nodata

        if resample_factor is not None:
            # Calculate new dimensions
            new_width = int(src.width * resample_factor)
            new_height = int(src.height * resample_factor)

            # Read with resampling
            elevation = src.read(
                1,
                out_shape=(new_height, new_width),
                resampling=Resampling.bilinear
            )
            # Update transform
            transform = src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )
        else:
            elevation = src.read(1)
            transform = src.transform

        # Convert to float and handle nodata
        elevation_float = elevation.astype(np.float32)
        if nodata is not None:
            elevation_float[elevation_float == nodata] = np.nan

        return elevation_float, transform, src.crs

# Test it
elevation_data, transform, crs = load_hgt_tile('../Data/Elevation/E44/N16E078.hgt')
print(f"Tile: N16E078.hgt")
print(f"Shape: {elevation_data.shape}")
print(f"CRS: {crs}")
print(f"Bounds: {rasterio.transform.array_bounds(elevation_data.shape[0], elevation_data.shape[1], transform)}")
print(f"Min elevation: {np.nanmin(elevation_data):.1f} m")
print(f"Max elevation: {np.nanmax(elevation_data):.1f} m")
print(f"Mean elevation: {np.nanmean(elevation_data):.1f} m")
print(f"NaN percentage: {100 * np.sum(np.isnan(elevation_data)) / elevation_data.size:.2f}%")

# You can also check what coordinates this tile covers:
# N16E078 means: Latitude 16°N to 17°N, Longitude 78°E to 79°E
