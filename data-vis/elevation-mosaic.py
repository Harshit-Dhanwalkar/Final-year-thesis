"""
sudo apt install gdal-bin

This script will create scalled down version of elevation map:
"""
import os
import glob
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import subprocess
import shutil

# 1. Find Files
search_path = "../Data/Elevation/HGT-files/**/*.hgt"
files_to_mosaic = glob.glob(search_path, recursive=True)

if not files_to_mosaic:
    print("No HGT files found! Check path.")
    exit()

print(f"Found {len(files_to_mosaic)} tiles. Building Virtual Dataset (VRT)...")

# 2. Build a VRT using the system's GDAL utility
vrt_path = "temp_mosaic.vrt"

# Use subprocess to call gdalbuildvrt
try:
    subprocess.run(['gdalbuildvrt', vrt_path] + files_to_mosaic, check=True)
except Exception as e:
    print(f"Error building VRT: {e}")
    exit()

# 3. Visualization (Downsampled)
with rasterio.open(vrt_path) as src:
    # Use a scale factor to avoid OOM
    target_width = 1024 * 15
    scale_factor = target_width / src.width
    # scale_factor = 0.1
    out_shape = (int(src.height * scale_factor), int(src.width * scale_factor))
    print(f"Reading downsampled preview {out_shape[1]}x{out_shape[0]}...")
    vis_data = src.read(
        1,
        out_shape=out_shape,
        resampling=rasterio.enums.Resampling.bilinear
    ).astype(float)

    # Clean up 'void' pixels (SRTM usually uses -32768)
    vis_data[vis_data < -100] = 0 
    # Normalize for plotting
    vmin = np.nanpercentile(vis_data, 2)
    vmax = np.nanpercentile(vis_data, 98)
    vis_data_normalized = np.clip((vis_data - vmin) / (vmax - vmin), 0, 1)

    plt.figure(figsize=(15, 10))
    # plt.imshow(vis_data_normalized, cmap='terrain', origin='upper')
    # plt.imshow(vis_data_normalized, cmap='gist_earth', origin='upper')
    plt.imshow(vis_data_normalized + 0.01, cmap='terrain', norm=LogNorm())
    plt.colorbar(label='Relative Elevation Intensity')
    plt.title(f"Mosaic of {len(files_to_mosaic)} tiles")
    plt.axis('off')

    output_png = "elevation-Indian-Subcontinent.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"Validation PNG saved as: {output_png}")

# The .vrt file is small (just text). Keep it to use in analyzer
print(f"VRT file remains at: {vrt_path}")
