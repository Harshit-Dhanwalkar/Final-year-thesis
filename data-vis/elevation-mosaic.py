"""
sudo apt install gdal-bin

This script will create scalled down version of elevation map:
"""

import os
import glob
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import subprocess

# Find Files
search_path = "../Data/Elevation/HGT-files/**/*.hgt"
files_to_mosaic = glob.glob(search_path, recursive=True)

if not files_to_mosaic:
    print("No HGT files found! Check path.")
    exit()

print(f"Found {len(files_to_mosaic)} tiles. Building Virtual Dataset (VRT)...")

# Build a VRT using the system's GDAL utility
vrt_path = "temp_mosaic.vrt"
if not os.path.exists(vrt_path):
    print(f"Building Virtual Dataset (VRT) from {len(files_to_mosaic)} tiles...")
    try:
        subprocess.run(["gdalbuildvrt", vrt_path] + files_to_mosaic, check=True)
    except Exception as e:
        print(f"Error building VRT: {e}")
        exit()
else:
    print(f"VRT already exists at {vrt_path}. Skipping build step.")

# Subprocess to call gdalbuildvrt
try:
    subprocess.run(["gdalbuildvrt", vrt_path] + files_to_mosaic, check=True)
except Exception as e:
    print(f"Error building VRT: {e}")
    exit()

# 3. Visualization (Downsampled)
with rasterio.open(vrt_path) as src:
    # Scale factor to avoid OOM
    target_width = 1024 * 10
    scale_factor = target_width / src.width
    out_shape = (int(src.height * scale_factor), int(src.width * scale_factor))
    print(f"Reading downsampled preview {out_shape[1]}x{out_shape[0]}...")
    vis_data = src.read(
        1, out_shape=out_shape, resampling=rasterio.enums.Resampling.bilinear
    ).astype(float)

    # Clean up 'void' pixels (SRTM usually uses -32768)
    vis_data[vis_data < 1] = 1
    log_norm = LogNorm(vmin=1, vmax=8000)

    plt.figure(figsize=(15, 10))
    im = plt.imshow(vis_data, cmap="terrain", norm=log_norm, origin="upper")
    # im = plt.imshow(vis_data, cmap='terrain', origin='upper')
    # im = plt.imshow(vis_data, cmap='gist_earth', origin='upper')

    cbar = plt.colorbar(im, orientation="vertical", pad=0.02)
    cbar.set_label("Elevation (Meters - Log Scale)")
    cbar.set_ticks([1, 10, 100, 1000, 8000])
    cbar.ax.set_yticklabels(["1m", "10m", "100m", "1km", "8km"])

    plt.title(f"{len(files_to_mosaic)} HGT tiles (SRTM data)")
    plt.axis("off")

    output_png = "elevation-Indian-Subcontinent.png"
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Saved as: {output_png}")
