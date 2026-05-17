import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
from PIL import Image

def plot_all_tracks_on_soil_map(csv_path, soil_image_path, image_extent, output_file="all_tracks_overlay.png"):
    df = pd.read_csv(csv_path)
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(image_extent, crs=ccrs.PlateCarree())

    if os.path.exists(soil_image_path):
        img = Image.open(soil_image_path)
        ax.imshow(img, extent=image_extent, transform=ccrs.PlateCarree(), origin='upper')
        print(f"Overlaying image: {soil_image_path}")
    else:
        print("Image not found, using basic land features.")
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#444444', zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor='gray', alpha=0.5, zorder=3)
    track_ids = df['track_id'].unique()
    cmap = plt.colormaps.get_cmap('tab10')
    track_colors = [cmap(i % 10) for i in range(len(track_ids))]
    for i, tid in enumerate(track_ids):
        track_data = df[df['track_id'] == tid]
        current_color = track_colors[i]
        ax.plot(track_data['longitude'], track_data['latitude'], color=current_color, linewidth=2, label=f'Track {tid}', transform=ccrs.PlateCarree(), zorder=5)
        ax.scatter(track_data['longitude'].iloc[0], track_data['latitude'].iloc[0],  color=current_color, marker='^', s=50, transform=ccrs.PlateCarree(), zorder=6)
    # Gridlines and Labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.show()
    plt.close(fig)
    print(f"Plot saved as {output_file}")

# Format: [West, East, South, North]
INDIA_EXTENT = [68, 97, 1, 40.0]

# Path to your image and csv
CSV_FILE = 'output/MD-coordinates-ALL.csv'
SOIL_IMG = 'esdac/Soil-map-of-India.png'

plot_all_tracks_on_soil_map(CSV_FILE, SOIL_IMG, INDIA_EXTENT)
