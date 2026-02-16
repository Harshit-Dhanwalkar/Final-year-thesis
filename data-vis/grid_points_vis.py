import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

ERA5_FILE = "../Data/single-levels/ncfiles/July-2024/data_stream-oper_stepType-avg.nc"
OUTPUT_DIR = "era5_grid_visualization"


def visualize_era5_grid(era5_file, output_dir="era5_grid_visualization"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading ERA5 file: {era5_file}")
    ds = xr.open_dataset(era5_file)

    # Get lat/lon coordinates
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    # Mesh grid
    LON, LAT = np.meshgrid(lon, lat)

    print(f"Latitude range: {lat.min():.2f}° to {lat.max():.2f}°")
    print(f"Longitude range: {lon.min():.2f}° to {lon.max():.2f}°")
    print(f"Number of latitude points: {len(lat)}")
    print(f"Number of longitude points: {len(lon)}")

    # Calculate grid spacing
    lat_spacing = np.mean(np.diff(lat))
    lon_spacing = np.mean(np.diff(lon))

    print(f"Latitude spacing: {lat_spacing:.4f}° ({lat_spacing*111:.2f} km)")
    print(
        f"Longitude spacing: {lon_spacing:.4f}° ({lon_spacing*111*np.cos(np.radians(np.mean(lat))):.2f} km)"
    )
    # Figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # Add map features
    ax.coastlines(resolution="50m", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.5)
    # Plot grid points
    ax.scatter(
        LON,
        LAT,
        s=1,
        c="red",
        alpha=0.5,
        transform=ccrs.PlateCarree(),
        label=f"Grid points ({len(lat)}x{len(lon)})",
    )
    # Plot grid lines (every 5th point for clarity if too dense)
    if len(lat) > 50:  # If too many points, skip some
        step = max(1, len(lat) // 20)
        for i in range(0, len(lat), step):
            ax.plot(
                lon,
                [lat[i]] * len(lon),
                "k-",
                linewidth=0.3,
                alpha=0.2,
                transform=ccrs.PlateCarree(),
            )
        for j in range(0, len(lon), step):
            ax.plot(
                [lon[j]] * len(lat),
                lat,
                "k-",
                linewidth=0.3,
                alpha=0.2,
                transform=ccrs.PlateCarree(),
            )
    else:
        # Plot all grid lines if not too many
        for i in range(len(lat)):
            ax.plot(
                lon,
                [lat[i]] * len(lon),
                "k-",
                linewidth=0.3,
                alpha=0.2,
                transform=ccrs.PlateCarree(),
            )
        for j in range(len(lon)):
            ax.plot(
                [lon[j]] * len(lat),
                lat,
                "k-",
                linewidth=0.3,
                alpha=0.2,
                transform=ccrs.PlateCarree(),
            )
    ax.set_global()
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    ax.set_title(
        f"ERA5 Grid Points\nSpacing: {lat_spacing:.4f}° × {lon_spacing:.4f}°",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=10)
    plot_path = os.path.join(output_dir, "ERA5_grid_over_globe.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.5)
    # India bounds (approximate)
    india_extent = [70, 95, 10, 30]  # [west, east, south, north]
    ax.set_extent(india_extent, crs=ccrs.PlateCarree())
    # Plot grid points
    ax.scatter(
        LON,
        LAT,
        s=1,
        c="red",
        alpha=0.5,
        edgecolors="darkred",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        label="Grid points",
    )
    # Plot grid lines
    for i in range(len(lat)):
        ax.plot(
            lon,
            [lat[i]] * len(lon),
            "k-",
            linewidth=0.5,
            alpha=0.2,
            transform=ccrs.PlateCarree(),
        )
    for j in range(len(lon)):
        ax.plot(
            [lon[j]] * len(lat),
            lat,
            "k-",
            linewidth=0.5,
            alpha=0.2,
            transform=ccrs.PlateCarree(),
        )
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    ax.set_title(
        f"ERA5 Grid Points - India Region\nSpacing: {lat_spacing:.4f}° x {lon_spacing:.4f}°",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=10)
    plot_path = os.path.join(output_dir, "ERA5_grid_india_region.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ds.close()


if __name__ == "__main__":
    visualize_era5_grid(ERA5_FILE, OUTPUT_DIR)
