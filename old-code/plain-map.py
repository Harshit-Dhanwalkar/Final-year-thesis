import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
import numpy as np

# Domain boundaries
LAT_MIN = 5
LAT_MAX = 30
LON_MIN = 70
LON_MAX = 100

# Major Indian states to label (with approximate coordinates)
STATE_LABELS = {
    'Maharashtra': (76, 20),
    'Karnataka': (76, 14),
    'Kerala': (76, 10),
    'Tamil Nadu': (79, 11),
    'Andhra Pradesh': (79, 15),
    'Telangana': (79, 18),
    'Odisha': (84, 20),
    'West Bengal': (88, 23),
    'Bihar': (86, 26),
    'Uttar Pradesh': (81, 27),
    'Madhya Pradesh': (78, 23),
    'Rajasthan': (74, 27),
    'Gujarat': (72, 23),
    'Assam': (93, 26),
    'Chhattisgarh': (82, 21),
    'Jharkhand': (85, 24),
    'Haryana': (76, 29),
    'Punjab': (75, 31),
    'Jammu & Kashmir': (76, 34),
    'Himachal Pradesh': (77, 32),
    'Uttarakhand': (79, 39),
}

# Water bodies and geographic features to label
GEOGRAPHIC_FEATURES = {
    'Bay of Bengal': (88, 18),
    'Arabian Sea': (68, 15),
    'Indian Ocean': (75, 8),
    'Gulf of Khambhat': (72, 21),
    'Palk Strait': (79.5, 9.5),
    'Gulf of Mannar': (78.5, 8.5),
    'Lakshadweep Sea': (73, 12),
}

# Neighboring countries
COUNTRY_LABELS = {
    'Pakistan': (69, 28),
    'Nepal': (83.5, 28.5),
    'Bhutan': (90.6, 27.5),
    'Bangladesh': (90, 24.5),
    'Myanmar': (96, 21),
    'Sri Lanka': (80.7, 7),
    'China': (82, 32),
}

def plot_study_domain_with_features(lat_min, lat_max, lon_min, lon_max, output_filename="study-domain-map.png"):
    try:
        # Create figure with PlateCarree projection
        fig = plt.figure(figsize=(14, 12))
        ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # type: ignore[assignment]
        
        # Set the extent to match your study domain
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # Add detailed geographic features
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor='black', zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
        ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.8, linewidth=1.0, zorder=2)
        ax.add_feature(cfeature.STATES, linewidth=0.8, linestyle='--', alpha=0.6, zorder=2)
        ax.add_feature(cfeature.RIVERS, alpha=0.6, zorder=1, linewidth=0.8)

        # Add country borders
        ax.add_feature(cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='50m',
            facecolor='none',
            edgecolor='darkgray',
            linewidth=1.2
        ))

        # Configure gridlines
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.6,
            color="gray",
            alpha=0.6,
            linestyle=":",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        
        # Set ticks at regular intervals
        lon_ticks = np.arange(70, 101, 5)  # Every 5 degrees from 70 to 100
        lat_ticks = np.arange(5, 36, 5)    # Every 5 degrees from 5 to 35
        
        gl.xlocator = plt.FixedLocator(lon_ticks)
        gl.ylocator = plt.FixedLocator(lat_ticks)

        # Labels and Title
        ax.set_title(
            f"Study Domain: Indian Subcontinent and Surrounding Oceans",
            fontsize=30,
            pad=25,
            fontweight='bold'
        )
        
        # Add subtitle with coordinates
        ax.text(0.5, 0.02, f"{lat_min}°N to {lat_max}°N, {lon_min}°E to {lon_max}°E", 
                transform=ax.transAxes, fontsize=26, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Add state labels (India)
        print("Adding Indian state labels...")
        for state, (lon, lat) in STATE_LABELS.items():
            # Check if within map bounds
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                ax.text(lon, lat, state,
                       transform=ccrs.PlateCarree(),
                       fontsize=11,
                       fontweight='bold',
                       ha='center',
                       va='center',
                       color='darkred',
                       bbox=dict(boxstyle="round,pad=0.2",
                                facecolor="white",
                                alpha=0.7,
                                edgecolor='none'),
                       zorder=4)

        # Add geographic feature labels (water bodies)
        print("Adding geographic feature labels...")
        for feature, (lon, lat) in GEOGRAPHIC_FEATURES.items():
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                ax.text(lon, lat, feature,
                       transform=ccrs.PlateCarree(),
                       fontsize=16,
                       fontstyle='italic',
                       ha='center',
                       va='center',
                       color='darkblue',
                       bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor="lightblue", 
                                alpha=0.5,
                                edgecolor='darkblue',
                                linewidth=0.5),
                       zorder=3)

        # Add neighboring country labels
        print("Adding neighboring country labels...")
        for country, (lon, lat) in COUNTRY_LABELS.items():
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                ax.text(lon, lat, country,
                       transform=ccrs.PlateCarree(),
                       fontsize=17,
                       fontweight='normal',
                       ha='center',
                       va='center',
                       color='darkgreen',
                       bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor="lightgreen", 
                                alpha=0.5,
                                edgecolor='darkgreen',
                                linewidth=0.5),
                       zorder=4)

        # Highlight major cities (optional)
        MAJOR_CITIES = {
            'Mumbai': (72.8, 19.1),
            'Chennai': (80.3, 13.1),
            'Kolkata': (88.2, 22),
            'Delhi': (77.2, 28.6),
            'Bangalore': (77.6, 13.0),
            'Hyderabad': (78.5, 17.4),
            'Ahmedabad': (72.6, 23.0),
            'Pune': (73.9, 18.5),
            'Visakhapatnam': (83.3, 17.7),
        }
        
        print("Adding major city markers...")
        for city, (lon, lat) in MAJOR_CITIES.items():
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                ax.plot(lon, lat, 'ko', markersize=3, transform=ccrs.PlateCarree(), zorder=5)
                ax.text(lon + 0.3, lat + 0.3, city, 
                       transform=ccrs.PlateCarree(),
                       fontsize=8,
                       ha='left',
                       va='bottom',
                       color='black',
                       zorder=5)

        # Add legend for different types of labels
        from matplotlib.patches import Rectangle
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='darkred', label='Indian States'),
            Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='darkblue', label='Water Bodies'),
            Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='darkgreen', label='Countries'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
                 framealpha=0.9, frameon=True)

        # Save the figure with high resolution
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nSuccessfully generated detailed map: {output_filename}")
        print(f"Domain: Latitude {lat_min}°N to {lat_max}°N, Longitude {lon_min}°E to {lon_max}°E")
        print(f"Added: {len(STATE_LABELS)} Indian states, {len(GEOGRAPHIC_FEATURES)} water bodies, {len(COUNTRY_LABELS)} countries")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

# Version with minimal labels (just water bodies and major features)
def plot_study_domain_minimal_labels(lat_min, lat_max, lon_min, lon_max, output_filename="study-domain-minimal.png"):
    """Version with minimal labels - just water bodies and major geographic features"""
    try:
        fig = plt.figure(figsize=(12, 10))
        ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # type: ignore[assignment]
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # Geographic features
        ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor='black', zorder=1)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle="-", alpha=0.8, linewidth=1.0, zorder=2)

        # Gridlines
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.6,
            color="gray",
            alpha=0.6,
            linestyle=":",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        
        lon_ticks = np.arange(70, 101, 5)
        lat_ticks = np.arange(5, 36, 5)
        
        gl.xlocator = plt.FixedLocator(lon_ticks)
        gl.ylocator = plt.FixedLocator(lat_ticks)

        ax.set_title(
            f"Study Domain with Major Geographic Features",
            fontsize=30,
            pad=20
        )

        # Add only water body labels
        WATER_FEATURES = {
            'Bay of Bengal': (88, 15),
            'Arabian Sea': (72, 14),
            'Indian Ocean': (73, 7),
            'Gulf of Khambhat': (72, 20),
        }
        
        for feature, (lon, lat) in WATER_FEATURES.items():
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                ax.text(lon, lat, feature, 
                       transform=ccrs.PlateCarree(),
                       fontsize=12,
                       fontstyle='italic',
                       fontweight='bold',
                       ha='center',
                       va='center',
                       color='darkblue',
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="white", 
                                alpha=0.8,
                                edgecolor='darkblue',
                                linewidth=1),
                       zorder=4)

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated minimal label map: {output_filename}")

    except Exception as e:
        print(f"Error in minimal labels version: {e}")

if __name__ == "__main__":
    print("Generating study domain maps with geographic features...")
    # Generate detailed version with all labels
    plot_study_domain_with_features(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, "study-domain-detailed-labels.png")
    # Generate minimal version
    plot_study_domain_minimal_labels(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, "study-domain-minimal.png")
    print("\nMaps created successfully!")
    print("1. study_domain_detailed_labels.png - With state names, water bodies, and country labels")
    print("2. study_domain_minimal.png - Only major water body labels")
