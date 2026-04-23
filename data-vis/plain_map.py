import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Domain boundaries
clipped_min_lat = 7.0  # N
clipped_max_lat = 30.0  # N
clipped_min_lon = 70.0  # E
clipped_max_lon = 95.0  # E
full_extent = [clipped_min_lon, clipped_max_lon, clipped_min_lat, clipped_max_lat]

# Font sizes
FS_TITLE = 14
FS_LABEL = 11
FS_TICK = 10
FS_LEGEND = 10
FS_ANNOT = 9

# Major Indian states to label
STATE_LABELS = {
    "Maharashtra": (76, 20),
    "Karnataka": (76, 14),
    "Kerala": (76, 10),
    "Tamil Nadu": (79, 11),
    "Andhra Pradesh": (79, 15),
    "Telangana": (79, 18),
    "Odisha": (84, 20),
    "West Bengal": (88, 23),
    "Bihar": (86, 26),
    "Uttar Pradesh": (81, 27),
    "Madhya Pradesh": (78, 23),
    "Rajasthan": (74, 27),
    "Gujarat": (72, 23),
    "Assam": (93, 26),
    "Chhattisgarh": (82, 21),
    "Jharkhand": (85, 24),
    "Haryana": (76, 29),
    "Punjab": (75, 31),
    "Jammu & Kashmir": (76, 34),
    "Himachal Pradesh": (77, 32),
    "Uttarakhand": (79, 39),
}

# Water bodies and geographic features to label
GEOGRAPHIC_FEATURES = {
    "Bay of Bengal": (88, 18),
    "Arabian Sea": (68, 15),
    "Indian Ocean": (75, 8),
    "Gulf of Khambhat": (72, 21),
    "Palk Strait": (79.5, 9.5),
    "Gulf of Mannar": (78.5, 8.5),
    "Lakshadweep Sea": (73, 12),
}

# Neighboring countries
COUNTRY_LABELS = {
    "Pakistan": (69, 28),
    "Nepal": (83.5, 28.5),
    "Bhutan": (90.6, 27.5),
    "Bangladesh": (90, 24.5),
    "Myanmar": (96, 21),
    "Sri Lanka": (80.7, 7),
    "China": (82, 32),
}


def plot_study_domain_with_features(output_filename):
    """Detailed version with all labels"""
    # Figure with PlateCarree projection
    fig = plt.figure(figsize=(14, 11))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]
    ax.set_extent(full_extent, crs=ccrs.PlateCarree())

    ax.add_feature(
        cfeature.LAND, facecolor="lightgray", edgecolor="black", linewidth=0.5
    )
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5, linewidth=0.8, zorder=6)
    ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

    # Country borders
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_0_countries",
            scale="50m",
            facecolor="none",
            edgecolor="darkgray",
            linewidth=1.2,
        )
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Subtitle with coordinates
    text = (
        f"{full_extent[2]}°N to {full_extent[3]}°N, {full_extent[0]}°E to {full_extent[1]}°E",
    )
    ax.text(
        0.5,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=FS_ANNOT,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # State labels (India)
    print("Adding Indian state labels...")
    for state, (lon, lat) in STATE_LABELS.items():
        # Check if within map bounds
        if (
            clipped_min_lon <= lon <= clipped_max_lon
            and clipped_min_lat <= lat <= clipped_max_lat
        ):
            ax.text(
                lon,
                lat,
                state,
                transform=ccrs.PlateCarree(),
                fontsize=FS_LABEL,
                fontweight="bold",
                ha="center",
                va="center",
                color="darkred",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                ),
                zorder=4,
            )

    # Geographic feature labels (water bodies)
    print("Adding geographic feature labels...")
    for feature, (lon, lat) in GEOGRAPHIC_FEATURES.items():
        if (
            clipped_min_lon <= lon <= clipped_max_lon
            and clipped_min_lat <= lat <= clipped_max_lat
        ):
            ax.text(
                lon,
                lat,
                feature,
                transform=ccrs.PlateCarree(),
                fontsize=FS_LABEL,
                fontstyle="italic",
                ha="center",
                va="center",
                color="darkblue",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="lightblue",
                    alpha=0.5,
                    edgecolor="darkblue",
                    linewidth=0.5,
                ),
                zorder=3,
            )

    # Neighboring country labels
    print("Adding neighboring country labels...")
    for country, (lon, lat) in COUNTRY_LABELS.items():
        if (
            clipped_min_lon <= lon <= clipped_max_lon
            and clipped_min_lat <= lat <= clipped_max_lat
        ):
            ax.text(
                lon,
                lat,
                country,
                transform=ccrs.PlateCarree(),
                fontsize=FS_LABEL,
                fontweight="normal",
                ha="center",
                va="center",
                color="darkgreen",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="lightgreen",
                    alpha=0.5,
                    edgecolor="darkgreen",
                    linewidth=0.5,
                ),
                zorder=4,
            )

    # Highlight major cities (optional)
    MAJOR_CITIES = {
        "Mumbai": (72.8, 19.1),
        "Chennai": (80.3, 13.1),
        "Kolkata": (88.2, 22),
        "Delhi": (77.2, 28.6),
        "Bangalore": (77.6, 13.0),
        "Hyderabad": (78.5, 17.4),
        "Ahmedabad": (72.6, 23.0),
        "Pune": (73.9, 18.5),
        "Visakhapatnam": (83.3, 17.7),
    }

    print("Adding major city markers...")
    for city, (lon, lat) in MAJOR_CITIES.items():
        if (
            clipped_min_lon <= lon <= clipped_max_lon
            and clipped_min_lat <= lat <= clipped_max_lat
        ):
            ax.plot(
                lon, lat, "ko", markersize=3, transform=ccrs.PlateCarree(), zorder=5
            )
            ax.text(
                lon + 0.3,
                lat + 0.3,
                city,
                transform=ccrs.PlateCarree(),
                fontsize=FS_TICK,
                ha="left",
                va="bottom",
                color="black",
                zorder=5,
            )

    legend_elements = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="lightgray",
            edgecolor="darkred",
            label="Indian States",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="lightblue",
            edgecolor="darkblue",
            label="Water Bodies",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="lightgreen",
            edgecolor="darkgreen",
            label="Countries",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=FS_LABEL,
        framealpha=0.9,
        frameon=True,
    )

    # Title
    ax.set_title(
        "Indian Subcontinent and Bay of Bengal",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=10,
    )

    # Save
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Successfully generated detailed map: {output_filename}")
    print(
        f"Added: {len(STATE_LABELS)} Indian states, {len(GEOGRAPHIC_FEATURES)} water bodies, {len(COUNTRY_LABELS)} countries"
    )


def plot_study_domain_minimal_labels(output_filename):
    """Version with minimal labels - just water bodies and major geographic features"""
    fig = plt.figure(figsize=(14, 11))
    ax: GeoAxes = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())  # type: ignore[assignment]
    ax.set_extent(full_extent, crs=ccrs.PlateCarree())

    # Geographic features
    ax.add_feature(cfeature.LAND, facecolor="lightgray", edgecolor="black", zorder=1)
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

    # Water body labels
    WATER_FEATURES = {
        "Bay of Bengal": (88, 15),
        "Arabian Sea": (72, 14),
        "Indian Ocean": (73, 7),
        "Gulf of Khambhat": (72, 20),
    }

    for feature, (lon, lat) in WATER_FEATURES.items():
        if (
            clipped_min_lon <= lon <= clipped_max_lon
            and clipped_min_lat <= lat <= clipped_max_lat
        ):
            ax.text(
                lon,
                lat,
                feature,
                transform=ccrs.PlateCarree(),
                fontsize=FS_LABEL,
                fontstyle="italic",
                fontweight="bold",
                ha="center",
                va="center",
                color="darkblue",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="darkblue",
                    linewidth=1,
                ),
                zorder=4,
            )

    ax.set_title(
        "Study Domain with Major Geographic Features",
        fontsize=FS_TITLE,
        fontweight="bold",
        pad=10,
    )

    # Save
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated minimal label map: {output_filename}")


if __name__ == "__main__":
    # Generate detailed version with all labels
    plot_study_domain_with_features("study_domain_detailed_labels.png")
    # Generate minimal version
    plot_study_domain_minimal_labels("study_domain_minimal.png")
