import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader

def is_point_land(lat: float, lon: float) -> bool:
    """Check if a point is over land."""
    try:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False
        point = sgeom.Point(lon, lat)
        # '110m'
        # land_shp = shpreader.natural_earth(
        #     resolution='110m',
        #     category='physical',
        #     name='land'
        # )
        land_shp = shpreader.natural_earth(
            resolution='50m',
            category='physical',
            name='land'
        )
        # '10m' if slow
        # land_shp = shpreader.natural_earth(
        #     resolution='10m',
        #     category='physical',
        #     name='land'
        # )
        reader = shpreader.Reader(land_shp)
        for geometry in reader.geometries():
            if geometry.contains(point):
                return True
        return False
    except Exception:
        return False
