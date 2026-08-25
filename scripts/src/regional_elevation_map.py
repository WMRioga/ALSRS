"""
Regional Elevation Map Module
==============================

Downloads an elevation map (DEM with a topographic color palette) of the
administrative region (state/department or country) that contains a given
point, using SRTM elevation data and FAO GAUL administrative boundaries.

This is a visual reference map only (not a feature source for the model —
see terrain_profile.py for the point-level elevation/slope/aspect stats
actually used in the pipeline). The DEM source here is intentionally kept
as SRTM for now; migrating it to Copernicus DEM (as terrain_profile.py
already does) is a possible future improvement, not done in this pass.

Dependencies:
    - requests
    - PIL (Pillow)
    - earthengine-api (ee)
    - tile_utils (custom module)
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import ee
import requests
from PIL import Image

from tile_utils import draw_marker, validate_coordinates

EE_PROJECT = "famous-strategy-376313"

try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)


# Topographic elevation palette: valleys/lowlands in green, mid-elevation zones in
# yellow/brown, peaks in white/gray (reference guide for elevation ranges
# in Colombia; adjust 'max' if your region is higher/lower)
ELEVATION_PALETTE = [
    '004400',  # 0 - 364 m: Deep valleys, lowlands or coastal plains
    '006600',  # 364 - 727 m: Lowlands / Tropical humid forests
    '38a800',  # 727 - 1,091 m: Low transition zone / Beginning of foothills
    '73d216',  # 1,091 - 1,455 m: Coffee-growing or agricultural zones, temperate climate
    'b2d235',  # 1,455 - 1,818 m: Mid-mountain slopes
    'fce94f',  # 1,818 - 2,182 m: Upper mid-elevation lands / Andean forests
    'e9b96e',  # 2,182 - 2,545 m: High Andean zones / Cold transition
    'c87d32',  # 2,545 - 2,909 m: Lower paramo / High mountain
    '8f5902',  # 2,909 - 3,273 m: Upper paramo / Cold rocky soils
    '5c3a21',  # 3,273 - 3,636 m: Super-paramo / Steep summits
    '8b8b8b',  # 3,636 - 4,000+ m: Highest peaks / Alpine rock gray
]

# GAUL feature collections and their name property, by region type
GAUL_LEVELS = {
    "country": ("FAO/GAUL/2015/level0", "ADM0_NAME"),
    "state": ("FAO/GAUL/2015/level1", "ADM1_NAME"),
}


def download_regional_elevation_map(
    lat: float,
    lon: float,
    region_type: str = "state",
    out_prefix: str = "regional_elevation",
    output_dir: str = "../../img/maps",
    dimensions: int = 720,
    min_elevation: float = 0,
    max_elevation: float = 4000,
    show_marker: bool = True,
) -> Path:
    """
    Downloads an elevation map (DEM with topographic palette) of the administrative
    region (state or country) that contains the given point.

    Uses SRTM digital elevation model data and FAO GAUL administrative boundaries
    to extract and visualize the terrain of the entire region. This is a visual
    reference map, not a source of model features.

    Args:
        lat: Latitude of the reference point in degrees (defines the region AND
             is marked on the map if show_marker=True)
        lon: Longitude of the reference point in degrees
        region_type: "state" (level 1, department/province) or "country" (level 0)
        out_prefix: Base name for the output PNG file
        output_dir: Directory where the image is saved (created if it doesn't exist)
        dimensions: Size in pixels of the longest side of the output image
        min_elevation: Lower bound of the color palette range (meters)
        max_elevation: Upper bound of the color palette range (meters)
        show_marker: If True, draws a red dot at the exact reference location

    Returns:
        pathlib.Path: Path to the saved elevation map image

    Raises:
        ValueError: If coordinates are invalid, region_type is not recognized,
                    or no administrative region contains the given point
    """
    # Validate inputs before proceeding
    validate_coordinates(lat, lon)

    if region_type not in GAUL_LEVELS:
        raise ValueError(
            f"region_type '{region_type}' not recognized. "
            f"Available options: {list(GAUL_LEVELS.keys())}"
        )

    collection_id, name_key = GAUL_LEVELS[region_type]

    # Create Earth Engine point geometry from lon/lat coordinates
    point = ee.Geometry.Point([lon, lat])

    # Filter the GAUL boundary collection to find the feature containing our point
    feature = ee.FeatureCollection(collection_id).filterBounds(point).first()

    # .first() returns null (not an exception) when no feature matches, so we
    # must check explicitly before calling .getInfo() on it, or Earth Engine
    # raises a cryptic server-side error instead of a clear message.
    if feature is None or feature.getInfo() is None:
        raise ValueError(
            f"No administrative region found containing point "
            f"(lat={lat}, lon={lon}) at region_type='{region_type}'. "
            f"The point may fall outside GAUL's coverage (e.g. over water) "
            f"or on a boundary gap."
        )

    # Extract the region name and geometry from Earth Engine
    region_name = feature.get(name_key).getInfo()
    geom = feature.geometry()
    print(f"[DEBUG] Detected zone: {region_name}")

    # Load SRTM elevation data (30m resolution, void-filled version)
    image = ee.Image('USGS/SRTMGL1_003').select('elevation')

    # Clip the elevation data to the administrative boundary
    clipped = image.clip(geom)

    # Apply visualization parameters: color palette and elevation range
    # This converts the single-band DEM into an RGB visualization image
    vis_image = clipped.visualize(
        min=min_elevation,
        max=max_elevation,
        palette=ELEVATION_PALETTE,
    )

    # Generate a thumbnail URL for the visualized image
    # The thumbnail is rendered server-side by Earth Engine
    thumbnail_url = vis_image.getThumbURL({
        'region': geom,
        'dimensions': dimensions,
        'format': 'png',
    })

    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    filename = f"{out_prefix}-v{timestamp}.png"

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_path = output_path / filename

    # Download the rendered thumbnail image from Earth Engine
    resp = requests.get(thumbnail_url, timeout=30)
    resp.raise_for_status()  # Raise an exception for HTTP errors

    # Open the downloaded image and convert to RGBA for marker drawing
    canvas = Image.open(BytesIO(resp.content)).convert("RGBA")

    if show_marker:
        # Linear interpolation within the geometry bounding box
        # (reasonable approximation for visual reference; at state/country scale
        # there is some distortion due to projection, but it works for locating the point).

        # Get the bounding box coordinates of the administrative region
        bounds_coords = geom.bounds().getInfo()["coordinates"][0]

        # Extract min/max longitude and latitude from the bounding box
        lons = [c[0] for c in bounds_coords]
        lats = [c[1] for c in bounds_coords]
        west, east = min(lons), max(lons)
        south, north = min(lats), max(lats)

        # Convert geographic coordinates to pixel coordinates on the canvas
        # Uses linear interpolation within the bounding box (approximate)
        px = (lon - west) / (east - west) * canvas.width
        py = (north - lat) / (north - south) * canvas.height  # Y is inverted (north is top)

        # Draw a red circular marker at the interpolated position
        draw_marker(canvas, px, py)

    # Save the final image to disk
    canvas.save(out_path)
    print(f"Image saved to {out_path} ({canvas.width}x{canvas.height}px)")
    return out_path


if __name__ == "__main__":

    # Reference points for quick access (commented out):
    # El Playon         --||     7.4584221918243045,    -73.222052853104
    # Finca Matanza     --||     7.300921,              -73.009794
    # Sugarcane_COL     --||     3.580109040361371,     -76.31299479308868
    # Sugarcane_QLD     --||     -19.689669877950884,   147.22717515914223

    # Test point: El Playon, Colombia
    lat_test = 7.4584221918243045
    lon_test = -73.222052853104

    # Download elevation map for the state/department containing the test point
    download_regional_elevation_map(lat_test, lon_test, region_type="state")
