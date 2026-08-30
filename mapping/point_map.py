"""
Point Map Visualization Module
==============================

Generates composite map images around geographic points by downloading
and stitching Web Mercator tiles from various tile providers.

This module is part of the agricultural land suitability pipeline and
provides visualization capabilities for point locations with optional
markers and area-of-interest bounding boxes. It is meant to be run
standalone (see __main__ below) or imported and called from the future
orchestrator alongside the other pipeline modules.

Supported tile providers:
    - ESRI World Imagery (satellite)
    - OpenTopoMap (topographic)

Example:
    >>> from point_map import get_point_map
    >>> from tile_utils import hectares_to_radius_meters
    >>> # Generate satellite map for 2 hectares around a point
    >>> area_meters = hectares_to_radius_meters(2)
    >>> image_path = get_point_map(
    ...     lat=7.4584,
    ...     lon=-73.2221,
    ...     area_meters=area_meters,
    ...     provider="esri"
    ... )

Dependencies:
    - requests
    - PIL (Pillow)
    - tile_utils (custom module)
"""

import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

# Make the local modules importable regardless of the working directory.
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "common", _PROJECT_ROOT / "extraction",
           _PROJECT_ROOT / "analysis", _PROJECT_ROOT / "mapping"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tile_utils import (
    draw_area_box,
    draw_marker,
    hectares_to_radius_meters,
    latlon_to_pixel,
    latlon_to_tile,
    validate_coordinates,
)


# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

# User-Agent for HTTP requests to tile servers
# Replace with your real email for responsible API usage
USER_AGENT = "agri_land_suitability_pipeline (your_real_email@domain.com)"

# Provider configuration: URL template, typical x/y ordering, typical max zoom,
# and recommended pause between requests (each service has its own usage policy).
PROVIDERS = {
    "opentopomap": {
        "url_template": "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        "max_zoom": 17,
        "sleep": 0.5,  # Seconds to wait between tile requests (rate limiting)
    },
    "esri": {
        # NOTE: ESRI uses {z}/{y}/{x} ordering, reversed from the standard {z}/{x}/{y}
        # This is a common gotcha when switching between tile providers
        "url_template": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "max_zoom": 17,
        "sleep": 0.2,  # Seconds to wait between tile requests (rate limiting)
    },
}


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

def get_point_map(
    lat: float,
    lon: float,
    area_meters: float = 56,
    provider: str = "esri",
    zoom: Optional[int] = None,
    radius: int = 2,
    out_prefix: Optional[str] = None,
    output_dir: str = "../../img/maps",
    show_marker: bool = True,
) -> Path:
    """
    Downloads map tiles around a point and composites them into a single image,
    optionally marking the exact location and area of interest.

    Args:
        lat: Latitude of the center point in decimal degrees
        lon: Longitude of the center point in decimal degrees
        area_meters: Buffer radius in meters for drawing the area box
                     (same value used in reduceRegion calculations across
                     terrain_profile.py / soil_profile.py)
        provider: Tile provider name ("esri" or "opentopomap")
        zoom: Zoom level; if None, uses the provider's max_zoom
        radius: Number of tiles to add around center in each direction
                (e.g., radius=2 gives a 5x5 tile grid)
        out_prefix: Base name for output file; defaults to provider name
        output_dir: Directory where the image is saved
        show_marker: If True, draws location marker and area box

    Returns:
        pathlib.Path: Path to the saved image file

    Raises:
        ValueError: If provider is not recognized, coordinates are invalid,
                    or area_meters/radius are non-positive/negative

    Example:
        >>> # Generate satellite map for a 2-hectare area
        >>> area_m = hectares_to_radius_meters(2)
        >>> img_path = get_point_map(
        ...     lat=7.4584,
        ...     lon=-73.2221,
        ...     area_meters=area_m,
        ...     provider="esri",
        ...     radius=2
        ... )
        >>> print(img_path)
        ../../img/maps/point_esri-v240815143022.png
    """
    # Validate inputs before proceeding
    validate_coordinates(lat, lon)

    if provider not in PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' not recognized. "
            f"Available options: {list(PROVIDERS.keys())}"
        )

    if area_meters <= 0:
        raise ValueError(f"area_meters must be positive. Got: {area_meters}")

    if radius < 0:
        raise ValueError(f"radius must be non-negative. Got: {radius}")

    # Load provider-specific configuration
    config = PROVIDERS[provider]
    zoom = zoom or config["max_zoom"]  # Use max zoom if not explicitly provided
    if zoom > config["max_zoom"]:
        print(
            f"[WARNING] zoom={zoom} exceeds the recommended max_zoom for "
            f"'{provider}' ({config['max_zoom']}); the provider may return blank or "
            f"lower-quality tiles at that level."
        )
        
    out_prefix = out_prefix or f"point_{provider}"  # Default filename prefix

    # Generate a unique filename with timestamp to avoid overwriting previous maps
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    filename = f"{out_prefix}-v{timestamp}.png"

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_path = output_path / filename

    # Calculate the tile coordinates for the center point
    # These are integer tile indices at the given zoom level
    x_center, y_center = latlon_to_tile(lat, lon, zoom)

    # Define the bounding box of tiles to download
    x_min, x_max = x_center - radius, x_center + radius
    y_min, y_max = y_center - radius, y_center + radius

    # Standard tile size for Web Mercator tiles (256x256 pixels)
    tile_size = 256

    # Calculate the total canvas dimensions in pixels
    # +1 because range is inclusive of both min and max
    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size

    # Create a blank RGB canvas to composite all tiles onto
    canvas = Image.new("RGB", (width, height))

    # Set HTTP headers with User-Agent for responsible API usage
    headers = {"User-Agent": USER_AGENT}

    # Download and stitch together all tiles in the bounding box
    failed_tiles = 0
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            # Build the tile URL using the provider's template
            url = config["url_template"].format(z=zoom, x=x, y=y)

            # Fetch the tile image with a timeout to avoid hanging
            resp = requests.get(url, headers=headers, timeout=10)

            # Skip this tile if the server returns an error
            if resp.status_code != 200:
                print(f"Tile {x},{y} failed with status {resp.status_code}")
                failed_tiles += 1
                time.sleep(config["sleep"])
                continue

            try:
                # Open the downloaded image from memory (avoids writing temp files)
                tile_img = Image.open(BytesIO(resp.content))

                # Paste the tile onto the canvas at the correct pixel offset
                # (x - x_min) and (y - y_min) convert tile coordinates to canvas pixel coordinates
                canvas.paste(
                    tile_img,
                    ((x - x_min) * tile_size, (y - y_min) * tile_size)
                )
            except Exception as e:
                print(f"Tile {x},{y} failed: {e}")
                failed_tiles += 1

            # Respect the provider's rate limiting to avoid being blocked
            time.sleep(config["sleep"])

    total_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
    if failed_tiles > 0:
        print(f"[WARNING] {failed_tiles}/{total_tiles} tiles failed and were left blank.")

    # Draw the point marker and area box on the composited canvas
    if show_marker:
        # Get the absolute world pixel coordinates for the given lat/lon
        px_world, py_world = latlon_to_pixel(lat, lon, zoom, tile_size)

        # Convert world pixel coordinates to canvas-local pixel coordinates
        # by subtracting the offset of the top-left tile in our grid
        px_canvas = px_world - x_min * tile_size
        py_canvas = py_world - y_min * tile_size

        # Draw a red circular marker at the exact point location
        draw_marker(canvas, px_canvas, py_canvas)

        # Draw a box representing the area used in reduceRegion calculations
        # This shows the actual spatial extent of the analysis
        draw_area_box(
            canvas,
            px_canvas,
            py_canvas,
            area_meters,
            lat=lat,
            zoom=zoom
        )

    # Save the final composited image to disk
    canvas.save(out_path)
    print(f"Image saved to {out_path} ({width}x{height}px)")
    return out_path


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: Generate maps for a 2-hectare area in Colombia

    # Convert hectares to buffer radius in meters (shared helper, same
    # formula used across terrain/soil profile scripts)
    ha = 2
    area_meters = hectares_to_radius_meters(ha)

    # Test point coordinates (El Playón, Colombia)
    latitude = 7.4584221918243045
    longitude = -73.222052853104

    # Generate satellite imagery map (ESRI World Imagery)
    satellite_map = get_point_map(
        lat=latitude,
        lon=longitude,
        area_meters=area_meters,
        provider="esri",
        radius=2,
    )

    # Generate topographic/elevation map (OpenTopoMap)
    topo_map = get_point_map(
        lat=latitude,
        lon=longitude,
        area_meters=area_meters,
        provider="opentopomap",
        radius=2,
    )

    print("\nGenerated maps:")
    print(f"  Satellite: {satellite_map}")
    print(f"  Topographic: {topo_map}")


# ---------------------------------------------------------------------------
# Reference Points
# ---------------------------------------------------------------------------

# Commonly used reference points for testing and documentation:
# El Playon         : 7.4584221918243045, -73.222052853104
# Finca Matanza     : 7.300921,          -73.009794
# Sugarcane_COL     : 3.580109040361371, -76.31299479308868
# Sugarcane_QLD     : -19.689669877950884, 147.22717515914223
