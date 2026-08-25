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
    >>> from point_map_refactored import get_point_map
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

from tile_utils import (
    DEFAULT_TILE_SIZE,
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

# User-Agent for HTTP requests to tile servers.
# Replace with your real email for responsible API usage.
USER_AGENT = "agri_land_suitability_pipeline (your_real_email@domain.com)"

# Seconds to wait for a tile server to respond before giving up.
TILE_REQUEST_TIMEOUT = 10

# Provider configuration: URL template, typical max zoom, and recommended
# pause between requests (each service has its own usage policy).
PROVIDERS = {
    "opentopomap": {
        "url_template": "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        "max_zoom": 17,
        "sleep": 0.5,  # Seconds to wait between tile requests (rate limiting)
    },
    "esri": {
        # NOTE: ESRI uses {z}/{y}/{x} ordering, reversed from the standard {z}/{x}/{y}.
        # This is a common gotcha when switching between tile providers.
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
    validate_coordinates(lat, lon)
    _validate_arguments(provider, area_meters, radius)

    config = PROVIDERS[provider]
    url_template = config["url_template"]
    max_zoom = config["max_zoom"]
    sleep_seconds = config["sleep"]

    zoom = max_zoom if zoom is None else zoom
    _warn_if_zoom_exceeds_max(provider, zoom, max_zoom)

    output_file = _build_output_path(out_prefix or f"point_{provider}", output_dir)

    # Tile indices of the center point at the given zoom level.
    tile_x_center, tile_y_center = latlon_to_tile(lat, lon, zoom)

    # Bounding box of tiles to download (radius tiles on every side).
    tile_x_min, tile_x_max = tile_x_center - radius, tile_x_center + radius
    tile_y_min, tile_y_max = tile_y_center - radius, tile_y_center + radius

    # Number of tiles per axis; +1 because the range is inclusive.
    tiles_across = tile_x_max - tile_x_min + 1
    tiles_down = tile_y_max - tile_y_min + 1

    canvas = Image.new(
        "RGB",
        (tiles_across * DEFAULT_TILE_SIZE, tiles_down * DEFAULT_TILE_SIZE),
    )

    failed_tiles = _stitch_tiles(
        canvas,
        url_template,
        zoom,
        tile_x_min,
        tile_x_max,
        tile_y_min,
        tile_y_max,
        sleep_seconds,
    )

    if failed_tiles:
        total_tiles = tiles_across * tiles_down
        print(f"[WARNING] {failed_tiles}/{total_tiles} tiles failed and were left blank.")

    if show_marker:
        _draw_annotations(canvas, lat, lon, zoom, area_meters, tile_x_min, tile_y_min)

    canvas.save(output_file)
    print(f"Image saved to {output_file} ({canvas.width}x{canvas.height}px)")
    return output_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_arguments(provider: str, area_meters: float, radius: int) -> None:
    """Raises ValueError for any invalid non-coordinate argument."""
    if provider not in PROVIDERS:
        raise ValueError(
            f"Provider '{provider}' not recognized. "
            f"Available options: {list(PROVIDERS.keys())}"
        )
    if area_meters <= 0:
        raise ValueError(f"area_meters must be positive. Got: {area_meters}")
    if radius < 0:
        raise ValueError(f"radius must be non-negative. Got: {radius}")


def _warn_if_zoom_exceeds_max(provider: str, zoom: int, max_zoom: int) -> None:
    """Warns when the requested zoom exceeds the provider's recommended max."""
    if zoom > max_zoom:
        print(
            f"[WARNING] zoom={zoom} exceeds the recommended max_zoom for "
            f"'{provider}' ({max_zoom}); the provider may return blank or "
            f"lower-quality tiles at that level."
        )


def _build_output_path(prefix: str, output_dir: str) -> Path:
    """Creates the output directory and returns a timestamped output file path."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    return output_dir_path / f"{prefix}-v{timestamp}.png"


def _stitch_tiles(
    canvas: Image.Image,
    url_template: str,
    zoom: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    sleep_seconds: float,
) -> int:
    """
    Downloads every tile in the bounding box and pastes it onto the canvas.

    Returns the number of tiles that could not be downloaded or decoded.
    Respects the provider's rate limiting with a pause after each request.
    """
    headers = {"User-Agent": USER_AGENT}
    failed = 0

    for tile_x in range(x_min, x_max + 1):
        for tile_y in range(y_min, y_max + 1):
            url = url_template.format(z=zoom, x=tile_x, y=tile_y)
            response = requests.get(url, headers=headers, timeout=TILE_REQUEST_TIMEOUT)

            if response.status_code != 200:
                print(f"Tile {tile_x},{tile_y} failed with status {response.status_code}")
                failed += 1
            else:
                try:
                    tile = Image.open(BytesIO(response.content))
                    canvas.paste(
                        tile,
                        (
                            (tile_x - x_min) * DEFAULT_TILE_SIZE,
                            (tile_y - y_min) * DEFAULT_TILE_SIZE,
                        ),
                    )
                except Exception as exc:
                    print(f"Tile {tile_x},{tile_y} failed: {exc}")
                    failed += 1

            time.sleep(sleep_seconds)

    return failed


def _draw_annotations(
    canvas: Image.Image,
    lat: float,
    lon: float,
    zoom: int,
    area_meters: float,
    tile_x_min: int,
    tile_y_min: int,
) -> None:
    """Draws the point marker and area-of-interest box on the canvas."""
    # Absolute world pixel coordinates for the point.
    px_world, py_world = latlon_to_pixel(lat, lon, zoom, DEFAULT_TILE_SIZE)

    # Convert to canvas-local pixels by removing the top-left tile offset.
    px_canvas = px_world - tile_x_min * DEFAULT_TILE_SIZE
    py_canvas = py_world - tile_y_min * DEFAULT_TILE_SIZE

    draw_marker(canvas, px_canvas, py_canvas)
    draw_area_box(canvas, px_canvas, py_canvas, area_meters, lat=lat, zoom=zoom)


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
