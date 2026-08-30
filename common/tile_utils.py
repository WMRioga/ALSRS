"""
Shared coordinate conversion, drawing utility functions, and geographic
validators, used by any script that works with Web Mercator tiles
(OpenTopoMap, ESRI World Imagery, or any other XYZ tile provider).

This module is the single source of truth for the Web Mercator projection
math used across the pipeline (point_map.py, regional_elevation_map.py).
"""

import math
from typing import Tuple

from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Web Mercator is mathematically undefined beyond this latitude (tan(lat)
# diverges as lat -> 90). This is the same clamp used by Google Maps / OSM.
# None of the pipeline's test points (Colombia, Queensland) are anywhere
# near this limit, but clamping avoids a silent OverflowError if the module
# is ever reused for a point closer to the poles.
MAX_MERCATOR_LAT = 85.05112878

# Standard tile size for Web Mercator XYZ tiles (ESRI, OpenTopoMap, OSM, etc.)
DEFAULT_TILE_SIZE = 256


# ---------------------------------------------------------------------------
# Geographic Validation
# ---------------------------------------------------------------------------

def validate_coordinates(lat: float, lon: float) -> None:
    """
    Validates that geographic coordinates are within valid ranges.

    Latitude must be between -90 and 90 degrees.
    Longitude must be between -180 and 180 degrees.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees

    Raises:
        ValueError: If coordinates are outside valid ranges

    Example:
        >>> validate_coordinates(7.4584, -73.2221)  # Valid, no error
        >>> validate_coordinates(95.5, -73.2221)    # Raises ValueError
    """
    if not -90 <= lat <= 90:
        raise ValueError(
            f"Latitude {lat} is out of range. Must be between -90 and 90 degrees."
        )

    if not -180 <= lon <= 180:
        raise ValueError(
            f"Longitude {lon} is out of range. Must be between -180 and 180 degrees."
        )


def hectares_to_radius_meters(hectares: float) -> float:
    """
    Converts hectares to equivalent buffer radius in meters.

    Calculates the side length of a square with the given area and
    returns half of that value, which represents the buffer radius
    around a central point.

    Args:
        hectares: Area in hectares (1 ha = 10,000 m²)

    Returns:
        Buffer radius in meters

    Raises:
        ValueError: If hectares is negative or zero

    Example:
        >>> hectares_to_radius_meters(2)
        70.71067811865476  # sqrt(20000) / 2
    """
    if hectares <= 0:
        raise ValueError(f"Hectares must be positive. Got: {hectares}")

    area_m2 = hectares * 10000  # Convert hectares to square meters
    side_length = math.sqrt(area_m2)  # Side length of equivalent square
    return side_length / 2  # Radius is half the side length


# ---------------------------------------------------------------------------
# Coordinate Conversion
# ---------------------------------------------------------------------------

def latlon_to_pixel(
    lat: float, lon: float, zoom: int, tile_size: int = DEFAULT_TILE_SIZE
) -> Tuple[float, float]:
    """
    Converts latitude/longitude to ABSOLUTE pixel coordinates (not just tile)
    at the given zoom level.

    This is the canonical Web Mercator projection used by the whole module;
    latlon_to_tile() derives from this function rather than reimplementing
    the same math, so both always stay in sync.

    Args:
        lat: Latitude in degrees (clamped internally to ±85.0511° — see
             MAX_MERCATOR_LAT — since Web Mercator is undefined beyond that)
        lon: Longitude in degrees
        zoom: Zoom level (integer)
        tile_size: Size of each tile in pixels (default 256 for standard tiles)

    Returns:
        tuple: (x_pixel, y_pixel) - absolute pixel coordinates at the given zoom level
    """
    lat_clamped = max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)
    lat_rad = math.radians(lat_clamped)
    n = 2 ** zoom
    x_pixel = (lon + 180.0) / 360.0 * n * tile_size
    y_pixel = (
        (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
        * tile_size
    )
    return x_pixel, y_pixel


def latlon_to_tile(
    lat: float, lon: float, zoom: int, tile_size: int = DEFAULT_TILE_SIZE
) -> Tuple[int, int]:
    """
    Converts latitude/longitude to the integer tile number (x, y) that
    contains it, at the given zoom level.

    Derived from latlon_to_pixel() (single source of truth for the
    projection math) instead of reimplementing it, so a future change
    to the projection only needs to happen in one place.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        zoom: Zoom level (integer)
        tile_size: Size of each tile in pixels (default 256)

    Returns:
        tuple: (tile_x, tile_y) - the tile coordinates containing the point
    """
    x_px, y_px = latlon_to_pixel(lat, lon, zoom, tile_size)
    return int(x_px // tile_size), int(y_px // tile_size)


def meters_to_pixels(
    distance_m: float, lat: float, zoom: int, tile_size: int = DEFAULT_TILE_SIZE
) -> float:
    """
    Converts a distance in meters to its equivalent in pixels, at the
    given zoom level.

    Depends on latitude because Web Mercator distorts the real scale
    as you move away from the equator (the higher the absolute latitude,
    each pixel represents FEWER real meters) -> therefore this calculation
    is not a fixed factor, it changes according to where your point is located.

    Args:
        distance_m: Distance in meters to convert
        lat: Latitude of the reference point in degrees (clamped to ±85.0511°)
        zoom: Zoom level (integer)
        tile_size: Size of each tile in pixels (default 256)

    Returns:
        float: Equivalent distance in pixels
    """
    lat_clamped = max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)
    meters_per_pixel = (156543.03392 * math.cos(math.radians(lat_clamped))) / (2 ** zoom)
    return distance_m / meters_per_pixel


# ---------------------------------------------------------------------------
# Drawing Functions
# ---------------------------------------------------------------------------

def draw_marker(
    canvas: Image.Image,
    x: float,
    y: float,
    radius: int = 4,
    color: Tuple[int, int, int] = (255, 40, 40),
) -> None:
    """
    Draws a circular marker with white outline at position (x, y) on the canvas.

    Args:
        canvas: PIL Image object to draw on
        x: X coordinate in pixels (center of the marker)
        y: Y coordinate in pixels (center of the marker)
        radius: Radius of the marker circle in pixels (default 4)
        color: RGB tuple for the fill color (default red)
    """
    draw = ImageDraw.Draw(canvas)
    draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color, outline=(255, 255, 255), width=2
    )


def draw_area_box(
    canvas: Image.Image,
    x: float,
    y: float,
    area_meters: float,
    lat: float,
    zoom: int,
    tile_size: int = DEFAULT_TILE_SIZE,
    color: Tuple[int, int, int] = (255, 215, 0),
    width: int = 4,
) -> None:
    """
    Draws the box representing the area used in reduceRegion
    (terrain_profile.py, soil_profile.py): a buffer of area_meters radius
    -> .bounds() -> square with side length 2*area_meters, centered at
    the point (x, y) in canvas pixels.

    Args:
        canvas: PIL Image object to draw on
        x: X coordinate of the center point in pixels
        y: Y coordinate of the center point in pixels
        area_meters: Buffer radius in meters (half the square side)
        lat: Latitude of the point in degrees (required for accurate meter-to-pixel conversion)
        zoom: Zoom level (integer)
        tile_size: Size of each tile in pixels (default 256)
        color: RGB tuple for the box outline color (default gold/yellow)
        width: Width of the box outline in pixels (default 4)
    """
    # Convert the buffer radius from meters to pixels at the given latitude and zoom
    half_side_px = meters_to_pixels(area_meters, lat, zoom, tile_size)

    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [(x - half_side_px, y - half_side_px), (x + half_side_px, y + half_side_px)],
        outline=color, width=width
    )
