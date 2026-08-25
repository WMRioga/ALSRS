# tile_utils.py
"""
Shared coordinate conversion, drawing utility functions, and geographic validators,
used by any script that works with Web Mercator tiles
(OpenTopoMap, ESRI World Imagery, or any other XYZ tile provider).
"""

import math
from PIL import ImageDraw


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

def latlon_to_tile(lat, lon, zoom):
    """
    Converts latitude/longitude to the tile number (x, y) that contains it,
    at the given zoom level.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        zoom: Zoom level (integer)
    
    Returns:
        tuple: (tile_x, tile_y) - the tile coordinates containing the point
    """
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def latlon_to_pixel(lat, lon, zoom, tile_size=256):
    """
    Converts latitude/longitude to ABSOLUTE pixel coordinates (not just tile)
    at the given zoom level.
    Allows locating the exact point within the canvas, not just the tile.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        zoom: Zoom level (integer)
        tile_size: Size of each tile in pixels (default 256 for standard tiles)
    
    Returns:
        tuple: (x_pixel, y_pixel) - absolute pixel coordinates at the given zoom level
    """
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x_pixel = (lon + 180.0) / 360.0 * n * tile_size
    y_pixel = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n * tile_size
    return x_pixel, y_pixel


def meters_to_pixels(distance_m, lat, zoom, tile_size=256):
    """
    Converts a distance in meters to its equivalent in pixels, at the
    given zoom level.
    
    Depends on latitude because Web Mercator distorts the real scale
    as you move away from the equator (the higher the absolute latitude,
    each pixel represents FEWER real meters) -> therefore this calculation
    is not a fixed factor, it changes according to where your point is located.
    
    Args:
        distance_m: Distance in meters to convert
        lat: Latitude of the reference point in degrees
        zoom: Zoom level (integer)
        tile_size: Size of each tile in pixels (default 256)
    
    Returns:
        float: Equivalent distance in pixels
    """
    meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2 ** zoom)
    return distance_m / meters_per_pixel


# ---------------------------------------------------------------------------
# Drawing Functions
# ---------------------------------------------------------------------------

def draw_marker(canvas, x, y, radius=4, color=(255, 40, 40)):
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


def draw_area_box(canvas, x, y, area_meters, lat, zoom, tile_size=256,
                   color=(255, 215, 0), width=4):
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