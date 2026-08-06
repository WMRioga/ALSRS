"""
Funciones compartidas de conversión de coordenadas y dibujo,
usadas por cualquier script que trabaje con tiles tipo Web Mercator
(OpenTopoMap, ESRI World Imagery, o cualquier otro proveedor XYZ).
"""
import math
from PIL import ImageDraw


def latlon_to_tile(lat, lon, zoom):
    """Convierte lat/lon al número de tile (x, y) que lo contiene, en el zoom dado."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def latlon_to_pixel(lat, lon, zoom, tile_size=256):
    """Convierte lat/lon a coordenadas de PIXEL absolutas (no solo tile) en el zoom dado.
    Permite ubicar el punto exacto dentro del canvas, no solo el tile."""
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x_pixel = (lon + 180.0) / 360.0 * n * tile_size
    y_pixel = (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n * tile_size
    return x_pixel, y_pixel


def draw_marker(canvas, x, y, radius=8, color=(255, 40, 40)):
    """Dibuja un marcador circular con contorno blanco en la posición (x, y) del canvas."""
    draw = ImageDraw.Draw(canvas)
    draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color, outline=(255, 255, 255), width=3
    )