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


def meters_to_pixels(distance_m, lat, zoom, tile_size=256):
    """
    Convierte una distancia en metros a su equivalente en pixeles, en el
    zoom dado. Depende de la latitud porque Web Mercator distorsiona la
    escala real a medida que te alejas del ecuador (a mayor |latitud|,
    cada pixel representa MENOS metros reales) -> por eso este cálculo
    no es un factor fijo, cambia según dónde esté tu punto.
    """
    meters_per_pixel = (156543.03392 * math.cos(math.radians(lat))) / (2 ** zoom)
    return distance_m / meters_per_pixel


def draw_marker(canvas, x, y, radius=4, color=(255, 40, 40)):
    """Dibuja un marcador circular con contorno blanco en la posición (x, y) del canvas."""
    draw = ImageDraw.Draw(canvas)
    draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=color, outline=(255, 255, 255), width=2
    )


def draw_area_box(canvas, x, y, area_meters, lat, zoom, tile_size=256,
                   color=(255, 215, 0), width=4):
    """
    Dibuja el cuadro que representa el área usada en reduceRegion
    (terrain_profile.py, soil_profile.py): un buffer de área_meters de
    radio -> .bounds() -> cuadrado de lado 2*area_meters, centrado en
    el punto (x, y) en pixeles del canvas.

    x, y: posición en pixeles del punto central (la misma que usás para el marker)
    area_meters: el mismo valor que le pasaste a get_terrain_profile_area /
                 get_soil_profile_area (el "radio" del buffer, no el lado del cuadro)
    lat: latitud del punto (necesaria para la conversión metros->pixeles)
    """
    half_side_px = meters_to_pixels(area_meters, lat, zoom, tile_size)

    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [(x - half_side_px, y - half_side_px), (x + half_side_px, y + half_side_px)],
        outline=color, width=width
    )