"""
Terrain Profile Module
=======================

Obtains average elevation, slope, and aspect for an area around a point,
using the Copernicus DEM via Google Earth Engine.

HOW TO READ THE RESULT
-----------------------
elevation_m       Average elevation of the area, in meters above sea level.
elevation_std_m   How much elevation varies within the area. A high value
                   indicates terrain with marked elevation differences
                   (not everything at the same height).

slope_deg         Average slope, in degrees (0 deg = flat, 90 deg = vertical).
                   As a general reference (not a strict rule, always check
                   specific agronomic criteria for your crop):
                     0-8 deg   : flat to gentle, easy to manage
                     8-15 deg  : moderate, may require conservation practices
                     15-25 deg : steep, usually needs contour lines/terraces
                     >25 deg   : very steep, more difficult/costly management

slope_std_deg     How much slope varies within the area. High = irregular
                   terrain (mix of flat and steep zones); low = consistent
                   slope throughout the area (the average represents it well).

aspect_deg        Direction the slope "faces", in compass degrees
                   (0 deg=North, 90 deg=East, 180 deg=South, 270 deg=West).
                   Relevant for sun exposure: in the northern hemisphere,
                   south/southwest-facing slopes receive more direct sun.
"""
import math
from datetime import datetime
from pathlib import Path

import ee
import pandas as pd

# Google Cloud project linked to Earth Engine. Required since Earth Engine
# now requires every ee.Initialize() call to be associated with a project.
EE_PROJECT = "famous-strategy-376313"

try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

# Data source label used in the download/acquire progress messages.
SOURCE_NAME = "Copernicus DEM"
ANALYSIS_VARIABLE = "terrain"


# Available Digital Elevation Model sources
DEM_SOURCES = {
    "copernicus": "COPERNICUS/DEM/GLO30_2024_1",  # 2024 release, recommended (30m resolution)
    "srtm": "USGS/SRTMGL1_003",                    # Classic SRTM, may have voids in steep terrain
}


def get_terrain_profile_area(lat, lon, area_meters=56, dem_source="copernicus", verbose=False):
    """
    Calculates average elevation, slope, and aspect over an area around
    a point. area_meters is the buffer radius; the actual evaluated area
    is a square with side length 2*area_meters.

    Args:
        lat: Latitude of the center point in degrees
        lon: Longitude of the center point in degrees
        area_meters: Buffer radius in meters; the actual extraction region
                     is a square bounding box around this circular buffer
        dem_source: "copernicus" (default, recommended) or "srtm"
        verbose: If True, prints the raw statistics returned by GEE (for debugging)

    Returns:
        dict: Terrain statistics including elevation, slope, aspect, and their
              standard deviations

    Raises:
        ValueError: If dem_source is not one of the available DEM sources
    """
    print(f"Downloading satellite information {SOURCE_NAME} to analyze "
          f"{ANALYSIS_VARIABLE}...")

    # Validate the requested DEM source
    if dem_source not in DEM_SOURCES:
        raise ValueError(f"dem_source must be one of {list(DEM_SOURCES.keys())}")

    # Create Earth Engine point geometry and buffer region
    center = ee.Geometry.Point([lon, lat])
    region = center.buffer(area_meters).bounds()

    # Load the appropriate DEM dataset
    if dem_source == "copernicus":
        # COPERNICUS/DEM/GLO30 is an ImageCollection of tiles -> must be mosaicked
        # .mosaic() composites all tiles into a single image
        dem = ee.ImageCollection(DEM_SOURCES[dem_source]).select('DEM').mosaic()
    else:
        # SRTM is a single global Image, load it directly
        dem = ee.Image(DEM_SOURCES[dem_source]).select('elevation')

    # Rename the elevation band for consistent reference
    elevation = dem.rename('elevation')

    # IMPORTANT: .mosaic() does not preserve a well-defined projection/pixel scale
    # -> ee.Terrain.slope()/aspect() need an explicit grid to calculate the
    # gradient correctly. Without this, they end up using a default scale
    # much coarser than 30m.
    # Reproject to EPSG:4326 (WGS84) at 30m resolution to match the native DEM
    elevation_for_terrain = elevation.reproject(crs='EPSG:4326', scale=30)

    # Calculate slope in degrees from the reprojected elevation
    slope = ee.Terrain.slope(elevation_for_terrain).rename('slope')

    # Calculate aspect in degrees (0 deg = North, clockwise)
    aspect_deg = ee.Terrain.aspect(elevation_for_terrain)

    # Aspect is circular data (0-360 deg) -> average via sine/cosine components,
    # not with a direct arithmetic mean (that would give incorrect results
    # when the area crosses north, e.g., values near 0 deg and 360 deg).
    # Convert to radians and compute sine and cosine components
    aspect_rad = aspect_deg.multiply(math.pi / 180)
    aspect_sin = aspect_rad.sin().rename('aspect_sin')
    aspect_cos = aspect_rad.cos().rename('aspect_cos')

    # Combine all bands into a single multi-band image for reduction
    combined = elevation.addBands(slope).addBands(aspect_sin).addBands(aspect_cos)

    # Create a combined reducer that computes mean, standard deviation, and count
    # sharedInputs=True applies all reducers to the same input bands
    combined_reducer = (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.count(), sharedInputs=True)
    )

    # Reduce the region: compute statistics for all bands within the area
    stats = combined.reduceRegion(
        reducer=combined_reducer,
        geometry=region,
        scale=30,        # 30m resolution matches the DEM native resolution
        bestEffort=True  # Allows EE to handle large/complex regions gracefully
    ).getInfo()

    if verbose:
        print(f"[DEBUG] Raw stats: {stats}")

    # Reconstruct the mean aspect angle from its sine and cosine components
    # atan2(sin_mean, cos_mean) gives the circular mean direction
    mean_aspect_rad = math.atan2(stats['aspect_sin_mean'], stats['aspect_cos_mean'])
    mean_aspect_deg = math.degrees(mean_aspect_rad)
    # Normalize to 0-360 deg range (atan2 returns -pi to pi)
    if mean_aspect_deg < 0:
        mean_aspect_deg += 360

    # Compile all terrain statistics into a results dictionary
    results = {
        # 'lat': lat,   # Uncomment to include coordinates in the output
        # 'lon': lon,   # Uncomment to include coordinates in the output
        'dem_source': dem_source,
        'elevation_m': stats['elevation_mean'],
        'elevation_std_m': stats['elevation_stdDev'],
        'slope_deg': stats['slope_mean'],
        'slope_std_deg': stats['slope_stdDev'],
        'aspect_deg': mean_aspect_deg,
        # Note: we don't calculate a standard deviation for aspect because
        # it's circular data (would require circular variance instead of
        # regular stdDev).
    }

    print(f"Satellite information {SOURCE_NAME} acquired.")
    return results


def save_terrain_profile(terrain_data, out_prefix="terrain_profile_data", output_dir="../../databases"):
    """
    Saves the terrain profile with a timestamp in the filename:
    {out_prefix}-vYYMMDDHHMMSS.csv

    Args:
        terrain_data: Dictionary of terrain statistics from get_terrain_profile_area()
        out_prefix: Base name for the output CSV file
        output_dir: Directory where the CSV is saved (created if it doesn't exist)

    Returns:
        tuple: (output_path as Path object, pandas DataFrame of the saved data)
    """
    # Convert the single dictionary to a one-row DataFrame
    df = pd.DataFrame([terrain_data])

    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    filename = f"{out_prefix}-v{timestamp}.csv"

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_path = output_path / filename

    # Save to CSV without the default pandas index
    df.to_csv(out_path, index=False)
    print(f"CSV saved to {out_path} ({df.shape[0]}x{df.shape[1]})")
    return out_path, df


if __name__ == "__main__":
    # Reference points for quick access (commented out):
    # El Playon         --||     7.4584221918243045,    -73.222052853104
    # Finca Matanza     --||     7.300921,              -73.009794
    # Sugarcane_COL     --||     3.580109040361371,     -76.31299479308868
    # Sugarcane_QLD     --||     -19.689669877950884,   147.22717515914223

    # Calculate the buffer radius from hectares
    # Formula: radius = sqrt(hectares * 10000) / 2
    # For 2 hectares: sqrt(20000) / 2 ~= 70.71 / 2 ~= 35.36 meters radius
    ha = 2
    area_meters = math.sqrt(ha * 10000) / 2

    # Test point: El Playon, Colombia
    Latitude = 7.4584221918243045
    Longitude = -73.222052853104

    # Extract terrain statistics using Copernicus DEM
    terrain_data = get_terrain_profile_area(Latitude, Longitude, area_meters, dem_source="copernicus")

    # Save the terrain profile to CSV and display it
    out_path, df = save_terrain_profile(terrain_data)
    print(df)
