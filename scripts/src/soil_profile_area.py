"""
Soil Profile Area Module
========================

Extracts soil properties (SoilGrids-ISRIC) for an area around a point and
calculates its hydraulic properties (field capacity, wilting point, and
available water capacity), using the Saxton & Rawls (2006) pedotransfer
functions implemented in soil_hydraulics.py.

This module is the Python equivalent of the Jupyter notebook
``MDS650_v260804_soil_profile_area.ipynb``. It is meant to be run
standalone (see __main__ below) or imported alongside the other pipeline
modules.

Dependencies:
    - earthengine-api (ee)
    - pandas
    - soil_hydraulics (custom module, same directory)
    - tile_utils (custom module, same directory)
"""
from datetime import datetime
from pathlib import Path
from typing import Tuple

import ee
import pandas as pd

from soil_hydraulics import DEPTHS, calculate_hydraulic_properties
from tile_utils import hectares_to_radius_meters

# Google Cloud project linked to Earth Engine. Required since Earth Engine
# now requires every ee.Initialize() call to be associated with a project.
EE_PROJECT = "famous-strategy-376313"

try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

# Data source label used in the download/acquire progress messages.
SOURCE_NAME = "SoilGrids (ISRIC)"
ANALYSIS_VARIABLE = "soil properties"


# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

# Mapping of soil property short names to their full SoilGrids collection IDs
# These are the ISRIC SoilGrids 2.0 datasets available in Earth Engine.
SOILGRIDS_PROPERTIES = {
    'phh2o': 'projects/soilgrids-isric/phh2o_mean',  # Soil pH in water
    'soc': 'projects/soilgrids-isric/soc_mean',      # Soil organic carbon content
    'clay': 'projects/soilgrids-isric/clay_mean',    # Clay fraction
    'sand': 'projects/soilgrids-isric/sand_mean',    # Sand fraction
    'silt': 'projects/soilgrids-isric/silt_mean',    # Silt fraction
    'bdod': 'projects/soilgrids-isric/bdod_mean',    # Bulk density of fine earth
    'cec': 'projects/soilgrids-isric/cec_mean',      # Cation exchange capacity
}


# ---------------------------------------------------------------------------
# Extraction Function
# ---------------------------------------------------------------------------

def get_soil_profile_area(lat: float, lon: float, area_meters: float = 56) -> dict:
    """
    Extracts soil properties (phh2o, soc, clay, sand, silt, bdod, cec)
    for 5 standard depth intervals, averaged over an area around the point.

    Uses SoilGrids-ISRIC datasets available through Google Earth Engine.
    Each property is extracted for all standard depth bands and spatially
    averaged over a square buffer region centered on the input coordinates.

    Args:
        lat: Latitude of the center point in degrees
        lon: Longitude of the center point in degrees
        area_meters: Buffer radius in meters; the actual extraction region
                     is a square bounding box around this circular buffer

    Returns:
        dict: Nested dictionary with structure:
              {property_name: {depth_interval: mean_value}}
              where property_name is one of:
              'phh2o', 'soc', 'clay', 'sand', 'silt', 'bdod', 'cec'
    """
    print(f"Downloading satellite information {SOURCE_NAME} to analyze "
          f"{ANALYSIS_VARIABLE}...")

    # Create an Earth Engine point geometry from lon/lat
    # Note: EE expects [longitude, latitude] order
    center = ee.Geometry.Point([lon, lat])

    # Create the extraction region: buffer the point by area_meters,
    # then get the bounding box (.bounds()) to create a square region.
    # This matches the area shown by draw_area_box() in tile_utils.py.
    region = center.buffer(area_meters).bounds()

    # Extract each soil property for all standard depth intervals
    results = {}
    for prop, collection_id in SOILGRIDS_PROPERTIES.items():
        results[prop] = {}

        # Load the SoilGrids image collection for this property
        # Each image contains multiple bands, one per depth interval
        image = ee.Image(collection_id)

        for depth in DEPTHS:
            # Build the band name following SoilGrids naming convention
            # e.g., 'clay_0-5cm_mean'
            band_name = f"{prop}_{depth}_mean"

            # Reduce the region: compute the spatial mean of the band.
            # scale=250 matches SoilGrids native resolution (250m).
            # NOTE: for buffers smaller than 250m (e.g. a 2 ha plot is
            # ~70m across), reduceRegion effectively samples a single
            # 250m pixel rather than averaging many cells.
            val = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=250,
                bestEffort=True
            ).get(band_name)

            # Convert the EE server-side object to a Python value.
            # getInfo() triggers the actual computation on EE servers.
            results[prop][depth] = val.getInfo()

    print(f"Satellite information {SOURCE_NAME} acquired.")
    return results


# ---------------------------------------------------------------------------
# Saving Functions
# ---------------------------------------------------------------------------

def save_soil_profile(
    soil_data: dict,
    out_prefix: str = "soil_profile_data",
    output_dir: str = "../../databases",
) -> Tuple[Path, pd.DataFrame]:
    """
    Saves the raw SoilGrids properties to a CSV file with a timestamp in the filename.

    The data is transposed so that soil properties become rows and depths become
    columns, making it easier to read and compare values across depths for each
    property.

    Args:
        soil_data: Nested dictionary from get_soil_profile_area()
        out_prefix: Base name for the output CSV file
        output_dir: Directory where the CSV is saved (created if it doesn't exist)

    Returns:
        tuple: (output_path as Path object, pandas DataFrame of the saved data)
    """
    # Transpose the nested dict: properties become rows, depths become columns
    df = pd.DataFrame(soil_data).T
    df.index.name = 'property'
    df.reset_index(inplace=True)

    # Generate a unique filename with timestamp to track different extractions
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


def save_hydraulic_profile(
    df_hydric: pd.DataFrame,
    out_prefix: str = "soil_hydraulic_data",
    output_dir: str = "../../databases",
) -> Path:
    """
    Saves the calculated hydraulic properties to a CSV file with a timestamp
    in the filename.

    The depth index is converted to a regular column for cleaner CSV output,
    making each row represent one depth layer with its hydraulic properties.

    Args:
        df_hydric: DataFrame from calculate_hydraulic_properties() in soil_hydraulics.py
        out_prefix: Base name for the output CSV file
        output_dir: Directory where the CSV is saved (created if it doesn't exist)

    Returns:
        pathlib.Path: Path to the saved CSV file
    """
    # Convert the depth index to a regular column for clean CSV output
    df = df_hydric.reset_index().rename(columns={'index': 'depth'})

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
    return out_path


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Reference points for quick access (commented out):
    # El Playon         --||     7.4584221918243045,    -73.222052853104
    # Finca Matanza     --||     7.300921,              -73.009794
    # Sugarcane_COL     --||     3.580109040361371,     -76.31299479308868
    # Sugarcane_QLD     --||     -19.689669877950884,   147.22717515914223
    # Emerald_QLD       --||     -23.596836971029173,   148.1870868479914

    # Calculate the buffer radius from hectares (shared helper from
    # tile_utils.py; same formula used across terrain/soil profile scripts)
    ha = 2
    area_meters = hectares_to_radius_meters(ha)

    # Example point: sugarcane field (Colombia)
    lat, lon = -19.689669877950884,   147.22717515914223

    # Step 1: Extract raw soil properties from SoilGrids
    soil_data = get_soil_profile_area(lat, lon, area_meters)

    # Step 2: Save the raw soil properties to CSV and display them
    _, df_soil_data = save_soil_profile(soil_data)
    print(df_soil_data.head(10))

    # Step 3: Calculate hydraulic properties (field capacity, wilting point, AWC)
    df_hydric = calculate_hydraulic_properties(soil_data)

    # Step 4: Save the calculated hydraulic properties to CSV
    save_hydraulic_profile(df_hydric)

    # Display the hydraulic properties table
    print(df_hydric)

    # Calculate and display the total available water capacity across all layers
    # This represents the total plant-available water in the 0-100cm root zone
    total_awc_mm = df_hydric['AWC_layer_mm'].sum()
    print(f"\nTotal available water depth (0-100cm): {total_awc_mm:.2f} mm")
