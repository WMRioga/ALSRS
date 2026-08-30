"""
Evapotranspiration Profile Module
=================================

Extracts the biweekly actual evapotranspiration (ET) series for a point,
using a dual source in Earth Engine:

1. MODIS/061/MOD16A2GF (Gap-Filled) - primary source, with gap filling,
   but "year-end gap-filled": the current year may have NO data at all
   until it closes.
2. MODIS/061/MOD16A2 (non gap-filled) - fallback source, near real-time,
   for biweekly periods where the primary source hasn't published yet.
   Without gap filling, so individual composites may still be missing
   due to persistent cloud cover.

This module is the Python equivalent of the Jupyter notebook
``MDS650_v260807_Evapotranspiration.ipynb``.

HOW TO READ THE RESULT
-----------------------
et_total_mm    Total water lost through evaporation + transpiration in
               that biweekly period (based on the current vegetation of
               the pixel, not necessarily the crop you plan to plant -
               see note below).
source         Which product the data came from: 'GF', 'no-GF', or
               'no_data' if neither source had a record for that
               biweekly period.

USE IN WATER BALANCE: together with precipitation_profile.py (input)
and soil_hydraulics.py (storage capacity), this closes the balance:
    Δstorage = precipitation - ET - surplus (when exceeding AWC)

IMPORTANT LIMITATION: Actual ET reflects the EXISTING vegetation in the
pixel today, not the crop you plan to plant (which may consume more or
less water). It serves as an indicative baseline for deciding whether
installed irrigation capacity is needed, not as exact system sizing.

TECHNICAL LIMITATION (composites): both products come in 8-day composites
that don't align exactly with the biweekly period boundaries (1-15,
16-end of month) -> there may be a small shift at the boundaries. For
the purpose of detecting general deficit, this does not affect the
conclusion.

CITATIONS (methodology):
Running, S., Mu, Q., & Zhao, M. (2021). MODIS/Terra Net
Evapotranspiration Gap-Filled 8-Day L4 Global 500m SIN Grid V061
[Data set]. NASA EOSDIS LP DAAC. https://doi.org/10.5067/MODIS/MOD16A2GF.061
Running, S., Mu, Q., & Zhao, M. (2021). MODIS/Terra Net
Evapotranspiration 8-Day L4 Global 500m SIN Grid V061 [Data set].
NASA EOSDIS LP DAAC. https://doi.org/10.5067/MODIS/MOD16A2.061

Dependencies:
    - earthengine-api (ee)
    - pandas
    - period_utils (custom module, same directory)
"""
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import ee
import pandas as pd

# Make the local modules importable regardless of the working directory.
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "common", _PROJECT_ROOT / "extraction",
           _PROJECT_ROOT / "analysis", _PROJECT_ROOT / "mapping"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from period_utils import build_biweekly_periods

# Google Cloud project linked to Earth Engine. Required since Earth Engine
# now requires every ee.Initialize() call to be associated with a project.
EE_PROJECT = "famous-strategy-376313"

try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)


# ---------------------------------------------------------------------------
# Extraction Function
# ---------------------------------------------------------------------------

def get_et_biweekly(
    lat: float,
    lon: float,
    start_date: str = "2016-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extracts biweekly actual evapotranspiration statistics using a dual-source
    approach with automatic fallback from MOD16A2GF to MOD16A2.

    Priority logic:
    1. Try MOD16A2GF (gap-filled, more complete) first
    2. Fall back to MOD16A2 (non gap-filled, near real-time) if GF unavailable
    3. Mark as 'no_data' if neither source has coverage

    MOD16 ET values are scaled: raw value x 0.1 = real mm accumulated in the
    composite. The scale factor is applied immediately when loading the
    collections.

    Args:
        lat: Latitude of the point in degrees
        lon: Longitude of the point in degrees
        start_date: Start date for the analysis period (str "YYYY-MM-DD")
        end_date: End date for the analysis period (str "YYYY-MM-DD");
                  None defaults to today

    Returns:
        pandas.DataFrame: Biweekly ET statistics with columns:
                          period_start, period_end, label, et_total_mm, source
    """
    # Parse date strings to Python date objects
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else date.today()

    # Generate the list of complete biweekly periods within the date range
    periods = build_biweekly_periods(start, end)
    print(f"[DEBUG] {len(periods)} biweekly periods to process, from {start} to {end}")

    # Create Earth Engine point geometry for the extraction location
    point = ee.Geometry.Point([lon, lat])

    # Load primary source: MOD16A2GF (Gap-Filled).
    # Apply the scale factor (0.1) immediately to convert raw values to real
    # mm. copyProperties preserves system:time_start so filterDate still
    # works after the band math (a defensive measure against property loss).
    col_gf = (
        ee.ImageCollection('MODIS/061/MOD16A2GF')
        .select('ET')
        .map(lambda img: img.multiply(0.1).copyProperties(img, ['system:time_start']))
    )

    # Load fallback source: MOD16A2 (non gap-filled, near real-time).
    # Same scale factor and property preservation as above.
    col_fallback = (
        ee.ImageCollection('MODIS/061/MOD16A2')
        .select('ET')
        .map(lambda img: img.multiply(0.1).copyProperties(img, ['system:time_start']))
    )

    # Build the list of periods as an ee.List of dictionaries for
    # server-side mapping (single network call at the end).
    ee_periods = ee.List([
        {'label': label, 'start': str(p_start), 'end': str(p_end)}
        for label, p_start, p_end in periods
    ])

    def compute_period(period):
        """
        Server-side function to compute ET statistics for a single period
        using dual-source logic with automatic fallback.
        This runs on Google Earth Engine servers, not locally.

        Priority: MOD16A2GF (gap-filled) > MOD16A2 (near real-time) > no_data

        Args:
            period: ee.Dictionary with keys 'label', 'start', 'end'

        Returns:
            ee.Feature with computed ET total and source indicator
        """
        period = ee.Dictionary(period)
        p_start = ee.Date(period.get('start'))
        p_end = ee.Date(period.get('end'))

        # Filter both collections to the current period
        filtered_gf = col_gf.filterDate(p_start, p_end)
        filtered_fb = col_fallback.filterDate(p_start, p_end)

        # Check if each source has data available for this period
        has_gf = filtered_gf.size().gt(0)
        has_fb = filtered_fb.size().gt(0)

        # Build images for each source with appropriate source labels.
        # If data exists: sum all composites and tag with source identifier.
        # If no data: create a self-masked image (returns NaN) tagged 'no_data'.
        # (The no_data branch of img_gf is defensive: final_img only selects
        # img_gf when has_gf is true, so that branch is normally unreachable.)
        img_gf = ee.Image(ee.Algorithms.If(
            has_gf,
            filtered_gf.sum().rename('et_total_mm').set('source', 'GF'),
            ee.Image.constant(0).rename('et_total_mm').selfMask().set('source', 'no_data')
        ))
        img_fb = ee.Image(ee.Algorithms.If(
            has_fb,
            filtered_fb.sum().rename('et_total_mm').set('source', 'no-GF'),
            ee.Image.constant(0).rename('et_total_mm').selfMask().set('source', 'no_data')
        ))

        # Prioritize GF (gap-filled); if unavailable, use the fallback
        # (which may itself end up as 'no_data' if no data exists there either).
        final_img = ee.Image(ee.Algorithms.If(has_gf, img_gf, img_fb))

        # Extract the pixel value at the point location.
        # scale=500 is MOD16 native resolution.
        stats = final_img.reduceRegion(
            reducer=ee.Reducer.first(),  # Take the first (only) pixel value at the point
            geometry=point,
            scale=500,
            maxPixels=1e9
        )

        # Return as a Feature with computed statistics, period metadata, and
        # source. The 'source' tag lives on the IMAGE (not inside reduceRegion's
        # output), so it is read from final_img.get('source').
        return ee.Feature(
            None,
            stats
            .set('label', period.get('label'))
            .set('period_start', p_start.format('YYYY-MM-dd'))
            .set('period_end', p_end.advance(-1, 'day').format('YYYY-MM-dd'))
            .set('source', final_img.get('source'))
        )

    # Map compute_period over all periods (server-side execution),
    # then retrieve everything in a single network call.
    result = ee.FeatureCollection(ee_periods.map(compute_period)).getInfo()

    # Parse the Earth Engine response into a list of dictionaries
    rows = []
    for feature in result['features']:
        props = feature['properties']
        rows.append({
            'period_start': props.get('period_start'),
            'period_end': props.get('period_end'),
            'label': props.get('label'),
            # 'lat': lat,   # Uncomment to include coordinates in the output
            # 'lon': lon,   # Uncomment to include coordinates in the output
            'et_total_mm': round(props.get('et_total_mm'), 2) if props.get('et_total_mm') is not None else None,
            'source': props.get('source'),
        })

    # Build the DataFrame. period_start/period_end are ISO strings
    # ("YYYY-MM-dd"), which sort chronologically when sorted as text, so no
    # datetime conversion is required (and it keeps the CSV dates clean).
    df = pd.DataFrame(rows)
    df = df.sort_values('period_start').reset_index(drop=True)

    # Check for biweekly periods where neither source had data
    null_rows = df['et_total_mm'].isna().sum()
    if null_rows > 0:
        first_nulls = df[df['et_total_mm'].isna()]['label'].tolist()
        print(f"[WARNING] {null_rows} biweekly period(s) without data in any "
              f"source (neither GF nor no-GF): {first_nulls}")

    return df


# ---------------------------------------------------------------------------
# Saving Function
# ---------------------------------------------------------------------------

def save_et_profile(
    df: pd.DataFrame,
    out_prefix: str = "et_biweekly",
    output_dir: str = "../../databases",
) -> Path:
    """
    Saves the evapotranspiration series with a timestamp in the filename:
    {out_prefix}-vYYMMDDHHMMSS.csv (same pattern as the rest of the pipeline).

    Args:
        df: DataFrame from get_et_biweekly()
        out_prefix: Base name for the output CSV file
        output_dir: Directory where the CSV is saved (created if it doesn't exist)

    Returns:
        pathlib.Path: Path to the saved CSV file
    """
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
    return out_path


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test point: Sugarcane field in Queensland, Australia
    Latitude, Longitude = -19.689669877950884, 147.22717515914223

    # Reference points for quick access (commented out):
    # El Playon         --||     7.4584221918243045,    -73.222052853104
    # Finca Matanza     --||     7.300921,              -73.009794
    # Sugarcane_COL     --||     3.580109040361371,     -76.31299479308868
    # Sugarcane_QLD     --||     -19.689669877950884,   147.22717515914223

    # Extract biweekly ET data from 2016 onwards using dual-source approach
    df = get_et_biweekly(Latitude, Longitude, start_date="2016-01-01")

    # Save the ET profile to CSV
    out_path = save_et_profile(df)

    # Display the first 10 rows for inspection
    print(df.head(10))
