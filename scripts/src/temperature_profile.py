"""
Temperature Profile Module
==========================

Extracts the biweekly temperature series (mean, standard deviation, and
TEMPORAL variance, i.e., between days within each biweekly period - not to
be confused with the SPATIAL std we calculate in terrain_profile.py) for a
point, using ERA5-Land Daily Aggregates via Google Earth Engine.

This module is the Python equivalent of the Jupyter notebook
``MDS650_v260806_temperature_profile.ipynb``.

HOW TO READ THE RESULT
-----------------------
mean_C    Mean temperature for that biweekly period, in °C.
std_C     How much the temperature varied day to day within that biweekly
          period. High = biweekly period with very dissimilar days (e.g.,
          mix of cool and hot days); low = stable temperature throughout
          the period.
var_C     The same variability, expressed as variance (std squared).

Dependencies:
    - earthengine-api (ee)
    - pandas
    - period_utils (custom module, same directory)
"""
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import ee
import pandas as pd

from period_utils import build_biweekly_periods

# Google Cloud project linked to Earth Engine. Required since Earth Engine
# now requires every ee.Initialize() call to be associated with a project.
EE_PROJECT = "famous-strategy-376313"

try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

# Data source label used in the download/acquire progress messages.
SOURCE_NAME = "ERA5-Land"
ANALYSIS_VARIABLE = "temperature"


# ---------------------------------------------------------------------------
# Extraction Function
# ---------------------------------------------------------------------------

def get_temperature_biweekly(
    lat: float,
    lon: float,
    start_date: str = "2016-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extracts biweekly temperature statistics from ERA5-Land Daily Aggregates
    for a given point.

    For each biweekly period, computes the mean, standard deviation, and
    variance of daily temperatures. Values are converted from Kelvin to
    Celsius.

    Args:
        lat: Latitude of the point in degrees
        lon: Longitude of the point in degrees
        start_date: Start date for the analysis period (str "YYYY-MM-DD")
        end_date: End date for the analysis period (str "YYYY-MM-DD");
                  None defaults to today.
                  NOTE: ERA5-Land has a few days of latency in its
                  publication -> the most recent biweekly periods may not
                  be available yet even though they have "already passed"
                  on the calendar.

    Returns:
        pandas.DataFrame: Biweekly temperature statistics with columns:
                          period_start, period_end, label,
                          mean_C, std_C, var_C
    """
    print(f"Downloading satellite information {SOURCE_NAME} to analyze "
          f"{ANALYSIS_VARIABLE}...")

    # Parse date strings to Python date objects
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else date.today()

    # Generate the list of complete biweekly periods within the date range
    periods = build_biweekly_periods(start, end)
    print(f"[DEBUG] {len(periods)} biweekly periods to process, from {start} to {end}")

    # Create Earth Engine point geometry for the extraction location
    point = ee.Geometry.Point([lon, lat])

    # Load ERA5-Land Daily Aggregates temperature collection.
    # Filter to the date range (adding 1 day to end to include the last day).
    era5_coll = (
        ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
        .select('temperature_2m')  # 2-meter air temperature in Kelvin
        .filterDate(str(start), str(end + timedelta(days=1)))
    )

    # Combined reducer: mean, standard deviation, and variance of the daily
    # images within each period. sharedInputs=True applies all reducers to
    # the same input data.
    combined_reducer = (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.variance(), sharedInputs=True)
    )

    # Build the list of periods as an ee.List of dictionaries, to process
    # them ALL on the server with .map() -> a single network call at the
    # end, instead of one per biweekly period (240 before).
    # This is a major performance optimization.
    ee_periods = ee.List([
        {'label': label, 'start': str(p_start), 'end': str(p_end)}
        for label, p_start, p_end in periods
    ])

    def compute_period(period):
        """
        Server-side function to compute temperature statistics for a single
        period. This runs on Google Earth Engine servers, not locally.

        The closure variables (era5_coll, point, combined_reducer) are all
        Earth Engine objects, so EE serializes them into the mapped function.

        Args:
            period: ee.Dictionary with keys 'label', 'start', 'end'

        Returns:
            ee.Feature with computed temperature statistics (mean, stdDev, variance)
        """
        period = ee.Dictionary(period)
        p_start = ee.Date(period.get('start'))
        p_end = ee.Date(period.get('end'))

        # Filter the collection to only include images within this period
        filtered = era5_coll.filterDate(p_start, p_end)

        # Reduce the filtered collection to get temporal statistics, then
        # extract the pixel value at the point location.
        stats = filtered.reduce(combined_reducer).reduceRegion(
            reducer=ee.Reducer.first(),  # Take the first (only) pixel value at the point
            geometry=point,
            scale=9000,  # ERA5-Land native resolution (~9 km)
            maxPixels=1e9
        )

        # Return as a Feature with computed statistics and period metadata.
        # p_end is exclusive, so subtract 1 day to get the actual last day included.
        return ee.Feature(
            None,
            stats
            .set('label', period.get('label'))
            .set('period_start', p_start.format('YYYY-MM-dd'))
            .set('period_end', p_end.advance(-1, 'day').format('YYYY-MM-dd'))
        )

    # Map compute_period over all periods (server-side execution),
    # then retrieve everything in a single network call.
    result = ee.FeatureCollection(ee_periods.map(compute_period)).getInfo()

    # Parse the Earth Engine response into a list of dictionaries
    rows = []
    for feature in result['features']:
        props = feature['properties']

        # Raw values from Earth Engine (in Kelvin)
        mean_k = props.get('temperature_2m_mean')
        std_k = props.get('temperature_2m_stdDev')
        var_k = props.get('temperature_2m_variance')

        # Convert Kelvin to Celsius (mean only). std and variance are
        # shift-invariant, so they need NO offset: 1 K == 1 °C as a difference.
        rows.append({
            'period_start': props.get('period_start'),
            'period_end': props.get('period_end'),
            'label': props.get('label'),
            # 'lat': lat,   # Uncomment to include coordinates in the output
            # 'lon': lon,   # Uncomment to include coordinates in the output
            'mean_C': round(mean_k - 273.15, 2) if mean_k is not None else None,
            'std_C': round(std_k, 2) if std_k is not None else None,
            'var_C': round(var_k, 2) if var_k is not None else None,
        })

    # Build the DataFrame. period_start/period_end are ISO strings
    # ("YYYY-MM-dd"), which sort chronologically when sorted as text, so no
    # datetime conversion is required (and it keeps the CSV dates clean).
    df = pd.DataFrame(rows)
    df = df.sort_values('period_start').reset_index(drop=True)
    print(f"Satellite information {SOURCE_NAME} acquired.")
    return df


# ---------------------------------------------------------------------------
# Saving Function
# ---------------------------------------------------------------------------

def save_temperature_profile(
    df: pd.DataFrame,
    out_prefix: str = "temperature_biweekly",
    output_dir: str = "../../databases",
) -> Path:
    """
    Saves the temperature series with a timestamp in the filename:
    {out_prefix}-vYYMMDDHHMMSS.csv (same pattern as terrain_profile/soil_profile).

    Args:
        df: DataFrame from get_temperature_biweekly()
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
    LAT = -19.689669877950884
    LON = 147.22717515914223

    # Reference points for quick access (commented out):
    # El Playon         --||     7.4584221918243045,    -73.222052853104
    # Finca Matanza     --||     7.300921,              -73.009794
    # Sugarcane_COL     --||     3.580109040361371,     -76.31299479308868
    # Sugarcane_QLD     --||     -19.689669877950884,   147.22717515914223

    # Extract biweekly temperature data from 2016 onwards
    df = get_temperature_biweekly(LAT, LON, start_date="2016-01-01")

    # Save the temperature profile to CSV
    out_path = save_temperature_profile(df)

    # Display the first 10 rows for inspection
    print(df.head(10))
