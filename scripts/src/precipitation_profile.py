"""
Precipitation Profile Module
============================

Extracts the biweekly precipitation series (accumulated total, rainy days,
and daily maximum) for a point, using CHIRPS Daily via Google Earth Engine.

Unlike temperature (which is averaged), precipitation is ACCUMULATED:
what matters agronomically is how much total rain fell, not the average
daily rate.

This module is the Python equivalent of the Jupyter notebook
``MDS650_v260806_precipitation_profile.ipynb``.

HOW TO READ THE RESULT
-----------------------
precip_total_mm          Total accumulated rainfall in the biweekly period, in mm.
precip_rainy_days        Number of days in the biweekly period with rainfall
                         (> threshold, standard 1 mm to distinguish actual rain
                         from moisture/dew measured by error).
                         Relevant for the "no more than one week without water"
                         criterion: a biweekly period can have the same total_mm
                         with very different distribution (e.g., all rain in
                         2 days vs spread over 8 days).
precip_max_daily_mm      The rainiest day of the biweekly period. Indicator of
                         intense rainfall events (erosion/waterlogging risk)
                         that the total or rainy days alone don't show.

LIMITATION TO KEEP IN MIND: precip_rainy_days counts rainy days WITHIN the
biweekly period, but does not detect dry spells that cross the boundary
between two periods (e.g., last 4 dry days of one period + first 5 dry days
of the next = 9 consecutive dry days, invisible in this table). If the
irrigation analysis needs to capture that, a separate "maximum dry spell"
calculation over the complete daily series can be added.

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


# ---------------------------------------------------------------------------
# Extraction Function
# ---------------------------------------------------------------------------

def get_precipitation_biweekly(
    lat: float,
    lon: float,
    start_date: str = "2016-01-01",
    end_date: Optional[str] = None,
    rain_threshold_mm: float = 1.0,
) -> pd.DataFrame:
    """
    Extracts biweekly precipitation statistics from CHIRPS Daily for a given point.

    For each biweekly period, computes:
    - Total accumulated precipitation (sum of all daily values)
    - Number of rainy days (days exceeding the threshold)
    - Maximum daily precipitation (peak intensity)

    Args:
        lat: Latitude of the point in degrees
        lon: Longitude of the point in degrees
        start_date: Start date for the analysis period (str "YYYY-MM-DD")
        end_date: End date for the analysis period (str "YYYY-MM-DD");
                  None defaults to today
        rain_threshold_mm: Threshold to consider a day as a "rainy day"
                          (default 1.0 mm, standard to exclude dew/moisture)

    Returns:
        pandas.DataFrame: Biweekly precipitation statistics with columns:
                          period_start, period_end, label, lat, lon,
                          precip_total_mm, precip_rainy_days, precip_max_daily_mm
    """
    # Parse date strings to Python date objects
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else date.today()

    # Generate the list of complete biweekly periods within the date range
    periods = build_biweekly_periods(start, end)
    print(f"[DEBUG] {len(periods)} biweekly periods to process, from {start} to {end}")

    # Create Earth Engine point geometry for the extraction location
    point = ee.Geometry.Point([lon, lat])

    # Load CHIRPS Daily precipitation collection.
    # Filter to the date range (adding 1 day to end to include the last day).
    chirps_coll = (
        ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        .select('precipitation')
        .filterDate(str(start), str(end + timedelta(days=1)))
    )

    # Build the list of periods as an ee.List of dictionaries, to process
    # them ALL on the server with .map() -> a single network call at the
    # end, instead of one per biweekly period.
    ee_periods = ee.List([
        {'label': label, 'start': str(p_start), 'end': str(p_end)}
        for label, p_start, p_end in periods
    ])

    def compute_period(period):
        """
        Server-side function to compute precipitation statistics for a single
        period. This runs on Google Earth Engine servers, not locally.

        The closure variables (chirps_coll, point, rain_threshold_mm) are
        serialized into the mapped function: chirps_coll and point are EE
        objects, while rain_threshold_mm is a plain float that EE captures
        as a constant.

        Args:
            period: ee.Dictionary with keys 'label', 'start', 'end'

        Returns:
            ee.Feature with computed precipitation statistics
        """
        period = ee.Dictionary(period)
        p_start = ee.Date(period.get('start'))
        p_end = ee.Date(period.get('end'))

        # Filter the collection to only include images within this period
        filtered = chirps_coll.filterDate(p_start, p_end)

        # 1. Total accumulated precipitation (sum of all daily values)
        total = filtered.sum().rename('precip_total_mm')

        # 2. Maximum daily precipitation (peak intensity day)
        max_daily = filtered.max().rename('precip_max_daily_mm')

        # 3. Number of rainy days: count days where precipitation > threshold.
        #    .gt() returns 1 where above the threshold, 0 otherwise; summing
        #    those binary masks counts the rainy days (strictly greater than).
        rainy_days = (
            filtered
            .map(lambda img: img.gt(rain_threshold_mm))
            .sum()
            .rename('precip_rainy_days')
        )

        # Combine all three bands into a single multi-band image
        combined = total.addBands(max_daily).addBands(rainy_days)

        # Extract the pixel value at the point location.
        # scale=5566 is CHIRPS native resolution (~0.05 degrees ~= 5.5 km).
        stats = combined.reduceRegion(
            reducer=ee.Reducer.first(),  # Take the first (only) pixel value at the point
            geometry=point,
            scale=5566,
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

        # Rainy-days is a sum of 0/1 masks -> an integer count. CHIRPS is
        # float-typed, so EE returns it as e.g. 5.0; cast to int for a clean CSV.
        rainy_days = props.get('precip_rainy_days')

        rows.append({
            'period_start': props.get('period_start'),
            'period_end': props.get('period_end'),
            'label': props.get('label'),
            'lat': lat,
            'lon': lon,
            'precip_total_mm': round(props.get('precip_total_mm'), 2) if props.get('precip_total_mm') is not None else None,
            'precip_rainy_days': int(rainy_days) if rainy_days is not None else None,
            'precip_max_daily_mm': round(props.get('precip_max_daily_mm'), 2) if props.get('precip_max_daily_mm') is not None else None,
        })

    # Build the DataFrame. period_start/period_end are ISO strings
    # ("YYYY-MM-dd"), which sort chronologically when sorted as text, so no
    # datetime conversion is required (and it keeps the CSV dates clean).
    df = pd.DataFrame(rows)
    df = df.sort_values('period_start').reset_index(drop=True)

    # CHIRPS typically has a 1-2 month lag in publishing the most recent data
    # (it needs ground station data for the final product) -> even though a
    # biweekly period has already "passed" on the calendar, the provider may
    # not have published it yet. We warn if null rows are found at the end of
    # the series, instead of silently ignoring them.
    null_rows = df['precip_total_mm'].isna().sum()
    if null_rows > 0:
        first_nulls = df[df['precip_total_mm'].isna()]['label'].tolist()
        print(f"[WARNING] {null_rows} biweekly period(s) without data yet "
              f"(CHIRPS publication lag): {first_nulls}")

    return df


# ---------------------------------------------------------------------------
# Saving Function
# ---------------------------------------------------------------------------

def save_precipitation_profile(
    df: pd.DataFrame,
    out_prefix: str = "precipitation_biweekly",
    output_dir: str = "../../databases",
) -> Path:
    """
    Saves the precipitation series with a timestamp in the filename:
    {out_prefix}-vYYMMDDHHMMSS.csv (same pattern as temperature_profile.py).

    Args:
        df: DataFrame from get_precipitation_biweekly()
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

    # Extract biweekly precipitation data from 2016 onwards
    df = get_precipitation_biweekly(LAT, LON, start_date="2016-01-01")

    # Save the precipitation profile to CSV
    out_path = save_precipitation_profile(df)

    # Display the first 10 rows for inspection
    print(df.head(10))
