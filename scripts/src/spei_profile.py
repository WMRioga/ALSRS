"""
SPEI Profile Module
===================

Calculates the Standardized Precipitation Evapotranspiration Index (SPEI)
for a point, using CHIRPS (precipitation) and ERA5-Land (temperature) data
via Google Earth Engine.

The SPEI (Vicente-Serrano et al., 2010) is a multi-scalar drought index that
combines precipitation and potential evapotranspiration to characterize
wet and dry conditions. Unlike SPI (which only uses precipitation), SPEI
incorporates temperature, making it sensitive to climate change effects
on water demand.

This module is the Python equivalent of the Jupyter notebook
``MDS650_v2608114_SPEI.ipynb``.

METHOD (L-moments / unbiased PWM):
The log-logistic distribution is fitted to the accumulated water balance
D = P - PET using L-moments estimated from unbiased Probability Weighted
Moments (PWM). This is the method recommended by Beguería et al. (2014)
and used by the reference R package ``SPEI`` (its ``pelglo`` generalized
logistic parameterization). It is robust for short series, which is the
case here with biweekly data (one value per year per biweekly position).

HOW TO READ THE RESULT
-----------------------
spei_1m    Short-term moisture conditions (last ~1 month)
           Captures immediate water stress during crop establishment.

spei_3m    Seasonal drought context (last ~3 months)
           Indicates whether the growing season has been dry or wet.

spei_6m    Medium-term agricultural drought (last ~6 months)
           Reflects cumulative water deficit over the crop cycle.

spei_12m   Long-term hydrological drought (last ~12 months)
           Distinguishes between chronic dry zones and occasional dry years.

SPEI values interpretation:
    >  2.0       : Extremely wet
     1.5 to 2.0  : Severely wet
     1.0 to 1.5  : Moderately wet
    -1.0 to 1.0  : Near normal
    -1.5 to -1.0 : Moderately dry
    -2.0 to -1.5 : Severely dry
    < -2.0       : Extremely dry

REFERENCES:
Vicente-Serrano, S.M., Beguería, S., & López-Moreno, J.I. (2010).
A multi-scalar drought index sensitive to global warming: The Standardized
Precipitation Evapotranspiration Index. Journal of Climate, 23(7), 1696-1718.
https://doi.org/10.1175/2009JCLI2909.1

Beguería, S., Vicente-Serrano, S.M., Reig, F., & Latorre, B. (2014).
Standardized precipitation evapotranspiration index (SPEI) revisited:
parameter fitting, evapotranspiration models, tools, datasets and drought
monitoring. International Journal of Climatology, 34(10), 3001-3023.
https://doi.org/10.1002/joc.3887

Hosking, J.R.M., & Wallis, J.R. (1997). Regional Frequency Analysis:
An Approach Based on L-Moments. Cambridge University Press.

Dependencies:
    - earthengine-api (ee)
    - pandas
    - numpy
    - scipy (scipy.stats.norm)
    - period_utils (custom module, same directory)
"""
from datetime import date, datetime, timedelta
import math
from pathlib import Path
from typing import Optional, Tuple

import ee
import numpy as np
import pandas as pd
from scipy.stats import norm

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
# PET Calculation
# ---------------------------------------------------------------------------

def _daylength_factor(month: int, lat: float) -> float:
    """
    Thornthwaite day-length correction factor: (N/30) * (h/12), where N is
    the number of days in the month and h is the mean daylight hours at the
    given latitude (from the solar declination at mid-month).

    Args:
        month: Calendar month (1-12).
        lat: Latitude in degrees.

    Returns:
        float: Dimensionless day-length correction factor (~1.0 near the
               equator, larger for long summer days at high latitudes).
    """
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    mid_month_doy = sum(days_in_month[:month - 1]) + 15

    lat_rad = math.radians(lat)
    declination = 0.4093 * math.sin(2 * math.pi / 365.0 * (mid_month_doy - 81))
    cos_hour_angle = -math.tan(lat_rad) * math.tan(declination)
    cos_hour_angle = max(-1.0, min(1.0, cos_hour_angle))
    hour_angle = math.acos(cos_hour_angle)
    daylight_hours = 24.0 / math.pi * hour_angle

    return (days_in_month[month - 1] / 30.0) * (daylight_hours / 12.0)


def calculate_pet_thornthwaite(temp_c: pd.Series, lat: float) -> pd.Series:
    """
    Calculates biweekly potential evapotranspiration (PET) using the
    Thornthwaite (1948) method.

    The method is applied at the MONTHLY scale and then distributed to
    biweekly periods, correcting the original notebook's approach (which fed
    biweekly temperature into the monthly formula and approximated the annual
    heat index as i * 12). The steps are:

    1. Biweekly temperature is aggregated to monthly mean temperature.
    2. The annual heat index I is the sum of the 12 monthly heat indices,
       computed from the climatological monthly mean temperatures.
    3. Monthly PET = 16 * (10 * T / I) ** a, with the standard day-length
       correction (N/30 * h/12).
    4. Monthly PET is split equally into its two biweekly periods.

    Args:
        temp_c: pandas Series of biweekly mean temperature (°C), indexed by
                a DatetimeIndex (period_start).
        lat: Latitude in degrees.

    Returns:
        pandas Series of biweekly PET (mm), with the same index as temp_c.
    """
    # 1. Aggregate biweekly -> monthly mean temperature (month start).
    monthly = temp_c.resample('MS').mean()

    # 2. Climatological monthly mean temperature (12 calendar months).
    clim = monthly.groupby(monthly.index.month).mean()

    # Monthly heat indices and annual heat index (sum of the 12 months).
    i_month = (clim / 5.0) ** 1.514
    i_month = i_month.where(clim > 0, 0.0)
    annual_i = float(i_month.sum())

    # Exponent a (empirical function of the annual heat index).
    a = (6.75e-7 * annual_i**3) - (7.71e-5 * annual_i**2) + (1.792e-2 * annual_i) + 0.49239

    # 3. Monthly PET (unadjusted). NaN temperature -> NaN PET (missing data);
    #    temperature <= 0 -> 0 mm (Thornthwaite is undefined below 0 °C).
    def _monthly_pet(t: float) -> float:
        if pd.isna(t):
            return np.nan
        if t <= 0:
            return 0.0
        return 16 * (10 * t / annual_i) ** a

    pet_monthly = monthly.apply(_monthly_pet)

    # Day-length correction per calendar month.
    daylength = pd.Series({m: _daylength_factor(m, lat) for m in range(1, 13)})
    pet_monthly = pet_monthly * monthly.index.month.map(daylength)

    # 4. Split each month's PET equally into its two biweekly periods.
    pet_monthly = pet_monthly.copy()
    pet_monthly.index = pet_monthly.index.to_period('M')
    pet_biweekly = pet_monthly.reindex(temp_c.index.to_period('M'))
    pet_biweekly.index = temp_c.index
    return pet_biweekly / 2.0


# ---------------------------------------------------------------------------
# L-Moments Helpers (unbiased PWM)
# ---------------------------------------------------------------------------

def _sample_lmoments(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes the first three unbiased sample L-moments of a 1-D array.

    Uses the unbiased probability weighted moments (PWM) estimators, which
    are the standard sample L-moments (Hosking & Wallis 1997, eq. 2.4-2.6).

    Args:
        x: 1-D array of values (any order; sorted internally).

    Returns:
        tuple: (l1, l2, l3) the first three L-moments.
    """
    x = np.sort(x)  # ascending order statistics x_(1) <= ... <= x_(n)
    n = len(x)

    b0 = np.mean(x)
    # Unbiased PWM estimators. Weights favour the larger order statistics,
    # consistent with b_r = E[X * F(X)^r].
    ranks = np.arange(1, n + 1)
    b1 = np.sum((ranks - 1) / (n - 1) * x) / n
    b2 = np.sum((ranks - 1) * (ranks - 2) / ((n - 1) * (n - 2)) * x) / n

    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    return l1, l2, l3


def _fit_glo_lmoments(x: np.ndarray) -> Tuple[float, float, float]:
    """
    Fits the generalized logistic (log-logistic) distribution by L-moments.

    This replicates the ``pelglo`` routine of the lmom Fortran code
    (Hosking), which is the estimator used by the reference R ``SPEI``
    package for its log-Logistic distribution.

    Args:
        x: 1-D array of values.

    Returns:
        tuple: (xi, alpha, k) location, scale and shape parameters.
    """
    l1, l2, l3 = _sample_lmoments(x)
    tau3 = l3 / l2

    # Shape parameter: for the GLO, the L-skewness tau_3 equals -k.
    k = -tau3

    if abs(k) < 1e-6:
        # k ~ 0 -> the logistic distribution.
        return l1, l2, 0.0

    gg = k * np.pi / np.sin(k * np.pi)
    alpha = l2 / gg
    xi = l1 - alpha * (1 - gg) / k
    return xi, alpha, k


def _glo_cdf(x: np.ndarray, xi: float, alpha: float, k: float) -> np.ndarray:
    """
    Cumulative distribution function of the generalized logistic (GLO)
    distribution, matching the lmom ``cdfglo`` parameterization.

    Args:
        x: Values at which to evaluate the CDF.
        xi: Location parameter.
        alpha: Scale parameter.
        k: Shape parameter.

    Returns:
        numpy array of CDF values in (0, 1).
    """
    if abs(k) < 1e-6:
        y = (x - xi) / alpha
    else:
        z = 1 - k * (x - xi) / alpha
        z = np.clip(z, 1e-12, None)  # guard against log of non-positive
        y = -np.log(z) / k

    return 1.0 / (1.0 + np.exp(-y))


# ---------------------------------------------------------------------------
# SPEI Calculation
# ---------------------------------------------------------------------------

def calculate_spei_lmoments(
    df_water_balance: pd.DataFrame,
    timescales: Tuple[int, ...] = (1, 3, 6, 12),
) -> dict:
    """
    Calculates SPEI using L-moments (unbiased PWM) following Beguería et al.
    (2014) and the reference R ``SPEI`` package.

    The water balance D = P - PET is accumulated over each timescale window,
    then, for each biweekly position of the year (24 groups), the generalized
    logistic distribution is fitted to the cross-year values via L-moments,
    and standardized to a standard normal (SPEI).

    Args:
        df_water_balance: DataFrame with 'period_start' (str "YYYY-MM-DD")
                          and 'water_balance' columns.
        timescales: SPEI scales in months (1, 3, 6, 12).

    Returns:
        dict: {scale_months: numpy array of SPEI values} aligned to the full
              input series (NaN for the initial accumulation window).
    """
    df = df_water_balance.sort_values('period_start').reset_index(drop=True)
    dates = pd.to_datetime(df['period_start'])
    water_balance = df['water_balance'].to_numpy(dtype=float)

    # Position key within the year: "MM_Q1" (days 1-15) / "MM_Q2" (days 16-end).
    # This groups the same biweekly position across all years (24 groups).
    position_key = np.array([
        f"{d.month:02d}_Q{1 if d.day <= 15 else 2}" for d in dates
    ])

    results = {}
    for scale_months in timescales:
        timescale = scale_months * 2  # 2 biweekly periods per month

        # Accumulate the water balance over the timescale window.
        accum = (
            pd.Series(water_balance)
            .rolling(timescale, min_periods=timescale)
            .sum()
            .to_numpy()
        )

        spei = np.full(len(df), np.nan)

        # Fit one distribution per biweekly position across all years.
        accum_by_pos = pd.DataFrame({'D': accum, 'pos': position_key})
        for _, grp in accum_by_pos.groupby('pos'):
            nonnan = grp['D'].dropna()
            # Same minimum sample size as the R SPEI package (>= 4 years).
            if len(nonnan) < 4:
                continue

            data = nonnan.to_numpy()
            xi, alpha, k = _fit_glo_lmoments(data)
            F = _glo_cdf(data, xi, alpha, k)
            F = np.clip(F, 1e-6, 1 - 1e-6)  # avoid norm.ppf(0/1) -> ±inf
            spei[nonnan.index] = norm.ppf(F)

        results[scale_months] = spei

    return results


# ---------------------------------------------------------------------------
# Extraction Function
# ---------------------------------------------------------------------------

def get_spei_biweekly(
    lat: float,
    lon: float,
    start_date: str = "2016-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculates SPEI at multiple temporal scales (1, 3, 6, 12 months) for a point.

    The SPEI is computed from the water balance (P - PET) accumulated over
    different time windows, then standardized using a log-logistic distribution
    fitted by L-moments (unbiased PWM), following Vicente-Serrano et al. (2010)
    and Beguería et al. (2014).

    Args:
        lat: Latitude of the point in degrees
        lon: Longitude of the point in degrees
        start_date: Start date for the analysis period (str "YYYY-MM-DD")
        end_date: End date for the analysis period (str "YYYY-MM-DD");
                  None defaults to today

    Returns:
        pandas.DataFrame: SPEI values for each biweekly period with columns:
                          period_start, period_end, label,
                          precip_mm, temp_c, pet_mm, water_balance,
                          spei_1m, spei_3m, spei_6m, spei_12m
    """
    # Parse date strings to Python date objects
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else date.today()

    # Generate the list of complete biweekly periods within the date range
    periods = build_biweekly_periods(start, end)
    print(f"[DEBUG] {len(periods)} biweekly periods to process, from {start} to {end}")

    # Create Earth Engine point geometry for the extraction location
    point = ee.Geometry.Point([lon, lat])

    # Load CHIRPS precipitation (mm/day), filtered to the date range.
    chirps_coll = (
        ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        .select('precipitation')
        .filterDate(str(start), str(end + timedelta(days=1)))
    )

    # Load ERA5-Land temperature (K) for PET calculation, filtered to the range.
    era5_coll = (
        ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
        .select('temperature_2m')
        .filterDate(str(start), str(end + timedelta(days=1)))
    )

    # Build the list of periods as an ee.List of dictionaries for server-side mapping.
    ee_periods = ee.List([
        {'label': label, 'start': str(p_start), 'end': str(p_end)}
        for label, p_start, p_end in periods
    ])

    def compute_period(period):
        """
        Server-side function to compute P and T for a single period.
        This runs on Google Earth Engine servers, not locally.

        Args:
            period: ee.Dictionary with keys 'label', 'start', 'end'

        Returns:
            ee.Feature with precipitation sum and mean temperature
        """
        period = ee.Dictionary(period)
        p_start = ee.Date(period.get('start'))
        p_end = ee.Date(period.get('end'))

        # Sum precipitation over the period (mm)
        precip_sum = chirps_coll.filterDate(p_start, p_end).sum().rename('precip_mm')

        # Mean temperature over the period (K -> C)
        temp_mean = era5_coll.filterDate(p_start, p_end).mean().rename('temp_c')
        temp_c = temp_mean.subtract(273.15)

        # Combine into a single image
        combined = precip_sum.addBands(temp_c)

        # Extract values at the point location (CHIRPS native resolution).
        stats = combined.reduceRegion(
            reducer=ee.Reducer.first(),
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
        rows.append({
            'period_start': props.get('period_start'),
            'period_end': props.get('period_end'),
            'label': props.get('label'),
            # 'lat': lat,   # Uncomment to include coordinates in the output
            # 'lon': lon,   # Uncomment to include coordinates in the output
            'precip_mm': round(props.get('precip_mm'), 2) if props.get('precip_mm') is not None else None,
            'temp_c': round(props.get('temp_c'), 2) if props.get('temp_c') is not None else None,
        })

    # Build the DataFrame. period_start/period_end are ISO strings
    # ("YYYY-MM-dd"), which sort chronologically when sorted as text.
    df = pd.DataFrame(rows)
    df = df.sort_values('period_start').reset_index(drop=True)

    # Calculate PET using the Thornthwaite method (monthly scale, then split
    # to biweekly). The function operates on the full series, so it needs a
    # DatetimeIndex built from period_start.
    temp_c_series = pd.Series(
        df['temp_c'].to_numpy(dtype=float),
        index=pd.to_datetime(df['period_start']),
    )
    df['pet_mm'] = calculate_pet_thornthwaite(temp_c_series, lat).to_numpy()

    # Calculate water balance (D = P - PET)
    df['water_balance'] = df['precip_mm'] - df['pet_mm']

    # Calculate SPEI from the (unrounded) water balance via L-moments
    spei_results = calculate_spei_lmoments(df[['period_start', 'water_balance']])

    # Add SPEI values to the DataFrame
    df['spei_1m'] = np.round(spei_results[1], 3)
    df['spei_3m'] = np.round(spei_results[3], 3)
    df['spei_6m'] = np.round(spei_results[6], 3)
    df['spei_12m'] = np.round(spei_results[12], 3)

    # Round PET and water balance for clean CSV output (SPEI was already
    # computed from the unrounded values, so this does not affect the index).
    df['pet_mm'] = df['pet_mm'].round(2)
    df['water_balance'] = df['water_balance'].round(2)

    # Select final columns for output (each line can be commented out)
    df = df[
        [
            'period_start',
            'period_end',
            'label',
            'precip_mm',
            'temp_c',
            'pet_mm',
            'water_balance',
            'spei_1m',
            'spei_3m',
            'spei_6m',
            'spei_12m',
        ]
    ]

    return df


# ---------------------------------------------------------------------------
# Saving Function
# ---------------------------------------------------------------------------

def save_spei_profile(
    df: pd.DataFrame,
    out_prefix: str = "spei_biweekly",
    output_dir: str = "../../databases",
) -> Path:
    """
    Saves the SPEI series with a timestamp in the filename:
    {out_prefix}-vYYMMDDHHMMSS.csv (same pattern as the rest of the pipeline).

    Args:
        df: DataFrame from get_spei_biweekly()
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
    # Test point: Finca Matanza, Colombia
    Latitude, Longitude = 7.300921, -73.009794

    # Reference points for quick access (commented out):
    # El Playon         --||     7.4584221918243045,    -73.222052853104
    # Finca Matanza     --||     7.300921,              -73.009794
    # Sugarcane_COL     --||     3.580109040361371,     -76.31299479308868
    # Sugarcane_QLD     --||     -19.689669877950884,   147.22717515914223

    # Calculate SPEI from 2016 onwards
    df = get_spei_biweekly(Latitude, Longitude, start_date="2016-01-01")

    # Save the SPEI profile to CSV
    out_path = save_spei_profile(df)

    # Display the first 10 rows for inspection
    print(df.head(10))
