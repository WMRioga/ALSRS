"""
ONI / ENSO Profile Module
=========================

Provides the Oceanic Nino Index (ONI) as a time series for the ML dataset.

The ONI is the standard index for the El Nino-Southern Oscillation (ENSO).
It is the 3-month running mean of sea-surface-temperature (SST) anomalies in
the Nino 3.4 region (5N-5S, 120-170W) of the equatorial Pacific. It is
published monthly by the NOAA Climate Prediction Center (CPC), free of charge,
from 1950 to the present, with ~1 month of publication lag.

WHY THIS MATTERS FOR ALSRS
--------------------------
ENSO is one of the few climate signals that is predictable months ahead (the
ocean has huge thermal inertia), so it carries genuine information for the
1-6 month irrigation forecast horizon. It modulates rainfall in both ALSRS
regions:
    - Colombia (cacao / coffee / plantain): El Nino -> drier Caribbean/Andes,
      La Nina -> wetter.
    - Australia (wheat / sorghum / canola / sugarcane): El Nino -> drier east,
      La Nina -> wetter.

A single scalar (the ONI value) therefore helps forecast water deficit for
crops in both continents.

DATA SOURCE
-----------
    NOAA CPC: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
    Columns: SEAS (3-month season, e.g. DJF), YR, TOTAL, ANOM (the ONI value).

MAPPING TO MONTHLY
------------------
The ONI table is a 3-month running mean, so each row is a season (DJF, JFM,
...). We map each season to its MIDDLE month (DJF -> January, JFM -> February,
..., NDJ -> December), producing one value per calendar month. Each biweekly
period is then assigned the ONI of its month.

REFERENCES
----------
- NOAA Climate Prediction Center. Oceanic Nino Index (ONI).
  https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
- Trenberth, K.E. (1997). The definition of El Nino. Bulletin of the American
  Meteorological Society, 78(12), 2771-2777.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

# Make the project's databases directory importable / writable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATABASES_DIR = _PROJECT_ROOT / "databases"

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Local cache (a single global monthly series, not a per-point extraction).
CACHE_PATH = _DATABASES_DIR / "oni_monthly.csv"

# Map each 3-month NOAA season to its MIDDLE month (1 = January, ... 12).
SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


# ---------------------------------------------------------------------------
# Download / Build
# ---------------------------------------------------------------------------

def download_oni() -> pd.DataFrame:
    """
    Downloads the raw NOAA ONI table.

    Returns:
        pandas.DataFrame with columns ``SEAS``, ``YR``, ``TOTAL``, ``ANOM``.
    """
    request = urllib.request.Request(ONI_URL, headers={"User-Agent": "ALSRS/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = pd.read_csv(response, sep=r"\s+")
    return raw


def build_monthly_oni(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Maps the seasonal ONI table to one value per calendar month.

    Each season (e.g. ``DJF``) is assigned to its middle month, giving a
    monthly series of the ONI.

    Args:
        raw: DataFrame from :func:`download_oni`.

    Returns:
        pandas.DataFrame with columns ``year``, ``month``, ``oni``, sorted
        chronologically.
    """
    rows = []
    for _, r in raw.iterrows():
        season = str(r["SEAS"]).strip()
        month = SEASON_TO_MONTH[season]
        rows.append({
            "year": int(r["YR"]),
            "month": month,
            "oni": float(r["ANOM"]),
        })
    monthly = pd.DataFrame(rows).sort_values(["year", "month"])
    return monthly.reset_index(drop=True)


def get_monthly_oni(force_download: bool = False) -> pd.DataFrame:
    """
    Returns the monthly ONI series, using the local cache when available.

    Args:
        force_download: If True, re-download even when the cache exists.

    Returns:
        pandas.DataFrame with columns ``year``, ``month``, ``oni``.
    """
    if not force_download and CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH)

    raw = download_oni()
    monthly = build_monthly_oni(raw)
    _DATABASES_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(CACHE_PATH, index=False)
    print(f"ONI monthly series cached to {CACHE_PATH}")
    return monthly


# ---------------------------------------------------------------------------
# Dataset integration
# ---------------------------------------------------------------------------

def add_oni_column(
    df: pd.DataFrame, label_col: str = "label"
) -> pd.DataFrame:
    """
    Adds an ``oni`` column to a dataset by joining on year + month.

    The year and month are parsed from the biweekly ``label`` column
    (format ``YYYY-MM_Qn``), so each row gets the ONI of its month.

    Args:
        df: DataFrame with a biweekly ``label`` column (or ``label_col``).
        label_col: Name of the label column to parse.

    Returns:
        pandas.DataFrame with the ``oni`` column added.
    """
    monthly = get_monthly_oni()
    monthly = monthly.rename(columns={"year": "_oni_year", "month": "_oni_month"})

    out = df.copy()
    out["_oni_year"] = out[label_col].str[:4].astype(int)
    out["_oni_month"] = out[label_col].str[5:7].astype(int)
    out = out.merge(
        monthly,
        on=["_oni_year", "_oni_month"],
        how="left",
    )
    out = out.drop(columns=["_oni_year", "_oni_month"])
    return out


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    monthly = get_monthly_oni(force_download=True)
    print(monthly.head(12))
    print("...")
    print(monthly.tail(12))
