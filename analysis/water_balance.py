"""
Water Balance / WRSI Module
===========================

Computes the Water Requirement Satisfaction Index (WRSI) for a single crop
at a given location, using a sequential soil-water balance with memory and a
rolling evaluation window. This produces the LABELED dataset used later by
the machine-learning step to forecast whether an initially-apt crop will need
irrigation.

This module has two layers:

1. Pure core: functions that take pandas DataFrames / numpy arrays and return
   the WRSI series and the labeled output. No Earth Engine dependency.

2. Runner (``main``): optionally calls the existing GEE extraction modules
   (``precipitation_profile``, ``soil_profile_area``, ``spei_profile``) to
   (re)generate the input CSVs, then feeds them to the pure core.

Method (FAO bucket model, biweekly):
    - Crop water requirement per biweekly period is distributed by reference
      evapotranspiration ET0 (Thornthwaite ``pet_mm`` from the SPEI CSV):
          ETc(t) = water_requirement_mm * ET0(t) / sum(ET0 over the window)
      so that the total requirement over any evaluation window equals
      ``water_requirement_mm``. (Constant fallback before the window exists.)
    - Sequential balance with memory:
          AET(t)     = min(P(t) + Storage(t-1), ETc(t))
          Storage(t) = clip(P(t) + Storage(t-1) - AET(t), 0, AWC)
      Warm-up: the first 4 biweekly periods only initialize the storage
      (mean of P-ETc, clipped to [0, AWC]) and are excluded from the labels.
    - Rolling WRSI over the evaluation window W (24 for perennial, cycle
      duration for annual):
          WRSI(t) = sum(AET over window) / sum(ETc over window)
          deficit(t) = (1 - WRSI(t)) * 100
    - Future labels: WRSI(t+1) ... WRSI(t+12) and their deficits, plus a
      suggestion from the WORST deficit in the next 12 biweekly periods:
          0-15% -> LOW, 15-30% -> MEDIUM, 30-50% -> HIGH, >50% -> NOT_SUITABLE.

Output:
    databases/water_balance_labels_{crop}-vYYMMDDHHMMSS.csv with features
    (AWC_mm, spei_1m/3m/6m/12m, P_acum_mm, Storage_mm, WRSI_actual) and labels
    (WRSI_t1 ... WRSI_t12, deficit_t1 ... deficit_t12, suggestion).

Dependencies:
    - numpy, pandas
    - precipitation_profile / soil_profile_area / soil_hydraulics / spei_profile
      (custom modules, same directory - imported lazily inside main())
"""
from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Make the local modules importable regardless of the working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "common", _PROJECT_ROOT / "extraction",
           _PROJECT_ROOT / "analysis", _PROJECT_ROOT / "mapping"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = _SCRIPT_DIR.parent
DATABASES_DIR = PROJECT_ROOT / "databases"

CROP_PARAMS_FILENAME = "crop_parameters_260822.csv"
PRECIP_PREFIX = "precipitation_biweekly"
SOIL_HYDRAULIC_PREFIX = "soil_hydraulic_data"
SPEI_PREFIX = "spei_biweekly"

# Biweekly periods used for the storage warm-up (2 months).
WARMUP_PERIODS = 4

# Prediction horizon for the suggestion and future labels (always 6 months).
HORIZON = 12

# Suggestion classes (English).
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
NOT_SUITABLE = "NOT_SUITABLE"


# ---------------------------------------------------------------------------
# Loading Functions (CSV -> plain structures; no Earth Engine)
# ---------------------------------------------------------------------------

def latest_csv(prefix: str, directory: Path = DATABASES_DIR) -> Path:
    """
    Returns the most recent CSV whose filename starts with ``prefix-v``.

    Filenames use a ``YYMMDDHHMMSS`` timestamp, so lexicographic sorting of
    the timestamp portion equals chronological sorting; the last match is the
    newest extraction.

    Args:
        prefix: Base name of the dataset (e.g. ``"precipitation_biweekly"``).
        directory: Directory to search.

    Returns:
        pathlib.Path: Path to the most recent matching CSV.

    Raises:
        FileNotFoundError: If no CSV matches the prefix.
    """
    candidates = sorted(Path(directory).glob(f"{prefix}-v*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found with prefix '{prefix}' in {directory}"
        )
    return candidates[-1]


def load_crop_parameters(path: Path) -> pd.DataFrame:
    """Loads the crop parameters table, indexed by the ``crop`` identifier."""
    df = pd.read_csv(path)
    df["crop"] = df["crop"].astype(str)
    return df.set_index("crop")


def load_precipitation(path: Path) -> pd.DataFrame:
    """
    Loads a biweekly precipitation CSV, keeping the join/analysis columns.

    Returns a DataFrame with ``period_start``, ``period_end``, ``label`` and
    ``precip_total_mm``.
    """
    df = pd.read_csv(path)
    df["precip_total_mm"] = pd.to_numeric(df["precip_total_mm"], errors="coerce")
    return df[["period_start", "period_end", "label", "precip_total_mm"]]


def load_awc(path: Path) -> float:
    """
    Loads a soil-hydraulic CSV and returns the total available water capacity
    (AWC) of the 0-100 cm profile, in mm (sum of ``AWC_layer_mm``).
    """
    df = pd.read_csv(path)
    if "AWC_layer_mm" not in df.columns:
        raise ValueError(f"'AWC_layer_mm' not found in {path}")
    return float(pd.to_numeric(df["AWC_layer_mm"], errors="coerce").sum())


def load_spei(path: Path) -> pd.DataFrame:
    """
    Loads a biweekly SPEI CSV, keeping the join/analysis columns.

    Returns a DataFrame with ``label``, ``pet_mm`` (reference ET0) and
    ``spei_1m``/``spei_3m``/``spei_6m``/``spei_12m``.
    """
    df = pd.read_csv(path)
    for col in ["pet_mm", "spei_1m", "spei_3m", "spei_6m", "spei_12m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["label", "pet_mm", "spei_1m", "spei_3m", "spei_6m", "spei_12m"]]


# ---------------------------------------------------------------------------
# Core: WRSI computation
# ---------------------------------------------------------------------------

def compute_etc(
    pet_mm: np.ndarray, water_requirement_mm: float, window: int
) -> np.ndarray:
    """
    Distributes the crop water requirement across biweekly periods using ET0.

    ETc(t) = water_requirement_mm * ET0(t) / rolling_sum(ET0, window)(t), so
    the total ETc over any full window equals ``water_requirement_mm``. Before
    the first window is complete, a constant ETc (requirement / window) is
    used; those periods only feed the storage warm-up and never enter the
    labeled dataset.

    Args:
        pet_mm: Reference evapotranspiration ET0 per period (mm).
        water_requirement_mm: Total crop water requirement per cycle (mm).
        window: Evaluation window in biweekly periods.

    Returns:
        numpy.ndarray: ETc per period (mm), same length as ``pet_mm``.
    """
    n = len(pet_mm)
    const = water_requirement_mm / window
    rolling_sum = (
        pd.Series(pet_mm).rolling(window, min_periods=window).sum().to_numpy()
    )

    etc = np.full(n, const, dtype=float)
    for t in range(window - 1, n):
        total = rolling_sum[t]
        if not np.isnan(total) and total > 0 and not np.isnan(pet_mm[t]):
            etc[t] = water_requirement_mm * pet_mm[t] / total
    return etc


def compute_water_balance(
    precip_mm: np.ndarray, etc_mm: np.ndarray, awc_mm: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Runs the sequential soil-water balance with memory (bucket model).

    Warm-up: the first 4 periods only initialize the storage as
    ``clip(mean(P - ETc), 0, AWC)`` and are excluded (their Storage/AET stay
    NaN). From period 4 onward:
        AET(t)     = min(P(t) + Storage(t-1), ETc(t))
        Storage(t) = clip(P(t) + Storage(t-1) - AET(t), 0, AWC)

    Args:
        precip_mm: Precipitation per period (mm).
        etc_mm: Crop water requirement per period (mm).
        awc_mm: Total available water capacity (mm).

    Returns:
        tuple: (storage_mm, aet_mm, storage_init) where storage/aet are
        arrays with NaN for the warm-up periods and ``storage_init`` is the
        warm-up storage value (mm).
    """
    n = len(precip_mm)
    storage = np.full(n, np.nan, dtype=float)
    aet = np.full(n, np.nan, dtype=float)

    warm = precip_mm[:WARMUP_PERIODS] - etc_mm[:WARMUP_PERIODS]
    storage_init = float(np.clip(np.nanmean(warm), 0.0, awc_mm))

    storage_prev = storage_init
    for t in range(WARMUP_PERIODS, n):
        available = precip_mm[t] + storage_prev
        aet[t] = min(available, etc_mm[t])
        storage[t] = min(available - aet[t], awc_mm)
        storage_prev = storage[t]

    return storage, aet, storage_init


def compute_wrsi_series(
    df: pd.DataFrame,
    water_requirement_mm: float,
    cycle_quincenas: int,
    crop_type: str,
    awc_mm: float,
) -> Tuple[pd.DataFrame, int, float]:
    """
    Computes ETc, the sequential balance, and the rolling WRSI for a crop.

    Args:
        df: Merged frame (precipitation + SPEI) with ``precip_total_mm`` and
            ``pet_mm``, sorted chronologically by ``label``.
        water_requirement_mm: Total crop water requirement per cycle (mm).
        cycle_quincenas: Crop cycle length in biweekly periods.
        crop_type: ``"perennial"`` or ``"annual"``.
        awc_mm: Total available water capacity (mm).

    Returns:
        tuple: (out_df, window, storage_init) where ``out_df`` is ``df`` plus
        columns ``ETc_mm``, ``Storage_mm``, ``P_acum_mm``, ``WRSI_actual`` and
        ``deficit_pct``.
    """
    window = 24 if crop_type == "perennial" else int(cycle_quincenas)

    precip = df["precip_total_mm"].to_numpy(dtype=float)
    pet = df["pet_mm"].to_numpy(dtype=float)
    n = len(df)

    etc = compute_etc(pet, water_requirement_mm, window)
    storage, aet, storage_init = compute_water_balance(precip, etc, awc_mm)

    # Rolling WRSI and accumulated precipitation over the window.
    wrsi = np.full(n, np.nan, dtype=float)
    p_acum = np.full(n, np.nan, dtype=float)
    for t in range(window - 1, n):
        s = slice(t - window + 1, t + 1)
        etc_sum = float(np.nansum(etc[s]))
        p_acum[t] = float(np.nansum(precip[s]))
        if etc_sum > 0:
            wrsi[t] = float(np.nansum(aet[s])) / etc_sum

    deficit = np.where(np.isnan(wrsi), np.nan, (1.0 - wrsi) * 100.0)

    out = df.copy()
    out["ETc_mm"] = etc
    out["Storage_mm"] = storage
    out["P_acum_mm"] = p_acum
    out["WRSI_actual"] = wrsi
    out["deficit_pct"] = deficit
    return out, window, storage_init


def classify_suggestion(worst_deficit: float) -> str:
    """
    Maps the worst future deficit to an irrigation suggestion class.

    0-15% -> LOW, 15-30% -> MEDIUM, 30-50% -> HIGH, >50% -> NOT_SUITABLE.
    Returns an empty string when there is no future data (NaN).
    """
    if pd.isna(worst_deficit):
        return ""
    if worst_deficit <= 15.0:
        return LOW
    if worst_deficit <= 30.0:
        return MEDIUM
    if worst_deficit <= 50.0:
        return HIGH
    return NOT_SUITABLE


def build_labeled_dataset(
    out: pd.DataFrame, crop: str, awc_mm: float
) -> pd.DataFrame:
    """
    Builds the final labeled frame: features at time t plus the future labels
    (WRSI_t1..t12, deficit_t1..t12) and the suggestion, then keeps only rows
    whose features are complete (no NaN).

    Args:
        out: DataFrame from compute_wrsi_series().
        crop: Crop identifier (added as a constant column).
        awc_mm: Total available water capacity (mm, constant column).

    Returns:
        pandas.DataFrame with one row per valid (fully-featured) period.
    """
    wrsi = out["WRSI_actual"].to_numpy(dtype=float)
    deficit = out["deficit_pct"].to_numpy(dtype=float)
    n = len(out)

    # Future labels: value at t+k.
    for k in range(1, HORIZON + 1):
        shifted_w = np.full(n, np.nan, dtype=float)
        shifted_d = np.full(n, np.nan, dtype=float)
        if k < n:
            shifted_w[:-k] = wrsi[k:]
            shifted_d[:-k] = deficit[k:]
        out[f"WRSI_t{k}"] = shifted_w
        out[f"deficit_t{k}"] = shifted_d

    deficit_cols = [f"deficit_t{k}" for k in range(1, HORIZON + 1)]
    out["suggestion"] = (
        out[deficit_cols].max(axis=1, skipna=True).apply(classify_suggestion)
    )

    # Constant identity columns.
    out["crop"] = crop
    out["AWC_mm"] = round(awc_mm, 2)

    # Keep only rows where every feature is defined.
    feature_cols = [
        "spei_1m", "spei_3m", "spei_6m", "spei_12m",
        "P_acum_mm", "Storage_mm", "WRSI_actual",
    ]
    labeled = out.dropna(subset=feature_cols).copy()

    # Column order for the CSV.
    output_cols = (
        ["crop", "period_start", "period_end", "label", "AWC_mm"]
        + ["spei_1m", "spei_3m", "spei_6m", "spei_12m"]
        + ["P_acum_mm", "Storage_mm", "WRSI_actual", "deficit_pct"]
        + [f"WRSI_t{k}" for k in range(1, HORIZON + 1)]
        + [f"deficit_t{k}" for k in range(1, HORIZON + 1)]
        + ["suggestion"]
    )
    labeled = labeled[output_cols]

    # Round numeric columns for a clean CSV.
    for col in ["P_acum_mm", "Storage_mm", "AWC_mm", "deficit_pct"]:
        labeled[col] = labeled[col].round(2)
    for col in ["spei_1m", "spei_3m", "spei_6m", "spei_12m"]:
        labeled[col] = labeled[col].round(3)
    for col in ["WRSI_actual"] + [f"WRSI_t{k}" for k in range(1, HORIZON + 1)]:
        labeled[col] = labeled[col].round(4)
    for col in [f"deficit_t{k}" for k in range(1, HORIZON + 1)]:
        labeled[col] = labeled[col].round(2)

    return labeled


# ---------------------------------------------------------------------------
# Saving / Reporting
# ---------------------------------------------------------------------------

def save_water_balance_labels(
    df: pd.DataFrame, crop: str, output_dir: Path = DATABASES_DIR
) -> Path:
    """
    Saves the labeled WRSI dataset to a timestamped, per-crop CSV.

    Args:
        df: DataFrame from build_labeled_dataset().
        crop: Crop identifier (embedded in the filename).
        output_dir: Directory where the CSV is saved (created if missing).

    Returns:
        pathlib.Path: Path to the saved CSV.
    """
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"water_balance_labels_{crop}-v{timestamp}.csv"
    df.to_csv(out_path, index=False)
    print(f"CSV saved to {out_path} ({df.shape[0]} rows x {df.shape[1]} cols)")
    return out_path


def print_report(
    crop: str,
    params: pd.Series,
    awc_mm: float,
    window: int,
    storage_init: float,
    labeled: pd.DataFrame,
    lat: float,
    lon: float,
) -> None:
    """
    Prints a concise WRSI summary to the console.

    Args:
        crop: Crop identifier.
        params: Crop parameter row.
        awc_mm: Total available water capacity (mm).
        window: Evaluation window in biweekly periods.
        storage_init: Warm-up storage value (mm).
        labeled: Labeled output frame.
        lat: Latitude of the evaluated point.
        lon: Longitude of the evaluated point.
    """
    print("\n" + "=" * 78)
    print("WATER BALANCE / WRSI REPORT")
    print("=" * 78)
    print(f"Location       : lat={lat}, lon={lon}")
    print(f"Crop           : {crop} ({params['common_name']}) [{params['type']}]")
    print(f"Water requirement: {params['water_requirement_mm']} mm/cycle")
    print(f"Cycle          : {params['cycle_quincenas']} quincenas")
    print(f"Evaluation window: {window} quincenas")
    print(f"AWC (0-100 cm) : {awc_mm:.2f} mm")
    print(f"Warm-up storage: {storage_init:.2f} mm")
    print(f"Labeled periods: {len(labeled)}")
    if len(labeled):
        print(f"Date range     : {labeled['label'].iloc[0]} -> "
              f"{labeled['label'].iloc[-1]}")
        print(f"WRSI (actual)  : min={labeled['WRSI_actual'].min():.3f}, "
              f"mean={labeled['WRSI_actual'].mean():.3f}, "
              f"max={labeled['WRSI_actual'].max():.3f}")
    print("-" * 78)
    if len(labeled):
        print("Suggestion distribution:")
        counts = labeled["suggestion"].replace("", "N/A").value_counts()
        for cls in [LOW, MEDIUM, HIGH, NOT_SUITABLE, "N/A"]:
            if cls in counts:
                print(f"  {cls:<14} {counts[cls]}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Runner (regenerates inputs via the GEE modules, then computes WRSI)
# ---------------------------------------------------------------------------

def regenerate_inputs(
    lat: float, lon: float, output_dir: Path = DATABASES_DIR
) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """
    (Re)generates precipitation, soil-hydraulic and SPEI CSVs by calling the
    existing GEE extraction modules, and returns the same structures the pure
    core consumes.

    The Earth Engine modules are imported lazily here so the pure core never
    needs the ``earthengine-api`` dependency.

    Args:
        lat: Latitude of the point.
        lon: Longitude of the point.
        output_dir: Directory where the intermediate CSVs are saved.

    Returns:
        tuple: (precip_df, awc_mm, spei_df).
    """
    from precipitation_profile import (
        get_precipitation_biweekly,
        save_precipitation_profile,
    )
    from soil_hydraulics import calculate_hydraulic_properties
    from soil_profile_area import get_soil_profile_area, save_hydraulic_profile
    from spei_profile import get_spei_biweekly, save_spei_profile

    # --- Precipitation (CHIRPS) ---
    precip_df = get_precipitation_biweekly(lat, lon, start_date="2016-01-01")
    save_precipitation_profile(precip_df, output_dir=str(output_dir))

    # --- Soil hydraulic properties (SoilGrids -> AWC) ---
    soil_dict = get_soil_profile_area(lat, lon, 70.0)
    df_hydric = calculate_hydraulic_properties(soil_dict)
    save_hydraulic_profile(df_hydric, output_dir=str(output_dir))
    awc_mm = float(df_hydric["AWC_layer_mm"].sum())

    # --- SPEI (CHIRPS + ERA5-Land) ---
    spei_df = get_spei_biweekly(lat, lon, start_date="2016-01-01")
    save_spei_profile(spei_df, output_dir=str(output_dir))

    return precip_df, awc_mm, spei_df


def main(
    lat: float,
    lon: float,
    crop: str,
    regenerate: bool = True,
    crop_params_path: Optional[Path] = None,
    output_dir: Path = DATABASES_DIR,
) -> Path:
    """
    Runs the WRSI computation for ONE crop and a point.

    Args:
        lat: Latitude of the point.
        lon: Longitude of the point.
        crop: Crop identifier (must exist in crop_parameters).
        regenerate: If True, call the GEE modules to (re)generate the input
                    CSVs; if False, read the latest local CSVs instead.
        crop_params_path: Path to the crop parameters CSV (defaults to the
                          file under ``databases``).
        output_dir: Directory for the output CSV.

    Returns:
        pathlib.Path: Path to the water_balance_labels CSV.

    Raises:
        ValueError: If ``crop`` is not present in the parameters table.
    """
    if crop_params_path is None:
        crop_params_path = DATABASES_DIR / CROP_PARAMS_FILENAME

    # 1. Inputs.
    if regenerate:
        precip_df, awc_mm, spei_df = regenerate_inputs(lat, lon, output_dir)
    else:
        precip_df = load_precipitation(latest_csv(PRECIP_PREFIX, output_dir))
        awc_mm = load_awc(latest_csv(SOIL_HYDRAULIC_PREFIX, output_dir))
        spei_df = load_spei(latest_csv(SPEI_PREFIX, output_dir))

    # 2. Crop parameters; validate the requested crop.
    crop_params = load_crop_parameters(crop_params_path)
    if crop not in crop_params.index:
        valid = ", ".join(crop_params.index.tolist())
        raise ValueError(f"Unknown crop '{crop}'. Valid crops: {valid}")
    params = crop_params.loc[crop]

    # 3. Merge precipitation and SPEI on the biweekly label.
    df = precip_df.merge(
        spei_df[["label", "pet_mm", "spei_1m", "spei_3m", "spei_6m", "spei_12m"]],
        on="label",
        how="inner",
    )
    df = df.sort_values("label").reset_index(drop=True)
    # Drop the trailing incomplete period(s) (e.g. CHIRPS publication lag).
    df = df.dropna(subset=["precip_total_mm"]).reset_index(drop=True)

    # 4. WRSI computation.
    out, window, storage_init = compute_wrsi_series(
        df,
        float(params["water_requirement_mm"]),
        int(params["cycle_quincenas"]),
        str(params["type"]),
        awc_mm,
    )

    # 5. Labeled dataset.
    labeled = build_labeled_dataset(out, crop, awc_mm)

    # 6. Save + report.
    out_path = save_water_balance_labels(labeled, crop, output_dir=output_dir)
    print_report(crop, params, awc_mm, window, storage_init, labeled, lat, lon)

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

    # Default demo: sugarcane at the Queensland, Australia point.
    LAT = 7.4584221918243045
    LON = -73.222052853104
    CROP = "cacao_ccn51"

    main(LAT, LON, CROP, regenerate=True)
