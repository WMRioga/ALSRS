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
    - Forecast targets (nested accumulated future deficits): for each biweek
      t, the deficit accumulated over the NEXT 1, 3 and 6 months:
          future_deficit_H(t) = sum(ETc - AET over [t+1 .. t+H]) /
                                sum(ETc over [t+1 .. t+H])
      with H = 2 (1m), 6 (3m) and 12 (6m) biweeks. These are genuinely
      forward-looking (no overlap with the features) and are the ML targets.
    - Irrigation suggestion = the WORST class among the three horizons (each
      accumulated deficit classified):
          <=0.15 -> LOW, <=0.30 -> MEDIUM, <=0.50 -> HIGH, >0.50 -> NOT_SUITABLE.

Output:
    databases/water_balance_labels_{crop}-vYYMMDDHHMMSS.csv with features
    (AWC_mm, spei_1m/3m/6m/12m, P_acum_mm, Storage_mm, WRSI, deficit_pct),
    forecast targets (future_deficit_1m/3m/6m and their mm counterparts) and
    the derived suggestion.

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

# Nested forecast horizons: (output suffix, number of future biweeks).
# 2 biweeks = 1 month, 6 = 3 months, 12 = 6 months. Each target is the deficit
# accumulated over that future window (fraction 0-1).
FORECAST_HORIZONS = [
    ("future_deficit_1m", 2),
    ("future_deficit_3m", 6),
    ("future_deficit_6m", 12),
]

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
        columns ``ETc_mm``, ``AET_mm``, ``Storage_mm``, ``P_acum_mm``, ``WRSI``
        and ``deficit_pct``.
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
    out["AET_mm"] = aet
    out["Storage_mm"] = storage
    out["P_acum_mm"] = p_acum
    out["WRSI"] = wrsi
    out["deficit_pct"] = deficit
    return out, window, storage_init


# Severity ranking of the irrigation-need classes (worst = highest).
_SUGGESTION_RANK = {LOW: 0, MEDIUM: 1, HIGH: 2, NOT_SUITABLE: 3}


def classify_suggestion(deficit_fraction: float) -> str:
    """
    Maps an accumulated water deficit (fraction 0-1) to an irrigation-need
    class.

    <=0.15 -> LOW, <=0.30 -> MEDIUM, <=0.50 -> HIGH, >0.50 -> NOT_SUITABLE.
    Returns an empty string when there is no future data (NaN).
    """
    if pd.isna(deficit_fraction):
        return ""
    if deficit_fraction <= 0.15:
        return LOW
    if deficit_fraction <= 0.30:
        return MEDIUM
    if deficit_fraction <= 0.50:
        return HIGH
    return NOT_SUITABLE


def worst_suggestion(*classes: str) -> str:
    """
    Returns the most severe irrigation-need class among the given ones.

    Severity order: LOW < MEDIUM < HIGH < NOT_SUITABLE. An empty string is
    ignored; if every class is empty, an empty string is returned.
    """
    worst = ""
    worst_rank = -1
    for cls in classes:
        rank = _SUGGESTION_RANK.get(cls, -1)
        if rank > worst_rank:
            worst_rank = rank
            worst = cls
    return worst


def _accumulate_future_deficit(
    etc: np.ndarray, aet: np.ndarray, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accumulates the water deficit over the next ``horizon`` biweeks, forward.

    For each period t, sums (ETc - AET) and ETc over [t+1 .. t+horizon]:

        fraction(t) = sum(ETc - AET) / sum(ETc)
        mm(t)       = sum(ETc - AET)

    Rows without ``horizon`` future biweeks (or with any NaN in the window)
    stay NaN.

    Args:
        etc: Crop evapotranspiration per period (mm).
        aet: Actual evapotranspiration per period (mm).
        horizon: Number of future biweeks to accumulate.

    Returns:
        tuple: (fraction, mm) arrays, same length as ``etc``.
    """
    n = len(etc)
    fraction = np.full(n, np.nan, dtype=float)
    mm = np.full(n, np.nan, dtype=float)

    for t in range(n):
        end = t + horizon
        if end >= n:
            break
        etc_win = etc[t + 1:end + 1]
        aet_win = aet[t + 1:end + 1]
        if np.isnan(etc_win).any() or np.isnan(aet_win).any():
            continue
        denom = float(np.sum(etc_win))
        numer = float(np.sum(etc_win - aet_win))
        if denom > 0:
            fraction[t] = numer / denom
            mm[t] = numer
    return fraction, mm


def build_labeled_dataset(
    out: pd.DataFrame, crop: str, awc_mm: float
) -> pd.DataFrame:
    """
    Builds the labeled frame: features at time t plus the nested future
    targets (future_deficit_1m/3m/6m as a fraction 0-1, plus their mm
    counterparts) and the irrigation suggestion = the WORST class among the
    three horizons. Keeps only rows whose features are complete AND that have
    a valid 6-month target (so the tail without enough future data is dropped).

    Args:
        out: DataFrame from compute_wrsi_series().
        crop: Crop identifier (added as a constant column).
        awc_mm: Total available water capacity (mm, constant column).

    Returns:
        pandas.DataFrame with one row per valid (fully-featured) period.
    """
    etc = out["ETc_mm"].to_numpy(dtype=float)
    aet = out["AET_mm"].to_numpy(dtype=float)

    # Nested accumulated future deficits (the ML targets).
    for name, horizon in FORECAST_HORIZONS:
        fraction, mm = _accumulate_future_deficit(etc, aet, horizon)
        out[name] = fraction
        out[f"{name}_mm"] = mm

    # Irrigation suggestion = worst class among the three horizons, derived
    # from the accumulated deficits (not predicted).
    horizon_class_cols = []
    for name, _ in FORECAST_HORIZONS:
        cls_col = f"_{name}_class"
        out[cls_col] = out[name].apply(classify_suggestion)
        horizon_class_cols.append(cls_col)
    out["suggestion"] = [
        worst_suggestion(*row) for row in out[horizon_class_cols].itertuples(
            index=False, name=None
        )
    ]

    # Constant identity columns.
    out["crop"] = crop
    out["AWC_mm"] = round(awc_mm, 2)

    # Keep only rows where every feature is defined and the 6-month target
    # exists (drops the tail with no future data).
    feature_cols = [
        "spei_1m", "spei_3m", "spei_6m", "spei_12m",
        "P_acum_mm", "Storage_mm", "WRSI",
    ]
    labeled = out.dropna(subset=feature_cols + ["future_deficit_6m"]).copy()

    # Column order for the CSV.
    output_cols = (
        ["crop", "period_start", "period_end", "label", "AWC_mm"]
        + ["spei_1m", "spei_3m", "spei_6m", "spei_12m"]
        + ["P_acum_mm", "Storage_mm", "WRSI", "deficit_pct"]
    )
    for name, _ in FORECAST_HORIZONS:
        output_cols += [name, f"{name}_mm"]
    output_cols += ["suggestion"]
    labeled = labeled[output_cols]

    # Round numeric columns for a clean CSV.
    for col in ["P_acum_mm", "Storage_mm", "AWC_mm", "deficit_pct"]:
        labeled[col] = labeled[col].round(2)
    for col in ["spei_1m", "spei_3m", "spei_6m", "spei_12m"]:
        labeled[col] = labeled[col].round(3)
    labeled["WRSI"] = labeled["WRSI"].round(4)
    for name, _ in FORECAST_HORIZONS:
        labeled[name] = labeled[name].round(4)
        labeled[f"{name}_mm"] = labeled[f"{name}_mm"].round(2)

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
        print(f"WRSI (current)  : min={labeled['WRSI'].min():.3f}, "
              f"mean={labeled['WRSI'].mean():.3f}, "
              f"max={labeled['WRSI'].max():.3f}")
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
