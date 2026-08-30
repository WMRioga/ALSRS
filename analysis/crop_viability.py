"""
Crop Physical Viability Filter Module (Level 1)
================================================

Validates the BASIC physical viability of a single crop at a given location,
using locally-generated feature data (no direct Earth Engine calls in the
core logic). This is the "Level 1" gate of the land-suitability pipeline:
it checks the static site characteristics (elevation, slope, soil texture,
pH, organic carbon) and the biweekly temperature regime against the crop's
minimum requirements.

The module has two layers:

1. Pure core: functions that take pandas DataFrames / dicts and return the
   per-filter status. These never touch Earth Engine and can be unit-tested
   offline against existing CSVs.

2. Runner (``main``): calls the existing GEE extraction modules
   (``soil_profile_area``, ``terrain_profile``, ``temperature_profile``) to
   (re)generate the input CSVs with fresh, consistent English columns, then
   feeds them to the pure core.

Evaluation rules (see README_crops.md):
    - Static (evaluated once per location):
        * Elevation : elevation_m vs [elev_min_m, elev_max_m].
        * Slope     : slope_deg and slope_deg + slope_std_deg vs slope_max_deg.
        * Texture   : depth-weighted sand/clay 0-60 cm vs universal limits.
        * pH        : depth-weighted pH 0-60 cm vs [ph_min, ph_max] (+/- 1.0).
        * SOC       : depth-weighted organic carbon 0-60 cm vs soc_min_pct.
    - Temperature (evaluated PER biweekly period, because it varies
      biweekly): each period's mean_C is compared against the crop's four
      thresholds. Non-aptitude "peaks" (too cold / too hot periods) are
      counted.

Output:
    databases/crop_viability_{crop}-static-vYYMMDDHHMMSS.csv (one row: crop
    identity + one status/detail pair per static filter) and
    databases/temperature_biweekly-vYYMMDDHHMMSS.csv (raw temperature enriched
    with crop, temp_status, temp_detail, temp_score), plus a human-readable
    console report listing the static parameters that fail, the number of
    temperature non-aptitude peaks and the final viability.

Dependencies:
    - pandas
    - soil_hydraulics (custom module, same directory - conversion constants)
    - soil_profile_area / terrain_profile / temperature_profile
      (custom modules, same directory - imported lazily inside main())
"""
from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Make the local modules importable regardless of the working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _PROJECT_ROOT / "common", _PROJECT_ROOT / "extraction",
           _PROJECT_ROOT / "analysis", _PROJECT_ROOT / "mapping"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# soil_hydraulics has no Earth Engine dependency (pandas only), so it is safe
# to import at module level. It provides the SoilGrids unit-conversion factors
# and the standard layer-thickness definitions used across the pipeline.
from soil_hydraulics import LAYER_THICKNESS_CM, SOILGRIDS_CONVERSION


# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

# Project folders (relative to this file).
PROJECT_ROOT = _SCRIPT_DIR.parent
DATABASES_DIR = PROJECT_ROOT / "databases"

# SoilGrids stores pH in water as pH*10 (integer); divide to get pH units.
PHH2O_TO_PH = 10.0

# Only the 0-60 cm layers are used for the depth-weighted texture/pH/SOC means
# (the 60-100 cm layer is excluded per the approved rules).
DEPTHS_0_60 = ["0-5cm", "5-15cm", "15-30cm", "30-60cm"]

# Status vocabulary (English, consistent across the pipeline).
VIABLE = "VIABLE"
VIABLE_WITH_RESERVE = "VIABLE_WITH_RESERVE"
VIABLE_WITH_ADVISORY = "VIABLE_WITH_ADVISORY"
NOT_RECOMMENDED = "NOT_RECOMMENDED"
MISSING = "MISSING"

# Input prefixes / filenames.
CROP_PARAMS_FILENAME = "crop_parameters_260822.csv"
SOIL_PREFIX = "soil_profile_data"
TERRAIN_PREFIX = "terrain_profile_data"
TEMPERATURE_PREFIX = "temperature_biweekly"


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
        prefix: Base name of the dataset (e.g. ``"soil_profile_data"``).
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
    """
    Loads the crop parameters table and indexes it by the ``crop`` identifier.

    Args:
        path: Path to ``crop_parameters*.csv``.

    Returns:
        pandas.DataFrame indexed by ``crop``.
    """
    df = pd.read_csv(path)
    df["crop"] = df["crop"].astype(str)
    return df.set_index("crop")


def load_soil_profile(path: Path) -> pd.DataFrame:
    """
    Loads a raw SoilGrids soil-profile CSV (property x depth, raw units).

    Values are left in their raw SoilGrids units; conversion to %/pH happens
    in :func:`soil_aggregates`.

    Args:
        path: Path to ``soil_profile_data*.csv``.

    Returns:
        pandas.DataFrame indexed by property (``phh2o``, ``soc``, ``clay``,
        ``sand``, ...) with depth columns.
    """
    df = pd.read_csv(path, index_col=0)
    return df.apply(pd.to_numeric, errors="coerce")


def load_terrain(path: Path) -> Dict[str, float]:
    """
    Loads a terrain-profile CSV (single row) into a small dict.

    Only the fields used by the viability filter are kept; any extra columns
    (e.g. ``lat``/``lon`` in older files) are ignored.

    Args:
        path: Path to ``terrain_profile_data*.csv``.

    Returns:
        dict with keys ``elevation_m``, ``slope_deg``, ``slope_std_deg``.

    Raises:
        ValueError: If the CSV is empty.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Terrain CSV is empty: {path}")
    row = df.iloc[0]
    return {
        "elevation_m": float(row["elevation_m"]),
        "slope_deg": float(row["slope_deg"]),
        "slope_std_deg": float(row["slope_std_deg"]),
    }


def load_temperature(path: Path) -> pd.DataFrame:
    """
    Loads a biweekly temperature CSV and keeps the analysis columns.

    Args:
        path: Path to ``temperature_biweekly*.csv``.

    Returns:
        pandas.DataFrame with columns ``period_start``, ``period_end``,
        ``label`` and ``mean_C``.
    """
    df = pd.read_csv(path)
    df["mean_C"] = pd.to_numeric(df["mean_C"], errors="coerce")
    cols = ["period_start", "period_end", "label", "mean_C"]
    for extra in ["std_C", "var_C"]:
        if extra in df.columns:
            cols.append(extra)
    return df[cols]


# ---------------------------------------------------------------------------
# Aggregation Helpers
# ---------------------------------------------------------------------------

def _scaled(raw: float, factor: float) -> float:
    """Divides ``raw`` by ``factor``, returning NaN for missing input."""
    if raw is None or pd.isna(raw):
        return float("nan")
    return round(float(raw) / factor, 2)


def weighted_mean_0_60(values: pd.Series) -> float:
    """
    Depth-weighted mean over the 0-60 cm layers (5/10/15/30 cm thickness).

    Args:
        values: Series indexed by depth interval (e.g. from a soil profile).

    Returns:
        float: Depth-weighted mean, or NaN if no valid layer value exists.
    """
    total_weight = 0.0
    total_value = 0.0
    for depth in DEPTHS_0_60:
        v = values.get(depth, None) if hasattr(values, "get") else None
        if v is None or pd.isna(v):
            continue
        weight = LAYER_THICKNESS_CM[depth]
        total_value += float(v) * weight
        total_weight += weight
    if total_weight == 0:
        return float("nan")
    return total_value / total_weight


def soil_aggregates(soil_df: pd.DataFrame) -> Dict[str, float]:
    """
    Converts the raw 0-60 cm soil properties to conventional units.

    SoilGrids units -> conventional units (see soil_hydraulics.py):
        - sand / clay : g/kg   -> %       (divide by 10)
        - soc         : dg/kg  -> %       (divide by 100)
        - phh2o       : pH*10  -> pH      (divide by 10)

    Args:
        soil_df: DataFrame indexed by property with depth columns (raw units).

    Returns:
        dict with keys ``sand_0_60_pct``, ``clay_0_60_pct``,
        ``soc_0_60_pct`` and ``ph_0_60``.
    """
    return {
        "sand_0_60_pct": _scaled(
            weighted_mean_0_60(soil_df.loc["sand"]), SOILGRIDS_CONVERSION["sand"]
        ),
        "clay_0_60_pct": _scaled(
            weighted_mean_0_60(soil_df.loc["clay"]), SOILGRIDS_CONVERSION["clay"]
        ),
        "soc_0_60_pct": _scaled(
            weighted_mean_0_60(soil_df.loc["soc"]), SOILGRIDS_CONVERSION["soc"]
        ),
        "ph_0_60": _scaled(weighted_mean_0_60(soil_df.loc["phh2o"]), PHH2O_TO_PH),
    }


# ---------------------------------------------------------------------------
# Static Evaluators (each returns status + human-readable detail)
# ---------------------------------------------------------------------------

def evaluate_elevation(elevation_m: float, params: pd.Series) -> Tuple[str, str]:
    """Evaluates elevation against [elev_min_m, elev_max_m]."""
    lo = float(params["elev_min_m"])
    hi = float(params["elev_max_m"])
    if elevation_m < lo:
        return NOT_RECOMMENDED, (
            f"elevation too low ({elevation_m:.1f} m < {lo:.0f} m)"
        )
    if elevation_m > hi:
        return NOT_RECOMMENDED, (
            f"elevation too high ({elevation_m:.1f} m > {hi:.0f} m)"
        )
    return VIABLE, f"elevation within range [{lo:.0f}, {hi:.0f}] m"


def evaluate_slope(
    slope_deg: float, slope_std_deg: float, params: pd.Series
) -> Tuple[str, str]:
    """Evaluates mean slope and its spread against slope_max_deg."""
    max_deg = float(params["slope_max_deg"])
    if slope_deg > max_deg:
        return NOT_RECOMMENDED, (
            f"slope exceeds crop limit ({slope_deg:.1f} deg > {max_deg:.0f} deg)"
        )
    if slope_deg + slope_std_deg > max_deg:
        return VIABLE_WITH_ADVISORY, (
            "some zones exceed slope limit, validate in field"
        )
    return VIABLE, "slope within limit"


def evaluate_texture(
    sand_pct: float, clay_pct: float, params: pd.Series
) -> Tuple[str, str]:
    """Evaluates sand/clay against the universal physical limits."""
    sand_max = float(params["sand_max_pct"])
    clay_max = float(params["clay_max_pct"])
    if sand_pct > sand_max:
        return NOT_RECOMMENDED, (
            f"soil too sandy ({sand_pct:.1f}% > {sand_max:.0f}%), "
            "does not retain water"
        )
    if clay_pct > clay_max:
        return NOT_RECOMMENDED, (
            f"soil too clayey ({clay_pct:.1f}% > {clay_max:.0f}%), "
            "poor drainage and root penetration"
        )
    return VIABLE, "texture within limits"


def evaluate_ph(ph: float, params: pd.Series) -> Tuple[str, str]:
    """Evaluates pH against [ph_min, ph_max] with a +/- 1.0 tolerance band."""
    lo = float(params["ph_min"])
    hi = float(params["ph_max"])
    if lo <= ph <= hi:
        return VIABLE, f"optimal pH range [{lo:.1f}, {hi:.1f}]"

    deviation = (lo - ph) if ph < lo else (ph - hi)
    if deviation <= 1.0:
        return VIABLE_WITH_ADVISORY, "requires pH correction with lime/sulfur"
    return VIABLE_WITH_ADVISORY, (
        "pH correction is costly; validate resources for amendments"
    )


def evaluate_soc(soc_pct: float, params: pd.Series) -> Tuple[str, str]:
    """Evaluates soil organic carbon against soc_min_pct."""
    min_soc = float(params["soc_min_pct"])
    if soc_pct < min_soc:
        return VIABLE_WITH_ADVISORY, (
            "soil poor in organic matter, requires amendments"
        )
    return VIABLE, "organic matter sufficient"


def evaluate_static(
    soil_agg: Dict[str, float], terrain: Dict[str, float], params: pd.Series
) -> Dict[str, Dict[str, str]]:
    """
    Evaluates every static filter once and returns a nested results dict.

    Args:
        soil_agg: Aggregated/converted soil properties (see soil_aggregates).
        terrain: Terrain dict (see load_terrain).
        params: Crop parameter row.

    Returns:
        dict keyed by filter name, each value being
        ``{"status": ..., "detail": ...}``.
    """
    elev_status, elev_detail = evaluate_elevation(terrain["elevation_m"], params)
    slope_status, slope_detail = evaluate_slope(
        terrain["slope_deg"], terrain["slope_std_deg"], params
    )
    text_status, text_detail = evaluate_texture(
        soil_agg["sand_0_60_pct"], soil_agg["clay_0_60_pct"], params
    )
    ph_status, ph_detail = evaluate_ph(soil_agg["ph_0_60"], params)
    soc_status, soc_detail = evaluate_soc(soil_agg["soc_0_60_pct"], params)

    return {
        "elevation": {"status": elev_status, "detail": elev_detail},
        "slope": {"status": slope_status, "detail": slope_detail},
        "texture": {"status": text_status, "detail": text_detail},
        "ph": {"status": ph_status, "detail": ph_detail},
        "soc": {"status": soc_status, "detail": soc_detail},
    }


# ---------------------------------------------------------------------------
# Temperature Evaluator (per biweekly period)
# ---------------------------------------------------------------------------

def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership score in [0, 1] for a temperature value.

    0 below ``a``, linear ramp 0->1 on ``[a, b]``, 1 on ``[b, c]``,
    linear ramp 1->0 on ``[c, d]``, 0 above ``d``.
    """
    if pd.isna(x):
        return float("nan")
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    if x <= c:
        return 1.0
    return (d - x) / (d - c) if d > c else 1.0


def classify_temperature(
    mean_c: float, params: pd.Series
) -> Tuple[str, str, str]:
    """
    Classifies a single biweekly mean temperature against the crop thresholds.

    Args:
        mean_c: Biweekly mean temperature (°C).
        params: Crop parameter row.

    Returns:
        tuple: (bucket, status, detail) where bucket is one of ``OPTIMAL``,
        ``COOL``, ``WARM``, ``TOO_COLD``, ``TOO_HOT``, ``MISSING``.
    """
    tmin_tol = float(params["temp_min_tolerable_c"])
    tmin_opt = float(params["temp_opt_min_c"])
    tmax_opt = float(params["temp_opt_max_c"])
    tmax_tol = float(params["temp_max_tolerable_c"])

    if pd.isna(mean_c):
        return "MISSING", MISSING, "no temperature data"

    if mean_c < tmin_tol:
        return "TOO_COLD", NOT_RECOMMENDED, "too cold for crop"
    if mean_c > tmax_tol:
        return "TOO_HOT", NOT_RECOMMENDED, "too hot for crop"
    if mean_c < tmin_opt:
        return "COOL", VIABLE_WITH_RESERVE, "cool for crop"
    if mean_c > tmax_opt:
        return "WARM", VIABLE_WITH_RESERVE, "warm for crop"
    return "OPTIMAL", VIABLE, "optimal temperature range"


def evaluate_temperature_series(
    temp_df: pd.DataFrame, params: pd.Series
) -> pd.DataFrame:
    """
    Evaluates the temperature for every biweekly period of the series.

    Args:
        temp_df: DataFrame with ``period_start``, ``period_end``, ``label``,
                 ``mean_C``.
        params: Crop parameter row.

    Returns:
        pandas.DataFrame: a copy of ``temp_df`` with added ``temp_bucket``,
        ``temp_status``, ``temp_detail`` and ``temp_score`` columns (one row
        per period).
    """
    out = temp_df.copy()
    tmin_tol = float(params["temp_min_tolerable_c"])
    tmin_opt = float(params["temp_opt_min_c"])
    tmax_opt = float(params["temp_opt_max_c"])
    tmax_tol = float(params["temp_max_tolerable_c"])

    buckets, statuses, details, scores = [], [], [], []
    for mean_c in out["mean_C"]:
        bucket, status, detail = classify_temperature(mean_c, params)
        buckets.append(bucket)
        statuses.append(status)
        details.append(detail)
        scores.append(_trapezoid(mean_c, tmin_tol, tmin_opt, tmax_opt, tmax_tol))

    out["temp_bucket"] = buckets
    out["temp_status"] = statuses
    out["temp_detail"] = details
    out["temp_score"] = scores
    return out


def summarize_temperature(series_df: pd.DataFrame) -> Dict[str, object]:
    """
    Summarizes the per-period temperature results.

    Args:
        series_df: DataFrame returned by evaluate_temperature_series().

    Returns:
        dict with counts (``n_optimal``, ``n_cool``, ``n_warm``, ``n_cold``,
        ``n_hot``, ``n_missing``), ``peaks`` (cold + hot) and the final
        ``status`` (NOT_RECOMMENDED if any peak, else VIABLE_WITH_RESERVE if
        any cool/warm, else VIABLE).
    """
    counts = series_df["temp_bucket"].value_counts().to_dict()
    n_optimal = int(counts.get("OPTIMAL", 0))
    n_cool = int(counts.get("COOL", 0))
    n_warm = int(counts.get("WARM", 0))
    n_cold = int(counts.get("TOO_COLD", 0))
    n_hot = int(counts.get("TOO_HOT", 0))
    n_missing = int(counts.get("MISSING", 0))

    peaks = n_cold + n_hot
    if peaks > 0:
        status = NOT_RECOMMENDED
    elif (n_cool + n_warm) > 0:
        status = VIABLE_WITH_RESERVE
    else:
        status = VIABLE

    return {
        "n_optimal": n_optimal,
        "n_cool": n_cool,
        "n_warm": n_warm,
        "n_cold": n_cold,
        "n_hot": n_hot,
        "n_missing": n_missing,
        "peaks": peaks,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Final Verdict + Frame Assembly
# ---------------------------------------------------------------------------

def compute_final_viability(
    static: Dict[str, Dict[str, str]], temp_status: str
) -> str:
    """
    Combines the static and temperature statuses into a single verdict.

    Severity order: NOT_RECOMMENDED > VIABLE_WITH_RESERVE >
    VIABLE_WITH_ADVISORY > VIABLE.

    Args:
        static: Nested static-results dict (see evaluate_static).
        temp_status: Final temperature status (see summarize_temperature).

    Returns:
        str: The final viability level.
    """
    statuses = [item["status"] for item in static.values()] + [temp_status]
    if NOT_RECOMMENDED in statuses:
        return NOT_RECOMMENDED
    if VIABLE_WITH_RESERVE in statuses:
        return VIABLE_WITH_RESERVE
    if VIABLE_WITH_ADVISORY in statuses:
        return VIABLE_WITH_ADVISORY
    return VIABLE


def build_static_frame(
    params: pd.Series,
    soil_agg: Dict[str, float],
    terrain: Dict[str, float],
    static: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """
    Builds the single-row static output frame: crop identity plus one
    status/detail pair per static filter and the measured values.

    Args:
        params: Crop parameter row.
        soil_agg: Aggregated soil properties.
        terrain: Terrain dict.
        static: Static results (evaluate_static).

    Returns:
        pandas.DataFrame with a single row.
    """
    return pd.DataFrame([{
        "crop": params.name,
        "common_name": params["common_name"],
        "type": params["type"],
        "elevation_status": static["elevation"]["status"],
        "elevation_detail": static["elevation"]["detail"],
        "elevation_m": round(terrain["elevation_m"], 2),
        "slope_status": static["slope"]["status"],
        "slope_detail": static["slope"]["detail"],
        "slope_deg": round(terrain["slope_deg"], 2),
        "slope_std_deg": round(terrain["slope_std_deg"], 2),
        "texture_status": static["texture"]["status"],
        "texture_detail": static["texture"]["detail"],
        "sand_0_60_pct": soil_agg["sand_0_60_pct"],
        "clay_0_60_pct": soil_agg["clay_0_60_pct"],
        "ph_status": static["ph"]["status"],
        "ph_detail": static["ph"]["detail"],
        "ph_0_60": soil_agg["ph_0_60"],
        "soc_status": static["soc"]["status"],
        "soc_detail": static["soc"]["detail"],
        "soc_0_60_pct": soil_agg["soc_0_60_pct"],
    }])


def build_temperature_frame(
    series_df: pd.DataFrame, params: pd.Series
) -> pd.DataFrame:
    """
    Builds the biweekly temperature output frame: crop identity plus the
    temperature analysis columns, one row per biweekly period.

    Args:
        series_df: Per-period temperature frame (evaluate_temperature_series).
        params: Crop parameter row.

    Returns:
        pandas.DataFrame with one row per biweekly period.
    """
    cols = ["period_start", "period_end", "label", "mean_C"]
    for extra in ["std_C", "var_C"]:
        if extra in series_df.columns:
            cols.append(extra)
    cols += ["temp_status", "temp_detail", "temp_score"]
    out = series_df[cols].copy()
    out["temp_score"] = out["temp_score"].round(4)
    out.insert(0, "crop", params.name)
    return out


# ---------------------------------------------------------------------------
# Saving / Reporting
# ---------------------------------------------------------------------------

def save_viability_reports(
    static_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    crop: str,
    output_dir: Path = DATABASES_DIR,
) -> Tuple[Path, Path]:
    """
    Saves the static viability CSV and the enriched temperature CSV
    (``temperature_biweekly`` with the crop analysis columns) with a shared
    timestamp.

    Args:
        static_df: Single-row static frame (build_static_frame).
        temp_df: Biweekly temperature frame (build_temperature_frame).
        crop: Crop identifier (embedded in the filenames).
        output_dir: Directory where the CSVs are saved (created if missing).

    Returns:
        tuple: (static_path, temperature_path).
    """
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    static_path = output_dir / f"crop_viability_{crop}-static-v{timestamp}.csv"
    temp_path = output_dir / f"temperature_biweekly-v{timestamp}.csv"

    static_df.to_csv(static_path, index=False)
    temp_df.to_csv(temp_path, index=False)

    print(f"CSV saved to {static_path} "
          f"({static_df.shape[0]} row x {static_df.shape[1]} cols)")
    print(f"CSV saved to {temp_path} "
          f"({temp_df.shape[0]} rows x {temp_df.shape[1]} cols)")
    return static_path, temp_path


def print_viability_report(
    params: pd.Series,
    static: Dict[str, Dict[str, str]],
    temp_summary: Dict[str, object],
    final_viability: str,
    lat: float,
    lon: float,
    area_ha: float,
) -> None:
    """
    Prints a human-readable Level-1 viability report.

    The report lists the static parameters that do not comply, and the number
    of biweekly temperature non-aptitude peaks found.

    Args:
        params: Crop parameter row.
        static: Static results (evaluate_static).
        temp_summary: Temperature summary (summarize_temperature).
        final_viability: Final verdict.
        lat: Latitude of the evaluated point.
        lon: Longitude of the evaluated point.
        area_ha: Area of the evaluated plot in hectares.
    """
    crop = params.name
    static_labels = {
        "elevation": "elevation",
        "slope": "slope",
        "texture": "soil texture",
        "ph": "soil pH",
        "soc": "soil organic carbon",
    }

    print("\n" + "=" * 78)
    print("CROP PHYSICAL VIABILITY REPORT (LEVEL 1)")
    print("=" * 78)
    print(f"Location      : lat={lat}, lon={lon}")
    print(f"Area          : {area_ha} ha")
    print(f"Crop          : {crop} ({params['common_name']}) [{params['type']}]")
    print(f"Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 78)

    # Static parameters.
    print("\nStatic parameters:")
    failures = []    # list of (label, detail) for params that fail
    advisories = []  # list of labels for params with advisory
    for key, label in static_labels.items():
        status = static[key]["status"]
        detail = static[key]["detail"]
        print(f"  {label:<18} {status:<22} {detail}")
        if status == NOT_RECOMMENDED:
            failures.append((label, detail))
        elif status == VIABLE_WITH_ADVISORY:
            advisories.append(label)

    # Temperature summary.
    print("\nTemperature (biweekly series):")
    print(f"  optimal periods : {temp_summary['n_optimal']}")
    print(f"  cool periods    : {temp_summary['n_cool']}")
    print(f"  warm periods    : {temp_summary['n_warm']}")
    print(f"  too cold periods: {temp_summary['n_cold']}")
    print(f"  too hot periods : {temp_summary['n_hot']}")
    if temp_summary["n_missing"]:
        print(f"  missing periods : {temp_summary['n_missing']}")
    print(f"  non-aptitude peaks (cold + hot): {temp_summary['peaks']}")
    print(f"  temperature status             : {temp_summary['status']}")

    # Verdict.
    print("\n" + "-" * 78)
    print(f"FINAL VIABILITY: {final_viability}")
    if failures or temp_summary["peaks"] > 0:
        reasons = [f"{label} ({detail})" for label, detail in failures]
        if temp_summary["peaks"] > 0:
            reasons.append(
                f"temperature ({temp_summary['peaks']} non-aptitude peaks: "
                f"{temp_summary['n_cold']} cold, {temp_summary['n_hot']} hot)"
            )
        print("The location does NOT meet: " + "; ".join(reasons))
    if advisories:
        print("Advisories (viable with correction): " + ", ".join(advisories))
    print("=" * 78)


# ---------------------------------------------------------------------------
# Runner (regenerates inputs via the GEE modules, then filters)
# ---------------------------------------------------------------------------

def regenerate_inputs(
    lat: float,
    lon: float,
    area_ha: float,
    output_dir: Path = DATABASES_DIR,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    (Re)generates soil, terrain and temperature CSVs by calling the existing
    GEE extraction modules, and returns the same structures the pure core
    consumes.

    The Earth Engine modules are imported lazily here so that the pure core
    (and its unit tests) never needs the ``earthengine-api`` dependency.

    Args:
        lat: Latitude of the point.
        lon: Longitude of the point.
        area_ha: Plot area in hectares (used to derive the buffer radius).
        output_dir: Directory where the intermediate CSVs are saved.

    Returns:
        tuple: (soil_df, terrain_dict, temperature_df) where:
            - soil_df is indexed by property with depth columns (raw units)
            - terrain_dict has elevation_m/slope_deg/slope_std_deg
            - temperature_df has period_start/period_end/label/mean_C
    """
    # Lazy imports: keep the GEE dependency out of the pure core.
    from soil_profile_area import get_soil_profile_area, save_soil_profile
    from terrain_profile import get_terrain_profile_area, save_terrain_profile
    from temperature_profile import get_temperature_biweekly

    # Buffer radius: same formula used across terrain/soil profile scripts.
    area_meters = math.sqrt(area_ha * 10000) / 2

    # --- Soil profile (raw SoilGrids) ---
    soil_dict = get_soil_profile_area(lat, lon, area_meters)
    save_soil_profile(soil_dict, output_dir=str(output_dir))
    soil_df = pd.DataFrame(soil_dict).T  # index=property, columns=depth

    # --- Terrain profile ---
    terrain_dict = get_terrain_profile_area(lat, lon, area_meters)
    save_terrain_profile(terrain_dict, output_dir=str(output_dir))

    # --- Biweekly temperature (saved later, enriched with the crop analysis) ---
    temp_df = get_temperature_biweekly(lat, lon, start_date="2016-01-01")

    return soil_df, terrain_dict, temp_df


def main(
    lat: float,
    lon: float,
    crop: str,
    area_ha: float = 2.0,
    regenerate: bool = True,
    crop_params_path: Optional[Path] = None,
    output_dir: Path = DATABASES_DIR,
) -> Tuple[Path, Path]:
    """
    Runs the Level-1 physical-viability filter for ONE crop and a point.

    Args:
        lat: Latitude of the point.
        lon: Longitude of the point.
        crop: Crop identifier (must exist in crop_parameters, e.g.
              ``arabica_coffee``).
        area_ha: Plot area in hectares (default 2).
        regenerate: If True, call the GEE modules to (re)generate the input
                    CSVs; if False, read the latest local CSVs instead.
        crop_params_path: Path to the crop parameters CSV (defaults to the
                          file under ``databases``).
        output_dir: Directory for the output CSVs.

    Returns:
        tuple: (static_path, temperature_path).

    Raises:
        ValueError: If ``crop`` is not present in the parameters table.
    """
    if crop_params_path is None:
        crop_params_path = DATABASES_DIR / CROP_PARAMS_FILENAME

    # 1. Inputs: either regenerate from GEE or read the latest local CSVs.
    if regenerate:
        soil_df, terrain, temp_df = regenerate_inputs(
            lat, lon, area_ha, output_dir=output_dir
        )
    else:
        soil_df = load_soil_profile(latest_csv(SOIL_PREFIX, output_dir))
        terrain = load_terrain(latest_csv(TERRAIN_PREFIX, output_dir))
        temp_df = load_temperature(latest_csv(TEMPERATURE_PREFIX, output_dir))

    # 2. Crop parameters; validate the requested crop.
    crop_params = load_crop_parameters(crop_params_path)
    if crop not in crop_params.index:
        valid = ", ".join(crop_params.index.tolist())
        raise ValueError(f"Unknown crop '{crop}'. Valid crops: {valid}")
    params = crop_params.loc[crop]

    # 3. Aggregations.
    soil_agg = soil_aggregates(soil_df)

    # 4. Static filters (once) + temperature (per biweekly period).
    static = evaluate_static(soil_agg, terrain, params)
    series_df = evaluate_temperature_series(temp_df, params)
    temp_summary = summarize_temperature(series_df)

    # 5. Final verdict (console only) + the two output frames.
    final_viability = compute_final_viability(static, temp_summary["status"])
    static_df = build_static_frame(params, soil_agg, terrain, static)
    temp_df = build_temperature_frame(series_df, params)

    # 6. Save + report.
    static_path, temp_path = save_viability_reports(
        static_df, temp_df, crop, output_dir=output_dir
    )
    print_viability_report(
        params, static, temp_summary, final_viability, lat, lon, area_ha
    )

    return static_path, temp_path


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

    # Default demo: Arabica coffee at Finca Matanza, Colombia.
    LAT = 7.4584221918243045
    LON = -73.222052853104
    CROP = "cacao_ccn51"

    main(LAT, LON, CROP, area_ha=2.0, regenerate=True)
