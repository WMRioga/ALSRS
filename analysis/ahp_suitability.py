"""
AHP Land Suitability Module
===========================

Combines the outputs of the previous pipeline stages (``crop_viability`` and
``water_balance``) into a single multi-criteria land-suitability score using
the Analytic Hierarchy Process (AHP, Saaty 1980).

The hierarchy (criteria and sub-criteria) and their weights are read from
``databases/ahp_weights.csv``. Each sub-criterion is scored in the range 0-1:

    - Temperature : mean of the per-period ``temp_score`` column produced by
                    ``crop_viability``, with an "extreme period" exception
                    (see below).
    - Elevation   : 1 inside [elev_min, elev_max], linear ramp to 0 over a
                    10% tolerance margin outside the range.
    - Slope       : 1 up to slope_max, linear ramp to 0 at slope_max + 15 deg.
    - Texture     : min(sand_score, clay_score), each 1 up to its limit and
                    ramping to 0 at 100%.
    - pH          : 1 inside [ph_min, ph_max]; 0.5 within +/- 1.0; 0.2 beyond.
    - SOC         : min(1, soc / soc_min).
    - Water (WRSI): proportion of labeled periods with ``deficit_pct <= 30``.

Temperature exception (a biweekly period outside the tolerable range has
``temp_score == 0``):
    - 0 extreme periods  -> score = mean(temp_score), no warning.
    - 1 extreme period   -> mean(temp_score) + a field-validation warning.
    - >1 extreme periods -> score = 0.0 (the temperature limiting factor then
                            caps the class to N).

The final suitability is the weighted sum of the sub-criterion scores, mapped
to the FAO classes S1/S2/S3/N, plus a "limiting factor" rule: a zero score in
a hard criterion (texture, temperature) caps the class to N, while a zero
slope score caps it to S3 and flags a field verification. Elevation is a SOFT
criterion (a proxy for temperature, which is measured directly): it lowers the
score but never hard-fails, and emits a "validate with local conditions"
warning when outside the literature range.

Output:
    databases/land_suitability_{crop}-vYYMMDDHHMMSS.csv  (one row per crop)
    plus a human-readable console report.

Dependencies:
    - pandas
    - crop_viability, water_balance (custom modules, same directory)
"""
from __future__ import annotations

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

# Upstream pipeline stages (imported as modules to call their main()).
import crop_viability
import water_balance
from crop_viability import CROP_PARAMS_FILENAME, DATABASES_DIR, load_crop_parameters


# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

AHP_WEIGHTS_FILENAME = "ahp_weights.csv"

# Tolerance parameters of the scoring functions.
ELEVATION_MARGIN_RATIO = 0.10   # elevation ramp = 10% of the crop range
SLOPE_MARGIN_DEG = 15.0         # slope ramp from slope_max to slope_max + 15 deg
PH_CORRECTION_SCORE = 0.5       # pH within +/- 1.0 of the optimal range
PH_COSTLY_SCORE = 0.2           # pH beyond +/- 1.0 (costly but always fixable)
WATER_STRESS_THRESHOLD_PCT = 30.0  # deficit <= 30% counts as "no water stress"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest(pattern: str, directory: Path = DATABASES_DIR) -> Path:
    """
    Returns the most recent CSV matching a glob pattern.

    Filenames use a ``YYMMDDHHMMSS`` timestamp, so lexicographic sorting of
    the timestamp portion equals chronological sorting; the last match is the
    newest file.
    """
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No CSV matching '{pattern}' in {directory}")
    return candidates[-1]


# ---------------------------------------------------------------------------
# Scoring Functions (raw value -> 0-1)
# ---------------------------------------------------------------------------

def evaluate_temperature_score(temp_score: pd.Series) -> Tuple[float, str]:
    """
    Temperature score with the "extreme period" exception.

    A biweekly period outside the tolerable range has ``temp_score == 0``:
        - 0 extreme periods  -> score = mean(temp_score), no warning.
        - 1 extreme period   -> mean(temp_score) + a field-validation warning.
        - >1 extreme periods -> score = 0.0 (the temperature limiting factor
                                then caps the class to N).

    Args:
        temp_score: Series of per-period trapezoidal scores in [0, 1].

    Returns:
        tuple: (score, warning) where ``warning`` is "" when none applies.
    """
    n_extreme = int((temp_score == 0.0).sum())
    if n_extreme > 1:
        return 0.0, ""
    mean_score = float(temp_score.mean())
    if n_extreme == 1:
        return mean_score, (
            "field validation recommended: temperature outside threshold in 1 period"
        )
    return mean_score, ""


def score_elevation(elevation_m: float, params: pd.Series) -> float:
    """1 inside [elev_min, elev_max], ramp to 0 over a 10% margin outside."""
    lo = float(params["elev_min_m"])
    hi = float(params["elev_max_m"])
    margin = ELEVATION_MARGIN_RATIO * (hi - lo)
    if lo <= elevation_m <= hi:
        return 1.0
    if margin <= 0:
        return 0.0
    if elevation_m < lo:
        return max(0.0, 1.0 - (lo - elevation_m) / margin)
    return max(0.0, 1.0 - (elevation_m - hi) / margin)


def score_slope(slope_deg: float, params: pd.Series) -> float:
    """1 up to slope_max, ramp to 0 at slope_max + 15 deg."""
    m = float(params["slope_max_deg"])
    if slope_deg <= m:
        return 1.0
    return max(0.0, 1.0 - (slope_deg - m) / SLOPE_MARGIN_DEG)


def score_texture(sand_pct: float, clay_pct: float, params: pd.Series) -> float:
    """min(sand, clay), each 1 up to its limit and ramping to 0 at 100%."""
    sand_max = float(params["sand_max_pct"])
    clay_max = float(params["clay_max_pct"])

    if sand_pct <= sand_max:
        sand_score = 1.0
    else:
        sand_score = max(0.0, 1.0 - (sand_pct - sand_max) / (100.0 - sand_max))

    if clay_pct <= clay_max:
        clay_score = 1.0
    else:
        clay_score = max(0.0, 1.0 - (clay_pct - clay_max) / (100.0 - clay_max))

    return min(sand_score, clay_score)


def score_ph(ph: float, params: pd.Series) -> float:
    """1 inside [ph_min, ph_max]; 0.5 within +/- 1.0; 0.2 beyond."""
    lo = float(params["ph_min"])
    hi = float(params["ph_max"])
    if lo <= ph <= hi:
        return 1.0
    deviation = (lo - ph) if ph < lo else (ph - hi)
    return PH_CORRECTION_SCORE if deviation <= 1.0 else PH_COSTLY_SCORE


def score_soc(soc_pct: float, params: pd.Series) -> float:
    """min(1, soc / soc_min): saturated linear "more is better"."""
    min_soc = float(params["soc_min_pct"])
    if min_soc <= 0:
        return 1.0
    return min(1.0, soc_pct / min_soc)


def score_water(
    deficit_pct: pd.Series, threshold: float = WATER_STRESS_THRESHOLD_PCT
) -> float:
    """Proportion of labeled periods with water deficit <= threshold."""
    valid = deficit_pct.dropna()
    if len(valid) == 0:
        return float("nan")
    return float((valid <= threshold).mean())


def classify_suitability(score: float) -> str:
    """Maps the suitability score to the FAO classes S1/S2/S3/N."""
    if pd.isna(score):
        return "N/A"
    if score >= 0.75:
        return "S1 (highly suitable)"
    if score >= 0.50:
        return "S2 (moderately suitable)"
    if score >= 0.25:
        return "S3 (marginally suitable)"
    return "N (not suitable)"


# Hard physiological/soil constraints: a zero score here caps the class to N
# (they cannot be corrected in the field). Elevation is intentionally excluded:
# it is a proxy for temperature, which is measured directly.
HARD_LIMITING_FACTORS = ("textura", "temperatura")

LIMITING_MESSAGES = {
    "textura": "soil texture beyond universal limits",
    "temperatura": "temperature regime outside tolerable range",
    "pendiente": "slope exceeds limit; verify workability in the field",
}


def apply_limiting_factor(
    score: float, scores: Dict[str, float]
) -> Tuple[str, str]:
    """
    Applies the FAO "limiting factor" rule on top of the weighted score.

    Hard constraints (elevation, texture, temperature) that score exactly 0
    are not correctable in the field, so they cap the class to "N". A zero
    slope score is potentially workable (terracing / field decision), so it
    caps the class to "S3" and flags a field verification.

    Args:
        score: Weighted suitability score in [0, 1].
        scores: Sub-criterion scores (see main()).

    Returns:
        tuple: (final_class, limiting_factor) where ``limiting_factor`` is the
        name of the capping criterion, or "" when none applies.
    """
    for key in HARD_LIMITING_FACTORS:
        if scores.get(key, 1.0) <= 1e-9:
            return "N (not suitable)", key
    if scores.get("pendiente", 1.0) <= 1e-9:
        return "S3 (marginally suitable)", "pendiente"
    return classify_suitability(score), ""


# ---------------------------------------------------------------------------
# Weights + Aggregation
# ---------------------------------------------------------------------------

def load_ahp_weights(path: Path) -> pd.DataFrame:
    """Loads the AHP hierarchy/weights CSV (criterion, subcriterion, weights)."""
    df = pd.read_csv(path)
    df["weight_global"] = pd.to_numeric(df["weight_global"], errors="coerce")
    return df


def compute_suitability(
    scores: Dict[str, float], weights_df: pd.DataFrame
) -> float:
    """
    Weighted sum of the sub-criterion scores using the global weights.

    Args:
        scores: dict keyed by subcriterion name -> score in [0, 1].
        weights_df: DataFrame from load_ahp_weights().

    Returns:
        float: suitability score in [0, 1].
    """
    w = dict(zip(weights_df["subcriterion"], weights_df["weight_global"]))
    total = 0.0
    for key, score in scores.items():
        if pd.isna(score):
            return float("nan")
        total += float(score) * float(w.get(key, 0.0))
    return total


# ---------------------------------------------------------------------------
# Saving / Reporting
# ---------------------------------------------------------------------------

def save_suitability_report(
    df: pd.DataFrame, crop: str, output_dir: Path = DATABASES_DIR
) -> Path:
    """Saves the single-row suitability result to a timestamped CSV."""
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"land_suitability_{crop}-v{timestamp}.csv"
    df.to_csv(out_path, index=False)
    print(f"CSV saved to {out_path} ({df.shape[0]} row x {df.shape[1]} cols)")
    return out_path


def print_suitability_report(
    params: pd.Series,
    scores: Dict[str, float],
    suitability: float,
    suitability_class: str,
    lat: float,
    lon: float,
    limiting_factor: str = "",
    warning: str = "",
) -> None:
    """Prints a human-readable AHP suitability summary."""
    labels = {
        "temperatura": "Temperature",
        "elevacion": "Elevation",
        "pendiente": "Slope",
        "textura": "Soil texture",
        "ph": "Soil pH",
        "soc": "Soil organic carbon",
        "wrsi": "Water (WRSI)",
    }
    print("\n" + "=" * 78)
    print("AHP LAND SUITABILITY REPORT")
    print("=" * 78)
    print(f"Location      : lat={lat}, lon={lon}")
    print(f"Crop          : {params.name} ({params['common_name']}) [{params['type']}]")
    print(f"Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 78)
    print("Sub-criterion scores (0-1):")
    for key, label in labels.items():
        value = scores.get(key, float("nan"))
        print(f"  {label:<22} {value:.4f}" if not pd.isna(value) else f"  {label:<22} N/A")
    print("-" * 78)
    print(f"SUITABILITY SCORE : {suitability:.4f}")
    print(f"SUITABILITY CLASS : {suitability_class}")
    if limiting_factor:
        print(f"LIMITING FACTOR   : {limiting_factor} "
              f"({LIMITING_MESSAGES.get(limiting_factor, '')})")
    if warning:
        print(f"WARNING           : {warning}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Runner (runs the upstream stages, then scores)
# ---------------------------------------------------------------------------

def main(
    lat: float,
    lon: float,
    crop: str,
    area_ha: float = 2.0,
    regenerate: bool = True,
    crop_params_path: Optional[Path] = None,
    ahp_weights_path: Optional[Path] = None,
    output_dir: Path = DATABASES_DIR,
) -> Path:
    """
    Computes the AHP land-suitability score for ONE crop and a point.

    Args:
        lat: Latitude of the point.
        lon: Longitude of the point.
        crop: Crop identifier (must exist in crop_parameters).
        area_ha: Plot area in hectares (default 2).
        regenerate: If True, run crop_viability and water_balance (GEE) to
                    (re)generate their outputs; if False, read the latest
                    per-crop CSVs instead.
        crop_params_path: Path to the crop parameters CSV.
        ahp_weights_path: Path to the AHP weights CSV.
        output_dir: Directory for the output CSV.

    Returns:
        pathlib.Path: Path to the land_suitability CSV.

    Raises:
        ValueError: If ``crop`` is not present in the parameters table.
    """
    if crop_params_path is None:
        crop_params_path = DATABASES_DIR / CROP_PARAMS_FILENAME
    if ahp_weights_path is None:
        ahp_weights_path = DATABASES_DIR / AHP_WEIGHTS_FILENAME

    # 1. Weights.
    weights_df = load_ahp_weights(ahp_weights_path)

    # 2. Upstream outputs (crop_viability + water_balance).
    if regenerate:
        cv_static_path, cv_temp_path = crop_viability.main(
            lat, lon, crop, area_ha=area_ha, regenerate=True,
            crop_params_path=crop_params_path, output_dir=output_dir,
        )
        wb_path = water_balance.main(
            lat, lon, crop, regenerate=True,
            crop_params_path=crop_params_path, output_dir=output_dir,
        )
    else:
        cv_static_path = _latest(f"crop_viability_{crop}-static-v*.csv", output_dir)
        cv_temp_path = _latest("temperature_biweekly-v*.csv", output_dir)
        wb_path = _latest(f"water_balance_labels_{crop}-v*.csv", output_dir)

    static_df = pd.read_csv(cv_static_path)
    temp_df = pd.read_csv(cv_temp_path)
    wb_df = pd.read_csv(wb_path)

    # 3. Crop parameters; validate the requested crop.
    crop_params = load_crop_parameters(crop_params_path)
    if crop not in crop_params.index:
        valid = ", ".join(crop_params.index.tolist())
        raise ValueError(f"Unknown crop '{crop}'. Valid crops: {valid}")
    params = crop_params.loc[crop]

    # 4. Static values (single row of the static viability CSV).
    s = static_df.iloc[0]
    elevation_m = float(s["elevation_m"])
    slope_deg = float(s["slope_deg"])
    sand_pct = float(s["sand_0_60_pct"])
    clay_pct = float(s["clay_0_60_pct"])
    ph = float(s["ph_0_60"])
    soc_pct = float(s["soc_0_60_pct"])

    # 5. Score every sub-criterion (temperature reads its per-period column).
    warnings = []
    temp_score, temp_warning = evaluate_temperature_score(temp_df["temp_score"])
    if temp_warning:
        warnings.append(temp_warning)

    # Elevation is a PROXY for temperature (already measured directly), so it
    # is a soft criterion: it lowers the score but never hard-fails. When it
    # falls outside the literature range, flag a local-validation warning.
    elev_lo = float(params["elev_min_m"])
    elev_hi = float(params["elev_max_m"])
    if elevation_m < elev_lo:
        warnings.append(
            f"elevation below literature range ({elevation_m:.1f} m < {elev_lo:.0f} m); "
            "validate with local conditions"
        )
    elif elevation_m > elev_hi:
        warnings.append(
            f"elevation above literature range ({elevation_m:.1f} m > {elev_hi:.0f} m); "
            "validate with local conditions"
        )
    warning = "; ".join(warnings)

    scores = {
        "temperatura": temp_score,
        "elevacion": score_elevation(elevation_m, params),
        "pendiente": score_slope(slope_deg, params),
        "textura": score_texture(sand_pct, clay_pct, params),
        "ph": score_ph(ph, params),
        "soc": score_soc(soc_pct, params),
        "wrsi": score_water(wb_df["deficit_pct"]),
    }

    # 6. Weighted aggregation + FAO class with limiting-factor rule.
    suitability = compute_suitability(scores, weights_df)
    suitability_class, limiting_factor = apply_limiting_factor(suitability, scores)

    # 7. Build the output row and save.
    row = {
        "crop": crop,
        "common_name": params["common_name"],
        "type": params["type"],
        "score_temperatura": round(scores["temperatura"], 4),
        "score_elevacion": round(scores["elevacion"], 4),
        "score_pendiente": round(scores["pendiente"], 4),
        "score_textura": round(scores["textura"], 4),
        "score_ph": round(scores["ph"], 4),
        "score_soc": round(scores["soc"], 4),
        "score_agua": round(scores["wrsi"], 4),
        "suitability_score": round(suitability, 4),
        "suitability_class": suitability_class,
        "limiting_factor": limiting_factor,
        "warning": warning,
    }
    out_df = pd.DataFrame([row])
    out_path = save_suitability_report(out_df, crop, output_dir=output_dir)
    print_suitability_report(
        params, scores, suitability, suitability_class, lat, lon,
        limiting_factor, warning,
    )

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

    # Default demo: cocoa at El Playon, Colombia.
    LAT = 7.4584221918243045
    LON = -73.222052853104
    CROP = "cacao_ccn51"

    main(LAT, LON, CROP, area_ha=2.0, regenerate=True)
