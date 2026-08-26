"""
AHP Land Suitability Module
===========================

Combines the physical-viability and water-balance results into a single
multi-criteria land-suitability score using the Analytic Hierarchy Process
(AHP, Saaty 1980).

The hierarchy (criteria and sub-criteria) and their weights are read from
``databases/ahp_weights.csv``. Each sub-criterion is scored in the range 0-1
with a fuzzy/trapezoidal membership function whose breakpoints come from the
crop thresholds in ``crop_parameters.csv``:

    - Temperature : trapezoidal per biweekly period, averaged over the series.
    - Elevation   : 1 inside [elev_min, elev_max], linear ramp to 0 over a
                    10% tolerance margin outside the range.
    - Slope       : 1 up to slope_max, linear ramp to 0 at slope_max + 15 deg.
    - Texture     : min(sand_score, clay_score), each 1 up to its limit and
                    ramping to 0 at 100%.
    - pH          : 1 inside [ph_min, ph_max]; 0.5 within +/- 1.0; 0.2 beyond.
    - SOC         : min(1, soc / soc_min).
    - Water (WRSI): proportion of labeled biweekly periods with water deficit
                    <= 30% (i.e. without severe water stress).

The final suitability is the weighted sum of the sub-criterion scores:

    suitability = sum(weight_global * score)   over all sub-criteria

and is mapped to the FAO land-suitability classes S1/S2/S3/N. A FAO
"limiting factor" rule is then applied: a zero score in a hard criterion
(elevation, texture, temperature) caps the class to N, while a zero slope
score caps it to S3 and flags a field verification.

This module reuses the pure (GEE-free) helpers of ``crop_viability`` and
``water_balance``; the runner still regenerates the input CSVs from Earth
Engine when ``regenerate=True``.

Output:
    databases/land_suitability_{crop}-vYYMMDDHHMMSS.csv  (one row per crop)
    plus a human-readable console report.

Dependencies:
    - numpy, pandas
    - crop_viability, water_balance (custom modules, same directory)
"""
from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Make the local modules importable regardless of the working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Reuse the pure, GEE-free helpers from the previous pipeline stages.
from crop_viability import (
    CROP_PARAMS_FILENAME,
    DATABASES_DIR,
    SOIL_PREFIX,
    TEMPERATURE_PREFIX,
    TERRAIN_PREFIX,
    latest_csv,
    load_crop_parameters,
    load_soil_profile,
    load_temperature,
    load_terrain,
    soil_aggregates,
)
from water_balance import (
    PRECIP_PREFIX,
    SOIL_HYDRAULIC_PREFIX,
    SPEI_PREFIX,
    compute_wrsi_series,
    load_awc,
    load_precipitation,
    load_spei,
)


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
# Scoring Functions (raw value -> 0-1)
# ---------------------------------------------------------------------------

def _trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership: 0 below ``a``, ramp 0->1 on ``[a, b]``, 1 on
    ``[b, c]``, ramp 1->0 on ``[c, d]``, 0 above ``d``.
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


def score_temperature(mean_c: pd.Series, params: pd.Series) -> float:
    """Average trapezoidal temperature score over the biweekly series."""
    a = float(params["temp_min_tolerable_c"])
    b = float(params["temp_opt_min_c"])
    c = float(params["temp_opt_max_c"])
    d = float(params["temp_max_tolerable_c"])
    vals = [max(0.0, min(1.0, _trapezoid(t, a, b, c, d))) for t in mean_c if not pd.isna(t)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


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
# (they cannot be corrected in the field).
HARD_LIMITING_FACTORS = ("elevacion", "textura", "temperatura")

LIMITING_MESSAGES = {
    "elevacion": "elevation outside physiological range",
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
    print("=" * 78)


# ---------------------------------------------------------------------------
# Runner (regenerates inputs via the GEE modules, then scores)
# ---------------------------------------------------------------------------

def regenerate_inputs(
    lat: float, lon: float, area_ha: float, output_dir: Path = DATABASES_DIR
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    """
    Regenerates all six input datasets (soil, terrain, temperature,
    precipitation, SPEI, soil-hydraulic) and returns the structures the
    scoring functions consume.

    Earth Engine modules are imported lazily so the pure core stays GEE-free.
    """
    from precipitation_profile import (
        get_precipitation_biweekly,
        save_precipitation_profile,
    )
    from soil_hydraulics import calculate_hydraulic_properties
    from soil_profile_area import (
        get_soil_profile_area,
        save_hydraulic_profile,
        save_soil_profile,
    )
    from spei_profile import get_spei_biweekly, save_spei_profile
    from temperature_profile import (
        get_temperature_biweekly,
        save_temperature_profile,
    )
    from terrain_profile import get_terrain_profile_area, save_terrain_profile

    area_m = math.sqrt(area_ha * 10000) / 2

    # Soil (raw for texture/pH/SOC + hydraulic for AWC).
    soil_dict = get_soil_profile_area(lat, lon, area_m)
    save_soil_profile(soil_dict, output_dir=str(output_dir))
    soil_df = pd.DataFrame(soil_dict).T
    df_hydric = calculate_hydraulic_properties(soil_dict)
    save_hydraulic_profile(df_hydric, output_dir=str(output_dir))
    awc_mm = float(df_hydric["AWC_layer_mm"].sum())

    # Terrain.
    terrain = get_terrain_profile_area(lat, lon, area_m)
    save_terrain_profile(terrain, output_dir=str(output_dir))

    # Temperature.
    temp_df = get_temperature_biweekly(lat, lon, start_date="2016-01-01")
    save_temperature_profile(temp_df, output_dir=str(output_dir))

    # Precipitation.
    precip_df = get_precipitation_biweekly(lat, lon, start_date="2016-01-01")
    save_precipitation_profile(precip_df, output_dir=str(output_dir))

    # SPEI.
    spei_df = get_spei_biweekly(lat, lon, start_date="2016-01-01")
    save_spei_profile(spei_df, output_dir=str(output_dir))

    return soil_df, terrain, temp_df, precip_df, awc_mm, spei_df


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
        regenerate: If True, call the GEE modules to (re)generate the input
                    CSVs; if False, read the latest local CSVs instead.
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

    # 2. Inputs.
    if regenerate:
        soil_df, terrain, temp_df, precip_df, awc_mm, spei_df = regenerate_inputs(
            lat, lon, area_ha, output_dir
        )
    else:
        soil_df = load_soil_profile(latest_csv(SOIL_PREFIX, output_dir))
        terrain = load_terrain(latest_csv(TERRAIN_PREFIX, output_dir))
        temp_df = load_temperature(latest_csv(TEMPERATURE_PREFIX, output_dir))
        precip_df = load_precipitation(latest_csv(PRECIP_PREFIX, output_dir))
        awc_mm = load_awc(latest_csv(SOIL_HYDRAULIC_PREFIX, output_dir))
        spei_df = load_spei(latest_csv(SPEI_PREFIX, output_dir))

    # 3. Crop parameters; validate the requested crop.
    crop_params = load_crop_parameters(crop_params_path)
    if crop not in crop_params.index:
        valid = ", ".join(crop_params.index.tolist())
        raise ValueError(f"Unknown crop '{crop}'. Valid crops: {valid}")
    params = crop_params.loc[crop]

    # 4. Aggregations shared by the static criteria.
    soil_agg = soil_aggregates(soil_df)

    # 5. WRSI series for the water criterion.
    merged = precip_df.merge(
        spei_df[["label", "pet_mm", "spei_1m", "spei_3m", "spei_6m", "spei_12m"]],
        on="label",
        how="inner",
    )
    merged = merged.sort_values("label").reset_index(drop=True)
    merged = merged.dropna(subset=["precip_total_mm"]).reset_index(drop=True)
    wrsi_out, _, _ = compute_wrsi_series(
        merged,
        float(params["water_requirement_mm"]),
        int(params["cycle_quincenas"]),
        str(params["type"]),
        awc_mm,
    )

    # 6. Score every sub-criterion.
    scores = {
        "temperatura": score_temperature(temp_df["mean_C"], params),
        "elevacion": score_elevation(terrain["elevation_m"], params),
        "pendiente": score_slope(terrain["slope_deg"], params),
        "textura": score_texture(
            soil_agg["sand_0_60_pct"], soil_agg["clay_0_60_pct"], params
        ),
        "ph": score_ph(soil_agg["ph_0_60"], params),
        "soc": score_soc(soil_agg["soc_0_60_pct"], params),
        "wrsi": score_water(wrsi_out["deficit_pct"]),
    }

    # 7. Weighted aggregation + FAO class with limiting-factor rule.
    suitability = compute_suitability(scores, weights_df)
    suitability_class, limiting_factor = apply_limiting_factor(suitability, scores)

    # 8. Build the output row and save.
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
    }
    out_df = pd.DataFrame([row])
    out_path = save_suitability_report(out_df, crop, output_dir=output_dir)
    print_suitability_report(
        params, scores, suitability, suitability_class, lat, lon, limiting_factor
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
