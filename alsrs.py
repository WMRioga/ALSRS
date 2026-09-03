"""
ALSRS Orchestrator
==================

Entry point of the Agricultural Land Suitability Recommendation System (ALSRS).

It asks for a location (latitude, longitude) and a crop, then runs the full
pipeline in order:

    1. crop_viability  -> physical viability filter (soil, terrain, temperature)
    2. water_balance   -> sequential water balance and WRSI
    3. ahp_suitability -> AHP multi-criteria land-suitability score

and finally prints a consolidated suggestion to the console.

Input modes:
    - Command line:  python alsrs.py --lat 7.45 --lon -73.22 --crop cacao_ccn51
    - Interactive:   python alsrs.py   (prompts for the values)

Only crops present in ``databases/crop_parameters_260822.csv`` are accepted.

The orchestrator always regenerates its own input documents from Earth Engine
(each analysis module can still be run independently).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

# Make the project's code folders importable (this file lives at the root).
_PROJECT_ROOT = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT / "common", _PROJECT_ROOT / "extraction",
           _PROJECT_ROOT / "analysis", _PROJECT_ROOT / "mapping"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ahp_suitability
import crop_viability
import water_balance
from crop_viability import CROP_PARAMS_FILENAME, DATABASES_DIR, load_crop_parameters


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


def validate_coordinates(lat: float, lon: float) -> None:
    """Validates latitude/longitude ranges."""
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude {lat} out of range [-90, 90]")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"Longitude {lon} out of range [-180, 180]")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parses the optional command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ALSRS - Agricultural Land Suitability Recommendation System"
    )
    parser.add_argument("--lat", type=float, help="Latitude (decimal degrees)")
    parser.add_argument("--lon", type=float, help="Longitude (decimal degrees)")
    parser.add_argument("--crop", type=str, help="Crop identifier (e.g. cacao_ccn51)")
    parser.add_argument(
        "--area", type=float, default=2.0, help="Plot area in hectares (default: 2.0)"
    )
    return parser.parse_args()


def prompt_interactive(crop_params: pd.DataFrame) -> Tuple[float, float, str]:
    """Prompts for latitude, longitude and crop (numbered menu)."""
    print("\n=== ALSRS - Interactive mode ===")
    while True:
        try:
            lat = float(input("Latitude : "))
            lon = float(input("Longitude: "))
            validate_coordinates(lat, lon)
            break
        except ValueError as err:
            print(f"  Invalid input ({err}), try again.")

    crops = crop_params.index.tolist()
    print("\nAvailable crops:")
    for i, crop in enumerate(crops, 1):
        row = crop_params.loc[crop]
        print(f"  {i:2d}. {crop:<16} {row['common_name']} ({row['type']})")

    while True:
        try:
            choice = int(input(f"\nSelect crop [1-{len(crops)}]: "))
            if 1 <= choice <= len(crops):
                return lat, lon, crops[choice - 1]
        except ValueError:
            pass
        print("  Invalid selection, try again.")


# ---------------------------------------------------------------------------
# Consolidated report
# ---------------------------------------------------------------------------

def print_consolidated_report(
    lat: float, lon: float, crop: str, crop_params: pd.DataFrame
) -> None:
    """
    Prints a final human-readable synthesis after the three pipeline stages.

    It reads the latest outputs of each stage (land suitability, water balance
    labels and static viability) and summarises them in a single block.
    """
    ls = pd.read_csv(_latest(f"land_suitability_{crop}-v*.csv")).iloc[0]
    wb = pd.read_csv(_latest(f"water_balance_labels_{crop}-v*.csv"))
    cv = pd.read_csv(_latest(f"crop_viability_{crop}-static-v*.csv")).iloc[0]

    params = crop_params.loc[crop]

    print("\n" + "=" * 78)
    print("ALSRS - FINAL SUGGESTION")
    print("=" * 78)
    print(f"Location      : lat={lat}, lon={lon}")
    print(f"Crop          : {crop} ({params['common_name']}) [{params['type']}]")
    print(f"Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 78)

    # Land suitability (AHP).
    print("\nLand suitability (AHP):")
    print(f"  Score        : {ls['suitability_score']:.4f}")
    print(f"  Class        : {ls['suitability_class']}")
    if ls.get("limiting_factor", "") and str(ls["limiting_factor"]) != "nan":
        print(f"  Limiting     : {ls['limiting_factor']}")
    if ls.get("warning", "") and str(ls["warning"]) != "nan":
        print(f"  Warning      : {ls['warning']}")

    # Water / WRSI.
    print("\nWater (WRSI):")
    print(f"  Mean WRSI    : {wb['WRSI'].mean():.3f}")
    dist = (
        wb["suggestion"].replace("", pd.NA).dropna().value_counts(normalize=True) * 100
    )
    parts = []
    for cls in ["LOW", "MEDIUM", "HIGH", "NOT_SUITABLE"]:
        if cls in dist:
            parts.append(f"{cls} {dist[cls]:.0f}%")
    print(f"  Irrigation   : {' | '.join(parts) if parts else 'N/A'}")

    # Static viability parameters.
    print("\nStatic parameters:")
    for key, label in [
        ("elevation", "elevation"),
        ("slope", "slope"),
        ("texture", "soil texture"),
        ("ph", "soil pH"),
        ("soc", "soil organic carbon"),
    ]:
        status = cv.get(f"{key}_status", "")
        detail = cv.get(f"{key}_detail", "")
        print(f"  {label:<18} {status}")
        if status not in ("VIABLE",):
            print(f"                     ({detail})")

    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Runs the ALSRS pipeline for a location and a crop."""
    # 1. Crop parameters (used for validation + menu).
    crop_params = load_crop_parameters(DATABASES_DIR / CROP_PARAMS_FILENAME)
    valid_crops = crop_params.index.tolist()

    # 2. Input: command-line args or interactive prompts.
    args = parse_args()
    if args.lat is not None and args.lon is not None and args.crop is not None:
        lat, lon, crop = args.lat, args.lon, args.crop
        area_ha = args.area
    else:
        lat, lon, crop = prompt_interactive(crop_params)
        area_ha = args.area

    # 3. Validation.
    validate_coordinates(lat, lon)
    if crop not in valid_crops:
        print(f"\n[ERROR] Unknown crop '{crop}'. Valid crops: {', '.join(valid_crops)}")
        sys.exit(1)

    # 4. Run the pipeline in order (each stage prints its own report).
    print("\n" + "#" * 78)
    print(f"# ALSRS - Running pipeline for {crop} at lat={lat}, lon={lon}")
    print("#" * 78)

    crop_viability.main(lat, lon, crop, area_ha=area_ha, regenerate=True)
    water_balance.main(lat, lon, crop, regenerate=True)
    ahp_suitability.main(lat, lon, crop, area_ha=area_ha, regenerate=False)

    # 5. Consolidated final suggestion.
    print_consolidated_report(lat, lon, crop, crop_params)


if __name__ == "__main__":
    main()
