# ALSRS — Agricultural Land Suitability Recommendation System

## Introduction

ALSRS is a system that recommends the agricultural suitability of a piece of
land for a given crop. Given a **location (latitude, longitude)** and a
**crop**, it runs a complete pipeline that:

1. **Extracts** free satellite / reanalysis data (climate, soil, terrain) from
   Google Earth Engine.
2. **Filters** the physical viability of the crop (soil, terrain, temperature).
3. **Computes** a sequential water balance and the Water Requirement
   Satisfaction Index (WRSI).
4. **Scores** the land suitability with the AHP multi-criteria method.
5. **Predicts** the future irrigation need from the water deficit using a
   Machine Learning model (a two-stage "hurdle" Random Forest).

The covered crops are: cacao CCN-51, arabica coffee and plantain (Colombia);
wheat, sorghum and canola (Australia); sugarcane (both regions).

The whole pipeline is orchestrated by `alsrs.py`:

```bash
python alsrs.py --lat 7.45 --lon -73.22 --crop cacao_ccn51
```

The full crop documentation is in
[`README_crops.md`](README_crops.md).

## Repository structure

- **`alsrs.py`** — pipeline orchestrator (entry point).
- **`analysis/`** — the three analysis stages:
  - `crop_viability.py` — physical viability filter (soil, terrain, temperature).
  - `water_balance.py` — sequential water balance and WRSI.
  - `ahp_suitability.py` — AHP land-suitability scoring.
- **`extraction/`** — Google Earth Engine downloaders (temperature,
  precipitation, evapotranspiration, SPEI, soil, terrain, ONI).
- **`common/`** — shared helpers (periods, soil hydraulics, tiles).
- **`mapping/`** — map generation scripts.
- **`databases/`** — configuration and per-crop data:
  - `crop_parameters_260822.csv` — crop parameters (7 crops).
  - `ahp_weights.csv` — AHP criteria weights.
  - `cacao_ccn51/`, `wheat/`, `sugarcane/` — per-point extraction and viability outputs.
- **`ml/`** — the machine-learning phase (see `ml/ML_MODEL.md` and
  `ml/DATASET_DICTIONARY.md`):
  - `collect_training.py` — builds the labeled training dataset.
  - `01_evaluate_hurdle.ipynb` to `06_reproduce_ch4_tables.ipynb` — experiments,
    train/validation/test results and authoritative table reproduction.
  - `evaluate_model.ipynb` — grid search and weight comparison.
  - `ml_dataset_{cacao_ccn51,sugarcane,wheat}.csv` — the labelled training datasets.
  - `points/` — sampled point coordinates and per-shard collection artifacts.
- **`img/`** — generated figures (confusion matrices, feature importance,
  learning curves) and `img/maps/` map images.
- **`test/`** — exploratory notebooks, unit tests and archived experiments.
- **`thesis/`** — the thesis manuscript (`md/` Markdown chapters, `tex/` LaTeX
  versions, `THESIS_PLAN.md`).
- **`presentation/`** — slide decks.
- **`scripts/`** — auxiliary scripts.

## Environment

Conda environment `agri_land_env` (Python 3.10). The ML notebooks require
`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn` and `joblib`; the
extraction phase requires `earthengine-api`.
