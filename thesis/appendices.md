# Appendices

> Supporting material referenced from the main chapters (Chapter 3, §3.1).
> Each appendix documents one component of the ALSRS implementation and
> points to its source in the project repository. In the final PDF, the
> "*(embed …)*" placeholders below are replaced with the full content of the
> referenced files.
>
> The complete source code, datasets, notebooks and configuration of the
> ALSRS pipeline are available in the public repository:
> [github.com/WMRioga/ALSRS](https://github.com/WMRioga/ALSRS).

---

## Appendix A — Dataset Dictionary

The complete 33-column dictionary of the ALSRS datasets (`ml/ml_dataset_cacao_ccn51.csv`,
`ml/ml_dataset_sugarcane.csv` and `ml/ml_dataset_wheat.csv`). The three datasets
share the same schema. For every column it records the meaning, the units, whether the value is *acquired*
(downloaded from satellite / reanalysis), *calculated* (derived in code) or
*metadata* (identifier / calendar), and whether the calculation consumes crop
parameters from `databases/crop_parameters_260822.csv`.

- **Authoritative document:** [`ml/DATASET_DICTIONARY.md`](../../ml/DATASET_DICTIONARY.md)

*(Embed the column dictionary table here in the final PDF.)*

---

## Appendix B — Model Documentation

The Phase-2 irrigation-need model specification: problem formulation, the 17
features and 3 targets, the two-stage (hurdle) Random Forest design, the
by-farm split and sample-weighting scheme, and the evaluation protocol — with
the bibliography supporting each design decision.

- **Authoritative document:** [`ml/ML_MODEL.md`](../../ml/ML_MODEL.md)

*(Embed the model documentation here in the final PDF.)*

---

## Appendix C — Analysis Notebooks

The notebooks and scripts that produce every table and figure in Chapter 4.

| Notebook / script | Produces |
|---|---|
| `test/MDS650_260903_dataset.ipynb` | Zero-inflation (Table 4.1, Figure 4.1) and the correlation matrix / ENSO (Figure 4.4) |
| `ml/01_evaluate_hurdle.ipynb` | Baseline, hurdle and gate evaluation (Tables 4.2–4.7; the gate-threshold sweep, Figure 4.x) |
| `ml/02_evaluate_class_bins.ipynb` | Class schemes and confusion matrices (Tables 4.8–4.10, Figure 4.2) |
| `ml/03_final_hurdle_model.ipynb` | Final hurdle model assembly |
| `ml/04_alsrs_model_process.ipynb` | End-to-end ALSRS modelling process |
| `ml/05_train_validation_test.ipynb` | Cacao 70/10/20 split and official held-out test results (Table 4.5) |
| `ml/05_train_validation_test Sugarcane.ipynb` | Sugarcane 70/10/20 split and official test results (Tables 4.13–4.15, Figure 4.3b) |
| `ml/02_evaluate_class_bins_wheat.ipynb` | Wheat class-scheme exploration (not reported; see Chapter 5 limitations) |
| `ml/05_train_validation_test_wheat.ipynb` | Wheat attempt (not reported; see Chapter 5 limitations) |
| `test/ml/evaluate_model.py` | Cacao learning curve (Table 4.12, Figure 4.3) and persistence baselines (Table 4.11) |

---

## Appendix D — Pipeline Source Code

The ALSRS pipeline, organised by stage.

| Module | Stage |
|---|---|
| `alsrs.py` | Orchestrator / entry point |
| `extraction/` | Data acquisition via Google Earth Engine: `precipitation_profile.py`, `temperature_profile.py`, `evapotranspiration_profile.py`, `spei_profile.py`, `terrain_profile.py`, `soil_profile_area.py`, `oni_profile.py` |
| `analysis/` | `crop_viability.py`, `water_balance.py` (WRSI labelling), `ahp_suitability.py` |
| `common/` | Shared utilities: `period_utils.py`, `tile_utils.py`, `soil_hydraulics.py` |
| `mapping/` | Map visualisation: `point_map.py`, `point_map_refactored.py`, `regional_elevation_map.py` |
| `ml/collect_training.py` | Dataset assembly |

*(Embed the source listings here in the final PDF.)*

The complete pipeline source code is available in the public repository:
[github.com/WMRioga/ALSRS](https://github.com/WMRioga/ALSRS).

---

## Appendix E — Configuration Data

Crop parameters and model configuration.

| File | Purpose |
|---|---|
| `databases/crop_parameters_260822.csv` | Crop-parameter table (7 crops) |
| `databases/ahp_weights.csv` | AHP pairwise-comparison weights |
| `ml/points/{cacao,sugarcane,wheat}_points.csv` | Sampled point coordinates per crop |
| `databases/{cacao_ccn51,wheat}/` | Per-crop extraction and viability outputs |
| `ml/hurdle_models/manifest.json` | Hurdle-model hyperparameters and thresholds |
| `ml/models/manifest.json` | Model configuration manifest |

Source: `databases/crop_parameters_260822.csv`.

| Crop | Type | T min tol (°C) | T opt min (°C) | T opt max (°C) | T max tol (°C) | Elev min (m) | Elev max (m) | Slope max (°) | pH min | pH max | Clay max (%) | Sand max (%) | SOC min (%) | Water req. (mm) | Max dry (mo.) | Cycle (quincenas) | Root depth (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sugarcane | perennial | 15 | 20 | 35 | 38 | 0 | 1500 | 8 | 5.5 | 7.5 | 50 | 70 | 0.5 | 2000 | 3 | 24 | 100 |
| CCN-51 Cocoa | perennial | 15 | 18 | 32 | 35 | 0 | 800 | 8 | 5.5 | 7.0 | 50 | 70 | 0.5 | 1500 | 3 | 24 | 70 |
| Arabica Coffee | perennial | 10 | 15 | 24 | 30 | 800 | 2000 | 15 | 5.0 | 6.5 | 50 | 70 | 0.5 | 1400 | 3 | 24 | 80 |
| Plantain | perennial | 10 | 18 | 32 | 38 | 0 | 1000 | 8 | 5.5 | 7.0 | 50 | 70 | 0.5 | 1500 | 2 | 24 | 60 |
| Wheat | annual | 5 | 10 | 24 | 30 | 0 | 1000 | 8 | 5.5 | 7.5 | 50 | 70 | 0.5 | 550 | 3 | 9 | 110 |
| Sorghum | annual | 10 | 15 | 35 | 40 | 0 | 1000 | 8 | 5.5 | 8.0 | 50 | 70 | 0.5 | 550 | 5 | 8 | 110 |
| Canola | annual | 5 | 10 | 25 | 30 | 0 | 800 | 8 | 5.5 | 7.5 | 50 | 70 | 0.5 | 475 | 3 | 9 | 100 |
