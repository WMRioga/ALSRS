# ALSRS — Machine Learning Model Documentation

Reference documentation for the Phase-2 irrigation-need model. It describes
the problem formulation, the features and targets, the model choice, how the
data is split and weighted, and how the model is evaluated — together with the
bibliography that supports each design decision.

The dataset itself is documented separately in
[`DATASET_DICTIONARY.md`](DATASET_DICTIONARY.md).

---

## 1. Objective (fixed)

> **Predict, from today's features, whether an irrigation system will be
> needed over the next 6 months, based on the water deficit.**

Formally: predict three nested **accumulated future water deficits**
(percentage 0–100) over the next 1, 3 and 6 months, then map them to an
irrigation-need class. The prediction of a *specific future biweek's weather*
is deliberately **not** attempted — that is physically chaotic beyond ~2 weeks.

---

## 2. Problem formulation: regression, not classification

- The model predicts a **continuous** deficit (percentage 0–100), not a class.
- The class is derived **after** prediction with the project's own thresholds
  (configurable; the adopted scheme is 3 classes):

  | Deficit | Class |
  |---|---|
  | `<= 15` | LOW |
  | `15–50` | MODERATE |
  | `>  50` | SEVERE |

  (The 4-class variant LOW / MEDIUM / HIGH / NOT_SUITABLE at 15 / 30 / 50 is
  kept available — the thresholds are just constants.)

- **Rationale:** the threshold is the project's *irrigation policy*, not a
  literature value. Keeping it outside the model (a) separates the physical
  model from the decision rule, and (b) lets the thresholds change without
  retraining. Because the model is **regression**, the class scheme can be
  re-binned freely (4 → 3 → 2 classes) with the same continuous predictions.

- **Why 3 classes:** the MEDIUM↔HIGH boundary (30) was the dominant source of
  error (a large share of HIGH rows were predicted MEDIUM). Merging them into
  MODERATE raised the exact-class recall from ~0.42–0.61 to ~0.84 and overall
  accuracy from ~0.69 to ~0.77, while keeping the "urgent" class SEVERE apart.

### Derived suggestion

`suggestion = worst class among the three horizons` (severity order
LOW < MODERATE < SEVERE). A short but intense drought therefore flags SEVERE
even when the 6-month average looks mild.

---

## 3. Features (17)

```text
month, biweek, mean_C, std_C, precip_total_mm, precip_rainy_days,
pet_mm, spei_1m, spei_3m, spei_6m, spei_12m, AWC_mm, Storage_mm,
P_acum_mm, WRSI_1m, deficit_1m, oni
```

- **Climate** (acquired): `mean_C`, `std_C`, `precip_total_mm`,
  `precip_rainy_days`.
- **Climate** (calculated): `pet_mm` (ET0, Thornthwaite), `spei_1m/3m/6m/12m`.
- **Soil**: `AWC_mm`.
- **Water-balance state** (crop-driven): `Storage_mm`, `P_acum_mm`, `WRSI_1m`,
  `deficit_1m`.
- **Seasonality**: `month`, `biweek`.
- **ENSO**: `oni` (Oceanic Niño Index, NOAA CPC).

Every feature is known at time `t`, so no future information leaks into the
model, and each maps to a physical driver of the deficit. The dataset carries
33 columns in total. Of these, 17 are features and 3 are the forecast targets;
the remaining 13 are metadata, reference values or derived outputs and never
enter the model. `point_id`, `lat` and `lon` would leak farm identity and let
the model memorize sites instead of learning a physical relationship that
generalizes to new coordinates. `crop` is constant in this single-crop dataset,
and `period_start`, `period_end` and `label` are calendar keys whose information
`month` and `biweek` already carry. The `_mm` companions duplicate the
percentage targets in mm, `past_deficit_3m` and `past_deficit_6m` exist only as
persistence baselines (as features they would leak the answer), and `suggestion`
is the derived target. Elevation and slope never enter the dataset, because
they act as suitability constraints in Phase 1 rather than as deficit
predictors.

---

## 4. Targets (3)

```text
future_deficit_1m, future_deficit_3m, future_deficit_6m
```

Each is the **accumulated future deficit** (percentage 0–100) over the next
2 / 6 / 12 biweeks:

```text
future_deficit_H(t) = 100 * sum(ETc - AET over [t+1 .. t+H]) / sum(ETc over [t+1 .. t+H])
```

They are **genuinely forward-looking** (no overlap with the features, which
stop at time `t`), avoiding the redundancy of the earlier "shifted point
deficit" design. Nested horizons give a coarse stress trajectory (short /
medium / long term).

---

## 5. Model: hurdle (two-stage) per horizon

Because the targets are zero-inflated, we use a **hurdle / two-part model**
(Cragg 1971; Mullahy 1986; Lambert 1992 — see Section 9). For each horizon
(1m, 3m, 6m) we fit **two models** instead of one:

- **Stage 1 — classifier** (`RandomForestClassifier`): predicts
  `P(deficit > 0)` on ALL rows, with `class_weight="balanced"` (the positive
  class is a minority at short horizons). It answers *"is there a deficit?"*.
- **Stage 2 — regressor** (`RandomForestRegressor`): trained ONLY on the rows
  with `deficit > 0`, with `sample_weight = deficit`. It answers *"given a
  deficit, how much?"*.

Each stage specializes on a simpler problem, instead of one model learning
"when" and "how much" at once. Six models in total (2 per horizon).

### The gate (combination rule)

Combining the stages as `expected = P(deficit>0) * magnitude` **shrinks** the
magnitude (the probability is < 1) and downgrades the severe classes. We
instead use a **gate**:

```text
prediction = magnitude   if P(deficit > 0) >= 0.5
prediction = 0           otherwise
```

Stage 1 only decides *whether* there is a deficit; stage 2 contributes the full
magnitude. This recovers the recall of the severe classes that the naive
`P * magnitude` combination lost, while keeping the large MAE gain from nailing
the zero rows. The threshold is per crop. Cacao uses 0.5 and sugarcane uses 0.75.

---

## 6. Class imbalance and weighting

The targets are **zero-inflated** (39–73% of rows have deficit = 0) and skewed
toward low values, while the rare high-deficit events are the ones that matter
for irrigation. The hurdle handles this in two complementary ways:

- **Stage 1 (classifier)** uses `class_weight="balanced"` so the minority
  positive class is not ignored.
- **Stage 2 (regressor)** uses `sample_weight = deficit` on the positive rows,
  so the rare extreme events dominate the fit (with `linear`, ~66% of the
  weight concentrates on the ~11% of rows with deficit > 50%).

The `linear` weight was chosen by explicit comparison (experiment A in
`evaluate_model.py`): three schemes were evaluated — **linear** (`deficit`),
**squared** (`deficit²`) and **plus-1-squared** (`1 + deficit²`). On the single
regressor, `plus1sq` gave the **lowest MAE** (~5–8 vs ~11–12 percentage points),
because it also weights the zero-deficit rows and thus "nails" them; but it gave
the **lowest severe-case recall** (~0.64). `linear` gave the **highest SEVERE
recall** (~0.72) at the cost of a higher MAE. Since the severe cases are the
decision-relevant ones, `linear` was adopted — the higher single-regressor MAE is
then removed by the hurdle's stage-1 classifier.

This prioritizes the extreme events without artificially resampling the data.

---

## 7. Train / validation / test split (70/10/20)

- **Split by point (farm), not by row**: a farm's whole time series goes
  entirely to one set, preventing leakage from the temporal autocorrelation
  within a farm.
- **70% train / 10% validation / 20% test**, fixed seed (42), no zone
  stratification (the model predicts behaviour, not zone):
  - **train** (70%) — fit the model.
  - **validation** (10%) — tune the gate threshold and the class scheme.
  - **test** (20%) — final report, touched **once**.
- Because the 10% validation set is small, the gate threshold is also confirmed
  with **5-fold CV on the train set** (grouped by farm) for a more robust
  tuning signal; the test set is never used for tuning.

---

## 8. Evaluation

### Regression (primary)

- **MAE** and **RMSE per horizon** (1m, 3m, 6m), cross-validated 5-fold by farm
  (`GroupKFold` grouped by `point_id`).
- Compared against a **single-RF baseline** (same features, `sample_weight =
  deficit`), to prove the hurdle improves on the plain regressor.

### Derived classification (secondary)

- After prediction, apply the configurable thresholds and derive `suggestion` =
  worst of 3.
- Report the **confusion matrix** and the **recall of MODERATE / SEVERE** (the
  classes that justify irrigation), not just global accuracy.
- Also report **cumulative recall** (`>= MODERATE`, `>= SEVERE`), the
  decision-relevant metric that is less strict than exact-class match.

### Key results (5-fold CV + holdout)

- The gated hurdle matches the single-RF recall of the severe classes while
  cutting MAE by ~31–63% (it nails the zero rows).
- 3 classes raise the exact-class recall of MODERATE to ~0.84 (vs 0.42–0.61 for
  the split MEDIUM / HIGH) and overall accuracy to ~0.77.

---

## 9. Bibliography

### Water balance / WRSI (target construction)

- Frère, M., & Popov, G.F. (1979). *Agrometeorological crop monitoring and
  forecasting.* FAO Plant Production and Protection Paper 17. Rome.
- FEWS NET (USGS). *Water Requirement Satisfaction Index (WRSI)* methodology
  and GeoWRSI tool documentation.
  https://help.fews.net/en/tools/v3/chapter-11-geowrsi
- An Improved Climatological Forecast Method for Projecting End-Of-Season
  Water Requirement Satisfaction Index (WRSI). ProQuest dissertation.
  https://www.proquest.com/docview/2455969164

### Reference evapotranspiration (ET0 / ETc)

- Thornthwaite, C.W. (1948). An approach toward a rational classification of
  climate. *Geographical Review, 38*(1), 55–94.
- Allen, R.G., Pereira, L.S., Raes, D., & Smith, M. (1998). *Crop
  evapotranspiration: Guidelines for computing crop water requirements.*
  FAO Irrigation and Drainage Paper 56. Rome.
- Doorenbos, J., & Kassam, A.H. (1979). *Yield response to water.* FAO
  Irrigation and Drainage Paper 33. Rome.

### Drought index (SPEI)

- Vicente-Serrano, S.M., Beguería, S., & López-Moreno, J.I. (2010). A
  multi-scalar drought index sensitive to global warming: The Standardized
  Precipitation Evapotranspiration Index. *Journal of Climate, 23*(7),
  1696–1718. https://doi.org/10.1175/2009JCLI2909.1
- Beguería, S., Vicente-Serrano, S.M., Reig, F., & Latorre, B. (2014).
  Standardized precipitation evapotranspiration index (SPEI) revisited.
  *International Journal of Climatology, 34*(10), 3001–3023.
- Hosking, J.R.M., & Wallis, J.R. (1997). *Regional Frequency Analysis: An
  Approach Based on L-Moments.* Cambridge University Press.

### Crop parameters (water requirement, drought tolerance)

- Carr, M.K.V., & Lockwood, R. (2011). The water relations and irrigation
  requirements of cocoa (*Theobroma cacao* L.): A review. *Experimental
  Agriculture, 47*(4), 653–676.
- ICCO (2017). *Growing Cocoa.* International Cocoa Organization.
- Carr, M.K.V. (2001). The water relations and irrigation requirements of
  coffee. *Experimental Agriculture, 37*(1), 1–36.
- Inman-Bamber, N.G., & Smith, D.M. (2005). Water relations in sugarcane and
  response to water deficits. *Field Crops Research, 92*(2–3), 185–202.
- DaMatta, F.M., & Ramalho, J.D.C. (2006). Impacts of drought and temperature
  stress on coffee physiology and production: A review. *Brazilian Journal of
  Plant Physiology, 18*(1), 55–81.
- Cenicafé (2016). *Guía para el cultivo de café en Colombia.* Cenicafé.

### Land evaluation / AHP (Phase 1)

- FAO (1976). *A Framework for Land Evaluation.* Soils Bulletin No. 32. Rome.
- Saaty, T.L. (1980). *The Analytic Hierarchy Process.* McGraw-Hill.

### ENSO / ONI

- Trenberth, K.E. (1997). The definition of El Niño. *Bulletin of the American
  Meteorological Society, 78*(12), 2771–2777.
- NOAA Climate Prediction Center. Oceanic Niño Index (ONI).
  https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

### Machine learning (model)

- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.

### Imbalanced regression and sample weighting

- Branco, P., Torgo, L., & Ribeiro, R.P. (2017). A survey of predictive
  modeling on imbalanced domains. *ACM Computing Surveys, 49*(2), 1–50.
- Resampling strategies for imbalanced regression: a survey and empirical
  analysis. *Artificial Intelligence Review* (2024).
  https://link.springer.com/article/10.1007/s10462-024-10724-3
- A Short Survey on Importance Weighting for Machine Learning (2024).
  https://arxiv.org/abs/2403.10175
- scikit-learn. `RandomForestRegressor` — `sample_weight` parameter.
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor

### Two-part / hurdle model (classifier + regressor for zero-inflated targets)

- Cragg, J.G. (1971). Some statistical models for limited dependent variables
  with application to the demand for durable goods. *Econometrica, 39*(5),
  829–844.
- Mullahy, J. (1986). Specification and testing of some modified count data
  models. *Journal of Econometrics, 33*(3), 341–365.
- Lambert, D. (1992). Zero-inflated Poisson regression, with an application to
  defects in manufacturing. *Technometrics, 34*(1), 1–14.
- A Two-Stage Machine Learning Framework for High-Resolution Multi-Source
  Precipitation Fusion in Complex Terrain (2025) — two-stage ML for
  zero-inflated, highly skewed precipitation.
  https://www.mdpi.com/2073-4433/17/8/762

### Zero-inflated / skewed targets (precipitation-like)

- A Semi-supervised Framework for Simultaneous Classification and Regression
  of Zero-Inflated Time Series Data with Application to Precipitation
  Prediction. https://ieeexplore.ieee.org/abstract/document/5360491

### Evaluation metrics for regression

- Torgo, L., & Ribeiro, R.P. (2009). Precision and Recall for Regression.
  *Discovery Science, LNCS 5808*, 332–346.
  https://www.semanticscholar.org/paper/45f639807caeb7e457904423972a2e1211e408ef
