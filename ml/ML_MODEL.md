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
- The class is derived **after** prediction with the project's own thresholds:

  | Deficit | Class |
  |---|---|
  | `<= 15` | LOW |
  | `<= 30` | MEDIUM |
  | `<= 50` | HIGH |
  | `>  50` | NOT_SUITABLE |

- **Rationale:** the threshold is the project's *irrigation policy*, not a
  literature value. Keeping it outside the model (a) separates the physical
  model from the decision rule, and (b) lets the thresholds change without
  retraining.

### Derived suggestion

`suggestion = worst class among the three horizons` (severity order
LOW < MEDIUM < HIGH < NOT_SUITABLE). A short but intense drought therefore
flags NOT_SUITABLE even when the 6-month average looks mild.

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

Excluded on purpose: `point_id`, `lat`, `lon` (identity/location — would leak
farm identity), `crop` (constant in this single-crop dataset), dates, `label`,
the `_mm` companions (informative only), and `suggestion` (derived target).

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

## 5. Model: 3 independent RandomForestRegressor

- One `RandomForestRegressor` per horizon (1m, 3m, 6m), fitted independently.
- Independent models give per-horizon metrics and **per-horizon feature
  importance** — e.g. the hypothesis that `oni` matters more at 6 months than
  at 1 month (the observed correlation already supports this: 0.15 → 0.21).
- Regression target; thresholds applied post-prediction (Section 2).

---

## 6. Class imbalance and `sample_weight`

The targets are **zero-inflated** (40–73% of rows have deficit = 0) and skewed
toward low values, while the rare high-deficit events are the ones that matter
for irrigation. Two candidate weighting functions, applied per model on its own
target:

```text
linear : weight = deficit       (0-deficit rows get weight 0 = ignored)
squared: weight = deficit^2     (emphasizes the tail strongly)
```

With `linear`, ~66% of the total weight concentrates on the ~11% of rows with
deficit > 50%; with `squared`, ~82%. This prioritizes the rare extreme events
without artificially rebalancing (resampling) the dataset.

---

## 7. Train / test split

- **Split by point (farm), not by row**: a farm's whole time series goes
  entirely to train or entirely to test, preventing leakage from the temporal
  autocorrelation within a farm.
- Ratio 80/20, fixed seed (42), **no zone stratification** (the model predicts
  behaviour, not zone).
- Primary evaluation: generalization to **unseen farms** (the real use case:
  a new user's farm). A **temporal split** is kept as a secondary robustness
  check of non-stationarity.

---

## 8. Evaluation

### Regression (primary)

- **MAE** and **RMSE per horizon** (1m, 3m, 6m).
- Compared against a **persistence baseline** (`future deficit = current
  `deficit_1m``), to prove the model beats "tomorrow ≈ today".

### Derived classification (secondary)

- After prediction, apply the 15/30/50 thresholds and derive `suggestion` =
  worst of 3.
- Report the **confusion matrix** and, crucially, the **recall of HIGH and
  NOT_SUITABLE** (the classes that justify irrigation), not just global
  accuracy (which the many LOW rows would inflate).
- For the skewed target, favour **range-based regression metrics** (e.g.
  Torgo & Ribeiro's precision/recall for regression) over global RMSE.

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

### Zero-inflated / skewed targets (precipitation-like)

- A Semi-supervised Framework for Simultaneous Classification and Regression
  of Zero-Inflated Time Series Data with Application to Precipitation
  Prediction. https://ieeexplore.ieee.org/abstract/document/5360491

### Evaluation metrics for regression

- Torgo, L., & Ribeiro, R.P. (2009). Precision and Recall for Regression.
  *Discovery Science, LNCS 5808*, 332–346.
  https://www.semanticscholar.org/paper/45f639807caeb7e457904423972a2e1211e408ef
