# ALSRS ML Dataset Dictionary

Reference documentation for `ml/ml_dataset_cacao_ccn51.csv` (30 columns). Each
row is one biweekly period for one sampled point in the cacao belt, with
features measured at that period and the water-deficit targets used to train
the irrigation-need model.

---

## 1. Column dictionary

For every column: **origin** marks whether the value is *acquired* (downloaded
from satellite / reanalysis), *calculated* (derived in code), or *metadata*
(identifier / calendar). **Uses crop?** marks whether the calculation consumes
crop parameters from `databases/crop_parameters_260822.csv`.

### 1.1 Identifiers and metadata (metadata — neither acquired nor calculated)

| # | Column | Description | Origin |
|---|---|---|---|
| 0 | `point_id` | Sampled-point identifier (`p001`…`p120`). | Manual label |
| 1 | `lat` | Point latitude (decimal degrees). | Input coordinate |
| 2 | `lon` | Point longitude (decimal degrees). | Input coordinate |
| 3 | `crop` | Crop identifier (`cacao_ccn51`). Constant across the dataset. | Fixed parameter |
| 4 | `period_start` | Start date of the biweekly period (`YYYY-MM-DD`). | Calendar |
| 5 | `period_end` | End date of the biweekly period (`YYYY-MM-DD`). | Calendar |
| 6 | `label` | Biweekly period key (`2016-01_Q1`), used to join tables. | Calendar |
| 7 | `month` | Month of the year (1–12), derived from `label`. | Calculated (from `label`) |
| 8 | `biweek` | Half of the month (1–2), derived from `label`. | Calculated (from `label`) |

### 1.2 Climate — acquired (satellite / reanalysis)

| # | Column | Description | Uses crop? |
|---|---|---|---|
| 9 | `mean_C` | Mean 2 m air temperature over the biweekly period (°C). Source: ERA5-Land, converted from Kelvin. | No |
| 10 | `std_C` | Standard deviation of the daily temperature within the biweekly period (°C); day-to-day variability. | No |
| 11 | `precip_total_mm` | Total accumulated precipitation over the biweekly period (mm). Source: CHIRPS (~5.5 km). | No |

### 1.3 Climate — calculated (no crop dependency)

| # | Column | Description | Uses crop? |
|---|---|---|---|
| 12 | `precip_rainy_days` | Rainy days in the period (days with precipitation > 1.0 mm). Calculated from daily CHIRPS. | No |
| 13 | `pet_mm` | Reference potential evapotranspiration **ET0** (Thornthwaite, 1948), from temperature + latitude. **This is not the crop ETc.** | No |
| 14 | `spei_1m` | SPEI at 1-month scale; standardized `P − PET` index (short-term moisture). | No |
| 15 | `spei_3m` | SPEI at 3-month scale (seasonal drought context). | No |
| 16 | `spei_6m` | SPEI at 6-month scale (medium-term agricultural drought). | No |
| 17 | `spei_12m` | SPEI at 12-month scale (long-term hydrological drought). | No |

### 1.4 Soil

| # | Column | Description | Uses crop? |
|---|---|---|---|
| 18 | `AWC_mm` | Available water capacity of the soil, summed over the 5 layers **0–100 cm** (mm). Source: SoilGrids. | No *(currently ignores `root_depth_cm`)* |

### 1.5 Water balance and WRSI — calculated, **uses the crop**

| # | Column | Description | Uses crop? |
|---|---|---|---|
| 19 | `Storage_mm` | Soil water storage (mm) from the sequential FAO bucket model: `Storage(t) = clip(P(t) + Storage(t−1) − AET(t), 0, AWC)`. | Yes (`water_requirement_mm` via ETc) |
| 20 | `P_acum_mm` | Precipitation accumulated over the rolling window (24 quincenas for perennials). | Yes (`type`/`cycle_quincenas` set the window) |
| 21 | `WRSI` | Water Requirement Satisfaction Index = `ΣAET / ΣETc` over the rolling window. | Yes (`water_requirement_mm`, `type`) |
| 22 | `deficit_pct` | Water deficit of the current period = `(1 − WRSI) × 100` (%). | Yes (via WRSI) |

### 1.6 Forecast targets (model labels) and suggestion

| # | Column | Description | Uses crop? |
|---|---|---|---|
| 23 | `future_deficit_1m` | **Target.** Accumulated water deficit over the next 2 biweeks (1 month), as a fraction 0–1: `Σ(ETc−AET)/ΣETc` over `[t+1 … t+2]`. | Yes |
| 24 | `future_deficit_3m` | **Target.** Accumulated deficit over the next 6 biweeks (3 months), fraction 0–1. | Yes |
| 25 | `future_deficit_6m` | **Target.** Accumulated deficit over the next 12 biweeks (6 months), fraction 0–1. | Yes |
| 26 | `future_deficit_1m_mm` | Companion of #23 in mm (numerator `Σ(ETc−AET)`). **Not a target.** | Yes |
| 27 | `future_deficit_3m_mm` | Companion of #24 in mm. **Not a target.** | Yes |
| 28 | `future_deficit_6m_mm` | Companion of #25 in mm. **Not a target.** | Yes |
| 29 | `suggestion` | Irrigation-need class derived from `future_deficit_6m`: LOW (≤0.15), MEDIUM (≤0.30), HIGH (≤0.50), NOT_SUITABLE (>0.50). **Derived, not predicted.** | Yes |

---

## 2. Intermediate values (calculated and used, but NOT stored in the dataset)

These values are computed inside `analysis/water_balance.py` and consumed by
the pipeline, but are not exported as columns of `ml_dataset_cacao_ccn51.csv`.

| Intermediate | Formula | Role | Depends on crop? |
|---|---|---|---|
| `ET0` (`pet_mm`) | Thornthwaite (1948): monthly PET from temperature + latitude, split into two biweekly halves. | Reference evapotranspiration; **stored** as `pet_mm`. | No |
| `ETc_mm` | `ETc(t) = water_requirement_mm × ET0(t) / ΣET0(window)` | Crop evapotranspiration per period; distributes the total requirement so `ΣETc(window) = water_requirement_mm`. | Yes (`water_requirement_mm`) |
| `AET_mm` | `AET(t) = min(P(t) + Storage(t−1), ETc(t))` | Actual evapotranspiration (water actually available). | Yes (via ETc) |
| `Storage_mm` | `Storage(t) = clip(P(t) + Storage(t−1) − AET(t), 0, AWC)` | Soil moisture state; **stored** as a feature. | Yes (via ETc/AET) |
| `WRSI` | `ΣAET(window) / ΣETc(window)` | Water satisfaction; **stored** as a feature. | Yes |

> **Note:** `ETc_mm` and `AET_mm` exist as intermediate columns inside
> `water_balance.py` (`out["ETc_mm"]`, `out["AET_mm"]`). They are used to
> compute the forecast targets (the accumulated deficits) and are dropped when
> the labeled CSV is written (`build_labeled_dataset`). They are deliberately
> excluded from the ML dataset because their information is already embedded in
> `WRSI`, `Storage_mm`, `deficit_pct`, and the targets.

---

## 3. How WRSI and the forecast targets work

### 3.1 The WRSI is a backward-looking rolling window, not a prediction

`water_balance.py` **does not forecast anything**. It computes the WRSI and
deficit **retrospectively** from already-observed precipitation (CHIRPS) and
temperature (ERA5-Land). The steps:

1. **Distribute the crop requirement** over each biweekly period using ET0 as
   the weight, so a full window sums to `water_requirement_mm` (1500 mm for
   cacao):

   ```
   ETc(t) = water_requirement_mm × ET0(t) / Σ ET0(window)
   ```

2. **Sequential bucket balance** (FAO method) with soil memory:

   ```
   AET(t)     = min(P(t) + Storage(t−1), ETc(t))
   Storage(t) = clip(P(t) + Storage(t−1) − AET(t), 0, AWC)
   ```

   The first 4 periods only warm up the storage and are excluded.

3. **Rolling WRSI** over the evaluation window `W` (24 quincenas for
   perennials, cycle length for annuals):

   ```
   WRSI(t)    = Σ AET(window) / Σ ETc(window)
   deficit(t) = (1 − WRSI(t)) × 100
   ```

### 3.2 The forecast targets are genuinely forward-looking

The ML targets (`future_deficit_1m/3m/6m`) are **accumulated future deficits**,
computed over the *next* H biweeks:

```
future_deficit_H(t) = Σ[ETc(t+j) − AET(t+j)] / Σ ETc(t+j)   , j = 1 … H
```

with H = 2 (1 month), 6 (3 months), 12 (6 months). They use **only future
data** (`t+1 … t+H`), so they do not overlap the features (which go up to time
`t`). This is why they avoid the redundancy problem of the previous design.

### 3.3 Why accumulated (not per-biweek) targets

The previous design used 12 per-biweek point targets (`deficit_t1…t12`), which
were `deficit_pct` shifted forward. Because `WRSI` is a **12-month rolling
window**, those labels shared most of their window with the present, so a
model predicted them trivially (redundancy) without learning the future.

Aggregating over the horizon fixes this in two ways:

1. **No overlap with features** — the accumulated window is entirely future.
2. **Predictable** — the chaotic weather of each individual biweek averages
   out inside the accumulated sum, leaving the seasonal signal + persistence
   that a model *can* learn.

The nested horizons (1m, 3m, 6m) give a coarse trajectory of the stress:
short term, medium term, and the full semester.

### 3.4 Model intent (fixed objective)

> **Predict, from today's features, whether an irrigation system will be
> needed over the next 12 biweekly windows (6 months), based on the water
> deficit.**

---

## 4. Crop dependency summary

| Group | Columns | Crop-dependent? |
|---|---|---|
| Pure climate | `mean_C`, `std_C`, `precip_total_mm`, `precip_rainy_days`, `pet_mm`, `spei_1m/3m/6m/12m` | No (identical for any crop) |
| Pure soil | `AWC_mm` | No (soil only) |
| Crop-driven | `Storage_mm`, `P_acum_mm` (window only), `WRSI`, `deficit_pct`, `future_deficit_*`, `suggestion` | Yes |

The engine of the crop dependency is a single parameter,
`water_requirement_mm = 1500` (plus `type = perennial` → 24-quincena window).
Only 2 of the 18 columns of `crop_parameters_260822.csv` feed the water
balance/WRSI; the other 16 feed the viability filter (`crop_viability.py`).

> **Known simplification:** `AWC_mm` sums the full 0–100 cm profile and
> ignores the crop's `root_depth_cm = 70`. This makes the storage bucket
> slightly more optimistic than reality for cacao.

---

## 5. References

- Allen, R.G., Pereira, L.S., Raes, D., & Smith, M. (1998). *Crop
  evapotranspiration: Guidelines for computing crop water requirements.*
  FAO Irrigation and Drainage Paper 56. Rome.
- Beguería, S., Vicente-Serrano, S.M., Reig, F., & Latorre, B. (2014).
  Standardized precipitation evapotranspiration index (SPEI) revisited:
  parameter fitting, evapotranspiration models, tools, datasets and drought
  monitoring. *International Journal of Climatology, 34*(10), 3001–3023.
  https://doi.org/10.1002/joc.3887
- Carr, M.K.V., & Lockwood, R. (2011). The water relations and irrigation
  requirements of cocoa (*Theobroma cacao* L.): A review. *Experimental
  Agriculture, 47*(4), 653–676.
- Cenicafé (2016). *Guía para el cultivo de café en Colombia.* Centro
  Nacional de Investigaciones de Café.
- DaMatta, F.M., & Ramalho, J.D.C. (2006). Impacts of drought and temperature
  stress on coffee physiology and production: A review. *Brazilian Journal of
  Plant Physiology, 18*(1), 55–81.
- Doorenbos, J., & Kassam, A.H. (1979). *Yield response to water.* FAO
  Irrigation and Drainage Paper 33. Rome.
- FAO (1976). *A Framework for Land Evaluation.* Soils Bulletin No. 32. Rome.
- FEWS NET (USGS). *Water Requirement Satisfaction Index (WRSI)* methodology
  and GeoWRSI tool documentation.
  https://help.fews.net/en/tools/v3/chapter-11-geowrsi
- Frère, M., & Popov, G.F. (1979). *Agrometeorological crop monitoring and
  forecasting.* FAO Plant Production and Protection Paper 17. Rome.
- Hosking, J.R.M., & Wallis, J.R. (1997). *Regional Frequency Analysis: An
  Approach Based on L-Moments.* Cambridge University Press.
- ICCO (2017). *Growing Cocoa.* International Cocoa Organization.
- Inman-Bamber, N.G., & Smith, D.M. (2005). Water relations in sugarcane and
  response to water deficits. *Field Crops Research, 92*(2–3), 185–202.
- Thornthwaite, C.W. (1948). An approach toward a rational classification of
  climate. *Geographical Review, 38*(1), 55–94.
- USDA Soil Taxonomy. Soil Survey Staff. Natural Resources Conservation
  Service.
- Vicente-Serrano, S.M., Beguería, S., & López-Moreno, J.I. (2010). A
  multi-scalar drought index sensitive to global warming: The Standardized
  Precipitation Evapotranspiration Index. *Journal of Climate, 23*(7),
  1696–1718. https://doi.org/10.1175/2009JCLI2909.1

### 5.1 Supporting literature for the accumulated-target design

- **End-of-season WRSI projection** (the standard "future" approach):
  An Improved Climatological Forecast Method for Projecting End-Of-Season
  Water Requirement Satisfaction Index (WRSI). ProQuest dissertation,
  https://www.proquest.com/docview/2455969164
- **ML for crop/water deficit forecasting:**
  Forecasting and Quantifying Risks of Crop and Water Supply Failures Using
  Machine Learning and Remote Sensing.
  https://discovery.researcher.life/article/forecasting-and-quantifying-risks-of-crop-and-water-supply-failures-using-machine-learning-and-remote-sensing/ea9e0cbc55e53d979a29522dcc494bec
- **ML irrigation management:** Scalable machine learning framework for
  adaptive irrigation management of maize and soybean in the U.S. Midwest.
  https://www.sciencedirect.com/science/article/pii/S0168169925008166
- **ML drought-index forecasting:** Characterization and forecasting of
  SPEI-based drought in Southern Telangana using statistical machine learning
  models. https://link.springer.com/article/10.1007/s44292-025-00070-6
