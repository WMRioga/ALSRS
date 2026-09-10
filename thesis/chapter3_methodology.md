# Chapter 3 — Research Methodology

> **Working draft.** Citations use IEEE-style numbered references, numbered
> in order of first appearance and consolidated in Chapter 7 (References). Numeric examples in this chapter are
> *illustrative* (rounded for pedagogy) unless stated otherwise. Study-area
> coordinates for wheat (Australia) are to be confirmed.

---

## 3.1 Overview and Research Workflow

The study follows a data-science pipeline of five stages, each feeding the next:

1. **Data acquisition** — open geospatial and remote-sensing data are retrieved
   through Google Earth Engine.
2. **Crop-viability screening** — each candidate location is filtered against
   crop-specific requirements (temperature, elevation, slope, soil).
3. **Physically-founded labelling** — a biweekly water balance (WRSI) converts
   the raw climate and soil inputs into a crop water-deficit label.
4. **Suitability scoring** — an AHP multi-criteria score quantifies overall
   land aptitude.
5. **Machine learning** — a two-stage (hurdle) Random Forest forecasts the
   deficit 1, 3 and 6 months ahead, and a post-hoc severity scheme produces the
   irrigation recommendation.

```mermaid
flowchart LR
    A["Data acquisition<br/>(GEE)"] --> B["Crop-viability<br/>screening"]
    B --> C["Water balance<br/>(WRSI labels)"]
    C --> D["AHP suitability<br/>scoring"]
    D --> E["Dataset<br/>(features + targets)"]
    E --> F["Hurdle model<br/>(classifier + regressor)"]
    F --> G["Recommendation<br/>(LOW/MODERATE/SEVERE)"]
```

> *(Render this as a figure for the final document.)*

**A walk-through example.** To fix ideas, consider a single cacao farm in
Colombia (e.g. *Finca Matanza*, 7.301, −73.010). The pipeline first pulls ten
years of biweekly climate and soil data for that coordinate from GEE, verifies
that the point falls within cacao's thermal and elevation tolerances, and then
computes, for each of the 240 biweekly periods, the water-balance deficit. The
result is a row per period: a feature vector of 17 variables known *at that
moment*, paired with the deficit accumulated over the *following* 1, 3 and 6
months. The machine-learning model is then trained on many such farms, and at
deployment time a new coordinate is pushed through the same stages to obtain a
LOW / MODERATE / SEVERE recommendation for the coming windows.

## 3.2 Study Area and Crop Selection

The ALSRS pipeline is parameterized for **seven crops** through a crop-parameter
table: cacao CCN-51, arabica coffee and plantain (Colombia); wheat, sorghum and
canola (Australia); and sugarcane (both countries). Each entry specifies the
crop's thermal and elevation tolerances, soil constraints, annual water
requirement, cycle length and root depth (Table 3.1). The crop-specific water
requirements and drought responses are substantiated by the agronomic
literature — cacao [32], sugarcane [33] and coffee [34], [35].

For the empirical validation, **three representative crops** spanning the two
case-study countries were selected:

| Crop | Type | Country | Water requirement | Cycle | Root depth |
|---|---|---|---|---|---|
| Cacao CCN-51 | perennial | Colombia | 1500 mm/yr | 24 quincenas | 70 cm |
| Wheat | annual | Australia | 550 mm/cycle | 9 quincenas | 110 cm |
| Sugarcane | perennial | Colombia + Australia | 2000 mm/yr | 24 quincenas | 100 cm |

The selection is deliberate: it spans a tropical perennial (cacao), a temperate
annual (wheat) and a crop grown in both countries (sugarcane). A successful
result on all three therefore provides evidence of transferability across
climate regimes and crop types, rather than a single-crop proof. It also
exercises both water-balance modes of the pipeline: the **perennial** mode (a
24-quincena cycle) and the **annual** mode (a crop-specific cycle — 9 quincenas
for wheat).

**Geographic context.** The Colombian case focuses on the cacao belt of the
north-west (e.g. the Urabá region of Antioquia, around Necoclí and El Playón),
a tropical, humid-to-seasonal climate. The Australian case targets the temperate
wheat belt and, for sugarcane, the sub-tropical coastal strip of Queensland
(e.g. around 19.7° S, 147.2° E). These two countries provide contrasting climate
regimes — the very contrast that makes the transferability claim meaningful.

## 3.3 Data Sources and Acquisition

All inputs are freely available and accessed through Google Earth Engine, with
server-side aggregation for efficiency (the pipeline maps over the point list
with `ee.List.map` and issues a single `getInfo` per stage). Table 3.2 summarises
the sources.

| Variable | Product | Native resolution | Coverage |
|---|---|---|---|
| Precipitation | CHIRPS [6] | daily, ~0.05° (≈5.5 km) | 1981–present |
| Temperature | ERA5-Land [7] | hourly, 0.1° (≈9 km) | 1950–present |
| ET (ET0/PET) | MODIS MOD16A2/GF [8] | 8-day, 500 m | 2000–present |
| Soil properties | SoilGrids 2.0 [9] | 250 m, 6 depths | static |
| Soil moisture (validation) | Sentinel-1 [22] | ~10 m SAR | 2014–present |
| Vegetation indices | Sentinel-2 (NDVI/EVI/NDWI) [23], [24], [25] | 10–20 m | 2015–present |
| ENSO teleconnection | ONI, NOAA CPC [3] | monthly | 1950–present |

Each variable is aggregated to **biweekly periods** (24 per year). For example,
the record labelled `2016-01_Q1` (the first quincena of January 2016) summarises
1–15 January: `precip_total_mm` is the *sum* of daily precipitation (an
accumulative variable), while `mean_C` is the *mean* of daily temperature. This
biweekly scale is a compromise: coarse enough to be robust to satellite noise
and cloud gaps, fine enough to be operationally useful for irrigation
scheduling. Two practical constraints follow from the products' coverage: the
harmonised Sentinel-2 archive begins in 2017 (so 2016 and early 2017 are missing
for those indices), and Sentinel-1 soil moisture is noisy at biweekly resolution,
so it is reserved for spot validation rather than used as a training feature.

## 3.4 Physically-Founded Labelling: the Water Balance (WRSI)

The irrigation-need label is not obtained from a statistical proxy but computed
from a physically interpretable water balance, following the Water Requirement
Satisfaction Index (WRSI) tradition [15], [16].

**Reference evapotranspiration.** ET0 is estimated with Thornthwaite's
temperature-based method [17] and stored as `pet_mm`; the FAO-56 formulation [2]
is the reference alternative when fuller meteorology is available.

**Crop evapotranspiration.** The crop water requirement is distributed across
the season in proportion to ET0:

```
ETc(t) = water_requirement_mm · ET0(t) / Σ ET0
```

so that the total ETc over the crop cycle equals the crop's annual (or seasonal)
requirement from the parameter table [2], [1]. For cacao (1500 mm/yr) with, say,
an annual ΣET0 of 2400 mm, a quincena with ET0 = 40 mm receives
`ETc = 1500 × 40 / 2400 = 25 mm`.

**Sequential soil-water balance.** For each biweekly period, actual
evapotranspiration and storage are updated with a bucket model:

```
AET(t)     = min(P(t) + Storage(t-1), ETc(t))
Storage(t) = min(AWC, Storage(t-1) + P(t) - AET(t))
```

where AWC is the available water capacity derived from soil texture and root
depth [9], [21]. The bucket means a crop can draw on stored soil moisture even
in a dry quincena, so deficit emerges only *after* the store is depleted.

**Worked example (illustrative).** Consider cacao with ETc = 25 mm per quincena
and AWC = 70 mm, and a dry spell with no rain:

| Period | P (mm) | Storage before | AET = min(P+storage, 25) | Deficit? | Storage after |
|---|---|---|---|---|---|
| t (rainy) | 30 | 50 | min(80, 25) = 25 | no | min(70, 55) = 55 |
| t+1 (dry) | 0 | 55 | min(55, 25) = 25 | no | 30 |
| t+2 (dry) | 0 | 30 | min(30, 25) = 25 | no | 5 |
| t+3 (dry) | 0 | 5 | min(5, 25) = 5 | **yes** (20 mm short) | 0 |

Notice how the storage buffer *delays* the onset of deficit: three consecutive
dry periods pass before the fourth one shows stress. This is exactly the
physically-grounded behaviour that a purely statistical label would miss.

**WRSI and deficit.** Over the one-month window (the last two quincenas, t+2 and
t+3):

```
WRSI    = Σ AET / Σ ETc = (25 + 5) / (25 + 25) = 0.60  (60%)
deficit = 100 · (Σ ETc - Σ AET) / Σ ETc = 100 · 20 / 50 = 40%
```

The WRSI is computed over a **one-month rolling window** (two biweeks). A longer
(12-month) window was trialled and found to flatten the seasonal signal, so it
was rejected in favour of one month. For perennial crops the cycle is 24
quincenas; for annual crops the cycle length from the parameter table is used
(wheat: 9 quincenas, matching its ~4.5-month growing season).

The rationale for this design is that the label is **physically founded** (a
water-balance deficit, not a statistical index) and therefore interpretable,
while still being cheaply computable from open data — a form of weak supervision
that avoids manual labelling.

## 3.5 Suitability Screening

Before the water balance, each candidate location is screened for basic
viability against the crop-parameter table (temperature range, elevation, slope,
pH, texture). Points that fall outside the crop's tolerances are flagged as not
viable.

**Example (cacao).** A point is viable only if its temperature range falls
within roughly 15–35 °C, elevation is 0–800 m, slope is below 8°, and soil pH is
in 5.5–7.0. A location at 2,000 m elevation would be flagged not viable for
cacao regardless of its rainfall, because cacao does not tolerate that altitude
range. This gate keeps the machine-learning problem focused on locations where
the crop is apt, so the model learns *when* irrigation is needed rather than
*whether* the land is suitable at all.

A separate AHP scoring step [5], following the FAO land-evaluation framework
[4], then combines soil, climate and terrain criteria into a single suitability
score using pairwise-comparison weights. The result is an ordinal aptitude score
that complements — but does not replace — the temporal deficit forecast.

## 3.6 Dataset Construction

The labelled dataset joins, for every point and biweekly period, the feature
vector (variables known at time `t`) with the forward-looking deficit targets.

**Features (17)** — all known at time `t`, so no future information is leaked:

```
month, biweek, mean_C, std_C, precip_total_mm, precip_rainy_days,
pet_mm, spei_1m, spei_3m, spei_6m, spei_12m, AWC_mm, Storage_mm,
P_acum_mm, WRSI_1m, deficit_1m, oni
```

Grouped by nature: **seasonality** (`month`, `biweek`); **climate** (`mean_C`,
`std_C`, `precip_total_mm`, `precip_rainy_days`); **calculated climate**
(`pet_mm`, the four `spei_*` drought indices [18], [19]); **soil** (`AWC_mm`);
**water-balance state** (`Storage_mm`, `P_acum_mm`, `WRSI_1m`, `deficit_1m`);
and **ENSO** (`oni` [3]). The SPEI features supply a multi-scalar
meteorological drought signal, and `oni` a large-scale climate teleconnection.
Identity and location columns (`point_id`, `lat`, `lon`, `crop`, dates, `label`)
are deliberately excluded to avoid leaking farm identity or location.

**Targets (3)** — the accumulated *future* deficit over the next 2, 6 and 12
biweeks (1, 3 and 6 months):

```
future_deficit_H(t) = 100 · Σ(ETc - AET over [t+1 .. t+H]) / Σ(ETc over [t+1 .. t+H])
```

The three nested horizons give a coarse stress trajectory (short / medium / long
term). The targets are zero-inflated: in the cacao dataset, 73%, 56% and 39% of
records have zero deficit at 1, 3 and 6 months, respectively. This zero-inflation
is the defining property that motivates the two-stage model of Section 3.7.

**Split.** Data are split **by farm (point)**, not by row: 80/20 with seed 42,
so that a farm's entire time series lies entirely in train or entirely in test.
This prevents the temporal autocorrelation within a farm from leaking into the
evaluation, and matches the real use case of generalising to an unseen farm.

## 3.7 Modelling Approach

This section describes the core methodological contribution of the study: a
**two-stage (hurdle) model with a gate**, designed for the zero-inflated water
deficit. It is presented in four parts — the single-regressor baseline (as the
contrast), the hurdle structure, the gate, and the post-hoc severity
classification — with the rationale for each design decision.

### 3.7.1 Baseline: single regressor

The baseline is one `RandomForestRegressor` [10] per horizon, trained on all
rows with `sample_weight = deficit` (zero-deficit rows receive no weight, so the
model focuses on the rows that actually exhibit stress). This single model must
learn two things at once — *whether* a deficit will occur and *how large* it
will be. As argued in Chapter 1, it performs both poorly: it emits a "phantom
deficit" on the many zero rows and under-predicts the rare severe tail. The
hurdle below is the remedy.

### 3.7.2 The hurdle (two-part) model

**Theoretical foundation.** The deficit is a *semicontinuous* outcome: it has a
probability mass at zero (no deficit) and a continuous, right-skewed positive
tail (a deficit of some size). For such variables, a single regression equation
confounds two logically distinct processes. The two-part — or hurdle — model
addresses this by decomposing the conditional mean into the product of an
*occurrence* component and a *magnitude* component [11], [12], [13]:

```
E[Y | x]  =  P(Y > 0 | x)  ·  E[Y | Y > 0, x]
             (occurrence)      (magnitude, given occurrence)
```

This decomposition originates in Cragg's two-part model for limited dependent
variables [11], was formalised as the "hurdle" formulation — a binary event that
must first be *cleared* — by Mullahy [12], and is paralleled by Lambert's
zero-inflated Poisson model for count data [13]. The key insight is that the
drivers of *occurrence* and of *magnitude* need not be the same, so they should
be modelled by separate functions rather than forced through one.

**Why it fits this problem.** A water deficit arises through a threshold
process: below a certain dryness, the crop's demand is met (zero deficit); above
it, a shortfall appears and grows with the severity of the dry spell. The
question "will there be a deficit?" is therefore conceptually different from
"how large will it be?", and the two may respond to different features. A single
regressor cannot represent this separation, which is precisely why it produces
phantom deficits and a diluted tail.

**Concrete instantiation.** Following the two-part structure, **two** Random
Forest models are fitted per horizon [10]:

- **Stage 1 — classifier** (`RandomForestClassifier`): estimates
  `P(deficit > 0)` on *all* rows, with `class_weight = "balanced"` so the
  minority positive class is not ignored. It answers *"is there a deficit?"*.
- **Stage 2 — regressor** (`RandomForestRegressor`): trained *only* on the rows
  with `deficit > 0`, with `sample_weight = deficit` so the rare extreme events
  dominate the fit. It answers *"given a deficit, how much?"*.

Each stage specialises on a single, simpler task, yielding six models in total
(two per horizon). This transplant of a classical econometric structure into a
machine-learning, remote-sensing setting is consistent with the recent adoption
of two-stage ML frameworks for zero-inflated, highly-skewed precipitation [29].

### 3.7.3 The gate: preserving the severe cases

**The shrinkage problem.** The decomposition above suggests combining the two
stages as the *expected* value `E[Y] = P(deficit>0) · magnitude`. This is the
correct quantity if the goal is to minimise squared error on the *average*
deficit. It is, however, the wrong quantity for an *irrigation decision*, for a
simple reason: because the probability `P(deficit>0)` is less than one, the
multiplication **systematically attenuates the magnitude** [11]. A genuinely
severe deficit is reported as only moderately severe, because its magnitude is
scaled down by the (uncertain) probability that it will occur at all. In the
study's data this attenuation is precisely what downgrades the SEVERE class that
the system exists to flag.

**The gate.** To avoid this shrinkage, the study replaces the probabilistic
expectation with a **thresholded decision rule** (the "gate"):

```
prediction = magnitude   if P(deficit > 0) >= 0.5
prediction = 0           otherwise
```

Stage 1 now serves only as a *switch*: it decides whether a deficit is
sufficiently likely, and if so, stage 2's full magnitude is used *unshrunken*;
otherwise the prediction is zero. This keeps the large MAE gain that comes from
correctly nailing the many zero rows, while **recovering the recall of the
severe classes** that the naive `P · magnitude` combination loses.

**Why a gate is justified.** The choice of a decision rule over an expectation
reflects an *asymmetric loss*: in irrigation, failing to flag a genuine severe
deficit (a false negative) is costlier than raising a false alarm. This is the
same principle that motivates cost-sensitive and imbalanced-learning methods
[30] and decision-relevant, recall-based evaluation [31]. The two-part structure
itself is drawn from the literature [11], [12], [13]; the gate is the study's
adaptation of that structure to a *decision* setting, in the same spirit as
two-stage prediction frameworks for skewed environmental variables [29].

**Worked example (illustrative).** Suppose three periods produce:

| Case | P(deficit>0) | magnitude (%) | `P·magnitude` | gated | Effect |
|---|---|---|---|---|---|
| A (severe) | 0.75 | 64 | 48 → MODERATE | **64 → SEVERE** | the gate rescues a severe case |
| B (moderate) | 0.85 | 38 | 32 → MODERATE | **38 → MODERATE** | no change |
| C (none) | 0.10 | 30 | 3 → LOW | **0 → LOW** | both agree |

Case A is the crucial one: multiplying by 0.75 drags a genuine 64% deficit down
to 48%, misclassifying it as merely MODERATE; the gate keeps the full 64% and so
preserves the SEVERE flag. Case C shows the gate still produces 0 (and therefore
LOW) when a deficit is unlikely, preserving the accuracy on the zero rows. The
threshold (0.5) is selected by sweeping the trade-off between recall and false
alarms on the binary "needs irrigation" decision.

### 3.7.4 Post-hoc severity classification

Because the model is a **regressor**, the class is derived after prediction by
applying configurable thresholds — an *irrigation policy*, not a property of the
model. The adopted scheme is three classes:

| Deficit | Class |
|---|---|
| ≤ 15 | LOW |
| 15–50 | MODERATE |
| > 50 | SEVERE |

The four-class variant (15/30/50) and two-class variants remain available without
retraining — a continuous prediction of, say, 38.96 maps to MODERATE under the
three-class scheme but to HIGH under the four-class scheme, and the same
prediction is re-binned *without re-fitting the model*. This separation of the
physical model from the decision rule is a deliberate design choice: the
thresholds encode the project's irrigation policy and can be changed by the
decision-maker without touching the model. The final `suggestion` is the **worst**
class across the three horizons, so a short but intense drought still flags
SEVERE even if the six-month average looks mild.

**Random Forest configuration.** Both stages use `n_estimators = 300`,
`max_depth = None`, `min_samples_leaf = 1`, `random_state = 42` and
`n_jobs = -1`.

## 3.8 Evaluation and Validation Strategy

**Cross-validation.** The official metric is obtained by five-fold
`GroupKFold` grouped by `point_id`, so that no farm appears in both training and
validation folds. Concretely, the ~239 farms are partitioned into five groups of
≈48 farms; the model is trained on four groups (~191 farms) and evaluated on the
held-out group, rotating until every group has been evaluated once, and the five
MAE/RMSE values are averaged. This is stricter than a single 80/20 holdout and
removes the leakage risk of splitting rows within a farm.

**Regression metrics.** MAE and RMSE are reported per horizon, against a
**persistence baseline** (the same-window backward deficit, i.e. "tomorrow ≈
today"), plus mean and median baselines, to establish that the model beats
simple, honest references.

**Decision-relevant metrics.** Because the target is zero-inflated, aggregate
regression error alone is misleading [30]. The model is therefore also evaluated
through the class-level **recall** and **cumulative recall** (≥ MODERATE,
≥ SEVERE) of the severity scheme, and through **confusion matrices** — the
metrics that matter for an irrigation decision, following precision-and-recall
formulations for regression [31]. For example, a recall of 0.83 for MODERATE
means that, of all periods that truly needed irrigation, 83% were flagged
correctly; the remaining 17% were under-flagged — a cost the decision-maker can
weigh against the false-alarm rate.

## 3.9 Tools, Technologies and Reproducibility

- **Language and libraries:** Python 3.10 with scikit-learn, pandas, NumPy,
  Matplotlib and Seaborn, in a dedicated conda environment (`agri_land_env`).
- **Geospatial compute:** Google Earth Engine, with server-side aggregation
  (`ee.List.map` and a single `getInfo`) for efficiency.
- **Reproducibility:** fixed random seed (42), a versioned `manifest.json`
  recording all model hyperparameters and thresholds, and trained models saved
  with `joblib` (excluded from version control as regenerable artefacts).
- **Data ethics and privacy:** only openly published environmental datasets are
  used; farm records are georeferenced coordinates with no personally
  identifiable information.
