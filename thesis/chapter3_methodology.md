# Chapter 3 — Research Methodology

## 3.1 Overview and Research Workflow

The study follows a data science pipeline of five stages, each of which feeds into the next:

1. Data collection. Open geospatial and remote-sensing data are obteined through Google Earth Engine.

2. Crop-viability screening. Each candidate location is filtered based on crop-specific requirements (temperature, elevation, slope, soil).

3. Physically based labelling. A biweekly water balance (WRSI) converts the raw climate and soil inputs into a crop water-deficit label.

4. Suitability scoring. An AHP multi-criteria score quantifies overall land aptitude.

5. Machine learning. A two-stage (hurdle) Random Forest forecasts the deficit 1, 3 and 6 months ahead, and a post-hoc severity scheme produces the irrigation recommendation.

```
flowchart LR    
    A["Data acquisition<br/>(GEE)"] --> B["Crop-viability<br/>screening"]    
    B --> C["Water balance<br/>(WRSI labels)"]    
    C --> D["AHP suitability<br/>scoring"]    
    D --> E["Dataset<br/>(features + targets)"]    
    E --> F["Hurdle model<br/>(classifier + regressor)"]    
    F --> G["Recommendation<br/>(LOW/MODERATE/SEVERE)"]
```

> *(Render this as a figure for the final document.)*

A walk-through example helps fix the idea. Consider a single cacao farm in Colombia, for instance Finca Matanza at 7.301, −73.010. The pipeline first pulls ten years of biweekly climate and soil data for that coordinate from GEE. It then verifies that the point falls within cacao's thermal and elevation tolerances, and computes, for each of the 240 biweekly periods, the water-balance deficit. The result is one row per period: a feature vector of 17 variables known at that moment, paired with the deficit accumulated over the following 1, 3 and 6 months. The machine-learning model is trained on many such farms. At deployment, a new coordinate is pushed through the same stages to obtain a LOW / MODERATE / SEVERE recommendation for the coming windows.

The implementation of each stage is documented in the appendices: the dataset dictionary (Appendix A), the model documentation (Appendix B), the analysis notebooks (Appendix C), the pipeline source code (Appendix D) and the configuration data (Appendix E).

## 3.2 Study Area and Crop Selection

The ALSRS pipeline is parameterised for seven crops through a crop-parameter table: cacao CCN-51, arabica coffee and plantain (Colombia); wheat, sorghum and canola (Australia); and sugarcane (both countries). Each entry specifies the crop's thermal and elevation tolerances, soil constraints, annual water requirement, cycle length and root depth (Table 3.1). The crop-specific water requirements and drought responses are supported by the agronomic literature: cacao [32], sugarcane [33] and wheat [34].

For the empirical validation, three representative crops spanning the two case-study countries were selected:

| Crop | Type | Country | Water requirement | Cycle | Root depth |
| - | - | - | - | - | - |
| Cacao CCN-51 | perennial | Colombia | 1500 mm/yr | 24 quincenas | 70 cm |
| Wheat | annual | Australia | 550 mm/cycle | 9 quincenas | 110 cm |
| Sugarcane | perennial | Colombia + Australia | 2000 mm/yr | 24 quincenas | 100 cm |


The selection covers a tropical perennial (cacao), a temperate annual (wheat) and a crop grown in both countries (sugarcane). Of these, cacao and sugarcane were validated in full; the wheat case was attempted but the model has not yet been adjusted to annual crops, and this is treated as a limitation in Chapter 5. The choice also exercises both water-balance modes of the pipeline: the perennial mode (a 24-quincena cycle) and the annual mode (a crop-specific cycle, 9 quincenas for wheat).

Geographic context. The Colombian case focuses on the cacao belt of the north-west, for example the Urabá region of Antioquia around Necoclí and El Playón, with a tropical, humid-to-seasonal climate. The Australian case targets the temperate wheat belt and, for sugarcane, the sub-tropical coastal strip of Queensland (around 19.7° S, 147.2° E). These two countries give contrasting climate regimes, which is exactly the contrast that makes the transferability claim meaningful.

## 3.3 Data Sources and Acquisition

All inputs are freely available and accessed through Google Earth Engine, with server-side aggregation for efficiency: the pipeline maps over the point list with `ee.List.map` and issues a single `getInfo` per stage. Table 3.2 summarises the sources.

| Variable | Product | Native resolution | Coverage |
| - | - | - | - |
| Precipitation | CHIRPS [6] | daily, ~0.05° (≈5.5 km) | 1981–present |
| Temperature | ERA5-Land [7] | hourly, 0.1° (≈9 km) | 1950–present |
| ET (ET0/PET) | MODIS MOD16A2/GF [8] | 8-day, 500 m | 2000–present |
| Soil properties | SoilGrids 2.0 [9] | 250 m, 6 depths | static |
| Soil moisture (validation) | Sentinel-1 [22] | ~10 m SAR | 2014–present |
| Vegetation indices | Sentinel-2 (NDVI/EVI/NDWI) [23], [24], [25] | 10–20 m | 2015–present |
| ENSO teleconnection | ONI, NOAA CPC [3] | monthly | 1950–present |


Each variable is aggregated to biweekly periods, 24 per year. For example, the record labelled `2016-01_Q1` (the first quincena of January 2016) summarises 1–15 January: `precip_total_mm` is the sum of daily precipitation (an accumulative variable), while `mean_C` is the mean of daily temperature. This biweekly scale is a compromise. It is coarse enough to be robust to satellite noise and cloud gaps, and fine enough to be operationally useful for irrigation scheduling. Two practical constraints follow from the products' coverage. The harmonised Sentinel-2 archive begins in 2017, so 2016 and early 2017 are missing for those indices. Sentinel-1 soil moisture is noisy at biweekly resolution, so it is reserved for spot validation rather than used as a training feature.

## 3.4 Physically Based Labelling: the Water Balance (WRSI)

The irrigation-need label is not obtained from a statistical proxy. It is computed from a physically interpretable water balance, following the Water Requirement Satisfaction Index (WRSI) tradition [15], [16].

Reference evapotranspiration. ET0 is estimated with Thornthwaite's temperature-based method [17] and stored as `pet_mm`. The FAO-56 formulation [2] is the reference alternative when fuller meteorology is available.

Crop evapotranspiration. The crop water requirement is distributed across the season in proportion to ET0:

```
ETc(t) = water_requirement_mm · ET0(t) / Σ ET0
```

so that the total ETc over the crop cycle equals the crop's annual (or seasonal) requirement from the parameter table [2], [1]. For cacao (1500 mm/yr) with, say, an annual ΣET0 of 2400 mm, a quincena with ET0 = 40 mm receives `ETc = 1500 × 40 / 2400 = 25 mm`.

Sequential soil-water balance. For each biweekly period, actual evapotranspiration and storage are updated with a bucket model:

```
AET(t)     = min(P(t) + Storage(t-1), ETc(t))    
Storage(t) = min(AWC, Storage(t-1) + P(t) - AET(t))
```

where AWC is the available water capacity derived from soil texture and root depth [9], [21]. The bucket means a crop can draw on stored soil moisture even in a dry quincena, so the deficit appears only after the store is depleted.

Worked example (illustrative). Consider cacao with ETc = 25 mm per quincena and AWC = 70 mm, and a dry spell with no rain:

| Period | P (mm) | Storage before | AET = min(P+storage, 25) | Deficit? | Storage after |
| - | - | - | - | - | - |
| t (rainy) | 30 | 50 | min(80, 25) = 25 | no | min(70, 55) = 55 |
| t+1 (dry) | 0 | 55 | min(55, 25) = 25 | no | 30 |
| t+2 (dry) | 0 | 30 | min(30, 25) = 25 | no | 5 |
| t+3 (dry) | 0 | 5 | min(5, 25) = 5 | yes (20 mm short) | 0 |


Notice how the storage buffer delays the onset of deficit: three dry periods pass before the fourth one shows stress. This is exactly the physically grounded behaviour that a purely statistical label would miss.

WRSI and deficit. Over the one-month window (the last two quincenas, t+2 and t+3):

```
WRSI    = Σ AET / Σ ETc = (25 + 5) / (25 + 25) = 0.60  (60%)    
deficit = 100 · (Σ ETc - Σ AET) / Σ ETc = 100 · 20 / 50 = 40%
```

The WRSI is computed over a one-month rolling window (two biweeks). A longer 12-month window was tried and found to flatten the seasonal signal. The wet/dry seasonality, for example Feb ≈ 83% deficit versus Jun ≈ 0% in the dry belt, averages out to a nearly flat ~9%. So the longer window was rejected in favour of one month, and this analysis is documented in the dataset dictionary (Appendix A). For perennial crops the cycle is 24 quincenas; for annual crops the cycle length from the parameter table is used (wheat: 9 quincenas, matching its ~4.5-month growing season).

The rationale for this design is that the label is physically based, a water-balance deficit rather than a statistical index, and therefore interpretable, while still being cheaply computable from open data. It is a form of weak supervision that avoids manual labelling.

## 3.5 Suitability Screening

Before the water balance, each candidate location passes through two complementary suitability checks: a hard viability screen against the crop-parameter table, and a soft, weighted suitability score computed with the Analytic Hierarchy Process (AHP) [5].

### 3.5.1 Crop-viability screening

Each candidate location is first screened against the crop-parameter table (temperature range, elevation, slope, pH, texture). Points that fall outside the crop's tolerances are flagged as not viable.

Example (cacao). A point is viable only if its temperature range falls within roughly 15–35 °C, elevation is 0–800 m, slope is below 8°, and soil pH is in 5.5–7.0. A location at 2,000 m elevation would be flagged not viable for cacao regardless of its rainfall, because cacao does not tolerate that altitude. This gate keeps the machine-learning problem focused on locations where the crop is apt, so the model learns when irrigation is needed rather than whether the land is suitable at all.

### 3.5.2 AHP suitability scoring: pairwise comparisons and weights

Locations that pass the screen are further scored across seven sub-criteria under four criteria, climate (temperature), soil (texture, pH, SOC), water (WRSI) and terrain (slope, elevation), following the FAO land-evaluation framework [4]. Each criterion's weight comes from a Saaty pairwise-comparison matrix [5], solved for its principal eigenvector, with a Consistency Ratio (CR) verifying the judgements are not self-contradictory (CR < 0.10 is the accepted threshold [5]); each judgement below is grounded in literature already used in this study, not asserted as unsupported opinion.

Climate, soil and water are judged equally important (FAO [4] treats them as parallel, co-primary land qualities), and each is judged moderately more important than terrain (value 3): water deficit is a direct, quantified yield determinant [1], whereas terrain is largely a management constraint already addressed upstream; elevation is a temperature proxy (Section 3.6) and severe slope is removed by the viability screen. Within soil, texture is judged slightly more important than pH, and pH than SOC, since texture directly drives the available-water-capacity calculation [21], while SOC is the most correctable through management. Within terrain, slope is judged slightly more important than elevation, consistent with elevation's redundant role as a temperature proxy. All three matrices are perfectly consistent (CR = 0.00).

| Criterion | Weight | Sub-criterion | Local weight | **Global weight** |
| - | - | - | - | :-: |
| Climate | 3.000 | Temperature | 1.000 | **0.300** |
| Soil | 3.000 | Texture | 0.571 | **0.171** |
|  |  | pH | 0.286 | **0.086** |
|  |  | SOC | 0.143 | **0.043** |
| Water | 3.000 | WRSI | 1.000 | **0.300** |
| Terrain | 1.000 | Slope | 0.667 | **0.067** |
|  |  | Elevation | 0.333 | **0.033** |


Each sub-criterion is scored in the range 0–1 (temperature as the mean per-period `temp_score`, with the extreme-period exception described below; elevation and slope as ramp functions around the crop's tolerance limits; texture and pH as ramp functions around their limiting values; SOC relative to its minimum; WRSI as the proportion of labelled periods with `deficit_1m ≤ 30`). The final suitability score is the weighted sum of these seven sub-criterion scores, mapped to the FAO suitability classes S1/S2/S3/N [4].

**Limiting-factor rule.** The weighted sum alone would allow a severe deficiency in one criterion to be compensated by strong scores elsewhere, which is agronomically unrealistic: a location with unworkable texture is not "rescued" by an excellent water balance. The scoring therefore applies a limiting-factor override on top of the weighted sum: a zero score in a hard criterion (texture or temperature) caps the class to N regardless of the weighted total, while a zero slope score caps the class to S3 and flags the location for field verification. Elevation remains a soft criterion throughout, consistent with its treatment elsewhere in this study as a proxy rather than a direct constraint, lowering the score without ever hard-failing the location, and emitting a "validate with local conditions" warning when a point falls outside the literature-reported elevation range. The temperature criterion carries its own graded exception: zero extreme periods (temperature outside the crop's tolerable range) contribute no penalty beyond the mean `temp_score`; exactly one extreme period adds a field-validation warning without penalising the class; more than one extreme period sets the temperature score to zero, which, being a hard criterion, caps the class to N.

Because classification depends heavily on these limiting-factor rules rather than the weighted sum alone, the final class is, by design, comparatively insensitive to the exact weights above; a sensitivity analysis quantifying this is reported in Chapter 4.

## 3.6 Dataset Construction

The labelled dataset joins, for every point and biweekly period, the feature vector (variables known at time `t`) with the forward-looking deficit targets.

Features (17). All are known at time `t`, so no future information is leaked:

```
month, biweek, mean_C, std_C, precip_total_mm, precip_rainy_days,    
pet_mm, spei_1m, spei_3m, spei_6m, spei_12m, AWC_mm, Storage_mm,    
P_acum_mm, WRSI_1m, deficit_1m, oni
```

Grouped by nature: seasonality (`month`, `biweek`); climate (`mean_C`, `std_C`, `precip_total_mm`, `precip_rainy_days`); calculated climate (`pet_mm`, the four `spei_\\\*` drought indices [18], [19]); soil (`AWC_mm`); water-balance state (`Storage_mm`, `P_acum_mm`, `WRSI_1m`, `deficit_1m`); and ENSO (`oni` [3]). The SPEI features supply a multi-scalar meteorological drought signal, and `oni` a large-scale climate teleconnection.

**Excluded variables and their individual justification.** Identity and location columns (`point_id`, `lat`, `lon`, dates, `label`), the categorical `crop`, and terrain (elevation, slope) are excluded from the feature set, each for a distinct reason rather than a single blanket rule:

- `lat`/`lon` are excluded so that the model learns the physical relationship between climate, soil and water need, a relationship intended to generalise to any new coordinate, rather than memorising the identity of the farms seen during training.

- `crop` is excluded because each crop is modelled with its own dataset and its own water-balance calculation, parameterised by that crop's specific annual water requirement (Table 3.1). Crop identity is therefore already embedded in the computed `WRSI_1m` and `deficit_1m` values, and would carry no additional information within a single-crop dataset.

- Elevation and slope, although used in the viability screening of Section 3.5, are excluded from the predictive features on different grounds.

Elevation is treated as a soft proxy for temperature rather than an independent driver of water deficit, since temperature is already captured directly through `mean_C` and `std_C`. Using the proxy instead of the direct measurement risks systematic error. The study's own reference plot (El Playón, Santander, ~1,000 m) exceeds the crop's nominal 800 m elevation ceiling yet stays thermally viable, which shows how weak the proxy can be.

Slope is excluded as a scope decision. It is treated as a physical and agronomic constraint already managed by the farmer when deciding whether to cultivate a plot at all, not as a variable the irrigation-forecasting model should infer. Its hydrological effect, surface run-off reducing effective infiltration, is acknowledged as an unmodelled limitation in Chapter 5.

Targets (3). The accumulated future deficit over the next 2, 6 and 12 biweeks (1, 3 and 6 months):

```
future_deficit_H(t) = 100 · Σ(ETc - AET over [t+1 .. t+H]) / Σ(ETc over [t+1 .. t+H])
```

The three nested horizons give a coarse stress trajectory (short, medium and long term). The targets are zero-inflated, but to different degrees by crop. For cacao, 72.8%, 56.4% and 39.2% of records have zero deficit at 1, 3 and 6 months, respectively. For sugarcane the same figures are 35.0%, 17.4% and 6.9%. This zero-inflation, and its variation across crops, is the defining property that motivates the two-stage model of Section 3.7.

**Sampling strategy for climatic variance.** An initial dataset restricted to the traditional cacao belt (humid-to-intermediate climate zones) produced an even more strongly zero-inflated target than the figures above, largely because the belt is, by construction, where the crop already thrives. Deficit events are rare almost by definition of the sampling frame. To increase deficit variance without compromising agronomic validity, additional points were added in marginal-but-thermally-viable dry zones: the upper Magdalena valley (Tolima–Huila), the dry Caribbean around Valledupar (Cesar), and the Chicamocha canyon. Each of these falls within cacao's tolerable temperature and elevation range but experiences markedly lower or more erratic rainfall than the traditional belt. Locations outside the crop's climatic envelope entirely, for example Mediterranean or desert regions with unrelated crop-suitability profiles, were deliberately excluded. Including them would answer a different question, whether the crop can grow there at all, rather than the target question of irrigation need for an already-viable crop.

Split. Data are split by farm (point), not by row, into three disjoint sets with seed 42: 70% train, 10% validation and 20% test. For cacao this gives 167, 24 and 48 farms, respectively; for sugarcane, 238, 34 and 68. A farm's entire time series lies in exactly one set, which prevents the temporal autocorrelation within a farm from leaking across sets. The three sets have distinct roles: train fits the model, validation selects the design decisions (gate threshold, class scheme), and test is held out untouched to report the final, unbiased performance once. The split is additionally stratified by climate zone (dry / intermediate / humid / very humid) so that each zone is represented in both training and test. This guards against, for example, an entire naturally-humid zone falling into training and none into test, which would prevent evaluating the model's behaviour in that zone under anomalous dry conditions such as an El Niño episode.

## 3.7 Modelling Approach

This section describes the core methodological contribution of the study: a two-stage (hurdle) model with a gate, designed for the zero-inflated water deficit. It is presented in four parts: the single-regressor baseline (as the contrast), the hurdle structure, the gate, and the post-hoc severity classification, with the rationale for each design decision.

### 3.7.1 Baseline: single regressor

The baseline is one `RandomForestRegressor` [10] per horizon, trained on all rows with `sample_weight = deficit`. Zero-deficit rows receive no weight, so the model focuses on the rows that actually show stress. The linear (`deficit`) weighting was chosen by explicit comparison against squared (`deficit²`) and plus-1-squared (`1 + deficit²`) alternatives on the validation set. Squared is dominated outright by linear, worse on both MAE and severe-case recall, and is not a useful compromise between the other two. The remaining pair traces a genuine trade-off: plus-1-squared gives the lowest MAE, because it also fits the zero rows closely, but it gives the lowest severe-case recall; linear gives the highest SEVERE recall, at the cost of a higher MAE, because it de-emphasises the zero rows in favour of the tail. Linear was adopted because the severe cases are the decision-relevant ones (Appendix B). This single model must learn two things at once: whether a deficit will occur, and how large it will be. It cannot occupy both ends of this trade-off at once. It either emits a phantom deficit on the many zero rows, or under-predicts the rare severe tail, depending on which weighting is chosen. The hurdle below is the remedy.

### 3.7.2 The hurdle model

Theoretical foundation. The deficit is a semicontinuous outcome: it has a probability mass at zero (no deficit) and a continuous, right-skewed positive tail (a deficit of some size). For such variables, a single regression equation confounds two logically distinct processes. The two-part, or hurdle, model addresses this by decomposing the conditional mean into the product of an occurrence component and a magnitude component [11], [12], [13]:

```
E[Y | x]  =  P(Y > 0 | x)  ·  E[Y | Y > 0, x]    
             (occurrence)      (magnitude, given occurrence)
```

This decomposition originates in Cragg's two-part model for limited dependent variables [11], was formalised as the hurdle formulation, a binary event that must first be cleared, by Mullahy [12], and is paralleled by Lambert's zero-inflated Poisson model for count data [13]. The key insight is that the drivers of occurrence and of magnitude need not be the same, so they should be modelled by separate functions rather than forced through one.

Why it fits this problem. A water deficit arises through a threshold process. Below a certain dryness, the crop's demand is met and the deficit is zero. Above it, a shortfall appears and grows with the severity of the dry spell. The question "will there be a deficit?" is therefore conceptually different from "how large will it be?", and the two may respond to different features. A single regressor cannot represent this separation, which is exactly why it produces phantom deficits and a diluted tail.

Concrete instantiation. Following the two-part structure, two Random Forest models are fitted per horizon [10]:

- Stage 1, classifier (`RandomForestClassifier`): estimates `P(deficit > 0)` on all rows, with `class_weight = "balanced"` so the minority positive class is not ignored. It answers "is there a deficit?".

- Stage 2, regressor (`RandomForestRegressor`): trained only on the rows with `deficit > 0`, with `sample_weight = deficit` so the rare extreme events dominate the fit. It answers "given a deficit, how much?".

Each stage is trained to answer a single, simpler question, giving six models in total, two per horizon. Whether each stage's specialisation translates into a measurable accuracy gain over the single-regressor baseline is examined empirically in Chapter 4 and discussed in Chapter 5. This transplant of a classical econometric structure into a machine-learning, remote-sensing setting is consistent with the recent adoption of two-stage ML frameworks for zero-inflated, highly-skewed precipitation [29].

### 3.7.3 The gate: preserving the severe cases

The shrinkage problem. The decomposition above suggests combining the two stages as the expected value `E[Y] = P(deficit>0) · magnitude`. This is the correct quantity if the goal is to minimise squared error on the average deficit. It is, however, the wrong quantity for an irrigation decision, for a simple reason: because the probability `P(deficit>0)` is less than one, the multiplication systematically attenuates the magnitude [11]. A genuinely severe deficit is reported as only moderately severe, because its magnitude is scaled down by the uncertain probability that it will occur at all. In the study's data this attenuation is precisely what downgrades the SEVERE class that the system exists to flag.

The gate. To avoid this shrinkage, the study replaces the probabilistic expectation with a thresholded decision rule, the gate:

```
prediction = magnitude   if P(deficit > 0) >= threshold    
prediction = 0           otherwise
```

Stage 1 now serves only as a switch: it decides whether a deficit is sufficiently likely, and if so, stage 2's full magnitude is used unshrunken; otherwise the prediction is zero. This keeps the large MAE gain from correctly nailing the many zero rows, while recovering the recall of the severe classes that the naive `P · magnitude` combination loses.

The threshold is tuned per crop. For cacao the value is 0.5, and for sugarcane 0.75. This is itself a finding: the same model structure transfers from one perennial to another with only a change in the gate threshold, as the crop-specific zero-inflation profile changes.

Two clarifications avoid a common confusion about the gate. First, at inference the regressor computes a magnitude for every row, it is only trained on the positive rows; the gate is what discards that magnitude when the classifier is not confident. Second, the gate is a fixed rule, never fitted: the zeros it produces are model outputs, not training data, and they enter no further learning.

Why a gate is justified. The choice of a decision rule over an expectation reflects an asymmetric loss. In irrigation, failing to flag a genuine severe deficit (a false negative) is costlier than raising a false alarm. This is the same principle that motivates cost-sensitive and imbalanced-learning methods [30] and decision-relevant, recall-based evaluation [31]. The two-part structure itself is drawn from the literature [11], [12], [13]; the gate is the study's adaptation of that structure to a decision setting, in the same spirit as two-stage prediction frameworks for skewed environmental variables [29].

Worked example (illustrative). Suppose three periods produce:

| Case | P(deficit>0) | magnitude (%) | `P·magnitude` | gated | Effect |
| - | - | - | :-: | - | - |
| A (severe) | 0.75 | 64 | 48 → MODERATE | 64 → SEVERE | the gate rescues a severe case |
| B (moderate) | 0.85 | 38 | 32 → MODERATE | 38 → MODERATE | no change |
| C (none) | 0.10 | 30 | 3 → LOW | 0 → LOW | both agree |


Case A is the crucial one: multiplying by 0.75 drags a genuine 64% deficit down to 48%, misclassifying it as merely MODERATE; the gate keeps the full 64% and so preserves the SEVERE flag. Case C shows the gate still produces 0, and therefore LOW, when a deficit is unlikely, preserving the accuracy on the zero rows. The threshold is selected by sweeping the trade-off between recall and false alarms on the binary "needs irrigation" decision.

### 3.7.4 Post-hoc severity classification

Because the model is a regressor, the class is derived after prediction by applying configurable thresholds. This is an irrigation policy, not a property of the model. The adopted scheme is three classes:

| Deficit | Class |
| - | - |
| ≤ 15 | LOW |
| 15–50 | MODERATE |
| > 50 | SEVERE |


**Why three classes rather than four.** An initial four-class scheme (LOW/MEDIUM/HIGH/NOT_SUITABLE at 15/30/50) was compared against the three-class scheme, and against a direct `RandomForestClassifier` trained on the four-class label rather than derived from the regression, in an exploratory by-farm holdout. The two approaches fail in opposite directions. The hurdle regressor under-predicts the tail, reading SEVERE cases down to MODERATE, but is accurate in the middle of the distribution. The direct classifier is the opposite: strong at the extremes but weak in the middle, because its errors on MEDIUM and HIGH leak outward to LOW rather than into the neighbouring class.

Merging MEDIUM and HIGH into MODERATE therefore helps the hurdle model, whose dominant confusion sits between those two classes, but it does not repair the direct classifier, whose error escapes outside the pair being merged. The three-class scheme is adopted for the hurdle model specifically, not as a universally superior choice. A hybrid that takes the more severe of the two predictions is noted as future work in Chapter 6, since the two approaches look complementary. The supporting recall figures are reported in Chapter 4 (Table 4.10).

Because thresholding is applied after prediction, the four-class and two-class variants remain available without retraining. A continuous prediction of, say, 38.96 maps to MODERATE under the three-class scheme but to HIGH under the four-class scheme, and the same prediction is re-binned without re-fitting the model. This separation of the physical model from the decision rule is a deliberate design choice: the thresholds encode the project's irrigation policy and can be changed by the decision-maker without touching the model. The final `suggestion` is the worst class across the three horizons, so a short but intense drought still flags SEVERE even if the six-month average looks mild.

Random Forest configuration. Both stages use `n_estimators = 300`, `max_depth = None`, `min_samples_leaf = 1`, `random_state = 42` and `n_jobs = -1`.

## 3.8 Evaluation and Validation Strategy

Train / validation / test protocol. Following the three-way split of Section 3.6, the model is tuned on the validation set and its final performance is reported once on the held-out test set. The test set is never used for any design decision, so the reported test metrics are an unbiased estimate of generalisation to unseen farms. Cross-validation is used additionally, on the train set only, to obtain a more robust tuning estimate. The 10% validation set is small, about 24 farms for cacao, so a single split is noisy. Five-fold `GroupKFold` grouped by `point_id` averages the tuning metric over five rotations without ever touching the test set.

Regression metrics. MAE and RMSE are reported per horizon, against a persistence baseline (the same-window backward deficit, "tomorrow ≈ today"), plus mean and median baselines, to establish that the model beats simple, honest references.

Decision-relevant metrics. Because the target is zero-inflated, aggregate regression error alone is misleading [30]. The model is therefore also evaluated through the class-level recall and cumulative recall (≥ MODERATE, ≥ SEVERE) of the severity scheme, and through confusion matrices. These are the metrics that matter for an irrigation decision, following precision-and-recall formulations for regression [31]. For example, a recall of 0.80 for MODERATE means that, of all periods that truly needed irrigation, 80% were flagged correctly; the remaining 20% were under-flagged, a cost the decision-maker can weigh against the false-alarm rate.

## 3.9 Tools, Technologies and Reproducibility

- Language and libraries: Python 3.10 with scikit-learn, pandas, NumPy, Matplotlib and Seaborn, in a dedicated conda environment (`agri_land_env`).

- Geospatial compute: Google Earth Engine, with server-side aggregation (`ee.List.map` and a single `getInfo`) for efficiency.

- Reproducibility: fixed random seed (42), a versioned `manifest.json` recording all model hyperparameters and thresholds, and trained models saved with `joblib` (excluded from version control as regenerable artefacts).

- Data ethics and privacy: only openly published environmental datasets are used; farm records are georeferenced coordinates with no personally identifiable information.

