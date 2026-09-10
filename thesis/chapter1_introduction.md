# Chapter 1 — Introduction

**Agricultural Land Suitability Recommendation System Based on Open Geospatial
Data and Remote Sensing: A Case Study in Colombia and Australia**

> **Working draft.** Citation style: IEEE (numbered). Scope: three crops —
> cacao CCN-51 (Colombia), wheat (Australia), and sugarcane (Colombia +
> Australia). The quantitative results cited reflect the current cacao CCN-51
> validation (239 farms); update numbers when wheat and sugarcane are trained.
> Reference numbers match Chapter 7 (References); references are numbered
> in order of first appearance.

---

## 1.1 Background and Research Context

Agricultural production is fundamentally constrained by water. For rain-fed and
irrigated crops alike, the timing and severity of **water deficit** — the
shortfall between crop water requirement and available water — determine yield
and profitability [1], [2]. Climate variability, including El Niño–Southern
Oscillation (ENSO) events, compounds this uncertainty by shifting rainfall
patterns across seasons [3].

Land suitability assessment has long addressed the question of *where* a crop
can be grown, by evaluating soil, climate and terrain against crop-specific
requirements [4]. Multi-criteria methods such as the Analytic Hierarchy Process
(AHP) formalise these evaluations into a suitability score [5]. Suitability,
however, answers a **static, spatial** question — *is this land apt?* — rather
than the **operational, temporal** question that follows it: *given a suitable
location, will this crop need irrigation in the coming weeks or months?*

Two developments make a predictive answer feasible. First, freely available
satellite and reanalysis products — precipitation (CHIRPS [6]), temperature
(ERA5-Land [7]), evapotranspiration (MODIS [8]), and soil properties
(SoilGrids [9]) — now provide continuous, global, multi-decadal coverage
accessible through platforms such as Google Earth Engine. Second,
machine-learning methods such as Random Forests [10] can learn non-linear,
data-driven relationships from these large, multi-source datasets.

This thesis develops the **Agricultural Land Suitability Recommendation System
(ALSRS)**, which bridges suitability assessment and operational water
management. ALSRS combines a physically-founded water-balance (WRSI) to derive
irrigation-need labels from free satellite data, with a two-stage (hurdle)
machine-learning model that forecasts water deficit 1, 3 and 6 months ahead. The
output is a per-biweekly recommendation of irrigation severity
(LOW / MODERATE / SEVERE) for a target crop and location.

The system is parameterized for seven crops through a crop-parameter table
describing soil, climate and water-requirement characteristics. For the
empirical validation reported in this thesis, three representative crops
spanning the two case-study countries were selected: **cacao CCN-51** (a
tropical perennial, Colombia), **wheat** (a temperate annual, Australia) and
**sugarcane** (a perennial grown in both countries). This selection is
deliberate: it spans a tropical and a temperate climate, a perennial and an
annual crop, and a crop shared by both countries, thereby providing evidence of
transferability across climate regimes and crop types.

## 1.2 Research Problem and Motivation

The central research problem is the **predictive** character of irrigation need.
Water deficit is a **zero-inflated** variable: in most biweekly periods there is
no deficit (value equal to zero), but when a deficit occurs it can be severe. In
the cacao dataset analysed in this study, 73%, 56% and 39% of records have zero
deficit at the 1-, 3- and 6-month horizons, respectively. A single regression
model must simultaneously learn *whether* a deficit will occur and *how large*
it will be — two tasks that one model serves poorly. In practice, a single
Random Forest regressor systematically predicts a "phantom deficit" of roughly
12 percentage points even in periods where no deficit occurs, inflating its
error.

Existing machine-learning research on agriculture has largely targeted crop
*yield* prediction or *drought classification*, frequently relying on commercial
or restricted data, and rarely modelling the *irrigation decision* directly as a
zero-inflated water deficit (Chapter 2 provides the critical review). The
two-part (hurdle) model — which separates the "occurrence" decision from the
"magnitude" regression — is a well-established approach in econometrics for
exactly this kind of semicontinuous data [11], [12], [13], yet it has seen
limited application to satellite-based agricultural water-deficit forecasting.

The motivation is therefore both practical and methodological. Practically,
farmers, irrigation districts and agronomists need a low-cost, transferable,
predictive tool that indicates, per biweekly window, whether irrigation will be
required and with what severity. Methodologically, the zero-inflated nature of
water deficit calls for a modelling approach that explicitly separates
occurrence from magnitude rather than forcing a single model to do both.

> *Note: this section motivates the problem. The research gap is established and
> justified critically in Chapter 2.*

## 1.3 Research Questions and Objectives

The study is guided by one main research question and three sub-questions.

**Main research question.** How can freely available satellite and climate data,
combined with physically-founded water-balance labels, be used to forecast
agricultural irrigation need (water deficit) with machine learning?

- **RQ1 (labelling).** Can a biweekly water-balance (WRSI) framework generate
  physically-founded irrigation-need labels from freely available Google Earth
  Engine data?
- **RQ2 (modelling).** Does a two-stage (hurdle) model — a classifier for
  deficit occurrence plus a regressor for magnitude — improve deficit
  forecasting over a single regressor, given the zero-inflated character of the
  target?
- **RQ3 (decision and transferability).** How does the choice of severity class
  scheme (four, three or two classes) affect decision-relevant performance
  (recall), and is the approach transferable across crops?

The corresponding objectives are:

- **O1.** Construct the ALSRS pipeline: satellite data extraction (Google Earth
  Engine), crop-viability filtering, biweekly water-balance (WRSI) labelling,
  and AHP-based suitability scoring.
- **O2.** Build and evaluate a hurdle (two-stage Random Forest) model that
  forecasts deficit at 1-, 3- and 6-month horizons, benchmarked against a
  single-regressor baseline and honest persistence baselines.
- **O3.** Evaluate the flexibility of the post-hoc severity classification
  (4/3/2 classes) on decision-relevant recall, and assess the transferability of
  the approach across crops (cacao, wheat and sugarcane).

## 1.4 Significance and Expected Contribution

The expected contribution is threefold.

1. **Methodological.** The study applies the two-part/hurdle model [11], [12],
   [13] to a new context — satellite-based water-deficit forecasting — and
   demonstrates *why* it helps: by separating occurrence from magnitude, it
   eliminates the "phantom deficit" that a single regressor produces on
   zero-inflated data. In five-fold cross-validation, the hurdle model reduces
   mean absolute error (MAE) by 64%, 58% and 34% at the 1-, 3- and 6-month
   horizons, respectively, while preserving the recall of severe-deficit cases.

2. **Empirical.** The study produces evidence from real, multi-year,
   multi-location datasets generated entirely from freely available data,
   demonstrating that a data-driven irrigation recommendation is feasible at
   scale without proprietary inputs (239 cacao farms in Colombia at the time of
   writing, with wheat and sugarcane to follow).

3. **Practical.** Because the model predicts a continuous deficit and derives
   the severity class as a post-hoc, configurable step, the classification scheme
   can be adjusted to the decision-maker's risk tolerance **without retraining**.
   The crop is a parameter of the pipeline rather than a property of the model,
   supporting transferability to other crops and regions.

For farmers, irrigation districts and agronomists, ALSRS offers a low-cost
decision-support signal: a per-window irrigation-severity recommendation
grounded in physically-interpretable water balance rather than a black-box
yield estimate.

## 1.5 Thesis Outline

The thesis follows a progression from problem and literature, through
methodology and evidence, to interpretation and conclusion.

- **Chapter 1 — Introduction** (this chapter): establishes the research problem,
  motivation, questions and objectives, and the expected contribution.
- **Chapter 2 — Literature Review**: critically reviews the relevant themes —
  land evaluation and AHP, water-balance and WRSI, drought indices,
  remote-sensing data, machine learning for agriculture, and two-part/hurdle
  models — and establishes the research gap.
- **Chapter 3 — Research Methodology**: describes the ALSRS pipeline (data
  extraction, water-balance labelling, suitability scoring), the dataset, the
  hurdle modelling approach, and the evaluation strategy (cross-validation,
  baselines and metrics).
- **Chapter 4 — Analysis, Experiments and Results**: presents the empirical
  evidence — the zero-inflated target analysis, the single-regressor baseline,
  the hurdle model, the gate, the threshold sweep, the flexible classification,
  and the cross-validation results, across the three crops.
- **Chapter 5 — Discussion**: interprets the findings, relates them to the
  literature, and discusses limitations and implications.
- **Chapter 6 — Conclusion and Recommendations**: summarises the findings and
  contribution, and proposes future research directions.

> *(Insert the thesis progression diagram from `THESIS_PLAN.md` here, as Figure
> 1.1.)*
