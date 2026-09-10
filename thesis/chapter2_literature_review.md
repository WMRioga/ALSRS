# Chapter 2 — Literature Review

> **Working draft.** Citations use IEEE-style numbered references, numbered
> in order of first appearance and consolidated in Chapter 7 (References).

---

## 2.1 Literature Review Introduction

This review examines the bodies of literature that jointly underpin the
research question: how open geospatial data and remote sensing can be combined
with physically-founded water-balance labels and machine learning to forecast
agricultural irrigation need. The review was compiled by searching academic
databases (Google Scholar, Scopus, IEEE Xplore and publisher archives) for
keywords including *land suitability assessment*, *Water Requirement
Satisfaction Index*, *Standardized Precipitation Evapotranspiration Index*,
*machine learning irrigation*, *zero-inflated regression*, *two-part model*,
*hurdle model* and *remote sensing agriculture*, followed by backward and
forward snowballing from the identified key works.

The literature is organised into seven themes, each of which corresponds to a
component of the proposed system or a question the system must answer:

1. Land suitability assessment and multi-criteria analysis;
2. Water balance and the Water Requirement Satisfaction Index (WRSI);
3. Reference evapotranspiration and drought indices;
4. Open geospatial data and remote sensing;
5. Machine learning for agricultural prediction;
6. Two-part and hurdle models for zero-inflated data;
7. Imbalanced regression and decision-relevant evaluation.

These themes map directly onto the research questions: themes 1–4 support RQ1
(the physically-founded labelling of irrigation need from open data), themes 5–6
support RQ2 (the modelling of a zero-inflated target), and themes 6–7 support
RQ3 (decision-relevant evaluation and transferability).

## 2.2 Relevant Literature Themes

### 2.2.1 Land suitability assessment and multi-criteria analysis

Land evaluation has a long and formalised tradition. The FAO *Framework for
Land Evaluation* [4] established the principle that land is assessed for a
*specific* use by matching land qualities against crop requirements, an approach
that remains the conceptual basis of most suitability systems. The Analytic
Hierarchy Process (AHP) [5] provides a structured method for weighting
heterogeneous criteria (soil, climate, terrain) into a single suitability score,
and is widely used in agricultural land evaluation because it makes expert
judgements explicit and reproducible. At the global scale, the Global
Agro-Ecological Zones (GAEZ) framework [14] operationalises this idea with
gridded climate and soil data to map crop suitability.

A critical reading of this literature reveals a consistent limitation: these
methods are **static and spatial**. They answer *where* a crop can be grown,
not *when* it will face water stress. Suitability scores are typically computed
once from long-term averages, so they do not capture inter-annual or seasonal
variability in water availability, nor do they produce the operational,
time-indexed answer that an irrigation decision requires. AHP-based approaches
are also subject to the subjectivity of the pairwise-comparison weights [5],
which must be justified rather than assumed. This theme establishes the
*foundation* of the proposed system (the suitability screening) while motivating
the need for a complementary *temporal, predictive* component.

### 2.2.2 Water balance and the Water Requirement Satisfaction Index (WRSI)

The Water Requirement Satisfaction Index (WRSI) originated in agrometeorological
crop monitoring for food-security early warning [15], and was later formalised
into a grid-cell implementation for operational use [16]. WRSI is computed from a
sequential soil-water balance: the crop's water requirement is compared against
available water (precipitation plus stored soil moisture), producing a
satisfaction index and, equivalently, a water-deficit percentage. Its key
strength is that it is **physically interpretable** — the deficit is derived
from a water-balance model rather than from a statistical proxy — and it is
well-suited to being computed from open climate data.

The critical limitation for the present study is one of *temporality and
direction*. WRSI is conventionally used to monitor the **current or
just-completed** season (a retrospective indicator), whereas the operational
question is **predictive**: what will the deficit be over the next one to six
months? Existing WRSI forecasting work is comparatively sparse and typically
projects end-of-season WRSI rather than forecasting short-horizon deficit
directly. The present study therefore repositions WRSI from a monitoring
metric to a **physically-founded labelling mechanism** for a supervised
learning target — a use that is consistent with its definition but not its
conventional deployment.

### 2.2.3 Reference evapotranspiration and drought indices

Quantifying crop water demand requires reference evapotranspiration (ET0).
Thornthwaite's temperature-based method [17] offers a parsimonious ET0 estimate
from temperature alone, while the FAO-56 Penman–Monteith formulation [2] is the
reference standard when the full set of meteorological inputs is available. Crop
water requirements are obtained by scaling ET0 by crop-specific coefficients,
and yield response to water deficit is characterised by the FAO-33 yield-response
factor [1].

At the drought-index level, the Standardized Precipitation Evapotranspiration
Index (SPEI) [18] extends the earlier SPI by incorporating temperature-driven
evapotranspiration, making it sensitive to warming-driven drought. Its revised
implementation [19] provides a consistent, multi-scalar (1- to 48-month) index
fitted by L-moments [20]. SPEI is attractive because it supplies a standardised,
scale-flexible drought signal that can act as a model *feature*.

The critical point is one of **abstraction**: SPEI and SPI measure
*meteorological* drought as a statistical anomaly, not the *crop-specific*
water deficit that drives an irrigation decision. They are therefore valuable as
predictors, but not sufficient as the decision target. This motivates the
combined design adopted here — ET0/WRSI-derived deficit as the target, and SPEI
as an input feature — which distinguishes the study from work that uses a
meteorological index as the endpoint.

### 2.2.4 Open geospatial data and remote sensing

The feasibility of the proposed system rests on the maturity of open geospatial
data. Gridded precipitation (CHIRPS [6]), reanalysis meteorology (ERA5-Land
[7]), satellite evapotranspiration (MODIS [8]) and soil properties (SoilGrids
[9], with pedotransfer functions for available water capacity [21]) together
provide the climatic and edaphic inputs needed for a water balance, at global
coverage and no direct data cost. Vegetation and moisture indices from
Sentinel-1 [22] and optical sensors (NDVI [23], EVI [24], NDWI/NDMI [25]) add
observational features and validation signals.

Two critical observations emerge from this theme. First, the products differ in
**temporal depth and quality**: for example, the harmonised Sentinel-2 archive
begins only in 2017, and synthetic-aperture-radar soil moisture is noisy at
biweekly resolution, which constrains which features can be used consistently
over a ten-year training window. Second, the products are **inputs, not
answers**: none of them directly yields the crop-specific deficit. Their value
is realised only when they are integrated through a water-balance model — the
design decision at the heart of this study, which contrasts with purely
data-driven approaches that feed raw indices directly to a learner.

### 2.2.5 Machine learning for agricultural prediction

Random Forests [10] are a standard, robust baseline for tabular agricultural
prediction, offering non-linearity, built-in variable importance and resistance
to overfitting. The applied literature has demonstrated machine learning for
irrigation-water estimation from remote sensing [26], for land-suitability
suggestion [27], and for soil-property classification [28], among many others.

A critical synthesis of this work reveals three recurring patterns relevant to
the gap. First, the dominant targets are **yield, drought class or water
consumption**, rather than the forward-looking *irrigation-decision* variable
(deficit) that a farmer actually acts on. Second, many studies rely on
**commercial, coarse or region-specific data**, limiting reproducibility and
transferability. Third, and most importantly for the modelling question, when
the target is a zero-inflated deficit, a **single regressor** is asked to learn
two distinct tasks — *occurrence* and *magnitude* — simultaneously, a mismatch
that the standard regression literature itself warns against for semicontinuous
data [11], [12], [13]. These patterns collectively position the present study:
a regression-based, physically-labelled, crop-transferable deficit forecast
using open data.

### 2.2.6 Two-part and hurdle models for zero-inflated data

The two-part (hurdle) model is the classical statistical solution for
semicontinuous outcomes: a first stage models the probability of a non-zero
value, and a second stage models the magnitude conditional on a non-zero
occurrence [11], [12]. Its roots lie in limited-dependent-variable econometrics
[11] and modified count-data models [12], with the closely related zero-inflated
family formalised for count regression [13]. The conceptual justification is
precisely the one the present problem demands: the processes that determine
*whether* a deficit occurs and *how large* it is may be governed by different
relationships, and a single equation confounds them.

Despite its maturity, the two-part model has seen **limited application to
satellite-based agricultural water-deficit forecasting**. Its adoption in
machine-learning pipelines is more recent, exemplified by two-stage frameworks
for zero-inflated, highly-skewed precipitation [29]. This is the decisive
positioning of the study: it transplants an established econometric structure
into a machine-learning, remote-sensing setting, where the zero-inflated nature
of the deficit target is not incidental but central — in the study's data, a
majority of biweekly records have zero deficit. The study contributes by
*applying and evaluating* this established approach in a new context rather than
by inventing a new method, in line with the scope appropriate to a Master's
project.

### 2.2.7 Imbalanced regression and decision-relevant evaluation

Because the deficit target is dominated by zeros, standard regression metrics
are misleading. Mean absolute error and R² are dominated by the majority
(no-deficit) class, while the minority (severe-deficit) cases are the ones that
matter for the irrigation decision. The imbalanced-learning literature
systematises this problem [30], and precision-and-recall formulations for
regression [31] provide decision-relevant alternatives. This informs the study's
evaluation design: alongside MAE and RMSE, the model is assessed through
class-level recall and cumulative recall on the severity scheme, and the
regression is weighted to focus on positive-deficit rows.

The critical gap in the applied literature is that agricultural ML evaluations
frequently report **aggregate accuracy alone**, which can look strong while
failing to detect the severe cases that an irrigation system exists to catch.
By adopting decision-relevant metrics and an explicit severity scheme, the study
aims to evaluate the model the way a decision-maker would use it, not merely
report average error.

## 2.3 Research Gap and Synthesis

Taken together, the seven themes reveal a set of well-developed **components**
that are rarely **integrated** for the specific task this study addresses:

- Suitability assessment is static and spatial [4], [5], [14];
- WRSI is physically interpretable but conventionally retrospective [15], [16];
- Meteorological drought indices are scale-flexible but crop-agnostic [18], [19];
- Open geospatial data are mature and free but are raw inputs, not answers [6], [7], [8], [9], [21], [22], [23], [24], [25];
- Agricultural ML targets yield or drought class rather than the zero-inflated
  irrigation-deficit decision, and often relies on non-open data [10], [26], [27], [28];
- The two-part/hurdle model is the established remedy for zero-inflated targets
  but is rarely applied to satellite-based water-deficit forecasting [11], [12], [13];
- Evaluation in agricultural ML rarely uses decision-relevant (recall-based)
  measures [30], [31].

The resulting **research gap** is the following. There is limited evidence on
how to combine (i) physically-founded water-balance (WRSI) labels derived from
open geospatial data, with (ii) a two-stage (hurdle) machine-learning model, to
**forecast** the zero-inflated irrigation-deficit decision across crop types and
regions, and on how such a system should be evaluated in decision-relevant
terms.

This gap maps directly onto the research questions. RQ1 asks whether the WRSI
balance can serve as a physically-founded labelling mechanism; RQ2 asks whether
the hurdle structure improves on a single regressor for the zero-inflated
deficit target; and RQ3 asks how the severity scheme affects decision-relevant
recall and whether the approach transfers across crops. The proposed study
addresses the gap by constructing the ALSRS pipeline and validating it on three
crops spanning the two case-study countries — cacao (a tropical perennial,
Colombia), wheat (a temperate annual, Australia) and sugarcane (grown in both) —
thereby demonstrating transferability across climate regimes and crop types
rather than merely asserting it.
