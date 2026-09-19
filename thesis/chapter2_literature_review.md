# Chapter 2 — Literature Review

> Working draft. IEEE numbered citations, in order of first appearance, consolidated in Chapter 7.

## 2.1 Literature Review Introduction

This chapter reviews the bodies of work that together support the research question: how open geospatial data and remote sensing, combined with physically based water-balance labels and machine learning, can forecast agricultural irrigation need. The review was put together by searching academic databases (Google Scholar, Scopus, IEEE Xplore and publisher archives) for keywords including *land suitability assessment*, *Water Requirement Satisfaction Index*, *Standardized Precipitation Evapotranspiration Index*, *machine learning irrigation*, *zero-inflated regression*, *two-part model*, *hurdle model* and *remote sensing agriculture*, followed by backward and forward snowballing from the key works found.

The literature is organised into seven themes. Each one matches a component of the proposed system, or a question the system must answer:

1. Land suitability assessment and multi-criteria analysis;
2. Water balance and the Water Requirement Satisfaction Index (WRSI);
3. Reference evapotranspiration and drought indices;
4. Open geospatial data and remote sensing;
5. Machine learning for agricultural prediction;
6. Two-part and hurdle models for zero-inflated data;
7. Imbalanced regression and decision-relevant evaluation.

These themes map onto the research questions. Themes 1–4 support RQ1 (the physically based labelling of irrigation need from open data), themes 5–6 support RQ2 (modelling a zero-inflated target), and themes 6–7 support RQ3 (decision-relevant evaluation and transferability).

## 2.2 Relevant Literature Themes

### 2.2.1 Land suitability assessment and multi-criteria analysis

Land evaluation has a long and formalised tradition. The FAO Framework for Land Evaluation [4] set the principle that land is assessed for a specific use by matching land qualities against crop requirements. This idea is still the conceptual base of most suitability systems. The Analytic Hierarchy Process (AHP) [5] offers a structured way to weight heterogeneous criteria (soil, climate, terrain) into a single score, and it is widely used in agricultural land evaluation because it makes expert judgements explicit and reproducible. At the global scale, the Global Agro-Ecological Zones (GAEZ) framework [14] applies the same idea with gridded climate and soil data to map crop suitability.

A careful reading of this literature shows a recurring limitation: these methods are static and spatial. They answer where a crop can be grown, not when it will face water stress. Suitability scores are usually computed once from long-term averages, so they miss inter-annual or seasonal variability in water availability, and they do not give the operational, time-indexed answer that an irrigation decision needs. AHP-based approaches also depend on subjective pairwise-comparison weights [5], which have to be justified rather than assumed. This theme gives the foundation of the proposed system (the suitability screening) and, at the same time, motivates the need for a complementary temporal, predictive component.

### 2.2.2 Water balance and the Water Requirement Satisfaction Index (WRSI)

The Water Requirement Satisfaction Index (WRSI) began as a tool for agrometeorological crop monitoring in food-security early warning [15], and was later turned into a grid-cell implementation for operational use [16]. WRSI is computed from a sequential soil-water balance: the crop's water requirement is compared against available water (precipitation plus stored soil moisture), which gives both a satisfaction index and a complementary water-deficit percentage. Its main strength is that it is physically interpretable. The deficit comes from a water-balance model, not from a statistical proxy, and it is well suited to being computed from open climate data.

For this study, the main limitation is one of temporality and direction. WRSI is normally used to monitor the current or just-completed season, which is a retrospective indicator. The operational question here is predictive: what will the deficit be over the next one to six months? This review's search identified limited work on forecasting WRSI beyond the current season; the literature that does exist typically projects end-of-season WRSI rather than short-horizon deficit. This study therefore moves WRSI from a monitoring metric to a physically based labelling mechanism for a supervised learning target. This use is consistent with the definition of WRSI, but not with its conventional deployment.

### 2.2.3 Reference evapotranspiration and drought indices

Quantifying crop water demand requires reference evapotranspiration (ET0). Thornthwaite's temperature-based method [17] gives a simple ET0 estimate from temperature alone, while the FAO-56 Penman–Monteith formulation [2] is the reference standard when the full set of meteorological inputs is available. Crop water requirements are obtained by scaling ET0 with crop-specific coefficients, and the yield response to water deficit is described by the FAO-33 yield-response factor [1].

At the drought-index level, the Standardized Precipitation Evapotranspiration Index (SPEI) [18] extends the earlier SPI by adding temperature-driven evapotranspiration, which makes it sensitive to warming-driven drought. Its revised implementation [19] gives a consistent, multi-scalar (1- to 48-month) index fitted by L-moments [20]. SPEI is attractive because it supplies a standardised, scale-flexible drought signal that can act as a model feature.

The key point here is abstraction. SPEI and SPI measure meteorological drought as a statistical anomaly, not the crop-specific water deficit that drives an irrigation decision. They are useful as predictors, but not as the decision target. This motivates the combined design used here: the ET0/WRSI-derived deficit as the target, and SPEI as an input feature. That is what separates this study from work that treats a meteorological index as the endpoint.

### 2.2.4 Open geospatial data and remote sensing

The proposed system is feasible because open geospatial data has matured. Gridded precipitation (CHIRPS [6]), reanalysis meteorology (ERA5-Land [7]), satellite evapotranspiration (MODIS [8]) and soil properties (SoilGrids [9], with pedotransfer functions for available water capacity [21]) together give the climatic and edaphic inputs needed for a water balance, at global coverage and no direct data cost. Vegetation and moisture indices from Sentinel-1 [22] and optical sensors (NDVI [23], EVI [24], NDWI/NDMI [25]) add observational features and validation signals.

Two observations follow. First, the products differ in temporal depth and quality. For example, the harmonised Sentinel-2 archive starts only in 2017, and synthetic-aperture-radar soil moisture is noisy at biweekly resolution, which limits which features can be used consistently over a ten-year window. Second, the products are inputs, not answers. None of them directly gives the crop-specific deficit. Their value appears only when they are integrated through a water-balance model. That is the design decision at the centre of this study, and it contrasts with purely data-driven approaches that feed raw indices straight to a learner.

### 2.2.5 Machine learning for agricultural prediction

Random Forests [10] are a standard and robust baseline for tabular agricultural prediction. They offer non-linearity, built-in variable importance and resistance to overfitting. The applied literature has used machine learning to estimate irrigation water from remote sensing [26], to suggest land suitability [27], and to classify soil properties [28], among many other tasks.

Three patterns in this work are relevant to the gap. First, the dominant targets are yield, drought class or water consumption, rather than the forward-looking irrigation-decision variable (deficit) that a farmer actually acts on. Second, many studies rely on commercial, coarse or region-specific data, which limits reproducibility and transferability. Third, and most important for the modelling question, when the target is a zero-inflated deficit, a single regressor is asked to learn two distinct tasks (occurrence and magnitude) at the same time. The standard regression literature itself warns against this for semicontinuous data [11], [12], [13]. These patterns position the present study: a regression-based, physically labelled, crop-transferable deficit forecast using open data.

### 2.2.6 Two-part and hurdle models for zero-inflated data

The two-part (hurdle) model is the classical statistical solution for semicontinuous outcomes. A first stage models the probability of a non-zero value, and a second stage models the magnitude given a non-zero occurrence [11], [12]. The approach comes from limited-dependent-variable econometrics [11] and modified count-data models [12], with the closely related zero-inflated family formalised for count regression [13]. The conceptual justification is exactly what the present problem needs: the processes that decide whether a deficit occurs and how large it is may follow different relationships, and a single equation mixes them up. In this thesis, the same structure is implemented as the two-stage, or hurdle, Random Forest model.

Despite its maturity, the two-part model has been rarely applied to satellite-based agricultural water-deficit forecasting. Its use in machine-learning pipelines is more recent, shown by two-stage frameworks for zero-inflated, highly-skewed precipitation [29]. This is the decisive positioning of the study: it moves an established econometric structure into a machine-learning, remote-sensing setting, where the zero-inflated nature of the deficit target is central rather than incidental. In the study's data, a majority of biweekly records have zero deficit for cacao, and a substantial minority for sugarcane. The study contributes by applying and evaluating this established approach in a new context, not by inventing a new method, which matches the scope of a Master's project.

### 2.2.7 Imbalanced regression and decision-relevant evaluation

Because the deficit target is dominated by zeros, standard regression metrics can be misleading. Mean absolute error and R² are dominated by the majority (no-deficit) class, while the minority (severe-deficit) cases are the ones that matter for the irrigation decision. The imbalanced-learning literature systematises this problem [30], and precision-and-recall formulations for regression [31] offer decision-relevant alternatives. This informs the study's evaluation design: alongside MAE and RMSE, the model is assessed through class-level recall and cumulative recall on the severity scheme, and the regression is weighted to focus on positive-deficit rows.

The gap in the applied literature is that agricultural ML evaluations often report aggregate accuracy alone. Such a number can look strong while the model still misses the severe cases that an irrigation system exists to catch. By adopting decision-relevant metrics and an explicit severity scheme, the study aims to evaluate the model the way a decision-maker would use it, not just report average error.

## 2.3 Research Gap and Synthesis

Together, the seven themes show a set of well-developed components that are rarely integrated for the specific task this study addresses:

- Suitability assessment is static and spatial [4], [5], [14].
- WRSI is physically interpretable but conventionally retrospective [15], [16].
- Meteorological drought indices are scale-flexible but crop-agnostic [18], [19].
- Open geospatial data are mature and free but are raw inputs, not answers [6], [7], [8], [9], [21], [22], [23], [24], [25].
- Agricultural ML usually targets yield or drought class rather than the zero-inflated irrigation-deficit decision, and often relies on non-open data [10], [26], [27], [28].
- The two-part/hurdle model is the established remedy for zero-inflated targets but is rarely applied to satellite-based water-deficit forecasting [11], [12], [13].
- Evaluation in agricultural ML rarely uses decision-relevant (recall-based) measures [30], [31].

The research gap is therefore the following. There is limited evidence on how to combine (i) physically based water-balance (WRSI) labels derived from open geospatial data with (ii) a two-stage (hurdle) machine-learning model to forecast the zero-inflated irrigation-deficit decision across crop types and regions, and on how such a system should be evaluated in decision-relevant terms.

This gap maps directly onto the research questions. RQ1 asks whether the WRSI balance can serve as a physically based labelling mechanism. RQ2 asks whether the hurdle structure improves on a single regressor for the zero-inflated deficit target. RQ3 asks how the severity scheme affects decision-relevant recall and whether the approach transfers across crops. The proposed study addresses the gap by constructing the ALSRS pipeline and validating it on cacao and sugarcane, with wheat still to be completed, in order to test transferability across climate regimes and crop types rather than merely assert it.
