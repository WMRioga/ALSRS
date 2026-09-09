# Bibliography

Consolidated reference list for the ALSRS (Agricultural Land Suitability
Recommendation System) project. This file unifies the citations that were
previously scattered across `ml/ML_MODEL.md`, `ml/DATASET_DICTIONARY.md`,
`README_crops.md`, and the module docstrings (`common/soil_hydraulics.py`,
`extraction/spei_profile.py`, `extraction/evapotranspiration_profile.py`).

References are grouped by the stage of the pipeline they support:
**Phase 1** (land evaluation / AHP), **Phase 2** (data extraction and water
balance), and **Phase 3** (machine learning).

---

## 1. Land evaluation framework (Phase 1)

- FAO (1976). *A Framework for Land Evaluation.* Soils Bulletin No. 32. Rome.
- Fischer, G., Nachtergaele, F.O., van Velthuizen, H.T., Chiozza, F.,
  Franceschini, G., Henry, M., Muchoney, D., & Tramberend, S. (2021).
  *Global Agro-Ecological Zones v4 – Model documentation.* FAO, Rome.

## 2. Multi-criteria decision analysis — AHP (Phase 1)

- Saaty, T.L. (1980). *The Analytic Hierarchy Process.* McGraw-Hill.

## 3. Soil — pedotransfer and data (Phase 2)

- Saxton, K.E., & Rawls, W.J. (2006). Soil water characteristic estimates by
  texture and organic matter for hydrologic solutions. *Soil Science Society
  of America Journal, 70*(5), 1569–1578. doi:10.2136/sssaj2005.0117
- Poggio, L., de Sousa, L.M., Batjes, N.H., Heuvelink, G.B.M., Kempen, B.,
  Ribeiro, E., & Rossiter, D. (2021). SoilGrids 2.0: producing soil
  information for the globe with quantified spatial uncertainty. *SOIL, 7*,
  217–240. doi:10.5194/soil-7-217-2021
- USDA Soil Taxonomy. Soil Survey Staff. Natural Resources Conservation
  Service.

## 4. Climate and satellite data sources (Phase 2)

- Muñoz-Sabater, J., Dutra, E., Agustí-Panareda, A., Albergel, C., Arduini,
  G., Balsamo, G., Boussetta, S., Choulga, M., Harrigan, S., Hersbach, H.,
  et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for
  land applications. *Earth System Science Data, 13*, 4349–4383.
  doi:10.5194/essd-13-4349-2021
- Funk, C., Peterson, P., Landsfeld, M., Pedreros, D., Verdin, J., Shukla, S.,
  Husak, G., Rowland, J., Harrison, L., Hoell, A., & Michaelsen, J. (2015).
  The climate hazards infrared precipitation with stations — a new
  environmental record for monitoring extremes. *Scientific Data, 2*, 150066.
  doi:10.1038/sdata.2015.66
- Running, S., Mu, Q., & Zhao, M. (2021). MODIS/Terra Net Evapotranspiration
  8-Day L4 Global 500m SIN Grid V061 [Data set]. NASA EOSDIS LP DAAC.
  https://doi.org/10.5067/MODIS/MOD16A2.061
- Running, S., Mu, Q., & Zhao, M. (2021). MODIS/Terra Net Evapotranspiration
  Gap-Filled 8-Day L4 Global 500m SIN Grid V061 [Data set]. NASA EOSDIS LP
  DAAC. https://doi.org/10.5067/MODIS/MOD16A2GF.061

## 5. Remote sensing — spectral indices and products (Phase 2)

- Bauer-Marschallinger, B., Freeman, V., Cao, S., Paulik, C., Schaufler, S.,
  Stachl, T., Modanesi, S., Massari, C., Ciabatta, L., Brocca, L., & Wagner,
  W. (2019). Toward global soil moisture monitoring with Sentinel-1:
  Harnessing assets and overcoming obstacles. *IEEE Transactions on
  Geoscience and Remote Sensing, 57*(1), 520–539.
- Rouse, J.W., Haas, R.H., Schell, J.A., & Deering, D.W. (1974). Monitoring
  vegetation systems in the Great Plains with ERTS. *NASA Special Publication
  351*, 309–317.
- Huete, A., Didan, K., Miura, T., Rodriguez, E.P., Gao, X., & Ferreira, L.G.
  (2002). Overview of the radiometric and biophysical performance of the
  MODIS vegetation indices. *Remote Sensing of Environment, 83*(1–2),
  195–213.
- Gao, B.-C. (1996). NDWI — a normalized difference water index for remote
  sensing of vegetation liquid water from space. *Remote Sensing of
  Environment, 58*(3), 257–266.
- Rikimaru, A., Roy, P.S., & Miyatake, S. (2002). Tropical forest cover
  density mapping. *Tropical Ecology, 43*(1), 39–47.
- Barnes, E.M., Clarke, T.R., Richards, S.E., Colaizzi, P.D., Haberland, J.,
  Kostrzewski, M., Waller, P., Choi, C., Riley, E., Thompson, T., et al.
  (2000). Coincident detection of crop water stress, nitrogen status and
  canopy density using ground-based multispectral data. *Proceedings of the
  5th International Conference on Precision Agriculture*, Bloomington, MN.
- Baetens, L., Desjardins, C., & Hagolle, O. (2019). Validation of Copernicus
  Sentinel-2 cloud masks obtained from MAJA, Sen2Cor, and FMask products
  using reference cloud masks generated with a supervised active learning
  procedure. *Remote Sensing, 11*(4), 433.
- White, M., Wulder, M.A., Hobart, G.W., Luther, J.E., Hermosilla, T.,
  Griffiths, P., Coops, N.C., Hall, R.J., Hostert, P., Dyk, A., & Guindon, L.
  (2014). Pixel-based image compositing for large-area dense time series
  applications and science. *Canadian Journal of Remote Sensing, 40*(3),
  192–212.

## 6. Reference evapotranspiration — ET0 / ETc (Phase 2)

- Thornthwaite, C.W. (1948). An approach toward a rational classification of
  climate. *Geographical Review, 38*(1), 55–94.
- Allen, R.G., Pereira, L.S., Raes, D., & Smith, M. (1998). *Crop
  evapotranspiration: Guidelines for computing crop water requirements.*
  FAO Irrigation and Drainage Paper 56. Rome.
- Doorenbos, J., & Kassam, A.H. (1979). *Yield response to water.* FAO
  Irrigation and Drainage Paper 33. Rome.

## 7. Water balance / WRSI (Phase 2)

- Frère, M., & Popov, G.F. (1979). *Agrometeorological crop monitoring and
  forecasting.* FAO Plant Production and Protection Paper 17. Rome.
- Verdin, J., & Klaver, R. (2002). Grid-cell-based crop water accounting for
  the famine early warning system. *Hydrological Processes, 16*(8),
  1617–1630.
- FEWS NET (USGS). *Water Requirement Satisfaction Index (WRSI)* methodology
  and GeoWRSI tool documentation.
  https://help.fews.net/en/tools/v3/chapter-11-geowrsi
- An Improved Climatological Forecast Method for Projecting End-Of-Season
  Water Requirement Satisfaction Index (WRSI). ProQuest dissertation.
  https://www.proquest.com/docview/2455969164

## 8. Drought index — SPEI (Phase 2)

- Vicente-Serrano, S.M., Beguería, S., & López-Moreno, J.I. (2010). A
  multi-scalar drought index sensitive to global warming: The Standardized
  Precipitation Evapotranspiration Index. *Journal of Climate, 23*(7),
  1696–1718. https://doi.org/10.1175/2009JCLI2909.1
- Beguería, S., Vicente-Serrano, S.M., Reig, F., & Latorre, B. (2014).
  Standardized precipitation evapotranspiration index (SPEI) revisited:
  parameter fitting, evapotranspiration models, tools, datasets and drought
  monitoring. *International Journal of Climatology, 34*(10), 3001–3023.
  https://doi.org/10.1002/joc.3887
- Hosking, J.R.M., & Wallis, J.R. (1997). *Regional Frequency Analysis: An
  Approach Based on L-Moments.* Cambridge University Press.

## 9. Crop parameters — water requirements and drought tolerance (Phase 2)

- Carr, M.K.V., & Lockwood, R. (2011). The water relations and irrigation
  requirements of cocoa (*Theobroma cacao* L.): A review. *Experimental
  Agriculture, 47*(4), 653–676.
- Carr, M.K.V. (2001). The water relations and irrigation requirements of
  coffee. *Experimental Agriculture, 37*(1), 1–36.
- Inman-Bamber, N.G., & Smith, D.M. (2005). Water relations in sugarcane and
  response to water deficits. *Field Crops Research, 92*(2–3), 185–202.
- DaMatta, F.M., & Ramalho, J.D.C. (2006). Impacts of drought and temperature
  stress on coffee physiology and production: A review. *Brazilian Journal of
  Plant Physiology, 18*(1), 55–81.
- Cenicafé (2016). *Guía para el cultivo de café en Colombia.* Centro
  Nacional de Investigaciones de Café.
- ICCO (2017). *Growing Cocoa.* International Cocoa Organization.

## 10. ENSO / ONI (Phase 2 feature)

- Trenberth, K.E. (1997). The definition of El Niño. *Bulletin of the
  American Meteorological Society, 78*(12), 2771–2777.
- NOAA Climate Prediction Center. Oceanic Niño Index (ONI).
  https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

## 11. Machine learning — model core (Phase 3)

- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
- Ratner, A., Bach, S.H., Ehrenberg, H., Fries, J., Wu, S., & Ré, C. (2017).
  Snorkel: Rapid training data creation with weak supervision. *Proceedings
  of the VLDB Endowment, 11*(3), 269–282.
  https://doi.org/10.14778/3157794.3157797

## 12. Imbalanced regression and sample weighting (Phase 3)

- Branco, P., Torgo, L., & Ribeiro, R.P. (2017). A survey of predictive
  modeling on imbalanced domains. *ACM Computing Surveys, 49*(2), 1–50.
- Resampling strategies for imbalanced regression: a survey and empirical
  analysis. *Artificial Intelligence Review* (2024).
  https://link.springer.com/article/10.1007/s10462-024-10724-3
- A Short Survey on Importance Weighting for Machine Learning (2024).
  https://arxiv.org/abs/2403.10175
- scikit-learn. `RandomForestRegressor` — `sample_weight` parameter.
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor

## 13. Two-part / hurdle model — classifier + regressor (Phase 3) ⭐

Core methodological foundation for the zero-inflated target design.

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

## 14. Zero-inflated / skewed targets (Phase 3)

- A Semi-supervised Framework for Simultaneous Classification and Regression
  of Zero-Inflated Time Series Data with Application to Precipitation
  Prediction. https://ieeexplore.ieee.org/abstract/document/5360491

## 15. Evaluation metrics for regression (Phase 3)

- Torgo, L., & Ribeiro, R.P. (2009). Precision and Recall for Regression.
  *Discovery Science, LNCS 5808*, 332–346.
  https://www.semanticscholar.org/paper/45f639807caeb7e457904423972a2e1211e408ef

## 16. Related work — ML for agriculture (active field)

Cited as evidence of an active research area and for the accumulated-target
design, not as core methodological precedents. **Wei et al. (2022) is the
closest methodological precedent** to the ALSRS forecasting approach.

- Wei, S., Xu, T., Niu, G.-Y., & Zeng, R. (2022). Estimating Irrigation Water
  Consumption Using Machine Learning and Remote Sensing Data in Kansas High
  Plains. *Remote Sensing, 14*(13), 3004.
  https://doi.org/10.3390/rs14133004
- Cao, Y., & Jiang, L. (2024). Machine Learning based Suggestion Method for
  Land Suitability Assessment and Production Sustainability. *Natural and
  Engineering Sciences, 9*(2), 55–72.
  https://doi.org/10.28978/nesciences.1569166
- Salehi Hikouei, I., Kim, S.S., & Mishra, D.R. (2021). Machine-Learning
  Classification of Soil Bulk Density in Salt Marsh Environments. *Sensors,
  21*(13), 4408. https://doi.org/10.3390/s21134408
- Forecasting and Quantifying Risks of Crop and Water Supply Failures Using
  Machine Learning and Remote Sensing.
  https://discovery.researcher.life/article/forecasting-and-quantifying-risks-of-crop-and-water-supply-failures-using-machine-learning-and-remote-sensing/ea9e0cbc55e53d979a29522dcc494bec
- Scalable machine learning framework for adaptive irrigation management of
  maize and soybean in the U.S. Midwest.
  https://www.sciencedirect.com/science/article/pii/S0168169925008166
- Characterization and forecasting of SPEI-based drought in Southern
  Telangana using statistical machine learning models.
  https://link.springer.com/article/10.1007/s44292-025-00070-6
