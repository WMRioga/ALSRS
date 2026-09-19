# Chapter 6 — Conclusion and Recommendations

> **Working draft.** Citations use IEEE-style numbered references, consolidated in Chapter 7 (References). This chapter is intentionally concise and does not introduce new results.

---

## 6.1 Conclusion

This thesis set out to answer a single question: how can freely available satellite and climate data, combined with physically based water-balance labels, be used to forecast agricultural irrigation need, the water deficit, with machine learning. The study addressed this by building the Agricultural Land Suitability Recommendation System (ALSRS) and validating its predictive core on cacao in Colombia and sugarcane in Colombia and Australia.

The three research objectives were met. O1 produced the ALSRS pipeline: open-data acquisition through Google Earth Engine, crop-viability screening, biweekly water-balance (WRSI) labelling, and AHP-based suitability scoring. O2 produced and evaluated the two-stage hurdle model against a single-regressor baseline and honest persistence references. O3 demonstrated the flexibility of the post-hoc severity classification and established the crop-parameterised, transferable architecture.

In doing so, the three research questions were answered. RQ1: the water balance yields a physically interpretable, zero-inflated deficit label, as its statistical structure confirms. RQ2: the hurdle model materially improves on a single regressor. It reduces the test MAE by 63.1%, 54.7% and 31.3% at the 1-, 3- and 6-month horizons for cacao (to 4.37, 5.15 and 7.29 percentage points), and by 63.9%, 39.2% and 15.5% for sugarcane, because separating occurrence from magnitude eliminates the phantom deficit that a single regressor emits on the no-deficit periods. RQ3: the severity scheme is a flexible post-hoc choice, in which fewer classes yield higher accuracy, and the approach transfers across crops. The transfer is now demonstrated empirically for a second perennial, sugarcane, where only the gate threshold had to change. The annual case, wheat, remains unresolved and is the main open direction.

The overall contribution is threefold. Methodologically, the study transplants the two-part/hurdle model [11], [12], [13] into a satellite-based forecasting context and shows that its value lies in modelling absence correctly, with the gate as a decision-aware combination rule, and that the benefit scales with how zero-inflated the target is. Empirically, it demonstrates that a usable irrigation recommendation can be produced entirely from open data, with physically based labels rather than proprietary inputs, in two countries and two perennial crops. Practically, it delivers a recommendation whose severity taxonomy and alarm threshold are configurable by the decision-maker without retraining.

## 6.2 Recommendations

The recommendations follow directly from the findings and their limitations.

For practice. Irrigation districts and agronomists can adopt a regression-based, physically labelled forecast as a low-cost decision-support signal, and should tune the severity thresholds and the alarm boundary to their own tolerance for false alarms versus missed severe events. Because the classification is a post-hoc layer, this tuning requires no retraining and no machine-learning expertise. The gate threshold is also crop-specific, which should be expected when the crops differ in how zero-inflated their deficit is.

For future research. Five directions follow from the study's limitations:

1. Extend to annual crops. The model transfers between perennials but not yet to an annual like wheat, whose deficit is almost never zero. This needs either a different target definition or a different class scheme, and it is the clearest next step for the transferability claim.
2. Improve severe-case recall. For cacao the 0.65 SEVERE recall and the 35% under-flagging of severe cases are the clearest remaining weakness. A hybrid of the hurdle and the direct classifier, taking the maximum severity of the two since they are complementary, is a promising, low-cost avenue.
3. Reduce deployment cost. Re-training with fewer trees (for example 100) or compressed serialisation would shrink the roughly 2.4 GB models substantially at little accuracy cost, enabling lightweight deployment.
4. Refine the water balance. Replacing Thornthwaite ET0 [17] with the FAO-56 Penman–Monteith formulation [2] where meteorology permits would improve the accuracy of the deficit labels on which everything downstream depends.
5. Complete the toolchain. A `predict_point(lat, lon, crop)` interface that chains the extraction and water-balance stages into the trained model would turn ALSRS from a validated model into an end-to-end service, and the re-tuning of thresholds per crop would accompany the multi-crop validation.

Taken together, the study establishes that forecasting the irrigation decision is feasible with open data, a physically based label, and an appropriately structured model, and that the resulting system is both interpretable and transferable across perennial crops. The remaining work is consolidation rather than discovery: extend the transfer to annual crops, sharpen the detection of severe cases, and package the result for practical use.
