# Chapter 6 — Conclusion and Recommendations

> **Working draft.** Citations use IEEE-style numbered references,
> consolidated in Chapter 7 (References). This chapter is intentionally concise and does
> not introduce new results.

---

## 6.1 Conclusion

This thesis set out to answer a single question: how can freely available
satellite and climate data, combined with physically-founded water-balance
labels, be used to forecast agricultural irrigation need — the water deficit —
with machine learning. The study addressed this by building the Agricultural
Land Suitability Recommendation System (ALSRS) and validating its predictive
core on cacao in Colombia.

The three research objectives were met. **O1** produced the ALSRS pipeline —
open-data acquisition through Google Earth Engine, crop-viability screening,
biweekly water-balance (WRSI) labelling, and AHP-based suitability scoring.
**O2** produced and evaluated the two-stage hurdle model against a
single-regressor baseline and honest persistence references. **O3** demonstrated
the flexibility of the post-hoc severity classification and established the
crop-parameterized, transferable architecture.

In doing so, the three research questions were answered. **RQ1** — the water
balance yields a physically interpretable, zero-inflated deficit label, as its
statistical structure confirms. **RQ2** — the hurdle model materially improves
on a single regressor, reducing cross-validated MAE by 64%, 58% and 34% at the
1-, 3- and 6-month horizons, because separating occurrence from magnitude
eliminates the "phantom deficit" that a single regressor emits on the many
no-deficit periods. **RQ3** — the severity scheme is a flexible post-hoc choice,
in which fewer classes yield higher accuracy (0.69 → 0.78 → 0.86), and the
approach is transferable by design, with the crop as a parameter of the pipeline.

The overall contribution is threefold. Methodologically, the study transplants
the two-part/hurdle model [11], [12], [13] into a satellite-based forecasting
context and shows that its value lies in modelling *absence* correctly, with the
gate as a decision-aware combination rule. Empirically, it demonstrates that a
usable irrigation recommendation can be produced entirely from open data, with
physically founded labels rather than proprietary inputs. Practically, it
delivers a recommendation whose severity taxonomy and alarm threshold are
configurable by the decision-maker without retraining.

## 6.2 Recommendations

The recommendations follow directly from the findings and their limitations.

**For practice.** Irrigation districts and agronomists can adopt a
regression-based, physically labelled forecast as a low-cost decision-support
signal, and should tune the severity thresholds (15/50) and the alarm boundary
to their own tolerance for false alarms versus missed severe events. Because the
classification is a post-hoc layer, this tuning requires no retraining and no
machine-learning expertise.

**For future research.** Five directions follow from the study's limitations:

1. *Empirically confirm transferability.* Complete the planned validation on
   wheat (Australia) and sugarcane (both countries). This would convert the
   transferability claim from a design property into demonstrated evidence, and
   would test the prediction that a less zero-inflated crop shows a smaller
   hurdle gain.
2. *Improve severe-case recall.* The 0.67 SEVERE recall and the 32%
   under-flagging of severe cases are the clearest remaining weakness. A hybrid
   of the hurdle and the direct classifier — taking the maximum severity of the
   two, since they are complementary — is a promising, low-cost avenue.
3. *Reduce deployment cost.* Re-training with fewer trees (e.g. 100) or
   compressed serialisation would shrink the ~2.4 GB models substantially at
   little accuracy cost, enabling lightweight deployment.
4. *Refine the water balance.* Replacing Thornthwaite ET0 [17] with the FAO-56
   Penman–Monteith formulation [2] where meteorology permits would improve the
   accuracy of the deficit labels on which everything downstream depends.
5. *Complete the toolchain.* A `predict_point(lat, lon, crop)` interface that
   chains the extraction and water-balance stages into the trained model would
   turn ALSRS from a validated model into an end-to-end service, and the
   re-tuning of thresholds per crop would accompany the multi-crop validation.

Taken together, the study establishes that forecasting the irrigation decision
is feasible with open data, a physically founded label, and an appropriately
structured model — and that the resulting system is both interpretable and
transferable. The remaining work is consolidation rather than discovery: confirm
the transfer across crops, sharpen the detection of severe cases, and package
the result for practical use.
