# Chapter 5 — Discussion

> **Working draft.** Citations use IEEE-style numbered references, consolidated in Chapter 7 (References). Findings cover cacao CCN-51 and sugarcane; the wheat case is discussed as a limitation.

---

## 5.1 Overview

Chapter 4 presented three headline results: the target is strongly zero-inflated, but to different degrees by crop; the hurdle model reduces the test MAE at every horizon for both cacao and sugarcane, while preserving the recall of the severe class; and the severity classification is a flexible post-hoc step in which fewer classes yield higher accuracy. This chapter interprets those results, explains why they arise, relates them to the existing literature, and identifies the study's strengths, limitations and implications.

## 5.2 Answering the Research Questions

RQ1. Can a biweekly water-balance (WRSI) generate physically based irrigation-need labels from open data? The answer is affirmative, and the evidence is the character of the resulting target itself (Table 4.1). The WRSI labelling produced a deficit variable with the expected semicontinuous structure: a dominant mass at zero and a right-skewed positive tail, in proportions consistent with the hydrological intuition that short horizons are usually satisfied while longer horizons are more often stressed. The label is interpretable. A 40% deficit means the crop received 40% less than its requirement over the window, which distinguishes it from a purely statistical proxy. This supports the design choice of deriving the target from a water balance [15], [16] rather than from a meteorological index, whose abstraction from crop demand would obscure the very quantity an irrigation decision requires.

RQ2. Does the hurdle improve on a single regressor? Yes. On the held-out test set the hurdle reduces the baseline MAE by 63.1%, 54.7% and 31.3% at 1, 3 and 6 months for cacao (Table 4.5), and by 63.9%, 39.2% and 15.5% for sugarcane (Table 4.13). The result is not merely a magnitude improvement but a mechanism, developed in Section 5.3. The two-stage structure [11], [12], [13] is therefore vindicated in this new, satellite-based context: the separation of occurrence from magnitude is precisely what the zero-inflated target demands, and a single regressor cannot supply it.

RQ3. How does the class scheme affect recall, and is the approach transferable? The class scheme has a monotonic, predictable effect on accuracy (0.688 → 0.784 → 0.855–0.860; Table 4.8), and the adopted three-class scheme yields a balanced recall profile on the test set. On transferability, the answer is now partly empirical. The model transfers from cacao to sugarcane, a second perennial in a different country, with only a change in the gate threshold (0.5 to 0.75). This is the first direct evidence that the crop is indeed a parameter of the pipeline rather than a property of the model. The remaining gap is the annual case: wheat did not transfer, and this is treated as a limitation in Section 5.8.

## 5.3 The Mechanism: Why the Hurdle Works

The most informative result is where the hurdle's gain is concentrated. The single regressor predicts a phantom deficit of roughly 11.7 percentage points on the rows where the true deficit is zero (Section 4.3); the hurdle predicts approximately zero on those rows, because its first stage classifies "no deficit" and the gate outputs zero. This single effect accounts for most of the MAE reduction, and it explains the otherwise puzzling pattern in Table 4.5: the improvement shrinks with horizon (63.1% → 54.7% → 31.3%) in lockstep with the zero-inflation rate (72.8% → 56.4% → 39.2%). The fewer the zero rows, the less phantom there is to eliminate.

The sugarcane results confirm this mechanism from the outside. Sugarcane is less zero-inflated than cacao, and its hurdle reduction shrinks faster with horizon (63.9% → 39.2% → 15.5%). At the one-month horizon, where both crops still have many zero periods, the reduction is nearly identical (about 63%). At six months, where sugarcane has only 6.9% zeros against cacao's 39.2%, the reduction falls to 15.5% against 31.3%. The hurdle helps in proportion to the phantom deficit available to remove, which is exactly what the mechanism predicts. Section 4.10 makes this comparison explicit.

Equally important is what the hurdle does not do: it does not predict the severe tail better than the baseline. On the cacao development holdout, the SEVERE recall is identical (0.674) for the baseline and the gated hurdle (Table 4.6). The two-stage model is not a better extrapolator of extreme drought; it is a better model of absence. This is a nuanced but consequential characterisation, and it has two implications. First, it reframes the contribution honestly: the hurdle's value is the elimination of false alarms on the many zero periods, not a breakthrough in predicting rare extremes. Second, it identifies the residual weakness. The model under-predicts the severe tail for cacao (35% of SEVERE cases fall to MODERATE, Table 4.9). For sugarcane this weakness is smaller, with a SEVERE recall of 0.92, because the positive mass is denser.

This mechanism is consistent with, but not identical to, the imbalanced-learning literature [30]. Rather than resampling or reweighting a single model, the study decomposes the problem so that the imbalance is handled structurally: the classifier absorbs the zero/non-zero imbalance, and the regressor is fitted only on the positive mass where it is well-defined.

The weight comparison on the single regressor (Table 4.3) anticipates this decomposition. A weighting that fits the zero rows (plus-1-squared) nearly halves the MAE but misses the severe cases, while the linear weight does the opposite. The single model cannot deliver both a low MAE on the zeros and a high recall on the tail, which is precisely why the two-stage structure is necessary: the classifier supplies the former and the regressor the latter.

## 5.4 The Gate and Decision-Relevant Evaluation

The comparison of combination rules in Table 4.6 demonstrates a subtle point. The natural combination `E[Y] = P(deficit>0) · magnitude` is correct if the loss is squared error on the average deficit. But multiplying by a probability attenuates the magnitude and systematically downgrades the severe cases (SEVERE recall falls to 0.575). For an irrigation decision, the loss is asymmetric: a missed severe deficit is costlier than a false alarm. So the relevant quantity is a thresholded decision, not an expectation. The gate implements exactly this: it keeps the full magnitude whenever occurrence is sufficiently likely.

This is the study's clearest design contribution. The two-part structure is borrowed from the literature [11], [12], [13]; the recognition that the combination rule must be chosen for the decision context, and that a probability threshold outperforms the expectation, is the adaptation that makes the structure fit the problem. It also motivates the decision-relevant evaluation adopted throughout: reporting recall and confusion matrices rather than aggregate accuracy alone, in line with precision-and-recall formulations for regression [31]. Had the study reported only MAE, it would have overstated the model by hiding the 0.65 SEVERE recall and the 35% under-flagging of severe cases for cacao.

The choice of the 0.5 threshold for cacao was validated empirically by sweeping the gate probability from 0.2 to 1.0 (Table 4.7). The sweep reveals a U-shaped MAE: it reaches its minimum around 0.5–0.6 and rises steeply at high thresholds, because an over-conservative gate rejects genuine deficits and sends them to zero. The recall of the severe class holds at the baseline level up to about 0.6–0.7 and only then collapses. The threshold 0.5 therefore sits in a flat zone (0.4–0.6) where the result is almost insensitive to the exact value.

The threshold is not universal, and this is a finding rather than a complication. Sugarcane required a threshold of 0.75 to reach the same flat region. The two crops differ in how much zero-inflation they carry, which shifts the trade-off curve and therefore the optimal gate. This is exactly the behaviour expected of a crop-parameterised system: the gate is a per-crop configuration value, not a fixed constant.

One technical caveat should be recorded. Because the stage-1 classifier uses `class_weight = "balanced"`, its output is a decision score rather than a well-calibrated probability. A value of 0.5 does not literally mean an even chance of deficit. The threshold is therefore justified on empirical grounds, from the sweep, not on a theoretical reading of 0.5 as an equiprobable split. Had a calibrated probability been required, a post-hoc calibration (for example Platt or isotonic) could be applied, but it was not necessary for the decision at hand.

## 5.5 Flexible Classification and the Boundary Principle

Table 4.8 exhibits a clean, monotonic regularity: every class boundary removed is a source of error removed. Merging MEDIUM and HIGH into MODERATE (four to three classes) raises accuracy from 0.688 to 0.784, and further merging (three to two) raises it to 0.855–0.860. The underlying reason is visible in the four-class confusion pattern: the dominant error sat on the MEDIUM/HIGH boundary. Merging that boundary removes the error at its source rather than training a better model.

The deeper point is architectural. Because the model is a regressor, the class scheme is a post-hoc policy layer, not part of the fitted model. The same continuous predictions can be re-binned into four, three or two classes without retraining, which means the severity taxonomy, and therefore the decision-maker's tolerance for false alarms, can be adjusted as a configuration choice. This separation of the physical model from the irrigation policy is, in the author's view, the single most reusable idea in the study: it decouples "how accurate is the deficit estimate" from "how conservatively should we flag", which are logically independent and should not be entangled in one fitted function.

## 5.6 Complementary Models: Hurdle versus Direct Classifier

The comparison in Table 4.10 reveals two models that fail in opposite directions. The hurdle is balanced but under-predicts the severe tail; the direct four-class classifier is a specialist at the extremes (LOW 0.98, SEVERE 0.83) but collapses the middle (MODERATE 0.44). This is not a contradiction but a complementarity: the direct classifier's error escapes to LOW (outside the merged classes), while the hurdle's error stays within the merged boundary.

This finding generalises the rule established in Section 5.5: merging classes helps only when the model's error crosses the boundary being merged. For the hurdle, the error sat on the MEDIUM/HIGH boundary, so merging helped; for the direct classifier, the error escaped to LOW, so merging MEDIUM+HIGH did not help (MODERATE remained 0.44). The two models therefore capture different signal, and a natural future direction is a hybrid that takes the maximum severity of the two: the direct classifier to catch SEVERE and the hurdle to preserve the middle. This is recorded as future work rather than pursued here, since the balanced hurdle already meets the study's objectives.

## 5.7 Relation to Existing Literature

The study positions itself within, and extends, three bodies of work.

First, it transplants the two-part/hurdle model [11], [12], [13] from its econometric origins into a remote-sensing, machine-learning setting. The empirical result, that separating occurrence from magnitude eliminates a phantom deficit on a zero-inflated target, provides concrete, quantitative evidence for a claim that is usually made only conceptually. The cross-crop results sharpen this: the benefit scales with the degree of zero-inflation, which is the mechanism in action. The work aligns with the recent adoption of two-stage frameworks for zero-inflated precipitation [29].

Second, it extends the applied ML-for-irrigation literature, of which Wei et al. [26] (irrigation water consumption from remote sensing) is a representative example. Where prior work estimates consumption or classifies drought, this study forecasts the forward-looking irrigation decision, the deficit over the next 1, 3 and 6 months, which is the variable a farmer actually acts on. It also demonstrates that this can be done entirely with open data and physically based labels, addressing the reproducibility and data-access concerns that motivate the research gap (Chapter 2).

Third, it contributes a small but general methodological lesson to the imbalanced-regression and evaluation literatures [30], [31]: on a semicontinuous target, the choice of combination rule (expectation versus gate) and the choice of metric (aggregate error versus decision recall) are not peripheral. They change the substantive conclusion about whether the model is good. A model that looks excellent by MAE may still miss a third of the severe cases, and the gate is a simple, principled way to make the model decision-aware.

## 5.8 Strengths and Limitations

Strengths. The study's principal strengths are: a physically interpretable target derived from a water balance rather than a statistical proxy; an evaluation that is honest and leakage-free (a by-farm train/validation/test split, persistence and mean/median baselines, decision-relevant recall); and an architecture in which the crop and the classification policy are parameters, not fixed properties of the model.

Limitations. Several limitations bound the generality of the findings.

- Annual crops. The model was validated on two perennials, cacao and sugarcane. It did not transfer to wheat, an annual. Wheat has almost no zero-deficit periods (Table 4.1), so the hurdle's stage-1 classifier finds no zero mass to separate, and the three-class scheme leaves the LOW class nearly empty. This means the current design is appropriate for perennial crops with a strong wet/dry seasonality, and the annual case needs either a different target definition or a different class scheme. This is the clearest remaining gap in transferability.
- Data completeness. For cacao, one farm (p074, Necoclí–Urabá) is missing from the dataset, and the harmonised Sentinel-2 archive begins in 2017, leaving 2016 and early 2017 without those indices.
- Simplified evapotranspiration. ET0 uses Thornthwaite's temperature-based method [17], which is less accurate than FAO-56 Penman–Monteith [2] when full meteorology is available; the deficit labels inherit this approximation.
- Design choices. The one-month WRSI window, the gate thresholds (0.5 for cacao, 0.75 for sugarcane) and the class thresholds (15/50) are policy or empirical choices validated on the two perennials, not optimised across crops or decision contexts. For cacao, the severe-class recall (0.65) and the 35% under-flagging of SEVERE are the clearest remaining weaknesses.
- Deployment cost. The fully-grown Random Forests (300 trees, unrestricted depth) total about 2.4 GB, which is a practical obstacle to lightweight deployment; preliminary evidence (not reported here) suggests reducing to 100 trees costs little accuracy for a threefold size reduction.

These limitations do not undermine the central findings but delineate the scope of their applicability, and each suggests a concrete avenue for future work.

## 5.9 Implications and Contribution to the Research Gap

The study addresses the research gap identified in Chapter 2, the limited evidence on combining physically based water-balance labels, a two-stage model, and open geospatial data to forecast the zero-inflated irrigation deficit, in three ways.

Methodologically, it shows that the hurdle model, with a decision-aware gate, is the appropriate structure for this target, and it quantifies why (the phantom deficit), including how the benefit scales with zero-inflation across two crops. Empirically, it produces evidence from two real multi-location datasets, in two countries, built entirely from open data, demonstrating feasibility without proprietary inputs. Practically, it delivers a recommendation whose severity taxonomy and alarm threshold are configurable by the decision-maker without retraining, and whose crop is a parameter, which is now supported by the transfer from cacao to sugarcane.

For practice, the implication is that irrigation decision support need not depend on commercial data or a black-box classifier. A physically interpretable, open-data, regression-based system can tell a farmer, per biweekly window, whether irrigation will be LOW, MODERATE or SEVERE, and the farmer can tune the boundary between "act" and "wait" to their own risk tolerance. For research, the study offers a reusable template: decompose a semicontinuous environmental target into occurrence and magnitude, combine the two stages with a decision rule rather than an expectation, and evaluate in decision-relevant terms.
