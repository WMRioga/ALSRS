# Chapter 5 — Discussion

> **Working draft.** Citations use IEEE-style numbered references,
> consolidated in Chapter 7 (References). Findings refer to the cacao CCN-51 validation;
> transferability across wheat and sugarcane is discussed as a design property
> pending empirical confirmation.

---

## 5.1 Overview

Chapter 4 presented three headline results: the target is strongly zero-inflated
(73% at one month); the hurdle model reduces the cross-validated MAE by 34–64%
while preserving the recall of the severe class; and the severity classification
is a flexible post-hoc step in which fewer classes yield higher accuracy. This
chapter interprets those results, explains *why* they arise, relates them to the
existing literature, and identifies the study's strengths, limitations and
implications.

## 5.2 Answering the Research Questions

**RQ1 — Can a biweekly water-balance (WRSI) generate physically-founded
irrigation-need labels from open data?** The answer is affirmative, and the
evidence is the character of the resulting target itself (Table 4.1). The WRSI
labelling produced a deficit variable with the expected semicontinuous structure:
a dominant mass at zero and a right-skewed positive tail, in proportions
consistent with the hydrological intuition that short horizons are usually
satisfied while longer horizons are more often stressed. The label is
interpretable — a 40% deficit means the crop received 40% less than its
requirement over the window — which distinguishes it from a purely statistical
proxy. This supports the design choice of deriving the target from a water
balance [15], [16] rather than from a meteorological index, whose abstraction from
crop demand would obscure the very quantity an irrigation decision requires.

**RQ2 — Does the hurdle improve on a single regressor?** Yes, and by a large
margin (−64%/−58%/−34% MAE; Table 4.4). The result is not merely a magnitude
improvement but a *mechanism*, developed in Section 5.3. The two-stage structure
[11], [12], [13] is therefore vindicated in this new, satellite-based context:
the separation of occurrence from magnitude is precisely what the zero-inflated
target demands, and a single regressor cannot supply it.

**RQ3 — How does the class scheme affect recall, and is the approach
transferable?** The class scheme has a monotonic, predictable effect on accuracy
(0.688 → 0.784 → 0.855–0.860; Table 4.6), and the adopted three-class scheme
yields a balanced recall profile (MODERATE 0.83, SEVERE 0.67). Transferability
is established at the design level — the crop is a parameter of the pipeline
rather than a property of the model — and is the subject of the pending
multi-crop validation, discussed as a limitation in Section 5.8.

## 5.3 The Mechanism: Why the Hurdle Works

The most informative result is *where* the hurdle's gain is concentrated. The
single regressor predicts a phantom deficit of roughly 12 percentage points on
the rows where the true deficit is zero (Section 4.3); the hurdle predicts
approximately zero on those rows, because its first stage classifies "no
deficit" and the gate outputs zero. This single effect accounts for most of the
MAE reduction, and it explains the otherwise puzzling pattern in Table 4.4: the
improvement *shrinks* with horizon (64% → 58% → 34%) in lockstep with the
zero-inflation rate (73% → 56% → 39%). The fewer the zero rows, the less
"phantom" there is to eliminate.

Equally important is what the hurdle does **not** do: it does not predict the
severe tail better than the baseline. The SEVERE recall is identical (0.674)
for the baseline and the gated hurdle (Table 4.5). The two-stage model is not a
better extrapolator of extreme drought; it is a better model of *absence*. This
is a nuanced but consequential characterisation, and it has two implications.
First, it reframes the contribution honestly: the hurdle's value is the
elimination of false alarms on the many zero periods, not a breakthrough in
predicting rare extremes. Second, it identifies the residual weakness — the
model under-predicts the severe tail (32% of SEVERE cases fall to MODERATE,
Table 4.7) — and points to the direct classifier as the complementary remedy
(Section 5.6).

This mechanism is consistent with, but not identical to, the imbalanced-learning
literature [30]: rather than resampling or reweighting a single model, the study
*decomposes* the problem so that the imbalance is handled structurally — the
classifier absorbs the zero/non-zero imbalance, and the regressor is fitted only
on the positive mass where it is well-defined.

## 5.4 The Gate and Decision-Relevant Evaluation

The comparison of combination rules in Table 4.5 demonstrates a subtle point.
The natural combination `E[Y] = P(deficit>0) · magnitude` is correct if the loss
is squared error on the average deficit; but multiplying by a probability
attenuates the magnitude and systematically downgrades the severe cases (SEVERE
recall falls to 0.575). For an *irrigation decision*, the loss is asymmetric — a
missed severe deficit is costlier than a false alarm — so the relevant quantity
is a thresholded decision, not an expectation. The gate implements exactly this:
it keeps the full magnitude whenever occurrence is sufficiently likely.

This is the study's clearest design contribution. The two-part structure is
borrowed from the literature [11], [12], [13]; the recognition that the
*combination* rule must be chosen for the decision context, and that a
probability threshold outperforms the expectation, is the adaptation that makes
the structure fit the problem. It also motivates the decision-relevant
evaluation adopted throughout: reporting recall and confusion matrices rather
than aggregate accuracy alone, in line with precision-and-recall formulations
for regression [31]. Had the study reported only MAE, it would have overstated
the model by hiding the 0.67 SEVERE recall and the 32% under-flagging of severe
cases.

## 5.5 Flexible Classification and the Boundary Principle

Table 4.6 exhibits a clean, monotonic regularity: every class boundary removed
is a source of error removed. Merging MEDIUM and HIGH into MODERATE (four → three
classes) raises accuracy from 0.688 to 0.784, and further merging (three → two)
raises it to 0.855–0.860. The underlying reason is visible in the four-class
confusion pattern: the dominant error sat on the MEDIUM/HIGH boundary. Merging
that boundary removes the error *at its source* rather than training a better
model.

The deeper point is architectural. Because the model is a regressor, the class
scheme is a post-hoc policy layer, not part of the fitted model. The same
continuous predictions can be re-binned into four, three or two classes without
retraining, which means the severity taxonomy — and therefore the
decision-maker's tolerance for false alarms — can be adjusted as a configuration
choice. This separation of the physical model from the irrigation policy is, in
the author's view, the single most reusable idea in the study: it decouples
"how accurate is the deficit estimate" from "how conservatively should we
flag", which are logically independent and should not be entangled in one
fitted function.

## 5.6 Complementary Models: Hurdle versus Direct Classifier

The comparison in Table 4.8 reveals two models that fail in opposite directions.
The hurdle is balanced but under-predicts the severe tail; the direct
four-class classifier is a specialist at the extremes (LOW 0.98, SEVERE 0.83)
but collapses the middle (MODERATE 0.44). This is not a contradiction but a
complementarity: the direct classifier's error escapes to LOW (outside the
merged classes), while the hurdle's error stays within the merged boundary.

This finding generalises the rule established in Section 5.5 — merging classes
helps only when the model's error *crosses the boundary being merged*. For the
hurdle, the error sat on the MEDIUM/HIGH boundary, so merging helped; for the
direct classifier, the error escaped to LOW, so merging MEDIUM+HIGH did not help
(MODERATE remained 0.44). The two models therefore capture different signal, and
a natural future direction is a hybrid that takes the maximum severity of the
two — the direct classifier to catch SEVERE and the hurdle to preserve the
middle — a trade-off between recall and precision that the decision-maker can
set. This is recorded as future work rather than pursued here, since the balanced
hurdle already meets the study's objectives.

## 5.7 Relation to Existing Literature

The study positions itself within, and extends, three bodies of work.

First, it transplants the two-part/hurdle model [11], [12], [13] from its
econometric origins into a remote-sensing, machine-learning setting. The
empirical result — that separating occurrence from magnitude eliminates a
phantom deficit on a zero-inflated target — provides concrete, quantitative
evidence for a claim that is usually made only conceptually, and it aligns with
the recent adoption of two-stage frameworks for zero-inflated precipitation
[29].

Second, it extends the applied ML-for-irrigation literature, of which
Wei et al. [26] (irrigation water consumption from remote sensing) is a
representative example. Where prior work estimates *consumption* or classifies
*drought*, this study forecasts the *forward-looking irrigation decision* — the
deficit over the next 1, 3 and 6 months — which is the variable a farmer
actually acts on. It also demonstrates that this can be done entirely with open
data and physically-founded labels, addressing the reproducibility and
data-access concerns that motivate the research gap (Chapter 2).

Third, it contributes a small but general methodological lesson to the
imbalanced-regression and evaluation literatures [30], [31]: on a semicontinuous
target, the *choice of combination rule* (expectation versus gate) and the
*choice of metric* (aggregate error versus decision recall) are not peripheral —
they change the substantive conclusion about whether the model is good. A model
that looks excellent by MAE may still miss a third of the severe cases, and the
gate is a simple, principled way to make the model decision-aware.

## 5.8 Strengths and Limitations

**Strengths.** The study's principal strengths are (i) a physically interpretable
target derived from a water balance rather than a statistical proxy; (ii) an
evaluation that is honest and leakage-free (by-farm GroupKFold, persistence and
mean/median baselines, decision-relevant recall); and (iii) an architecture in
which the crop and the classification policy are parameters, not fixed
properties of the model.

**Limitations.** Several limitations bound the generality of the findings.

- *Single-crop evidence.* The quantitative results concern cacao only. The
  transferability claim is currently a property of the design, not an empirical
  demonstration; the planned wheat (Australia) and sugarcane (both countries)
  validations will confirm or qualify it. A crop whose deficit is less
  zero-inflated than cacao's would be expected to show a smaller hurdle gain,
  by the mechanism of Section 5.3.
- *Data completeness.* One farm (p074, Necoclí–Urabá) is missing from the
  dataset, and the harmonised Sentinel-2 archive begins in 2017, leaving
  2016–early 2017 without those indices.
- *Simplified evapotranspiration.* ET0 uses Thornthwaite's temperature-based
  method [17], which is less accurate than FAO-56 Penman–Monteith [2] when full
  meteorology is available; the deficit labels inherit this approximation.
- *Design choices.* The one-month WRSI window, the gate threshold (0.5) and the
  class thresholds (15/50) are policy or empirical choices validated on cacao,
  not optimised across crops or decision contexts. The severe-class recall
  (0.67) and the 32% under-flagging of SEVERE are the clearest remaining
  weaknesses.
- *Deployment cost.* The fully-grown Random Forests (300 trees, unrestricted
  depth) total ~2.4 GB, which is a practical obstacle to lightweight deployment;
  preliminary evidence (not reported here) suggests reducing to 100 trees costs
  little accuracy for a threefold size reduction.

These limitations do not undermine the central findings but delineate the scope
of their applicability, and each suggests a concrete avenue for future work.

## 5.9 Implications and Contribution to the Research Gap

The study addresses the research gap identified in Chapter 2 — the limited
evidence on combining physically-founded water-balance labels, a two-stage
model, and open geospatial data to forecast the zero-inflated irrigation
deficit — in three ways.

*Methodologically*, it shows that the hurdle model, with a decision-aware gate,
is the appropriate structure for this target, and it quantifies why (the phantom
deficit). *Empirically*, it produces evidence from a real multi-location dataset
built entirely from open data, demonstrating feasibility without proprietary
inputs. *Practically*, it delivers a recommendation whose severity taxonomy and
alarm threshold are configurable by the decision-maker without retraining, and
whose crop is a parameter, supporting transfer to other crops and regions.

For practice, the implication is that irrigation decision support need not
depend on commercial data or a black-box classifier: a physically interpretable,
open-data, regression-based system can tell a farmer, per biweekly window,
whether irrigation will be LOW, MODERATE or SEVERE — and, crucially, the
farmer can tune the boundary between "act" and "wait" to their own risk
tolerance. For research, the study offers a reusable template: decompose a
semicontinuous environmental target into occurrence and magnitude, combine the
two stages with a decision rule rather than an expectation, and evaluate in
decision-relevant terms.
