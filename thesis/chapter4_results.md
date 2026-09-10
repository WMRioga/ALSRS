# Chapter 4 — Analysis, Experiments and Results

> **Working draft.** Citations use IEEE-style numbered references,
> consolidated in Chapter 7 (References). All results in this chapter are from the cacao
> CCN-51 validation (239 farms); wheat and sugarcane results will be appended
> once trained. Figures referenced as "Figure 4.x" should be exported from the
> notebooks `ml/01_evaluate_hurdle.ipynb`, `ml/02_evaluate_class_bins.ipynb` and
> `ml/04_alsrs_model_process.ipynb`. Interpretation is deferred to Chapter 5.

---

## 4.1 Experimental Setup

The experiments use the dataset and modelling approach described in Chapter 3.
In summary: 52,341 biweekly records from 239 cacao farms in Colombia, 17
features known at time `t`, and three accumulated future-deficit targets (1, 3
and 6 months). All results are reported on **unseen farms**: an 80/20 train/test
split by farm (seed 42) for the holdout analyses, and five-fold `GroupKFold`
grouped by farm for the official cross-validated metrics. Two model families are
compared: the **single Random Forest regressor** (baseline, Section 4.3) and the
**hurdle model** — a classifier plus a regressor combined through a gate
(Sections 4.4–4.5). Regression performance is measured with MAE and RMSE;
decision performance with class-level and cumulative recall.

## 4.2 The Zero-Inflated Target (RQ1)

The first result is the character of the target itself. Table 4.1 reports the
proportion of records with a zero deficit at each horizon.

**Table 4.1 — Zero-inflation of the accumulated future deficit (cacao).**

| Horizon | Zero-deficit share |
|---|---|
| 1 month | 73% |
| 3 months | 56% |
| 6 months | 39% |

*(Figure 4.1: histogram of the three targets, showing the probability mass at
zero and the long positive tail.)*

The targets are strongly zero-inflated — especially at the short horizon, where
nearly three-quarters of all biweekly records have no deficit at all. This
confirms, empirically, the premise that motivated the two-stage model: any
single regressor must simultaneously learn *whether* a deficit occurs and *how
large* it is, against a target dominated by zeros. It also establishes RQ1 in the
affirmative at the data level: the water-balance (WRSI) labelling yields a
physically interpretable deficit whose statistical structure is exactly the
semicontinuous form anticipated.

## 4.3 Baseline: Single Random Forest

The single `RandomForestRegressor` (one per horizon, `sample_weight = deficit`)
provides the reference performance. On the holdout, it attains MAE of 11.59,
10.88 and 10.87 percentage points at 1, 3 and 6 months (Table 4.2).

**Table 4.2 — Holdout regression error, single Random Forest (baseline).**

| Horizon | MAE | RMSE |
|---|---|---|
| 1 month | 11.59 | 14.11 |
| 3 months | 10.88 | 12.91 |
| 6 months | 10.87 | 13.54 |

The diagnostic value of this baseline lies not in its average error but in
*where* the error is concentrated. In the rows where the true deficit is zero —
the majority of the data — the single regressor predicts, on average, a deficit
of roughly **11.8 percentage points** (1-month horizon). This is the "phantom
deficit": the model reports a substantial water shortfall in periods where none
occurred. It is the direct consequence of asking one model to perform the two
distinct tasks identified in Section 4.2.

## 4.4 The Hurdle Model (RQ2)

The hurdle model replaces the single regressor with two specialised models per
horizon (Section 3.7.2). Table 4.3 reports the holdout MAE of the gated hurdle
against the baseline.

**Table 4.3 — Holdout MAE: baseline vs hurdle (gated).**

| Horizon | Baseline MAE | Hurdle (gated) MAE |
|---|---|---|
| 1 month | 11.59 | **4.33** |
| 3 months | 10.88 | **4.86** |
| 6 months | 10.87 | **7.49** |

The hurdle reduces the holdout MAE by roughly 60% at short and medium horizons
and by about 30% at the long horizon. To obtain a reliable, leakage-free
estimate, Table 4.4 reports the official five-fold, by-farm cross-validation.

**Table 4.4 — Five-fold cross-validation (GroupKFold by farm): baseline vs
hurdle.**

| Horizon | Baseline MAE (RMSE) | Hurdle MAE (RMSE) | MAE reduction |
|---|---|---|---|
| 1 month | 11.21 (13.81) | **4.01 (9.55)** | −64% |
| 3 months | 11.00 (12.84) | **4.65 (8.82)** | −58% |
| 6 months | 10.60 (12.76) | **7.03 (10.61)** | −34% |

The cross-validated results confirm the holdout: the hurdle cuts MAE by 64%,
58% and 34% at 1, 3 and 6 months, respectively, and reduces RMSE at every
horizon. The improvement is largest at the short horizon, where zero-inflation
is most severe (73%), and smallest at the long horizon, where zero-inflation is
milder (39%) — a pattern consistent with the mechanism that the hurdle
eliminates the phantom deficit on the zero rows. This evidence directly answers
RQ2 in the affirmative.

## 4.5 The Gate: Preserving the Severe Cases

The gate (Section 3.7.3) determines how the two stages are combined. Table 4.5
reports the decision-level recall on the holdout under three combination rules:
the single-regressor baseline, the naive `P · magnitude` expectation, and the
gate.

**Table 4.5 — Holdout decision recall (3-class scheme) under three combination
rules.**

| Class | Baseline | `P · magnitude` | Gate (0.5) |
|---|---|---|---|
| MODERATE | 0.899 | 0.683 | **0.832** |
| SEVERE | 0.674 | 0.575 | **0.674** |

The naive `P · magnitude` combination achieves the lowest recall on both
decision classes: multiplying by the occurrence probability attenuates the
magnitude and systematically downgrades the cases that matter. The gate
recovers this loss — it restores the SEVERE recall to the baseline level (0.674)
and raises the MODERATE recall to 0.832 — while retaining the large MAE
improvement of Table 4.4. In other words, the gate is what allows the hurdle to
gain on *average* error without sacrificing the detection of the *severe* cases
that the system exists to flag.

## 4.6 Flexible Severity Classification (RQ3)

Because the model is a regressor, the severity class is a post-hoc, configurable
step. Table 4.6 reports overall accuracy under four class schemes, using the
*same* continuous predictions (no retraining).

**Table 4.6 — Accuracy of the same continuous predictions under different class
schemes.**

| Scheme | Thresholds | Classes | Accuracy |
|---|---|---|---|
| 4 classes | 15 / 30 / 50 | LOW, MEDIUM, HIGH, NOT_SUITABLE | 0.688 |
| **3 classes** | **15 / 50** | **LOW, MODERATE, SEVERE** | **0.784** |
| 2 classes | 20 | MODERATE, SEVERE | 0.855 |
| 2 classes | 30 | MODERATE, SEVERE | 0.860 |

Accuracy rises monotonically as the number of classes falls, from 0.688 (four
classes) to 0.784 (three) to 0.855–0.860 (two). Each removed boundary removes
the errors that crossed it. The three-class scheme is adopted because it
captures the accuracy gain of merging the fragile MEDIUM/HIGH boundary while
retaining the operationally important distinction of an urgent SEVERE class.

**Table 4.7 — Row-normalised confusion matrix, hurdle (gated), 3 classes
(diagonal = recall).**

| True \ Predicted | LOW | MODERATE | SEVERE |
|---|---|---|---|
| LOW | 0.79 | 0.20 | 0.00 |
| MODERATE | 0.13 | 0.83 | 0.04 |
| SEVERE | 0.01 | 0.32 | 0.67 |

*(Figure 4.2: heatmap of Table 4.7.)* The three-class scheme yields recall of
0.79 (LOW), 0.83 (MODERATE) and 0.67 (SEVERE). The residual error is
concentrated in the direction MODERATE→LOW (13%) and SEVERE→MODERATE (32%) — the
model tends to *under-predict* the severity rather than over-predict it.

**Comparison with a direct classifier.** As a point of comparison, a single
four-class `RandomForestClassifier` (a "direct" classifier that optimises the
class labels directly rather than regressing and binning) was trained on the
same features. Table 4.8 reports its three-class recall after merging
MEDIUM+HIGH, alongside the hurdle.

**Table 4.8 — Three-class recall: hurdle (gated) vs direct classifier (merged).**

| Class | Hurdle (gated) | Direct (merged) |
|---|---|---|
| LOW | 0.79 | **0.98** |
| MODERATE | **0.83** | 0.44 |
| SEVERE | 0.67 | **0.83** |

The two models fail in opposite directions. The direct classifier excels at the
extremes (LOW 0.98, SEVERE 0.83) but collapses the middle (MODERATE 0.44),
whereas the hurdle is balanced (MODERATE 0.83) but slightly under-predicts the
severe tail. This complementarity is returned to in Chapter 5.

## 4.7 Baselines and Cross-Validation

Table 4.9 situates the hurdle against the honest baselines introduced in
Chapter 3: a persistence reference ("tomorrow ≈ today", using the same-window
backward deficit) and the single-regressor baseline, all under five-fold
by-farm cross-validation.

**Table 4.9 — Five-fold CV: persistence, single regressor and hurdle.**

| Horizon | Persistence MAE (RMSE) | Single RF MAE (RMSE) | Hurdle MAE (RMSE) |
|---|---|---|---|
| 1 month | 10.80 (22.49) | 11.21 (13.81) | **4.01 (9.55)** |
| 3 months | 15.85 (25.70) | 11.00 (12.84) | **4.65 (8.82)** |
| 6 months | 14.34 (20.00) | 10.60 (12.76) | **7.03 (10.61)** |

Two observations follow. First, persistence is a deceptively strong baseline at
the 1-month horizon — its MAE (10.80) is slightly lower than the single
regressor's — because "tomorrow ≈ today" correctly predicts zero in the many
zero periods; but its RMSE (22.49) is far worse, because persistence is wildly
wrong on the rare large-deficit periods. The hurdle dominates both metrics at
every horizon. Second, the hurdle's advantage over the single regressor is
robust and large (Table 4.4), confirming that the gain is not an artefact of a
single split.

## 4.8 Supporting Analyses

**Learning curve.** Training the single-regressor baseline on increasing numbers
of farms (47 → 95 → 143 → 191) reduces the 6-month MAE from 12.79 to 11.59 to
11.22 to 10.87 (Figure 4.3). The gains diminish as farms are added, indicating
that the available 239 farms are sufficient for the present study and that
marginal new farms contribute little beyond ~190.

**ENSO teleconnection.** The Oceanic Niño Index feature (`oni`) correlates with
the future deficit increasingly with horizon: 0.15, 0.19 and 0.21 at 1, 3 and 6
months, respectively (Figure 4.4). The teleconnection signal is weak but
monotonic — stronger at the horizons where large-scale climate forcing matters
more than local weather — which supports its inclusion as a feature.

---

## Summary of Chapter 4

The evidence establishes three findings. **(i)** The target is strongly
zero-inflated (73% at 1 month), and a single regressor responds to this with a
phantom deficit of ~12 points on the zero rows. **(ii)** The hurdle model,
combining a classifier and a regressor through a gate, reduces MAE by 34–64% in
cross-validation while preserving the recall of the severe class. **(iii)** The
severity classification is a flexible post-hoc step: fewer classes yield higher
accuracy (0.69 → 0.78 → 0.86), and the three-class scheme offers a balanced
recall profile (MODERATE 0.83, SEVERE 0.67). The interpretation of these
findings — and their limitations — is developed in Chapter 5.
