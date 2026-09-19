# 	Chapter 4 — Analysis, Experiments and Results

> **Working draft.** Citations use IEEE-style numbered references, consolidated in Chapter 7 (References). Results cover cacao CCN-51 (Colombia) and sugarcane (Colombia + Australia); the wheat case is noted where tested and treated as a limitation in Chapter 5. Figures referenced as "Figure 4.x" should be exported from the notebooks listed in Appendix C. Interpretation is deferred to Chapter 5.


## 4.1 Experimental Setup

The experiments use the dataset and modelling approach described in Chapter 3. Two crops were validated in full.

For cacao, the dataset has 52,341 biweekly records from 239 farms in Colombia, with 17 features known at time `t` and three accumulated future-deficit targets (1, 3 and 6 months). For sugarcane, the dataset has 75,140 records from 340 farms across Colombia and Australia, with the same features and targets. In both cases the data are split by farm into 70% train, 10% validation and 20% test (seed 42). For cacao this gives 167, 24 and 48 farms; for sugarcane, 238, 34 and 68. The train set fits the model, the validation set selects the design decisions, and the held-out test set reports the final, unbiased metrics once.

Two model families are compared: the single Random Forest regressor (baseline, Section 4.3) and the hurdle model, a classifier plus a regressor combined through a gate (Sections 4.4–4.5). Regression performance is measured with MAE and RMSE; decision performance with class-level recall.

All tables and figures in this chapter are produced by the analysis notebooks (Appendix C).

## 4.2 The Zero-Inflated Target (RQ1)

The first result is the character of the target itself. Table 4.1 reports the proportion of records with a zero deficit at each horizon, for the three crops.

**Table 4.1 — Zero-inflation of the accumulated future deficit.**

| Crop | 1 month | 3 months | 6 months |
| - | - | - | - |
| Cacao | 72.8% | 56.4% | 39.2% |
| Sugarcane | 35.0% | 17.4% | 6.9% |
| Wheat | 10.1% | 1.8% | 0.0% |


*(Figure 4.1: histogram of the three targets, showing the probability mass at zero and the long positive tail. Export from `test/MDS650\_260903\_dataset.ipynb`.)*




Two things stand out. First, the target is strongly zero-inflated, especially at the short horizon. For cacao, nearly three quarters of all biweekly records have no deficit at one month. This confirms the premise that motivated the two-stage model: any single regressor must learn whether a deficit occurs and how large it is, against a target dominated by zeros. Second, the degree of zero-inflation varies a lot by crop. Cacao is heavily zero-inflated, sugarcane is moderate, and wheat has almost no zero-deficit periods at all. This gradient is important, because Section 4.10 shows that it predicts how much the hurdle model helps.

The results also establish RQ1 at the data level: the water-balance (WRSI) labelling yields a physically interpretable deficit whose statistical structure is exactly the semicontinuous form anticipated.

## 4.3 Baseline: Single Random Forest

The single `RandomForestRegressor` (one per horizon, `sample\_weight = deficit`) provides the reference performance. On the cacao development holdout it attains MAE of 11.59, 10.88 and 10.87 percentage points at 1, 3 and 6 months (Table 4.2).

**Table 4.2 — Development-holdout regression error, single Random Forest (baseline), cacao.**

| Horizon | MAE | RMSE |
| - | - | - |
| 1 month | 11.59 | 14.11 |
| 3 months | 10.88 | 12.91 |
| 6 months | 10.87 | 13.54 |


The diagnostic value of this baseline lies not in its average error but in where the error is concentrated. In the rows where the true deficit is zero, which are the majority of the data, the single regressor predicts on average a deficit of roughly 11.7 percentage points (1-month horizon). This is the phantom deficit: the model reports a substantial water shortfall in periods where none occurred. It is the direct consequence of asking one model to perform the two distinct tasks identified in Section 4.2.

**Table 4.3 — Weight comparison on the single regressor, cacao validation set (10%).** (Appendix C)

| Weight | MAE 1m | MAE 3m | MAE 6m | recall MODERATE | recall SEVERE |
| - | - | - | - | - | - |
| linear (`deficit`) | 11.56 | 11.52 | 11.05 | 0.905 | 0.722 |
| squared (`deficit²`) | 12.11 | 13.14 | 11.65 | 0.921 | 0.704 |
| plus-1-squared (`1 + deficit²`) | 4.98 | 6.74 | 8.45 | 0.747 | 0.637 |


The weight comparison exposes the single regressor's trade-off. A weighting that also fits the zero rows (plus-1-squared) attains a much lower MAE, because it nails the many no-deficit periods, but at the cost of severe-case recall (0.637). A weighting that ignores the zero rows (linear) maximises the SEVERE recall (0.722) but inflates the MAE by predicting the phantom deficit. Linear was adopted for its decision-relevant recall, and the inflated MAE is precisely what the hurdle's stage-1 classifier removes (Section 4.4).

## 4.4 The Hurdle Model (RQ2)

The hurdle model replaces the single regressor with two specialised models per horizon (Section 3.7.2). Table 4.4 reports the development-holdout MAE of the gated hurdle against the baseline, for cacao.

**Table 4.4 — Development holdout MAE: baseline vs hurdle (gated), cacao.**

| Horizon | Baseline MAE | Hurdle (gated) MAE |
| - | - | - |
| 1 month | 11.59 | 4.33 |
| 3 months | 10.88 | 4.86 |
| 6 months | 10.87 | 7.49 |


The hurdle reduces the development holdout MAE by roughly 60% at short and medium horizons and by about 30% at the long horizon.

The mechanism behind this reduction is visible directly in where the two models place their predictions on the no-deficit rows. The table 4.5 below reports the mean prediction on the development-holdout rows where the true 1-month deficit is zero (exported from `ml/01\_evaluate\_hurdle.ipynb`):

**Table 4.5**

| Model | Mean prediction on true-zero rows (1-month) |
| - | - |
| Baseline (single RF) | 11.66 |
| Hurdle (P·magnitude) | 1.84 |
| Hurdle (gated) | 1.02 |


The baseline invents about 11.7 points of deficit on periods that had none, while the gated hurdle predicts about 1 point on those same rows. Eliminating this phantom deficit is what drives the MAE gain, and it is the same effect the weight comparison in Table 4.3 anticipated.

The official, unbiased metric is then reported once on the held-out test set (48 farms for cacao), which is never used for any design decision.

**Table 4.6 — Official test-set MAE/RMSE: baseline vs hurdle (gated), cacao.** (Appendix C)

| Horizon | Baseline MAE (RMSE) | Hurdle MAE (RMSE) | MAE reduction |
| - | - | - | - |
| 1 month | 11.84 (14.31) | 4.37 (10.03) | −63.1% |
| 3 months | 11.37 (13.25) | 5.15 (9.57) | −54.7% |
| 6 months | 10.61 (13.02) | 7.29 (11.17) | −31.3% |


On the held-out test set, the hurdle reduces the baseline MAE by 63.1%, 54.7% and 31.3% at 1, 3 and 6 months, and lowers the RMSE at every horizon. The improvement is largest at the short horizon, where zero-inflation is most severe (72.8%), and smallest at the long horizon, where it is milder (39.2%). This pattern is consistent with the mechanism that the hurdle eliminates the phantom deficit on the zero rows. The evidence directly answers RQ2 in the affirmative.

## 4.5 The Gate: Preserving the Severe Cases

The gate (Section 3.7.3) determines how the two stages are combined. Table 4.7 reports the decision-level recall on the cacao development holdout under three combination rules: the single-regressor baseline, the naive `P · magnitude` expectation, and the gate.

**Table 4.7 — Development-holdout decision recall (3-class scheme) under three combination rules, cacao.**

| Class | Baseline | `P · magnitude` | Gate (0.5) |
| - | - | :-: | - |
| MODERATE | 0.899 | 0.683 | 0.832 |
| SEVERE | 0.674 | 0.575 | 0.674 |


The naive `P · magnitude` combination achieves the lowest recall on both decision classes: multiplying by the occurrence probability attenuates the magnitude and systematically downgrades the cases that matter. The gate recovers this loss. It restores the SEVERE recall to the baseline level (0.674) and raises the MODERATE recall to 0.832, while retaining the large MAE improvement of Table 4.6. In other words, the gate is what allows the hurdle to gain on average error without sacrificing the detection of the severe cases that the system exists to flag.

**Table 4.8 — Gate probability-threshold sweep on the validation set (10%), three-class scheme, cacao.** (Appendix C)

| Threshold | MAE (worst) | recall LOW | recall MODERATE | recall SEVERE |
| - | - | - | - | - |
| 0.2 | 11.477 | 0.527 | 0.893 | 0.730 |
| 0.3 | 10.309 | 0.609 | 0.883 | 0.730 |
| 0.4 | 9.305 | 0.677 | 0.865 | 0.730 |
| 0.5 | 8.372 | 0.746 | 0.836 | 0.730 |
| 0.6 | 7.703 | 0.826 | 0.761 | 0.730 |
| 0.7 | 7.786 | 0.891 | 0.653 | 0.721 |
| 0.8 | 8.862 | 0.944 | 0.475 | 0.693 |
| 0.9 | 11.303 | 0.983 | 0.254 | 0.587 |
| 1.0 | 19.863 | 1.000 | 0.004 | 0.043 |


*(Figure 4.x: recall and MAE versus the gate threshold, showing the U-shaped MAE and the collapse of severe recall above ~0.7. Export from `ml/01\_evaluate\_hurdle.ipynb`.)* img: recall-MAE\_vs\_gate-th(cocoa).png

The sweep justifies the 0.5 threshold. The MAE is U-shaped, reaching its minimum around 0.5–0.6 and rising sharply at high thresholds, where an over-conservative gate rejects genuine deficits and sends them to zero. The severe-class recall holds at the baseline level (0.730) from 0.2 to 0.6 and only then falls. The per-class pattern is also informative: recall LOW rises with the threshold, recall MODERATE falls, and recall SEVERE stays flat until the threshold becomes very high. The choice of 0.5 therefore lies in a flat, robust region where the result is almost insensitive to the exact threshold.

## 4.6 Flexible Severity Classification (RQ3)

Because the model is a regressor, the severity class is a post-hoc, configurable step. Table 4.9 reports overall accuracy under four class schemes, using the same continuous predictions and no retraining.

**Table 4.9 — Accuracy on the development holdout of the same continuous predictions under different class schemes, cacao.**

| Scheme | Thresholds | Classes | Accuracy |
| - | - | - | - |
| 4 classes | 15 / 30 / 50 | LOW, MEDIUM, HIGH, NOT\_SUITABLE | 0.688 |
| 3 classes | 15 / 50 | LOW, MODERATE, SEVERE | 0.784 |
| 2 classes | 20 | MODERATE, SEVERE | 0.855 |
| 2 classes | 30 | MODERATE, SEVERE | 0.860 |


Accuracy rises as the number of classes falls, from 0.688 (four classes) to 0.784 (three) to 0.855–0.860 (two). Each removed boundary removes the errors that crossed it. The three-class scheme is adopted because it captures the accuracy gain of merging the fragile MEDIUM/HIGH boundary while keeping the operationally important distinction of an urgent SEVERE class.

**Table 4.10 — Row-normalised confusion matrix on the test set, hurdle (gated), 3 classes, cacao (diagonal = recall).** (Appendix C)

| True \\ Predicted | LOW | MODERATE | SEVERE |
| - | - | - | - |
| LOW | 0.79 | 0.20 | 0.00 |
| MODERATE | 0.15 | 0.80 | 0.04 |
| SEVERE | 0.00 | 0.35 | 0.65 |


*(Figure 4.2: heatmap of Table 4.9. Export from `ml/02\_evaluate\_class\_bins.ipynb`.)* The three-class scheme yields recall of 0.79 (LOW), 0.80 (MODERATE) and 0.65 (SEVERE). The residual error is concentrated in the direction MODERATE→LOW (15%) and SEVERE→MODERATE (35%). The model tends to under-predict the severity rather than over-predict it.

Comparison with a direct classifier. As a point of comparison, a single four-class `RandomForestClassifier`, which optimises the class labels directly rather than regressing and binning, was trained on the same features. Table 4.11 reports its three-class recall after merging MEDIUM+HIGH, alongside the hurdle.

**Table 4.11 — Three-class recall (development holdout): hurdle (gated) vs direct classifier (merged), cacao.**

| Class | Hurdle (gated) | Direct (merged) |
| - | - | - |
| LOW | 0.79 | 0.98 |
| MODERATE | 0.83 | 0.44 |
| SEVERE | 0.67 | 0.83 |


The two models fail in opposite directions. The direct classifier excels at the extremes (LOW 0.98, SEVERE 0.83) but collapses the middle (MODERATE 0.44), whereas the hurdle is balanced (MODERATE 0.83) but slightly under-predicts the severe tail. This complementarity is returned to in Chapter 5.

## 4.7 Baselines (Model Development)

Table 4.12 situates the hurdle against the honest baselines introduced in Chapter 3: a persistence reference ("tomorrow ≈ today", using the same-window backward deficit) and the single-regressor baseline, all under five-fold by-farm cross-validation on the train set.

**Table 4.12 — Five-fold CV: persistence, single regressor and hurdle, cacao.**

| Horizon | Persistence MAE (RMSE) | Single RF MAE (RMSE) | Hurdle MAE (RMSE) |
| - | - | - | - |
| 1 month | 10.80 (22.49) | 11.21 (13.81) | 4.01 (9.55) |
| 3 months | 15.85 (25.70) | 11.00 (12.84) | 4.65 (8.82) |
| 6 months | 14.34 (20.00) | 10.60 (12.76) | 7.03 (10.61) |


Two observations follow. First, persistence is a deceptively strong baseline at the 1-month horizon: its MAE (10.80) is slightly lower than the single regressor's, because "tomorrow ≈ today" correctly predicts zero in the many zero periods. But its RMSE (22.49) is far worse, because persistence is wildly wrong on the rare large-deficit periods. The hurdle dominates both metrics at every horizon. Second, the hurdle's advantage over the single regressor is robust and large (Table 4.6), confirming that the gain is not an artefact of a single split.

## 4.8 Supporting Analyses

Learning curve. Training both models on increasing numbers of farms (47 → 95 → 143 → 191) reduces the 6-month MAE: the single regressor falls from 12.65 to 10.66 percentage points, while the hurdle model falls from 10.09 to 7.38 (Table 4.13, Figure 4.3; exported from `test/ml/evaluate\_model.py`). The gains diminish as farms are added, yet the hurdle's advantage, roughly 2.6–3.3 points, is present at every training size. This shows the hurdle's benefit is not an artefact of the full training set but holds even with modest data. The available 239 farms are sufficient for both models: marginal new farms contribute little beyond ~190.

**Table 4.13 — Learning curve (development holdout): single regressor vs hurdle (gated), 6-month MAE, cacao.**

| Farms | Single regressor | Hurdle (gated) |
| - | - | - |
| 47 | 12.65 | 10.09 |
| 95 | 11.66 | 8.84 |
| 143 | 11.21 | 8.11 |
| 191 | 10.66 | 7.38 |


ENSO teleconnection. The Oceanic Niño Index feature (`oni`) correlates with the future deficit increasingly with horizon: 0.15, 0.19 and 0.21 at 1, 3 and 6 months (Figure 4.4; exported from `test/MDS650\_260903\_dataset.ipynb`, see Appendix C). The teleconnection signal is weak but monotonic, stronger at the horizons where large-scale climate forcing matters more than local weather, which supports its inclusion as a feature.

## 4.9 Sugarcane: a Second Perennial

The same pipeline, with the same features and the same hurdle structure, was applied to sugarcane with only one change: the gate threshold was tuned per crop (Section 3.7.3) and set to 0.75 instead of 0.5. Table 4.14 reports the official test-set results.

**Table 4.14 — Official test-set MAE/RMSE: baseline vs hurdle (gated), sugarcane.**

| Horizon | Baseline MAE (RMSE) | Hurdle MAE (RMSE) | MAE reduction |
| - | - | - | - |
| 1 month | 9.10 (14.00) | 3.29 (6.92) | −63.9% |
| 3 months | 4.75 (8.06) | 2.89 (5.59) | −39.2% |
| 6 months | 3.97 (6.72) | 3.36 (5.96) | −15.5% |


Two results stand out. First, the hurdle again beats the single regressor at every horizon, with a reduction of 63.9% at one month. Second, the reduction shrinks with horizon, from 63.9% to 15.5%, and it shrinks faster than it did for cacao. This is expected: sugarcane is much less zero-inflated than cacao (35.0% zeros at one month versus 72.8%), so at the longer horizons there is less phantom deficit to eliminate.

The decision-level results are also strong. On the test set the gated hurdle reaches a recall of 0.95 for MODERATE and 0.92 for SEVERE (Table 4.15), notably higher than cacao's 0.80 and 0.65. The confusion matrix shows a well-separated diagonal.

**Table 4.15 — Row-normalised confusion matrix on the test set, hurdle (gated), 3 classes, sugarcane (diagonal = recall).**

| True \\ Predicted | LOW | MODERATE | SEVERE |
| - | - | - | - |
| LOW | 0.83 | 0.17 | 0.00 |
| MODERATE | 0.03 | 0.95 | 0.03 |
| SEVERE | 0.00 | 0.08 | 0.92 |


The higher recall makes sense. Sugarcane's positive mass is denser than cacao's, so the stage-2 regressor is trained on a larger share of the data and resolves the deficit magnitudes better. This is the first empirical evidence that the model transfers from one perennial to another, in a different country, with only a change in the gate threshold.

Learning curve. Training the sugarcane hurdle on increasing numbers of farms (59 → 119 → 178 → 238) reduces the 6-month MAE from 6.82 to 3.07 percentage points (Table 4.16, Figure 4.3b; exported from `ml/05\_train\_validation\_test Sugarcane.ipynb`). As with cacao, the gains diminish as farms are added, and the error is already low at modest training sizes.

**Table 4.16 — Learning curve: sugarcane hurdle (gated), 6-month MAE.**

| Farms | Hurdle (gated) MAE |
| - | - |
| 59 | 6.82 |
| 119 | 4.50 |
| 178 | 3.56 |
| 238 | 3.07 |


## 4.10 Cross-Crop Comparison

Putting the two crops side by side confirms the mechanism proposed in Section 4.4. Table 4.17 relates the zero-inflation rate to the hurdle's MAE reduction.

**Table 4.17 — Zero-inflation and hurdle MAE reduction by crop and horizon.**

| Crop | Zero-inflation (1m / 3m / 6m) | MAE reduction (1m / 3m / 6m) |
| - | - | - |
| Cacao | 72.8% / 56.4% / 39.2% | 63.1% / 54.7% / 31.3% |
| Sugarcane | 35.0% / 17.4% / 6.9% | 63.9% / 39.2% / 15.5% |


The pattern is clear. At the one-month horizon, both crops are sufficiently zero-inflated that the hurdle delivers a large and nearly equal reduction (about 63%). As the horizon lengthens, zero-inflation falls for both crops, and the hurdle's reduction falls with it, but faster for sugarcane because its zero-inflation drops more sharply. The reduction at six months is 31.3% for cacao against 15.5% for sugarcane. In other words, the hurdle helps in proportion to how much phantom deficit there is to eliminate, which is exactly the mechanism the two-stage design was built to exploit.

This also reframes the contribution. The hurdle model is not a universally better regressor; it is a better model of absence. Its value is largest where the target is dominated by zeros, and it shrinks as the target becomes denser. That the sugarcane results follow this pattern, with no change to the model except the gate threshold, is the strongest evidence in the thesis for transferability across crops.

## Summary of Chapter 4

The evidence establishes three findings. First, the target is strongly zero-inflated, but to different degrees by crop: 72.8% of cacao records have zero deficit at one month, against 35.0% for sugarcane. A single regressor responds to this with a phantom deficit of about 11.7 points on the zero rows. Second, the hurdle model, combining a classifier and a regressor through a gate, reduces the test MAE at every horizon for both crops, by 31.3–63.1% for cacao and 15.5–63.9% for sugarcane, while preserving the recall of the severe class. Third, the severity classification is a flexible post-hoc step: fewer classes yield higher accuracy, and the three-class scheme offers a balanced recall profile. The interpretation of these findings, and their limitations, is developed in Chapter 5.

