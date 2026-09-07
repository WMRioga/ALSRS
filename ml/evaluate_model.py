"""
ALSRS — Model Evaluation Script
===============================

Runs the four Phase-2 experiments to tune and evaluate the irrigation-need
model. Written in plain, readable code (no pipelines) so each step can be
explained in the thesis.

Experiments:
    A. Weight comparison  : linear vs squared vs (1 + deficit^2).
    B. Hyperparameter grid: small manual grid, cross-validated.
    C. Learning curve     : train on growing fractions of farms.
    D. Cross-validation   : 5-fold by point (farm); model vs persistence baseline.

Every fit prints a start/end timestamp so execution time can be measured.
Results are saved as CSVs in ml/.

Run:  python ml/evaluate_model.py
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, train_test_split

ML_DIR = Path(__file__).resolve().parent
DATASET = ML_DIR / "ml_dataset_cacao_ccn51.csv"

FEATURES = [
    "month", "biweek", "mean_C", "std_C",
    "precip_total_mm", "precip_rainy_days", "pet_mm",
    "spei_1m", "spei_3m", "spei_6m", "spei_12m",
    "AWC_mm", "Storage_mm", "P_acum_mm", "WRSI_1m", "deficit_1m",
    "oni",
]
TARGETS = ["future_deficit_1m", "future_deficit_3m", "future_deficit_6m"]

SEVERITY = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "NOT_SUITABLE": 3}
CLASS_LABELS = ["LOW", "MEDIUM", "HIGH", "NOT_SUITABLE"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def now() -> str:
    """Current wall-clock time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def classify_deficit(v: float) -> str:
    """Map a deficit (percentage 0-100) to its irrigation-need class."""
    if pd.isna(v):
        return "NA"
    if v <= 15:
        return "LOW"
    if v <= 30:
        return "MEDIUM"
    if v <= 50:
        return "HIGH"
    return "NOT_SUITABLE"


def worst_of(*classes: str) -> str:
    """Most severe class among the given ones."""
    return max(classes, key=lambda c: SEVERITY.get(c, -1))


def weight_linear(deficit):
    """weight = deficit  (0-deficit rows get weight 0 = ignored)."""
    return deficit


def weight_squared(deficit):
    """weight = deficit^2  (emphasizes the tail strongly)."""
    return deficit ** 2


def weight_plus1sq(deficit):
    """weight = 1 + deficit^2  (floor of 1, so no row is ignored)."""
    return 1 + deficit ** 2


WEIGHTS = {
    "linear": weight_linear,
    "squared": weight_squared,
    "plus1sq": weight_plus1sq,
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_model(n_estimators=300, max_depth=None, min_samples_leaf=1):
    """A Random Forest with fixed seed and all cores."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )


def timed_fit(model, X, y, w, label):
    """Fit a model, printing start/end time and elapsed seconds."""
    start = datetime.now()
    model.fit(X, y, sample_weight=w)
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    print(f"    {label:<28} {start.strftime('%H:%M:%S')} -> "
          f"{end.strftime('%H:%M:%S')}  ({elapsed:5.1f}s)")
    return model


def class_recall(y_true, y_pred, cls) -> float:
    """Recall of one class: among true 'cls', how many were predicted 'cls'."""
    mask = y_true == cls
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] == cls).mean())


# ---------------------------------------------------------------------------
# Experiment A: weight comparison (train/test split)
# ---------------------------------------------------------------------------

def experiment_a(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EXPERIMENT A — Weight comparison (linear vs squared vs plus1sq)")
    print("=" * 78)

    # Same 80/20 by-point split as the notebook.
    point_ids = df["point_id"].unique()
    train_pts, test_pts = train_test_split(point_ids, test_size=0.20,
                                           random_state=42)
    tr = df[df["point_id"].isin(train_pts)]
    te = df[df["point_id"].isin(test_pts)]
    Xtr = tr[FEATURES]
    Xte = te[FEATURES]

    rows = []
    for wname, wfn in WEIGHTS.items():
        print(f"\n  Weight: {wname}")
        preds = {}
        for target in TARGETS:
            model = make_model()
            w = wfn(tr[target])
            label = f"{target}"
            timed_fit(model, Xtr, tr[target], w, label)
            preds[target] = model.predict(Xte)

            mae = mean_absolute_error(te[target], preds[target])
            r = rmse(te[target], preds[target])
            rows.append([wname, target, mae, r])

        # Derived classification (worst of 3) recall on HIGH / NOT_SUITABLE.
        true_cls = te[TARGETS].applymap(classify_deficit)
        # Build predicted classes with the SAME index as the test rows, so the
        # boolean masks below align correctly.
        pred_cls = pd.DataFrame({
            t: pd.Series(preds[t], index=te.index).apply(classify_deficit)
            for t in TARGETS
        })
        true_sugg = true_cls.apply(lambda r: worst_of(*r), axis=1)
        pred_sugg = pred_cls.apply(lambda r: worst_of(*r), axis=1)

        for cls in ["HIGH", "NOT_SUITABLE"]:
            rc = class_recall(true_sugg, pred_sugg, cls)
            print(f"      recall {cls:<13}: {rc:.3f}")

    out = pd.DataFrame(rows, columns=["weight", "target", "mae", "rmse"])
    out.to_csv(ML_DIR / "eval_weights.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Experiment B: hyperparameter grid (cross-validated on train farms)
# ---------------------------------------------------------------------------

def cv_eval(df, target, weight_fn, model_params, n_splits=5):
    """5-fold by-point cross-validation for one target.

    Returns a dict with the model's (mae, rmse, r2) plus three baselines
    evaluated on the SAME folds:

      - base_persist: same-window persistence (honest, matched window size).
        For each horizon it uses the deficit accumulated over the SAME-sized
        window immediately BEFORE t, so both share the same variance/dilution:
            1m -> ``deficit_1m`` (1-month deficit)
            3m -> ``past_deficit_3m``
            6m -> ``past_deficit_6m``
      - base_mean   : always predict the training mean
      - base_median : always predict the training median

    R2 is reported for reference only: it is unreliable on zero-inflated
    targets (many 0-deficit rows depress it).
    """
    # Same-window persistence baseline for each horizon.
    persist_col = {
        "future_deficit_1m": "deficit_1m",
        "future_deficit_3m": "past_deficit_3m",
        "future_deficit_6m": "past_deficit_6m",
    }[target]

    gkf = GroupKFold(n_splits=n_splits)
    maes, rmses, r2s = [], [], []
    p_maes, p_rmses, p_r2s = [], [], []   # persistence
    m_maes, m_rmses, m_r2s = [], [], []   # mean
    d_maes, d_rmses, d_r2s = [], [], []   # median

    for tr_idx, va_idx in gkf.split(df, groups=df["point_id"]):
        m = RandomForestRegressor(random_state=42, n_jobs=-1, **model_params)
        Xtr, ytr = df.iloc[tr_idx][FEATURES], df.iloc[tr_idx][target]
        Xva, yva = df.iloc[va_idx][FEATURES], df.iloc[va_idx][target]
        m.fit(Xtr, ytr, sample_weight=weight_fn(ytr))
        pred = m.predict(Xva)

        maes.append(mean_absolute_error(yva, pred))
        rmses.append(rmse(yva, pred))
        r2s.append(r2_score(yva, pred))

        # Persistence baseline: same-window deficit immediately before t.
        # Read from the full df (past_deficit_* are not in FEATURES).
        # Filter rows where the baseline is NaN (early rows of each point
        # have no full past window).
        persist = df.iloc[va_idx][persist_col]
        valid = persist.notna() & yva.notna()
        p_maes.append(mean_absolute_error(yva[valid], persist[valid]))
        p_rmses.append(rmse(yva[valid], persist[valid]))
        p_r2s.append(r2_score(yva[valid], persist[valid]))

        # Mean / median baselines: constant value from the TRAIN fold.
        mean_val = float(ytr.mean())
        median_val = float(ytr.median())
        m_maes.append(mean_absolute_error(yva, np.full(len(yva), mean_val)))
        m_rmses.append(rmse(yva, np.full(len(yva), mean_val)))
        m_r2s.append(r2_score(yva, np.full(len(yva), mean_val)))
        d_maes.append(mean_absolute_error(yva, np.full(len(yva), median_val)))
        d_rmses.append(rmse(yva, np.full(len(yva), median_val)))
        d_r2s.append(r2_score(yva, np.full(len(yva), median_val)))

    return {
        "mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)),
        "r2": float(np.mean(r2s)),
        "base_persist_mae": float(np.mean(p_maes)),
        "base_persist_rmse": float(np.mean(p_rmses)),
        "base_persist_r2": float(np.mean(p_r2s)),
        "base_mean_mae": float(np.mean(m_maes)),
        "base_mean_rmse": float(np.mean(m_rmses)),
        "base_mean_r2": float(np.mean(m_r2s)),
        "base_median_mae": float(np.mean(d_maes)),
        "base_median_rmse": float(np.mean(d_rmses)),
        "base_median_r2": float(np.mean(d_r2s)),
    }


def experiment_b(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EXPERIMENT B — Hyperparameter grid (5-fold CV, 6-month target)")
    print("=" * 78)

    # Tune on the 6-month horizon (the hardest), with linear weight.
    target = "future_deficit_6m"
    wfn = weight_linear

    grid = [
        dict(n_estimators=100, max_depth=None, min_samples_leaf=1),
        dict(n_estimators=100, max_depth=10, min_samples_leaf=1),
        dict(n_estimators=100, max_depth=20, min_samples_leaf=1),
        dict(n_estimators=300, max_depth=None, min_samples_leaf=1),
        dict(n_estimators=300, max_depth=10, min_samples_leaf=1),
        dict(n_estimators=300, max_depth=20, min_samples_leaf=1),
        dict(n_estimators=300, max_depth=None, min_samples_leaf=5),
        dict(n_estimators=500, max_depth=None, min_samples_leaf=1),
        dict(n_estimators=500, max_depth=10, min_samples_leaf=1),
        dict(n_estimators=500, max_depth=20, min_samples_leaf=1),
        dict(n_estimators=500, max_depth=10, min_samples_leaf=5),
        dict(n_estimators=500, max_depth=20, min_samples_leaf=5),
    ]

    rows = []
    for params in grid:
        t0 = time.time()
        res = cv_eval(df, target, wfn, params)
        mae, r = res["mae"], res["rmse"]
        elapsed = time.time() - t0
        desc = (f"n={params['n_estimators']} d={params['max_depth']} "
                f"leaf={params['min_samples_leaf']}")
        rows.append([desc, mae, r, elapsed])
        print(f"    {desc:<34} CV MAE={mae:6.2f}  RMSE={r:6.2f}  ({elapsed:.1f}s)")

    out = pd.DataFrame(rows, columns=["params", "cv_mae", "cv_rmse", "elapsed_s"])
    out = out.sort_values("cv_rmse").reset_index(drop=True)
    out.to_csv(ML_DIR / "eval_tuning.csv", index=False)

    best = out.iloc[0]
    print(f"\n  Best: {best['params']}  (RMSE={best['cv_rmse']:.2f})")
    return out


# ---------------------------------------------------------------------------
# Experiment C: learning curve (growing fraction of farms)
# ---------------------------------------------------------------------------

def experiment_c(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EXPERIMENT C — Learning curve (25/50/75/100% of train farms)")
    print("=" * 78)

    point_ids = df["point_id"].unique()
    train_pts, test_pts = train_test_split(point_ids, test_size=0.20,
                                           random_state=42)
    te = df[df["point_id"].isin(test_pts)]
    Xte = te[FEATURES]

    rows = []
    for frac in [0.25, 0.50, 0.75, 1.0]:
        n_farms = int(len(train_pts) * frac)
        subset_pts = train_pts[:n_farms]  # deterministic (seeded split order)
        tr = df[df["point_id"].isin(subset_pts)]
        Xtr = tr[FEATURES]
        print(f"\n  Training on {len(subset_pts)} farms ({frac:.0%}):")

        for target in TARGETS:
            model = make_model()
            w = weight_linear(tr[target])
            timed_fit(model, Xtr, tr[target], w, f"{target} ({len(subset_pts)}f)")
            pred = model.predict(Xte)
            mae = mean_absolute_error(te[target], pred)
            r = rmse(te[target], pred)
            rows.append([frac, len(subset_pts), target, mae, r])

    out = pd.DataFrame(rows, columns=["frac", "n_farms", "target", "mae", "rmse"])
    out.to_csv(ML_DIR / "eval_learning_curve.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Experiment D: 5-fold cross-validation by point (robust metric)
# ---------------------------------------------------------------------------

def experiment_d(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EXPERIMENT D — 5-fold CV by point: model vs 3 baselines")
    print("=" * 78)

    rows = []
    for target in TARGETS:
        t0 = time.time()
        res = cv_eval(df, target, weight_linear, {})
        elapsed = time.time() - t0
        rows.append([target, res["mae"], res["rmse"], res["r2"],
                     res["base_persist_mae"], res["base_persist_rmse"],
                     res["base_mean_mae"], res["base_mean_rmse"],
                     res["base_median_mae"], res["base_median_rmse"],
                     elapsed])
        print(f"    {target:<22} ({elapsed:.1f}s)")
        print(f"      model          : MAE={res['mae']:6.2f}  RMSE={res['rmse']:6.2f}  R2={res['r2']:6.3f}")
        print(f"      base persist   : MAE={res['base_persist_mae']:6.2f}  RMSE={res['base_persist_rmse']:6.2f}")
        print(f"      base mean      : MAE={res['base_mean_mae']:6.2f}  RMSE={res['base_mean_rmse']:6.2f}")
        print(f"      base median    : MAE={res['base_median_mae']:6.2f}  RMSE={res['base_median_rmse']:6.2f}")

    out = pd.DataFrame(rows, columns=[
        "target", "cv_mae", "cv_rmse", "cv_r2",
        "base_persist_mae", "base_persist_rmse",
        "base_mean_mae", "base_mean_rmse",
        "base_median_mae", "base_median_rmse", "elapsed_s",
    ])
    out.to_csv(ML_DIR / "eval_cv.csv", index=False)

    print("\n  Note: R2 is reported for reference only — it is unreliable on")
    print("        zero-inflated targets (many 0-deficit rows depress it).")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("ALSRS — Model Evaluation")
    print(f"Started: {now()}")
    print("=" * 78)

    df = pd.read_csv(DATASET)
    print(f"Dataset: {DATASET.name}  ({df.shape[0]} rows x {df.shape[1]} cols)")

    t0 = time.time()
    experiment_a(df)
    experiment_b(df)
    experiment_c(df)
    experiment_d(df)

    print("\n" + "=" * 78)
    print(f"All experiments done in {time.time() - t0:.1f}s  (ended {now()})")
    print("=" * 78)


if __name__ == "__main__":
    main()
