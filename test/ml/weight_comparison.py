import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

ML_DIR = Path("ml")
df = pd.read_csv(ML_DIR / "ml_dataset_cacao_ccn51.csv")

FEATURES = ["month","biweek","mean_C","std_C","precip_total_mm","precip_rainy_days",
            "pet_mm","spei_1m","spei_3m","spei_6m","spei_12m","AWC_mm","Storage_mm",
            "P_acum_mm","WRSI_1m","deficit_1m","oni"]
TARGETS = ["future_deficit_1m","future_deficit_3m","future_deficit_6m"]
CLASS_THRESHOLDS = [15, 50]
CLASS_NAMES = ["LOW","MODERATE","SEVERE"]

point_ids = df["point_id"].unique()
train_pts, rest = train_test_split(point_ids, test_size=0.30, random_state=42)
val_pts, test_pts = train_test_split(rest, test_size=2/3, random_state=42)
tr = df[df["point_id"].isin(train_pts)]
va = df[df["point_id"].isin(val_pts)]
te = df[df["point_id"].isin(test_pts)]
print(f"Train {len(train_pts)} | Val {len(val_pts)} | Test {len(test_pts)} farms")

def weight_linear(d): return d
def weight_squared(d): return d ** 2
def weight_plus1sq(d): return 1 + d ** 2
WEIGHTS = {"linear": weight_linear, "squared": weight_squared, "plus1sq": weight_plus1sq}

def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))
def bin_vector(v, t): return np.searchsorted(t, np.asarray(v), side="left")
def worst(preds): return np.maximum(np.maximum(preds[TARGETS[0]], preds[TARGETS[1]]), preds[TARGETS[2]])
def sugg(preds): return np.array(CLASS_NAMES)[bin_vector(worst(preds), CLASS_THRESHOLDS)]

Xtr = tr[FEATURES].to_numpy()
Xva = va[FEATURES].to_numpy()
Xte = te[FEATURES].to_numpy()

rows = []
for wname, wfn in WEIGHTS.items():
    va_preds, te_preds = {}, {}
    for t in TARGETS:
        m = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=1,
                                  random_state=42, n_jobs=-1)
        m.fit(Xtr, tr[t].to_numpy(), sample_weight=wfn(tr[t].to_numpy()))
        va_preds[t] = m.predict(Xva)
        te_preds[t] = m.predict(Xte)
    row = {"weight": wname}
    for tag, dset, preds in [("val", va, va_preds), ("test", te, te_preds)]:
        for i, t in enumerate(TARGETS):
            h = ["1m","3m","6m"][i]
            row[f"{tag}_mae_{h}"] = round(mean_absolute_error(dset[t], preds[t]), 2)
            row[f"{tag}_rmse_{h}"] = round(rmse(dset[t], preds[t]), 2)
        ts = sugg({t: dset[t].to_numpy() for t in TARGETS})
        ps = sugg(preds)
        for cls in ["MODERATE","SEVERE"]:
            m = ts == cls
            row[f"{tag}_recall_{cls}"] = round((ps[m] == cls).mean(), 3)
    rows.append(row)

out = pd.DataFrame(rows)
print("\n" + out.to_string(index=False))
out.to_csv(ML_DIR / "eval_weights_701020.csv", index=False)
print("\nSaved to ml/eval_weights_701020.csv")
