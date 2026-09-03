"""
Training Dataset Collector
==========================

Runs the ALSRS pipeline for every point in a points CSV (e.g. the cacao belt)
and accumulates the ML training rows into a single dataset.

For each point it:

    1. runs ``crop_viability`` and ``water_balance`` (downloads + computes),
    2. joins ``water_balance_labels`` + ``temperature_biweekly`` +
       ``precipitation_biweekly`` + ``spei_biweekly`` on the biweekly label,
    3. derives the seasonal position (``mes`` 1-12, ``quincena`` 1-2) from the
       label,
    4. appends the resulting rows (features + future-deficit labels) to the
       output CSV.

It is crash-safe: a checkpoint JSON records the completed points, so a re-run
resumes where it left off. Each point is retried a few times on failure.

Parallelism (sharding)
----------------------
Points are independent, so the list can be split into shards and each shard
run in its own process/terminal:

    python ml/collect_training.py --shard 0 --nshards 4
    python ml/collect_training.py --shard 1 --nshards 4
    ...

Each shard writes its own CSV + checkpoint (no coordination needed). Keep the
number of parallel shards low (2-4) to stay well within Earth Engine's
concurrent-request limits. After all shards finish, merge them:

    python ml/collect_training.py --merge --nshards 4

Output columns (approved ML schema):
    point_id, lat, lon, crop, period_start, period_end, label, mes, quincena,
    mean_C, std_C, precip_total_mm, precip_rainy_days, pet_mm,
    spei_1m, spei_3m, spei_6m, spei_12m, AWC_mm, Storage_mm, P_acum_mm,
    WRSI_actual, deficit_pct,  deficit_t1 ... deficit_t12,  suggestion

Note: to restart a shard from scratch, delete BOTH its checkpoint JSON and its
output CSV (they are kept in sync automatically).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Make the project's code folders importable (this file lives in ml/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT / "common", _PROJECT_ROOT / "extraction",
           _PROJECT_ROOT / "analysis", _PROJECT_ROOT / "mapping"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import crop_viability
import water_balance
from crop_viability import DATABASES_DIR


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CROP = "cacao_ccn51"
POINTS_CSV = _PROJECT_ROOT / "ml" / "cacao_points.csv"
MAX_RETRIES = 3
RETRY_BACKOFF_S = 5


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _final_output(crop: str) -> Path:
    return _PROJECT_ROOT / "ml" / f"ml_dataset_{crop}.csv"


def _full_checkpoint(crop: str) -> Path:
    return _PROJECT_ROOT / "ml" / f"checkpoint_{crop}.json"


def _shard_output(crop: str, shard: int) -> Path:
    return _PROJECT_ROOT / "ml" / f"ml_dataset_{crop}_shard{shard}.csv"


def _shard_checkpoint(crop: str, shard: int) -> Path:
    return _PROJECT_ROOT / "ml" / f"checkpoint_{crop}_shard{shard}.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_csv(pattern: str, directory: Path = DATABASES_DIR) -> Path:
    """Returns the most recent CSV matching a glob pattern."""
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No CSV matching '{pattern}' in {directory}")
    return candidates[-1]


def load_checkpoint(path: Path) -> dict:
    """Loads the checkpoint JSON (done/failed point lists)."""
    if path.exists():
        return json.loads(path.read_text())
    return {"done": [], "failed": []}


def save_checkpoint(path: Path, state: dict) -> None:
    """Writes the checkpoint JSON."""
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def build_ml_rows(
    wb_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    precip_df: pd.DataFrame,
    spei_df: pd.DataFrame,
    point_id: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """
    Joins the four per-point CSVs and builds the approved ML rows.

    Features come from: water_balance_labels (AWC, SPEI, storage, WRSI,
    deficit) + temperature (mean_C, std_C) + precipitation (total, rainy days)
    + SPEI (pet_mm). Seasonal position (mes, quincena) is derived from label.
    Labels are the 12 future deficits (deficit_t1..t12); suggestion is kept for
    validation (it is derived, not predicted).
    """
    df = wb_df.merge(
        temp_df[["label", "mean_C", "std_C"]], on="label", how="inner"
    )
    df = df.merge(
        precip_df[["label", "precip_total_mm", "precip_rainy_days"]],
        on="label", how="inner",
    )
    df = df.merge(
        spei_df[["label", "pet_mm"]], on="label", how="inner"
    )

    # Seasonal position: "2016-01_Q1" -> mes=1, quincena=1.
    df["mes"] = df["label"].str.split("-").str[1].str.split("_").str[0].astype(int)
    df["quincena"] = df["label"].str.split("_").str[1].map({"Q1": 1, "Q2": 2})

    # Point identifiers.
    df.insert(0, "point_id", point_id)
    df.insert(1, "lat", lat)
    df.insert(2, "lon", lon)

    feature_cols = [
        "mean_C", "std_C", "precip_total_mm", "precip_rainy_days", "pet_mm",
        "spei_1m", "spei_3m", "spei_6m", "spei_12m",
        "AWC_mm", "Storage_mm", "P_acum_mm", "WRSI_actual", "deficit_pct",
    ]
    label_cols = [f"deficit_t{k}" for k in range(1, 13)]
    order = (
        ["point_id", "lat", "lon", "crop", "period_start", "period_end",
         "label", "mes", "quincena"]
        + feature_cols
        + label_cols
        + ["suggestion"]
    )
    return df[order]


def process_point(point_id: str, lat: float, lon: float) -> pd.DataFrame:
    """
    Runs the pipeline for one point and returns its ML rows.

    The verbose console reports of the pipeline are silenced so the batch log
    stays readable; only the collector's own progress lines are printed.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        _, temp_path = crop_viability.main(
            lat, lon, CROP, area_ha=2.0, regenerate=True
        )
        wb_path = water_balance.main(lat, lon, CROP, regenerate=True)

    wb_df = pd.read_csv(wb_path)
    temp_df = pd.read_csv(temp_path)
    precip_df = pd.read_csv(latest_csv("precipitation_biweekly-v*.csv"))
    spei_df = pd.read_csv(latest_csv("spei_biweekly-v*.csv"))

    return build_ml_rows(wb_df, temp_df, precip_df, spei_df, point_id, lat, lon)


def append_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Appends rows to the output CSV (header only on first write)."""
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def merge_shards(crop: str, nshards: int) -> None:
    """Merges the per-shard CSVs into the final dataset CSV."""
    frames = []
    for shard in range(nshards):
        path = _shard_output(crop, shard)
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        print("No shard files found to merge.")
        return

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["point_id", "label"], keep="first"
    )
    merged.to_csv(_final_output(crop), index=False)
    print(f"Merged {len(frames)} shard(s) -> {_final_output(crop).name} "
          f"({len(merged)} rows)")


# ---------------------------------------------------------------------------
# Args + Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect ALSRS ML training data (optional sharding)."
    )
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index to process (0-based).")
    parser.add_argument("--nshards", type=int, default=1,
                        help="Total number of shards (default: 1).")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all shard CSVs into the final dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.merge:
        merge_shards(CROP, args.nshards)
        return

    points = pd.read_csv(POINTS_CSV)
    total_all = len(points)

    # Shard selection (contiguous chunk) or all points.
    if args.shard is not None:
        if not 0 <= args.shard < args.nshards:
            print(f"Invalid shard {args.shard} (must be 0..{args.nshards - 1}).")
            return
        chunk = math.ceil(total_all / args.nshards)
        start = args.shard * chunk
        end = min(start + chunk, total_all)
        points = points.iloc[start:end].reset_index(drop=True)
        output_csv = _shard_output(CROP, args.shard)
        checkpoint_json = _shard_checkpoint(CROP, args.shard)
        tag = f"shard {args.shard}/{args.nshards}"
    else:
        output_csv = _final_output(CROP)
        checkpoint_json = _full_checkpoint(CROP)
        tag = "all points"

    # Keep checkpoint and output CSV in sync: if the output was deleted,
    # start fresh.
    state = load_checkpoint(checkpoint_json)
    if not output_csv.exists():
        state = {"done": [], "failed": []}
        save_checkpoint(checkpoint_json, state)

    done = set(state["done"])
    failed = set(state["failed"])
    total = len(points)

    print(f"Collecting {CROP} [{tag}]: {total} points "
          f"({len(done)} already done).", flush=True)

    for i, row in points.iterrows():
        point_id = row["point_id"]
        if point_id in done:
            continue
        lat, lon = float(row["lat"]), float(row["lon"])
        name = row["name"]
        start_ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{i + 1}/{total}] {point_id} {name} ... - [{start_ts}]", flush=True)

        ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                rows = process_point(point_id, lat, lon)
                append_to_csv(rows, output_csv)
                done.add(point_id)
                state["done"] = sorted(done)
                save_checkpoint(checkpoint_json, state)
                end_ts = datetime.now().strftime("%H:%M:%S")
                print(f"      OK: {len(rows)} rows - [{end_ts}]", flush=True)
                ok = True
                break
            except Exception as err:
                print(f"      attempt {attempt}/{MAX_RETRIES} failed: {err}",
                      flush=True)
                time.sleep(RETRY_BACKOFF_S * attempt)

        if not ok:
            failed.add(point_id)
            state["failed"] = sorted(failed)
            save_checkpoint(checkpoint_json, state)
            print(f"      FAILED after {MAX_RETRIES} attempts", flush=True)

    print(f"\nDone [{tag}]. OK: {len(done)} | Failed: {len(failed)}")
    if output_csv.exists():
        n_rows = len(pd.read_csv(output_csv))
        print(f"Dataset: {output_csv.name} ({n_rows} rows)")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# How to run the collector (quick reference)
# ---------------------------------------------------------------------------
#
# 1. Launch the shards in parallel (one terminal per shard, or nohup in the
#    background). Each shard writes its own CSV + checkpoint, so they never
#    collide:
#
#        python ml/collect_training.py --shard 0 --nshards 4
#        python ml/collect_training.py --shard 1 --nshards 4
#        python ml/collect_training.py --shard 2 --nshards 4
#        python ml/collect_training.py --shard 3 --nshards 4
#
#    To keep them running after you log out and capture the logs:
#
#        nohup python ml/collect_training.py --shard 0 --nshards 4 \
#            > ml/log_shard0.txt 2>&1 &
#        ... same for shards 1, 2, 3 ...
#
# 2. Watch progress. Each point prints a start timestamp and a finish
#    timestamp, e.g.:
#
#        [1/30] p001 Aguachica_Cesar ... - [14:02:33]
#              OK: 231 rows - [14:05:18]
#
#    Follow a running log with:
#
#        tail -f ml/log_shard0.txt
#
#    Count how many points each shard has finished with:
#
#        grep -c "OK:" ml/log_shard*.txt
#
# 3. Merge everything at the end (only after all shards have finished):
#
#        python ml/collect_training.py --merge --nshards 4
#
#    This writes the final dataset ml/ml_dataset_cacao_ccn51.csv.
#
# 4. Resume after a failure / machine crash. Just re-run the same shard
#    command: the checkpoint JSON records finished points, so the run picks
#    up where it left off (it does not redo finished points). To force a
#    full restart of a shard, delete BOTH its checkpoint JSON and its output
#    CSV, e.g.:
#
#        rm ml/checkpoint_cacao_ccn51_shard0.json \
#           ml/ml_dataset_cacao_ccn51_shard0.csv
#
#    Failed points are also recorded in the checkpoint; after fixing the
#    cause, simply re-run the shard and they will be retried.
