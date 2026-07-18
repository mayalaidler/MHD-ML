#!/usr/bin/env python3
"""
Post-hoc analysis from saved prediction files — no retraining needed.

Part 1: variance-aware metrics table for every saved prediction set:
        R², RMSE (dex), fraction within 0.5 / 1 dex, and the same metrics
        restricted to cells above the species' physical floor.

Part 2: hurdle-model prototype for floor-dominated ions (Si IV):
        stage 1 = classifier (above/below floor), stage 2 = regressor on
        above-floor cells only.  LOSO over simulations, compared against the
        plain single-regressor approach.

Usage:  python3 posthoc_metrics.py            # both parts
        python3 posthoc_metrics.py metrics    # part 1 only
        python3 posthoc_metrics.py hurdle     # part 2 only
"""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, \
    HistGradientBoostingRegressor
from sklearn.metrics import f1_score, r2_score

SEED = 42

# Prediction files: (label, path)  — silently skipped if absent
PRED_FILES = [
    ("single-chk SiI  ridge", "siresults_ridge_fixed/single_spatial_test.npz"),
    ("single-chk SiI  nn",    "siresults_nn_fixed/nn_single_spatial_test.npz"),
    ("single-chk SiII ridge", "results_SiII_ridge/single_spatial_test.npz"),
    ("single-chk SiII nn",    "results_SiII_nn/nn_single_spatial_test.npz"),
    ("single-chk SiIV ridge", "results_SiIV_ridge/single_spatial_test.npz"),
    ("single-chk SiIV nn",    "results_SiIV_nn/nn_single_spatial_test.npz"),
    ("LOSO v2   SiI  ridge",  "multi_sim_v2/SiI/ridge_loso_preds.npz"),
    ("LOSO v2   SiI  hgb",    "multi_sim_v2/SiI/hgb_loso_preds.npz"),
    ("LOSO v2   SiI  nn",     "multi_sim_v2/SiI/nn_loso_preds.npz"),
    ("LOSO v2   SiII ridge",  "multi_sim_v2/SiII/ridge_loso_preds.npz"),
    ("LOSO v2   SiII hgb",    "multi_sim_v2/SiII/hgb_loso_preds.npz"),
    ("LOSO v2   SiII nn",     "multi_sim_v2/SiII/nn_loso_preds.npz"),
    ("LOSO v2   SiIV ridge",  "multi_sim_v2/SiIV/ridge_loso_preds.npz"),
    ("LOSO v2   SiIV hgb",    "multi_sim_v2/SiIV/hgb_loso_preds.npz"),
    ("LOSO v2   SiIV nn",     "multi_sim_v2/SiIV/nn_loso_preds.npz"),
]


def detect_floor(y_log, lo=-33.0, hi=-5.0, nbins=120):
    """
    Auto-detect the numerical-floor threshold of a log10 target:
    the least-populated histogram bin between the distribution's two
    outermost peaks.  Returns None when the target has no floor branch
    (spread too small to be bimodal).
    """
    if y_log.min() > -12:          # no floor branch present
        return None
    counts, edges = np.histogram(y_log, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Peaks: highest bin below -14 (floor cluster) and above -14 (physical)
    below = centers < -14
    if counts[below].sum() == 0 or counts[~below].sum() == 0:
        return None
    p_floor = centers[below][np.argmax(counts[below])]
    p_phys  = centers[~below][np.argmax(counts[~below])]
    between = (centers > p_floor) & (centers < p_phys)
    if between.sum() == 0:
        return None
    return float(centers[between][np.argmin(counts[between])])


def metrics_block(y_true, y_pred):
    err = y_pred - y_true
    return {
        "R2":    r2_score(y_true, y_pred),
        "RMSE":  float(np.sqrt(np.mean(err**2))),
        "f0.5":  float((np.abs(err) < 0.5).mean()),
        "f1.0":  float((np.abs(err) < 1.0).mean()),
        "N":     len(y_true),
    }


def part1_metrics():
    print(f"{'prediction set':24s} {'N':>8s} {'R²':>7s} {'RMSE':>6s} "
          f"{'<0.5dex':>8s} {'<1dex':>6s}   above-floor: R² / RMSE / N")
    print("-" * 110)
    rows = {}
    for label, path in PRED_FILES:
        if not os.path.exists(path):
            continue
        with np.load(path) as d:
            y_true, y_pred = d["y_true"], d["y_pred"]
        m = metrics_block(y_true, y_pred)
        floor = detect_floor(y_true)
        if floor is not None:
            keep = y_true > floor
            mf = metrics_block(y_true[keep], y_pred[keep])
            extra = (f"(floor {floor:6.1f})  {mf['R2']:6.3f} / "
                     f"{mf['RMSE']:5.2f} / {mf['N']}")
        else:
            extra = "—"
        print(f"{label:24s} {m['N']:>8d} {m['R2']:7.3f} {m['RMSE']:6.2f} "
              f"{m['f0.5']:8.2%} {m['f1.0']:6.2%}   {extra}")
        rows[label] = {"all": m, "floor": floor,
                       "above_floor": mf if floor is not None else None}
    with open("posthoc_metrics.json", "w") as fh:
        json.dump(rows, fh, indent=2, default=float)
    print("\n[SAVED] posthoc_metrics.json")


def part2_hurdle(tag="SiIV"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from multi_sim_v2 import make_features, DATASET_PATH, SIM_NAMES

    with np.load(DATASET_PATH) as d:
        X_raw, sim_ids = d["X"], d["sim_ids"]
        y = d[f"y_{tag}"]

    X, _ = make_features(X_raw)
    y_log = np.log10(np.clip(y, 1e-35, None))
    floor = detect_floor(y_log)
    if floor is None:
        print(f"[HURDLE {tag}] no floor branch detected — plain regression is fine")
        return
    print(f"\n[HURDLE {tag}] auto-detected floor threshold: {floor:.1f}")
    above = y_log > floor
    print(f"[HURDLE {tag}] {above.mean():.1%} of cells above floor")

    rows = []
    for held_out in np.unique(sim_ids):
        tr, te = sim_ids != held_out, sim_ids == held_out

        clf = HistGradientBoostingClassifier(max_iter=200, random_state=SEED)
        clf.fit(X[tr], above[tr])
        cls_pred = clf.predict(X[te])
        f1 = f1_score(above[te], cls_pred) if above[te].any() else np.nan

        reg = HistGradientBoostingRegressor(
            max_iter=300, early_stopping=True, validation_fraction=0.1,
            random_state=SEED)
        reg.fit(X[tr & above], y_log[tr & above])

        # Combined prediction: floor median where classified below
        combined = np.where(cls_pred, reg.predict(X[te]),
                            np.median(y_log[tr & ~above]))
        r2_combined = r2_score(y_log[te], combined)

        # Regression quality on truly-above-floor cells only
        te_above = te & above
        r2_above = (r2_score(y_log[te_above], reg.predict(X[te_above]))
                    if te_above.sum() > 10 else np.nan)

        rows.append((int(held_out), f1, r2_combined, r2_above))
        print(f"  sim {held_out:2d} ({SIM_NAMES[held_out]:>18s})  "
              f"clf F1={f1:5.3f}  combined R²={r2_combined:7.3f}  "
              f"above-floor R²={r2_above:7.3f}", flush=True)

    arr = np.array(rows)
    print(f"\n[HURDLE {tag}] mean clf F1={np.nanmean(arr[:,1]):.3f}  "
          f"mean combined R²={np.nanmean(arr[:,2]):.3f}  "
          f"mean above-floor R²={np.nanmean(arr[:,3]):.3f}")
    np.save(f"multi_sim_v2/{tag}_hurdle_results.npy", arr)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "metrics"):
        part1_metrics()
    if mode in ("all", "hurdle"):
        for tag in ("SiI", "SiII", "SiIV"):
            part2_hurdle(tag)
