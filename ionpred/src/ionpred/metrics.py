"""Variance-aware evaluation for log-space abundance predictions.

R² alone misleads when comparing species: it is error normalized by the
target's variance, and different ions span wildly different ranges (a
narrow Si I distribution can score a *lower* R² than Si II at identical
absolute accuracy).  Always read R² together with RMSE in dex.
"""

from __future__ import annotations

import numpy as np

from .floors import detect_floor


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    floor: float | None = "auto",
) -> dict:
    """Metrics dict for log-space predictions.

    Returns R², RMSE (dex), fraction within 0.5 and 1 dex, N — and, when
    the target has a numerical floor branch, the same metrics restricted
    to physically meaningful (above-floor) cells plus the accuracy of
    floor-vs-physical separation.

    Pass ``floor=None`` to skip floor handling, a number to force a
    threshold, or the default ``"auto"`` to detect it.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = _block(y_true, y_pred)

    if floor == "auto":
        floor = detect_floor(y_true)
    out["floor"] = floor
    if floor is not None:
        above_t = y_true > floor
        above_p = y_pred > floor
        out["floor_separation_accuracy"] = float((above_t == above_p).mean())
        if above_t.sum() >= 10:
            out["above_floor"] = _block(y_true[above_t], y_pred[above_t])
    return out


def _block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rmse_dex": float(np.sqrt(np.mean(err**2))),
        "frac_within_0.5dex": float((np.abs(err) < 0.5).mean()),
        "frac_within_1dex": float((np.abs(err) < 1.0).mean()),
        "n": int(len(y_true)),
    }
