"""Leakage-aware train/test splits.

Never split turbulence-simulation cells randomly: neighboring cells are
nearly identical, so a random split leaks the test set into training.
Split along a dimension the model must generalize across instead —
space, time, or simulation.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


def spatial_split(
    x_pos: np.ndarray, strategy: str = "median"
) -> tuple[np.ndarray, np.ndarray]:
    """Split cells by x-coordinate. Returns (train_mask, test_mask).

    'median'     train x < median, test x >= median
    'percentile' train bottom 25%, test top 25% (gap discarded — least
                 leakage across the boundary)
    'thirds'     train bottom third, test top third
    """
    if strategy == "median":
        t = np.median(x_pos)
        return x_pos < t, x_pos >= t
    if strategy == "percentile":
        return x_pos < np.percentile(x_pos, 25), x_pos > np.percentile(x_pos, 75)
    if strategy == "thirds":
        return (x_pos < np.percentile(x_pos, 33.33),
                x_pos > np.percentile(x_pos, 66.67))
    raise ValueError(f"unknown strategy {strategy!r}")


def temporal_split(
    chk_ids: np.ndarray, test_frac: float = 0.2, mode: str = "interpolate"
) -> tuple[np.ndarray, np.ndarray]:
    """Split cells by checkpoint index. Returns (train_mask, test_mask).

    'interpolate' holds out checkpoints from the *middle* of the time
    range, so the model interpolates in time.  'extrapolate' holds out
    the last checkpoints — a strictly harder ask that models with any
    time-dependent feature tend to fail (they extrapolate a fitted
    trend), so use it only when forecasting really is the question.
    """
    unique = np.unique(chk_ids)
    n_test = max(1, int(round(test_frac * len(unique))))
    if mode == "extrapolate":
        test_chks = unique[-n_test:]
    elif mode == "interpolate":
        mid = len(unique) // 2
        lo = max(0, mid - n_test // 2)
        test_chks = unique[lo:lo + n_test]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    test_mask = np.isin(chk_ids, test_chks)
    return ~test_mask, test_mask


def loso_folds(
    sim_ids: np.ndarray,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Leave-one-simulation-out folds: yields (sim_id, train_mask,
    test_mask) for each unique simulation."""
    for sid in np.unique(sim_ids):
        test_mask = sim_ids == sid
        yield int(sid), ~test_mask, test_mask
