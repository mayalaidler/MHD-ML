import numpy as np

from ionpred.floors import detect_floor


def test_detects_bimodal_floor():
    rng = np.random.default_rng(0)
    physical = rng.normal(-4.0, 0.5, 5000)
    floor = rng.normal(-25.0, 1.5, 5000)
    y = np.concatenate([physical, floor])
    t = detect_floor(y)
    assert t is not None
    assert -22 < t < -7
    # threshold separates the two populations
    assert (physical > t).mean() > 0.99
    assert (floor < t).mean() > 0.99


def test_no_floor_returns_none():
    rng = np.random.default_rng(0)
    y = rng.normal(-4.0, 0.5, 5000)
    assert detect_floor(y) is None
