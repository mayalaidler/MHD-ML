import numpy as np

from ionpred.metrics import evaluate


def test_perfect_prediction():
    y = np.random.default_rng(0).normal(-4, 1, 1000)
    m = evaluate(y, y, floor=None)
    assert m["r2"] == 1.0
    assert m["rmse_dex"] == 0.0
    assert m["frac_within_0.5dex"] == 1.0


def test_r2_is_variance_normalized():
    """Identical absolute error scores lower R² on a narrower target —
    the effect that makes cross-species R² comparisons misleading."""
    rng = np.random.default_rng(0)
    err = rng.normal(0, 0.3, 20000)
    narrow = rng.normal(-4, 0.4, 20000)
    wide = rng.normal(-5, 2.0, 20000)
    m_narrow = evaluate(narrow, narrow + err, floor=None)
    m_wide = evaluate(wide, wide + err, floor=None)
    assert abs(m_narrow["rmse_dex"] - m_wide["rmse_dex"]) < 0.02
    assert m_wide["r2"] > m_narrow["r2"] + 0.2


def test_above_floor_block_present_for_bimodal_target():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(-4, 0.5, 3000),
                        rng.normal(-25, 1.0, 3000)])
    pred = y + rng.normal(0, 0.2, len(y))
    m = evaluate(y, pred)  # floor='auto'
    assert m["floor"] is not None
    assert "above_floor" in m
    assert m["above_floor"]["n"] >= 2900
    assert m["floor_separation_accuracy"] > 0.99
