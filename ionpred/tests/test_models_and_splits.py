import numpy as np
import pytest

from ionpred.models import HurdleModel, make_model
from ionpred.splits import loso_folds, spatial_split, temporal_split


def _hurdle_data(n=6000, seed=0):
    """Synthetic floor-dominated species: abundance tracks temperature
    above a cutoff, floor noise below it."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))
    present = X[:, 0] > 0.2
    y = np.where(present,
                 -5 + 1.5 * X[:, 0] + rng.normal(0, 0.2, n),
                 rng.normal(-25, 1.0, n))
    return X, y


def test_hurdle_beats_plain_regressor_on_floor_data():
    X, y = _hurdle_data()
    tr, te = slice(0, 4000), slice(4000, None)

    plain = make_model("gbm").fit(X[tr], y[tr])
    hurdle = HurdleModel(floor=-15.0).fit(X[tr], y[tr])

    def rmse(pred):
        return float(np.sqrt(np.mean((pred - y[te]) ** 2)))

    assert rmse(hurdle.predict(X[te])) <= rmse(plain.predict(X[te])) + 0.1


def test_hurdle_rejects_unimodal_target():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (500, 3))
    y = rng.normal(-4, 0.5, 500)
    with pytest.raises(ValueError):
        HurdleModel(floor=-15.0).fit(X, y)


def test_spatial_split_disjoint():
    x = np.random.default_rng(0).uniform(0, 1, 1000)
    for strategy in ("median", "percentile", "thirds"):
        tr, te = spatial_split(x, strategy)
        assert not (tr & te).any()
        assert tr.sum() > 0 and te.sum() > 0


def test_temporal_split_interpolate_holds_out_middle():
    chk = np.repeat(np.arange(10), 5)
    tr, te = temporal_split(chk, test_frac=0.2, mode="interpolate")
    held = np.unique(chk[te])
    assert held.min() > 0 and held.max() < 9


def test_loso_folds_cover_everything():
    sids = np.repeat([0, 1, 2], 10)
    folds = list(loso_folds(sids))
    assert [f[0] for f in folds] == [0, 1, 2]
    for _, tr, te in folds:
        assert not (tr & te).any()
        assert (tr | te).all()
