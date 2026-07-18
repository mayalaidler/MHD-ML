"""Model zoo: ridge, gradient boosting, optional torch NN, and the
hurdle model for floor-dominated species."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42


def make_model(kind: str, seed: int = SEED):
    """'ridge' (degree-2 polynomial RidgeCV), 'gbm', or 'nn' (needs the
    [nn] extra).

    Notes from validation on the Si ions:
    - ridge extrapolates violently on wide (>10 dex) targets; prefer
      'gbm' or 'nn' there, or use HurdleModel.
    - 'gbm' is the strongest and cheapest default on tabular features.
    """
    if kind == "ridge":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-5, 2, 20), cv=5)),
        ])
    if kind == "gbm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("gbm", HistGradientBoostingRegressor(
                max_iter=300, early_stopping=True, validation_fraction=0.1,
                random_state=seed)),
        ])
    if kind == "nn":
        return _make_nn(seed)
    raise ValueError(f"unknown model kind {kind!r}")


class HurdleModel:
    """Two-stage model for floor-dominated species.

    Stage 1 classifies whether a cell is above the species' numerical
    floor ("is the ion physically present?"); stage 2 regresses the
    abundance on above-floor cells only.  Cells classified as floor get
    the training floor median.  This keeps solver noise out of the
    regression and makes the two questions separately reportable.
    """

    def __init__(self, floor: float, regressor="gbm", seed: int = SEED):
        self.floor = floor
        self.classifier = HistGradientBoostingClassifier(
            max_iter=200, random_state=seed)
        self.regressor = (make_model(regressor, seed)
                          if isinstance(regressor, str) else regressor)
        self.floor_fill_: float | None = None

    def fit(self, X: np.ndarray, y_log: np.ndarray):
        above = y_log > self.floor
        if above.all() or not above.any():
            raise ValueError(
                "target has no floor/physical mix at this threshold; "
                "use a plain regressor instead")
        self.classifier.fit(X, above)
        self.regressor.fit(X[above], y_log[above])
        self.floor_fill_ = float(np.median(y_log[~above]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        above = self.classifier.predict(X)
        out = np.full(len(X), self.floor_fill_, dtype=np.float64)
        if above.any():
            out[above] = self.regressor.predict(X[above])
        return out


def _make_nn(seed: int):
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "the 'nn' model needs pytorch: pip install ionpred[nn]") from e
    from ._nn import TorchRegressor
    return TorchRegressor(seed=seed)
