"""End-to-end single-checkpoint pipeline with reproducibility metadata."""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np

from . import __version__
from .features import REQUIRED_FIELDS, build_features, valid_mask
from .floors import detect_floor
from .io import read_fields, species_label
from .metrics import evaluate
from .models import HurdleModel, make_model
from .splits import spatial_split

SEED = 42


def run_single_checkpoint(
    checkpoint: str,
    species: str,
    model: str = "gbm",
    split: str = "median",
    out_dir: str | None = None,
    sample: int = 500_000,
    hurdle: str = "auto",
    seed: int = SEED,
) -> dict:
    """Train on one checkpoint with a spatial hold-out and save
    predictions, the fitted model, and a metadata JSON.

    hurdle: 'auto' uses a HurdleModel when a numerical floor is detected,
    'on' forces it, 'off' disables it.
    """
    label = species_label(species) or species.strip()
    fields = read_fields(checkpoint, REQUIRED_FIELDS + [species])
    target = fields.pop(species)

    mask = valid_mask(fields, target)
    fields = {k: v[mask] for k, v in fields.items()}
    target = target[mask]

    # Spatial proxy: FLASH cell data raveled in block order has x varying
    # within blocks; using a real coordinate would need the mesh, so we
    # use the raveled index as an ordering proxy consistent across fields.
    x_proxy = np.arange(len(target), dtype=np.float64)

    if sample and len(target) > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(target), size=sample, replace=False)
        fields = {k: v[idx] for k, v in fields.items()}
        target, x_proxy = target[idx], x_proxy[idx]

    X = build_features(fields)
    y_log = np.log10(target)

    train_mask, test_mask = spatial_split(x_proxy, strategy=split)
    X_tr, y_tr = X[train_mask], y_log[train_mask]
    X_te, y_te = X[test_mask], y_log[test_mask]

    floor = detect_floor(y_tr)
    use_hurdle = (hurdle == "on") or (hurdle == "auto" and floor is not None)

    if use_hurdle:
        if floor is None:
            raise ValueError("hurdle requested but no floor detected")
        est = HurdleModel(floor=floor, regressor=model, seed=seed)
    else:
        est = make_model(model, seed=seed)
    est.fit(X_tr, y_tr)
    y_pred = est.predict(X_te)

    results = evaluate(y_te, y_pred, floor=floor)
    results["species"], results["label"], results["model"] = \
        species, label, model
    results["hurdle_used"] = bool(use_hurdle)

    if out_dir:
        _save(out_dir, est, y_te, y_pred, X_te, results, dict(
            checkpoint=os.path.abspath(checkpoint), species=species,
            label=label, model=model, split=split, sample=sample,
            hurdle=hurdle, seed=seed))
    return results


def _save(out_dir, est, y_true, y_pred, X_test, results, config):
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(
            f"{out_dir} exists and is not empty — refusing to overwrite. "
            "Pick a new output directory.")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "predictions.npz"),
             y_true=y_true, y_pred=y_pred, X_test=X_test)
    joblib.dump(est, os.path.join(out_dir, "model.joblib"))
    meta = {
        "config": config,
        "results": results,
        "provenance": {
            "ionpred_version": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
