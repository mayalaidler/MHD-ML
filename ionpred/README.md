# ionpred

Predict ion abundances in FLASH MHD turbulence simulations from local gas
properties (density, temperature, velocity, magnetic field) with machine
learning — for **any tracked species**, with honest evaluation.

Developed for silicon ionization states (Si I / Si II / Si IV) in the
MAIHEM-chemistry ISM boxes; works unchanged for any of the ~40 species
fields FLASH tracks (C, N, O, Ne, Na, Mg, S, Ca, Fe ions).

## Install

```bash
pip install -e .           # core (ridge + gradient boosting)
pip install -e ".[nn]"     # + PyTorch neural network
pip install -e ".[dev]"    # + pytest
```

## Quickstart

```bash
# What can I predict in this checkpoint?
ionpred species /path/to/ISM_hdf5_chk_0006

# Train gradient boosting on Si IV with a spatial hold-out
ionpred run /path/to/ISM_hdf5_chk_0006 --species si3p --out results/si4_gbm
```

Outputs: `predictions.npz` (y_true / y_pred / test features),
`model.joblib`, and `metadata.json` recording the full configuration,
seed, package versions, and metrics. Output directories are never
overwritten.

## Why this isn't just `model.fit(X, y)`

Three lessons are baked into the defaults, learned the hard way on the
silicon ions:

**1. Trace species are dominated by a numerical floor.** In cells where
an ion physically doesn't exist, the chemistry solver stores an arbitrary
tiny value (10⁻²⁰–10⁻³³). Those digits are noise. Asking one regressor to
fit both branches injects irreducible error, and R² computed over both is
mostly measuring floor-vs-physical classification. `ionpred` detects the
floor automatically (`floors.detect_floor`) and, when one exists, uses a
two-stage **hurdle model**: a classifier for "is the ion present?" and a
regressor for "how much?", each reported separately.

**2. R² is not comparable across species.** R² normalizes error by the
target's variance, and Si I (σ ≈ 0.4 dex in a cold box) can score *lower*
than Si II (σ ≈ 0.8 dex) at identical absolute accuracy — or −86 on a
box where its variance is nearly zero. Every evaluation therefore also
reports **RMSE in dex** and the fraction of cells within 0.5/1 dex
(`metrics.evaluate`).

**3. Cells are not independent samples.** Random train/test splits leak:
neighboring cells are nearly identical. Split along a generalization
dimension instead — space (`spatial_split`), time (`temporal_split`,
which holds out *middle* checkpoints by default because extrapolating a
fitted time trend fails), or simulation (`loso_folds`). And pair features
with targets **from the same cells**: an early version of this pipeline
paired features and targets from independent random subsamples and scored
LOSO R² ≈ −40 without erroring.

Model guidance: `gbm` (gradient boosting) is the strongest cheap default.
`ridge` (degree-2 polynomial) is interpretable but extrapolates violently
on targets spanning more than ~10 dex. `nn` matches gbm but single runs
vary by ΔR² ≈ 0.1 with initialization — run several seeds and report the
spread (all ionpred models are seeded and deterministic by default).

## Library use

```python
from ionpred import (read_fields, build_features, valid_mask,
                     detect_floor, make_model, HurdleModel,
                     spatial_split, evaluate)
from ionpred.features import REQUIRED_FIELDS
import numpy as np

fields = read_fields(chk, REQUIRED_FIELDS + ["si3p"])
target = fields.pop("si3p")
m = valid_mask(fields, target)
X = build_features({k: v[m] for k, v in fields.items()})
y = np.log10(target[m])

floor = detect_floor(y)
model = HurdleModel(floor=floor) if floor is not None else make_model("gbm")
train, test = spatial_split(np.arange(len(y)))
model.fit(X[train], y[train])
print(evaluate(y[test], model.predict(X[test])))
```

## Tests

```bash
pytest
```

## License

MIT
