# ionpred

**Predict ion abundances in astrophysical simulations with machine learning —
instead of recomputing the expensive chemistry.**

`ionpred` trains models to predict how ionized each element is in a cell of a
FLASH magnetohydrodynamics (MHD) simulation, using only the local gas
properties (density, temperature, velocity, magnetic field). It is built for
**any tracked ion species**, and it evaluates itself honestly so you know when
to trust a prediction and when not to.

---

## Why this exists

Simulations of interstellar and circumgalactic gas (here, the
[MAIHEM](https://maihem.org) non-equilibrium chemistry package built on FLASH)
track the ionization state of dozens of elements — how many electrons each
atom has lost. This chemistry network is **computationally expensive**, often
as costly as the hydrodynamics itself. If you finish a simulation and later
want the abundance of an ion it didn't save, you normally have to **rerun the
whole simulation** — days of supercomputer time.

`ionpred` asks a simpler question: *can a machine-learning model predict that
ion's abundance directly from the gas conditions already in the snapshot?* If
so, a days-long recomputation becomes a prediction that takes seconds. The
package was developed and validated on three silicon ionization states
(Si I, Si II, Si IV) and the code runs unchanged on any of the ~64 species
fields the reference simulations track (H, He, C, N, O, Ne, Na, Mg, Si, S,
Ca, Fe ions).

> **Accuracy caveat.** Prediction accuracy has only been *measured* for the
> silicon states. It depends strongly on whether a species is a **dominant
> state** (abundant in most cells → predicted to ~0.2 dex) or a **trace
> state** (nearly absent in most cells → much harder). Always run the built-in
> evaluation before trusting a new species; the metrics are designed to tell
> you the truth, not a flattering number.

---

## Install

```bash
pip install -e .           # core (ridge + gradient boosting)
pip install -e ".[nn]"     # + PyTorch neural network
pip install -e ".[dev]"    # + pytest
```

Requires Python ≥ 3.10.

## Quickstart

```bash
# 1. See which ion species a checkpoint contains
ionpred species /path/to/ISM_hdf5_chk_0006

# 2. Train and evaluate a model on one ion (Si IV here) with a spatial split
ionpred run /path/to/ISM_hdf5_chk_0006 --species si3p --out results/si4_gbm
```

Species names follow FLASH's convention: `si  ` = Si I (neutral),
`sip ` = Si II, `si2p` = Si III, `si3p` = Si IV, and so on for other elements.

Every run writes three files to the output directory:

| File | Contents |
|------|----------|
| `predictions.npz` | true values, predicted values, and the test features |
| `model.joblib`    | the trained model, ready to reload and reuse |
| `metadata.json`   | full configuration, random seed, package versions, and metrics — so any result can be reproduced exactly |

Output directories are **never overwritten** — a safeguard against silently
destroying a previous result.

---

## The three things `ionpred` gets right (that a naive `model.fit(X, y)` does not)

These are the lessons that separate a trustworthy scientific tool from one
that merely runs. Each is baked into the defaults.

### 1. Trace species hide at a numerical floor

In cells where an ion physically doesn't exist, the chemistry solver still
stores a tiny placeholder value (10⁻²⁰–10⁻³³) for numerical stability. **Those
digits are noise, not physics.** A single regressor forced to fit both the
real values and the noise floor wastes its effort and produces a misleading
score. `ionpred` **automatically detects the floor** (`floors.detect_floor`)
and, when one exists, switches to a two-stage **hurdle model**:

- a **classifier** answers *"is the ion present at all?"*
- a **regressor** answers *"how much?"* — trained only on the cells where it's
  really present.

Both are reported separately, so "we can locate the ion" and "we can measure
its abundance" are never conflated.

### 2. R² alone is misleading — always read RMSE too

R² measures error relative to how spread out the target is. That means the
*same* absolute accuracy can score very different R² values on different ions:
a narrow-ranged species like Si I can score a **lower** R² than a wide-ranged
one like Si II even when it is predicted *more* accurately in real units. On a
simulation where an ion barely varies, R² can even go strongly negative while
the actual error is small. Every evaluation therefore also reports **RMSE in
dex** (the scale-free, physically meaningful error) and the fraction of cells
predicted within 0.5 and 1 dex (`metrics.evaluate`).

### 3. Neighboring cells leak — split deliberately

Cells next to each other in a turbulence simulation are nearly identical, so a
random train/test split puts near-duplicates on both sides and **inflates the
score without measuring real generalization**. `ionpred` provides three
leakage-aware splits, each answering a different scientific question:

| Split | Question it answers |
|-------|--------------------|
| `spatial_split` | Does the model generalize across *space* within one snapshot? |
| `temporal_split` | Does it hold at *times* it never trained on? (holds out *middle* checkpoints by default, since extrapolating a fitted time trend fails) |
| `loso_folds` | Does it transfer to an *entirely unseen simulation*? |

The package also enforces that each row's features and target come from the
**same cell** — an early version of this pipeline accidentally paired them
from independent random samples and scored a meaningless R² ≈ −40 *without
ever raising an error*, which is exactly why silent-failure guards matter.

---

## Which model should I use?

| Model | When to use it |
|-------|---------------|
| `gbm` (gradient boosting) | **Best default.** Strong, cheap, deterministic, degrades gracefully outside the training range. |
| `ridge` (degree-2 polynomial) | Interpretable baseline — but extrapolates violently on targets spanning more than ~10 dex, so avoid it for wide-ranged trace species. |
| `nn` (neural network) | Matches `gbm`, but a single run varies by ΔR² ≈ 0.1 with random initialization — run several seeds and report the spread. |

All models are seeded and deterministic by default, so results reproduce
exactly.

---

## Library use

Everything the CLI does is available as a small, composable API:

```python
from ionpred import (read_fields, build_features, valid_mask,
                     detect_floor, make_model, HurdleModel,
                     spatial_split, evaluate)
from ionpred.features import REQUIRED_FIELDS
import numpy as np

# Load features + one ion target from the same cells of a checkpoint
fields = read_fields(chk, REQUIRED_FIELDS + ["si3p"])
target = fields.pop("si3p")
m = valid_mask(fields, target)
X = build_features({k: v[m] for k, v in fields.items()})
y = np.log10(target[m])

# Pick a hurdle model automatically if the ion sits at a numerical floor
floor = detect_floor(y)
model = HurdleModel(floor=floor) if floor is not None else make_model("gbm")

# Leakage-aware split, fit, and honest evaluation
train, test = spatial_split(np.arange(len(y)))
model.fit(X[train], y[train])
print(evaluate(y[test], model.predict(X[test])))   # R², RMSE (dex), within-0.5/1 dex
```

## Tests

```bash
pytest
```

## License

MIT
