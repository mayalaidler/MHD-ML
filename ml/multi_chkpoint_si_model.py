#muti checkpoint, time included as a feature, 

"""
MHD Simulation ML: Ridge + Neural Network
Predicts silicon abundance (si  ) from thermo-hydrodynamic features.
Leakage prevention: train/test split is done on checkpoint files (temporal),
never on individual cells within a checkpoint.
"""

import glob
import math
import os
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = "/scratch/ebuie/ISO_Turb/midway/mhd_1e8/1E23_S100_z01_mhd/"
TEST_FRAC   = 0.2
RANDOM_SEED = 42
GAMMA       = 5.0 / 3.0
MU0         = 4.0 * math.pi * 1e-7   # SI; change to CGS (1.0) if needed

# Target ion: 'si  '=Si I (neutral), 'sip '=Si II, 'si2p'=Si III, 'si3p'=Si IV
# Override with e.g.:  ION_FIELD='sip ' python3 multi_chkpoint_si_model.py
ION_FIELD = os.environ.get("ION_FIELD", "si  ")
ION_TAG   = {"si  ": "SiI", "sip ": "SiII", "si2p": "SiIII", "si3p": "SiIV"}.get(
    ION_FIELD, ION_FIELD.strip())

# Temporal split mode:
#   'extrapolate' (original) — test on the LAST test_frac of checkpoints.
#     Hard by construction: checkpoint_idx is a feature and test values lie
#     outside the training range, so models extrapolate a fitted time trend.
#   'interpolate' — test on a middle block of checkpoints, training on both
#     earlier and later ones.
TEMPORAL_SPLIT_MODE = os.environ.get("TEMPORAL_SPLIT_MODE", "extrapolate")

_suffix = ("" if ION_FIELD == "si  " else f"_{ION_TAG}") + (
    "_interp" if TEMPORAL_SPLIT_MODE == "interpolate" else "")
RESULTS_TXT = f"results{_suffix}.txt"
MODEL_DIR   = f"saved_models{_suffix}"

# Random cells kept per checkpoint (0 = keep all).  Full checkpoints are
# 16.7M cells each, so loading every file unsampled needs >25 GB of RAM.
CELLS_PER_CHK = int(os.environ.get("CELLS_PER_CHK", "0"))

FEATURE_COLS = [
    "log_rho", "log_T",
    "vx", "vy", "vz", "vmag",
    "log_B", "mach", "log_beta",
    "checkpoint_idx",
]
TARGET_COL = "log_si"

def _read_field(f: h5py.File, name: str) -> np.ndarray:
    """Return a flattened float64 array for a FLASH HDF5 variable."""
    # FLASH 4-char names may be stored with or without trailing spaces
    for key in [name, name.rstrip(), name + " " * (4 - len(name.rstrip()))]:
        if key in f:
            return f[key][()].astype(np.float64).ravel()
    raise KeyError(f"Field '{name}' not found in file. Available: {list(f.keys())}")


def load_checkpoint(path: str, ckpt_idx: int) -> tuple:
    """
    Load one FLASH HDF5 checkpoint and return (X dict, y array).
    All feature engineering happens here so it is always fit on raw data.
    """
    with h5py.File(path, "r") as f:
        dens = _read_field(f, "dens")
        temp = _read_field(f, "temp")
        velx = _read_field(f, "velx")
        vely = _read_field(f, "vely")
        velz = _read_field(f, "velz")
        magx = _read_field(f, "magx")
        magy = _read_field(f, "magy")
        magz = _read_field(f, "magz")
        pres = _read_field(f, "pres")
        si   = _read_field(f, ION_FIELD)

    n = len(dens)

    vmag  = np.sqrt(velx**2 + vely**2 + velz**2)
    Bmag  = np.sqrt(magx**2 + magy**2 + magz**2)
    cs    = np.sqrt(np.clip(GAMMA * pres / dens, 1e-30, None))
    mach  = vmag / cs
    beta  = 2.0 * pres / np.clip(Bmag**2 / MU0, 1e-30, None)

    # Guard against non-positive values before log10
    eps = 1e-30
    X = {
        "log_rho":        np.log10(np.clip(dens,  eps, None)),
        "log_T":          np.log10(np.clip(temp,  eps, None)),
        "vx":             velx,
        "vy":             vely,
        "vz":             velz,
        "vmag":           vmag,
        "log_B":          np.log10(np.clip(Bmag,  eps, None)),
        "mach":           mach,
        "log_beta":       np.log10(np.clip(beta,  eps, None)),
        "checkpoint_idx": np.full(n, ckpt_idx, dtype=np.float64),
    }
    y = np.log10(np.clip(si, eps, None))

    return X, y

def load_all_checkpoints(data_dir: str) -> tuple:
    """
    Load all checkpoint files, return (X_array, y_array, ckpt_idx_per_cell).
    Files are sorted by name so FLASH time-ordering is preserved.
    """
    for pat in ["*_hdf5_chk_*", "*.chk", "*plt_cnt*"]:
        files = sorted(glob.glob(os.path.join(data_dir, pat)))
        if files:
            break
    if not files:
        sys.exit(f"No checkpoint files found in {data_dir!r}. "
                 f"Check DATA_DIR and GLOB_PAT in the config block.")

    print(f"Found {len(files)} checkpoint files.")

    X_parts, y_parts, idx_parts = [], [], []
    for i, fpath in enumerate(files):
        print(f"  [{i+1}/{len(files)}] loading {os.path.basename(fpath)} ...", end=" ", flush=True)
        X_dict, y = load_checkpoint(fpath, i)
        n = len(y)
        X_row = np.stack([X_dict[c] for c in FEATURE_COLS], axis=1)
        if CELLS_PER_CHK and n > CELLS_PER_CHK:
            rng = np.random.default_rng(RANDOM_SEED + i)
            keep = rng.choice(n, size=CELLS_PER_CHK, replace=False)
            X_row, y = X_row[keep], y[keep]
            n = CELLS_PER_CHK
        X_parts.append(X_row)
        y_parts.append(y)
        idx_parts.append(np.full(n, i, dtype=np.int32))
        print(f"{n:,} cells")

    X_all   = np.concatenate(X_parts, axis=0)
    y_all   = np.concatenate(y_parts, axis=0)
    idx_all = np.concatenate(idx_parts, axis=0)
    n_files = len(files)
    return X_all, y_all, idx_all, n_files, files


# ── Train / test split ────────────────────────────────────────────────────────

def temporal_split(X, y, idx_all, n_files, files, test_frac=TEST_FRAC):
    """
    Split cells by checkpoint index.  All cells from the last `test_frac`
    fraction of files become the test set.  No cell from a test timestep
    appears in training data.
    """
    n_test = max(1, round(test_frac * n_files))
    if TEMPORAL_SPLIT_MODE == "interpolate":
        start = (n_files - n_test) // 2
        test_idx = set(range(start, start + n_test))
    else:  # 'extrapolate' — original behavior
        test_idx = set(range(n_files - n_test, n_files))
    test_mask  = np.isin(idx_all, sorted(test_idx))
    train_mask = ~test_mask

    print(f"\nTemporal split ({TEMPORAL_SPLIT_MODE}): "
          f"test files {sorted(test_idx)}")
    print(f"  Train files : {[os.path.basename(f) for i, f in enumerate(files) if i not in test_idx]}")
    print(f"  Test  files : {[os.path.basename(f) for i, f in enumerate(files) if i in test_idx]}")
    print(f"  Train cells : {train_mask.sum():,}")
    print(f"  Test  cells : {test_mask.sum():,}\n")

    return (X[train_mask], X[test_mask],
            y[train_mask], y[test_mask])

# ── Evaluation helper ─────────────────────────────────────────────────────────

def _metrics(name, y_true, y_pred_train, y_pred_test, y_train, label_train="train", label_test="test"):
    r2_tr  = r2_score(y_train,  y_pred_train)
    r2_te  = r2_score(y_true,   y_pred_test)
    rmse   = math.sqrt(mean_squared_error(y_true, y_pred_test))
    mae    = mean_absolute_error(y_true, y_pred_test)
    lines = [
        f"\n{'='*50}",
        f"Model: {name}",
        f"  R²  ({label_train})  : {r2_tr:.4f}",
        f"  R²  ({label_test})   : {r2_te:.4f}",
        f"  RMSE ({label_test})  : {rmse:.4f}",
        f"  MAE  ({label_test})  : {mae:.4f}",
    ]
    return lines


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test):
    results_lines = [
        "MHD Simulation ML Results",
        f"Train cells: {len(y_train):,}  |  Test cells: {len(y_test):,}",
        f"Features: {FEATURE_COLS}",
        f"Target: {TARGET_COL}  (log10 of silicon abundance)",
    ]

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Ridge ─────────────────────────────────────────────────────────────────
    print("Training Ridge (RidgeCV for alpha selection)...")
    alphas = np.logspace(-3, 4, 30)
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  RidgeCV(alphas=alphas, cv=5)),
    ])
    ridge_pipe.fit(X_train, y_train)
    best_alpha = ridge_pipe.named_steps["ridge"].alpha_
    print(f"  Best alpha: {best_alpha:.4g}")

    y_pred_ridge_train = ridge_pipe.predict(X_train)
    y_pred_ridge_test  = ridge_pipe.predict(X_test)

    lines = _metrics("Ridge Regression", y_test, y_pred_ridge_train,
                     y_pred_ridge_test, y_train)
    lines.append(f"  Best alpha     : {best_alpha:.4g}")

    # Feature coefficients (after scaling, so they're comparable)
    coefs = ridge_pipe.named_steps["ridge"].coef_
    lines.append("  Feature coefficients (scaled):")
    for feat, coef in sorted(zip(FEATURE_COLS, coefs), key=lambda x: abs(x[1]), reverse=True):
        lines.append(f"    {feat:<18s}: {coef:+.4f}")

    for l in lines:
        print(l)
    results_lines.extend(lines)

    joblib.dump(ridge_pipe, os.path.join(MODEL_DIR, "ridge_pipeline.pkl"))
    print(f"  Saved → {MODEL_DIR}/ridge_pipeline.pkl")

    # ── MLP ───────────────────────────────────────────────────────────────────
    print("\nTraining Neural Network (MLP)...")
    mlp_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_SEED,
            verbose=True,
        )),
    ])
    mlp_pipe.fit(X_train, y_train)

    mlp_model = mlp_pipe.named_steps["mlp"]
    print(f"  Converged in {mlp_model.n_iter_} iterations")

    y_pred_mlp_train = mlp_pipe.predict(X_train)
    y_pred_mlp_test  = mlp_pipe.predict(X_test)

    lines = _metrics("Neural Network (MLP)", y_test, y_pred_mlp_train,
                     y_pred_mlp_test, y_train)
    lines.append(f"  Iterations     : {mlp_model.n_iter_}")
    lines.append(f"  Final loss     : {mlp_model.loss_:.6f}")

    for l in lines:
        print(l)
    results_lines.extend(lines)

    joblib.dump(mlp_pipe, os.path.join(MODEL_DIR, "mlp_pipeline.pkl"))
    print(f"  Saved → {MODEL_DIR}/mlp_pipeline.pkl")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(RESULTS_TXT, "w") as fh:
        fh.write("\n".join(results_lines) + "\n")
    print(f"\nResults written to {RESULTS_TXT}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Data directory : {DATA_DIR}")
    print(f"Test fraction  : {TEST_FRAC} (last {int(TEST_FRAC*100)}% of files)")
    print(f"Random seed    : {RANDOM_SEED}\n")

    X_all, y_all, idx_all, n_files, files = load_all_checkpoints(DATA_DIR)
    X_train, X_test, y_train, y_test = temporal_split(
        X_all, y_all, idx_all, n_files, files
    )
    train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
