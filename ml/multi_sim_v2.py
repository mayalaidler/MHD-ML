#!/usr/bin/env python3
"""
Multi-simulation LOSO training, v2 — fixes the cell-pairing bug in
new_multi_sim.py and extends it to multiple Si ionization states.

What was wrong in v1 (new_multi_sim.py):
    build_dataset() drew features from the *equilibrium* checkpoint and Si
    values from checkpoint 12 using two independent random subsamples (the
    same rng object advanced between the two load_snapshot calls, and each
    checkpoint has its own validity mask).  Row i's features and row i's Si
    therefore came from two UNRELATED cells.  The only learnable signal was
    the per-simulation meta features, which is why LOSO R² was ~ -40.

What v2 does instead:
    1. Features and ion targets are read from the SAME cells of the SAME
       checkpoint (the local-conditions -> local-abundance task, matching the
       single-checkpoint models).
    2. Reads FLASH HDF5 directly with h5py (no yt index build — much faster).
    3. Extracts Si I ('si  '), Si II ('sip '), Si IV ('si3p') in one pass;
       one cached dataset serves all three ions.
    4. Uses two checkpoints per simulation (equilibrium + latest available)
       for temporal variety.
    5. Models per ion, LOSO over simulations: RidgeCV, HistGradientBoosting,
       and the same NN architecture as v1.

Usage:
    python3 multi_sim_v2.py build   # build/refresh the cached dataset
    python3 multi_sim_v2.py run     # LOSO for all ions (builds cache if absent)
    python3 multi_sim_v2.py run SiII  # LOSO for a single ion
"""

import json
import os
import sys
import time

import h5py
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR      = "/scratch/ebuie/ISO_Turb/midway/mhd_1e8/"
OUT_ROOT      = "multi_sim_v2"
DATASET_PATH  = os.path.join(OUT_ROOT, "paired_dataset.npz")
CELLS_PER_CHK = 25000          # random cells kept per checkpoint per sim
NN_MAX_TRAIN  = 100000         # NN per-fold training subsample
SEED          = 42

IONS = {"SiI": "si  ", "SiII": "sip ", "SiIV": "si3p"}

# Equilibrium checkpoint index per simulation (same as new_multi_sim.py)
equilibrium_files = {
    "1E23_S100_z01_mhd": 8,  "1E23_S100_z1_mhd": 7,
    "1E23_S30_z01_mhd":  5,  "1E23_S30_z1_mhd":  6,
    "1E23_S60_z01_mhd":  7,  "1E24_S100_z1_mhd": 8,
    "1E25_S100_z1_mhd":  5,  "1E25_S30_z01_mhd": 5,
    "1E25_S30_z1_mhd":   6,  "1E26_S100_z01_mhd": 9,
    "1E26_S100_z1_mhd":  9,  "1E26_S30_z01_mhd": 5,
    "1E26_S30_z1_mhd":   5,
}
SIM_NAMES = sorted(equilibrium_files.keys())


def parse_simulation_metadata(sim_name):
    parts       = sim_name.split("_")
    density     = float(parts[0].replace("E", "e"))
    turbulence  = float(parts[1].replace("S", ""))
    metallicity = 0.1 if parts[2] == "z01" else 1.0
    return density, turbulence, metallicity


def available_checkpoints(sim_path):
    idxs = []
    for f in os.listdir(sim_path):
        if "ISM_hdf5_chk_" in f and not f.endswith(".part"):
            try:
                idxs.append(int(f.split("_")[-1]))
            except ValueError:
                pass
    return sorted(idxs)


def read_checkpoint_paired(path, n_keep, seed):
    """Read features + all ion targets from the SAME cells of one checkpoint."""
    names = ["dens", "temp", "velx", "vely", "velz",
             "magx", "magy", "magz"] + list(IONS.values())
    data = {}
    with h5py.File(path, "r") as f:
        for name in names:
            data[name] = f[name][()].ravel().astype(np.float64)

    n = min(len(v) for v in data.values())
    for k in data:
        data[k] = data[k][:n]

    B = np.sqrt(data["magx"]**2 + data["magy"]**2 + data["magz"]**2)

    mask = (
        np.isfinite(data["dens"]) & np.isfinite(data["temp"])
        & np.isfinite(data["velx"]) & np.isfinite(B)
        & (data["dens"] > 0) & (data["temp"] > 0) & (B > 0)
    )
    for ion_field in IONS.values():
        mask &= np.isfinite(data[ion_field]) & (data[ion_field] > 0)

    idx_valid = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    if len(idx_valid) > n_keep:
        idx_valid = rng.choice(idx_valid, size=n_keep, replace=False)

    vmag = np.sqrt(data["velx"][idx_valid]**2
                   + data["vely"][idx_valid]**2
                   + data["velz"][idx_valid]**2)

    cols = {
        "rho":  data["dens"][idx_valid],
        "T":    data["temp"][idx_valid],
        "vmag": vmag,
        "B":    B[idx_valid],
    }
    ions = {tag: data[field][idx_valid] for tag, field in IONS.items()}
    return cols, ions


def build_dataset():
    print("[INFO] Building paired multi-sim dataset (v2)")
    t0 = time.time()
    rows_X, rows_ions, rows_sim, rows_chk = [], {t: [] for t in IONS}, [], []

    for sim_idx, sim_name in enumerate(SIM_NAMES):
        sim_path = os.path.join(ROOT_DIR, sim_name)
        if not os.path.isdir(sim_path):
            print(f"[WARN] ({sim_idx}) {sim_name}: directory missing — skipped")
            continue

        idxs = available_checkpoints(sim_path)
        eq   = equilibrium_files[sim_name]
        picks = [i for i in {eq, idxs[-1] if idxs else eq} if i in idxs]
        if not picks:
            print(f"[WARN] ({sim_idx}) {sim_name}: no usable checkpoints "
                  f"(have {idxs[:3]}...{idxs[-3:] if len(idxs) > 3 else ''}) — skipped")
            continue

        density, turb, Z = parse_simulation_metadata(sim_name)

        for chk_i in sorted(picks):
            path = os.path.join(sim_path, f"ISM_hdf5_chk_{chk_i:04d}")
            t1 = time.time()
            try:
                cols, ions = read_checkpoint_paired(
                    path, CELLS_PER_CHK, seed=SEED + 1000 * sim_idx + chk_i)
            except Exception as e:
                print(f"[WARN] ({sim_idx}) {sim_name} chk {chk_i}: {e} — skipped")
                continue

            n = len(cols["rho"])
            meta = np.column_stack([
                np.full(n, density), np.full(n, turb), np.full(n, Z)])
            rows_X.append(np.column_stack(
                [cols["rho"], cols["T"], cols["vmag"], cols["B"], meta]))
            for tag in IONS:
                rows_ions[tag].append(ions[tag])
            rows_sim.append(np.full(n, sim_idx, dtype=np.int32))
            rows_chk.append(np.full(n, chk_i, dtype=np.int32))
            print(f"[INFO] ({sim_idx}) {sim_name} chk {chk_i:04d}: "
                  f"{n} cells in {time.time()-t1:.0f}s")

    X = np.vstack(rows_X)
    out = {
        "X": X,
        "sim_ids": np.concatenate(rows_sim),
        "chk_ids": np.concatenate(rows_chk),
    }
    for tag in IONS:
        out[f"y_{tag}"] = np.concatenate(rows_ions[tag])

    os.makedirs(OUT_ROOT, exist_ok=True)
    np.savez_compressed(DATASET_PATH, **out)
    sims_included = sorted(set(out["sim_ids"].tolist()))
    print(f"[SUCCESS] {X.shape[0]} rows from sims {sims_included} "
          f"in {(time.time()-t0)/60:.1f} min -> {DATASET_PATH}")


def make_features(X):
    """Same 12 features as new_multi_sim._load_and_transform."""
    eps = 1e-30
    rho, T, vmag, B = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    density_meta, turb_meta, Z_meta = X[:, 4], X[:, 5], X[:, 6]

    log_rho   = np.log10(rho + eps)
    log_T     = np.log10(T + eps)
    log_B     = np.log10(B + eps)
    mach      = vmag / (np.sqrt(T) + eps)
    beta      = rho * T / (B**2 + eps)
    log_beta  = np.log10(np.clip(beta, eps, None))

    feats = np.column_stack([
        log_rho, log_T, vmag, log_B, mach, log_beta,
        log_rho * turb_meta,
        log_rho * Z_meta,
        log_T * Z_meta,
        np.log10(density_meta + eps),
        turb_meta,
        Z_meta,
    ])
    names = ["log_rho", "log_T", "vmag", "log_B_mag", "mach", "log_beta",
             "rho_turb_interaction", "log_rho_Z", "log_T_Z",
             "log_density_meta", "turb_meta", "Z_meta"]
    return feats, names


class SiPredictor(nn.Module):
    """Same architecture as new_multi_sim.NN."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.01),
            nn.Linear(256, 128),       nn.LeakyReLU(0.01),
            nn.Linear(128, 64),        nn.LeakyReLU(0.01),
            nn.Linear(64, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def train_nn_fold(X_tr, y_tr, X_te):
    if len(X_tr) > NN_MAX_TRAIN:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_tr), size=NN_MAX_TRAIN, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    Xtr = torch.from_numpy(X_tr)
    ytr = torch.from_numpy(y_tr).view(-1, 1)
    Xte = torch.from_numpy(X_te)

    model = SiPredictor(Xtr.shape[1]).double()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-6)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=1024, shuffle=True)

    best_loss, patience, max_patience = float("inf"), 0, 15
    for epoch in range(100):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        scheduler.step(avg)
        if avg < best_loss:
            best_loss, patience = avg, 0
        else:
            patience += 1
        if patience >= max_patience:
            break

    model.eval()
    with torch.no_grad():
        return model(Xte).numpy().flatten()


def run_ion(tag, X_feats, y_log, sim_ids):
    """LOSO CV for one ion with Ridge, HGB, and NN."""
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    alphas = np.logspace(-3, 3, 20)
    unique = np.unique(sim_ids)

    results = {m: {} for m in ("ridge", "hgb", "nn")}
    preds   = {m: {"y_true": [], "y_pred": [], "sim_ids": []}
               for m in ("ridge", "hgb", "nn")}

    print(f"\n{'='*70}\n[ION {tag}] target log10 range "
          f"[{y_log.min():.2f}, {y_log.max():.2f}], std {y_log.std():.2f}, "
          f"{len(y_log)} rows, LOSO over {len(unique)} sims\n{'='*70}")

    for held_out in unique:
        tr, te = sim_ids != held_out, sim_ids == held_out
        X_tr, X_te = X_feats[tr], X_feats[te]
        y_tr, y_te = y_log[tr], y_log[te]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        fold_preds = {}
        ridge = RidgeCV(alphas=alphas, cv=5, scoring="r2")
        ridge.fit(X_tr_s, y_tr)
        fold_preds["ridge"] = ridge.predict(X_te_s)

        hgb = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, max_depth=None,
            early_stopping=True, validation_fraction=0.1, random_state=SEED)
        hgb.fit(X_tr_s, y_tr)
        fold_preds["hgb"] = hgb.predict(X_te_s)

        fold_preds["nn"] = train_nn_fold(X_tr_s, y_tr, X_te_s)

        line = f"  sim {held_out:2d} ({SIM_NAMES[held_out]:>18s})"
        for m in ("ridge", "hgb", "nn"):
            r2 = r2_score(y_te, fold_preds[m])
            results[m][int(held_out)] = r2
            preds[m]["y_true"].append(y_te)
            preds[m]["y_pred"].append(fold_preds[m])
            preds[m]["sim_ids"].append(
                np.full(te.sum(), held_out, dtype=np.int32))
            line += f"  {m} R²={r2:7.3f}"
        print(line, flush=True)

    summary = {}
    for m in ("ridge", "hgb", "nn"):
        vals = list(results[m].values())
        summary[m] = {
            "mean_r2": float(np.mean(vals)),
            "median_r2": float(np.median(vals)),
            "min_r2": float(np.min(vals)),
            "max_r2": float(np.max(vals)),
            "per_sim": results[m],
        }
        np.savez(
            os.path.join(out_dir, f"{m}_loso_preds.npz"),
            y_true=np.concatenate(preds[m]["y_true"]),
            y_pred=np.concatenate(preds[m]["y_pred"]),
            sim_ids=np.concatenate(preds[m]["sim_ids"]),
        )
        print(f"[RESULT {tag}] {m:5s} LOSO mean R² = {summary[m]['mean_r2']:.4f} "
              f"(median {summary[m]['median_r2']:.4f}, "
              f"min {summary[m]['min_r2']:.4f}, max {summary[m]['max_r2']:.4f})")

    # Final models trained on everything (for downstream use)
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X_feats)
    ridge_full = RidgeCV(alphas=alphas, cv=5, scoring="r2").fit(X_full, y_log)
    hgb_full = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, early_stopping=True,
        validation_fraction=0.1, random_state=SEED).fit(X_full, y_log)
    joblib.dump(ridge_full, os.path.join(out_dir, "ridge_final.joblib"))
    joblib.dump(hgb_full, os.path.join(out_dir, "hgb_final.joblib"))
    joblib.dump(scaler_full, os.path.join(out_dir, "scaler_final.joblib"))

    with open(os.path.join(out_dir, "loso_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def run(ion_filter=None):
    if not os.path.exists(DATASET_PATH):
        build_dataset()

    with np.load(DATASET_PATH) as d:
        X_raw   = d["X"]
        sim_ids = d["sim_ids"]
        ys      = {tag: d[f"y_{tag}"] for tag in IONS}

    X_feats, _ = make_features(X_raw)
    assert np.isfinite(X_feats).all()

    all_summaries = {}
    for tag in IONS:
        if ion_filter and tag != ion_filter:
            continue
        y_log = np.log10(np.clip(ys[tag], 1e-35, None))
        all_summaries[tag] = run_ion(tag, X_feats, y_log, sim_ids)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY (LOSO mean R², cell-paired v2 dataset)")
    print("=" * 70)
    for tag, s in all_summaries.items():
        print(f"  {tag:5s}: " + "  ".join(
            f"{m}={s[m]['mean_r2']:.4f}" for m in s))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "build":
        build_dataset()
    elif mode == "run":
        run(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(f"Unknown mode {mode!r} — use 'build' or 'run'")
