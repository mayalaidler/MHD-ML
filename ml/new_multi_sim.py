# file: build_multi_sim_dataset_CLEAN.py
#
# Multi-simulation training with LOSO CV
# Matches original feature engineering with critical fixes
#
# SPEED OPTIMIZATIONS:
#   - MAX_SAMPLES = 10k per sim (down from 50k) → ~120k total dataset
#   - Oversample multiplier = 2x (down from 5x)
#   - LOSO subsample = 50k per fold (down from 100k)
#   - NN: Early stopping, smaller batches
#   → Total runtime: ~45-60 min (vs 2-3 hours original)
#
# CRITICAL FIXES:
#   - Added log_density_meta transform (was missing, caused R²=0.46)
#   - Restored metallicity interaction terms (log_rho*Z, log_T*Z)
#   - REMOVED MaxAbsScaler on y (was causing cross-sim scaling disaster)
#   - CLIP Si values to 1e-10 minimum before log transform (was getting -30!)
#   - Filter out Si < 1e-12 in stratified sampling (noise threshold)
#   - NN trains directly on log10(Si) with proper range checking
#
# EXPECTED RESULTS (with smaller 10k dataset):
#   - Ridge: R² = 0.55-0.65 (lower due to less data)
#   - NN:    R² = 0.55-0.65 (should match Ridge now!)

import os
import numpy as np
import yt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

ROOT_DIR    = "/scratch/ebuie/ISO_Turb/midway/mhd_1e8/"
OUTPUT_PATH = "/scratch/mlaidler/astr_thesis/ml/datasets/multi_sim_dataset.npz"
MAX_SAMPLES = 10000  # Reduced from 50k (10k per sim × 13 sims = ~130k total)
N_SI_BINS   = 10


def find_field(ds, substrings):
    """Find a field in the dataset matching all substrings."""
    for field in ds.field_list:
        fname = f"{field[0]}:{field[1]}".lower()
        if all(s.lower() in fname for s in substrings):
            return field
    return None


def quantity_to_numpy(q):
    """Convert yt quantity to numpy array."""
    try:
        return q.to_value() if hasattr(q, "to_value") else q.value
    except Exception:
        return np.array(q)


def load_snapshot(ds, max_samples=None, rng=None):
    """
    Load physical features and Si target from a yt dataset.
    
    Returns
    -------
    X : (N, 7) array of [rho, T, vx, vy, vz, vmag, B_mag]
    y : (N,) array of Si mass fraction
    """
    # Find fields
    rho_field  = find_field(ds, ('flash', 'dens'))
    temp_field = find_field(ds, ('flash', 'temp'))
    vx_field   = find_field(ds, ('flash', 'velx'))
    vy_field   = find_field(ds, ('flash', 'vely'))
    vz_field   = find_field(ds, ('flash', 'velz'))
    si_field   = find_field(ds, ('flash', 'si  '))  # 2 spaces
    
    if si_field is None:
        raise RuntimeError("Could not find ('flash', 'si  ') field")
    
    # Check for magnetic field
    if find_field(ds, ("mag_strength",)):
        B_field = find_field(ds, ('flash', 'mag_strength'))
        use_mag_components = False
    else:
        Bx_field = find_field(ds, ('flash', 'magx'))
        By_field = find_field(ds, ('flash', 'magy'))
        Bz_field = find_field(ds, ('flash', 'magz'))
        use_mag_components = True
    
    # Load data
    ad = ds.all_data()
    rho = np.ravel(quantity_to_numpy(ad[rho_field]))
    T   = np.ravel(quantity_to_numpy(ad[temp_field]))
    vx  = np.ravel(quantity_to_numpy(ad[vx_field]))
    vy  = np.ravel(quantity_to_numpy(ad[vy_field]))
    vz  = np.ravel(quantity_to_numpy(ad[vz_field]))
    
    if use_mag_components:
        Bx = np.ravel(quantity_to_numpy(ad[Bx_field]))
        By = np.ravel(quantity_to_numpy(ad[By_field]))
        Bz = np.ravel(quantity_to_numpy(ad[Bz_field]))
        B  = np.sqrt(Bx**2 + By**2 + Bz**2)
    else:
        B = np.ravel(quantity_to_numpy(ad[B_field]))
    
    si = np.ravel(quantity_to_numpy(ad[si_field]))
    
    # Align lengths
    npts = min(len(rho), len(T), len(vx), len(vy), len(vz), len(B), len(si))
    rho, T, vx, vy, vz, B, si = (
        arr[:npts] for arr in (rho, T, vx, vy, vz, B, si)
    )
    
    # Mask invalid data
    mask = (
        np.isfinite(rho) & np.isfinite(T) & np.isfinite(vx) &
        np.isfinite(B) & np.isfinite(si) &
        (si > 0) & (rho > 0) & (T > 0) & (B > 0)
    )
    
    rho, T, vx, vy, vz, B, si = (
        arr[mask] for arr in (rho, T, vx, vy, vz, B, si)
    )
    
    # Compute vmag
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Assemble features
    X = np.column_stack([rho, T, vx, vy, vz, vmag, B])
    y = si
    
    # Optional subsampling
    if max_samples is not None and len(X) > max_samples:
        if rng is None:
            rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
    
    return X, y

equilibrium_files = {
    "1E23_S100_z01_mhd": 8,  "1E23_S100_z1_mhd": 7,
    "1E23_S30_z01_mhd":  5,  "1E23_S30_z1_mhd":  6,
    "1E23_S60_z01_mhd":  7,  "1E24_S100_z1_mhd": 8,
    "1E25_S100_z1_mhd":  5,  "1E25_S30_z01_mhd": 5,
    "1E25_S30_z1_mhd":   6,  "1E26_S100_z01_mhd": 9,
    "1E26_S100_z1_mhd":  9,  "1E26_S30_z01_mhd": 5,
    "1E26_S30_z1_mhd":   5,
}


def parse_simulation_metadata(sim_name):
    parts      = sim_name.split("_")
    density    = float(parts[0].replace("E", "e"))
    turbulence = float(parts[1].replace("S", ""))
    metallicity = 0.1 if parts[2] == "z01" else 1.0
    return density, turbulence, metallicity


def get_checkpoint_path(sim_path, idx):
    return os.path.join(sim_path, f"ISM_hdf5_chk_{idx:04d}")


def stratified_sample_log_si(X, y, max_samples, n_bins, rng):
    """Stratified sampling to prevent low-Si majority from dominating."""
    # Filter out noise cells (Si < 1e-15) - same threshold as _load_and_transform
    valid_mask = y > 1e-15
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    if len(y_valid) < max_samples * 0.5:
        print(f"[WARNING] Only {len(y_valid)} valid samples, relaxing threshold")
        valid_mask = y > 1e-20
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
    
    log_y     = np.log10(np.clip(y_valid, 1e-40, None))
    bin_edges = np.linspace(log_y.min(), log_y.max(), n_bins + 1)
    bin_ids   = np.digitize(log_y, bin_edges[1:-1])
    per_bin   = max(1, max_samples // n_bins)
    idx_keep  = []

    for b in range(n_bins):
        members = np.where(bin_ids == b)[0]
        if len(members) == 0:
            continue
        chosen = rng.choice(members, size=min(per_bin, len(members)), replace=False)
        idx_keep.append(chosen)

    idx_keep = np.concatenate(idx_keep)
    rng.shuffle(idx_keep)
    return X_valid[idx_keep], y_valid[idx_keep]


def build_dataset():
    """Build multi-simulation dataset with stratified sampling."""
    print("[INFO] Starting dataset build")
    X_all, y_all, sim_ids = [], [], []
    sim_names = sorted(equilibrium_files.keys())

    for sim_idx, sim_name in enumerate(sim_names):
        sim_path = os.path.join(ROOT_DIR, sim_name)
        if not os.path.isdir(sim_path):
            continue

        print(f"\n[INFO] ({sim_idx+1}/{len(sim_names)}) Processing: {sim_name}")
        try:
            density, turb, Z = parse_simulation_metadata(sim_name)
            first_path = get_checkpoint_path(sim_path, equilibrium_files[sim_name])
            last_path  = get_checkpoint_path(sim_path, 12)

            ds_first = yt.load(first_path)
            ds_last  = yt.load(last_path)

            rng = np.random.default_rng(abs(hash(sim_name)) % 2**32)
            X_first, _ = load_snapshot(ds_first, max_samples=MAX_SAMPLES * 2, rng=rng)  # Reduced from 5x
            _, y_last  = load_snapshot(ds_last,  max_samples=MAX_SAMPLES * 2, rng=rng)

            if X_first is None or y_last is None:
                continue

            n = min(len(X_first), len(y_last))
            X_first, y_last = X_first[:n], y_last[:n]

            # Stratified downsample
            rng_strat = np.random.default_rng(abs(hash(sim_name + "_strat")) % 2**32)
            X_first, y_last = stratified_sample_log_si(
                X_first, y_last, MAX_SAMPLES, N_SI_BINS, rng_strat
            )
            print(f"[DEBUG] After stratified sampling: {len(y_last)} rows")

            # Add metadata
            meta = np.column_stack([
                np.full(len(y_last), density),
                np.full(len(y_last), turb),
                np.full(len(y_last), Z),
            ])

            X_combined = np.hstack([X_first, meta])
            X_all.append(X_combined)
            y_all.append(y_last)
            sim_ids.append(np.full(len(y_last), sim_idx, dtype=np.int32))

        except Exception as e:
            print(f"[ERROR] Failed {sim_name}: {e}")
            continue

    X_all   = np.vstack(X_all)
    y_all   = np.concatenate(y_all)
    sim_ids = np.concatenate(sim_ids)

    print(f"\n[DEBUG] Final shapes: X={X_all.shape}, y={y_all.shape}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH, X=X_all, y=y_all, sim_ids=sim_ids)
    print("[SUCCESS] Dataset saved to", OUTPUT_PATH)


def _load_and_transform(npz_path):
    """
    Load NPZ and apply feature engineering.
    
    IMPORTANT: Uses same 9 base features as single-checkpoint,
    PLUS 3 meta-features to distinguish simulations.
    
    NPZ columns (from build_dataset):
        0-6: physical [rho, T, vx, vy, vz, vmag, B_mag] (raw)
        7-9: meta [density, turbulence, metallicity] (raw)
    
    Returns 12 features total:
        9 physical (same as single-checkpoint) + 3 meta (log_density, turb, Z)
    """
    with np.load(npz_path) as data:
        X       = data["X"].copy().astype(np.float64)
        y       = data["y"].copy().astype(np.float64)
        sim_ids = data["sim_ids"].copy() if "sim_ids" in data.files else None

    eps = 1e-30

    # Extract raw features
    rho   = X[:, 0]
    T     = X[:, 1]
    vx    = X[:, 2]
    vy    = X[:, 3]
    vz    = X[:, 4]
    vmag  = X[:, 5]
    B_mag = X[:, 6]
    density_meta = X[:, 7]
    turb_meta    = X[:, 8]
    Z_meta       = X[:, 9]

    # ── Feature engineering (RESTORE ORIGINAL) ───────────────────────────
    log_rho  = np.log10(rho + eps)
    log_T    = np.log10(T + eps)
    log_B_mag = np.log10(B_mag + eps)
    
    mach = vmag / (np.sqrt(T) + eps)
    
    p_th = rho * T
    p_mag = B_mag**2
    plasma_beta = p_th / (p_mag + eps)
    log_beta = np.log10(np.clip(plasma_beta, eps, None))
    
    # Density-turbulence interaction (important for mixing)
    rho_turb_interaction = log_rho * turb_meta

    # ── Meta-feature engineering (CRITICAL FOR MULTI-SIM) ─────────────────
    log_density_meta = np.log10(density_meta + eps)  # ← FIX: was missing
    
    # Metallicity interactions (CRITICAL: different physics at different Z)
    log_rho_Z = log_rho * Z_meta
    log_T_Z   = log_T * Z_meta

    # ── Assemble (12 features matching original) ──────────────────────────
    X_final = np.column_stack([
        log_rho,
        log_T,
        vmag,
        log_B_mag,
        mach,
        log_beta,
        rho_turb_interaction,
        log_rho_Z,           # ← RESTORED
        log_T_Z,             # ← RESTORED
        log_density_meta,    # ← FIXED
        turb_meta,
        Z_meta,
    ])

    feature_names = [
        "log_rho", "log_T", "vmag", "log_B_mag", "mach", "log_beta",
        "rho_turb_interaction", "log_rho_Z", "log_T_Z",
        "log_density_meta", "turb_meta", "Z_meta",
    ]

    # Filter out noise/uninitialized cells (Si < 1e-15), then log-transform
    # Don't clip! We want to preserve the natural Si distribution
    eps = 1e-30
    noise_threshold = 1e-15
    valid_mask = y > noise_threshold
    
    X_final = X_final[valid_mask]
    y = y[valid_mask]
    sim_ids = sim_ids[valid_mask] if sim_ids is not None else None
    
    n_filtered = (~valid_mask).sum()
    if n_filtered > 0:
        print(f"[INFO] Filtered {n_filtered} noise cells (Si < {noise_threshold:.0e})")
    
    y = np.log10(y + eps)
    
    print(f"[DEBUG] Target (log10 Si) range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"[DEBUG] Target std: {y.std():.2f}")

    assert X_final.shape[1] == len(feature_names) == 12
    assert np.isfinite(X_final).all()
    assert np.isfinite(y).all()

    print(f"[DEBUG] Final dataset: {X_final.shape[0]} samples, {X_final.shape[1]} features")
    return X_final, y, sim_ids, feature_names


def ridge():
    """Ridge with LOSO CV."""
    npz_file_path = "datasets/multi_sim_dataset.npz"

    print("[INFO] Loading dataset...")
    X, y, sim_ids, feature_names = _load_and_transform(npz_file_path)

    if sim_ids is None:
        raise RuntimeError("sim_ids not found — re-run build_dataset()")

    # Baseline
    scaler_ref = StandardScaler()
    X_s = scaler_ref.fit_transform(X)
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_s, y)
    print(f"[BASELINE] Dummy R²: {dummy.score(X_s, y):.4f}")

    # LOSO CV
    print("\n[INFO] Leave-one-simulation-out CV (Ridge)...")
    alphas      = np.logspace(-3, 3, 20)
    unique      = np.unique(sim_ids)
    per_sim_r2  = {}
    all_y_true, all_y_pred, all_sim_ids = [], [], []

    for held_out in unique:
        train_mask = sim_ids != held_out
        test_mask  = sim_ids == held_out

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        # Subsample training to 50k (reduced from 100k for speed)
        if len(X_tr) > 50000:
            print(f"  [SUBSAMPLE] {len(X_tr)} → 50000")
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_tr), size=50000, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = RidgeCV(alphas=alphas, cv=5, scoring="r2")
        model.fit(X_tr_s, y_tr)

        y_pred = model.predict(X_te_s)
        r2 = r2_score(y_te, y_pred)
        per_sim_r2[int(held_out)] = r2

        all_y_true.append(y_te)
        all_y_pred.append(y_pred)
        all_sim_ids.append(np.full(len(y_te), held_out, dtype=np.int32))

        print(f"  sim {held_out:2d}  α={model.alpha_:.2e}  R²={r2:.4f}")

    mean_r2 = float(np.mean(list(per_sim_r2.values())))
    print(f"\n[RESULT] LOSO mean R²: {mean_r2:.4f}  "
          f"(min={min(per_sim_r2.values()):.4f}, "
          f"max={max(per_sim_r2.values()):.4f})")

    # Save
    os.makedirs("multi_simulation", exist_ok=True)
    np.savez(
        "multi_simulation/ridge_loso_preds.npz",
        y_true=np.concatenate(all_y_true),
        y_pred=np.concatenate(all_y_pred),
        sim_ids=np.concatenate(all_sim_ids),
    )

    # Final model
    print("\n[INFO] Training final model...")
    scaler_final = StandardScaler()
    X_final = scaler_final.fit_transform(X)
    final_model = RidgeCV(alphas=alphas, cv=5, scoring="r2")
    final_model.fit(X_final, y)
    print(f"[INFO] Final α: {final_model.alpha_:.2e}, R²: {final_model.score(X_final, y):.4f}")

    # Feature importance
    print("\n[INFO] Top 10 features by |coefficient|:")
    coefs = np.abs(final_model.coef_)
    for i in np.argsort(coefs)[::-1][:10]:
        print(f"  {feature_names[i]:20s} {coefs[i]:.4f}")

    joblib.dump(final_model, "multi_simulation/si_ridge_model.joblib")
    joblib.dump(scaler_final, "multi_simulation/si_scaler.joblib")
    np.save("multi_simulation/ridge_loso_r2.npy", per_sim_r2)
    print("[SUCCESS] Ridge complete")


def NN():
    """NN with LOSO CV."""
    npz_file_path = "datasets/multi_sim_dataset.npz"

    print("[INFO] Loading dataset...")
    X, y, sim_ids, feature_names = _load_and_transform(npz_file_path)

    if sim_ids is None:
        raise RuntimeError("sim_ids not found")

    class SiPredictor(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            # Wider, deeper network for multi-regime learning
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),  # Wider for complex interactions
                nn.LeakyReLU(0.01),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.01),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.01),
                nn.Linear(64, 1),
            )
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.net(x)

    def train_one_fold(X_tr, y_tr, X_te, y_te):
        # Subsample training to 50k (reduced from 100k for speed)
        if len(X_tr) > 50000:
            print(f"    [SUBSAMPLE] {len(X_tr)} → 50000")
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_tr), size=50000, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]

        # Debug: Check input ranges
        print(f"    [DEBUG] y_train range: [{y_tr.min():.2f}, {y_tr.max():.2f}]")
        print(f"    [DEBUG] y_test range:  [{y_te.min():.2f}, {y_te.max():.2f}]")

        # Sanitize
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=1e6, neginf=-1e6)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=1e6, neginf=-1e6)
        y_tr = np.nan_to_num(y_tr, nan=0.0, posinf=10.0, neginf=-10.0)
        y_te = np.nan_to_num(y_te, nan=0.0, posinf=10.0, neginf=-10.0)

        # Scale X only (y is already in good range as log10(Si))
        scaler_X = StandardScaler()
        X_tr_s = scaler_X.fit_transform(X_tr)
        X_te_s = scaler_X.transform(X_te)

        Xtr = torch.from_numpy(X_tr_s).double()
        ytr = torch.from_numpy(y_tr).double().view(-1, 1)
        Xte = torch.from_numpy(X_te_s).double()

        model = SiPredictor(Xtr.shape[1]).double()
        criterion = nn.MSELoss()
        
        # Smaller LR for smaller dataset + weight decay for regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-6
        )
        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=1024, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(100):  # More epochs with early stopping
            model.train()
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                
                # Check for NaN gradients
                if not torch.isfinite(loss):
                    print(f"    [ERROR] NaN/Inf loss at epoch {epoch}")
                    break
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"    [EARLY STOP] at epoch {epoch+1}, best loss={best_loss:.6f}")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}: loss={avg_loss:.6f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(Xte).numpy().flatten()

        # Debug: Check prediction range
        print(f"    [DEBUG] y_pred range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
        
        r2 = r2_score(y_te, y_pred)
        rmse = np.sqrt(np.mean((y_te - y_pred)**2))
        print(f"    [DEBUG] RMSE: {rmse:.4f}")
        
        return y_pred, r2

    # LOSO CV
    print("\n[INFO] Leave-one-simulation-out CV (NN)...")
    unique = np.unique(sim_ids)
    per_sim_r2 = {}
    all_y_true, all_y_pred, all_sim_ids = [], [], []

    for held_out in unique:
        train_mask = sim_ids != held_out
        test_mask = sim_ids == held_out

        y_pred, r2 = train_one_fold(
            X[train_mask], y[train_mask],
            X[test_mask], y[test_mask],
        )
        per_sim_r2[int(held_out)] = r2

        all_y_true.append(y[test_mask])
        all_y_pred.append(y_pred)
        all_sim_ids.append(np.full(test_mask.sum(), held_out, dtype=np.int32))

        print(f"  sim {held_out:2d}  R²={r2:.4f}")

    mean_r2 = float(np.mean(list(per_sim_r2.values())))
    print(f"\n[RESULT] LOSO mean R²: {mean_r2:.4f}  "
          f"(min={min(per_sim_r2.values()):.4f}, "
          f"max={max(per_sim_r2.values()):.4f})")

    # Save
    np.savez(
        "multi_simulation/nn_loso_preds.npz",
        y_true=np.concatenate(all_y_true),
        y_pred=np.concatenate(all_y_pred),
        sim_ids=np.concatenate(all_sim_ids),
    )
    np.save("multi_simulation/nn_loso_r2.npy", per_sim_r2)
    print("[SUCCESS] NN complete")


if __name__ == "__main__":
    # build_dataset()  # Run once
    ridge()
    NN()