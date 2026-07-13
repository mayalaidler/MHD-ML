# Fixed Ridge training — stricter spatial hold-out, si field only
import os
import numpy as np
import joblib
import yt
import re
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

USE_POLYNOMIAL_FEATURES = True  # Set True for polynomial degree-2 expansion
POLY_DEGREE = 2

# Target ion: 'si  '=Si I (neutral), 'sip '=Si II, 'si2p'=Si III, 'si3p'=Si IV
# Override with e.g.:  ION_FIELD='sip ' python3 si_model.py
ION_FIELD = os.environ.get("ION_FIELD", "si  ")
ION_TAG   = {"si  ": "SiI", "sip ": "SiII", "si2p": "SiIII", "si3p": "SiIV"}.get(
    ION_FIELD, ION_FIELD.strip())

# Spatial split strategy — choose one:
#   'percentile' — train on x < 25th, test on x > 75th (largest gap, least leakage)
#   'median'     — train on x < 50th, test on x >= 50th (original, more leakage)
#   'thirds'     — train on x < 33rd, test on x > 67th (middle ground)
SPATIAL_SPLIT_STRATEGY = 'median'

# ══════════════════════════════════════════════════════════════════════════════

def get_chk_number(path):
    match = re.search(r"chk_(\d+)", path)
    return int(match.group(1)) if match else -1

def find_field(ds, substrings):
    for field in ds.field_list:
        fname = f"{field[0]}:{field[1]}".lower()
        if all(s.lower() in fname for s in substrings):
            return field
    return None

def quantity_to_numpy(q):
    try:
        return q.to_value() if hasattr(q, "to_value") else q.value
    except Exception:
        return np.array(q)

def build_dataset_with_coords(ds):
    """Build dataset using ONLY the 'si  ' field (2 spaces)."""
    rho_field  = find_field(ds, ('flash', 'dens'))
    temp_field = find_field(ds, ('flash', 'temp'))
    vx_field   = find_field(ds, ('flash', 'velx'))
    vy_field   = find_field(ds, ('flash', 'vely'))
    vz_field   = find_field(ds, ('flash', 'velz'))
    si_field   = find_field(ds, ('flash', ION_FIELD))

    if si_field is None:
        raise RuntimeError(f"Could not find ('flash', {ION_FIELD!r}) field in dataset")

    # Magnetic field
    if find_field(ds, ("mag_strength",)):
        B_field = find_field(ds, ('flash', 'mag_strength'))
        use_mag_components = False
    else:
        Bx_field = find_field(ds, ('flash', 'magx'))
        By_field = find_field(ds, ('flash', 'magy'))
        Bz_field = find_field(ds, ('flash', 'magz'))
        use_mag_components = True

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

    si    = np.ravel(quantity_to_numpy(ad[si_field]))
    x_pos = np.ravel(quantity_to_numpy(ad[('index', 'x')]))

    # Align lengths
    npts = min(len(rho), len(T), len(vx), len(vy), len(vz), len(B), len(si), len(x_pos))
    rho, T, vx, vy, vz, B, si, x_pos = (
        arr[:npts] for arr in (rho, T, vx, vy, vz, B, si, x_pos)
    )

    # Valid data mask
    mask = (
        np.isfinite(rho) & np.isfinite(T) & np.isfinite(vx) & 
        np.isfinite(B) & np.isfinite(si) & np.isfinite(x_pos) & 
        (si > 0) & (rho > 0) & (T > 0) & (B > 0)
    )

    rho, T, vx, vy, vz, B, si, x_pos = (
        arr[mask] for arr in (rho, T, vx, vy, vz, B, si, x_pos)
    )

    # Feature engineering
    eps = 1e-30
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    mach = vmag / (np.sqrt(T) + eps)
    p_th = rho * T
    p_mag = B**2
    plasma_beta = p_th / (p_mag + eps)

    log_rho  = np.log10(rho)
    log_T    = np.log10(T)
    log_B    = np.log10(B)
    log_beta = np.log10(np.clip(plasma_beta, eps, None))
    y = np.log10(si)

    X = np.column_stack([
        log_rho, log_T, vx, vy, vz, vmag, log_B, mach, log_beta
    ])

    # Subsample
    sample_limit = int(os.environ.get('DATASET_SAMPLE_SIZE', 100000))
    n_total = len(X)
    if n_total > sample_limit:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_total, size=sample_limit, replace=False)
        X, y, x_pos = X[idx], y[idx], x_pos[idx]
        print(f"[INFO] Subsampled dataset: {n_total} -> {len(X)} samples")
    else:
        print(f"[INFO] Built dataset: {len(X)} samples")

    return X, y, x_pos


def spatial_split(x_pos, strategy='percentile'):
    """
    Split data by x-coordinate using specified strategy.
    
    Returns
    -------
    train_mask, test_mask : boolean arrays
    gap_info : dict with split statistics
    """
    if strategy == 'median':
        threshold = np.median(x_pos)
        train_mask = x_pos < threshold
        test_mask  = x_pos >= threshold
        gap = 0.0  # no gap
        
    elif strategy == 'percentile':
        # Train on bottom 25%, test on top 25%, discard middle 50%
        p25 = np.percentile(x_pos, 25)
        p75 = np.percentile(x_pos, 75)
        train_mask = x_pos < p25
        test_mask  = x_pos > p75
        gap = p75 - p25
        
    elif strategy == 'thirds':
        # Train on bottom 33%, test on top 33%, discard middle
        p33 = np.percentile(x_pos, 33.33)
        p67 = np.percentile(x_pos, 66.67)
        train_mask = x_pos < p33
        test_mask  = x_pos > p67
        gap = p67 - p33
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    gap_info = {
        'strategy': strategy,
        'gap_size': gap,
        'n_train': train_mask.sum(),
        'n_test': test_mask.sum(),
        'n_discarded': len(x_pos) - train_mask.sum() - test_mask.sum(),
    }
    
    return train_mask, test_mask, gap_info


def run_ridge_pipeline(X_tr, y_tr, X_te, y_te):
    """Unified Ridge pipeline with optional polynomial expansion."""
    if USE_POLYNOMIAL_FEATURES:
        print(f"[INFO] Generating polynomial features (degree={POLY_DEGREE})...")
        poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
        X_tr_p = poly.fit_transform(X_tr)
        X_te_p = poly.transform(X_te)
        print(f"       Features: {X_tr.shape[1]} -> {X_tr_p.shape[1]}")
    else:
        poly = None
        X_tr_p = X_tr
        X_te_p = X_te
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_p)
    X_te_s = scaler.transform(X_te_p)
    
    alphas = np.logspace(-5, 2, 20)
    ridge  = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_tr_s, y_tr)
    
    y_pred_te = ridge.predict(X_te_s)
    y_pred_tr = ridge.predict(X_tr_s)
    
    return ridge, scaler, poly, y_pred_tr, y_pred_te


def train_single_checkpoint_spatial(base_dir, out_dir):
    """Train Ridge on single checkpoint with spatial hold-out."""
    os.makedirs(out_dir, exist_ok=True)
    chk_files = sorted([
        os.path.join(base_dir, f) 
        for f in os.listdir(base_dir) 
        if "ISM_hdf5_chk_" in f
    ])
    chk_files = [f for f in chk_files if get_chk_number(f) >= 4]

    target_path = chk_files[2] 
    print(f"[INFO] Loading checkpoint: {target_path}")
    ds = yt.load(target_path)

    X, y, x_pos = build_dataset_with_coords(ds)

    # Apply spatial split
    train_mask, test_mask, gap_info = spatial_split(x_pos, strategy=SPATIAL_SPLIT_STRATEGY)
    
    print(f"\n[INFO] Spatial split strategy: {gap_info['strategy']}")
    print(f"       Gap between train/test: {gap_info['gap_size']:.3f}")
    print(f"       Train samples: {gap_info['n_train']}")
    print(f"       Test samples:  {gap_info['n_test']}")
    if gap_info['n_discarded'] > 0:
        print(f"       Discarded (middle): {gap_info['n_discarded']}")

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    model, scaler, poly, _, y_pred = run_ridge_pipeline(
        X_train, y_train, X_test, y_test
    )

    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    
    print(f"\n[INFO] Best alpha: {model.alpha_:.2e}")
    print(f"[RESULT] Single-checkpoint (spatial hold-out)")
    print(f"         Polynomial features: {USE_POLYNOMIAL_FEATURES}")
    print(f"         MSE: {mse:.6e}")
    print(f"         R²:  {r2:.4f}")

    # Save
    joblib.dump(model,  os.path.join(out_dir, "ridge_single_spatial.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler_single_spatial.joblib"))
    if poly is not None:
        joblib.dump(poly, os.path.join(out_dir, "poly_single_spatial.joblib"))
    
    np.savez(
        os.path.join(out_dir, "single_spatial_test.npz"),
        y_true=y_test,
        y_pred=y_pred,
        X=X_test
    )
    
    # Save split info for reproducibility
    with open(os.path.join(out_dir, "split_info.txt"), "w") as f:
        f.write(f"Strategy: {gap_info['strategy']}\n")
        f.write(f"Gap size: {gap_info['gap_size']:.3f}\n")
        f.write(f"Train samples: {gap_info['n_train']}\n")
        f.write(f"Test samples: {gap_info['n_test']}\n")
        f.write(f"Discarded: {gap_info['n_discarded']}\n")
        f.write(f"R²: {r2:.4f}\n")
        f.write(f"RMSE: {np.sqrt(mse):.4f} dex\n")
    
    print(f"[SUCCESS] Saved to {out_dir}/")


def train_multi_checkpoint(base_dir, out_dir):
    """Train Ridge on early checkpoint, test on late checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    chk_files = sorted([
        os.path.join(base_dir, f) 
        for f in os.listdir(base_dir) 
        if "ISM_hdf5_chk_" in f
    ])
    chk_files = [f for f in chk_files if get_chk_number(f) >= 4]

    early_path = chk_files[0]  
    late_path  = chk_files[-1]  

    print(f"[INFO] Train checkpoint: {early_path}")
    print(f"[INFO] Test  checkpoint: {late_path}")

    ds_train = yt.load(early_path)
    ds_test  = yt.load(late_path)

    X_train, y_train, _ = build_dataset_with_coords(ds_train)
    X_test,  y_test,  _ = build_dataset_with_coords(ds_test)

    print(f"[INFO] Train: {len(X_train)} samples")
    print(f"[INFO] Test:  {len(X_test)} samples")

    model, scaler, poly, y_pred_train, y_pred_test = run_ridge_pipeline(
        X_train, y_train, X_test, y_test
    )

    mse = mean_squared_error(y_test, y_pred_test)
    r2  = r2_score(y_test, y_pred_test)
    
    print(f"\n[INFO] Best alpha: {model.alpha_:.2e}")
    print(f"[RESULT] Multi-checkpoint (temporal generalization)")
    print(f"         Polynomial features: {USE_POLYNOMIAL_FEATURES}")
    print(f"         MSE: {mse:.6e}")
    print(f"         R²:  {r2:.4f}")

    joblib.dump(model,  os.path.join(out_dir, "ridge_multi_checkpoint.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler_multi_checkpoint.joblib"))
    if poly is not None:
        joblib.dump(poly, os.path.join(out_dir, "poly_multi_checkpoint.joblib"))
    
    np.savez(
        os.path.join(out_dir, "multi_checkpoint_test.npz"),
        X=X_test,
        y_true=y_test,
        y_pred=y_pred_test
    )
    np.savez(
        os.path.join(out_dir, "multi_checkpoint_train.npz"),
        X=X_train,
        y_true=y_train,
        y_pred=y_pred_train
    )
    print(f"[SUCCESS] Saved to {out_dir}/")


if __name__ == "__main__":
    BASE_DIR = "/scratch/ebuie/ISO_Turb/midway/mhd_1e8/1E23_S100_z1_mhd/"
    OUT_DIR  = os.environ.get("OUT_DIR_OVERRIDE") or (
        "siresults_ridge_fixed" if ION_FIELD == "si  "
        else f"results_{ION_TAG}_ridge")

    print("=" * 70)
    print(f"CONFIG: Ion = {ION_TAG} (field {ION_FIELD!r})")
    print(f"CONFIG: Polynomial features = {USE_POLYNOMIAL_FEATURES}")
    print(f"        Spatial split = {SPATIAL_SPLIT_STRATEGY}")
    print("=" * 70)
    
    print("\nMODE 1 — Single-checkpoint with spatial hold-out")
    print("=" * 70)
    train_single_checkpoint_spatial(BASE_DIR, OUT_DIR)

    # print("\n" + "=" * 70)
    # print("MODE 2 — Multi-checkpoint temporal generalization")
    # print("=" * 70)
    # train_multi_checkpoint(BASE_DIR, OUT_DIR)