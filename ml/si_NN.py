# Matching NN training — same features and spatial split as Ridge
import os
import numpy as np
import yt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader
import re

# Import the same helper functions from Ridge script
import sys
sys.path.insert(0, os.path.dirname(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# If Ridge uses polynomial features, NN needs more capacity to compete
# Set WIDE_NETWORK=True to match polynomial Ridge (128→64→32 instead of 64→32)
WIDE_NETWORK = True

# Match Ridge's spatial split strategy
SPATIAL_SPLIT_STRATEGY = 'median'  # 'percentile', 'median', or 'thirds'

# Target ion: 'si  '=Si I (neutral), 'sip '=Si II, 'si2p'=Si III, 'si3p'=Si IV
# Override with e.g.:  ION_FIELD='sip ' python3 si_NN.py
ION_FIELD = os.environ.get("ION_FIELD", "si  ")
ION_TAG   = {"si  ": "SiI", "sip ": "SiII", "si2p": "SiIII", "si3p": "SiIV"}.get(
    ION_FIELD, ION_FIELD.strip())

# Reproducibility: fix torch's RNG (weight init + batch shuffling).
TORCH_SEED = int(os.environ.get("TORCH_SEED", "42"))
torch.manual_seed(TORCH_SEED)

# Training subsample cap for the NN (test set is never subsampled here).
NN_MAX_TRAIN = int(os.environ.get("NN_MAX_TRAIN", "100000"))

# ══════════════════════════════════════════════════════════════════════════════

# Copy helper functions from Ridge script
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
    """Build dataset using ONLY the 'si  ' field — IDENTICAL to Ridge version."""
    rho_field  = find_field(ds, ('flash', 'dens'))
    temp_field = find_field(ds, ('flash', 'temp'))
    vx_field   = find_field(ds, ('flash', 'velx'))
    vy_field   = find_field(ds, ('flash', 'vely'))
    vz_field   = find_field(ds, ('flash', 'velz'))
    si_field   = find_field(ds, ('flash', ION_FIELD))

    if si_field is None:
        raise RuntimeError(f"Could not find ('flash', {ION_FIELD!r}) field")

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

    npts = min(len(rho), len(T), len(vx), len(vy), len(vz), len(B), len(si), len(x_pos))
    rho, T, vx, vy, vz, B, si, x_pos = (
        arr[:npts] for arr in (rho, T, vx, vy, vz, B, si, x_pos)
    )

    mask = (
        np.isfinite(rho) & np.isfinite(T) & np.isfinite(vx) & 
        np.isfinite(B) & np.isfinite(si) & np.isfinite(x_pos) & 
        (si > 0) & (rho > 0) & (T > 0) & (B > 0)
    )
    rho, T, vx, vy, vz, B, si, x_pos = (
        arr[mask] for arr in (rho, T, vx, vy, vz, B, si, x_pos)
    )

    # IDENTICAL feature engineering to Ridge
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

    sample_limit = int(os.environ.get('DATASET_SAMPLE_SIZE', 500000))
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
    """IDENTICAL spatial split logic to Ridge."""
    if strategy == 'median':
        threshold = np.median(x_pos)
        train_mask = x_pos < threshold
        test_mask  = x_pos >= threshold
        gap = 0.0
    elif strategy == 'percentile':
        p25 = np.percentile(x_pos, 25)
        p75 = np.percentile(x_pos, 75)
        train_mask = x_pos < p25
        test_mask  = x_pos > p75
        gap = p75 - p25
    elif strategy == 'thirds':
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


class SiPredictor(nn.Module):
    def __init__(self, input_dim, wide=False):
        super().__init__()
        if wide:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.01),
                nn.Linear(32, 1),
            )
        else:
            # Standard network for fair comparison with linear Ridge
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LeakyReLU(0.01),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.01),
                nn.Linear(32, 1),
            )
        
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def train_nn(X_train, y_train, X_test, y_test, num_epochs=50, max_samples=NN_MAX_TRAIN):
    """Train NN with R² computed in log10(Si) space."""
    # Subsample training set
    if len(X_train) > max_samples:
        print(f"[INFO] Subsampled training: {len(X_train)} -> {max_samples}")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), size=max_samples, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    # Sanitize
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6,  neginf=-1e6)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=1e6,  neginf=-1e6)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=10.0, neginf=-10.0)
    y_test  = np.nan_to_num(y_test,  nan=0.0, posinf=10.0, neginf=-10.0)

    # Scale y to [-1, 1] for stable training
    scaler_y = MaxAbsScaler()
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_s  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    y_train_s = np.clip(y_train_s, -1, 1)
    y_test_s  = np.clip(y_test_s,  -1, 1)

    # Scale X
    scaler_X  = StandardScaler()
    X_train_s = np.clip(scaler_X.fit_transform(X_train), -5, 5)
    X_test_s  = np.clip(scaler_X.transform(X_test),      -5, 5)

    # Tensors
    Xtr = torch.from_numpy(X_train_s).double()
    ytr = torch.from_numpy(y_train_s).double().view(-1, 1)
    Xte = torch.from_numpy(X_test_s).double()

    model     = SiPredictor(Xtr.shape[1], wide=WIDE_NETWORK).double()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-5
    )
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=1024, shuffle=True)

    print("[INFO] Training NN...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")

    model.eval()
    with torch.no_grad():
        pred_scaled = model(Xte).numpy().flatten()

    y_pred_log = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    r2 = r2_score(y_test, y_pred_log)
    return r2, y_pred_log


def train_single_checkpoint_spatial_nn(base_dir, out_dir):
    """Train NN on single checkpoint with spatial hold-out."""
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

    train_mask, test_mask, gap_info = spatial_split(x_pos, strategy=SPATIAL_SPLIT_STRATEGY)
    
    print(f"\n[INFO] Spatial split strategy: {gap_info['strategy']}")
    print(f"       Gap: {gap_info['gap_size']:.3f}")
    print(f"       Train: {gap_info['n_train']}, Test: {gap_info['n_test']}")
    if gap_info['n_discarded'] > 0:
        print(f"       Discarded: {gap_info['n_discarded']}")

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    r2, y_pred = train_nn(X_train, y_train, X_test, y_test, num_epochs=50)

    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    print(f"\n[RESULT] Single-checkpoint NN (spatial hold-out)")
    print(f"         Wide network: {WIDE_NETWORK}")
    print(f"         R²: {r2:.4f}")
    print(f"         RMSE: {rmse:.4f} dex")

    np.savez(
        os.path.join(out_dir, "nn_single_spatial_test.npz"),
        y_true=y_test,
        y_pred=y_pred,
        X=X_test
    )
    print(f"[SUCCESS] Saved to {out_dir}/")


def train_multi_checkpoint_nn(base_dir, out_dir):
    """Train NN on early checkpoint, test on late checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    chk_files = sorted([
        os.path.join(base_dir, f) 
        for f in os.listdir(base_dir) 
        if "ISM_hdf5_chk_" in f
    ])
    chk_files = [f for f in chk_files if get_chk_number(f) >= 4]

    early_path = chk_files[0]
    late_path  = chk_files[-1]

    print(f"[INFO] Train: {early_path}")
    print(f"[INFO] Test:  {late_path}")

    ds_train = yt.load(early_path)
    ds_test  = yt.load(late_path)

    X_train, y_train, _ = build_dataset_with_coords(ds_train)
    X_test,  y_test,  _ = build_dataset_with_coords(ds_test)

    print(f"[INFO] Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    r2, y_pred = train_nn(X_train, y_train, X_test, y_test, num_epochs=50)

    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    
    print(f"\n[RESULT] Multi-checkpoint NN")
    print(f"         Wide network: {WIDE_NETWORK}")
    print(f"         R²: {r2:.4f}")
    print(f"         RMSE: {rmse:.4f} dex")

    np.savez(
        os.path.join(out_dir, "nn_multi_checkpoint_test.npz"),
        y_true=y_test,
        y_pred=y_pred,
        X=X_test
    )
    print(f"[SUCCESS] Saved to {out_dir}/")


if __name__ == "__main__":
    BASE_DIR = "/scratch/ebuie/ISO_Turb/midway/mhd_1e8/1E23_S100_z1_mhd/"
    OUT_DIR  = os.environ.get("OUT_DIR_OVERRIDE") or (
        "siresults_nn_fixed" if ION_FIELD == "si  "
        else f"results_{ION_TAG}_nn")

    print("=" * 70)
    print(f"CONFIG: Ion = {ION_TAG} (field {ION_FIELD!r})")
    print(f"CONFIG: Wide network = {WIDE_NETWORK}")
    print(f"        Spatial split = {SPATIAL_SPLIT_STRATEGY}")
    print("=" * 70)

    print("\nMODE 1 — Single-checkpoint with spatial hold-out (NN)")
    print("=" * 70)
    train_single_checkpoint_spatial_nn(BASE_DIR, OUT_DIR)

    # print("\n" + "=" * 70)
    # print("MODE 2 — Multi-checkpoint temporal generalization (NN)")
    # print("=" * 70)
    # train_multi_checkpoint_nn(BASE_DIR, OUT_DIR)