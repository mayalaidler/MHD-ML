# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # SVM
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn import metrics # check model accuracy
from sklearn.model_selection import train_test_split  # split data into train & test
import yt 

import os
import re
import numpy as np
import pandas as pd
import h5py
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib


# Train Si IV mass fraction from early checkpoint features.

# Inputs (early checkpoint ISM_hdf5_chk_0004):
# - gas:density
# - gas:temperature
# - gas:velocity_x, velocity_y, velocity_z
# - flash:magx/magy/magz or flash:mag_strength

# Target (final checkpoint ISM_hdf5_chk_0017):
# - Si IV mass fraction (attempts to locate common names such as
#   ('flash','si4p'), ('flash','si4'), ('flash','si')).

# The script samples up to 20k cells, trains a Ridge regressor, and
# saves model weights to `si4_ridge_model.npz` in the simulation folder.

def find_field(ds, substrings):
    fl = ds.field_list
    for field in fl:
        fname = f"{field[0]}:{field[1]}".lower()
        if all(s.lower() in fname for s in substrings):
            return field
    return None


def quantity_to_numpy(q):
    try:
        return q.to_value() if hasattr(q, "to_value") else q.value
    except Exception:
        return np.array(q)


def main():
    base_dir = "/scratch/mlaidler/astr_thesis/mhd_1e8/1E25_S100_z01_mhd/Simulation/"
    early_chk = os.path.join(base_dir, "ISM_hdf5_chk_0017")

    if not os.path.exists(early_chk):
        raise FileNotFoundError(f"Early checkpoint not found: {early_chk}")

    print(f"Loading early checkpoint: {early_chk}")
    early_ds = yt.load(early_chk)

    # locate Si field in the early dataset (we only use the early snapshot)
    si_field = ('flash', 'si  ')
    if si_field is None:
        available = ",".join(f"{f[0]}:{f[1]}" for f in early_ds.field_list)
        raise RuntimeError(f"Could not locate Si IV field in early dataset. Available fields: {available}")

    print(f"Using Sitarget field from late checkpoint: {si_field}")

    early_ad = early_ds.all_data()

    # required early fields
    req = [ ('flash', 'dens'), ('flash', 'temp'), ("flash","velx"), ("flash","vely"), ("flash","velz") ]
    for r in req:
        if r not in early_ds.field_list:
            raise RuntimeError(f"Required early field missing: {r}")

    # magnetic field
    if ("flash","mag_strength") in early_ds.field_list:
        B_field = ("flash","mag_strength")
    else:
        comps = [("flash","magx"), ("flash","magy"), ("flash","magz")]
        if all(c in early_ds.field_list for c in comps):
            B_field = None
        else:
            B_field = None

    rho = quantity_to_numpy(early_ad[("gas","density")])
    T = quantity_to_numpy(early_ad[("gas","temperature")])
    vx = quantity_to_numpy(early_ad[("gas","velocity_x")])
    vy = quantity_to_numpy(early_ad[("gas","velocity_y")])
    vz = quantity_to_numpy(early_ad[("gas","velocity_z")])

    if B_field is not None:
        B = quantity_to_numpy(early_ad[B_field])
    else:
        Bx = quantity_to_numpy(early_ad[("flash","magx")])
        By = quantity_to_numpy(early_ad[("flash","magy")])
        Bz = quantity_to_numpy(early_ad[("flash","magz")])
        B = np.sqrt(Bx**2 + By**2 + Bz**2)

    si = quantity_to_numpy(early_ad[si_field])

    rho = np.ravel(np.array(rho, dtype=float))
    T = np.ravel(np.array(T, dtype=float))
    vx = np.ravel(np.array(vx, dtype=float))
    vy = np.ravel(np.array(vy, dtype=float))
    vz = np.ravel(np.array(vz, dtype=float))
    B = np.ravel(np.array(B, dtype=float))
    si = np.ravel(np.array(si, dtype=float))

    npts = min(len(rho), len(T), len(vx), len(B), len(si))
    if npts == 0:
        raise RuntimeError("No data points found in dataset arrays")

    N_SAMPLES = min(50000, npts)
    rng = np.random.default_rng(42)
    x = quantity_to_numpy(early_ad[("index", "x")])
    ycoord = quantity_to_numpy(early_ad[("index", "y")])
    z = quantity_to_numpy(early_ad[("index", "z")])

    x = np.ravel(x)
    ycoord = np.ravel(ycoord)
    z = np.ravel(z)

    split_val = np.median(x)

    train_mask = x < split_val
    test_mask  = x >= split_val

    def build_features(mask):
        return np.vstack([
            rho[mask],
            T[mask],
            vx[mask],
            vy[mask],
            vz[mask],
            np.sqrt(vx[mask]**2 + vy[mask]**2 + vz[mask]**2),
            B[mask]
        ]).T, si[mask]

    X_all, y_all = build_features(np.ones_like(rho, dtype=bool))
    X_train_full, y_train_full = build_features(train_mask)
    X_test_full,  y_test_full  = build_features(test_mask)

    N_SAMPLES = 60000

    def sample_subset(X, y, N):
        n = len(X)
        if n <= N:
            return X, y
        idx = rng.choice(n, size=N, replace=False)
        return X[idx], y[idx]

    X_train, y_train = sample_subset(X_train_full, y_train_full, int(0.8 * N_SAMPLES))
    X_test, y_test = sample_subset(X_test_full, y_test_full, int(0.2 * N_SAMPLES))

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    feature_names = ["rho","temperature","velx","vely","velz","v_mag","B_mag"]

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test  = pd.DataFrame(X_test,  columns=feature_names)

    y_train = pd.Series(y_train, name="si_mass_frac")
    y_test  = pd.Series(y_test,  name="si_mass_frac")

    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)


    from sklearn.dummy import DummyRegressor

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train_s, y_train)
    print("Dummy R²:", dummy.score(X_test_s, y_test))

 
    model = Ridge(alpha=0.01)
    model.fit(X_train_s, y_train)

    pred = model.predict(X_test_s)
    mse = mean_squared_error(y_test, pred)

    print(f"Trained Ridge on {len(X_train)} samples; MSE={mse:.6e}")
    print(f"RIDGE R²: {model.score(X_test_s, y_test):.4f}")

   
    # ----------------------------------------
    y_true = np.array(y_test)
    y_pred = np.array(pred)

    nplot = min(len(y_true), 10)
    order = np.arange(len(y_true))
    if len(order) > nplot:
        rng = np.random.default_rng(1)
        order = rng.choice(order, size=nplot, replace=False)

    plt.figure(figsize=(10,5))
    plt.scatter(order, y_true[order], s=8, alpha=0.7, label='True')
    plt.scatter(order, y_pred[order], s=8, alpha=0.6, label='Predicted')
    plt.xlabel('Sample index')
    plt.ylabel('Si mass fraction')
    plt.title(f'Si: True vs Predicted (MSE={mse:.2e})')
    plt.legend()
    plt.grid(alpha=0.3)

    out_fig = "siresults/latestage_si_true_and_pred_series.png"
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    print(f"Saved plot to: {out_fig}")


    out_model = "siresults/latestage_si_ridge_model.npz"
    np.savez_compressed(
        out_model,
        coef=model.coef_,
        intercept=model.intercept_,
        features=feature_names
    )
    print(f"Saved model to: {out_model}")

    # save dataframe (NO leakage now — already split correctly)
    df_train = X_train.copy()
    df_train["si_mass_frac"] = y_train.values

    df_test = X_test.copy()
    df_test["si_mass_frac"] = y_test.values

    np.savez_compressed(
        "siresults/train_data.npz",
        **{col: df_train[col].values for col in df_train.columns}
    )

    np.savez_compressed(
        "siresults/test_data.npz",
        **{col: df_test[col].values for col in df_test.columns}
    )

    print("Saved train/test datasets separately (leakage-safe)")

    # save scaler
    scaler_path = "siresults/latestage_si_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved feature scaler to: {scaler_path}")

if __name__ == "__main__":
    main()