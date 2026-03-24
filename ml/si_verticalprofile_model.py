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

# requires yt, numpy, glob
import numpy as np, yt, os
from glob import glob

def vertical_profile_from_ds(ds, field):
    """Return x,y-averaged profile vs z for `field` (field is a tuple like ('gas','density'))."""
    ad = ds.all_data()
    dims = tuple(int(d) for d in ds.domain_dimensions)
    arr3 = ad[field].v.reshape(dims)
    # average over x,y -> shape (nz,)
    profile = np.mean(arr3, axis=(0,1))
    return profile

def time_average_profiles(file_pattern, field, t1=0, t2=None, verbose=False):
    """Read all files matching pattern, compute vertical profiles, stack and time-average.
       Returns array shape (nz,) (time-averaged)."""
    files = sorted(glob(file_pattern))
    if not files:
        raise FileNotFoundError(file_pattern)
    profs = []
    for i,f in enumerate(files):
        if verbose: print(f"Loading {f} ({i+1}/{len(files)})")
        ds = yt.load(f)
        profs.append(vertical_profile_from_ds(ds, field))
        del ds
    profs = np.stack(profs, axis=0)   # (nt, nz)
    if t2 is None:
        t2 = profs.shape[0]
    return np.mean(profs[t1:t2], axis=0)


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
    base_dir = "/scratch/ebuie/ISO_Turb/midway/mhd_1e8/1E23_S100_z01_mhd/"
    early_chk = os.path.join(base_dir, "ISM_hdf5_chk_0004")
    #out_directory should be currrent directoyr 
    out_dir = ('time_avgsiresults')

    if not os.path.exists(early_chk):
        raise FileNotFoundError(f"Early checkpoint not found: {early_chk}")

    print(f"Loading early checkpoint: {early_chk}")
    early_ds = yt.load(early_chk)

    # locate Si field in the early dataset (use finder with fallbacks)
    si_field = find_field(early_ds, ["si5p"]) or find_field(early_ds, ["si5"]) or find_field(early_ds, ["si"]) or find_field(early_ds, ["si","iv"])
    if si_field is None:
        available = ",".join(f"{f[0]}:{f[1]}" for f in early_ds.field_list)
        raise RuntimeError(f"Could not locate Si field in early dataset. Available fields: {available}")

    print(f"Using Si target field from early checkpoint: {si_field}")

    # time-average over multiple checkpoint files (pattern derived from base_dir)
    base_pattern = os.path.join(base_dir, "ISM_hdf5_chk_*")

    rho_prof = time_average_profiles(base_pattern, ('gas','density'), t1=0, t2=5)
    T_prof   = time_average_profiles(base_pattern, ('gas','temperature'), t1=0, t2=5)
    vxp = time_average_profiles(base_pattern, ('gas','velocity_x'), t1=0, t2=5)
    vyp = time_average_profiles(base_pattern, ('gas','velocity_y'), t1=0, t2=5)
    vzp = time_average_profiles(base_pattern, ('gas','velocity_z'), t1=0, t2=5)
    Bxm = time_average_profiles(base_pattern, ('flash','magx'), t1=0, t2=5)
    Bym = time_average_profiles(base_pattern, ('flash','magy'), t1=0, t2=5)
    Bzm = time_average_profiles(base_pattern, ('flash','magz'), t1=0, t2=5)
    # use the discovered si_field for the Si profile
    si_prof = time_average_profiles(base_pattern, si_field, t1=0, t2=5)

    vmag = np.sqrt(vxp**2 + vyp**2 + vzp**2)
    Bmag = np.sqrt(Bxm**2 + Bym**2 + Bzm**2)

    # Build per-z features: each z-bin is a sample with 4 features
    features_matrix = np.vstack([rho_prof, T_prof, vmag, Bmag]).T  # shape (nz, 4)
    feature_names = ["rho","temperature","v_mag","B_mag"]
    target = si_prof  # shape (nz,)

    df = pd.DataFrame(features_matrix, columns=feature_names)
    df["si_mass_frac"] = target

    X = df[feature_names]
    y = df["si_mass_frac"]
    #save this dataframe to a npz file in time_avgsiresults directory
    out_npz = os.path.join('time_avgsiresults', "si_timeavg_profiles.npz")
    np.savez_compressed(out_npz, rho=rho_prof, temperature=T_prof, v_mag=vmag, B_mag=Bmag, si_mass_frac=si_prof)
    print(f"Saved time-averaged profiles to: {out_npz}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("x Training samples:", X_train.head(3), "y train samples:", y_train.head(3))

    # Scale features (fit on train only)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    from sklearn.dummy import DummyRegressor

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    print(dummy.score(X_test, y_test))

    model = Ridge(alpha=0.01)
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    mse = mean_squared_error(y_test, pred)

    print(f"Trained Ridge on {len(X_train)} samples; MSE={mse:.6e}")
    
    y_true = np.array(y_test)
    y_pred = np.array(pred)

    nplot = min(len(y_true), 20)
    order = np.arange(len(y_true))
    if len(order) > nplot:
        rng = np.random.default_rng(1)
        order = rng.choice(order, size=nplot, replace=False)

    plt.figure(figsize=(10,5))
    plt.scatter(order, y_true[order], s=8, alpha=0.7, c='tab:blue', label='True Si 2')
    plt.scatter(order, y_pred[order], s=8, alpha=0.6, c='tab:orange', label='Predicted Si 2')
    plt.xlabel('Sample index')
    plt.ylabel('Si mass fraction')
    plt.title(f'Si: True vs Predicted (MSE={mse:.2e})')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    out_dir = 'time_avgsiresults'
    os.makedirs(out_dir, exist_ok=True)
    out_fig = os.path.join(out_dir, 'si_true_and_pred_series.png')
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    print(f"Saved plot to: {out_fig}")

    out_model = os.path.join(out_dir, "si_ridge_model.npz")
    np.savez_compressed(out_model, coef=model.coef_, intercept=model.intercept_, features=feature_names)
    print(f"Saved model to: {out_model}")

    # save scaler
    scaler_path = os.path.join(out_dir, 'si_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Saved feature scaler to: {scaler_path}")
    
    print(f"RIDGE score: {model.score(X_test_s,y_test)}")
    print(f"Coefficients: {model.coef_}")

if __name__ == "__main__":
    main()