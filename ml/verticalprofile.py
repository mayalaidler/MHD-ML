from ridge_fits import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
from glob import glob
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler


#df15 = gen_df_tave(fname='mhd_dataset.npz',t1=1500,t2=-1,verbose=0)

# def read_mf_norm(fname='mfields.npz'):
#     # Normalize everything by urms = B_eq
#     mf = np.load(fname)
#     bxm = mf['bxm']/mf['uave']
#     bym = mf['bym']/mf['uave']
#     jxm = mf['jxm']/mf['uave']
#     jym = mf['jym']/mf['uave']
#     Exm = mf['emfx']/mf['uave']
#     Eym = mf['emfy']/mf['uave']
#     return bxm, bym, jxm, jym, Exm, Eym

#hdf5 to pd instead of npz to pd
#fname = '/scratch/mlaidler/astr_thesis/mhd_1e8/1E25_S100_z01_mhd/Simulation/ISM_hdf5_chk_0004'

def read_mf_norm(
    file_pattern: str = None,
    cache_file: str = "meanfields.npz",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mean-field quantities either from cache or by reading FLASH HDF5
    checkpoint files via `yt`.

    If `cache_file` exists it is loaded and returned. Otherwise `file_pattern`
    must be provided and matched files will be processed, the results saved to
    `cache_file`, and the arrays returned.
    """
    import yt

    # If cache exists, load and return immediately
    if cache_file and os.path.exists(cache_file):
        if verbose:
            print(f"Loading mean-fields from cache: {cache_file}")
        data = np.load(cache_file)
        return data['bxm'], data['bym'], data['jxm'], data['jym'], data['Exm'], data['Eym']

    # Otherwise we must have a file pattern to read
    if file_pattern is None:
        raise ValueError("file_pattern must be provided when cache_file does not exist")

    files: List[str] = sorted(glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    bx_list, by_list = [], []
    jx_list, jy_list = [], []
    Ex_list, Ey_list = [], []

    for i, fname in enumerate(files):
        try:
            if verbose:
                print(f"[{i+1}/{len(files)}] Processing {fname}")

            ds = yt.load(fname)
            ad = ds.all_data()
            dims = tuple(int(d) for d in ds.domain_dimensions)

            # Load 3D fields
            bx = ad[("flash", "magx")].v.reshape(dims)
            by = ad[("flash", "magy")].v.reshape(dims)
            bz = ad[("flash", "magz")].v.reshape(dims)

            vx = ad[("flash", "velx")].v.reshape(dims)
            vy = ad[("flash", "vely")].v.reshape(dims)
            vz = ad[("flash", "velz")].v.reshape(dims)

            # Compute EMF = v × B
            Ex = vy * bz - vz * by
            Ey = vz * bx - vx * bz

            # Compute current density J = curl(B)
            dx = (ds.domain_right_edge - ds.domain_left_edge) / np.array(dims)

            dBy_dz = np.gradient(by, dx[2], axis=2)
            dBx_dz = np.gradient(bx, dx[2], axis=2)
            dBz_dx = np.gradient(bz, dx[0], axis=0)
            dBz_dy = np.gradient(bz, dx[1], axis=1)

            jx = dBz_dy - dBy_dz
            jy = dBx_dz - dBz_dx

        except Exception as e:
            print(f"Skipping file {fname} due to error: {e}")
            continue

        # Spatial average over x,y → (z,)
        bx_list.append(np.mean(bx, axis=(0, 1)))
        by_list.append(np.mean(by, axis=(0, 1)))
        jx_list.append(np.mean(jx, axis=(0, 1)))
        jy_list.append(np.mean(jy, axis=(0, 1)))
        Ex_list.append(np.mean(Ex, axis=(0, 1)))
        Ey_list.append(np.mean(Ey, axis=(0, 1)))

        # free memory for this timestep
        del bx, by, bz, vx, vy, vz, Ex, Ey, jx, jy

    if not bx_list:
        raise RuntimeError("No valid checkpoint files were processed")

    # Stack over time → (nt, nz)
    bxm = np.stack(bx_list, axis=0)
    bym = np.stack(by_list, axis=0)
    jxm = np.stack(jx_list, axis=0)
    jym = np.stack(jy_list, axis=0)
    Exm = np.stack(Ex_list, axis=0)
    Eym = np.stack(Ey_list, axis=0)

    # Save cache for future runs
    np.savez_compressed(
        cache_file,
        bxm=bxm, bym=bym,
        jxm=jxm, jym=jym,
        Exm=Exm, Eym=Eym
    )
    if verbose:
        print(f"Saved mean-fields to cache: {cache_file}")

    return bxm, bym, jxm, jym, Exm, Eym

def scale_df(df):
    '''
    Call:   df_ss, scl = scale_df(df)
    Inv. Transform: dfn = scl.inverse_transform(df_ss)
    Check equality: np.allclose(df.to_numpy(),dfn.to_numpy())
    '''
    
    df_ss = df.copy()
    scl   = StandardScaler()
    df_ss = scl.fit_transform(df_ss)
    return pd.DataFrame(df_ss,columns=df.columns),scl


#function to average the fields over time
def ave_t(arr,tone,ttwo,verbose=None):
    if verbose:
        print(f't1: {tone}, t2: {ttwo}')
    return np.mean(arr[tone:ttwo,:],axis=0)

def gen_df_tave(file_pattern,t1=5,t2=-1,verbose=None, cache_file: str = "meanfields.npz"):
    '''
    Generate a dataframe by averaging squared fields over t from t1 to t2
    '''
    
    if verbose:
        print(f"Generating time averaged dataframe with t1: {t1} and t2: {t2}")
  
    bxm, bym, jxm, jym, Exm, Eym = read_mf_norm(file_pattern=file_pattern, cache_file=cache_file, verbose=verbose)
    
    print("bxm", bxm.shape, "bym", bym.shape, "jxm", jxm.shape, "jym", jym.shape, "Exm", Exm.shape, "Eym", Eym.shape)
    
    return pd.DataFrame.from_dict({
        'Bx': ave_t(bxm,tone=t1,ttwo=t2),
        'By': ave_t(bym,tone=t1,ttwo=t2),
        'Jx': ave_t(jxm,tone=t1,ttwo=t2),
        'Jy': ave_t(jym,tone=t1,ttwo=t2),
        'Ex': -1. * ave_t(Exm,tone=t1,ttwo=t2), # changed signs
        'Ey': -1. * ave_t(Eym,tone=t1,ttwo=t2)  # Since PENCIL computes -VxB
        })


df15 = gen_df_tave(file_pattern='/scratch/mlaidler/astr_thesis/mhd_1e8/1E25_S100_z01_mhd/Simulation/ISM_hdf5_chk*',t1=5,t2=-1,verbose=1)
print("printing the averaged dataframe:")
print(df15.head())
print("Dataframe shape:", df15.shape)


#linear regression
from sklearn.linear_model import LinearRegression
#from preprocess import train_test_seq
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error #mean_squared_error

#from preprocess of M2
def train_test_seq(X,y,test_size=0.2):
    '''
    Assume X,y are dataframes
    '''
    
    print(f"Test size: {test_size}")
    train_size = 1.0 - test_size
    totsz = X.shape[0]
    X_train, X_test = X.iloc[:int(train_size*totsz)], X.iloc[int(train_size*totsz):]
    y_train, y_test = y.iloc[:int(train_size*totsz)], y.iloc[int(train_size*totsz):]
    
    return X_train, X_test, y_train, y_test


lr = LinearRegression(fit_intercept=False)

fld  = ['Ex']
flds = ['Ex','Ey']
tst_sz = 0.2

df15_ss, scl = scale_df(df15)
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_seq(df15_ss.drop(flds,axis=1),df15_ss[fld],test_size=tst_sz)
lr.fit(X_train_lr,y_train_lr)
y_pred_lr = lr.predict(X_test_lr)
print(lr.coef_)

#scale time-averaged df
df15_scaled, scl = scale_df(df15)

# ridge regression with grid search and cross validation
y_train,y_pred,y_test = ridge_gridcv(df15_scaled)


