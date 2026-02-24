import yt
import numpy as np
from pathlib import Path

import numpy as np


# #Field list for reference 
# field list: [('flash', 'accx'), ('flash', 'accy'), ('flash', 'accz'), ('flash', 'c   '), 
#              ('flash', 'c2p '), ('flash', 'c3p '), ('flash', 'c4p '), ('flash', 'c5p '), 
#              ('flash', 'ca  '), ('flash', 'ca2p'), ('flash', 'ca3p'), ('flash', 'ca4p'), 
#              ('flash', 'cap '), ('flash', 'chdt'), ('flash', 'cjto'), ('flash', 'cp  '), 
#              ('flash', 'dens'), ('flash', 'divb'), ('flash', 'eint'), ('flash', 'elec'), 
#              ('flash', 'ener'), ('flash', 'fe  '), ('flash', 'fe2p'), ('flash', 'fe3p'), 
#              ('flash', 'fe4p'), ('flash', 'fep '), ('flash', 'gamc'), ('flash', 'game'), 
#              ('flash', 'h   '), ('flash', 'he  '), ('flash', 'he2p'), ('flash', 'hep '), 
#              ('flash', 'hp  '), ('flash', 'jtdt'), ('flash', 'lc  '), ('flash', 'lca '), 
#              ('flash', 'lct '), ('flash', 'lfe '), ('flash', 'lfl '), ('flash', 'lh  '), 
#              ('flash', 'lhe '), ('flash', 'lmg '), ('flash', 'ln  '), ('flash', 'lna '), 
#              ('flash', 'lne '), ('flash', 'lo  '), ('flash', 'lph '), ('flash', 'ls  '), 
#              ('flash', 'lsi '), ('flash', 'magp'), ('flash', 'magx'), ('flash', 'magy'), 
#              ('flash', 'magz'), ('flash', 'metl'), ('flash', 'mg  '), ('flash', 'mg2p'), 
#              ('flash', 'mg3p'), ('flash', 'mgp '), ('flash', 'n   '), ('flash', 'n2p '), 
#              ('flash', 'n3p '), ('flash', 'n4p '), ('flash', 'n5p '), ('flash', 'n6p '), 
#              ('flash', 'na  '), ('flash', 'na2p'), ('flash', 'nap '), ('flash', 'ne  '), 
#              ('flash', 'ne2p'), ('flash', 'ne3p'), ('flash', 'ne4p'), ('flash', 'ne5p'), 
#              ('flash', 'ne6p'), ('flash', 'ne7p'), ('flash', 'ne8p'), ('flash', 'ne9p'), 
#              ('flash', 'nep '), ('flash', 'np  '), ('flash', 'o   '), ('flash', 'o2p '), 
#              ('flash', 'o3p '), ('flash', 'o4p '), ('flash', 'o5p '), ('flash', 'o6p '), 
#              ('flash', 'o7p '), ('flash', 'oden'), ('flash', 'op  '), ('flash', 'otmp'), 
#              ('flash', 'pres'), ('flash', 's   '), ('flash', 's2p '), ('flash', 's3p '), 
#              ('flash', 's4p '), ('flash', 'shok'), ('flash', 'si  '), ('flash', 'si2p'), 
#              ('flash', 'si3p'), ('flash', 'si4p'), ('flash', 'si5p'), ('flash', 'sip '), 
#              ('flash', 'sp  '), ('flash', 'temp'), ('flash', 'velx'), ('flash', 'vely'), 
#              ('flash', 'velz')]

def build_dataset(chkpt, out="mhd_dataset.npz"):
    """
    Build ML dataset from FLASH 3D checkpoints.
    
    Parameters
    ----------
    chkpt: 
        FLASH checkpoint file paths
    out : str
        Output npz filename
    """

    all_X = []
    all_Y = []

    print(f"Processing {len(chkpt)} snapshots...")

    for fname in chkpt:
        print(f"Loading {fname}")
        ds = yt.load(fname)
        ad = ds.all_data()
        dims = ds.domain_dimensions
        break  # just do one snapshot for now to test

        # --- Load 3D fields ---
        rho = ad[("flash","density")].reshape(dims)
        temp = ad[("flash","temp")].reshape(dims)
        vx = ad[("flash","velx")].reshape(dims)
        vy = ad[("flash","vely")].reshape(dims)
        vz = ad[("flash","velz")].reshape(dims)
        Bx = ad[("flash","magx")].reshape(dims)
        By = ad[("flash","magy")].reshape(dims)
        Bz = ad[("flash","magz")].reshape(dims)
        P  = ad[("flash","pres")].reshape(dims)

        # --- Horizontal average → vertical profiles ---
        rho_z = np.mean(rho, axis=(0,1))
        T_z   = np.mean(temp, axis=(0,1))
        vx_z  = np.mean(vx, axis=(0,1))
        vy_z  = np.mean(vy, axis=(0,1))
        vz_z  = np.mean(vz, axis=(0,1))
        P_z   = np.mean(P, axis=(0,1))

        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
        Bmag_z = np.mean(Bmag, axis=(0,1))

        # --- Derived physics feature ---
        dT_dz = np.gradient(T_z)

        # --- Stack features ---
        X_snapshot = np.column_stack([
            T_z,
            vx_z,
            vy_z,
            vz_z,
            Bmag_z,
            P_z,
            dT_dz
        ])

        Y_snapshot = rho_z

        all_X.append(X_snapshot)
        all_Y.append(Y_snapshot)

    # --- Stack across all snapshots ---
    X = np.vstack(all_X)
    Y = np.concatenate(all_Y)

    print("Stacked dataset shape:")
    print("X:", X.shape)
    print("Y:", Y.shape)

    # --- Global normalization ---
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0] = 1.0  # avoid divide by zero

    X_norm = (X - mean) / std

    # --- Save dataset ---
    np.savez_compressed(
        out,
        X=X_norm,
        Y=Y,
        mean=mean,
        std=std
    )

    print(f"Saved dataset to {out}")
    
build_dataset(chkpt=[
    "/scratch/mlaidler/astr_thesis/mhd_1e8/1E25_S100_z01_mhd/Simulation/ISM_hdf5_chk_0004",])


def read_mf_norm(fname='mfields.npz'):
    # Normalize everything by urms = B_eq
    mf = np.load(fname)
    bxm = mf['bxm']/mf['uave']
    bym = mf['bym']/mf['uave']
    jxm = mf['jxm']/mf['uave']
    jym = mf['jym']/mf['uave']
    Exm = mf['emfx']/mf['uave']
    Eym = mf['emfy']/mf['uave']
    return bxm, bym, jxm, jym, Exm, Eym

def read_mf_norm(fname='mfields.npz'):
    # Normalize everything by urms = B_eq
    mf = np.load(fname)
    bxm = mf()/mf['uave']
    bym = mf['bym']/mf['uave']
    jxm = mf['jxm']/mf['uave']
    jym = mf['jym']/mf['uave']
    Exm = mf['emfx']/mf['uave']
    Eym = mf['emfy']/mf['uave']
    return bxm, bym, jxm, jym, Exm, Eym

def scale_df(df):
    '''
    Call:   df_ss, scl = scale_df(df)
    Inv. Transform: dfn = scl.inverse_transform(df_ss)
    Check equality: np.allclose(df.to_numpy(),dfn.to_numpy())
    '''

    from sklearn.preprocessing import StandardScaler
    df_ss = df.copy()
    scl   = StandardScaler()
    df_ss = scl.fit_transform(df_ss)
    return pd.DataFrame(df_ss,columns=df.columns),scl

def ave_t(arr,tone=1000,ttwo=2000,verbose=None):
    if verbose:
        print(f't1: {tone}, t2: {ttwo}')
    return np.mean(arr[tone:ttwo,:],axis=0)


#train and test
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


