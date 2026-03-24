def get_all_checkpoints(base_dir):
    files = os.listdir(base_dir)
    chk_files = sorted([
        os.path.join(base_dir, f)
        for f in files if "chk_" in f
    ])
    return chk_files

def load_snapshot(ds, max_samples=10000, rng=None):
    ad = ds.all_data()

    rho = quantity_to_numpy(ad[("gas","density")])
    T   = quantity_to_numpy(ad[("gas","temperature")])
    vx  = quantity_to_numpy(ad[("gas","velocity_x")])
    vy  = quantity_to_numpy(ad[("gas","velocity_y")])
    vz  = quantity_to_numpy(ad[("gas","velocity_z")])

    if ("flash","mag_strength") in ds.field_list:
        B = quantity_to_numpy(ad[("flash","mag_strength")])
    else:
        Bx = quantity_to_numpy(ad[("flash","magx")])
        By = quantity_to_numpy(ad[("flash","magy")])
        Bz = quantity_to_numpy(ad[("flash","magz")])
        B = np.sqrt(Bx**2 + By**2 + Bz**2)

    si = quantity_to_numpy(ad[("flash","si  ")])

    # flatten
    rho = rho.flatten()
    T   = T.flatten()
    vx  = vx.flatten()
    vy  = vy.flatten()
    vz  = vz.flatten()
    B   = B.flatten()
    si  = si.flatten()

    npts = len(rho)
    if npts == 0:
        return None

    n_use = min(max_samples, npts)
    idx = rng.choice(npts, size=n_use, replace=False)

    vmag = np.sqrt(vx[idx]**2 + vy[idx]**2 + vz[idx]**2)

    X = np.vstack([
        rho[idx], T[idx], vx[idx], vy[idx], vz[idx], vmag, B[idx]
    ]).T

    y = si[idx]

    return X, y

def main():
    base_dir = "/scratch/ebuie/ISO_Turb/midway/mhd_1e8/1E23_S100_z01_mhd/"

    chk_files = get_all_checkpoints(base_dir)
    print(f"Found {len(chk_files)} checkpoints")
    #extract checkpoint file number and put it in the X_values
    
    
    

    rng = np.random.default_rng(42)

    X_all = []
    y_all = []

    for i, chk in enumerate(chk_files):
        print(f"Loading {chk}")
        try:
            ds = yt.load(chk)
            result = load_snapshot(ds, max_samples=6000, rng=rng)

            if result is None:
                continue

            X, y = result
            X_all.append(X)
            y_all.append(y)

        except Exception as e:
            print(f"Skipping {chk}: {e}")
            continue

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    print(f"Total dataset size: {X_all.shape}")

    feature_names = ["rho","temperature","velx","vely","velz","v_mag","B_mag"]
    df = pd.DataFrame(X_all, columns=feature_names)
    df["si_mass_frac"] = y_all

    X = df[feature_names]
    y = df["si_mass_frac"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = Ridge(alpha=0.01)
    model.fit(X_train_s, y_train)

    pred = model.predict(X_test_s)

    mse = mean_squared_error(y_test, pred)
    score = model.score(X_test_s, y_test)

    print(f"MSE: {mse:.6e}")
    print(f"R^2: {score:.4f}")