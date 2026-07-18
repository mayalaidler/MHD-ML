#!/usr/bin/env python3
"""
Single-checkpoint accuracy improvement experiments.

Per ion (Si I / Si II / Si IV), on checkpoint 0006 of 1E23_S100_z1:

  A  ridge_poly base 250k     — current Ridge baseline (reference)
  B  gbm base 250k            — gradient boosting, same data budget
  C  gbm base 2M              — more training data
  D  gbm +phys 2M             — + log_pres, shok, cell-level |grad log T|,
                                 |grad log rho| (mixing indicators)
  E  gbm +phys+elec 2M        — + log electron abundance (chemistry-informed:
                                 only valid if the target sim tracks e-)
  F  hurdle(+phys) 2M         — presence classifier + above-floor regressor
                                 (only when a numerical floor is detected)

Split: median of block x-centers (whole blocks on one side — no leakage
across the boundary).  All configs share the same split and test cells.

Reads FLASH block arrays directly with h5py; gradients are computed within
8x8x8 blocks (edges approximate, interior exact).
"""

import json
import os
import time

import h5py
import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

CHK = "/scratch/ebuie/ISO_Turb/midway/mhd_1e8/1E23_S100_z1_mhd/ISM_hdf5_chk_0006"
IONS = {"SiI": "si  ", "SiII": "sip ", "SiIV": "si3p"}
OUT_DIR = os.environ.get("OUT", "final_results/improve_single_chk")
SEED = 42
EPS = 1e-30

TRAIN_SMALL, TRAIN_BIG, TEST_CAP = 250_000, 2_000_000, 1_000_000


def detect_floor(y_log, split_guess=-14.0, lo=-35.0, hi=-2.0, nbins=130):
    if y_log.min() > split_guess:
        return None
    counts, edges = np.histogram(y_log, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    below = centers < split_guess
    if counts[below].sum() == 0 or counts[~below].sum() == 0:
        return None
    p_f = centers[below][np.argmax(counts[below])]
    p_p = centers[~below][np.argmax(counts[~below])]
    between = (centers > p_f) & (centers < p_p)
    return float(centers[between][np.argmin(counts[between])]) if between.any() else None


def block(name, f):
    return f[name][()].astype(np.float64)          # (nb, nz, ny, nx)


def grad_mag(logfield):
    gz, gy, gx = np.gradient(logfield, axis=(1, 2, 3))
    return np.sqrt(gx**2 + gy**2 + gz**2)


def load_everything():
    t0 = time.time()
    with h5py.File(CHK, "r") as f:
        dens = block("dens", f); temp = block("temp", f)
        velx = block("velx", f); vely = block("vely", f); velz = block("velz", f)
        magx = block("magx", f); magy = block("magy", f); magz = block("magz", f)
        pres = block("pres", f); shok = block("shok", f); elec = block("elec", f)
        ions = {tag: block(field, f) for tag, field in IONS.items()}
        bbox = f["bounding box"][()]               # (nb, ndim, 2)

    nb, nz, ny, nx = dens.shape
    log_rho, log_T = np.log10(dens + EPS), np.log10(temp + EPS)
    g_T, g_rho = grad_mag(log_T), grad_mag(log_rho)

    B = np.sqrt(magx**2 + magy**2 + magz**2)
    vmag = np.sqrt(velx**2 + vely**2 + velz**2)

    def rav(a):
        return a.ravel()

    base = np.column_stack([
        rav(log_rho), rav(log_T), rav(velx), rav(vely), rav(velz), rav(vmag),
        rav(np.log10(B + EPS)),
        rav(vmag / (np.sqrt(temp) + EPS)),
        rav(np.log10(np.clip(dens * temp / (B**2 + EPS), EPS, None))),
    ])
    phys = np.column_stack([
        rav(np.log10(pres + EPS)), rav(shok), rav(g_T), rav(g_rho)])
    chem = rav(np.log10(elec + EPS)).reshape(-1, 1)

    block_x = bbox[:, 0, :].mean(axis=1)           # block x-centers
    cell_block = np.repeat(np.arange(nb), nz * ny * nx)
    in_train_block = block_x[cell_block] < np.median(block_x)

    ys = {tag: np.log10(np.clip(rav(a), 1e-35, None)) for tag, a in ions.items()}
    ok = np.isfinite(base).all(axis=1) & np.isfinite(phys).all(axis=1) \
        & np.isfinite(chem).all(axis=1)
    print(f"[load] {nb * nz * ny * nx} cells in {time.time()-t0:.0f}s, "
          f"valid {ok.mean():.3f}, train-side {in_train_block.mean():.3f}")
    return base[ok], phys[ok], chem[ok], {t: y[ok] for t, y in ys.items()}, \
        in_train_block[ok]


def metrics(y_true, y_pred, floor):
    err = y_pred - y_true
    out = {"r2": r2_score(y_true, y_pred),
           "rmse": float(np.sqrt(np.mean(err**2))),
           "f05": float((np.abs(err) < 0.5).mean()),
           "f10": float((np.abs(err) < 1.0).mean())}
    if floor is not None:
        keep = y_true > floor
        if keep.sum() > 10:
            out["r2_above"] = r2_score(y_true[keep], y_pred[keep])
            out["rmse_above"] = float(np.sqrt(np.mean(err[keep]**2)))
            out["presence_acc"] = float(((y_pred > floor) == keep).mean())
    return out


def gbm(seed=SEED):
    return HistGradientBoostingRegressor(
        max_iter=400, early_stopping=True, validation_fraction=0.1,
        random_state=seed)


def run_ion(tag, base, phys, chem, y, train_side):
    rng = np.random.default_rng(SEED)
    tr_all = np.flatnonzero(train_side)
    te_all = np.flatnonzero(~train_side)
    te = rng.choice(te_all, size=min(TEST_CAP, len(te_all)), replace=False)
    tr_small = rng.choice(tr_all, size=min(TRAIN_SMALL, len(tr_all)), replace=False)
    tr_big = rng.choice(tr_all, size=min(TRAIN_BIG, len(tr_all)), replace=False)

    floor = detect_floor(y[tr_big])
    results = {}
    print(f"\n═══ {tag}: floor={floor}, "
          f"target std={y[te].std():.2f} dex", flush=True)

    def record(name, y_pred):
        m = metrics(y[te], y_pred, floor)
        results[name] = m
        extra = (f"  above-floor R²={m.get('r2_above', float('nan')):.3f} "
                 f"RMSE={m.get('rmse_above', float('nan')):.2f} "
                 f"presence={m.get('presence_acc', float('nan')):.3f}"
                 if floor is not None else "")
        print(f"  {name:22s} R²={m['r2']:7.3f}  RMSE={m['rmse']:5.2f}  "
              f"<0.5dex={m['f05']:.2%}{extra}", flush=True)

    # A: current Ridge baseline
    t0 = time.time()
    poly = PolynomialFeatures(degree=2, include_bias=False)
    sc = StandardScaler()
    Xtr = sc.fit_transform(poly.fit_transform(base[tr_small]))
    ridge = RidgeCV(alphas=np.logspace(-5, 2, 20), cv=5).fit(Xtr, y[tr_small])
    record("A ridge_poly base 250k",
           ridge.predict(sc.transform(poly.transform(base[te]))))

    # B/C: gbm on base features, small vs big budget
    record("B gbm base 250k",
           gbm().fit(base[tr_small], y[tr_small]).predict(base[te]))
    record("C gbm base 2M",
           gbm().fit(base[tr_big], y[tr_big]).predict(base[te]))

    # D: + physics features
    Xp = np.hstack([base, phys])
    record("D gbm +phys 2M",
           gbm().fit(Xp[tr_big], y[tr_big]).predict(Xp[te]))

    # E: + electron abundance (chemistry-informed)
    Xpe = np.hstack([base, phys, chem])
    record("E gbm +phys+elec 2M",
           gbm().fit(Xpe[tr_big], y[tr_big]).predict(Xpe[te]))

    # F: hurdle on D's features
    if floor is not None:
        above_tr = y[tr_big] > floor
        if above_tr.any() and not above_tr.all():
            clf = HistGradientBoostingClassifier(
                max_iter=200, random_state=SEED).fit(Xp[tr_big], above_tr)
            reg = gbm().fit(Xp[tr_big][above_tr], y[tr_big][above_tr])
            pred = np.full(len(te), np.median(y[tr_big][~above_tr]))
            is_above = clf.predict(Xp[te])
            pred[is_above] = reg.predict(Xp[te][is_above])
            record("F hurdle +phys 2M", pred)

    print(f"  ({tag} done in {time.time()-t0:.0f}s)", flush=True)
    return results, floor


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base, phys, chem, ys, train_side = load_everything()
    all_results = {}
    for tag in IONS:
        res, floor = run_ion(tag, base, phys, chem, ys[tag], train_side)
        all_results[tag] = {"floor": floor, "configs": res}
    with open(os.path.join(OUT_DIR, "improvement_results.json"), "w") as fh:
        json.dump(all_results, fh, indent=2, default=float)
    print(f"\n[SAVED] {OUT_DIR}/improvement_results.json")


if __name__ == "__main__":
    main()
