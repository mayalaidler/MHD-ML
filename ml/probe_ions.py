# Quick probe: mass-fraction stats for Si ionization states in one checkpoint
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/scratch/ebuie/ISO_Turb/midway/mhd_1e8/1E23_S100_z1_mhd/ISM_hdf5_chk_0006"

print(f"Probing: {path}\n")
with h5py.File(path, "r") as f:
    for name, label in [("si  ", "Si I "), ("sip ", "Si II"),
                        ("si2p", "Si III"), ("si3p", "Si IV")]:
        arr = f[name][()].ravel()
        pos = arr[arr > 0]
        n, npos = len(arr), len(pos)
        if npos == 0:
            print(f"{label} ({name!r}): NO positive values out of {n:,} cells")
            continue
        lg = np.log10(pos)
        print(f"{label} ({name!r}): {npos:,}/{n:,} cells > 0 "
              f"({100*npos/n:.1f}%)")
        print(f"   log10 range [{lg.min():.2f}, {lg.max():.2f}]  "
              f"median {np.median(lg):.2f}  "
              f"p5 {np.percentile(lg,5):.2f}  p95 {np.percentile(lg,95):.2f}  "
              f"std {lg.std():.2f}")
