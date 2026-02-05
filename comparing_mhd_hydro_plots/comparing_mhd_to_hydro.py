import yt
import os
import numpy as np
import matplotlib.pyplot as plt

mhd_dir = "/scratch/ebuie/ISO_Turb/midway/1E23_S100_z1_mhd"
hydro_dir = "/scratch/ebuie/ISO_Turb/midway/1E23_S100_z1"

# Map Hydro to MHD checkpoints
pairs = {
    "7": ("ISM_hdf5_chk_0007", "ISM_hdf5_chk_0007"),
    "10": ("ISM_hdf5_chk_0010", "ISM_hdf5_chk_0010"),
    "11": ("ISM_hdf5_chk_0011", "ISM_hdf5_chk_0011"),
    "12": ("ISM_hdf5_chk_0012", "ISM_hdf5_chk_0013"),
}

for label, (hydro_file, mhd_file) in pairs.items():
    hydro = yt.load(os.path.join(hydro_dir, hydro_file))
    mhd = yt.load(os.path.join(mhd_dir, mhd_file))
    mhd.field_list
    # Hydro phase plot: density vs temperature
    p = yt.PhasePlot(
        hydro.all_data(),
        ("gas", "density"),
        ("flash", "OVI"),
        ("gas", "mass"),
        weight_field=None,
    )
    p.set_unit(("gas", "mass"), "Msun")
    # lock axis ranges for fair comparison
    p.set_xlim(1e-29, 1e-20)
    p.set_ylim(1e2, 1e6)
    p.set_zlim(field = ("gas", "mass"), zmin=1e-5,zmax=1e3)
    p.save("hydro_phase.png")

    # MHD phase plot: density vs mag_strength
    m = yt.PhasePlot(
        mhd.all_data(),
        ("gas", "density"),
        ("flash", "OVI"),
        ("gas", "mass"),
        weight_field=None,
    )
    m.set_unit(("gas", "mass"), "Msun")
    m.set_xlim(1e-29, 1e-20)
    m.set_ylim(1e2, 1e6)
    m.set_zlim(field = ("gas", "mass"), zmin=1e-5,zmax=1e3)
    m.save("mhd_phase.png")

    # Stitch side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(plt.imread("hydro_phase.png"))
    axes[0].axis("off")
    axes[0].set_title(f"Hydro {label}")

    axes[1].imshow(plt.imread("mhd_phase.png"))
    axes[1].axis("off")
    axes[1].set_title(f"MHD {label}")

    plt.tight_layout()
    plt.savefig(f"OVI{label}.png", dpi=300)
    plt.close()

    print(f"Saved {label}.png")

