import yt
import numpy as np
import os
from PIL import Image

# ==== DIRECTORIES ====
mhd_dir = "/scratch/ebuie/ISO_Turb/midway/1E23_S100_z1_mhd"
hydro_dir = "/scratch/ebuie/ISO_Turb/midway/1E23_S100_z1"
output_dir = "/scratch/mlaidler/astr_thesis/comparing_mhd_hydro_plots"
os.makedirs(output_dir, exist_ok=True)

# ==== FILE PAIRS ====
pairs = {
    "7": ("ISM_hdf5_chk_0007", "ISM_hdf5_chk_0007"),
    "10": ("ISM_hdf5_chk_0010", "ISM_hdf5_chk_0010"),
    "11": ("ISM_hdf5_chk_0011", "ISM_hdf5_chk_0011"),
    "12": ("ISM_hdf5_chk_0012", "ISM_hdf5_chk_0013"),
}

# ==== PARAMETERS ====
field = ("gas", "density")
axis = "z"
cmap = "viridis"

# ==== HELPER FUNCTION ====
def stitch_images(img_path1, img_path2, out_path, title1="Hydro", title2="MHD"):
    """Stitches two images side-by-side."""
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    # Make same height
    h = min(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * h / img1.height), h))
    img2 = img2.resize((int(img2.width * h / img2.height), h))

    total_width = img1.width + img2.width
    stitched = Image.new("RGB", (total_width, h))
    stitched.paste(img1, (0, 0))
    stitched.paste(img2, (img1.width, 0))
    stitched.save(out_path)

# ==== MAIN LOOP ====
for label, (hydro_file, mhd_file) in pairs.items():
    print(f"\n=== Processing checkpoint {label} ===")
    
    hydro_path = os.path.join(hydro_dir, hydro_file)
    mhd_path = os.path.join(mhd_dir, mhd_file)

    hydro_ds = yt.load(hydro_path)
    mhd_ds = yt.load(mhd_path)

    # Find consistent limits for density
    hydro_data = hydro_ds.all_data()
    mhd_data = mhd_ds.all_data()

    dens_min = min(hydro_data[field].min(), mhd_data[field].min())
    dens_max = max(hydro_data[field].max(), mhd_data[field].max())
    vmin = np.log10(dens_min)
    vmax = np.log10(dens_max)

    # ---- HYDRO PLOTS ----
    slc_h = yt.SlicePlot(hydro_ds, axis, field)
    slc_h.set_log(field, True)
    slc_h.set_zlim(field, vmin, vmax)
    slc_h.set_cmap(field, cmap)
    slc_h.annotate_title(f"Hydro {label} Density Slice (z-axis)")
    hydro_slice_path = os.path.join(output_dir, f"hydro_slice_{label}.png")
    slc_h.save(hydro_slice_path)

    prj_h = yt.ProjectionPlot(hydro_ds, axis, field, weight_field=None)
    prj_h.set_log(field, True)
    prj_h.set_zlim(field, vmin, vmax)
    prj_h.set_cmap(field, cmap)
    prj_h.annotate_title(f"Hydro {label} Density Projection (z-axis)")
    hydro_proj_path = os.path.join(output_dir, f"hydro_projection_{label}.png")
    prj_h.save(hydro_proj_path)

    # ---- MHD PLOTS ----
    slc_m = yt.SlicePlot(mhd_ds, axis, field)
    slc_m.set_log(field, True)
    slc_m.set_zlim(field, vmin, vmax)
    slc_m.set_cmap(field, cmap)
    slc_m.annotate_title(f"MHD {label} Density Slice (z-axis)")
    mhd_slice_path = os.path.join(output_dir, f"mhd_slice_{label}.png")
    slc_m.save(mhd_slice_path)

    prj_m = yt.ProjectionPlot(mhd_ds, axis, field, weight_field=None)
    prj_m.set_log(field, True)
    prj_m.set_zlim(field, vmin, vmax)
    prj_m.set_cmap(field, cmap)
    prj_m.annotate_title(f"MHD {label} Density Projection (z-axis)")
    mhd_proj_path = os.path.join(output_dir, f"mhd_projection_{label}.png")
    prj_m.save(mhd_proj_path)

    # ---- STITCH TOGETHER ----
    slice_out = os.path.join(output_dir, f"slice_comparison_{label}.png")
    proj_out = os.path.join(output_dir, f"projection_comparison_{label}.png")

    stitch_images(hydro_slice_path, mhd_slice_path, slice_out)
    stitch_images(hydro_proj_path, mhd_proj_path, proj_out)

    print(f" Saved {slice_out} and {proj_out}")

print("\nAll slice and projection comparisons complete!")

