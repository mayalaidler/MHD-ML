#code to see what the EMF looks like in my simulatioin data 
import yt
import numpy as np
import matplotlib.pyplot as plt

file_path = '/scratch/mlaidler/astr_thesis/mhd_1e8/1E25_S100_z01_mhd/Simulation/ISM_hdf5_chk_0004' 

def _Ex(field, data):
    vx = data[("flash", "velx")]
    vy = data[("flash", "vely")]
    vz = data[("flash", "velz")]
    
    bx = data[("flash", "magx")]
    by = data[("flash", "magy")]
    bz = data[("flash", "magz")]

    return vy * bz - vz * by

def _Ey(field, data):
    vx = data[("flash", "velx")]
    vy = data[("flash", "vely")]
    vz = data[("flash", "velz")]
    
    bx = data[("flash", "magx")]
    by = data[("flash", "magy")]
    bz = data[("flash", "magz")]

    return vz * bx - vx * bz

ds = yt.load(file_path)
ds.add_field(("flash", "Ex"),
             function=_Ex,
             sampling_type="cell",
             units="code_velocity*code_magnetic")

ds.add_field(("flash", "Ey"),
             function=_Ey,
             sampling_type="cell",
             units="code_velocity*code_magnetic")

import os

# Ensure output directory exists
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# --- Additional visualizations: Ex/Ey vs magnetic field values ---
ad = ds.all_data()

# Extract arrays (convert to physical units using dataset metadata where possible)
try:
    # Convert velocities to SI (m/s)
    vx = ad[("flash", "velx")].in_units("m/s").v.flatten()
    vy = ad[("flash", "vely")].in_units("m/s").v.flatten()
    vz = ad[("flash", "velz")].in_units("m/s").v.flatten()

    # Try magnetic field conversion to Tesla first, fall back to Gauss if needed
    try:
        bx = ad[("flash", "magx")].in_units("T").v.flatten()
        by = ad[("flash", "magy")].in_units("T").v.flatten()
        bz = ad[("flash", "magz")].in_units("T").v.flatten()
        b_unit_label = 'T'
    except Exception:
        bx = ad[("flash", "magx")].in_units("G").v.flatten()
        by = ad[("flash", "magy")].in_units("G").v.flatten()
        bz = ad[("flash", "magz")].in_units("G").v.flatten()
        b_unit_label = 'G'

    # Compute electric field (EMF) in SI units: E = v x B
    ex = vy * bz - vz * by
    ey = vz * bx - vx * bz
    e_unit_label = f'm/s*{b_unit_label}'
except Exception as exc:
    print('Warning: unit conversion using dataset metadata failed, falling back to code units:', exc)
    ex = ad[("flash", "Ex")].in_units("code_velocity*code_magnetic").v.flatten()
    ey = ad[("flash", "Ey")].in_units("code_velocity*code_magnetic").v.flatten()
    bx = ad[("flash", "magx")].in_units("code_magnetic").v.flatten()
    by = ad[("flash", "magy")].in_units("code_magnetic").v.flatten()
    bz = ad[("flash", "magz")].in_units("code_magnetic").v.flatten()
    b_unit_label = 'code_magnetic'
    e_unit_label = 'code_velocity*code_magnetic'

# Magnetic field magnitude
bmag = np.sqrt(bx**2 + by**2 + bz**2)

# Downsample if the dataset is large to keep plotting fast
N = ex.size
max_points = 200000
if N > max_points:
    idx = np.random.choice(N, max_points, replace=False)
    ex_s = ex[idx]
    ey_s = ey[idx]
    bmag_s = bmag[idx]
    bx_s = bx[idx]
else:
    ex_s, ey_s, bmag_s, bx_s = ex, ey, bmag, bx

# Hexbin: Ex vs |B|
plt.figure(figsize=(6,5))
hb = plt.hexbin(bmag_s, ex_s, gridsize=200, cmap='viridis', bins='log')
plt.colorbar(hb, label='log10(N)')
plt.xlabel(f'B magnitude ({b_unit_label})')
plt.ylabel(f'Ex ({e_unit_label})')
plt.title('Ex vs |B|')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Ex_vs_Bmag_hexbin.png'), dpi=200)
plt.close()

# Hexbin: Ey vs |B|
plt.figure(figsize=(6,5))
hb2 = plt.hexbin(bmag_s, ey_s, gridsize=200, cmap='viridis', bins='log')
plt.colorbar(hb2, label='log10(N)')
plt.xlabel(f'B magnitude ({b_unit_label})')
plt.ylabel(f'Ey ({e_unit_label})')
plt.title('Ey vs |B|')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Ey_vs_Bmag_hexbin.png'), dpi=200)
plt.close()

# Scatter: Ex vs Bx (downsampled)
plt.figure(figsize=(6,5))
plt.scatter(bx_s, ex_s, s=1, alpha=0.2, color='k')
plt.xlabel(f'B_x ({b_unit_label})')
plt.ylabel(f'Ex ({e_unit_label})')
plt.title('Ex vs B_x (downsampled)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Ex_vs_Bx_scatter.png'), dpi=200)
plt.close()

print('Saved additional Ex/Ey vs B visualizations to', output_dir)

# --- yt-based visualizations for paper-quality figures ---
# Add E magnitude field (computed from v x B) for plotting
def _Emag(field, data):
    vx = data[("flash", "velx")]
    vy = data[("flash", "vely")]
    vz = data[("flash", "velz")]
    bx = data[("flash", "magx")]
    by = data[("flash", "magy")]
    bz = data[("flash", "magz")]
    ex = vy * bz - vz * by
    ey = vz * bx - vx * bz
    ez = vx * by - vy * bx
    return np.sqrt(ex**2 + ey**2 + ez**2)

ds.add_field(("flash", "E_mag"), function=_Emag, sampling_type="cell", units="code_velocity*code_magnetic")

# 1) High-resolution slice of |E| with magnetic-field quivers and density contours
p = yt.SlicePlot(ds, "z", ("flash", "E_mag"))
p.set_cmap(("flash", "E_mag"), "inferno")
p.set_log(("flash", "E_mag"), True)
# overlay quiver of B-field on the slice
p.annotate_quiver(("flash", "magx"), ("flash", "magy"), factor=12)

# overlay density contours for context
# p.annotate_contour(("gas", "density"), ncontours=6, clim=None)
# p.annotate_title("|E| (slice) with B-field vectors and density contours")
# p.set_font_size(18)
# p.save(os.path.join(output_dir, "E_mag_slice_vectors.png"))

# 2) Density-weighted projection of |E|
pp = yt.ProjectionPlot(ds, "z", ("flash", "E_mag"), weight_field=("gas", "density"))
pp.set_cmap(("flash", "E_mag"), "viridis")
pp.set_log(("flash", "E_mag"), True)
pp.annotate_title("Density-weighted projection of |E|")
pp.save(os.path.join(output_dir, "E_mag_proj_densityweighted.png"))

print('Saved yt slice and projection visualizations to', output_dir)


import glob


def timeseries_emf_b(base_dir, out_dir, max_snapshots=50):
    pattern = os.path.join(base_dir, 'ISM_hdf5_chk_*')
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        print('No checkpoint files found with pattern', pattern)
        return

    times = []
    meanE = []
    meanB = []
    medE = []
    medB = []

    for i, f in enumerate(files[:max_snapshots]):
        print('Loading', f)
        ds_i = yt.load(f)
        ad_i = ds_i.all_data()

        # Convert fields to SI where possible (vel -> m/s, B -> T or G)
        try:
            vx = ad_i[("flash", "velx")].in_units('m/s').v.flatten()
            vy = ad_i[("flash", "vely")].in_units('m/s').v.flatten()
            vz = ad_i[("flash", "velz")].in_units('m/s').v.flatten()
            try:
                bx = ad_i[("flash", "magx")].in_units('T').v.flatten()
                by = ad_i[("flash", "magy")].in_units('T').v.flatten()
                bz = ad_i[("flash", "magz")].in_units('T').v.flatten()
                b_unit = 'T'
            except Exception:
                bx = ad_i[("flash", "magx")].in_units('G').v.flatten()
                by = ad_i[("flash", "magy")].in_units('G').v.flatten()
                bz = ad_i[("flash", "magz")].in_units('G').v.flatten()
                b_unit = 'G'

            e_mag = np.sqrt((vy*bz - vz*by)**2 + (vz*bx - vx*bz)**2 + (vx*by - vy*bx)**2)
            bmag = np.sqrt(bx**2 + by**2 + bz**2)
            time_val = ds_i.current_time.in_units('yr').v
        except Exception as exc:
            print('Warning: failed to convert to SI on', f, ' — falling back to code units:', exc)
            ex_i = ad_i[("flash", "Ex")].in_units('code_velocity*code_magnetic').v.flatten()
            ey_i = ad_i[("flash", "Ey")].in_units('code_velocity*code_magnetic').v.flatten()
            bx = ad_i[("flash", "magx")].in_units('code_magnetic').v.flatten()
            by = ad_i[("flash", "magy")].in_units('code_magnetic').v.flatten()
            bz = ad_i[("flash", "magz")].in_units('code_magnetic').v.flatten()
            e_mag = np.sqrt(ex_i**2 + ey_i**2)
            bmag = np.sqrt(bx**2 + by**2 + bz**2)
            time_val = ds_i.current_time.in_units('code_time').v
            b_unit = 'code_magnetic'

        times.append(time_val)
        meanE.append(np.mean(e_mag))
        meanB.append(np.mean(bmag))
        medE.append(np.median(e_mag))
        medB.append(np.median(bmag))

    times = np.array(times)
    meanE = np.array(meanE)
    meanB = np.array(meanB)
    medE = np.array(medE)
    medB = np.array(medB)

    # Sort by time
    order = np.argsort(times)
    times = times[order]
    meanE = meanE[order]
    meanB = meanB[order]
    medE = medE[order]
    medB = medB[order]

    # Plot time series: mean |E| and mean |B|
    fig, ax1 = plt.subplots(figsize=(7,5))
    ax1.plot(times, meanE, '-o', label='mean |E|')
    ax1.set_yscale('log')
    ax1.set_xlabel('Time (yr)')
    ax1.set_ylabel(f'mean |E| (units depend on conversion)')

    ax2 = ax1.twinx()
    ax2.plot(times, meanB, '-s', color='C1', label='mean |B|')
    ax2.set_yscale('log')
    ax2.set_ylabel(f'mean |B| ({b_unit})')

    fig.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'timeseries_meanE_meanB.png'), dpi=200)
    plt.close(fig)

    # Scatter meanE vs meanB colored by time
    plt.figure(figsize=(6,5))
    sc = plt.scatter(meanB, meanE, c=times, cmap='plasma', s=40)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(f'mean |B| ({b_unit})')
    plt.ylabel('mean |E|')
    cbar = plt.colorbar(sc)
    cbar.set_label('Time (yr)')
    plt.title('Mean |E| vs Mean |B| (colored by time)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'meanE_vs_meanB_times.png'), dpi=200)
    plt.close()

    print('Saved time-series plots to', out_dir)


# Run the timeseries analysis on the available checkpoint files
#running on buies files while I rerun mine 
base_dir = '/scratch/ebuie/ISO_Turb/midway/mhd_1e8/1E23_S100_z01_mhd/'
timeseries_emf_b(base_dir, output_dir, max_snapshots=50)
