import yt
import numpy as np
import matplotlib.pyplot as plt

def mag_field_stats_over_time(runs):
    """
    Compare mean and standard deviation of magnetic field strength over time
    for multiple simulations with different densities/turbulent velocities.
    
    Parameters
    ----------
    runs : list of tuples
        [(label, path), ...] where 'label' is the run label (e.g. 'n=1e23, S=100'),
        and 'path' is the directory containing the ISM_hdf5_chk_ files.
    indices : array-like
        List or array of checkpoint indices to include.
    """

    all_results = {}

    for label, path in runs:
        results = []

        #for i in indices:
            # Build filename
         #   if i < 10:
          #      chk = f"{path}/ISM_hdf5_chk_000{i}"
           # elif i < 100:
           #     chk = f"{path}/ISM_hdf5_chk_00{i}"
           # else:
            #    chk = f"{path}/ISM_hdf5_chk_0{i}"

            print(f"Loading {chk} ...")
            ds = yt.load(ch)
            ad = ds.all_data()

            # Magnetic field magnitude
            B = ad[("gas", "magnetic_field_strength")].to("G").v

            mean_B = np.mean(B)
            std_B = np.std(B)
            time = ds.current_time.to("Myr").v

            results.append((time, mean_B, std_B))

        all_results[label] = np.array(results)

    # === Plot Mean Magnetic Field Strength ===
    plt.figure(figsize=(8, 6))
    for label, data in all_results.items():
        times, means, stds = data[:, 0], data[:, 1], data[:, 2]
        plt.plot(times, means, marker='o', label=label)
    plt.xlabel("Time (Myr)")
    plt.ylabel(r"Mean $|B|$ [G]")
    plt.title("Evolution of Magnetic Field Strength (Mean)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("magnetic_field_mean_comparison.png", dpi=200)
    plt.close()

    # === Plot Standard Deviation ===
    plt.figure(figsize=(8, 6))
    for label, data in all_results.items():
        times, means, stds = data[:, 0], data[:, 1], data[:, 2]
        plt.plot(times, stds, marker='s', label=label)
    plt.xlabel("Time (Myr)")
    plt.ylabel(r"Std Dev of $|B|$ [G]")
    plt.title("Evolution of Magnetic Field Strength (Std Dev)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("magnetic_field_std_comparison.png", dpi=200)
    plt.close()

    print("Saved magnetic_field_mean_comparison.png and magnetic_field_std_comparison.png")

# === Example usage ===
if __name__ == "__main__":
    runs = [
        ("n=1e23, S=100", "/scratch/ebuie/ISO_Turb/midway/1E23_S100_z1_mhd"),
        ("n=1e23, S=30", "/scratch/ebuie/ISO_Turb/midway/1E23_S30_z01_mhd"),
        ("n=1e26, S=100", "/scratch/ebuie/ISO-Turb/midway/1E26_S100_z01_mhd")]
    #indices = np.arange(7, 13, 1)
    mag_field_stats_over_time(runs)

