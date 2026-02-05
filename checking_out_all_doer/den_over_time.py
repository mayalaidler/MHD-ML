import yt
import numpy as np
import matplotlib.pyplot as plt

def temp_spread_over_time():
    path = "/scratch/ebuie/ISO_Turb/midway/1E23_S100_z1_mhd/"
    name = "1E23_S100_z1_mhd"
    indices = [0, 1, 2, 3, 4,5, 6, 7, 8,9,10, 11, 12, 13, 14]  # adjust for however many checkpoints exist

    results = []

    for i in indices:
        if i < 10:
            chk = f"{path}/ISM_hdf5_chk_000{i}"
        elif i < 100:
            chk = f"{path}/ISM_hdf5_chk_00{i}"
        else:
            chk = f"{path}/ISM_hdf5_chk_0{i}"

        print(f"Loading {chk} ...")
        ds = yt.load(chk)
        ad = ds.all_data()

        dens = ad[("gas", "density")].v

        mean_dens = np.mean(dens)
        std_dens = np.std(dens)/mean_dens
        time = ds.current_time.to("Myr").v  # time in Myr

        results.append((time, mean_dens, std_dens))

    results = np.array(results)
    times = results[:, 0]
    means = results[:, 1]
    stds = results[:, 2]

    np.savetxt("temperature_spread_over_time.dat",
               np.column_stack([times, means, stds]),
               header="time(Myr) mean_den(K) std_den(K)")
    plt.figure(figsize=(8, 6))
    plt.plot(times, stds, 's--', label='Density Std Dev', color='tab:orange')
    plt.xlabel("Time (Myr)")
    plt.ylabel("Density")
    plt.title(f"Density  Evolution for {name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("density_spread_over_time.png", dpi=200)
    plt.close()

    print("Saved density_spread_over_time.png and data file.")

# Run it
if __name__ == "__main__":
    temp_spread_over_time()

