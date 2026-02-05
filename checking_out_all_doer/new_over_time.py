import yt
import numpy as np
import matplotlib.pyplot as plt

def mag_strength(field, data):
    return np.sqrt(
        data["flash", "magx"]**2 +
        data["flash", "magy"]**2 +
        data["flash", "magz"]**2
    )

yt.add_field( 
    ("flash", "mag_strength"),
    function=mag_strength,
    units="gauss",
    sampling_type="cell"
)

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

        temps = ad[("temperature")].to("K").v
        

        mean_temp = np.mean(temps)
        std_temp = np.std(temps)/mean_temp
        time = ds.current_time.to("Myr").v  # time in Myr
        if i == 7:
                    uvrad = time
        print(f"Checkpoint {i}: mean={mean_temp:.3e} K, std={std_temp:.3e} K")

        results.append((time, mean_temp, std_temp))

    results = np.array(results)
    times = results[:, 0]
    means = results[:, 1]
    stds = results[:, 2]
   

    #np.savetxt("temperature_spread_over_time.dat",
    #          np.column_stack([times, means, stds]),
    #          header="time(Myr) mean_temp(K) std_temp(K)")
    plt.figure(figsize=(8, 6))
    plt.plot(times, stds, 's--', label='Temp Std Dev', color='tab:purple')
    plt.plot(uvrad, stds[7], 'o', label='UV Radiation Equilibrium', color='tab:red', markersize=10)
    plt.xlabel("Time (Myr)")
    plt.ylabel("Temp (K)")
    plt.title(f"Temperature Evolution for {name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("temp_spread_over_time.png", dpi=200)
    plt.close()

    print("Saved temp_spread_over_time.png and data file.")

# Run it
if __name__ == "__main__":
    temp_spread_over_time()




