import yt
import numpy as np
import matplotlib.pyplot as plt

chk = "/scratch/ebuie/ISO_Turb/midway/1E23_S100_z1_mhd/ISM_hdf5_chk_0007"

ds = yt.load(chk)
ad = ds.all_data()

field = ("gas", "temperature")

values = ad[field].to("K")

mean_val = np.mean(values)
abs_dev = np.abs(values - mean_val)
new = mean_val/abs_dev

print(f"Mean {field}: {mean_val:.3e} K")
print(f"Standard deviation: {np.std(values):.3e} K")

plt.figure(figsize=(8, 6))
hb = plt.scatter(
    values,
    new,
    gridsize=200,
    bins="log",
    cmap="viridis",
)
cb = plt.colorbar(hb)
cb.set_label("log10(Number of cells)")

plt.xlabel("Temperature (K)")
plt.ylabel("|Temperature - mean| (K)")
plt.title(f"Temperature Deviation from Mean\n{chk.split('/')[-1]}")
plt.tight_layout()
plt.savefig("temperature_deviation_hexbin.png", dpi=200)
plt.close()

print("Saved temperature_deviation_hexbin.png")

