from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import build_dataset


build_dataset("/scratch/mlaidler/astr_thesis/mhd_1e8/1E25_S100_z01_mhd/Simulation/ISM_hdf5_chk_0017", out="datasets/flasht0017.npz")

data = np.load("datasets/flasht0017.npz")
data.head()


X = data["X"]
Y = data["Y"]

#creating test and train
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("x and y train shaps: ", X_train.shape, Y_train.shape)
print("x and y test shapes: ", X_test.shape, Y_test.shape)

#make sure there are no nans or infs
assert np.all(np.isfinite(X))
assert np.all(np.isfinite(Y))
print(X.shape, Y.shape)

print(data)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

print("fitting model")

model.fit(X_train,Y_train)

import matplotlib.pyplot as plt

print("Showing correlation exits...")
y_delta = np.log10(Y[:,0]) - X[:,0]

plt.scatter(X[:,0], y_delta, s=1, alpha=0.1)
plt.xlabel("log density at t")
plt.ylabel("density at t+dt")
plt.savefig("density_t_vs_tpdt.png", dpi=200)
plt.close()

#testing the model predictions
Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("MSE:", mse)
print("R²:", r2)

print("Y_test mean:", Y_test.mean(axis=0))
print("Y_test std :", Y_test.std(axis=0))


#feature by feature extraction 
names = ["density", "temperature", "Bx", "By", "Bz"]

for i, name in enumerate(names):
    mse_i = mean_squared_error(Y_test[:, i], Y_pred[:, i])
    r2_i = r2_score(Y_test[:, i], Y_pred[:, i])
    print(f"{name}: MSE={mse_i:.3e}, R2={r2_i:.3f}")

i = 0  # density
plt.figure()
plt.scatter(Y_test[:, i], Y_pred[:, i], s=1, alpha=0.3)
plt.plot(
    [Y_test[:, i].min(), Y_test[:, i].max()],
    [Y_test[:, i].min(), Y_test[:, i].max()],
    'r'
)

plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Density: True vs Predicted")
plt.savefig("true_vs_pred.png")

#true vs predicted for density in a cell
rho_true = Y_test[:Nx*Ny, 0].reshape(Nx, Ny)
rho_pred = Y_pred[:Nx*Ny, 0].reshape(Nx, Ny)

plt.figure()
plt.imshow(np.log10(rho_true))
plt.title("True Density")
plt.colorbar()

plt.figure()
plt.imshow(np.log10(rho_pred))
plt.title("Predicted Density")
plt.colorbar()
plt.show()
            

