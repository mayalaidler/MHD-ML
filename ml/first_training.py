import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cpu"
print(f"Using {device} device")
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#Step up from ML class 
# @title Setup: RUN THIS FIRST
# run this first
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

import sklearn as sk

# suppress warnings
pd.options.mode.chained_assignment = None
pd.options.mode.copy_on_write = True

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # SVM
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn import metrics # check model accuracy
from sklearn.model_selection import train_test_split  # split data into train & test
import yt 

file_path = '/scratch/mlaidler/astr_thesis/mhd_1e8/1E25_S100_z01_mhd/Simulation/ISM_hdf5_chk_0004'

# Initialize an empty dictionary -> convert this simualtion data into a pandas DataFrame
data_dict = {}

with h5py.File(file_path, 'r') as f:
    for key in f.keys():
        print(f"Reading dataset: {key}")
        dataset_array = f[key][()] 
        data_dict[key] = dataset_array

# Convert the dictionary to a pandas DataFrame
# df = pd.DataFrame.from_dict(data_dict)
# print(df)

# from sklearn.model_selection import train_test_split
# train_df, test_df = train_test_split(df, test_size=0.2)
# print(f"Train DataFrame shape: {train_df.shape}")
# print(f"Test DataFrame shape: {test_df.shape}")



# print(ds.field_list)        
# print(ds.derived_field_list)

#Thermodynamic outcome

# final mean temperature
# final temperature std dev
# Structural outcome
# final max density mass above density threshold (star-forming proxy)
# Dynamical outcome
# final turbulent kinetic energy final magnetic energy You can either: 
# predict each separately, or do multi-output regression


import numpy as np

def extract_early_features(ds):
    ad = ds.all_data()

    # Density
    rho = ad[("gas", "density")]
    rho_mean = rho.mean().to("g/cm**3").value
    rho_std  = rho.std().to("g/cm**3").value
    rho_max  = rho.max().to("g/cm**3").value

    # Temperature
    T = ad[("gas", "temperature")]
    T_mean = T.mean().value
    T_std  = T.std().value

    # Velocity RMS
    vx = ad[("gas", "velocity_x")]
    vy = ad[("gas", "velocity_y")]
    vz = ad[("gas", "velocity_z")]

    v_rms = ((vx**2 + vy**2 + vz**2).mean())**0.5
    v_rms = v_rms.to("km/s").value

    # Magnetic field
    if ("flash", "mag_strength") in ds.field_list:
        B = ad[("flash", "mag_strength")]
    else:
        Bx = ad[("flash", "magx")]
        By = ad[("flash", "magy")]
        Bz = ad[("flash", "magz")]
        B = (Bx**2 + By**2 + Bz**2)**0.5

    B_mean = B.mean().to("gauss").value
    B_std  = B.std().to("gauss").value

    # Plasma beta
    P_gas = ad[("gas", "pressure")]
    P_mag = B**2 / (8 * np.pi)
    beta = (P_gas / P_mag).mean().value

    return {
        "rho_mean_early": rho_mean,
        "rho_std_early": rho_std,
        "rho_max_early": rho_max,
        "T_mean_early": T_mean,
        "T_std_early": T_std,
        "v_rms_early": v_rms,
        "B_mean_early": B_mean,
        "B_std_early": B_std,
        "beta_early": beta,
    }

def extract_final_targets(ds):
    ad = ds.all_data()

    rho = ad[("gas", "density")]
    T   = ad[("gas", "temperature")]

    return {
        "rho_max_final": rho.max().to("g/cm**3").value,
        "T_mean_final":  T.mean().value,
        "T_std_final":   T.std().value,
    }

from pathlib import Path

base_dir = "/scratch/mlaidler/astr_thesis/mhd_1e8/1E25_S100_z01_mhd/Simulation/"
rows = []

chk4_files = base_dir + "ISM_hdf5_chk_0004"
chk22_files = base_dir + "ISM_hdf5_chk_0017"

# make sure I have files

if not chk4_files or not chk22_files:
    print("Checkpoint files not found!")

else:
    early_ds = yt.load(chk4_files)
    final_ds = yt.load(chk22_files)

    row = {}

    # Extract numeric fields from early_ds
    for field in early_ds.field_list:
        row[f"early_{field[1]}"] = early_ds.all_data()[field].mean()
            # Extract numeric fields from final_ds
    for field in final_ds.field_list:
        row[f"final_{field[1]}"] = final_ds.all_data()[field].mean()

            # Keep track of run
        s = len(base_dir)
        rows.append(row)
        print(f"Processed {chk4_files[s:]}")

df = pd.DataFrame(rows)

print("num of rows :", len(rows))
print("rows: ", rows[:2])
print("columns: ", df.columns)
print("df.head(): ", df.head())
print(df.isna().sum())

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

target = ['final_temp', 'final_velx']

# Split features / targets
xtrain_df = train_df.drop(columns=target)
ytrain_df = train_df[target]

xtest_df = test_df.drop(columns=target)
ytest_df = test_df[target]

# 🔑 DROP NON-NUMERIC COLUMNS HERE
xtrain_df = xtrain_df.select_dtypes(include='number')
xtest_df  = xtest_df.select_dtypes(include='number')

import matplotlib.pyplot as plt

plt.scatter(df["early_temp"], df["final_temp"])
plt.xlabel("Early Mean Density")
plt.ylabel("Final Max Density")
plt.show()

#Actually training the model

from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
print("started model: ridge")
model.fit(xtrain_df, ytrain_df)
print("fitting...next")
y_pred = model.predict(xtest_df)


from sklearn.metrics import mean_squared_error


mse = mean_squared_error(ytest_df, y_pred)

print(f"Mean Squared Error: {mse}")