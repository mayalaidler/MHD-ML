import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Define a simple neural network

device = "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#To train a model, we need a loss function and an optimizer.

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

#### CV data
import torchvision
data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True)

# preprocess the data into numpy arrays
images = data.data.numpy().astype(float)
targets = data.targets.numpy() # integer encoding of class labels
class_dict = {i:class_name for i,class_name in enumerate(data.classes)}
labels = np.array([class_dict[t] for t in targets]) # raw class labels
n = len(images)

import pandas as pd
import h5py

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

import yt
import pandas as pd
from pathlib import Path

base_dir = Path("/scratch/mlaidler/astr_thesis/mhd_1e8/1E25_S100_z01_mhd/Simulation")
rows = []

for run_dir in base_dir.iterdir():
    if not run_dir.is_dir():
        continue

    try:
        early_ds = yt.load(run_dir / f"ISM_hdf5_chk_0004")
        final_ds = yt.load(run_dir / f"ISM_hdf5_chk_0022")

        row = {}
        row.update(extract_early_features(early_ds))
        row.update(extract_final_targets(final_ds))

        row["run_name"] = run_dir.name
        rows.append(row)

        print(f"Processed {run_dir.name}")

    except Exception as e:
        print(f"Skipping {run_dir.name}: {e}")

df = pd.DataFrame(rows)

print(df.head())
print(df.describe())
print(df.isna().sum())

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

target = ['rho_max_final', 'T_mean_final', 'T_std_final']

xtrain_df = train_df.drop(columns=target)
ytrain_df = train_df[target]

xtest_df = test_df.drop(columns=target)
ytest_df = test_df[target]

import matplotlib.pyplot as plt

plt.scatter(df["rho_mean_early"], df["rho_max_final"])
plt.xlabel("Early Mean Density")
plt.ylabel("Final Max Density")
plt.show()

#Actually training the model

from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)

model.fit(xtrain_df, ytrain_df)
y_pred = model.predict(xtest_df)


from sklearn.metrics import mean_squared_error


mse = mean_squared_error(ytest_df, y_pred)

print(f"Mean Squared Error: {mse}")
