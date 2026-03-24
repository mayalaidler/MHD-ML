# si_NN_safe.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def load_npz_to_df(path):
    data = np.load(path)
    return pd.DataFrame({k: data[k] for k in data.files})

train_df = load_npz_to_df("siresults/train_data.npz")
test_df  = load_npz_to_df("siresults/test_data.npz")

feature_names = ["rho","temperature","velx","vely","velz","v_mag","B_mag"]

X_train = train_df[feature_names].values
y_train = train_df["si_mass_frac"].values

X_test = test_df[feature_names].values
y_test = test_df["si_mass_frac"].values

EPS = 1e-30

mask_train = np.isfinite(y_train) & (y_train > 0)
mask_test  = np.isfinite(y_test)  & (y_test > 0)

X_train, y_train = X_train[mask_train], y_train[mask_train]
X_test,  y_test  = X_test[mask_test],  y_test[mask_test]

y_train = np.log10(np.clip(y_train, EPS, None))
y_test  = np.log10(np.clip(y_test, EPS, None))

for idx in [0,1,6]:  # rho, T, B_mag
    X_train[:, idx] = np.log10(np.clip(X_train[:, idx], EPS, None))
    X_test[:, idx]  = np.log10(np.clip(X_test[:, idx], EPS, None))

# --- Force finite & clip extreme values ---
X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
X_test  = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=10.0, neginf=-10.0)
y_test  = np.nan_to_num(y_test, nan=0.0, posinf=10.0, neginf=-10.0)

# --- Scale targets to [-1,1] ---
scaler_y = MaxAbsScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1,1)).flatten()
y_test  = scaler_y.transform(y_test.reshape(-1,1)).flatten()

y_train = np.clip(y_train, -1, 1)
y_test  = np.clip(y_test, -1, 1)

# --- Scale features ---
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test  = scaler_X.transform(X_test)

X_train = np.clip(X_train, -5, 5)
X_test  = np.clip(X_test, -5, 5)

# --- Convert to float64 tensors (safe for CPU) ---
X_train = torch.from_numpy(X_train).double()
y_train = torch.from_numpy(y_train).double().view(-1,1)
X_test  = torch.from_numpy(X_test).double()
y_test  = torch.from_numpy(y_test).double().view(-1,1)

# --- Define small, safe network ---
class SiPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

model = SiPredictor(X_train.shape[1]).double()

# --- Training setup ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  

batch_size = 1024
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

num_epochs = 50

# --- Training loop ---
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.6f}")

# --- Evaluation ---
model.eval()
with torch.no_grad():
    preds = model(X_test)

y_true = y_test.numpy().flatten()
y_pred = preds.numpy().flatten()

print("Test R²:", r2_score(y_true, y_pred))

# --- Plots ---
plt.figure()
plt.scatter(y_true, y_pred, s=5, alpha=0.5)
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
plt.plot(lims, lims, 'r--')
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("NN: True vs Predicted")
plt.grid(alpha=0.3)
plt.savefig("NN_results/nn_true_vs_pred.png")

plt.figure()
plt.scatter(y_pred, y_true - y_pred, s=5, alpha=0.5)
plt.axhline(0, linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals")
plt.grid(alpha=0.3)
plt.savefig("NN_results/nn_residuals.png")