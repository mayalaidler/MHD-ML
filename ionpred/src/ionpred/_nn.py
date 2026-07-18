"""Torch MLP regressor with a sklearn-style fit/predict interface."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class TorchRegressor:
    """256-128-64 LeakyReLU MLP on standardized features, trained on
    log-space targets with Adam, LR plateau decay, and early stopping.

    Deterministic for a fixed seed.  Run several seeds and report the
    spread: single unseeded runs of this architecture varied by
    ΔR² ≈ 0.1 in validation.
    """

    def __init__(self, seed: int = 42, max_epochs: int = 100,
                 batch_size: int = 1024, lr: float = 5e-4,
                 weight_decay: float = 1e-4, patience: int = 15):
        self.seed = seed
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.scaler_: StandardScaler | None = None
        self.model_: nn.Module | None = None

    def _build(self, input_dim: int) -> nn.Module:
        torch.manual_seed(self.seed)
        model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.01),
            nn.Linear(256, 128), nn.LeakyReLU(0.01),
            nn.Linear(128, 64), nn.LeakyReLU(0.01),
            nn.Linear(64, 1),
        )
        for m in model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        return model.double()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        Xt = torch.from_numpy(Xs)
        yt = torch.from_numpy(np.asarray(y, dtype=np.float64)).view(-1, 1)

        self.model_ = self._build(Xt.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr,
            weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-6)
        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.seed))

        best, stale = float("inf"), 0
        for _ in range(self.max_epochs):
            self.model_.train()
            total = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
            avg = total / len(loader)
            scheduler.step(avg)
            if avg < best:
                best, stale = avg, 0
            else:
                stale += 1
            if stale >= self.patience:
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.from_numpy(Xs)).numpy().ravel()
