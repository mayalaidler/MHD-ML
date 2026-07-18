"""Feature engineering from raw FLASH variables."""

from __future__ import annotations

import numpy as np

EPS = 1e-30

#: FLASH variables needed to build the standard feature set.
REQUIRED_FIELDS = ["dens", "temp", "velx", "vely", "velz",
                   "magx", "magy", "magz"]

FEATURE_NAMES = ["log_rho", "log_T", "vx", "vy", "vz", "vmag",
                 "log_B", "mach", "log_beta"]


def valid_mask(fields: dict, target: np.ndarray) -> np.ndarray:
    """Cells where all features and the target are finite and positive
    where positivity is required."""
    B = _bmag(fields)
    m = (
        np.isfinite(fields["dens"]) & (fields["dens"] > 0)
        & np.isfinite(fields["temp"]) & (fields["temp"] > 0)
        & np.isfinite(fields["velx"]) & np.isfinite(fields["vely"])
        & np.isfinite(fields["velz"])
        & np.isfinite(B) & (B > 0)
        & np.isfinite(target) & (target > 0)
    )
    return m


def build_features(fields: dict) -> np.ndarray:
    """The 9 standard features: log rho, log T, velocities, |v|,
    log |B|, an isothermal Mach proxy, and log plasma beta.

    Rows correspond to the input arrays' rows; apply `valid_mask` first.
    """
    rho, T = fields["dens"], fields["temp"]
    vx, vy, vz = fields["velx"], fields["vely"], fields["velz"]
    B = _bmag(fields)

    vmag = np.sqrt(vx**2 + vy**2 + vz**2)
    mach = vmag / (np.sqrt(T) + EPS)
    plasma_beta = rho * T / (B**2 + EPS)

    return np.column_stack([
        np.log10(rho + EPS),
        np.log10(T + EPS),
        vx, vy, vz, vmag,
        np.log10(B + EPS),
        mach,
        np.log10(np.clip(plasma_beta, EPS, None)),
    ])


def _bmag(fields: dict) -> np.ndarray:
    return np.sqrt(fields["magx"]**2 + fields["magy"]**2
                   + fields["magz"]**2)
