#!/usr/bin/env python3
"""
Visualizations for the per-ion runs.

1. Single-checkpoint Ridge-vs-NN comparison figure per ion (reuses the panel
   styling from visualizations.py).
2. Multi-sim v2 LOSO parity panels per ion x model (ridge / hgb / nn), if the
   multi_sim_v2 outputs exist.

Usage:  python3 viz_ions.py
"""

import json
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from visualizations import make_panel, load_predictions, SIM_NAMES

ION_LABELS = {"SiI": "Si I", "SiII": "Si II", "SiIV": "Si IV"}

SINGLE_CHK_PATHS = {
    "SiI":  ("siresults_ridge_fixed/single_spatial_test.npz",
             "siresults_nn_fixed/nn_single_spatial_test.npz"),
    "SiII": ("results_SiII_ridge/single_spatial_test.npz",
             "results_SiII_nn/nn_single_spatial_test.npz"),
    "SiIV": ("results_SiIV_ridge/single_spatial_test.npz",
             "results_SiIV_nn/nn_single_spatial_test.npz"),
}


def plot_single_checkpoint_ion(tag, ridge_path, nn_path, out_dir="ion_plots"):
    os.makedirs(out_dir, exist_ok=True)
    label = ION_LABELS.get(tag, tag)

    y_true_r, y_pred_r = load_predictions(ridge_path)
    y_true_n, y_pred_n = load_predictions(nn_path)

    fig = plt.figure(figsize=(12, 5.4))
    gs = GridSpec(1, 2, figure=fig, left=0.08, right=0.98,
                  bottom=0.12, top=0.82, wspace=0.25)
    ax_ridge = fig.add_subplot(gs[0, 0])
    ax_nn    = fig.add_subplot(gs[0, 1])

    make_panel(ax_ridge, y_true_r, y_pred_r, "Ridge Regression")
    make_panel(ax_nn,    y_true_n, y_pred_n, "Neural Network")

    for ax in (ax_ridge, ax_nn):
        ax.set_xlabel(rf"True $\log_{{10}}$({label})", fontsize=12)
        ax.set_ylabel(rf"Predicted $\log_{{10}}$({label})", fontsize=12)

    fig.suptitle(
        f"{label} — Single-Checkpoint (spatial hold-out, chk 0006, "
        f"1E23_S100_z1)", fontsize=15, fontweight="600", y=0.98)

    base = os.path.join(out_dir, f"single_checkpoint_{tag}")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] {base}.png / .pdf")


def plot_multisim_v2_ion(tag, out_dir="multi_sim_v2/plots"):
    """3-panel LOSO parity figure (ridge / hgb / nn) for one ion."""
    ion_dir = os.path.join("multi_sim_v2", tag)
    if not os.path.isdir(ion_dir):
        print(f"[SKIP] {ion_dir} not found")
        return
    os.makedirs(out_dir, exist_ok=True)
    label = ION_LABELS.get(tag, tag)

    with open(os.path.join(ion_dir, "loso_summary.json")) as fh:
        summary = json.load(fh)

    models = [("ridge", "Ridge"), ("hgb", "Gradient Boosting"),
              ("nn", "Neural Network")]

    fig = plt.figure(figsize=(16.5, 5.4))
    gs = GridSpec(1, 3, figure=fig, left=0.05, right=0.99,
                  bottom=0.12, top=0.82, wspace=0.22)

    for col, (m, m_label) in enumerate(models):
        path = os.path.join(ion_dir, f"{m}_loso_preds.npz")
        with np.load(path) as d:
            y_true, y_pred = d["y_true"], d["y_pred"]
        ax = fig.add_subplot(gs[0, col])
        make_panel(ax, y_true, y_pred,
                   f"{m_label} (LOSO mean R²={summary[m]['mean_r2']:.3f})")
        ax.set_xlabel(rf"True $\log_{{10}}$({label})", fontsize=12)
        ax.set_ylabel(rf"Predicted $\log_{{10}}$({label})", fontsize=12)

    fig.suptitle(
        f"{label} — Multi-Simulation LOSO (cell-paired v2 dataset)",
        fontsize=15, fontweight="600", y=0.97)

    base = os.path.join(out_dir, f"loso_parity_{tag}")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] {base}.png / .pdf")


if __name__ == "__main__":
    for tag, (ridge_path, nn_path) in SINGLE_CHK_PATHS.items():
        if os.path.exists(ridge_path) and os.path.exists(nn_path):
            plot_single_checkpoint_ion(tag, ridge_path, nn_path)
        else:
            print(f"[SKIP] single-checkpoint {tag}: results not found yet")
    for tag in ION_LABELS:
        plot_multisim_v2_ion(tag)
