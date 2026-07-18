import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.image import imread
import yt

BASE_DIR    = "/scratch/ebuie/ISO_Turb/midway/mhd_1e8/1E23_S100_z1_mhd/"
OUT_DIR     = "multi_simulation/plots"
FIELDS      = [('flash', 'si  ')]
ZLIM        = (1e-10, 1e-2)          # fixed color scale across all frames
CMAP        = "viridis"
AXES_UNIT   = "pc"
FPS         = 3                       # frames per second in the GIF
DPI         = 150


def _discover_checkpoints(base_dir: str) -> list[str]:
    """Return all ISM checkpoint paths in base_dir, sorted by index."""
    pattern = os.path.join(base_dir, "ISM_hdf5_chk_*")
    paths   = sorted(glob.glob(pattern))          # lexicographic == numeric here
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints matching {pattern}"
        )
    return paths

def _render_frame(ds, field: tuple, temp_path: str, chk_index: int) -> None:
    """
    Render a single yt slice plot for *field* into *temp_path*.
    Whitespace is minimised: the colorbar is kept but the figure is
    cropped tightly around the actual plot area.
    """
    slc = yt.SlicePlot(ds, "z", field)
    slc.set_cmap(field, CMAP)
    slc.set_zlim(field, *ZLIM)
    slc.set_log(field, True)
    slc.set_axes_unit(AXES_UNIT)

    # pull simulation time for the annotation
    t_myr = float(ds.current_time.in_units("Myr"))

    slc.render()
    fig = slc.plots[field].figure

    # ── strip yt's outer whitespace ──────────────────────────────────────
    # yt places the plot inside axes[0]; shrink the figure to just that.
    ax = fig.axes[0]
    ax.set_title(
        f"chk {chk_index:04d}   |   t = {t_myr:.3f} Myr\n"
        f"field: {field[1].strip()}   |   sim: {os.path.basename(BASE_DIR.rstrip('/'))}",
        fontsize=9, pad=4
    )

    # tight_layout collapses padding; constrained_layout already disabled by yt
    fig.tight_layout(pad=0.5)
    fig.savefig(temp_path, dpi=DPI, bbox_inches="tight")
    plt.close("all")

def _frames_to_gif(frame_paths: list[str], gif_path: str) -> None:
    """Assemble a list of PNG paths into an animated GIF."""
    if not frame_paths:
        print("  [WARN] no frames to assemble.")
        return

    # read all frames
    imgs = [imread(p) for p in frame_paths]

    # build figure sized to the first frame
    h, w = imgs[0].shape[:2]
    fig, ax = plt.subplots(figsize=(w / DPI, h / DPI), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")

    im = ax.imshow(imgs[0])

    def _update(frame_idx):
        im.set_data(imgs[frame_idx])
        return (im,)

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(imgs),
        interval=int(1000 / FPS),
        blit=True,
    )

    writer = animation.PillowWriter(fps=FPS)
    ani.save(gif_path, writer=writer)
    plt.close(fig)
    print(f"  → saved: {gif_path}")

def comparelsitosi_gif() -> None:
    """
    For each field in FIELDS, iterate over all available checkpoints,
    render a tight slice plot per frame, then assemble into a GIF for my slides.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    temp_dir = os.path.join(OUT_DIR, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    checkpoints = _discover_checkpoints(BASE_DIR)
    print(f"Found {len(checkpoints)} checkpoints in {BASE_DIR}")

    for field in FIELDS:
        field_tag  = field[1].strip()            # "si" or "lsi"
        gif_path   = os.path.join(OUT_DIR, f"{field_tag}_evolution.gif")
        frame_paths = []

        print(f"\nRendering field '{field_tag}' ...")

        for chk_path in checkpoints:
            # parse the checkpoint index from the filename (last 4 chars before extension)
            chk_index = int(chk_path.split("_")[-1])
            temp_path = os.path.join(temp_dir, f"{field_tag}_chk{chk_index:04d}.png")

            print(f"  [{chk_index:04d}] {os.path.basename(chk_path)}", end=" ... ")

            try:
                ds = yt.load(chk_path)
                _render_frame(ds, field, temp_path, chk_index)
                frame_paths.append(temp_path)
                print("ok")
            except Exception as e:
                print(f"SKIPPED ({e})")

        print(f"Assembling {len(frame_paths)} frames into GIF ...")
        _frames_to_gif(frame_paths, gif_path)

    print("\nDone. GIFs written to:", OUT_DIR)
#PLOTS for NN and Ridge for multi_sim

# file: plot_scatter.py
#
# Produces: multi_simulation/plots/scatter_pred_vs_true.pdf  
#           multi_simulation/plots/scatter_pred_vs_true.png  
# Layout: 1 row × 2 panels
#   Left  — Ridge regression, LOSO held-out predictions
#   Right — Neural network,   LOSO held-out predictions
#
# Each point is one simulation cell.
# Diagonal = perfect prediction. Marginal histograms show coverage.

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

matplotlib.rcParams.update({
    "font.family"      : "sans-serif",
    "font.size"        : 9,
    "axes.linewidth"   : 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction"  : "in",
    "ytick.direction"  : "in",
    "pdf.fonttype"     : 42,   
    "ps.fonttype"      : 42,
})

SIM_NAMES = [
    "1E23_S100_z01", "1E23_S100_z1",
    "1E23_S30_z01",  "1E23_S30_z1",
    "1E23_S60_z01",  "1E24_S100_z1",
    "1E25_S100_z1",  "1E25_S30_z01",
    "1E25_S30_z1",   "1E26_S100_z01",
    "1E26_S100_z1",  "1E26_S30_z01",
    "1E26_S30_z1",
]

OUT_DIR = "multi_simulation/plots"

def _load_preds(path):
    with np.load(path) as d:
        return d["y_true"].copy(), d["y_pred"].copy(), d["sim_ids"].copy()


def _panel_stats(y_true, y_pred):
    """Return R², RMSE, and Pearson r for annotation."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    #rmse   = np.sqrt(np.mean((y_true - y_pred) ** 2))
    #r, _   = pearsonr(y_true, y_pred)
    return r2


def _draw_panel(ax, y_true, y_pred, sim_ids, cmap, norm, title, alpha=0.25):
    """
    Draw one scatter panel with:
      - per-simulation colored points (thin, semi-transparent for density)
      - grey identity line
      - shaded ±1 dex band around identity
    """
    # ── identity line and ±1 dex band ────────────────────────────────────
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    diag = np.array([lo, hi])

    ax.fill_between(diag, diag - 1, diag + 1,
                    color="gray", alpha=0.07, linewidth=0, zorder=0)
    ax.plot(diag, diag, color="gray", linewidth=0.8,
            linestyle="--", zorder=1, label="y = x")

    # ── scatter, one sim at a time so legend works cleanly ───────────────
    unique = np.unique(sim_ids)
    for sid in unique:
        mask  = sim_ids == sid
        color = cmap(norm(sid))
        label = SIM_NAMES[sid] if sid < len(SIM_NAMES) else str(sid)
        ax.scatter(
            y_true[mask], y_pred[mask],
            c=[color], s=4, alpha=alpha,
            linewidths=0, rasterized=True,   # rasterize for small PDF size
            label=label, zorder=2,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\log_{10}(\mathrm{Si}_\mathrm{true})$", fontsize=9)
    ax.set_ylabel(r"$\log_{10}(\mathrm{Si}_\mathrm{pred})$", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="normal", pad=6)
    r2 = _panel_stats(y_true, y_pred)
    stats_str   = (
        f"$R^2 = {r2:.3f}$\n"
    )
    ax.text(
        0.04, 0.96, stats_str,
        transform=ax.transAxes,
        fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="lightgray", linewidth=0.5, alpha=0.9),
    )
    ax.text(0.97, 0.03, f"N = {len(y_true):,}",
            transform=ax.transAxes, fontsize=7,
            ha="right", va="bottom", color="gray")

    return ax


def plot_scatter(
    ridge_path = "multi_simulation/ridge_loso_preds.npz",
    nn_path    = "multi_simulation/nn_loso_preds.npz",
    out_dir    = OUT_DIR,
    point_alpha= 0.20,):
    os.makedirs(out_dir, exist_ok=True)

    y_true_r, y_pred_r, sids_r = _load_preds(ridge_path)
    y_true_n, y_pred_n, sids_n = _load_preds(nn_path)

    n_sims = 9
    cmap   = matplotlib.colormaps["tab20"].resampled(n_sims)
    norm   = matplotlib.colors.Normalize(vmin=0, vmax=n_sims - 1)

    # Two equal scatter panels side by side, shared legend below.
    fig = plt.figure(figsize=(7.0, 3.8))
    gs  = gridspec.GridSpec(
        1, 2,
        figure=fig,
        left=0.09, right=0.97,
        bottom=0.22, top=0.93,
        wspace=0.30,
    )

    ax_r = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[0, 1])

    _draw_panel(ax_r, y_true_r, y_pred_r, sids_r, cmap, norm,
                "Ridge regression (LOSO)", alpha=point_alpha)
    _draw_panel(ax_n, y_true_n, y_pred_n, sids_n, cmap, norm,
                "Neural network (LOSO)",   alpha=point_alpha)

    # ── shared legend — one entry per simulation ──────────────────────────
    legend_handles = []
    for sid in range(n_sims):
        color = cmap(norm(sid))
        label = SIM_NAMES[sid] if sid < len(SIM_NAMES) else str(sid)
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=color, markersize=5, label=label)
        )
    # Add the identity line entry
    legend_handles.append(
        Line2D([0], [0], color="gray", linewidth=0.8,
               linestyle="--", label="y = x")
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        fontsize=7,
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
        bbox_to_anchor=(0.53, 0.01),
        columnspacing=0.8,
        handletextpad=0.4,
    )

    base = os.path.join(out_dir, "NNandRidge_MultiSim_scatter_pred_vs_true")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Scatter plot saved to {base}.pdf / .png")
    

# file: plot_single_checkpoint_presentation.py
#
# Output:
#   siresults_presentation/single_checkpoint_comparison.pdf (vector, editable)
#   siresults_presentation/single_checkpoint_comparison.png (raster, slides)
#
# Layout: 1 row × 2 columns
#   Left:  Ridge regression
#   Right: Neural network
#
# Each panel shows:
#   - Hex-binned density scatter (handles 100k+ points cleanly)
#   - Identity line (perfect prediction)
#   - ±1 dex error band (gray shaded)
#   - Stats box (R², RMSE, Pearson r)
#   - Marginal histograms on axes

# Presentation-quality settings
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 11,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUT_DIR = "siresults_presentation"


def load_predictions(path):
    """Load y_true and y_pred from .npz file."""
    with np.load(path) as data:
        return data["y_true"].copy(), data["y_pred"].copy()


def compute_stats(y_true, y_pred):
    """Compute R², RMSE, ."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    #rmse   = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Fraction of predictions 
    abs_error = np.abs(y_true - y_pred)
    
    return r2


def make_panel(ax, y_true, y_pred, title, color="#1f77b4"):
    """
    Draw one comparison panel with hex-binned density scatter.
    
    Parameters
    ----------
    ax     : matplotlib axes
    y_true : true log10(Si) values
    y_pred : predicted log10(Si) values
    title  : panel title (e.g., "Ridge Regression")
    color  : primary color for hexbins
    """
    # Compute stats
    r2 = compute_stats(y_true, y_pred)
    
    # Axis limits
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    diag = np.array([lo, hi])
    
    # ── Background: ±1 dex error band ────────────────────────────────────
    ax.fill_between(diag, diag - 1, diag + 1,
                    color="#EEEEEE", linewidth=0, zorder=0,
                    label="±1 dex")
    
    # ── Identity line ─────────────────────────────────────────────────────
    ax.plot(diag, diag, color="#444444", linewidth=1.5,
            linestyle="--", zorder=1, label="Perfect prediction")
    
    # ── Hex-binned density scatter ────────────────────────────────────────
    # Hexbins handle large datasets (100k+ points) cleanly without 
    # overwhelming the plot or bloating the PDF file size.
    hexbin = ax.hexbin(
        y_true, y_pred,
        gridsize=50,
        cmap="Blues",
        mincnt=1,
        linewidths=0.2,
        edgecolors="face",
        zorder=2,
        alpha=0.8,
    )
    
    # ── Axis formatting ───────────────────────────────────────────────────
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"True $\log_{10}(\mathrm{Si})$", fontsize=12)
    ax.set_ylabel(r"Predicted $\log_{10}(\mathrm{Si})$", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="600", pad=10)
    
    # ── Stats box (top-left corner) ───────────────────────────────────────
    stats_text = (
        f"$R^2 = {r2:.3f}$\n"
       
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#CCCCCC",
            linewidth=1.0,
            alpha=0.95,
        ),
        zorder=10,
    )
    
    # ── Sample count (bottom-right) ───────────────────────────────────────
    ax.text(
        0.97, 0.03,
        f"$N = {len(y_true):,}$",
        transform=ax.transAxes,
        fontsize=9,
        ha="right", va="bottom",
        color="#666666",
        zorder=10,
    )
    
    # Grid for readability
    ax.grid(True, alpha=0.2, linewidth=0.5, zorder=0)
    
    return hexbin


def plot_single_checkpoint_comparison(
    ridge_path = "siresults_ridge_fixed/single_spatial_test.npz",
    nn_path    = "siresults_nn_fixed/nn_single_spatial_test.npz",
    out_dir    = OUT_DIR,
):
    """
    Create side-by-side comparison of Ridge vs NN for single-checkpoint.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    print("[INFO] Loading Ridge predictions...")
    y_true_ridge, y_pred_ridge = load_predictions(ridge_path)
    
    print("[INFO] Loading NN predictions...")
    y_true_nn, y_pred_nn = load_predictions(nn_path)
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 2, figure=fig, left=0.08, right=0.98,
                  bottom=0.12, top=0.90, wspace=0.25)
    
    ax_ridge = fig.add_subplot(gs[0, 0])
    ax_nn    = fig.add_subplot(gs[0, 1])
    
    # Draw panels
    print("[INFO] Rendering Ridge panel...")
    make_panel(ax_ridge, y_true_ridge, y_pred_ridge,
               "Ridge Regression (Spatial Hold-Out)", color="#2E86AB")
    
    print("[INFO] Rendering NN panel...")
    make_panel(ax_nn, y_true_nn, y_pred_nn,
               "Neural Network (Spatial Hold-Out)", color="#A23B72")
    
    # Super-title
    fig.suptitle(
        "Single-Checkpoint Training: Ridge vs Neural Network",
        fontsize=16, fontweight="600", y=0.97,
    )
    
    # Subtitle
    fig.text(
        0.5, 0.93,
        "Train on checkpoint 0004 (x < median), test on checkpoint 0004 (x ≥ median)",
        ha="center", fontsize=10, color="#555555",
    )
    
    # Save
    base = os.path.join(out_dir, "single_checkpoint_comparison")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n[SUCCESS] Saved to:")
    print(f"  {base}.pdf  (vector, editable)")
    print(f"  {base}.png  (raster, for slides)")


# ══════════════════════════════════════════════════════════════════════════════
# BONUS: 3-panel version with residual analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_single_checkpoint_detailed(
    ridge_path = "siresults_ridge_fixed/single_spatial_test.npz",
    nn_path    = "siresults_nn_fixed/nn_single_spatial_test.npz",
    out_dir    = OUT_DIR,
):
    """
    More detailed 2×3 layout:
      Row 1: Ridge scatter | NN scatter | Comparison bar chart
      Row 2: Ridge residuals | NN residuals | Error distribution
    """
    os.makedirs(out_dir, exist_ok=True)
    
    y_true_r, y_pred_r = load_predictions(ridge_path)
    y_true_n, y_pred_n = load_predictions(nn_path)
    
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 3, figure=fig, left=0.06, right=0.98,
                  bottom=0.06, top=0.94, wspace=0.30, hspace=0.35)
    
    # ── Row 1: Scatter plots ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    make_panel(ax1, y_true_r, y_pred_r, "Ridge Regression")
    make_panel(ax2, y_true_n, y_pred_n, "Neural Network")
    
    # Bar chart comparing metrics
    r2_r = compute_stats(y_true_r, y_pred_r)
    r2_n = compute_stats(y_true_n, y_pred_n)
    
    metrics = ["R²"]
    ridge_vals = [r2_r]
    nn_vals    = [r2_n]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, ridge_vals, width, label="Ridge", color="#2E86AB", alpha=0.8)
    ax3.bar(x + width/2, nn_vals, width, label="NN", color="#A23B72", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=10)
    ax3.set_ylabel("Value", fontsize=11)
    ax3.set_title("Performance Comparison", fontsize=14, fontweight="600", pad=10)
    ax3.legend(fontsize=10, framealpha=0.9)
    ax3.grid(axis="y", alpha=0.3)
    
    # ── Row 2: Residual analysis ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Ridge residuals vs true
    residuals_r = y_pred_r - y_true_r
    ax4.hexbin(y_true_r, residuals_r, gridsize=50, cmap="RdBu_r",
               mincnt=1, vmin=-2, vmax=2, linewidths=0.2, alpha=0.8)
    ax4.axhline(0, color="black", linewidth=1.5, linestyle="--")
    ax4.axhline(1, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax4.axhline(-1, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax4.set_xlabel(r"True $\log_{10}(\mathrm{Si})$", fontsize=11)
    ax4.set_ylabel("Residual (pred − true)", fontsize=11)
    ax4.set_title("Ridge Residuals", fontsize=12, fontweight="600")
    ax4.grid(alpha=0.2)
    
    # NN residuals vs true
    residuals_n = y_pred_n - y_true_n
    ax5.hexbin(y_true_n, residuals_n, gridsize=50, cmap="RdBu_r",
               mincnt=1, vmin=-2, vmax=2, linewidths=0.2, alpha=0.8)
    ax5.axhline(0, color="black", linewidth=1.5, linestyle="--")
    ax5.axhline(1, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax5.axhline(-1, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax5.set_xlabel(r"True $\log_{10}(\mathrm{Si})$", fontsize=11)
    ax5.set_ylabel("Residual (pred − true)", fontsize=11)
    ax5.set_title("NN Residuals", fontsize=12, fontweight="600")
    ax5.grid(alpha=0.2)
    
    # Overlapping residual histograms
    ax6.hist(residuals_r, bins=50, alpha=0.6, label="Ridge", color="#2E86AB", density=True)
    ax6.hist(residuals_n, bins=50, alpha=0.6, label="NN", color="#A23B72", density=True)
    ax6.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax6.set_xlabel("Residual (dex)", fontsize=11)
    ax6.set_ylabel("Density", fontsize=11)
    ax6.set_title("Error Distribution", fontsize=12, fontweight="600")
    ax6.legend(fontsize=10, framealpha=0.9)
    ax6.grid(alpha=0.2)
    
    # Save
    base = os.path.join(out_dir, "single_checkpoint_detailed")
    fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n[SUCCESS] Detailed figure saved to:")
    print(f"  {base}.pdf")
    print(f"  {base}.png")

if __name__ == "__main__":
    plot_scatter()
