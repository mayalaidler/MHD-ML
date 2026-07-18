# Si Ionization-State ML Pipeline — Results Report (2026-07-13 → 07-18)

All artifacts referenced here live on jr.vassar.edu under
`/scratch/mlaidler/astr_thesis/ml/` ("ml/" below). Every number can be
recomputed from the saved `*.npz` prediction files without retraining.
Original scripts (pre-modification) are archived in `ml/backups_20260713/`.

## 1. Pipelines and what ran

| Pipeline | Script | Ions run | Outputs |
|---|---|---|---|
| Single-checkpoint, spatial hold-out | `si_model.py` (Ridge), `si_NN.py` (NN) | Si I, Si II, Si IV | `ml/results_*`, `ml/final_results/final_*` |
| Multi-checkpoint, temporal split | `multi_chkpoint_si_model.py` | Si II, Si IV | `ml/results_SiII.txt`, `ml/final_results/results_SiIV.txt` |
| Multi-simulation LOSO (fixed) | `multi_sim_v2.py` | Si I, Si II, Si IV | `ml/multi_sim_v2/` |
| Post-hoc metrics + hurdle models | `posthoc_metrics.py` | all | `ml/final_results/posthoc_metrics.json`, `*_hurdle_results.npy` |
| Figures | `viz_ions.py` | all | `ml/final_results/ion_plots/`, `ml/final_results/multi_sim_v2_plots/` |

Scripts changed only additively (env-var switches; defaults reproduce old
behavior exactly): `ION_FIELD` (target ion), `OUT_DIR_OVERRIDE`,
`DATASET_SAMPLE_SIZE`, `NN_MAX_TRAIN`, `TORCH_SEED` (NN now seeded),
`CELLS_PER_CHK` (multi-checkpoint RAM cap).

## 2. Headline numbers

### Single-checkpoint (1E23_S100_z1, chk 0006, median spatial split)

Harmonized runs: Ridge and NN share the *identical* 500k-cell sample,
identical train/test split; NN = mean ± std over torch seeds {42, 1000, 2000}.

| Ion | log₁₀ range (p5–p95) | Ridge R² | NN R² | NN RMSE (dex) |
|---|---|---|---|---|
| Si I  | −4.0 → −3.2 | 0.684 | 0.788 ± 0.013 | 0.22 |
| Si II | −5.9 → −3.3 | 0.878 | 0.873 ± 0.004 | 0.29 |
| Si IV | −27.8 → −12.0 | 0.439 | 0.577 ± 0.004 | 3.28 |

Reproducibility check: Ridge Si I rerun matched the stored R² = 0.7106
exactly (its config: 100k sample). The stored Si I NN R² = 0.903 did NOT
reproduce (rerun: 0.797) because the NN was previously unseeded; the
seeded harmonized ensemble gives 0.788 ± 0.013.

### Multi-simulation LOSO (v2 cell-paired dataset; 600k rows, 12 sims)

Median held-out-simulation R² (medians, not means — see §3):

| Ion | Ridge | Grad. Boosting | NN |
|---|---|---|---|
| Si I  | 0.754 | 0.879 | 0.846 |
| Si II | 0.680 | 0.799 | 0.766 |
| Si IV | 0.333 | 0.867 | 0.773 |

Old pipeline (new_multi_sim.py) scored mean R² ≈ −41 (Ridge) / −40 (NN):
its dataset paired features and targets from unrelated cells (two
independent random subsamples of two different checkpoints), so no
cell-level relation was learnable. Those numbers are invalid, not merely
worse.

Hurdle model (classifier: ion above numerical floor? + regressor on
above-floor cells), LOSO means: classifier F1 = 0.98 (Si I), 0.91 (Si II),
0.98 (Si IV). Above-floor regression works well for Si II (R² 0.75,
RMSE ≈ 1.7 dex); for Si I and Si IV the within-branch variance is small
relative to cross-sim error, so above-floor R² is low/negative while RMSE
stays 1–2 dex — quote RMSE there.

### Multi-checkpoint temporal split (1E23_S100_z01, train chks 0–11, test 12–14)

| Ion | Ridge train/test R² | MLP train/test R² |
|---|---|---|
| Si II | 0.73 / **−0.71** | 1.00 / **−13.2** |
| Si IV | 0.87 / **−4.18** | 0.94 / **−14.0** |

This pipeline fails by design for extrapolation: `checkpoint_idx` is a
feature but all test checkpoints lie beyond the training range, and
log_beta is an exact linear combination of log_rho, log_T, log_B
(coefficients blow up). Fix (not yet run): drop `checkpoint_idx` and/or
hold out middle checkpoints (interpolation). No Si I baseline exists —
the earlier Si I run never completed (empty `saved_models/`).

## 3. Findings that shape interpretation

1. **R² is variance-normalized.** Si I and Si II are predicted with
   nearly identical absolute accuracy (0.22 vs 0.29 dex) yet differ in
   R² because Si II spans twice the range. Always read R² next to RMSE.
2. **Every ion is floor-dominated somewhere.** Trace states sit at the
   chemistry solver's numerical floor (10⁻²⁰–10⁻³³ = noise): Si IV in
   cold boxes, Si I/Si II in the hot 1E26_S100 boxes (100% of cells at
   floor there). Fold-mean LOSO R² is destroyed by those two sims
   (e.g. −86 with RMSE only ~3 dex on floor noise) — a metric artifact,
   hence medians + the hurdle decomposition.
3. **Transfer quality tracks parameter-space coverage.** The only 1E24
   sim scores R² 0.1–0.5 held-out (no density sibling in training);
   well-covered regimes score 0.8–0.99. Maya's own unique
   `1E25_S100_z01_mhd` sim (258 GB, in her scratch — NOT in ebuie's
   suite) could fill a coverage gap if added.
4. **Sim 6 (1E25_S100_z1) has B ≡ 0** despite the `_mhd` name —
   excluded by the B > 0 mask (also silently absent from the old
   dataset). v2 recovered sims 2, 11, 12 (old code hardcoded chk 12,
   which they lack) → 12/13 sims vs 9/13.
5. **NN seed variance:** unseeded historical runs varied by ΔR² ≈ 0.1;
   seeded harmonized ensembles vary by ~0.01. All scripts now seeded.

## 4. Paper changes required

- **Replace all multi-sim numbers** with v2 LOSO results + a sentence on
  the pairing fix. Old values are invalid.
- **Replace the single Si I NN value (0.903)** with the seeded ensemble
  0.79 ± 0.01, and state Ridge results are exactly reproducible.
- **Use harmonized same-test-set runs** for any Ridge-vs-NN comparison;
  quote LOSO medians with the floor-sim explanation, or hurdle metrics.
- **Report RMSE (dex) beside every R²**; add the variance-normalization
  caveat when comparing ions.
- **Multi-checkpoint section:** reframe as a negative result about
  temporal extrapolation with a time feature, or rerun with the
  interpolation split before quoting.
- **New material available:** ion-dependence hypothesis (dominant-state
  vs trace-state predictability), hurdle architecture, coverage effect,
  per-sim diagnosis table, publication figures in
  `ml/final_results/{ion_plots,multi_sim_v2_plots}/`.

## 5. Open-source package

`ionpred` scaffold on Maya's Mac at `~/Astro/ionpred/` (pip-installable,
CLI, hurdle model, auto floor detection, leakage-aware splits,
provenance metadata, 13 passing tests). Git repo initialized, nothing
committed — review then commit. Name is a placeholder.
