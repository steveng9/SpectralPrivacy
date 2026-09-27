# SpectralPrivacy

Is privacy leakage in synthetic data smooth or rough over the geometry of the records?
First experiment: `spectral_memorization_experiment_plan.md` (MAMA-MIA on SNAKE, fixed pool, graph Laplacian spectra).
Background: `research-vision-geometry-of-privacy.md`.

## Layout

| file | what |
|---|---|
| `spectral_core.py` | kNN graph, normalized Laplacian, GFT, roughness, split-half cross-spectrum (plan appendix + noise-robust helpers) |
| `mamamia_bridge.py` | imports the MAMA-MIA code from `../SyntheticData_MIA` **without modifying it** (patches its hard-coded Mac `DATA_DIR`) |
| `pool_experiment.py` | Step 0, "mode S": focal-point shadow modelling, then R runs per (SDG, ε) over a fixed 500-record pool |
| `analysis.py` | Steps 1–3: per-record signals, graphs, cross-spectra, permutation null, density baseline (H0), run bootstrap for H1 |
| `figures.py` | Step 4: F1–F4 |
| `workspace/` | pool ids, focal points, per-run results (`results/pool/{sdg}_e{eps}/run_XXX.npz` + `_synth.parquet`) |
| `results/` | `summary_{metric}_k{k}.csv`, `figures/` |
| `logs/` | run logs |

## Running

```bash
PY=~/miniconda3/envs/mamamia_/bin/python        # py3.10; PrivBayes .so is built for 3.10
$PY pool_experiment.py pool_ids                  # once (seeded; already done)
$PY pool_experiment.py run --sdgs mst priv --runs 64 --workers 16   # resumable
$PY analysis.py --figures                        # hamming, k=10
$PY analysis.py --metric gower --k 20 --figures  # robustness
```

Workers are single-threaded (`OMP/MKL/OPENBLAS=1`, JAX on CPU), so `--workers N` ≈ N cores.

## Choices and deviations from the plan

- **Focal points regenerated.** The `FP4_*` files were not on this server. They were re-created with the same procedure
  (15 shadow fits on n=1000 aux samples per SDG×ε), stored in `workspace/experiment_artifacts/focalpoints/`.
- **Private-GSD not run yet.** MAMA-MIA-GSD depends on a locally modified `private_gsd` (`fit_dp(..., only_determine_fps=True)`
  returning query ids). The submodule is empty here and the pinned upstream commit `a640d54` lacks that change.
- **Paired design.** One pool (seed 20260925) and one membership mask per run index are shared by every (SDG, ε),
  so differences across ε are not confounded by pool or mask draws.
- **Score used for per-record signals:** per run, z-scored log raw score (the input to `activate_3`), which removes
  run-level shifts (synthetic-data quality varies run to run) that would otherwise be common-mode noise.
- **Third attack, DCR:** nearest-neighbour Gower distance to a reference aux sample minus that to the synthetic data.
  Added because KDE (DOMIAS) is near chance here (AUC ≈ 0.52), too weak for the attack-independence control, and
  because MAMA-MIA is built from marginals and may be smooth on the attribute graph by construction.
- **Graph distance:** Hamming over the 15 raw columns (= fraction of attributes that differ) as primary; Gower
  (ordinal columns range-normalized) as robustness. One-hot + Hamming would double-count nominal mismatches.
- **Density baseline (H0):** v regressed on both within-pool and population (20k aux sample) mean 10-NN distance;
  residuals computed separately per half so the cross-spectrum remains noise-cancelling.
- **Roughness:** besides `R(v)` we report the cross-roughness `R×(v) = v_aᵀ L v_b / v_aᵀ v_b`, which is unbiased by
  white estimation noise (naive `R` of a noisy v drifts toward 1, the "estimation-noise trap").

## TabDDPM follow-ups (env `recon_`, GPU)

- `tabddpm_experiment.py` — existing 5 disjoint TabDDPM releases per dataset in `~/data/reconstruction_data`
  (adult, cdc, california); DCR + MeLoMIA-ND (from `~/MIA_on_bulkRNAseq_CAMDA2026`) → `results/tabddpm/`.
  Caveat: one in-release per record, so per-record memorization is confounded with release noise.
- `tabddpm_pool.py run --runs 32 --concurrent 8 --gpu 1` — fixed-pool design on adult: 1,000 pool records, each
  release trains on 500 pool members + 500 filler records (≈58 min per wave of 8) → `workspace/tabddpm_pool/`.
- `tabddpm_pool_analysis.py features --gpu 1` then `analyze --k {5,10,20}` → `results/tabddpm_pool/`.
  Result: per-record v is reliable (split-half r 0.82 DCR, 0.67 MeLoMIA); density explains 23–53% of it; the
  density residual is reliable (r½ ≈ 0.55–0.65) but ≈ white on the Gower graph (R× 0.91–0.99, high-band share at
  null) — idiosyncratic per-record leakage, not high-frequency. Open: repeat with a model-embedding graph.
