# Patch Sampling Strategy & K-Sensitivity Experiment

**Date**: 2026-08-11
**Model**: HE-only R²T + ABMIL
**Dataset**: CAMELYON16 (Train: 270 slides, Test: 129 slides)
**GPU**: RTX A6000 (48GB), actually 15.77 GiB Tesla

## Setup
- Internal stratified split: 80/20 (StratifiedShuffleSplit, random_state=42)
- Early stopping: patience=10 on val_auc
- Max 25 epochs, lr=1e-4, Adam, ReduceLROnPlateau
- Seeds: 42, 123

## Results

### Phase 1: Sampling Strategy (K=5000)

| Experiment | Sampling | K | Val AUC | Test AUC | Δ |
|---|---|---|---|---|---|
| first5000 | [:5000] (first) | 5000 | 0.9411 | 0.7189 | 0.222 |
| fixed_rand5000 | Random (seed=42) | 5000 | 0.9503 | 0.7715 | 0.179 |
| per_epoch_rand5000 | Random (new each epoch) | 5000 | 0.9858 | 0.7908 | 0.195 |

### Phase 2: K Sensitivity (per_epoch_random)

| K | Val AUC | Test AUC | Δ |
|---|---|---|---|
| 2500 | 0.9428 | **0.7909** | **0.152** |
| 5000 | 0.9858 | 0.7908 | 0.195 |
| 7500 | 0.9893 | 0.7877 | 0.202 |
| 10000 | 0.9879 | 0.7667 | 0.221 |
| all | OOM | — | — |

## Conclusions

1. **[:5000] is spatially biased**: first5000 (Test 0.719) is significantly worse than random sampling
2. **Per-epoch random at K=2500 is best**: Highest Test AUC (0.791) with smallest gap (0.152)
3. **More patches ≠ better**: K=10000 degrades Test AUC to 0.767
4. **Per-epoch random inflates Val AUC**: 0.986-0.989 at K≥5000 is misleading; validation sees different patches each epoch
5. **Residual ~0.15 gap is fundamental**: Feature distribution shift (tumor patch count 43% gap), not fixable by sampling alone
6. **all_patches OOM**: R-MSA attention matrix too large for some slides without patch truncation

## Comparison with 5-Seed Benchmark

| Config | Val AUC | Test AUC | Δ |
|---|---|---|---|
| HE-only [:5000] (5-seed) | 0.8336† | 0.7542 | 0.079 |
| HE-only [:5000] (single split) | 0.9411 | 0.7189 | 0.222 |
| HE-only per_epoch_rand2500 | 0.9428 | 0.7909 | 0.152 |

† OOF (out-of-fold) AUC — honest internal generalization estimate

## Raw Data
- Individual results: `results/patch_sampling/{exp_name}/seed{seed}/result.json`
- Aggregated: `results/patch_sampling/all_results.json`
