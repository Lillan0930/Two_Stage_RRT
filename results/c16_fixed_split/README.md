# C16 Fixed-Split — HE-only vs HE+PR Two-stage Comparison

**Date**: 2026-08-12  
**Project**: Two_Stage_RRT (Lillan0930/Two_Stage_RRT)

## Protocol (LOCKED)

| Parameter | Value |
|-----------|-------|
| Patch count | K=2500 |
| Sampling | Random (per-epoch train, fixed val/test) |
| HE/PR indices | Shared (generated once per slide) |
| Data split | Fixed 80/20 (StratifiedShuffleSplit, random_state=42) |
| Train slides | 216 (127 normal, 89 tumor) |
| Val slides | 54 (32 normal, 22 tumor) |
| Test slides | 129 (80 normal, 49 tumor) — Official CAMELYON16 Test |
| Seeds | 42, 123, 456, 789, 1024 |
| Epochs | 25 (early stopping patience=10 on Val AUC) |
| Optimizer | Adam (lr=1e-4, wd=1e-5) |
| Scheduler | ReduceLROnPlateau |

**Strict constraints:**
- Official Test NEVER used for training, early stopping, or scheduler
- Same fixed split for all seeds
- Seeds only affect model init, dropout, train randomness, per-epoch sampling
- No 5-fold, no OOF, no K search, no test-time ensemble, no Random PR, no fusion changes

## Results

### Per-Seed

| Model | Seed | Val AUC | Test AUC | Acc | Sens | Spec |
|-------|------|---------|----------|-----|------|------|
| HE_only | 42 | 0.9787 | **0.8080** | 0.8372 | 0.6735 | 0.9375 |
| HE_only | 123 | 0.9602 | 0.7948 | 0.7442 | 0.6531 | 0.8000 |
| HE_only | 456 | 0.9744 | 0.7681 | 0.7209 | 0.5918 | 0.8000 |
| HE_only | 789 | 0.9616 | 0.7832 | 0.7597 | 0.6531 | 0.8250 |
| HE_only | 1024 | 0.9766 | 0.7500 | 0.6512 | 0.7551 | 0.5875 |
| Two_stage | 42 | 0.8736 | 0.6865 | 0.5891 | 0.6735 | 0.5375 |
| Two_stage | 123 | 0.8949 | 0.6673 | 0.5736 | 0.7143 | 0.4875 |
| Two_stage | 456 | 0.9588 | **0.7832** | 0.7364 | 0.6531 | 0.7875 |
| Two_stage | 789 | 0.9261 | 0.7304 | 0.7209 | 0.4490 | 0.8875 |
| Two_stage | 1024 | 0.9688 | 0.7261 | 0.6667 | 0.6735 | 0.6625 |

### Aggregated (mean ± std)

| Model | Val AUC | Test AUC | Acc | Sens | Spec |
|-------|---------|----------|-----|------|------|
| HE_only | 0.9703 ± 0.0078 | **0.7808 ± 0.0203** | 0.7426 ± 0.0601 | 0.6653 ± 0.0526 | 0.7900 ± 0.1133 |
| Two_stage | 0.9244 ± 0.0364 | 0.7187 ± 0.0401 | 0.6574 ± 0.0664 | 0.6327 ± 0.0940 | 0.6725 ± 0.1497 |

### Paired Difference (Two_stage − HE_only)

| Metric | Value |
|--------|-------|
| Mean Δ Test AUC | −0.0621 |
| 95% CI | [−0.1389, +0.0147] |
| Paired t-test | t = −2.246, p = 0.0881 |
| Significance | **NOT significant** at α=0.05 |

### Per-Slide Prediction Agreement

| Seed | Agreement | Cohen's κ | Prob Correlation |
|------|-----------|-----------|-----------------|
| 42 | 0.628 | 0.281 | 0.341 |
| 123 | 0.643 | 0.318 | 0.376 |
| 456 | 0.829 | 0.632 | 0.677 |
| 789 | 0.744 | 0.399 | 0.414 |
| 1024 | 0.783 | 0.569 | 0.585 |

## Key Findings

1. **HE-only consistently outperforms Two-stage HE+PR** under the corrected sampling protocol. Mean Test AUC: 0.7808 vs 0.7187 (Δ = −0.0621, p = 0.0881 → not significant at α=0.05).

2. **HE-only is more stable**: Test AUC std = 0.0203 vs Two-stage 0.0401. Two-stage shows seed 42/123 catastrophic failure (Test AUC 0.667–0.687) while seed 456/789/1024 perform comparably to HE-only.

3. **Large Val→Test generalization gap**: HE_only Val 0.9703 → Test 0.7808 (Δ = −0.1895). This is much larger than the OOF→Test gap found in 5-fold (Δ ≈ 0.05), suggesting the single-split Val AUC is severely inflated.

4. **Two-stage Val AUC is unreliable**: Val AUC 0.9244 but Test AUC only 0.7187. The gap of −0.2057 is larger than for HE-only, suggesting the more complex two-stage model overfits to the small val set (54 slides).

5. **Per-slide agreement between HE_only and Two_stage varies drastically by seed** (κ: 0.281–0.632). When Two_stage works (seed 456, κ=0.632), it makes similar predictions to HE_only. When it fails (seed 42, κ=0.281), the predictions diverge significantly.

6. **The fused PR modality does not improve C16 classification** under this architecture and protocol. This aligns with the 5-fold OOF comparison results (OOF: HE-only ≈ 0.847 vs Two-stage ≈ 0.817–0.826).

## Directory Structure

```
results/c16_fixed_split/
├── all_results.json         — full per-experiment results
├── paired_analysis.json     — paired t-test and difference statistics
├── HE_only/
│   ├── seed{42,123,456,789,1024}/
│   │   ├── config.json      — experiment configuration
│   │   ├── result.json      — training + test metrics
│   │   ├── test_predictions.csv — per-slide probabilities
│   │   ├── ckpt/best_model.pt   — best checkpoint
│   │   └── logs/            — training logs
│   ...
└── Two_stage/
    └── seed{42,123,456,789,1024}/
        └── ...
```

## Reproduction

```bash
cd /home/Public/lillan/Two_Sage_RRT-/TwoStageRRT
/home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_c16_fixed_split.py
```
