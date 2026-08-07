# AGENTS.md

## Project totals （规模总数） — commit f342cce

- **Code**: 24 `.py` files (incl. `scripts/`), plus C16 label CSVs now in `TwoStageRRT/data/C16_labels/`.
- **Model parameters** (instantiated `MM_RRT_ABMIL` with the config in `run_c16_two_stage.py`: HE+PR, input_dim=768, mlp_dim=512, region_num=4, n_layers=2, n_heads=4, epeg_k=9, crmsa_k=3, crmsa_heads=8, abmil_hidden_dim=256):
  - **Active in `two_stage_region` path: 6,380,117 (~6.38M)** = rrt_he 2,105,896 + rrt_ihc 2,105,896 (independent official RRTEncoder ×2) + cross_region_mod 1,117,442 (default `stage2_type='staining_msa'`) + mil 263,427 + patch_to_emb 787,456.
  - Total instantiated: 10,852,776 — includes legacy `self.rrt_encoder` (MM_RRTEncoder, 4,472,659) which is created unconditionally but **unused** by the two-stage path.

## Layout

- Repo root contains only `TwoStageRRT/`; there is no top-level package, manifest, or config. All paths below are relative to `TwoStageRRT/`.
- Entrypoints: `run_c16_two_stage.py` (standalone smoke test, builds config inline as a Python dict) and `train.py` (`Trainer` class, YAML-config-driven).
- `train.py` docstring references `config/config_dual_raw_er.yaml`, but **no `config/` directory exists in the repo** — use the inline dict in `run_c16_two_stage.py` as the config template.
- `data/C16_labels/` CSVs are in the repo (since f342cce); all feature directories are **not in the repo**.

## Run

```bash
cd TwoStageRRT && python run_c16_two_stage.py
```

Gotchas:
- `run_c16_two_stage.py` hardcodes Linux: `os.environ['CUDA_VISIBLE_DEVICES']='7'`, features at `/home/Public/lillan/features_result/C16_features`, outputs to `/tmp/ts_standalone/`. It will not run unmodified on Windows — edit paths/device first.
- Scripts rely on `sys.path.insert(0, <script dir>)`; run them from inside `TwoStageRRT/` because label paths (`data/C16_labels/...`) are relative.
- Local env has **CPU-only torch 2.12.0+cpu** (Python 3.11); real training targets CUDA.
- No requirements.txt, no tests, no lint/typecheck config, no CI. Deps (import from source): torch, numpy, pandas, scikit-learn, matplotlib, tqdm, pyyaml, **timm** (DropPath, required since f342cce).
- Since f342cce, the `two_stage_region` path uses the **official-faithful `RRTEncoder`** (rmsa.py/mm_rrt_encoder.py, "exact match with official RRT-MIL") with **independent weights per modality** (`rrt_he`/`rrt_ihc`); CR-MSA is TransLayer-wrapped (pre-LN + residual + DropPath). The old multi-modal `MM_RRTEncoder` (`self.rrt_encoder`) only serves legacy fusion paths.
- Stage 2 has two variants via model kwarg `stage2_type`: `'staining_msa'` (**default**, `cross_staining_region_msa.py` — "staining-as-region" symmetric MSA fusion over nK region tokens with phantom-region masking, zero-init `fusion_gate`, outputs concat over stainings `[B, ΣN_m, D]`) and `'he_anchor'` (legacy `cross_region_reembedding.py`, kept for ablation).

## Code conventions / quirks

- `MM_RRT_ABMIL.forward()` returns a **tuple**: `out[0]`=logits, `out[3]`=fusion_stats dict, `out[4]`=aux_loss. With `return_features=True` it returns a dict instead (`'logits'`, `'embedded_features'`, `'fusion_stats'`). KD/aux losses in `train.py` read per-path logits (`logits_he`, `logits_pr`, `logits_ihc_list`) out of `fusion_stats`.
- `Trainer` supports two dataset types via `config['data']['dataset_type']`: `'c16'` (separate `train_label_file`/`val_label_file`, dirs `normal|tumor|test/*.pt` per modality) and default `'c17'` (single label file, patient split by `data_split.val_start`).
- Feature dir names are derived: `{base}/C17_{MOD}_features` (or `C17_raw_features` for `RAW`) unless overridden by `data.dir_mapping`.
- Many fusion/KD/ablation flags in `create_model` (train.py:451) are legacy experiments; the verified two-stage path uses `fusion_type='two_stage_region'` + `use_shared_rrt=False` and trains with AMP off, batch_size=1, modality_dropout=0, aux_loss_weight=0, kd_enabled=False.
- Reproducibility: `set_seed` uses `cudnn.benchmark=True` (deliberately not deterministic); reported C16 result (seed=42, 25 epochs): HE-only AUC 0.7811, two-stage HE+PR AUC 0.8346.
