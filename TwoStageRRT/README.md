# TwoStageRRT: Two-stage Modality-aware R²T

Clean standalone implementation of Two-stage Modality-aware Region R²T.

## Architecture

```
HE feature          IHC feature
    |                    |
Projection          Projection
    |                    |
R²T Encoder         R²T Encoder (shared weights)
    |                    |
Z_HE [B,N,D]        Z_IHC [B,N,D]
    \                  /
     Cross-Region Re-embedding
              |
     Z_final [B,N,D]
              |
           ABMIL
```

## Key Results (C16, seed=42, 25 epochs)

| Experiment | AUC |
|---|---|
| HE-only | 0.7811 |
| Two-stage (HE+PR) | 0.8346 |
| Random IHC | 0.8042 |

## Usage

```bash
python run_c16_two_stage.py
```

## Project Structure

```
TwoStageRRT/
├── models/
│   ├── cross_region_reembedding.py  # Stage 2 module
│   ├── mm_rrt_abmil.py              # Main model
│   ├── mm_rrt_encoder.py            # R²T encoder
│   ├── rmsa.py                      # Region/MSA attention
│   ├── abmil.py                     # ABMIL aggregator
│   └── mil_registry.py              # MIL registry
├── data/
│   ├── c16_multimodal_dataset.py
│   └── C16_labels/
├── utils/
│   └── metrics.py
├── train.py
└── run_c16_two_stage.py
```
