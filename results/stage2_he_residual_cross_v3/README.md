# HE Residual Cross v3 —— Train / Val-as-Test 协议

**Evaluation protocol: Train / Val-as-Test**

> official test set 在训练中被用于 checkpoint selection / early stopping，因此属既定 val-as-test 开发协议，非严格意义的独立 untouched test。

v3 = v2（cosine τ=0.2 + bias-free QKV）+ dataset-prototype PR value centering（减 train-set EMA prototype μ_PR 而非 per-slide 均值）。HE-only/v1/v2 复用，不重训。

## AUC（逐 seed；HE-only / v1 / v2 复用）

| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| HE-only (reuse) | 0.8344 | 0.8615 | 0.8143 | 0.8367 ± 0.0193 |
| v1 (reuse) | 0.8406 | 0.8908 | 0.8551 | 0.8622 ± 0.0211 |
| v2 (reuse) | 0.8418 | 0.8755 | 0.8139 | 0.8438 ± 0.0252 |
| HE Residual Cross v3 | 0.8753 | 0.8755 | 0.8724 | 0.8744 ± 0.0014 |

## paired ΔAUC

| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| v3 − HE | 0.0408 | 0.0140 | 0.0582 | 0.0377 ± 0.0182 |
| v3 − v1 | 0.0347 | -0.0153 | 0.0173 | 0.0122 ± 0.0207 |
| v3 − v2 | 0.0334 | -0.0000 | 0.0585 | 0.0307 ± 0.0240 |

## Stage2 config (v3)
```
{
  "region_num": 4,
  "crmsa_heads": 8,
  "crmsa_k": 3,
  "drop_out": 0.1,
  "drop_path": 0.0,
  "epeg": false,
  "epeg_k": 15,
  "crmsa_mlp": false,
  "ffn": false,
  "qkv_bias": false,
  "temperature": 0.2,
  "residual_scale": 0.1,
  "disable_cross": false,
  "prototype_momentum": 0.99
}
```

