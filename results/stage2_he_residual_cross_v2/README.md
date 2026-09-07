# HE Residual Cross v2 —— Train / Val-as-Test 协议

**Evaluation protocol: Train / Val-as-Test**

> official test set 在训练中被用于 checkpoint selection / early stopping，因此属既定 val-as-test 开发协议，非严格意义的独立 untouched test。

v2 = cosine cross-attention (τ=0.2) + PR value centering + bias-free QKV。HE-only 与 v1 结果复用上一轮，不重训。

## AUC（v2 逐 seed；HE-only / v1 复用）

| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| HE-only (reuse) | 0.8344 | 0.8615 | 0.8143 | 0.8367 ± 0.0193 |
| HE Residual Cross v1 (reuse) | 0.8406 | 0.8908 | 0.8551 | 0.8622 ± 0.0211 |
| HE Residual Cross v2 | 0.8418 | 0.8755 | 0.8139 | 0.8438 ± 0.0252 |

## paired ΔAUC

| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| v2 − HE | 0.0074 | 0.0140 | -0.0004 | 0.0070 ± 0.0059 |
| v2 − v1 | 0.0013 | -0.0153 | -0.0412 | -0.0184 ± 0.0175 |

## Stage2 config (v2)
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
  "value_centering": true,
  "residual_scale": 0.1,
  "disable_cross": false
}
```

