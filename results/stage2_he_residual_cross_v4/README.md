# HE Residual Cross v4 —— Train / Val-as-Test 协议

**Evaluation protocol: Train / Val-as-Test**

> official test set 在训练中被用于 checkpoint selection / early stopping，因此属既定 val-as-test 开发协议，非严格意义的独立 untouched test。

v4 = v3（cosine τ=0.2 + bias-free QKV + dataset-prototype PR value centering）+ Q/K slide-internal common-direction removal（只作用于 attention score）。HE-only/v1/v2/v3 复用，不重训。

## AUC（逐 seed；HE-only / v1 / v2 / v3 复用）

| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| HE-only (reuse) | 0.8344 | 0.8615 | 0.8143 | 0.8367 ± 0.0193 |
| v1 (reuse) | 0.8406 | 0.8908 | 0.8551 | 0.8622 ± 0.0211 |
| v2 (reuse) | 0.8418 | 0.8755 | 0.8139 | 0.8438 ± 0.0252 |
| v3 (reuse) | 0.8753 | 0.8755 | 0.8724 | 0.8744 ± 0.0014 |
| HE Residual Cross v4 | 0.8707 | 0.8653 | 0.8444 | 0.8601 ± 0.0113 |

## paired ΔAUC

| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| v4 − HE | 0.0362 | 0.0038 | 0.0301 | 0.0234 ± 0.0141 |
| v4 − v1 | 0.0301 | -0.0255 | -0.0107 | -0.0020 ± 0.0235 |
| v4 − v2 | 0.0288 | -0.0102 | 0.0305 | 0.0164 ± 0.0188 |
| v4 − v3 | -0.0046 | -0.0102 | -0.0281 | -0.0143 ± 0.0100 |

## Stage2 config (v4)
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

