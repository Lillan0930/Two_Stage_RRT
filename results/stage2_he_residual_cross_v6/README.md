# HE Residual Cross v6 —— Train / Val-as-Test 协议（paired v3 vs v6）

**Evaluation protocol: Train / Val-as-Test**

> official test set 在训练中被用于 checkpoint selection / early stopping，因此属既定 val-as-test 开发协议，非严格意义的独立 untouched test。

v6 = v3 前向结构 + 训练期梯度分工：HE 投影/RRT + 主 ABMIL 只由 HE CE 更新；PR 投影/RRT + 全部 Stage2 只由 fused CE 更新。前向一次、两个 loss、两组各自裁剪。

## AUC（逐 seed；v3 为配对对照）

| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| v3 (control) | 0.8753 | 0.8755 | 0.8724 | 0.8744 ± 0.0014 |
| v6 (target)  | 0.8500 | 0.8737 | 0.8219 | 0.8486 ± 0.0212 |

## paired ΔAUC（v6 − v3）

| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| v6 − v3 | -0.0253 | -0.0018 | -0.0505 | -0.0259 ± 0.0199 |

## v6 两个 loss（末 epoch 均值）

| Seed | CE_HE | CE_fused |
|---|---|---|
| 42 | 0.4785 | 0.4811 |
| 123 | 0.4916 | 0.4449 |
| 456 | 0.7441 | 0.5718 |

## Stage2 config (v3 = v6)
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

