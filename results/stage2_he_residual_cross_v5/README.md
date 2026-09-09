# HE Residual Cross v5 —— 固定 split 内部验证配对对照

**协议：fixed_split（216/54）+ dev test(129，仅背景）**

> checkpoint selection / early stopping 在内部 val(54) 上进行；129 集仅作开发背景（早期 val-as-test 已污染），不能称独立测试收益。

v5 = v3（cosine τ=0.2 + bias-free QKV + dataset-prototype PR value centering）+ 训练期对 merged PR value memory 的辅助 ABMIL 分类监督（β=0.1）。对照组 v5_beta0 用同一模块、β=0，与 v3 主路径逐位一致。

## dev test AUC（129，仅背景）

| seed | v5_beta0 (β=0) | v5_beta01 (β=0.1) | Δ |
|---|---|---|---|
| 42 | 0.7468 | 0.7472 | 0.0004 |
| 123 | 0.8531 | 0.7755 | -0.0776 |
| 456 | 0.7997 | 0.7668 | -0.0329 |
| **mean** | 0.7999 | 0.7632 | -0.0367 |

## 内部 val AUC（54，primary）

| seed | v5_beta0 (β=0) | v5_beta01 (β=0.1) | Δ |
|---|---|---|---|
| 42 | 0.9645 | 0.9659 | 0.0014 |
| 123 | 0.9730 | 0.9815 | 0.0085 |
| 456 | 0.9886 | 0.9830 | -0.0057 |
| **mean** | 0.9754 | 0.9768 | 0.0014 |

## Stage2 config (v5)
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
  "prototype_momentum": 0.99,
  "aux_hidden_dim": 64,
  "aux_dropout": 0.1
}
```

