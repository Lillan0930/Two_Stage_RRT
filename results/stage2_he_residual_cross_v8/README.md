# HE Residual Cross v8（HE region queries 读取 PR patch memory）—— Train / Val-as-Test 协议

**Evaluation protocol: Train / Val-as-Test**

> official test set 在训练中被用于 checkpoint selection / early stopping，因此属既定 val-as-test 开发协议，非严格意义的独立 untouched test。

v8 = 恢复 v3 训练（单 fused CE、全局裁剪）+ PR memory 结构对照：routed（PR routing 压缩，≡v3）vs patch（跳过 PR routing pooling，K/V = 全部有效 Z_PR patch token）。

## AUC（逐 seed；v3 为同期基线，复用）

| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| v3 (baseline) | 0.8753 | 0.8755 | 0.8724 | 0.8744 |
| v8-routed | 0.8753 | 0.8755 | 0.8724 | 0.8744 ± 0.0014 |
| v8-patch | 0.8559 | 0.8270 | 0.8758 | 0.8529 ± 0.0200 |

## paired ΔAUC

| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| routed − v3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 |
| patch − v3 | -0.0194 | -0.0485 | 0.0033 | -0.0215 ± 0.0212 |
| patch − routed | -0.0194 | -0.0485 | 0.0033 | -0.0215 ± 0.0212 |

## 初始化核对

routed/patch 同 seed init hash 一致: {'42': True, '123': True, '456': True}


## Stage2 config (v3 = v8)
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


## 最终判定（§7）：patch 未超过 v3 —— FAILED

- routed（对照）在完整 270/129 训练下逐位复现同期 v3（0.8753/0.8755/0.8724，routed−v3=+0.0000），配对有效性成立。
- patch（K/V = 全部有效 Z_PR patch token，无 routing 压缩）= 0.8529，**patch−v3 = −0.0215**（2/3 seed 变差，seed456 +0.0034 微涨）。
- **同一 checkpoint 内**：patch 的残差贡献（normal − disable_cross）≈ 0（|Δ|≤0.0008），正确配对 PR 的帮助（normal − replaced_pr）≈ 0（|Δ|≤0.0003，5 次替换逐 seed 几乎相同）。
- 结论：即使 PR 侧完全去掉 routing 压缩（K 从 48 扩到 2500），cross-attention 依旧学不到 PR 内容——**PR routing 压缩不是瓶颈**。结合 v5（PR value 可被训出判别性）与 v4/v3：瓶颈是 cross-attention 通路端到端无法利用 PR content，与梯度分工（v6/v7）、Q/K 方向（v4）、value 表征（v5）、routing 压缩（v8）均无关。
