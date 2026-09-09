# HE Residual Cross v7（恢复融合梯度）—— Train / Val-as-Test 协议

**Evaluation protocol: Train / Val-as-Test**

> official test set 在训练中被用于 checkpoint selection / early stopping，因此属既定 val-as-test 开发协议，非严格意义的独立 untouched test。

v7 = v3 前向 + 训练期双 loss（同一 MIL，无 detach）：`L = CE(M(H),y) + CE(M(F),y)`。A: `F=Stage2(H.detach(),P)`；B: `F=Stage2(H,P)`（主候选）。三参数组各自 clip。

## AUC（逐 seed；v3 为同期基线，复用）

| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| v3 (baseline) | 0.8753 | 0.8755 | 0.8724 | 0.8744 |
| v7A | 0.8327 | 0.8645 | 0.8204 | 0.8392 ± 0.0186 |
| v7B | 0.8166 | 0.8492 | 0.8152 | 0.8270 ± 0.0157 |

## paired ΔAUC

| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| v7A − v3 | -0.0426 | -0.0110 | -0.0520 | -0.0352 ± 0.0176 |
| v7B − v3 | -0.0587 | -0.0263 | -0.0573 | -0.0474 ± 0.0150 |
| v7B − v7A | -0.0161 | -0.0153 | -0.0052 | -0.0122 ± 0.0049 |

## 初始化核对

A/B 同 seed init hash 一致: {'42': True, '123': True, '456': True}


## Stage2 config (v3 = v7)
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


## 最终判定（§7）：FAILED —— 终止梯度隔离修补路线

- A（0.8392）与 B（0.8270）**都未超过同期 v3（0.8744）**：A−v3 = −0.0352、B−v3 = −0.0474，均 3/3 seed 变差。
- 双 loss（CE(M(H)) + CE(M(F))）本身损害训练：B 的 HE 路径（disable_cross）崩到 0.749–0.803（v3 为 0.872–0.876）；B 的残差在**同模型内**能补回 +0.014/+0.050/+0.066，但净预测仍远低于 v3。
- 正确配对 PR 依旧无帮助：normal − replaced_pr ≈ 0（6/6 组 |Δ| ≤ 0.001）——cross 对 PR content 不敏感的结论第三次复现（v3/v5/v7）。
- 依 §7：不自动追加新 loss / gate / 新版本，梯度隔离修补路线到此结束。
