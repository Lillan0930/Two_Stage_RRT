# symmetric_r8_noepeg

对称双模态 official-style Cross-Staining CR-MSA（Stage2），270 train / 129 test-as-val，5 seeds。

## Stage2 配置
```
{
  "region_num": 8,
  "crmsa_heads": 8,
  "crmsa_k": 3,
  "drop_out": 0.1,
  "drop_path": 0.0,
  "epeg": false,
  "epeg_k": 15,
  "crmsa_mlp": false,
  "ffn": false,
  "qkv_bias": true
}
```

## Stage1 encoder（固定，各自 best）
```
{
  "HE": {
    "region_num": 4,
    "epeg_k": 9,
    "crmsa_k": 3,
    "n_heads": 4,
    "drop_path": 0.0
  },
  "PR": {
    "region_num": 8,
    "epeg_k": 15,
    "crmsa_k": 5,
    "n_heads": 8,
    "drop_path": 0.11554210024949738
  }
}
```

## Per-seed 指标

| seed | AUC | Acc | F1 | Sens | Spec | epoch |
|---|---|---|---|---|---|---|
| 42 | 0.6806 | 0.6899 | 0.6551 | 0.6511 | 0.6511 | 8 |
| 123 | 0.6811 | 0.6357 | 0.6267 | 0.6351 | 0.6351 | 14 |
| 456 | 0.7075 | 0.6822 | 0.6708 | 0.6765 | 0.6765 | 14 |
| 789 | 0.6971 | 0.6357 | 0.6176 | 0.6193 | 0.6193 | 16 |
| 1024 | 0.7128 | 0.7054 | 0.6646 | 0.6597 | 0.6597 | 15 |

- **AUC**: 0.6958 ± 0.0132
- **Acc**: 0.6698 ± 0.0288
- **F1**: 0.6470 ± 0.0210

## 与 HE-only 同 seed 的 paired ΔAUC

HE-only 5-seed: AUC 0.8052 ± 0.0233

| seed | this AUC | HE-only AUC | ΔAUC |
|---|---|---|---|
| 42 | 0.6806 | 0.7911 | -0.1105 |
| 123 | 0.6811 | 0.7899 | -0.1088 |
| 456 | 0.7075 | 0.7852 | -0.0777 |
| 789 | 0.6971 | 0.8119 | -0.1148 |
| 1024 | 0.7128 | 0.8480 | -0.1352 |

- **paired ΔAUC**: -0.1094 ± 0.0185

## 与旧 symmetric r4 epeg=True 对比

旧结果（Stage2 region_num=4, epeg=True，仅 3 seeds 42/123/456）: AUC 0.7347 ± 0.0216

本版本在公共 3 seeds 上: AUC 0.6898，Δ = -0.0449

## 附加工件

- `padding_stats.json` — Stage2 padding 统计（train/test 每 slide N/H/W/add_length/padding_ratio + 汇总）
- `seed42/stage2_magnitude_seed42.json` — seed42 幅度诊断 ||delta||/||z||（HE/PR）

