# symmetric_r4_noepeg

对称双模态 official-style Cross-Staining CR-MSA（Stage2），270 train / 129 test-as-val，5 seeds。

## Stage2 配置
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
| 42 | 0.6709 | 0.6124 | 0.5997 | 0.6045 | 0.6045 | 7 |
| 123 | 0.6980 | 0.6202 | 0.3828 | 0.5000 | 0.5000 | 0 |
| 456 | 0.7040 | 0.7364 | 0.7035 | 0.6966 | 0.6966 | 17 |
| 789 | 0.6861 | 0.6667 | 0.6418 | 0.6403 | 0.6403 | 16 |
| 1024 | 0.7301 | 0.6977 | 0.6655 | 0.6614 | 0.6614 | 15 |

- **AUC**: 0.6978 ± 0.0197
- **Acc**: 0.6667 ± 0.0468
- **F1**: 0.5986 ± 0.1131

## 与 HE-only 同 seed 的 paired ΔAUC

HE-only 5-seed: AUC 0.8052 ± 0.0233

| seed | this AUC | HE-only AUC | ΔAUC |
|---|---|---|---|
| 42 | 0.6709 | 0.7911 | -0.1202 |
| 123 | 0.6980 | 0.7899 | -0.0920 |
| 456 | 0.7040 | 0.7852 | -0.0813 |
| 789 | 0.6861 | 0.8119 | -0.1258 |
| 1024 | 0.7301 | 0.8480 | -0.1179 |

- **paired ΔAUC**: -0.1074 ± 0.0175

## 与旧 symmetric r4 epeg=True 对比

旧结果（Stage2 region_num=4, epeg=True，仅 3 seeds 42/123/456）: AUC 0.7347 ± 0.0216

本版本在公共 3 seeds 上: AUC 0.6909，Δ = -0.0437

## 附加工件

- `padding_stats.json` — Stage2 padding 统计（train/test 每 slide N/H/W/add_length/padding_ratio + 汇总）
- `seed42/stage2_magnitude_seed42.json` — seed42 幅度诊断 ||delta||/||z||（HE/PR）

