# Stage2 改造 —— Train / Val-as-Test 协议

**Evaluation protocol: Train / Val-as-Test**

> official test set 在训练过程中被用于 checkpoint selection / early stopping，因此这里的结果属于既定 val-as-test 开发协议，不作为严格意义上的独立 untouched test。

3 条件 × 3 Model seeds（42/123/456），train=270 / val-as-test=129，统一 LR=1e-4，全部从头随机初始化，无 pretrained。

## AUC 表格

| Model | Seed42 AUC | Seed123 AUC | Seed456 AUC | Mean ± Std |
|---|---|---|---|---|
| HE-only | 0.8344 | 0.8615 | 0.8143 | 0.8367 ± 0.0193 |
| Joint CR-MSA | 0.8505 | 0.8360 | 0.8449 | 0.8438 ± 0.0060 |
| HE Residual Cross | 0.8406 | 0.8908 | 0.8551 | 0.8622 ± 0.0211 |

## paired ΔAUC（逐 seed + 均值）

| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| Joint − HE | 0.0161 | -0.0255 | 0.0306 | 0.0071 ± 0.0238 |
| ResidualCross − HE | 0.0061 | 0.0293 | 0.0408 | 0.0254 ± 0.0144 |
| ResidualCross − Joint | -0.0099 | 0.0548 | 0.0102 | 0.0184 ± 0.0271 |

## ACC / F1 / Sensitivity / Specificity（均值，val-as-test）

| Model | ACC | F1 | Sens(tumor) | Spec(tumor) | Sens(macro) | Spec(macro) |
|---|---|---|---|---|---|---|
| HE-only | 0.8062 | 0.7759 | 0.5782 | 0.9458 | 0.7620 | 0.7620 |
| Joint CR-MSA | 0.8191 | 0.7866 | 0.5646 | 0.9750 | 0.7698 | 0.7698 |
| HE Residual Cross | 0.7855 | 0.7498 | 0.5578 | 0.9250 | 0.7414 | 0.7414 |

## 逐 seed 详细指标

```
{
  "he_only": {
    "per_seed_auc": {
      "42": 0.8344387755102041,
      "123": 0.8614795918367347,
      "456": 0.8142857142857143
    },
    "per_seed_acc": {
      "42": 0.813953488372093,
      "123": 0.8062015503875969,
      "456": 0.7984496124031008
    },
    "per_seed_f1": {
      "42": 0.7881773399014779,
      "123": 0.7719397496640973,
      "456": 0.76759977827051
    },
    "per_seed_sensitivity_tumor": {
      "42": 0.6122448979591837,
      "123": 0.5510204081632653,
      "456": 0.5714285714285714
    },
    "per_seed_specificity_tumor": {
      "42": 0.9375,
      "123": 0.9625,
      "456": 0.9375
    },
    "best_epoch": {
      "42": 7,
      "123": 4,
      "456": 6
    },
    "actual_optimizer_lrs": {
      "he_projection": 0.0001,
      "he_rrt": 0.0001,
      "abmil": 0.0001
    }
  },
  "staining_msa": {
    "per_seed_auc": {
      "42": 0.8505102040816327,
      "123": 0.835969387755102,
      "456": 0.8448979591836735
    },
    "per_seed_acc": {
      "42": 0.8217054263565892,
      "123": 0.8217054263565892,
      "456": 0.813953488372093
    },
    "per_seed_f1": {
      "42": 0.7901845696909695,
      "123": 0.7871134390471407,
      "456": 0.7825842696629214
    },
    "per_seed_sensitivity_tumor": {
      "42": 0.5714285714285714,
      "123": 0.5510204081632653,
      "456": 0.5714285714285714
    },
    "per_seed_specificity_tumor": {
      "42": 0.975,
      "123": 0.9875,
      "456": 0.9625
    },
    "best_epoch": {
      "42": 10,
      "123": 11,
      "456": 11
    },
    "actual_optimizer_lrs": {
      "he_projection": 0.0001,
      "pr_projection": 0.0001,
      "he_rrt": 0.0001,
      "pr_rrt": 0.0001,
      "stage2": 0.0001,
      "abmil": 0.0001
    }
  },
  "he_residual_cross": {
    "per_seed_auc": {
      "42": 0.8405612244897959,
      "123": 0.8908163265306123,
      "456": 0.8551020408163265
    },
    "per_seed_acc": {
      "42": 0.7674418604651163,
      "123": 0.8217054263565892,
      "456": 0.7674418604651163
    },
    "per_seed_f1": {
      "42": 0.7059270516717325,
      "123": 0.7901845696909695,
      "456": 0.753188775510204
    },
    "per_seed_sensitivity_tumor": {
      "42": 0.40816326530612246,
      "123": 0.5714285714285714,
      "456": 0.6938775510204082
    },
    "per_seed_specificity_tumor": {
      "42": 0.9875,
      "123": 0.975,
      "456": 0.8125
    },
    "best_epoch": {
      "42": 9,
      "123": 12,
      "456": 6
    },
    "actual_optimizer_lrs": {
      "he_projection": 0.0001,
      "pr_projection": 0.0001,
      "he_rrt": 0.0001,
      "pr_rrt": 0.0001,
      "stage2": 0.0001,
      "abmil": 0.0001
    }
  }
}
```

## 实际实例化配置 / 实际 optimizer LR

- Stage1 encoder（HE/PR 各自 best，绝对未改）：
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
- Stage2（region_num=4, crmsa_k=3, heads=8, dropout=0.1, epeg=False, ffn=False；he_residual_cross 另加 residual_scale=0.1, disable_cross=False）：
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
- 实际 optimizer LR（每活跃模块，从 result.json 读取）：
```
{
  "he_only": {
    "he_projection": 0.0001,
    "he_rrt": 0.0001,
    "abmil": 0.0001
  },
  "staining_msa": {
    "he_projection": 0.0001,
    "pr_projection": 0.0001,
    "he_rrt": 0.0001,
    "pr_rrt": 0.0001,
    "stage2": 0.0001,
    "abmil": 0.0001
  },
  "he_residual_cross": {
    "he_projection": 0.0001,
    "pr_projection": 0.0001,
    "he_rrt": 0.0001,
    "pr_rrt": 0.0001,
    "stage2": 0.0001,
    "abmil": 0.0001
  }
}
```

## 附加工件

- 每个 seed 目录含 `config.json` / `logs/run.log` / `ckpt/best_model.pt` / `result.json` / `test_predictions.csv` / `protocol_check.json`

