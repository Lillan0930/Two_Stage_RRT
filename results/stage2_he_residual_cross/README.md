# Stage2 改造实验 — he_residual_cross vs staining_msa vs HE-only

3 条件 × 3 seeds（42/123/456），固定 split 216/54，Test 129，统一 LR=1e-4，全部从头随机初始化。

## Test AUC 表格

| seed | HE-only | staining_msa | he_residual_cross |
|---|---|---|---|
| 42 | 0.7772 | 0.8156 | 0.7543 |
| 123 | 0.8413 | 0.8334 | 0.8283 |
| 456 | 0.7625 | 0.7987 | 0.7793 |
| **mean** | 0.7937 | 0.8159 | 0.7873 |
| **std** | 0.0342 | 0.0142 | 0.0307 |

## best_epoch

```
{
  "he_rrt_he_only": {
    "42": 26,
    "123": 37,
    "456": 6
  },
  "staining_msa": {
    "42": 15,
    "123": 24,
    "456": 14
  },
  "he_residual_cross": {
    "42": 18,
    "123": 8,
    "456": 12
  }
}
```

## 附加工件

- 每个 seed 目录含 `result.json` / `test_predictions.csv` / `config.json` / `ckpt/best_model.pt` / `logs/run.log`

