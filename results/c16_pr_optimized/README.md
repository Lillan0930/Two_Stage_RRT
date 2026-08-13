# PR-only 调参结果（Optuna + 5-seed 官方 Test）

## 目的

调 PR-only 的 R²T+ABMIL 最优 baseline。objective = **3-fold 内部 CV mean val AUC**（不污染官方 Test，不依赖单次 Val）。

## 流程

1. `tune/tune_pr.py` 用 Optuna（TPE + MedianPruner）搜 30 trials → `tune/tune.db`。
2. 固定 best config，`scripts/_run_fixed_split_exp.py` 跑 5 seeds 官方 Test。

## 最优配置（第一轮）

| 参数 | 最优值 | 搜索范围 |
|------|--------|---------|
| region_num | **8** | {2, 4, 8} |
| epeg_k | **15** | {5, 9, 15} |
| crmsa_k | **5** | {1, 3, 5} |
| dropout | 0.4153 | 0.1 ~ 0.5 |
| lr | 7.79e-05 | 1e-5 ~ 3e-4 |
| weight_decay | 5.94e-06 | 1e-6 ~ 1e-3 |

> region_num / epeg_k / crmsa_k **全部命中上界** → 第二轮已扩展搜索（见 `tune/search_space_round2.json`）。

## 5-seed 官方 Test 结果

| seed | Val AUC | **Test AUC** | Sens | Spec |
|------|---------|-------------|------|------|
| 42 | 0.9361 | 0.7258 | 0.7143 | 0.4875 |
| 123 | 0.9162 | 0.7574 | 0.5918 | 0.7875 |
| 456 | 0.9432 | 0.7212 | 0.4694 | 0.8125 |
| 789 | 0.9148 | 0.7283 | 0.7959 | 0.5375 |
| 1024 | 0.9105 | 0.7798 | 0.5510 | 0.7750 |

**Test AUC = 0.7425 ± 0.0253**

## 对比

| 模型 | Test AUC |
|------|----------|
| PR-only（未调参，seed42） | 0.6571 |
| **PR-only（调参，5-seed mean）** | **0.7425 ± 0.025** |
| HE-only | 0.8080 |

调参带来 **+0.085**，与 HE 的差距从 0.15 缩到 0.066。
注意 val 仍虚高（0.91 ~ 0.94 vs test 0.72 ~ 0.78），最终以官方 Test 为准。
