# tune/ — PR-only R²T+ABMIL 超参调优（Optuna）

## 目标

调 PR-only 的 R²T+ABMIL 最优 baseline。**objective = 3-fold 内部 CV 的 mean val AUC**，
而不是单次 Val（避免 "幸运 split" 虚高），更不是官方 Test（避免污染 Test）。

## 文件

| 文件 | 说明 |
|------|------|
| `tune_pr.py` | 主脚本（单 GPU 进程内 Optuna，SQLite 存储） |
| `tune.db` | Optuna study（SQLite，所有 trial 的 params + objective + fold AUCs） |
| `folds/` | 3 份 216/54 分层划分 CSV（一次性生成，复用） |
| `best_config.json` | 跑完后自动写出 best params，供下一步 5-seed 评估用 |
| `search_space_round2.json` | 第二轮扩展搜索空间（复用第一轮 trial 做 warm-start） |

## 搜索空间

**第一轮（默认，写在 `tune_pr.py` 的 `SPACE`）**

| 参数 | 范围 | 分布 |
|------|------|------|
| `region_num` | {2, 4, 8} | categorical |
| `epeg_k` | {5, 9, 15} | categorical |
| `crmsa_k` | {1, 3, 5} | categorical |
| `dropout` | [0.1, 0.5] | uniform |
| `lr` | [1e-5, 3e-4] | log |
| `weight_decay` | [1e-6, 1e-3] | log |

**第二轮（`search_space_round2.json`）**：第一轮 best 三个 categorical 全部命中上界
（region_num=8, epeg_k=15, crmsa_k=5），故扩展上界为超集，**保留旧选择**使历史 trial 能
干净映射到 TPE；三个 float 范围**保持不变**，让 TPE 自动收敛到已发现的优值
（lr~8e-5, wd~5e-6, dropout~0.4）。

| 参数 | 第一轮 | 第二轮 |
|------|--------|--------|
| `region_num` | {2,4,8} | {2,4,8,**16**} |
| `epeg_k` | {5,9,15} | {5,9,15,**25**} |
| `crmsa_k` | {1,3,5} | {1,3,5,**7**} |
| `dropout` / `lr` / `weight_decay` | 不变 | 不变 |

非调参项全部锁定（`mlp_dim=512`, `n_layers=2`, `n_heads=4`, `trans_dropout=0.1`,
`drop_path=0`, `crmsa_heads=8`, `abmil_hidden_dim=256`, `max_patches=2500`,
`batch_size=1`, `num_epochs=25`, early_stop patience=10 on val_auc）。

## 用法

```bash
cd /home/Public/lillan/Two_Sage_RRT-/TwoStageRRT
PY=/home/cxl/miniconda3/envs/rrtmil/bin/python

# 前台跑（30 trials）
$PY tune/tune_pr.py --n-trials 30 --gpu 7

# 后台跑
nohup $PY tune/tune_pr.py --n-trials 30 --gpu 7 > tune/tune.log 2>&1 &

# 断点续跑 / 补足到 30 trials（study 名相同会自动 load）
$PY tune/tune_pr.py --n-trials 30 --gpu 7

# 第二轮：新 study + warm-start（Optuna 不允许在原 study 改 categorical 分布）
#   n-trials = warm-start 复制的 26 + 本轮新增 30 = 56
$PY tune/tune_pr.py --n-trials 56 --gpu 7 --space tune/search_space_round2.json \
    --study-name pr_rrt_tune_v2 --warm-start-from pr_rrt_tune
```

> `--n-trials` 语义 = **study 最终应包含的总 trial 数**（含 warm-start 复制的）。
> `--warm-start-from` 会把旧 study 里所有 `COMPLETE` trial（含 value / params / fold_aucs / 中间值）
> 复制进新的 `--study-name`，让 TPE 直接站在第一轮结果上继续，不重复探索低分区域。

## 轻量化说明

- **不落盘任何 checkpoint / 训练曲线 / per-trial 日志**——best val AUC 只在内存中追踪。
- 所有结果存 `tune.db`（SQLite，几 KB~几百 KB）。
- 3 份 fold CSV 只写一次（`folds/`，共 6 个小文件）。
- 用 `MedianPruner` 在跑完 ≥2 folds 后剪掉明显差的 trial，省算力。

## 协议对齐（与 results/c16_pr_baseline 一致）

- 特征：`C16_PR_features`，K=2500，`sampling='random'`, `sample_seed=42`。
- 单模态官方 `RRTEncoder`（R-MSA + CR-MSA）+ ABMIL。
- 训练：`per_epoch=False`（固定 random，等价于基线 Trainer 的有效行为）。

## 下一步（调优完成后）

把 `best_config.json` 里的 `best_params` 写进一个 PR-only 的 `config.json`，
用 `scripts/_run_fixed_split_exp.py` 跑 **5 seeds**（42/123/456/789/1024）官方 Test，
得到 `PR-only optimized: Test AUC mean ± std`。
