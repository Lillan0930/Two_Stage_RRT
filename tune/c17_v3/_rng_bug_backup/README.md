# `_rng_bug_backup/` —— sampler-RNG 缺陷下的作废结果

**这里的 5 个 study 全部是非正式结果，不进入最终 Optuna 结果。**

归档时间：2026-10-15。原因：`TPESampler` 的 RNG 在"一条 trial 一个进程"的
driver 下每轮从头重放，导致 **TPE 阶段提案锁死成不动点**。修复见
`../optuna_v3.py` 的 `sampler_seed_for()`。

目录是**移动**过来的，不是删除；内容一律只读保留，不要就地续跑。

---

## 1. 缺陷机制

driver 用 `--trials-per-process 1`，每个 trial 起一个新进程；而每个进程都
`TPESampler(seed=2026)` 从零构造，`_rng` 于是每轮回到同一起点。

TPE 把"候选点云"交给 `mpe_below` / `mpe_above` 打分后取 `argmax`：

- `mpe_above` 随 history 增长而变；
- `mpe_below` 只由 below 集的**参数**决定，而
  `n_below = min(ceil(0.1·n), 25)` 在 `n ≤ 10` 时**恒等于 1** —— below 集
  就只剩 argmax 那一个点。

于是 argmax 一旦稳定，`mpe_below` 逐位不变，候选点云又被常数种子原样重放，
`np.argmax` 每轮挑中同一个下标 —— **提案变成不动点，之后每一轮都在重复同一次
尝试**。早期（#5/#6/#7）偶有变化，是因为 `mpe_above` 还在动。

## 2. 证据

三个 study.db 并排打开，同一张表：

| 现象 | 位置 |
|---|---|
| **连续 16 条提案逐位相同且全部 FAIL** | `HE_Ki67` #8–#23 == #7 |
| **跨组合重放同一条 RNG 流** | `HE_PR_ER` #4/#5/#6/#7 与 `HE_PR_HER2_ER` #4/#5/#6/#7 逐位相同 |
| 胜出配置本身出自被污染的流 | `HE_PR_ER` #8 的 lr 原样出现在 `HE_PR_HER2_ER` #8 |

`HE_Ki67` 的 `#7`（lr=0.000292028481330547）拿到 0.9420281064，成为史上最好；
紧接着 #8–#23 把**同一个刚跑过的配置**又提了 16 次，全部判重 FAIL。搜索在
这 16 轮里一点新信息都没产生。

## 3. 归档时的各组合状态

来自 `_state_at_archive.txt`（`有效` = `n_valid()` 认可的 trial 数）：

| 组合 | 有效 | 库内条数 | FAIL | best_trial | best_value | 是否收尾 |
|---|---|---|---|---|---|---|
| HE+Ki67 | 8 | 24 | 16 | 7 | 0.9420281064 | 否（中断） |
| HE+PR | ? | 库已删 | ? | **0** | 0.9492467900 | 是（已收尾） |
| HE+HER2+Ki67 | ? | 库已删 | ? | 8 | 0.9411586291 | 是（已收尾） |
| HE+PR+ER | 10 | 11 | 0 | 8 | 0.9424628450 | 否（中断） |
| HE+PR+HER2+ER | 8 | 9 | 0 | 2 | 0.9413102821 | 否（中断） |

两点值得记下：

- **HE+PR 的 best_trial 是 0** —— 即基线自己赢了，整轮 TPE 什么都没搜到更好的。
  这正是提案锁死的直接后果。
- `best_params.json` / `best_result.json` / `holdout/` 只对**已收尾**的
  HE+PR、HE+HER2+Ki67 存在（HE+HER2+Ki67 的 holdout 只跑完 seed 3611）。
  这两个"收尾"是 driver 在有效 trial 数够时自动触发的，**不代表结果可信** ——
  它们的搜索过程同样跑在坏掉的提案流上。

## 4. 为什么没有就地续跑

- 缺陷是**采样器状态**层面的：库里的 12 条 history 已经被污染，继续在上面
  采样只会继续被同一个不动点吸住。
- 而且旧种子是常数、`argmax` 已稳定，重启后仍会立刻回到同一个不动点 ——
  **不修代码就没法靠重启脱身**。
- 用户 2026-10-15 的决定：5 个 study 全部视为非正式结果，归档后**全新重启**。

## 5. 目录内容

```
HE_Ki67/            study.db（24 条）, best_model.pt
HE_PR/              best_model.pt, best_params.json, best_result.json,
                    holdout/（3 seeds）, holdout_result.json
HE_HER2_Ki67/       best_model.pt, best_params.json, best_result.json,
                    holdout/（仅 seed 3611）
HE_PR_ER/           study.db（11 条，含 1 条 RUNNING）, best_model.pt
HE_PR_HER2_ER/      study.db（9 条，含 1 条 RUNNING）, best_model.pt
tmp/                driver_10151117.log（89 MB，tqdm 撑大的）,
                    *_run.log, holdout_worker.log,
                    trial_HE_PR_ER_010/, trial_HE_PR_HER2_ER_008/
_state_at_archive.txt
summary.csv, summary_holdout.csv
```

三个 study.db 归档时都过了 `PRAGMA integrity_check` = `ok`。

## 6. 这些结果不能怎么用

- **不能**把上表的 `best_value` 当成"Optuna 搜出来的最优超参"——它们是坏采样器
  之下的产物，`HE+PR` 更是连一次有效 TPE 提案都没落地。
- **不能**用它们和重跑后的结果比较优劣：重跑是干净采样，两者不同源。
- 保留它们只是为了留下缺陷证据（第 2 节那张表），以及万一需要回溯
  `best_model.pt` 是从哪条提案来的。
