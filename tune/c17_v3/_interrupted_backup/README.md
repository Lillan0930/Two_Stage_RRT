# 中断归档：2026-09-18 NVMe 硬件故障

**这份归档只作故障记录，禁止 resume、禁止复用其中的 study.db。**

## 发生了什么

2026-09-18 约 17:50，承载 `/home/Public` 的 NVMe（当时枚举为 `nvme1`）控制器
I/O 超时后掉出 PCIe 总线，设备节点消失。ext4 journal aborted，内核将该卷重挂
为只读，5 条 Optuna 链与 GPU7 worker 全部被杀。

内核证据：

```
nvme nvme1: I/O 239 QID 1 timeout, reset controller
nvme nvme1: Device not ready; aborting reset, CSTS=0x1
blk_update_request: I/O error, dev nvme1n1, sector 672466952 op 0x0:(READ)
EXT4-fs error (device nvme1n1): ext4_check_bdev_write_error:217:
             Error while async write back metadata
EXT4-fs error (device nvme1n1) in ext4_reserve_inode_write:5789: Journal has aborted
```

最后一次成功落盘：`tmp/driver_09181343.log` 的 mtime `2026-09-18 17:49:51`。

事后硬件做了调换：`/home/Public` 现挂在 `nvme0n1`，系统盘换成 `nvme1n1p2`。
数据完好，恢复后验证 5 个 `study.db` 的 `pragma integrity_check` 均为 `ok`。

## 中断时的进度：11/60

| 组合 | COMPLETE | 孤儿 RUNNING | 已完成值 |
|---|---|---|---|
| HE_Ki67 | 3 | T3 | T0=0.9407, T1=0.9258, T2=0.9258 |
| HE_PR | 3 | T3 | T0=0.9492, T1=0.9307, T2=0.9307 |
| HE_HER2_Ki67 | 2 | T2 | T0=0.9396, T1=0.9308 |
| HE_PR_ER | 2 | T2 | T0=0.9357, T1=0.9270 |
| HE_PR_HER2_ER | 1 | T1 | T0=0.9342 |

5 个组合的 Trial 0 全部完成，且**5/5 逐位复现历史 seed42**（差 0.00e+00）——
这条结论与被中断的搜索无关，作为 harness 正确性的证据保留在这里。

## 为什么不 resume

除了故障本身，这批 study 还带着一个**采样缺陷**：`T1 == T2`（HE_Ki67 都是
0.9258、HE_PR 都是 0.9307）不是巧合。

driver 用 `--trials-per-process 1`，每条 trial 都是新进程；而 `TPESampler` 在
`len(COMPLETE+PRUNED) < n_startup_trials` 时会把采样委托给内部的
`RandomSampler(seed)`，`LazyRandomState` 又在构造时重新 `seed(2026)`。于是
**每个新进程的 startup 抽样都从同一个 RNG 状态开始，永远抽到同一个点**。

5 个组合中断时 COMPLETE 数为 3/3/2/2/1，全部还在 startup 窗口内，直接 resume
会再白扔 9 条 trial。因此改为归档重建：新版 `optuna_v3.py` 在 startup 阶段
自己做确定性采样并入队，不再依赖那个会重置的 RNG。

## 目录内容

```
_interrupted_backup/
├── README.md                     本文件
├── HE_Ki67/{study.db,best_model.pt}      中断时的 study 与当时的 best checkpoint
├── HE_PR/...
├── HE_HER2_Ki67/...
├── HE_PR_ER/...
├── HE_PR_HER2_ER/...
└── tmp/
    ├── driver_09181343.log       5 条链 + GPU7 的完整 driver 日志（27 MB）
    ├── driver_console.log        启动时的 dry-run 与映射表
    ├── holdout_worker.log        GPU7 的等待日志（当时一直处于 [wait]）
    ├── <组合>_run.log            各链的 study 级日志
    └── trial_*/                  故障瞬间正在跑的 trial 目录（含孤儿 RUNNING 的现场）
```
