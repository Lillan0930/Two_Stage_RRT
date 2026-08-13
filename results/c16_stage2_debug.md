# Two-stage CR-MSA 调试记录（Stage2 忠实重写后）

**日期**: 2026-08-13
**项目**: Two_Stage_RRT (Lillan0930/Two_Stage_RRT)
**协议**: 与 `results/c16_fixed_split/README.md` 完全一致（K=2500，per-epoch random train / fixed random val / fixed random test，StratifiedShuffleSplit(random_state=42)，seed=42）

## 背景

把 Stage2 从简化版 `CrossStainingRegionMSA` 重写为官方式 `CrossStainingCRMSA` 后，seed=42 单跑 Two-stage 得到 **Test AUC 0.6691**，远低于 HE-only 0.8080，甚至略低于旧版 0.6865。为定位原因，做了三项诊断。

## 结果总表（全部 seed=42）

| 实验 | Val AUC | **Test AUC** | Sens | Spec | 说明 |
|------|---------|-------------|------|------|------|
| HE-only | 0.9787 | **0.8080** | 0.6735 | 0.9375 | 基线，两次复现一致 |
| PR-only | 0.8722 | 0.6571 | 0.6939 | 0.5125 | 诊断 2 |
| Two-stage 纯 concat | 0.9077 | 0.6564 | 0.5102 | 0.6000 | 诊断 1 |
| Two-stage CR-MSA（新） | 0.8949 | 0.6691 | 0.7347 | 0.5000 | 忠实重写版 |
| Two-stage（旧 CrossStainingRegionMSA） | 0.8736 | 0.6865 | 0.6735 | 0.5375 | fixed-split 5-seed 均值 0.7187 |

## 三项诊断

### 诊断 1 — 纯 concat 对照（去掉跨染色融合）

把 Stage2 换成 `z_final = concat([z_he, z_pr])`（不做任何跨染色注意力），直接 ABMIL。

- 结果：**Test AUC 0.6564**，与完整 CR-MSA 0.6691 几乎相同（差 0.013，噪声级）。
- **结论**：融合机制不是元凶。哪怕最简单的 concat，结果一样差。

### 诊断 2 — PR-only 基线

只跑 PR 单模态（`num_modalities=1` 走官方 RRTEncoder，与 HE-only 同构）。

- 结果：**Test AUC 0.6571**。
- **方法论核验**：PR 特征 std=0.1476（HE=0.1448，非退化）；模型参数 4,868,052 与 HE-only 完全一致；固定 split 无泄露。
- **结论**：PR 对 CAMELYON16 metastasis 任务是**弱模态**。生物上合理——PR（孕激素受体）只在部分乳腺癌（PR+）表达，不是判别"淋巴结有无转移"的好标记。

### 诊断 3 — Stage2 更新幅度（脚本 `scripts/diag_stage2_magnitude.py`）

在训练好的 CR-MSA checkpoint 上手工重放 Stage2 内部，测 ‖δ‖/‖z‖。

| 量 | mean | median | max |
|----|------|--------|-----|
| ‖z_he‖ | 1041 | 1095 | 1100 |
| ‖z_pr‖ | 1072 | 1128 | 1129 |
| δ_he / z_he | 0.981 | 1.070 | 1.379 |
| δ_pr / z_pr | 1.160 | 1.214 | 1.763 |

- **结论**：δ 与 z 同量级（≈1.0），跨染色融合在**实质重写**特征（不是 no-op，也不是爆炸）。融合是"真在起作用，但帮倒忙"。

## 核心结论

1. **PR 是弱模态**（0.6571），且 PR 特征健康、无泄露——这是真实结果，不是跑错。
2. **所有 Two-stage 变体都塌缩到 PR-only 水平（~0.66）**：concat 0.6564 ≈ CR-MSA 0.6691 ≈ 旧版 0.6865，全部远低于 HE-only 0.8080。
3. **根因在上游的 bag 组合方式**：把 2500 HE + 2500 PR 拼成 5000 token 喂给单个 ABMIL 后，HE 的强信号被 PR 淹没（Spec 从 0.9375 掉到 0.50–0.60），整体塌回 PR-only。这与融合是否"忠实"无关。
4. **忠实重写的 CR-MSA 不是问题**：它 ≈ 纯 concat，融合机制工作正常（诊断 3）。

## 下一步方向（待定）

- A. dump ABMIL 在 HE vs PR token 上的注意力权重，直接证实"HE 被稀释"。
- B. 改决策级融合（HE/PR 各自出分再加权，模型已有 `use_logit_fusion` 通道），保住 HE 强信号。
- C. 复查 PR 特征对齐 / 换更适合 metastasis 的次级染色（如 pan-CK）。

## 复现

```bash
cd /home/Public/lillan/Two_Sage_RRT-/TwoStageRRT
PY=/home/cxl/miniconda3/envs/rrtmil/bin/python

# HE-only 复确认
$PY scripts/_run_fixed_split_exp.py results/c16_he_baseline/seed42
# 新 CR-MSA Two-stage
$PY scripts/_run_fixed_split_exp.py results/c16_twostage_crmsa/seed42
# 诊断1：concat 对照
$PY scripts/_run_fixed_split_exp.py results/c16_concat_ablation/seed42
# 诊断2：PR-only
$PY scripts/_run_fixed_split_exp.py results/c16_pr_baseline/seed42
# 诊断3：delta 幅度
$PY scripts/diag_stage2_magnitude.py
```

## 目录

```
results/
├── c16_he_baseline/seed42/       HE-only 复确认
├── c16_twostage_crmsa/seed42/    新 CrossStainingCRMSA Two-stage
├── c16_concat_ablation/seed42/   诊断1：纯 concat 对照
├── c16_pr_baseline/seed42/       诊断2：PR-only
└── c16_stage2_debug.md           本文件
scripts/
└── diag_stage2_magnitude.py      诊断3：Stage2 δ/z 幅度
```
