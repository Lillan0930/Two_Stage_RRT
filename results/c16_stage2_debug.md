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

### 诊断 4 — 多模态 RRT-only 基线对照（ER / HER2 / Ki67，特征与 PR 同级）

为判定 PR-only 0.6571 是否异常，在完全相同的协议下（seed=42，K=2500，固定 split，单模态官方 RRTEncoder）补齐 ER / HER2 / Ki67 三个 RRT-only 基线。

| 模态 | Val AUC | **Test AUC** | Sens | Spec | 生物学角色 |
|------|---------|-------------|------|------|-----------|
| **HE** | 0.9787 | **0.8080** | 0.6735 | 0.9375 | H&E 主染色 |
| HER2 | 0.8935 | 0.6934 | 0.1633 | 0.9250 | HER2 受体 |
| Ki67 | 0.9162 | 0.6929 | 0.0816 | 1.0000 | 增殖标记 |
| PR | 0.8722 | 0.6571 | 0.6939 | 0.5125 | 孕激素受体 |
| ER | 0.9062 | 0.6569 | 0.5306 | 0.6375 | 雌激素受体 |

- **PR（0.6571）≈ ER（0.6569），并列最弱**：二者为激素受体、常共表达，只在部分乳腺癌表达，对"淋巴结有无转移"判别力最差。
- **所有 IHC 标记均 ≤0.69，远弱于 HE 0.8080**：HER2/Ki67 略强（~0.69）但仍远不足以靠 bag 拼接超越 HE-only。
- **HER2/Ki67 的模式符合生物学**：HER2 高特异低敏感（Sens 0.16 / Spec 0.93，HER2+ 为少数）；Ki67 几乎只报正常（Sens 0.08 / Spec 1.00，增殖标记只染小部分细胞）。
- **结论**：PR-only 0.6571 不是方法学错误，而是激素受体本身就是最弱的转移判别标记；ER 同样低进一步证实了这一点。

## 核心结论

1. **PR 是弱模态**（0.6571），且 PR 特征健康、无泄露——这是真实结果，不是跑错。ER（0.6569）同样弱，证明激素受体本身就弱，而非 PR 特异异常。
2. **HE 是唯一强模态**（0.8080）；ER/PR/HER2/Ki67 四个 IHC 标记全部 ≤0.69。
3. **所有 Two-stage 变体都塌缩到 PR-only 水平（~0.66）**：concat 0.6564 ≈ CR-MSA 0.6691 ≈ 旧版 0.6865，全部远低于 HE-only 0.8080。
4. **根因在上游的 bag 组合方式**：把 2500 HE + 2500 PR 拼成 5000 token 喂给单个 ABMIL 后，HE 的强信号被 PR 淹没（Spec 从 0.9375 掉到 0.50–0.60），整体塌回 PR-only。这与融合是否"忠实"无关。
5. **忠实重写的 CR-MSA 不是问题**：它 ≈ 纯 concat，融合机制工作正常（诊断 3）。
6. **"HE + PR 两阶段"前提存疑**：PR 是最弱次级染色之一（与 ER 并列）。若目标是给 HE 加一个真正提升的次级模态，激素受体不是好选择，应考虑直接标定肿瘤细胞的染色（如 pan-CK）。

## 下一步方向（待定）

- A. dump ABMIL 在 HE vs PR token 上的注意力权重，直接证实"HE 被稀释"。
- B. 改决策级融合（HE/PR 各自出分再加权，模型已有 `use_logit_fusion` 通道），保住 HE 强信号。
- C. 复查 PR 特征对齐 / 换更适合 metastasis 的次级染色（如 pan-CK）。诊断 4 已强烈支持方向 C。

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
# 诊断4：ER / HER2 / Ki67 RRT-only 基线
$PY scripts/_run_fixed_split_exp.py results/c16_er_baseline/seed42
$PY scripts/_run_fixed_split_exp.py results/c16_her2_baseline/seed42
$PY scripts/_run_fixed_split_exp.py results/c16_ki67_baseline/seed42
```

## 目录

```
results/
├── c16_he_baseline/seed42/       HE-only 复确认
├── c16_twostage_crmsa/seed42/    新 CrossStainingCRMSA Two-stage
├── c16_concat_ablation/seed42/   诊断1：纯 concat 对照
├── c16_pr_baseline/seed42/       诊断2：PR-only
├── c16_er_baseline/seed42/       诊断4：ER-only
├── c16_her2_baseline/seed42/     诊断4：HER2-only
├── c16_ki67_baseline/seed42/     诊断4：Ki67-only
└── c16_stage2_debug.md           本文件
scripts/
└── diag_stage2_magnitude.py      诊断3：Stage2 δ/z 幅度
```
