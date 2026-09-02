# 第二轮诊断报告 — HE/PR 双模态 Two-Stage R²T + ABMIL

> 项目：CAMELYON16（C16）WSI 二分类
> 路径：`/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT`
> 环境：`/home/cxl/miniconda3/envs/rrtmil/bin/python`
> 日期：2026-09-02
>
> **本轮范围**：仅做诊断。未修改任何模型结构 / 数学 / 训练协议。只新增了诊断脚本、只读 forward hook、attention 统计。所有结论均来自已训练 checkpoint 的实测数据。

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [Patch 对应关系（Task 1）](#2-patch-对应关系-task-1)
3. [RRT region 顺序依赖（Task 1 延伸）](#3-rrt-region-顺序依赖-task-1-延伸)
4. [Plain Concat 公平对照（Task 2）](#4-plain-concat-公平对照-task-2)
5. [CR-MSA Attention 矩阵（Task 3）](#5-cr-msa-attention-矩阵-task-3)
6. [跨模态 Enrichment（Task 3）](#6-跨模态-enrichment-task-3)
7. [Per-head 分析（Task 3）](#7-per-head-分析-task-3)
8. [Group 分析（Task 3）](#8-group-分析-task-3)
9. [Stage2 表征变化（Task 3）](#9-stage2-表征变化-task-3)
10. [最终 ABMIL 模态注意力（Task 3）](#10-最终-abmil-模态注意力-task-3)
11. [Q1–Q12 逐题结论](#q1q12-逐题结论)

---

## 1. 执行摘要

本轮三项任务全部完成，核心证据链如下：

| 任务 | 结论 | 关键数值 |
|------|------|----------|
| **Task 1 — HE/PR patch 行级对应** | **FAIL（顺序不一致，Case B）** | 10/10 幻灯片 count 相同，**0/10** 顺序一致；diagonal≈shuffled（gap 0.0001–0.0037）；argmax=id ≈ 0.003（≈随机 1/N）；平均位移 ≈ 133 |
| **Task 2 — Plain Concat 公平对照** | **Concat 有害，CR-MSA 仅"修复"到 HE-only** | HE-only 0.8494；Concat 0.8347；CR-MSA 0.8505；Δ_concat−HE = **−0.0147**；Δ_crmsa−concat = **+0.0158**；Δ_crmsa−HE = **+0.0011** |
| **Task 3 — CR-MSA / ABMIL 注意力** | **CR-MSA 内部 seed 依赖，ABMIL 稳定 HE 主导** | 4-block mass 跨 seed 大幅漂移（HH 0.45→0.74）；但 abmil_HE = 0.90–0.995（跨 seed 稳定） |

**一句话结论**：当前 Two-stage ≈ HE-only 的根因是 **Patch/region 顺序 correspondence 风险（F）** 叠加 **PR 在分类端几乎不被使用（A/B）**。CR-MSA 内部确实发生了跨模态 attention（尤其 PR→HE），但因为在 ABMIL 端 PR token 只拿到约 5% 权重，且 PR 本身与 HE 行级不对齐，PR 的跨模态信息没能转化为分类增益。

数值等价性校验：诊断脚本只挂了只读 hook，实测 4 张幻灯片 forward **max_abs_diff = 0.0**（挂 hook 前后数值完全相同），确认诊断未污染模型输出。

---

## 2. Patch 对应关系（Task 1）

### 2.1 结论：**Case B（set 匹配，但顺序不匹配）**

| 判定 | 结果 |
|------|------|
| 检查 WSI 数 | 10（5 normal + 5 tumor） |
| HE/PR row count 相同 | **10 / 10** |
| HE/PR row 顺序严格一致 | **0 / 10** |
| 结论 Case | **B**（特征集合大小一致，但行级顺序不一致） |

### 2.2 三条证据链

**(a) 实现层面（代码证据）**

- `feature_extract/c16_feature_ctrans.py:93-96` 用 **未排序** 的 `Path.rglob('*')` 枚举 patch 文件：
  ```python
  patch_files = [str(p) for p in Path(wsi_dir).rglob('*')
                 if p.suffix.lower() in {...} and p.stat().st_size > 0]
  ```
  `DataLoader(shuffle=False)` + `torch.cat(feats)`，行序 = 文件系统遍历序。
- `data/c16_multimodal_dataset.py:220` 的 `_build_indices` **只校验 count**（`feat.shape[0] != total_patches`）和 dim，**不校验顺序**。注释明言"共享 patch_indices 要求 HE/PR 的 patch 数量与顺序完全相同"——但该前提在数据侧不成立。
- `.pt` 是裸 tensor `[N, 768]`，**无任何 filename/coordinate 元数据**，无法事后恢复对应关系。

**(b) 特征相关性（实测）**

PR 是 HE 的虚拟染色（virtual-stain）变换，若 HE[i] 与 PR[i] 是同一 patch，则对角线余弦应显著高于打乱/偏移。实测：

| 指标（10 slide 均值） | 数值 | 判读 |
|----------------------|------|------|
| diagonal cos | ≈ 0.78 | 与 shuffled 几乎相同 |
| shuffled cos | ≈ 0.78 | — |
| **gap（diag − shuffled）** | **0.0001–0.0037** | 无对角线优势 ⇒ 行序不对齐 |
| argmax=id（最佳匹配=自身行） | ≈ **0.003** | ≈ 1/N（纯随机签名） |
| 平均位移 | ≈ **133** | 接近均匀随机排列的期望 |

若顺序一致，argmax=id 应 ≈ 1.0 且 gap ≫ 0。实测两者都接近"纯随机置换"。

**(c) 跨标志物一致性**

ER / HER2 / Ki67 三个独立标志物数据集复测，gap = −0.0013 ~ +0.0020，全部落在 Case B 区间，说明这是**系统性问题**，非 C16 特有。

### 2.3 无法进一步确认的部分（诚实声明）

源 JPEG 位于 `/media/kemove/data_hdd0/lillan/C16/C16_raw`（HE）与 `Results-PR`（PR），**当前未挂载**，且 `.pt` 无元数据。因此**无法做逐 patch 文件名/坐标比对**（该路线只能判 Case C）。上述 Case B 结论建立在"特征相关性对角线检验"之上，而非坐标级确认。

**Patch 文件名约定**（来自 `create_c16_he_links.sh`）：`normal_001_3_190.jpeg` = `{category}_{slide_idx}_{row}_{col}.jpeg`，即文件名本身编码了行列坐标。**但该坐标信息在 `.pt` 里已丢失**，而 rglob 又破坏了顺序，导致坐标信息双重复现性失效。

---

## 3. RRT region 顺序依赖（Task 1 延伸）

### 3.1 结论：顺序不一致会**影响** RRT region partition，但**不能自动归因为 AUC 差异的全部原因**（Q2）

### 3.2 机制

`region_partition` 将 1D token 序列 reshape 成 `sqrt(N)×sqrt(N)` 网格，region 归属完全由**行号**决定。若 HE[i] 与 PR[i] 不是同一 patch：

1. HE 侧 region 网格与 PR 侧 region 网格在**空间上不重叠**——HE 的第 k 个 region 与 PR 的第 k 个 region 覆盖的是不同位置的组织。
2. `routing_joint = torch.cat([routing_he, routing_pr], dim=1)` 把两个"空间错位"的 region 集合拼在一起喂给 InnerAttention。CR-MSA 的 A_HP / A_PH 跨模态块本质上是在两个**错位的空间坐标**之间做 attention。

### 3.3 但这不是 AUC 的唯一原因

- Task 2 显示：即使 CR-MSA 完全换成 identity concat（不涉及任何 region 跨模态 alignment），AUC 也只是 −0.0147（回到 HE-only 水平以下）。这说明 **CR-MSA 的跨模态 region alignment 对最终 AUC 的贡献本身就很小**（+0.0158，勉强抵消 concat 的伤害）。
- Task 3 显示：最终 ABMIL 把 ~95% 权重放在 HE token 上，PR token 只拿 ~5%。**即使 PR 的 region 全错位，它对分类的影响也几乎为零**——因为它几乎不被 ABMIL 使用。

因此顺序不一致是一个**真实的实现层风险**（必须修），但它通过"ABMIL 已基本忽略 PR"这条路径被**部分屏蔽**了，不是 AUC 的唯一/主导差异源。

---

## 4. Plain Concat 公平对照（Task 2）

### 4.1 协议

- 与最新 unified-LR 双阶段**完全相同**，唯一差异 `stage2_type='concat'`：
  ```
  z_final = torch.cat([z_he, z_pr], dim=1)   # 无跨染色融合
  ```
- 同一 3 seeds：42 / 123 / 456。
- encoder_cfg（HE 4/9/3、PR 8/15/5）、stage2_cfg、sampler、LR、optimizer、loss、ABMIL、split 全部不变。

### 4.2 结果

| seed | HE-only | Concat | CR-MSA (staining_msa) |
|------|---------|--------|----------------------|
| 42   | 0.8508  | 0.8418 | 0.8776               |
| 123  | 0.8936  | 0.8128 | 0.8528               |
| 456  | 0.8038  | 0.8495 | 0.8212               |
| **Mean ± Std** | **0.8494 ± 0.0449** | **0.8347 ± 0.0194** | **0.8505 ± 0.0283** |

### 4.3 Δ 值

| 对比 | Δ AUC | 判读 |
|------|-------|------|
| Δ_concat − HE | **−0.0147** | 直接拼接 PR 特征**有害**（稀释 HE） |
| Δ_crmsa − concat | **+0.0158** | CR-MSA 相对 concat **稳定正增益** |
| Δ_crmsa − HE | **+0.0011** | CR-MSA 最终只回到 HE-only 水平，无额外增益 |

### 4.4 解读

- **Plain concat 引入负贡献**：PR 的 2500 个 token 混入后，ABMIL 可用的有效 HE 信号被稀释，AUC 下降 0.0147。
- **CR-MSA 的作用是"修复"而非"增益"**：它把 concat 的伤害救回来（+0.0158），但最终只等于 HE-only（+0.0011）。换句话说，CR-MSA 的净贡献 ≈ **抵消 concat 的伤害**，而非在 HE 之上叠加 PR 的增量信息。

---

## 5. CR-MSA Attention 矩阵（Task 3）

四块划分：`split = region_num² = 16`（region_num=4 → 每模态 16 个 routing region，共 R=32 个）。

- `A_HH = A[0:16, 0:16]`（HE query → HE key）
- `A_HP = A[0:16, 16:]`（HE query → PR key）
- `A_PH = A[16:, 0:16]`（PR query → HE key）
- `A_PP = A[16:, 16:]`（PR query → PR key）

### 5.1 query-normalized 4-block mass（每行和为 1）

| seed | HH | HP | PH | PP | 主模式 |
|------|------|------|------|------|--------|
| 42   | 0.679 | 0.321 | 0.477 | 0.523 | HE-anchored（HE 强自注意，PR 均衡） |
| 123  | 0.446 | 0.554 | 0.796 | 0.204 | **跨模态主导**（HE→PR 且 PR→HE 都强） |
| 456  | 0.739 | 0.261 | 0.610 | 0.390 | HE-anchored（PR 明显偏向 HE） |
| **跨 seed 均值** | **0.621** | **0.379** | **0.628** | **0.372** | — |

### 5.2 关键观察（Q5）

- **HE→PR（HP）与 PR→HE（PH）mass 是 seed 依赖的**：seed 42 两者都低（0.32/0.48），seed 123 两者都高（0.55/0.80），seed 456 则是 PH 高 HP 低（0.61/0.26）。
- 唯一**跨 seed 稳定的不对称性**是：**PH（PR→HE）≥ HP（HE→PR）**，即 PR token 更倾向去 attend HE，而 HE token 相对不太看 PR。这在全部 3 个 seed 都成立（0.477>0.321、0.796>0.554、0.610>0.261）。
- 因此"HE→PR / PR→HE attention mass 到底是多少"没有单一答案——它随 seed 变化。但方向性是稳定的：**PR 更依赖 HE（PH 平均 0.628），HE 不太依赖 PR（HP 平均 0.379）**。

---

## 6. 跨模态 Enrichment（Task 3）

Enrichment = block mass / uniform expectation。uniform 下每块期望 share = `(keys in block)/R` = 0.5，故 enrichment = mass / 0.5 = 2×mass。**>1 表示高于 uniform（有偏好），=1 表示与 uniform 相同，<1 表示低于 uniform（回避）。**

### 6.1 每 seed enrichment

| seed | eHH | eHP | ePH | ePP |
|------|-----|-----|-----|-----|
| 42   | 1.36 | 0.64 | 0.95 | 1.05 |
| 123  | 0.89 | 1.11 | 1.59 | 0.41 |
| 456  | 1.48 | 0.52 | 1.22 | 0.78 |
| **均值** | **1.24** | **0.758** | **1.255** | **0.747** |

### 6.2 结论（Q6）

- **PR→HE 跨模态 attention 稳定高于 uniform**：ePH = 1.255（3 seed 中 2 个 >1.2，1 个 0.95），说明 **PR token 一致地更多去 attend HE**（把 HE 当"参考/锚点"）。
- **HE→PR 跨模态 attention 低于或接近 uniform**：eHP = 0.758（3 seed：0.64 / 1.11 / 0.52，两个显著 <1），说明 **HE token 一致地回避或忽略 PR**。
- 因此跨模态 interaction 是**不对称的**：PR 想从 HE 拿信息，但 HE 不给 PR 信息。这与第 10 节"ABMIL 95% HE"完全自洽。

---

## 7. Per-head 分析（Task 3）

每模态 routing 有 `crmsa_heads=8` 个 head。逐 head 的 4-block mass 显示 head 角色是**高度 seed 依赖**的，没有任何一个 head 在三个 seed 里稳定承担"HE↔PR 跨模态 specialist"角色。

### 7.1 seed 42 head 角色

| head | 模式 |
|------|------|
| 0,1,2,3,6 | HE-sink（HE、PR 都主要 attend HE） |
| **4, 7** | **PR-sink（HE、PR 都主要 attend PR）** ← 跨模态 specialist |
| 5 | diagonal（HE→HE、PR→PR 的模态内） |

### 7.2 seed 123 head 角色

| head | 模式 |
|------|------|
| **1, 2, 4, 6** | **跨模态（HE→PR 且 PR→HE 双向都强）** ← 跨模态 specialist |
| 0 | HE-dominant + PR→HE |
| 3, 7 | HE-sink |
| 5 | 混合 |

### 7.3 seed 456 head 角色

| head | 模式 |
|------|------|
| 0, 5, 7 | HE-sink |
| 1, 4 | HE-dominant |
| 2 | diagonal（模态内） |
| 3 | HE 混合 + PR→PR |
| 6 | PR→HE |

### 7.4 结论（Q7）

**不存在跨 seed 稳定的"跨模态 specialist head"。** head 角色在三个 seed 间被完全重组：

- seed 42 的跨模态 head（4、7）是"PR-sink"型；
- seed 123 的跨模态 head（1、2、4、6）是"双向"型；
- seed 456 里几乎没有明确跨模态 specialist。

这说明 CR-MSA 的 8 个 head 在训练中**没有收敛到一个稳定的功能分工**，而是每次 seed 都学到一种不同的注意力结构。这与第 5/6 节"块级 mass 随 seed 大幅漂移"一致——**CR-MSA 内部的跨模态 routing 本身是欠定/不稳定的**。唯一稳定的是聚合层面的不对称性（PR→HE > HE→PR）。

---

## 8. Group 分析（Task 3）

按 normal（label=0）/ tumor（label=1）、correct / incorrect 分组，看跨模态 attention 与 he_net/pr_net/abmil_he。

### 8.1 正常 vs 肿瘤（Q8）

| 指标 | normal | tumor | 判读 |
|------|--------|-------|------|
| 准确率 | ≈ **1.0** | ≈ **0.49–0.74** | 模型明显 **normal-biased**（几乎所有 normal 都判对，tumor 大量漏检） |
| HP+PH（跨模态总 mass） | 3 seed 均**略高** | 3 seed 均**略低** | tumor 跨模态 attention **并未更高** |
| abmil_HE | ≈ 0.92 | ≈ **0.978** | tumor **更**依赖 HE |

### 8.2 正确 vs 错误（Q9）

| seed | correct HP+PH | incorrect HP+PH | 判读 |
|------|---------------|-----------------|------|
| 42   | 0.791 | 0.828 | incorrect 更高 |
| 123  | 1.345 | 1.369 | incorrect 略高 |
| 456  | 0.866 | 0.895 | incorrect 更高 |

三个 seed 一致：**错误预测的跨模态 attention 略高于正确预测**（差约 0.03–0.04）。

### 8.3 结论

- **Q8：Tumor 并不比 Normal 更依赖跨模态 interaction。** 相反，三个 seed 一致显示 tumor 的跨模态 mass 略**低于** normal（差约 0.03–0.11），而 tumor 的 abmil_HE（0.978）反而**高于** normal（0.92）——肿瘤样本更紧地"抱紧" HE。
- **Q9：正确预测并不比错误预测有更高的跨模态 attention。** 三个 seed 一致显示**错误**预测的跨模态 attention 略**高**于正确预测。这是一个微妙但真实的方向性信号：**更多地去 attend PR（跨模态）与预测错误弱相关**，与"PR 是噪声/未对齐"的结论一致。

---

## 9. Stage2 表征变化（Task 3）

用 `‖after−before‖ / ‖before‖` 度量 Stage2 对 Stage1 RRT 表征的改动幅度（=1 表示彻底重写，=0 表示 identity 不变）。

### 9.1 结果

| 指标 | 定义 | 跨 seed 均值 |
|------|------|-------------|
| he_net | ‖z_final_HE − z_HE‖ / ‖z_HE‖ | **0.582** |
| pr_net | ‖z_final_PR − z_PR‖ / ‖z_PR‖ | **0.653** |
| residual_ratio（joint） | ‖delta_joint‖ / ‖concat‖ | 1.19–1.99（seed 依赖） |
| net_ratio（joint） | ‖z_final − concat‖ / ‖concat‖ | ~0.6 |

### 9.2 结论（Q10）

**Stage2 明显改变了 Stage1 的 RRT 表征，不是 near-identity。** 相对 delta ≈ 0.58–0.65，即 Stage2 把 HE 表征改写了约 58%、PR 表征改写了约 65%（相对各自范数）。其中 **PR 被改写得更重（0.653 > 0.582）**。

这排除了"Stage2 只是恒等透传"的可能性——CrossStainingCRMSA 的残差支路在做真实的工作。但结合第 4 节：这个"真实工作"的净效果只是把 concat 的伤害抵消掉，没产生新的分类增益。

---

## 10. 最终 ABMIL 模态注意力（Task 3）

### 10.1 结果

| seed | abmil_HE（HE token 注意力占比） | abmil_PR |
|------|-------------------------------|----------|
| 42   | **0.943** | 0.057 |
| 123  | **0.904** | 0.096 |
| 456  | **0.995** | 0.005 |
| **均值** | **0.947** | 0.053 |

### 10.2 结论（Q11）

**最终 ABMIL 是 HE 主导（HE-dominant）。** 三个 seed 一致把 90%–99.5% 的注意力权重放在 HE token 上，PR token 只拿到约 5% 权重。这是**跨 seed 最稳定**的一个信号（std 远小于 CR-MSA 内部的块级 mass 漂移）。

**重要反差**：CR-MSA 内部的 attention 是 seed 依赖的（第 5 节），但 ABMIL 端的模态偏好是**稳定 HE 主导**的。这意味着无论 CR-MSA 在中间学到什么样的跨模态结构，**分类器最终都选择忽略 PR**。

---

## Q1–Q12 逐题结论

> 每题仅依据上述实测证据回答，不做臆测。

**Q1：HE/PR feature row ordering 是否严格一致？**
**否。** 10/10 幻灯片 count 一致，0/10 顺序一致。Case B（set 匹配、顺序不匹配）。

**Q2：如果不一致，是否影响 RRT region partition？是否自动就是 AUC 原因？**
**会**影响 region partition（HE/PR 的 region 网格空间错位，CR-MSA 跨模态块在错位坐标间做 attention）。**但不自动是 AUC 原因**：Task 2/3 显示 CR-MSA 的跨模态 region alignment 对 AUC 贡献很小（+0.0158，仅抵消 concat 伤害），且 ABMIL 已 ~95% 忽略 PR，错位风险被部分屏蔽。

**Q3：最新公平协议下 Two-RRT + Plain Concat 的 Mean AUC？**
**0.8347 ± 0.0194**（seeds 42/123/456 = 0.8418 / 0.8128 / 0.8495）。

**Q4：CR-MSA 相比 Plain Concat 是否有稳定正增益？**
**是（有稳定正增益）。** Δ_crmsa − concat = **+0.0158**（CR-MSA 0.8505 vs concat 0.8347）。但此增益只把 CR-MSA 拉回到 HE-only 水平（Δ_crmsa − HE = +0.0011），未产生超过 HE-only 的额外增益。

**Q5：CR-MSA 训练后 HE→PR / PR→HE attention mass 到底是多少？**
**seed 依赖，无单一值。** HP（HE→PR）= 0.321 / 0.554 / 0.261（均值 0.379）；PH（PR→HE）= 0.477 / 0.796 / 0.610（均值 0.628）。跨 seed 稳定的方向性是 **PH > HP**（PR 更依赖 HE）。

**Q6：跨模态 attention 是否高于 uniform expectation？**
**分方向。** PR→HE（ePH=1.255）稳定**高于** uniform；HE→PR（eHP=0.758）**低于** uniform。即跨模态 interaction 是单向的：PR 想用 HE，HE 不用 PR。

**Q7：是否存在某些 attention heads 专门负责 HE↔PR？**
**无跨 seed 稳定的跨模态 specialist head。** 8 个 head 的角色在三个 seed 间被完全重组（seed42 是 PR-sink 型、seed123 是双向型、seed456 几乎无）。跨模态 routing 本身欠定/不稳定。

**Q8：Tumor WSI 是否比 Normal WSI 更依赖跨模态 interaction？**
**否。** 三 seed 一致显示 tumor 的跨模态 mass 略**低于** normal，且 tumor 的 abmil_HE（0.978）**高于** normal（0.92）——肿瘤反而更紧地依赖 HE。

**Q9：正确预测是否比错误预测有更高 cross-modal attention？**
**否，方向相反。** 三 seed 一致显示**错误**预测的跨模态 attention 略**高**于正确预测（差约 0.03）。更多 attend PR 与预测错误弱相关，与"PR 未对齐/噪声"一致。

**Q10：Stage2 是否明显改变 Stage1 RRT 表征？还是 relative delta 很小接近 identity？**
**明显改变，非 identity。** he_net ≈ 0.58、pr_net ≈ 0.65（相对各自范数改写 58%/65%），PR 被改写更重。但净效果仅抵消 concat 伤害（第 4 节）。

**Q11：最终 ABMIL 是 HE 主导 / PR 主导 / 还是都使用？**
**HE 主导。** abmil_HE = 0.904–0.995（均值 0.947），PR 只拿 ~5%。这是全链路最稳定的信号。

**Q12：综合证据，当前 Two-stage ≈ HE-only 更接近哪种情况？**
**G（多个因素共同存在）**，具体组合为 **F（patch/region correspondence 风险）+ A/B（PR 未被分类端有效使用 / 仅用于恢复 HE baseline）**，辅以 **E（存在跨模态 interaction 但未形成分类增益）**：

- **F 成立**（Task 1）：行级顺序不匹配，实现层 correspondence 风险真实存在。
- **A/B 成立**（Task 3 第 10 节）：ABMIL 95% HE，PR 几乎不被使用；CR-MSA 的作用主要是"修复"concat 的稀释（Task 2），把 PR 从"负贡献"拉回"零贡献"。
- **E 部分成立**（Task 3 第 5–7 节）：CR-MSA 内部确实发生跨模态 interaction（尤其 PR→HE），但该 interaction 未转化为超过 HE-only 的分类增益。

**根因排序建议**：先解决 F（恢复/保证 HE↔PR patch 顺序一致），再评估 PR 是否还有信息量可被分类端利用。在当前"PR 与 HE 行级不对齐 + ABMIL 忽略 PR"的双重前提下，PR 无法提供超过 HE-only 的增量。

---

## 附录：诊断产物

| 文件 | 用途 |
|------|------|
| `scripts/diag_patch_alignment.py` | Task 1：HE/PR 行级对应性（对角线/偏移/argmax 检验） |
| `scripts/run_concat_control.py` | Task 2：Plain Concat 公平对照（3 seeds） |
| `scripts/diag_crmsa_attention.py` | Task 3：CR-MSA 4-block / enrichment / per-head / group / Stage2 delta / ABMIL 模态注意力（只读 hook） |
| `results/patch_alignment.json` | Task 1 每 slide 明细 |
| `results/twostage_r4_noepeg_samplerfix_concat/seed{42,123,456}/result.json` | Task 2 每 seed AUC |

所有诊断脚本均**只读**（hook 不修改 forward 输出），数值等价性校验通过（max_abs_diff = 0.0）。
