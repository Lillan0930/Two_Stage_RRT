# 第三轮诊断报告 — 因果消融（PR 是否真的影响最终 prediction）

> 项目：CAMELYON16（C16）WSI 二分类
> 路径：`/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT`
> 日期：2026-09-03
>
> **本轮范围**：只用已有训练 checkpoint，做测试时（test-time）因果消融。不训练、不改网络结构、不改权重。所有 ablation 通过只读 forward_pre_hook 实现，默认 `ablation=None` 时与原始 forward **逐位一致**。

---

## 0. Numerical Equivalence Test（必需）

| Seed | max_abs_diff(原始 vs 挂 hook 且 ablation=None) |
|------|----------------------------------------------|
| 42   | **0.0** |
| 123  | **0.0** |
| 456  | **0.0** |

诊断 hook 在 `ablation=None` 时完全不触碰 forward，三个 seed 全部逐位一致，确认诊断代码未改变正常数值结果。

---

## 1. PR Shuffle（Task 1）

实验：保持 HE_i 不变，用 derangement π（无固定点，i≠j）把 PR slide 打乱成 PR_{π(i)}，只破坏 slide-specific HE–PR 配对。≥5 个 shuffle seed。

| Seed | Full AUC | Shuffled AUC (mean ± std) | Δ = shuffled − full |
|------|---------:|--------------------------:|--------------------:|
| 42   | 0.8776   | 0.8851 ± 0.0045           | **+0.0075** |
| 123  | 0.8528   | 0.8623 ± 0.0021           | **+0.0095** |
| 456  | 0.8212   | 0.8266 ± 0.0013           | **+0.0054** |
| **均值** | 0.8505 | 0.8580 | **+0.0075** |

每次 5 个 shuffle seed 的 AUC（seed 42 / 123 / 456）：

- 42: 0.8839, 0.8893, 0.8778, 0.8903, 0.8839
- 123: 0.8622, 0.8645, 0.8587, 0.8640, 0.8620
- 456: 0.8276, 0.8253, 0.8281, 0.8247, 0.8273

**严格 count-preserved 子集（只保留 patch 数恰为 2500 的 115 个 slide，打乱后 token 数不变）：**

| Seed | Full | Shuffled (mean ± std) | Δ |
|------|-----:|----------------------:|---:|
| 42   | 0.8805 | 0.8827 ± 0.0023 | +0.0022 |
| 123  | 0.8978 | 0.8976 ± 0.0003 | −0.0002 |
| 456  | 0.8687 | 0.8681 ± 0.0006 | −0.0006 |

**结论**：打乱 PR 后 AUC **没有下降**——反而平均上升 +0.0075（全 129 slide）；在 count-preserved 的 115-slide 子集上 Δ ≈ 0（−0.0006 ~ +0.0022）。说明 **PR 的 slide-specific 信息对最终 prediction 没有正向贡献**。

---

## 2. Cross-block Ablation（Task 2）

测试时 mask routing-token attention logits（softmax 前），再按剩余 key 重新归一化：

- **Full**：不 mask。
- **No-Cross**：mask A_HP（HE q → PR k）与 A_PH（PR q → HE k），只留 A_HH、A_PP。
- **No HE←PR**：只 mask A_HP（切断 PR → HE-side representation），留 A_HH、A_PH、A_PP。

| Seed | Full | No-Cross | No HE←PR |
|------|-----:|---------:|---------:|
| 42   | 0.8776 | 0.8857 (**+0.0082**) | 0.8839 (**+0.0064**) |
| 123  | 0.8528 | 0.8551 (**+0.0023**) | 0.8528 (**+0.0000**) |
| 456  | 0.8212 | 0.8227 (**+0.0015**) | 0.8207 (**−0.0005**) |

- **Δ_no_cross = AUC_no_cross − AUC_full**：+0.0082 / +0.0023 / +0.0015（全部 ≥ 0）。
- **Δ_no_HP = AUC_no_HP − AUC_full**：+0.0064 / +0.0000 / −0.0005（≈ 0）。

**结论**：禁掉所有 HE↔PR cross attention（No-Cross）后 AUC **不降反略升**；只禁 HE←PR（A_HP）后 AUC **基本不变**。说明跨模态 attention（包括最重要的 PR→HE 路径）对分类**没有正向贡献**，甚至略有害。

---

## 3. Patch Alignment（Task 3）

源 JPEG 本轮已可访问（`/media/kemove/SANDISK ELE/C16-raw&translation/`，此前在 `data_hdd0` 上未挂载）。对 1 个 normal（normal_001）+ 1 个 tumor（tumor_001）从原 JPEG 用 `sorted()` 重提取，保存 `{features, filenames, coords}`。

### 3.1 文件名 / 坐标直接证据

| Slide | HE patch 数 | PR patch 数 | 坐标集合是否完全一致 | 排序后顺序是否一致 |
|-------|-----------:|-----------:|-------------------:|------------------:|
| normal_001 | 395 | 395 | **一致（set equal = True）** | **一致** |
| tumor_001  | 1488 | 1488 | **一致（set equal = True）** | **一致** |

文件名对应示例：HE `normal_001_28_212.jpeg` ↔ PR `28_212.jpg`（同一坐标 row=28, col=212）。HE 文件名编码 `{category}_{slide}_{row}_{col}`，PR 文件名编码 `{row}_{col}`（目录为 `{category}_{slide}`），坐标一一对应。

### 3.2 排序重提取后的 cosine（验证 cosine 方法本身）

| Slide | tile 级 diag | tile 级 shuffled | tile 级 gap | patch 级 diag | patch 级 shuffled | patch 级 gap |
|-------|-------------:|-----------------:|------------:|--------------:|------------------:|-------------:|
| normal_001 | 0.6327 | 0.4953 | **+0.1374** | 0.6580 | 0.5848 | **+0.0732** |
| tumor_001  | 0.6308 | 0.4566 | **+0.1742** | 0.6523 | 0.5321 | **+0.1203** |

### 3.3 现有 .pt（未排序 rglob 提取）cosine —— 同口径对比

| Slide | 现有 .pt diag | 现有 .pt shuffled | 现有 .pt gap |
|-------|--------------:|------------------:|-------------:|
| normal_001 | 0.4953 | 0.4938 | **+0.0015** |
| tumor_001  | 0.4560 | 0.4567 | **−0.0007** |

（注：现有 .pt 与排序重提取的 shuffled 值完全一致，说明两者含同一组特征，只是行序不同；但 diag 一个接近 0、一个接近 +0.14~0.17。）

### 3.4 判定

- **源数据（HE vs PR 原始 patch）：CONFIRMED ALIGNED** —— 坐标集合完全一致、排序后逐行一一对应。
- **现有 .pt 特征：CONFIRMED MISALIGNED** —— diag ≈ shuffled（gap ≈ 0），是 `c16_feature_ctrans.py:93-96` 用未排序 `Path.rglob('*')` 提取、且 `data/c16_multimodal_dataset.py:220` 只校验 count 不校验顺序所致。
- **cosine 方法本身可靠**：对排序重提取的特征，diag 显著高于 shuffled（+0.14~0.17）；对现有 .pt，diag ≈ shuffled。方法能正确区分 aligned / misaligned。

---

## 4. 最终回答（Q1–Q5）

**Q1：PR shuffle 后 AUC 是否明显下降？**
**否。** 3 个 seed 的 shuffled AUC 全部 ≥ Full（Δ = +0.0054 ~ +0.0095，均值 +0.0075）。PR 的 slide-specific 信息对最终 prediction **没有正向贡献**。

**Q2：禁掉所有 HE↔PR cross attention 后 AUC 是否下降？**
**否。** No-Cross 3 个 seed 全部 ≥ Full（Δ = +0.0015 ~ +0.0082）。跨模态 attention **没有正向贡献，甚至略有害**。

**Q3：只禁掉 HE←PR（A_HP）后 AUC 是否下降？**
**否。** No HE←PR 的 Δ = +0.0064 / +0.0000 / −0.0005（≈ 0）。PR → HE-side representation 这条路径 **没有正向贡献**。

**Q4：PR 是否确实通过 CR-MSA 对最终 HE-side representation / prediction 产生有效贡献？**
**否。** 三个独立的因果消融（打乱 PR、禁 cross attention、禁 HE←PR 路径）一致显示：PR 及其跨模态路径对最终 prediction 没有产生有效（正）贡献。结合第二轮证据（ABMIL ~95% 权重在 HE token），PR 在分类端事实上几乎不被使用。

**Q5：根据因果实验，当前 Two-stage ≈ HE-only 最接近哪种情况？**
**E（多个因素共同存在）**，具体是 **A（PR 基本没有实际贡献）+ D（当前 HE/PR correspondence 存在真实实现问题）**：

- **A 成立**：Q1/Q2/Q3 的因果消融直接证实 PR 及其跨模态路径无正向贡献。
- **D 成立**：Task 3 直接证实源数据对齐、但现有 `.pt` 因未排序 rglob 而 misaligned（diag ≈ shuffled，gap ≈ 0，而非对齐时的 +0.14~0.17）。
- **两者关系（因果链）**：D（提取时 misalignment）→ 跨模态 attention 在空间错位的 region 上运算（等同噪声）→ 模型在训练中学会忽略 PR（第二轮：ABMIL 95% HE）→ A（PR 无实际贡献）→ 残余的少量跨模态 attention 反而略有害（No-Cross ≥ Full）。

排除的选项：C 不成立（Q3 显示 HE←PR 路径无增益）；B 不成立（Q2 显示 cross fusion 略有害而非"PR 用于抵消融合干扰"）。

---

## 结论（是否要改 Stage2）

本轮因果证据明确：**在当前已训练的 checkpoint 上，PR 对最终 prediction 没有真实的正向影响，且这个影响也不是通过 HE←PR 跨模态路径产生的。**

但注意：**这不是"PR 本身无用"的最终判定**，而是"PR 在 `当前 misaligned 的 .pt` 上无用"。Task 3 已证明源 HE/PR 是对齐的、且 misalignment 来自提取环节。因此下一步（第四轮）的正确顺序是：

1. **先修复特征提取的 row-order**（用 `sorted()` 按坐标排序重提取 HE 与 PR 特征，保证 HE[i]↔PR[i] 同一 patch），
2. **再在修正后的对齐特征上重做因果消融 / 重新评估** PR 是否提供增量信息。

在此之前修改 Stage2 融合结构没有意义——因为输入的行级对应关系本身就是坏的，任何融合结构都建立在错位的 HE↔PR 之上。

---

## 附录：诊断产物

| 文件 | 用途 |
|------|------|
| `scripts/diag_causal_ablation.py` | Task 1（PR shuffle）+ Task 2（cross-block ablation）+ 数值等价校验 |
| `feature_extract/recheck_c16_alignment.py` | Task 3：normal_001 / tumor_001 用 sorted() 重提取 + filename/coords/cosine |
| `results/causal_ablation_seed{42,123,456}.json` | 每 seed 的 Full / No-Cross / No-HP / shuffle AUC |
| `results/recheck_alignment/recheck_alignment_report.json` | 重提取 cosine + 坐标一致性 |
| `results/recheck_alignment/{normal,tumor}_001_{HE,PR}_sorted.pt` | 排序重提取特征（含 filenames / coords） |

所有 ablation 均不改权重、不改结构、默认等价（max_abs_diff = 0.0）。
