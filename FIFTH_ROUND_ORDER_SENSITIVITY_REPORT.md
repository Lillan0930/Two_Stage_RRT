# 第五轮诊断报告 — HE-RRT / PR-RRT 对 patch token 顺序的依赖

> 项目：CAMELYON16（C16）WSI 二分类
> 路径：`/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT`
> 日期：2026-09-06
>
> **本轮目标**：不重新训练、不重新提取特征、不修改模型结构，只回答一个问题——
> **单模态 HE-RRT / PR-RRT 到底有多依赖当前 patch token 顺序？**
>
> **唯一动作**：测试时对已训练好的单模态 RRT checkpoint 做「同 WSI 内 patch token 行 shuffle」。

---

## 0. 实验对象

| 模态 | checkpoint（seeds 42/123/456） | 结构参数 | 协议 |
|------|-------------------------------|----------|------|
| HE-only RRT | `results/he_rrt_samplerfix_lr1e4/seed{42,123,456}/ckpt/best_model.pt` | region_num=4 / epeg_k=9 / crmsa_k=3 / n_heads=4 | test-as-val（`c16_test_labels.csv`，random 采样，max_patches=2500） |
| PR-only RRT | `results/c16_test_as_val/seed{42,123,456}/ckpt/best_model.pt` | region_num=8 / epeg_k=15 / crmsa_k=5 / n_heads=4 | test-as-val（`c16_test_labels.csv`，random 采样，max_patches=2500） |

两者均 `fusion_type='two_stage_region'`、`num_modalities=1`，即模型实际走 `MM_RRT_ABMIL.forward` 的单模态分支
`x_emb → rrt_he(RRTEncoder) → mil(ABMIL)`；checkpoint 里多余的 `rrt_encoder.*`（多模态 MM_RRTEncoder）参数在加载时被丢弃、不参与前向。

测试集 = 129 个 test slide（`c16_test_labels.csv`），评估方式与训练期 val 完全一致。

---

## 1. 数值等价性验证（必须满足）

在 `X[permutation]` 中取 identity permutation（`perm = arange(N)`），前向必须与「不 shuffle」完全一致。

| Modality | seed | max_abs_diff(identity vs original) |
|----------|------|-----------------------------------:|
| HE | 42 / 123 / 456 | **0.00e+00 / 0.00e+00 / 0.00e+00** |
| PR | 42 / 123 / 456 | **0.00e+00 / 0.00e+00 / 0.00e+00** |

全部为 `0`（bit-exact），证明 shuffle 机制本身（索引取行）不引入任何数值变化。

**附加正确性旁证**：本 harness 复现的「Original AUC」与训练记录的 `result.json` 完全一致
（HE seed42 = 0.8508 vs 记录 0.850765；seed123 = 0.8936 vs 0.893622；seed456 = 0.8038 vs 0.803827），
确认评估路径与训练期 val 逐位等价。

---

## 2. Intra-slide Shuffle 结果

每张 WSI 的 tile feature `X ∈ R^{N×768}` 在同 WSI 内随机重排行（`X_shuffle = X[perm]`），
feature 数值 / label / patch 数量完全不变，不跨 WSI 交换，不修改 checkpoint；每模型 5 个 shuffle seed。

| Modality | Model seed | Original AUC | Shuffled AUC (mean±std) | Δ (shuffled−orig) | 5 个单独 AUC |
|----------|-----------:|-------------:|------------------------:|------------------:|:-------------|
| HE | 42 | **0.8508** | 0.8306 ± 0.0061 | **−0.0202** | 0.8301 / 0.8314 / 0.8245 / 0.8416 / 0.8255 |
| HE | 123 | **0.8936** | 0.8874 ± 0.0021 | **−0.0062** | 0.8865 / 0.8876 / 0.8857 / 0.8913 / 0.8857 |
| HE | 456 | **0.8038** | 0.8130 ± 0.0061 | **+0.0091** | 0.8061 / 0.8130 / 0.8143 / 0.8079 / 0.8235 |
| PR | 42 | **0.7574** | 0.7557 ± 0.0070 | **−0.0017** | 0.7523 / 0.7446 / 0.7571 / 0.7587 / 0.7656 |
| PR | 123 | **0.7672** | 0.7420 ± 0.0154 | **−0.0252** | 0.7375 / 0.7319 / 0.7217 / 0.7638 / 0.7554 |
| PR | 456 | **0.7885** | 0.7906 ± 0.0129 | **+0.0021** | 0.7842 / 0.7760 / 0.8043 / 0.8077 / 0.7809 |

**跨 seed 汇总**：

| Modality | Δ 均值（3 seeds） | Δ 跨 seed 范围 | Δ 跨 seed std |
|----------|------------------:|---------------:|--------------:|
| HE | **−0.0058** | −0.0202 … +0.0091 | 0.0147 |
| PR | **−0.0083** | −0.0252 … +0.0021 | 0.0148 |

---

## 3. 解释

RRT 里唯一依赖顺序的环节是：token `i` → 二维位置 `(i//W, i%W)` → `region_partition` 划进
`region_num×region_num` 个窗口 → 窗口内 R-MSA + EPEG 位置编码 → CR-MSA 跨区域路由。
shuffle 会改变每个 token 落入的窗口、以及 EPEG 赋予的邻域上下文。但最终 ABMIL 是对
所有 token 的**注意力加权全局池化**（permutation-invariant），加上 `all_shortcut` 残差
`x = x + x_shortcut` 把原始 token 特征直通到输出，所以窗口级扰动在全局聚合后被大幅稀释。

实测结论与这一机制一致：

- **HE-RRT**：Δ 均值 −0.006，但三个 seed 分别 −0.020 / −0.006 / **+0.009**，符号不一致；
  跨 seed std（0.015）大于均值绝对值。**无稳健的负向效应。**
- **PR-RRT**：Δ 均值 −0.008，三个 seed 分别 −0.002 / −0.025 / **+0.002**，同样符号不一致；
  跨 seed std（0.015）大于均值绝对值。**无稳健的负向效应。**
- 单模型内的 shuffle-to-shuffle 噪声（std 0.006–0.015）与 |Δ| 同量级甚至更大，说明 Δ 主要来自
  shuffle 抽样噪声，而非「顺序被破坏」的系统性伤害。

---

## 4. 最终回答（只回答三个问题）

### Q1：HE-RRT 是否对 intra-slide patch 顺序敏感？

**不敏感（或极弱）。** 平均 Δ ≈ −0.006，三个 seed 中有一个为正（+0.009），符号不一致，
幅度（≤0.02）落在 shuffle 抽样噪声量级内，无法构成「顺序敏感」的证据。

### Q2：PR-RRT 是否对 intra-slide patch 顺序敏感？

**不敏感（或极弱）。** 平均 Δ ≈ −0.008，三个 seed 中同样有一个为正（+0.002），
符号不一致、幅度（≤0.025）在噪声量级内，不构成敏感证据。

### Q3：是否有足够证据说明「未 coordinate-sorted」显著损害现有单模态 RRT 性能？

**没有。** 推理链：

1. 若 RRT 已经学会利用当前（未排序）顺序里的某种空间结构，那么「进一步随机打乱顺序」会破坏它已学到的结构 → AUC 应显著下降。
2. 实测进一步打乱几乎不改变 AUC（|Δ| ≤ 0.025，均值 −0.006~−0.008，符号不稳定）→ RRT **没有**依赖当前顺序的结构。
3. 因此当前未排序顺序**不是**单模态 RRT 的性能瓶颈；单靠「按坐标重排序重新提取特征」**预计无法**提升 HE-only / PR-only RRT 的 AUC。

**一句话结论**：单模态 HE-RRT / PR-RRT 对 patch token 顺序**近似不变**，现有未排序特征并未显著拖累单模态 RRT。

> **边界说明**（不展开，仅记录）：本结论只针对**单模态** RRT。两阶段 HE+PR 里「HE[i] ↔ PR[i] ↔ 同一坐标」
> 的跨模态对齐是另一个独立问题——它关系到 Stage-2 跨染色 CR-MSA 的路由是否把 HE 区域配到对应 PR 区域，
> 不是本轮的 intra-slide 单模态 shuffle 能覆盖的。是否需要在两阶段设定里重提取对齐特征，应由后续轮次单独判定。

---

## 附录：产物与约束

| 文件 | 用途 |
|------|------|
| `scripts/diag_order_sensitivity.py` | 单模态 RRT 的 intra-slide shuffle 诊断脚本 |
| `results/order_sens_{he,pr}_seed{42,123,456}.json` | 6 组原始/shuffle AUC、Δ、数值等价性 |

**本轮未**：重新训练、重新提取特征、修改模型结构 / RRT / CR-MSA / ABMIL、修改 checkpoint、讨论新 Stage2、开始全量重提取。
