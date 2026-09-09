# HE Residual Cross v3 —— Dataset-Prototype Centering 报告

**结论（一句话）：v3 在 AUC 上「看起来」涨了（0.8744 vs HE-only 0.8367，+0.038），但因果消融证明这个涨幅**全部来自联合训练变强的 HE encoder**（`disable_cross`=0.8741 ≈ matched 0.8744），PR 残差本身在推理时贡献 ≈ 0（Δ≈±0.0003）。判定：**FAILED**（残差坍缩，机制层面 v3 未达成 `(HE,PR) 真实优于 HE`）。

---

## 0. 摘要

| 指标 | v3 | HE-only | v1 | v2 |
|---|---|---|---|---|
| mean AUC (3 seeds) | **0.8744** | 0.8367 | 0.8622 | 0.8437 |
| per-seed AUC | 0.8753 / 0.8755 / 0.8724 | 0.8344 / 0.8615 / 0.8143 | 0.8406 / 0.8908 / 0.8551 | 0.8418 / 0.8755 / 0.8139 |

| 因果（均值） | matched | disable_cross | random_mismatch | within_class_mismatch |
|---|---|---|---|---|
| AUC | 0.8744 | 0.8741 | 0.8744 | 0.8744 |

- `matched ≈ disable`（Δ≈±0.0003）→ PR 残差对推理贡献 ≈ 0。
- `matched == random_mismatch == within_class_mismatch`（逐 seed 完全相等）→ 残差与 PR 内容无关。
- 关键诊断：`rho = ‖0.1·Δ‖/‖z_he‖` = **0.0207 / 0.0023 / 0.0002**（残差坍缩）；`entropy_norm` = 0.99–0.9999（**注意力近乎均匀**）。

---

## 1. §1 两个快速确认（Phase 1，基于 v2 checkpoint，无重训）

### A. PR centroid residual test —— **v3 假设机制上成立**

对每个 PR slide 取 Stage2 value centroid `c_i = mean_t V_{i,t}`，用 train 集计算 `μ_train = mean_i c_i`，再在 val-as-test 上算 `δ_i = c_i − μ_train`：

| seed | cos_c（原始质心） | cos_δ（去公共质心后） | ‖δ‖/‖c‖ (ratio_mean) | δ 有效秩 |
|---|---|---|---|---|
| 42 | 0.9643 | **0.0402** | 0.183 | 35.3 |
| 123 | 0.9828 | **0.0494** | 0.118 | 29.0 |
| 456 | 0.9928 | **0.0866** | 0.087 | 19.3 |

- 原始 PR 质心**高度共线**（cos_c 0.96–0.99）；减去 train 集公共质心后 `cos_δ` 骤降到 0.04–0.09，**不同 slide 重新获得可分辨的 slide 特异性**。
- `ratio_mean` = 0.087–0.183（非 ≈0），`δ` 有效秩 19–35（非 1）→ **δ_i 不是近似 0，slides 也不是相同**。
- §2 判定：**继续 v3**（δ_i 非零且 slide 有区分度）。

### B. HE X→Embed 控制 —— **patch_to_emb 非 PR 特异坍缩源**

| | pairwise_cosine | effective_rank | token_variance |
|---|---|---|---|
| HE_X | 0.713 | 217.3 | 0.0059 |
| HE_E (768→512) | 0.796 | 152.3 | 0.0033 |
| PR_X | 0.777 | 180.2 | 0.0048 |
| PR_E (768→512) | 0.899 | 85.6 | 0.0018 |

- HE 与 PR **同方向**被压缩（cos↑、秩↓、方差↓）；这是共享 `patch_to_emb` 的**共享行为**，不是 PR 特异。→ 按 §16，**不改 patch_to_emb**。

---

## 2. v3 实现（`models/he_residual_cross_crmsa_v3.py`）

v3 继承 v2（cosine τ=0.2 + bias-free QKV + 独立 routing + HE identity 残差 + residual_scale=0.1），唯一核心改动：

```
v2 centering (per slide):  Ṽ_{i,t} = V_{i,t} − mean_t(V_{i,t})   (删掉 slide-global + dataset-common)
v3 centering (prototype): Ṽ_{i,t} = V_{i,t} − μ_PR               (只删 dataset-common，保留 slide-global)
```

`μ_PR` 是 **EMA running prototype**（β=0.99），只从 train PR 学习（train-only `torch.no_grad()` 更新，eval 冻结），`register_buffer`（不进 optimizer、不带梯度、自动 checkpoint）。6 项实现测试全部通过（prototype train/eval 门控、dataset-common 删除+slide-specific 保留、最终输出 PR 内容敏感性、fallback identity、mask/空-PR/NaN、梯度）。冒烟测试确认完整训练管线可构建/训练（6,319,531 可训参数全被 optimizer 覆盖，μ_PR 首步 0→92.9 初始化）。

---

## 3. 训练结果（§11，Train/Val-as-Test，3 seeds，LR=1e-4，wd=1e-5，batch=1，80 epochs，cosine，patience=10）

| Model | seed42 | seed123 | seed456 | mean ± std |
|---|---|---|---|---|
| HE-only（复用） | 0.8344 | 0.8615 | 0.8143 | 0.8367 ± 0.0193 |
| v1（复用） | 0.8406 | 0.8908 | 0.8551 | 0.8622 ± 0.0211 |
| v2（复用） | 0.8418 | 0.8755 | 0.8139 | 0.8437 ± 0.0252 |
| **v3** | **0.8753** | **0.8755** | **0.8724** | **0.8744 ± 0.0014** |

paired ΔAUC（逐 seed）：

| Δ | seed42 | seed123 | seed456 | mean ± std |
|---|---|---|---|---|
| v3 − HE | +0.041 | +0.014 | +0.058 | **+0.0377 ± 0.0181** |
| v3 − v1 | +0.035 | −0.015 | +0.017 | +0.0122 ± 0.0209 |
| v3 − v2 | +0.033 | −0.000 | +0.058 | +0.0307 ± 0.0240 |

表面结论：`(HE,PR) > HE` 在 **3/3 seeds** 成立（paired Δ 全为正）。

---

## 4. 因果评估（§12-13，5 replacement seeds）

| seed | matched | disable_cross | Δ(matched−disable) | random_mm (5 seeds) | within_class_mm (5 seeds) |
|---|---|---|---|---|---|
| 42 | 0.8753 | 0.8750 | **+0.0003** | 0.8753（全部相同） | 0.8753（全部相同） |
| 123 | 0.8755 | 0.8758 | **−0.0003** | 0.8755（全部相同） | 0.8755（全部相同） |
| 456 | 0.8724 | 0.8724 | **+0.0000** | 0.8724（全部相同） | 0.8724（全部相同） |

- `matched ≈ disable`（|Δ| ≤ 0.0003）→ 去掉 PR cross（纯 z_he 进 ABMIL）AUC 基本不变。
- `matched == random_mismatch == within_class_mismatch`（逐 seed 到小数点后 4 位完全相等）→ 交换 PR slide（含同类别内交换）**零影响**。

**关键对照**：`disable_cross`（= v3 模型的 HE encoder 直接进 ABMIL）AUC = **0.8741**，远高于 HE-only 基线 0.8367。⇒ v3 相对 HE-only 的 +0.0377 涨幅**几乎全部来自联合训练变强的 HE encoder**，而非 PR 残差在推理时的贡献。

---

## 5. 原型诊断（§9，val-as-test 逐 slide）

| 指标 | seed42 | seed123 | seed456 | 含义 |
|---|---|---|---|---|
| **rho** = ‖0.1·Δ‖/‖z_he‖ | 0.0207 | 0.0023 | **0.0002** | 最终残差相对 z_he 近乎为 0（坍缩） |
| **entropy_norm** | 0.992 | 0.998 | 0.9999 | 注意力**近乎均匀**（1=最大熵） |
| score_std | 0.170 | 0.105 | 0.029 | 注意力 logit 几乎无区分度 |
| value_diversity | 0.993 | 0.999 | 0.9995 | PR value 高度共线（坍缩） |
| selective_ratio | 0.219 | 0.141 | 0.034 | A·Ṽ / A·V（≈ r_slide，见下） |
| **r_slide** | 0.217 | 0.139 | 0.034 | ‖mean(V_i)−μ_PR‖/‖mean(V_i)‖（slide-global 残余存在） |
| **r_region** | 0.230 | 0.141 | 0.039 | mean_t ‖V_{i,t}−μ_PR‖/‖V_{i,t}‖（region 残余存在） |
| μ_norm = ‖μ_PR‖ | 19.84 | 15.00 | 10.81 | 原型幅值可观（公共分量占主导） |

### 诊断解读（失败机理）

1. v3 的 prototype centering **确实**保留了 slide/region 特异性结构：`r_slide`/`r_region` = 0.03–0.23 非零（这是 v3 相比 v2 的进步——v2 的 per-slide 居中把这些信息删成了 0）。
2. 但 `entropy_norm ≈ 0.99–0.9999` → 注意力**均匀**（`selective_ratio ≈ r_slide`，即 A·Ṽ 退化为 slide-global 分量 D_i，而非 region 选择性信号）。
3. `w_out` 投影 + `residual_scale=0.1` 把这个本就很小的 D_i 进一步压到 `rho ≈ 0.0002–0.02`。
4. 净效果：PR 残差在推理时 ≈ 0，`matched ≈ disable ≈ mismatch`。

**根因**：跨模态注意力**学不会「选择性」**（entropy≈1、score_std≈0、value_diversity≈1），与居中方案无关——无论 per-slide（v2）还是 dataset-prototype（v3）居中，只要注意力均匀，残差就坍缩。

---

## 6. 成功判定（§14-15）

- 首要目标 `(HE_i, PR_i) > HE_i`（PR 残差**真实**贡献）：**FAILED**。AUC 数字上成立，但 `matched ≈ disable`（Δ≈±0.0003）证明这是 HE encoder 联合训练的产物，不是 PR 残差的推理贡献。
- 次要目标 `(HE_i, PR_i) > (HE_i, PR_j)`（内容敏感）：**FAILED**。`matched == random_mismatch == within_class_mismatch`（完全相等）。
- 判定：**FAILED**（残差坍缩；不是 TRUE/PARTIAL/GLOBAL-PRIOR，因为残差根本不为 0 也不贡献）。

---

## 7. 事实与建议（§16）

**事实（不臆测）**：
- v3 相对 HE-only 的 AUC 涨幅（+0.0377）可复现于 3 seeds，但**可完全归因于 `disable_cross` 路径（0.8741）**，即联合训练让 HE encoder 更强。
- PR cross-attention 注意力均匀、PR value 共线、最终残差 rho≈0 —— 三个症状一致，指向**「PR 路由 token 坍缩 / 注意力不具选择性」**这一老问题（与 round-6/7/8 一致）。
- v3 的 prototype centering 在**中间表示层面**成功保留了 slide-specific 信息（r_slide/r_region 非零），但被均匀注意力 + w_out + 0.1 缩放压平。

**按 §16 约束，本轮不自动修改** `patch_to_emb`、RRT、routing、loss 或 `residual_scale`。后续若继续，优先级应是**先让 cross-attention 变选择性**（例如直接诊断/修正 PR 路由 token 的坍缩，或对注意力施加非均匀约束），而不是再改居中方案——居中方案（v2/v3）已经证明不是瓶颈。

---

## 8. 复现入口

- 训练驱动：`scripts/run_stage2_v3.py --gpus <g1> <g2>`
- 评估：`scripts/eval_v3.py --section repro causal diagnostics summary --gpu <g>`
- 实现测试：`python tests/test_he_residual_cross_v3.py`
- Phase 1 确认：`scripts/diag_v3_prototype_centroid.py --gpu <g> --seed 42 123 456`
- 结果目录：`results/stage2_he_residual_cross_v3/`（`summary.json`、`_eval/{causal,diagnostics,repro,summary}.json`）
