# TRUE Multimodal Fusion — HE Residual Cross v2 报告

**日期**: 2026-09-07
**对象**: `models/he_residual_cross_crmsa_v2.py`（`HEResidualCrossCRMSAv2`）
**协议**: Train / Val-as-Test（train=270 / val-as-test=129，val-as-test 同时用于每 epoch 验证、早停、best checkpoint 选择、最终报告）
**产物目录**: `results/stage2_he_residual_cross_v2/`（训练） + `_eval/`（评估）
**评估脚本**: `scripts/eval_v2.py`

---

## 0. 最终评级

# **FAILED**

**v2 的三个改动（cosine cross-attention τ=0.2 + PR value centering + bias-free QKV）全部实现正确、5/5 测试通过、3/3 可复现——但 causal 检验证明 cross-attention 已完全失效（残差塌缩为 ≈0），模型在功能上退化为 HE-only。+0.007 相对 HE-only 的提升与 PR 完全无关。**

一句话：**centering 成功杀掉了 v1 的「内容无关平均 PR 向量」残差（ρ 从 0.08–0.13 → ≈1e-4–1e-7），但模型没有学会用 selective attention 补位，于是 cross 残差直接塌缩为 0，cross 机制比 v1 更惰性。**

---

## 1. 主结果

**Evaluation protocol: Train / Val-as-Test**（HE-only 与 v1 复用上一轮 `results/stage2_he_residual_cross_val_as_test/`，不重训）

| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| HE-only (reuse) | 0.8344 | 0.8615 | 0.8143 | 0.8367 ± 0.0193 |
| v1 HE Residual Cross (reuse) | 0.8406 | 0.8908 | 0.8551 | 0.8622 ± 0.0211 |
| **v2 HE Residual Cross** | 0.8418 | 0.8755 | 0.8139 | **0.8438 ± 0.0252** |

**paired ΔAUC（逐 seed + 均值）**

| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| v2 − HE | +0.0074 | +0.0140 | −0.0004 | **+0.0070 ± 0.0059** |
| v2 − v1 | +0.0013 | −0.0153 | −0.0412 | **−0.0184 ± 0.0175** |

- v2 相对 HE-only 的 +0.0070 **小于** v1 的 +0.0254，且 3 seed 中 1 个为负（seed456 −0.0004）。
- v2 相对 v1 **下降** −0.0184（3 seed 中 2 个为负，seed456 大幅 −0.0412）。

---

## 2. 实现验证 —— 三改动实现正确、测试全过、可复现 ✅

### 2.1 三改动逐项核对

| 改动 | 结论 | 证据 |
|---|---|---|
| Cosine cross-attention，τ=0.2 固定（不可学习、不搜索） | ✅ | `Q̂=Q/‖Q‖+ε`、`K̂=K/‖K‖+ε`、`S=Q̂K̂ᵀ/τ`、`τ=0.2` 为普通 float 属性（非 nn.Parameter、非 buffer、不进 optimizer） |
| PR value centering（核心，per slide per head 对有效 PR token 求 μ_V） | ✅ | `μ_V=(Σ_{j∈valid} V_j)/|valid|`，`Ṽ=V−μ_V`，`O=A·Ṽ`；均匀 attention ⇒ O≈0（见 §2.2 Test 2，实测 rel=0.00e+00） |
| Cross 投影 bias=False（W_q/W_k/W_v/W_out） | ✅ | `m.w_q.bias is None` 等四个均为 None；参数数 6,319,531 = v1 的 6,321,579 − 2048（4×512 bias） |

**Stage2 参数量**：v2 = 1,055,744（v1 = 1,057,792，差 2048 = 4 个 bias）。HE 分支参数量 2,499,624 与 HE-only/v1 完全一致。

### 2.2 5 个实现测试 —— 5/5 通过

| Test | 内容 | 结果 |
|---|---|---|
| 1 | HE fallback / identity + bias-free | ✅ disable_cross / residual_scale=0 严格恒等；四个投影 bias=None；τ=0.2 |
| 2 | 均匀 attention 抵消（常数 slide） | ✅ `‖Δ‖/‖Z_HE‖ = 0.00e+00`（centering 精确抵消，无浮点残差） |
| 3 | PR 内容敏感性 | ✅ 换 PR → `‖Δa−Δb‖=13.9`（合成随机输入下 delta 随 PR 变） |
| 4 | mask / 空 PR | ✅ 全空 PR → 恒等；部分 mask → 有限、无效 HE slot 归零 |
| 5 | 梯度 | ✅ 有限且主路径全活；**w_q=19.1 / w_k=20.1 / w_v=14.7**（v1 为 0.003/0.003/1.7，cosine 显著激活了 Q/K 路径） |

### 2.3 Checkpoint 复现 —— 3/3 精确

| Seed | Saved AUC | Recomputed AUC | Diff |
|---|---|---|---|
| 42 | 0.841837 | 0.841837 | 0.00e+00 |
| 123 | 0.875510 | 0.875510 | 0.00e+00 |
| 456 | 0.813903 | 0.813903 | 0.00e+00 |

### 2.4 配置公平性

- effective LR=1e-4 实测作用于 6 个活跃模块（he/pr projection、he/pr rrt、stage2、abmil），统一 LR 分支。
- Stage1 encoder 配置与 HE-only / v1 完全一致（HE 4/9/3，PR 8/15/5），未改动。
- v2 与 v1 唯一差异 = stage2_type + stage2_cfg（temperature=0.2 / value_centering=True / qkv_bias=False）。
- 无新增 loss、无 contrastive/orthogonal/reconstruction/matched-mismatched、无辅助 PR 分类头、无熵正则、无 gate/FFN/双向 cross、无 modality dropout、无 residual-scale 搜索。

---

## 3. 因果检验 —— cross 完全失效 ❌（核心负面发现）

4 个 causal test 全部指向同一个结论：**换 PR、关闭 cross 都不改变预测，残差幅度 ≈0**。

### 3.1 matched vs disable_cross

| Seed | matched AUC | disable_cross AUC | Δ |
|---|---|---|---|
| 42 | 0.8418 | 0.8418 | **0.0000** |
| 123 | 0.8755 | 0.8755 | **0.0000** |
| 456 | 0.8139 | 0.8139 | **0.0000** |

**关闭 cross 与保留 cross 的 AUC 完全相同（到 4 位小数）。** v1 时 disable 还有 ±0.0003–0.005 的（内容无关）扰动，v2 因为残差塌缩为 0，disable == matched 严格相等。

### 3.2 matched vs random-mismatched（derangement，5 replacement seeds × 3 model seeds）

| Seed | matched | 5 次 random-mismatch AUC | 结论 |
|---|---|---|---|
| 42 | 0.8418 | 0.8421 / 0.8421 / 0.8418 / 0.8418 / 0.8421 | ≈ matched（±0.0003 非方向抖动） |
| 123 | 0.8755 | 0.8755 × 5 | **完全相等** |
| 456 | 0.8139 | 0.8139 × 5 | **完全相等** |

### 3.3 matched vs within-class-mismatched（同类内 derangement，5 replacement seeds）

| Seed | matched | 5 次 within-class AUC | 结论 |
|---|---|---|---|
| 42 | 0.8418 | 0.8421 / 0.8421 / 0.8418 / 0.8418 / 0.8418 | ≈ matched（±0.0003） |
| 123 | 0.8755 | 0.8755 × 5 | **完全相等** |
| 456 | 0.8139 | 0.8139 × 5 | **完全相等** |

**把 PR 换成任意其他 slide（随机 derangement）或同类 slide（within-class derangement）后，预测完全不变。** 15/15 次替换中 seed123/456 严格 0 变化，seed42 仅 ±0.0003 的残差噪声抖动（无一致方向，有时 mismatch 反而略高）。

### 因果结论

**matched 并不优于 mismatched，也不优于 disable_cross。** 三个 seed 一致：喂入哪个 PR slide 完全不影响预测，关闭 cross 也完全不影响预测。这与残差幅度诊断（§5）互证：**cross 残差塌缩为 ≈0**。

---

## 4. 四个诊断量（val-as-test，mean / p10 / p50 / p90）

| Seed | 诊断量 | mean | p10 | p50 | p90 |
|---|---|---|---|---|---|
| 42 | entropy_norm | 0.9954 | 0.9923 | 0.9958 | 0.9980 |
| | score_std | 0.164 | 0.1118 | 0.1641 | 0.2130 |
| | value_diversity | 0.976 | 0.9618 | 0.9775 | 0.9884 |
| | selective_ratio | 0.0183 | 0.0074 | 0.0158 | 0.0329 |
| | ρ = ‖0.1·Δ‖/‖Z_HE‖ | 1.77e-4 | 6.5e-5 | 1.58e-4 | 3.37e-4 |
| 123 | entropy_norm | 0.9997 | 0.9995 | 0.9998 | 0.9999 |
| | score_std | 0.0414 | 0.0252 | 0.0373 | 0.0586 |
| | value_diversity | 0.996 | 0.993 | 0.998 | 0.999 |
| | selective_ratio | 0.0020 | 0.0006 | 0.0011 | 0.0031 |
| | ρ | 2.46e-5 | 7.1e-6 | 1.39e-5 | 3.36e-5 |
| 456 | entropy_norm | 0.9996 | 0.9992 | 0.9997 | 0.9999 |
| | score_std | 0.0468 | 0.0258 | 0.0451 | 0.0678 |
| | value_diversity | 0.999 | 0.998 | 0.999 | 0.9997 |
| | selective_ratio | 0.0016 | 0.0003 | 0.0013 | 0.0029 |
| | ρ | 2.37e-7 | 5.4e-8 | 2.0e-7 | 4.4e-7 |

**关键读法**：

- **残差幅度 ρ 塌缩为 ≈0**（1.77e-4 / 2.46e-5 / 2.37e-7），对比 v1 的 ρ≈0.08–0.13，缩小了 **3–6 个数量级**。centering 把「均匀 attention ⇒ 0 残差」这一性质硬性执行了：只要 attention 接近均匀（entropy≈1），O=A·Ṽ≈0，bias-free W_out(0)=0，残差即 0。
- **attention entropy 仍然接近 1**（0.9954 / 0.9997 / 0.9996）：cosine 没有让 attention 变得有选择性，仅 seed42 略降到 0.995。
- **score_std 显著增大**（0.041–0.164 vs v1 的 ~0.001–0.004）：cosine/τ=0.2 放大了方向差异到 score 尺度，但**没有**转化为选择性 attention（softmax 仍近乎均匀），因为 PR routing token 的**方向本身近乎平行**（value_diversity 0.976–0.999）。
- **selective_ratio 极小**（0.018 / 0.002 / 0.0016）：输出里「选择性 PR 分量」占比几乎为 0，即 O≈A·μ_V 被 centering 抵消后所剩无几。

---

## 5. 根因分析（v2 为什么失败）

v2 的目标是**让残差真正依赖「匹配 PR 的内容」**，方法是：杀掉 v1 的「内容无关平均 PR 向量」路径（centering），逼模型走「选择性 attention」路径（cosine + bias-free）。

结果分两步看：

1. **centering 完全成功**：v1 的 ρ≈0.08–0.13（非零但内容无关的残差）被压到 ≈1e-4–1e-7。disable_cross 的 Δ 从 v1 的 ±0.0003–0.005 变成 v2 的严格 0。即「平均 PR 向量」这条已被证伪的路径被彻底堵死。

2. **选择性路径没有建立**：attention 仍然接近均匀（entropy 0.995–0.9997）。根因是 **PR routing token 本身塌缩为近常数**（value_diversity = mean pairwise cosine ≈ 0.976–0.999，即 token 之间方向几乎平行）。当所有 PR routing token 方向都近乎相同，cosine 相似度都趋近 1，`/τ=0.2` 的放大不足以在 softmax 里拉开差距 → attention 仍近乎均匀 → centering 把唯一（内容无关的）均值分量也减掉了 → 残差 = 0。

即：**v2 正确地堵死了错误路径，但暴露了更底层的问题——PR 独有信息本身不足以支撑「选择性 PR cross-attention」**。这与之前多轮诊断（[[c16-round6-pr-residual]]、[[c16-round7-he-residual-cross]]、[[c16-round3-causal-ablation]]）一脉相承：问题从来不在 Stage2 的残差方向/形式，而在 PR 独有特征判别力弱、routing token 塌缩。

训练目标（CE-only）没有给「让 attention 变 selective」任何压力——因为残差已经是 ≈0、HE-only 已经是一个足够好的解，模型没有动力去学习利用 PR 的细粒度信息。

---

## 6. 七个问题直接回答

1. **三改动是否实现正确并验证？** 是。§2.1 逐项核对通过；5/5 测试（含均匀抵消 rel=0.00e+00、bias-free、τ=0.2 固定）通过；3/3 checkpoint 浮点 0 误差复现。实现层**完全 SOLID**。

2. **(HE_i, PR_i) > HE_i 是否成立？** 名义上成立（mean +0.0070），但幅度极小、3 seed 中 1 个为负（−0.0004），且 **disable_cross == matched 严格相等** → 这个 +0.0070 **不是来自 PR cross**，而是不同随机初始化 HE encoder + test-selection 的差异。

3. **(HE_i, PR_i) > (HE_i, PR_j) 是否成立？** **不成立**。random-mismatch 与 within-class-mismatch 15/15 次与 matched 相等（seed123/456 严格 0，seed42 ±0.0003 非方向抖动）。matched 没有任何优势。

4. **disable_cross 是否降低 AUC？** **否**。Δ=0.0000（三个 seed 严格相等）。cross 对预测贡献为 0。

5. **attention entropy 是否从接近 uniform 变得更有选择性？** **几乎没有**。v1 entropy_norm ≈ 0.999999/1.0/0.999997 → v2 ≈ 0.9954/0.9997/0.9996。仅 seed42 从 1.0 微降到 0.995，seed123/456 仍 ≈1。score_std 虽大幅上升（cosine/τ 放大），但因 PR token 方向近乎平行，未能转化为选择性 attention。

6. **+0.007 的提升能否因果归因于匹配 PR？** **不能**。残差 ρ≈1e-4–1e-7（≈0），disable=matched=replacement 严格相等。+0.007 与 v1 的 +0.0254 同源：val-as-test 上选 checkpoint 的选择性抬高 + 不同随机初始化 HE encoder 差异，**与 PR 无关**。

7. **最终评级？** **FAILED**。v2 在结构/实现/协议/可复现层面是 SOLID 的，但核心科学目标（让残差依赖匹配 PR）未达成——cross 残差塌缩为 0，模型功能上退化为 HE-only，且没有走通「选择性 PR cross-attention」这条替代路径。

---

## 7. 与 v1 的对比总结

| 维度 | v1 he_residual_cross | v2 he_residual_cross |
|---|---|---|
| cross 残差幅度 ρ | 0.08–0.13（非零，内容无关） | **1e-4–1e-7（≈0）** |
| attention entropy_norm | ≈1.0（完全均匀） | 0.995–0.9997（仍近均匀） |
| disable_cross Δ | +0.0003 / −0.0008 / −0.0051 | **0.0000 × 3** |
| PR replacement | 15/15 不变 | **15/15 不变（更严格：等于）** |
| 相对 HE-only | +0.0254 | +0.0070 |
| 相对 v1 | — | −0.0184 |

**v2 没有变好，反而更惰性**：它把 v1 里那个「内容无关但仍非零」的残差也归零了，却没有补上一个「内容相关」的残差。这为下一步指明了真正方向：**问题不在 Stage2 融合形式，而在 Stage1 的 PR 特征/routing token 需要先获得可被选择性注意利用的判别性**——但在本任务约定的改动范围内（不改 loss、不加监督、不重提特征），v2 无法解决这一点。

---

## 8. 建议

**不建议以 v2 作为论文主模型。** 但 v2 的 negative result 是有价值的诊断结论：

- centering + cosine + bias-free 三件套**精确地证伪了「PR 平均向量」路径**（这是 v1 未被因果检验拆穿的残余假设）。
- 它把问题**收敛到 PR routing token 塌缩**（value_diversity≈0.98–0.999）这一可量化的根因上。
- 若后续允许，应先解决「PR routing token 判别性塌缩」（例如对 PR 分支的监督、或让 Stage1 PR 编码器学到更分散的 routing token），再回来做 Stage2 cross；否则任何 Stage2 融合形式都难以让残差真正依赖匹配 PR。

（本轮按约定未做任何超参搜索、未重提特征、未改动 Stage1、未新增 loss，仅如实报告。）
