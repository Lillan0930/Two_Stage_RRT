# HE Residual Cross v4 —— Q/K Common-Direction Removal 报告

**结论（一句话）：v4 的 Q/K slide 内 common-direction 去除在「机制层面」第一次成功了**——注意力从均匀（v3 entropy≈0.99）变成选择性（v4 entropy≈0.83）、attention logit 方差从 ≈0.1 涨到 ≈1.7、Q/K 的 pairwise cosine 从 0.82/0.99 骤降到 0.07/0.12；**但端到端因果仍是失败的**：`matched ≈ disable ≈ random_mismatch ≈ within_class_mismatch`（到小数点后 4 位），PR 残差在推理时贡献 ≈ 0，AUC 涨幅（v4 0.8601 vs HE-only 0.8367，+0.023）**全部来自联合训练变强的 HE encoder**。判定：**GLOBAL-PRIOR / TRAINING-DYNAMICS**（机制已修，但 PR 残差仍未真正依赖 matched PR content）。

核心问题的答案（§12）：**否**——去掉 Q/K 的 common direction 后，cross-attention 变得「选择性」了（非均匀），但**不是「内容选择性」**：它的非均匀权重不随 PR slide 改变（matched ≈ random_mismatch ≈ within_class 到 4 位小数），所以残差仍是 HE/slot 位置的固定函数，而非 matched PR content 的函数。

---

## 0. 摘要

### 主表（§10）

| Seed | HE-only | v3 | v4 Matched | Random mismatch | Within-class mismatch | Disable |
|---|---|---|---|---|---|---|
| 42 | 0.8344 | 0.8753 | **0.8707** | 0.8707 | 0.8707 | 0.8707 |
| 123 | 0.8615 | 0.8755 | **0.8653** | 0.8648–0.8653 | 0.8648–0.8651 | 0.8656 |
| 456 | 0.8143 | 0.8724 | **0.8444** | 0.8444 | 0.8444 | 0.8459 |
| mean | 0.8367 | 0.8744 | **0.8601 ± 0.0113** | 0.8600 | 0.8600 | 0.8607 |

### 核心 Δ（逐 seed paired，mean ± std）

| Δ | 定义 | seed42 | seed123 | seed456 | mean ± std |
|---|---|---|---|---|---|
| **Δ_HE** | matched − HE-only | +0.0362 | +0.0038 | +0.0301 | **+0.0234 ± 0.0140** |
| **Δ_cross** | matched − disable | +0.0000 | −0.0003 | −0.0015 | **−0.0006 ± 0.0007** |
| **Δ_random** | matched − random_mm | −0.0000 | +0.0003 | −0.0000 | **+0.0001 ± 0.0001** |
| **Δ_within** | matched − within_mm | −0.0000 | +0.0003 | −0.0000 | **+0.0001 ± 0.0001** |

- `Δ_HE > 0`（3/3 seeds，+0.0234）→ v4 相对 HE-only 有真实涨幅。
- `Δ_cross ≈ 0`（|Δ| ≤ 0.0015）→ 去掉 PR cross（纯 z_he 进 ABMIL）AUC 基本不变 ⇒ **PR 残差推理贡献 ≈ 0**。
- `Δ_random ≈ Δ_within ≈ 0`（到 4 位小数）→ 交换 PR slide（含同类别内交换）**零影响** ⇒ 残差不依赖 matched PR content。

---

## 1. 核心 8 问（§Final deliverable）

### Q1. Q/K centering 是否降低 pairwise cosine / 提高 score 方差？——**是（强）**

| 指标 | seed42 | seed123 | seed456 | mean | v3 对照 |
|---|---|---|---|---|---|
| q_raw_cos | 0.901 | 0.733 | 0.829 | **0.821** | — |
| q_centered_cos | 0.016 | 0.076 | 0.114 | **0.069** | — |
| k_raw_cos | 0.997 | 0.990 | 0.9995 | **0.996** | — |
| k_centered_cos | 0.228 | 0.153 | −0.008 | **0.124** | — |
| score_std | 1.645 | 2.203 | 1.326 | **1.725** | 0.170 / 0.105 / 0.029 |

- Q 的 raw pairwise cosine 0.73–0.90（高度共线）→ 居中后 0.02–0.11（近正交）。
- K 的 raw pairwise cosine ≈1（几乎完全共线）→ 居中后 −0.01–0.23。
- score_std 从 v3 的 ≈0.1 涨到 ≈1.7（**logit 终于有区分度**）。

### Q2. entropy 是否下降？——**是**

| | seed42 | seed123 | seed456 | mean |
|---|---|---|---|---|
| v3 entropy_norm | 0.992 | 0.998 | 0.9999 | 0.997 |
| **v4 entropy_norm** | **0.826** | **0.797** | **0.869** | **0.830** |

熵从 ≈1（最大熵，均匀注意力）降到 ≈0.83（中度选择性）。**这是 v4 相对 v2/v3 第一次真正让注意力离开「均匀」态。**

### Q3. 残差是否不再坍缩？——**部分（幅度回升，但端到端仍不贡献）**

| | seed42 | seed123 | seed456 | mean |
|---|---|---|---|---|
| v3 rho = ‖0.1·Δ‖/‖z_he‖ | 0.0207 | 0.0023 | 0.0002 | 0.008 |
| **v4 rho** | **0.0032** | **0.0167** | **0.0205** | **0.0135** |

- 残差幅度从 v3 的 ≈0.008 升到 ≈0.0135（seed123/456 升 7–100×），但仍只占 z_he 的 ~1%。
- 关键：`Δ_cross ≈ 0`（matched ≈ disable）→ 这个非零残差**没有改变最终预测**。所以「坍缩」在幅度上缓解了，在「对输出的贡献」上没有缓解。

### Q4. Matched > Disable？——**否**（Δ_cross = −0.0006 ± 0.0007，基本相等）

### Q5. Matched > Random mismatch？——**否**（Δ_random = +0.0001 ± 0.0001，到 4 位小数相等）

### Q6. Matched > Within-class mismatch？——**否**（Δ_within = +0.0001 ± 0.0001，到 4 位小数相等）

### Q7. v4 > HE-only？——**是**（+0.0234 ± 0.0140，3/3 seeds 全正）

### Q8. 最终评级 —— **GLOBAL-PRIOR / TRAINING-DYNAMICS**

- 满足「Matched > HE」且「Matched ≈ mismatch ≈ disable」→ 严格对应 §11 的 **GLOBAL-PRIOR/TRAINING-DYNAMICS** 档（AUC 涨幅来自联合训练，非 PR 残差）。
- 但**关键机制收获**：本轮第一次让 cross-attention 变得选择性（Q1–Q3 全成立），这是 v2/v3 都没做到的。问题从「注意力学不会选择性」变成了「注意力选择性但不内容相关」。

---

## 2. v4 实现（`models/he_residual_cross_crmsa_v4.py`）

v4 继承 v3（cosine τ=0.2 + bias-free QKV + dataset-prototype PR value centering `V'=V−μ_PR` + HE identity 残差 + 独立 routing + 全 slot 有向 HE→PR cross），**唯一核心改动**：

```
v3:  A = softmax( Q̂ K̂ᵀ / τ ),      Q̂=Q/‖Q‖, K̂=K/‖K‖
v4:  A = softmax( Q̂_c K̂_cᵀ / τ ),   Q̂_c=Q_c/‖Q_c‖, K̂_c=K_c/‖K_c‖
     Q_c = Q − Q̄,  K_c = K − K̄
     Q̄ = mean over valid HE routes(Q)   （per slide, per head）
     K̄ = mean over valid PR routes(K)   （per slide, per head）
```

- **只作用于 attention score**；V 的 dataset-prototype centering 完全不变。
- 数值安全：Q̄/K̄ 只在有效 token 上取均值；count.clamp(min=1)；ε=1e-6 安全归一化；无有效 PR token 时残差=0（identity 回退）；无效 token 不参与 mean。
- 新增 5 个观测诊断（无 loss）：`q_raw_cos / q_centered_cos / k_raw_cos / k_centered_cos / attn_weight_std`。
- 5 项实现测试全部通过（fallback、common-direction invariance、token-specific sensitivity、mask/empty-PR/NaN、gradient）。冒烟测试确认完整训练管线可构建（6,319,531 参数全被 optimizer 覆盖，μ_PR 首步 0→99.6 初始化）。

---

## 3. 训练结果（Train/Val-as-Test，3 seeds，LR=1e-4，wd=1e-5，batch=1，80 epochs，cosine，patience=10）

| Model | seed42 | seed123 | seed456 | mean ± std |
|---|---|---|---|---|
| HE-only（复用） | 0.8344 | 0.8615 | 0.8143 | 0.8367 ± 0.0193 |
| v1（复用） | 0.8406 | 0.8908 | 0.8551 | 0.8622 ± 0.0211 |
| v2（复用） | 0.8418 | 0.8755 | 0.8139 | 0.8437 ± 0.0252 |
| v3（复用） | 0.8753 | 0.8755 | 0.8724 | 0.8744 ± 0.0014 |
| **v4** | **0.8707** | **0.8653** | **0.8444** | **0.8601 ± 0.0113** |

paired ΔAUC（逐 seed）：

| Δ | seed42 | seed123 | seed456 | mean ± std |
|---|---|---|---|---|
| v4 − HE | +0.0362 | +0.0038 | +0.0301 | **+0.0234 ± 0.0140** |
| v4 − v1 | +0.0301 | −0.0255 | −0.0107 | −0.0020 ± 0.0235 |
| v4 − v2 | +0.0288 | −0.0102 | +0.0305 | +0.0164 ± 0.0188 |
| v4 − v3 | −0.0046 | −0.0102 | −0.0281 | **−0.0143 ± 0.0100** |

- v4 相对 HE-only 涨 +0.023（3/3 正），但**低于 v3**（−0.014），且基本等于 v1（−0.002）。
- 关键：v4 的 `disable_cross`（= 联合训练的 HE encoder 直接进 ABMIL）AUC = **0.8607**，≈ v4 matched 0.8601，远高于 HE-only 0.8367。⇒ **v4 相对 HE-only 的涨幅同样来自联合训练变强的 HE encoder，而非 PR 残差。**

---

## 4. 因果评估（matched / random-mismatch×5 / within-class×5 / disable）

| seed | matched | disable | Δ(matched−disable) | random_mm（5 seeds） | within_class_mm（5 seeds） |
|---|---|---|---|---|---|
| 42 | 0.8707 | 0.8707 | **+0.0000** | 0.8707（全部相同） | 0.8707（全部相同） |
| 123 | 0.8653 | 0.8656 | **−0.0003** | 0.8648–0.8653 | 0.8648–0.8651 |
| 456 | 0.8444 | 0.8459 | **−0.0015** | 0.8444（全部相同） | 0.8444（全部相同） |

- `matched ≈ disable`（|Δ| ≤ 0.0015）→ 去掉 PR cross AUC 基本不变。
- `matched ≈ random_mm ≈ within_mm`（到 4 位小数）→ 交换 PR slide（含同类别内）零影响。
- **重要新观察**：v4 的注意力已经非均匀（entropy≈0.83），但因果仍是 matched ≈ mismatch ⇒ 非均匀注意力是**slide 不变的固定模式**（如固定关注某些 slot 位置），而非由 PR content 驱动。

---

## 5. 诊断（§8，val-as-test 逐 slide；v3 对照）

| 指标 | seed42 | seed123 | seed456 | mean | v3 mean |
|---|---|---|---|---|---|
| **entropy_norm** | 0.826 | 0.797 | 0.869 | **0.830** | 0.997 |
| **score_std** | 1.645 | 2.203 | 1.326 | **1.725** | 0.101 |
| **rho** = ‖0.1·Δ‖/‖z_he‖ | 0.0032 | 0.0167 | 0.0205 | **0.0135** | 0.008 |
| **q_raw_cos → q_centered_cos** | 0.901 → 0.016 | 0.733 → 0.076 | 0.829 → 0.114 | **0.821 → 0.069** | — |
| **k_raw_cos → k_centered_cos** | 0.997 → 0.228 | 0.990 → 0.153 | 0.9995 → −0.008 | **0.996 → 0.124** | — |
| **attn_weight_std** | 0.0256 | 0.0267 | 0.0219 | **0.0247** | — |
| selective_ratio | 0.078 | 0.259 | 0.146 | 0.161 | 0.131 |
| r_slide | 0.077 | 0.257 | 0.145 | 0.160 | 0.130 |
| r_region | 0.080 | 0.262 | 0.146 | 0.163 | 0.137 |
| value_diversity | 0.9995 | 0.9974 | 0.9999 | 0.999 | 0.997 |
| μ_norm = ‖μ_PR‖ | 20.89 | 18.26 | 28.01 | 22.39 | 15.22 |

### 诊断解读（机制成功 + 端到端失败）

1. **Q/K centering 机制上完全按设计生效**：Q 共线度 0.82→0.07，K 共线度 0.996→0.124，注意力熵 0.997→0.830，score_std 0.101→1.725。**v3 的「均匀注意力」病根（Q/K 强 common direction）确实被去掉了。**
2. 但 `attn_weight_std` 仍很小（≈0.025，均匀权重下 12 个 key 权重 ≈0.083，std≈0）——注意力只是「略」离开均匀，且**模式与 PR slide 无关**（matched ≈ mismatch）。
3. `value_diversity ≈ 0.999` 仍然说明 PR value 高度共线（未变）——即使注意力选择性了，A·Ṽ 仍主要是 slide-global 分量（`selective_ratio ≈ r_slide ≈ 0.16`），被 w_out + 0.1 压到 rho≈1%。
4. 净效果：残差非零但 slide 不变 ⇒ matched ≈ disable ≈ mismatch。

**结论**：瓶颈从「注意力学不会选择性」下移到「注意力能选择性，但选择性不内容相关 + PR value 仍共线」。Q/K common direction 只是「均匀」的一个来源，去掉后暴露了更深一层：**注意力即使非均匀也不由 PR content 驱动**（可能因为 PR value 共线、或 w_out/0.1 压平、或没有信号梯度引导它学内容相关性）。

---

## 6. 成功判定（§11）

- 首要目标 `(HE_i, PR_i) > HE_i`（PR 残差**真实**贡献）：**FAILED**。`Δ_cross ≈ 0`、`Δ_random ≈ 0`，AUC 数字上的涨幅是联合训练 HE encoder 的产物。
- 次要目标 `(HE_i, PR_i) > (HE_i, PR_j)`（内容敏感）：**FAILED**。matched ≈ random_mismatch ≈ within_class_mismatch。
- **机制目标（注意力选择性）**：**SUCCESS**（entropy 0.83、score_std 1.7、Q/K cosine 骤降）——这是 v2/v3 都没做到的第一次。
- 综合评级：**GLOBAL-PRIOR / TRAINING-DYNAMICS**（Matched > HE 但 Matched ≈ mismatch ≈ disable；且注意力**未**再次坍缩）。

---

## 7. 事实与建议（§16 约束下）

**事实（不臆测）**：
- v4 的 Q/K centering 精确命中设计目标：Q/K 共线度骤降、注意力熵降、logit 方差升——注意力**第一次非均匀**。
- 但非均匀注意力是 **slide 不变的固定模式**（matched ≈ mismatch 到 4 位小数），PR 残差端到端贡献仍 ≈0。
- v4 AUC（0.8601）低于 v3（0.8744）——去掉 common direction 让注意力「用掉」了更多自由度去形成固定非均匀模式，但没换来内容相关性，反而略损了联合训练收益（+0.023 vs v3 的 +0.038）。
- PR value 共线（value_diversity≈0.999）自 v2 以来未变，仍是独立于 attention 的另一处坍缩。

**按 §12 约束，本轮不自动修改** V centering、routing、patch_to_emb、RRT；不加 entropy/contrastive/gate/auxiliary loss；不调 temperature / residual scale。**对核心问题的诚实回答：Q/K common-direction 去除让注意力「选择性」但未让注意力「依赖 matched PR content」。** 后续若要继续，优先级应转为：诊断「注意力为何 slide 不变」（例如冻结权重下直接看 attention 模式是否随 PR slide 变化、或 PR value 共线是否是 A·Ṽ 坍缩的主因），而非继续改注意力归一化/居中方案。

---

## 8. 复现入口

- 训练驱动：`scripts/run_stage2_v4.py --gpus <g1> <g2> <g3>`
- 评估：`scripts/eval_v4.py --section repro causal diagnostics summary --gpu <g>`
- 实现测试：`python tests/test_he_residual_cross_v4.py`（5/5）
- 冒烟测试：`python scripts/smoke_v4_train.py`
- 结果目录：`results/stage2_he_residual_cross_v4/`（`summary.json`、`_eval/{repro,causal,diagnostics,summary}.json`）
