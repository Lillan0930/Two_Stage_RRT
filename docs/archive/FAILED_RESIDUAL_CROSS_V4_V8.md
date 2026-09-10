# v4–v8 归档：HE 残差交叉分支的五轮失败尝试

**结论先行：v4、v5、v6、v7、v8 全部失败，没有一轮在主指标上超过同期 v3。**
`he_residual_cross` 这条线到 v3 为止，v4 之后都是在解释"为什么 v3 的 PR 残差≈0"，
而每一轮都只证明了：**换掉那个环节也不能让 PR 起作用**。

归档日期：2026-09-10。本文件是 v4–v8 的**唯一留存物**——对应代码、脚本、测试、
结果目录与 checkpoint 已在本轮清理中删除（见 §7）。

---

## 0. 共同前提

所有 v4–v8 都建立在 v3 之上，共享同一套协议与同一份数据：

| 项目 | 取值 |
|---|---|
| 数据 | C16，HE + PR(IHC)，共享 patch 索引 |
| Stage 1 | 每染色独立 `RRTEncoder`（HE: region 4 / epeg_k 9 / crmsa_k 3 / heads 4；PR: region 8 / epeg_k 15 / crmsa_k 5 / heads 8） |
| Stage 2 (v3) | 有向 HE→PR cross-attention，cosine 打分 τ=0.2，无 bias QKV，PR value 用 dataset prototype（EMA β=0.99）中心化，`residual_scale=0.1` 写到 HE 上 |
| MIL | ABMIL(hidden 256, dropout 0.25) |
| 训练 | batch 1，lr 1e-4 统一，cosine，80 epoch，early stopping patience 10 on val_auc，单 CE，全局梯度裁剪 1.0 |
| 协议 | Train / Val-as-Test（270 train / 129 val-as-test），`val_auc` 选模 |
| 初始化 | 全部 `random_from_scratch`，`pretrained_loaded=false` |

**v3 参照值**：`mean AUC = 0.8744 ± 0.0014`（seeds 42/123/456 → 0.8753 / 0.8755 / 0.8724）。
这是 v6/v7/v8 轮里**同期重跑的配对对照**（`results/stage2_he_residual_cross_v6/v3`），
不是历史 `results/stage2_he_residual_cross_v3`（历史目录未被动过）。

---

## 1. v4 — Q/K 共同方向移除

| | |
|---|---|
| **假设** | v3 的注意力熵≈1（均匀）是因为 Q/K 里存在 per-slide per-head 的共同方向，把 cosine 打分压平了；去掉它注意力就该变选择性 |
| **改动** | 只在 attention score 上做 per-slide per-head 的 Q/K 共同方向移除；V 的中心化沿用 v3；always-on，无新超参 |
| **机制验证** | **成功**。`entropy_norm` 0.99 → **0.826 ± 0.020**，`score_std` 1.65，`selective_ratio` 0.078，`r_slide` 0.077 |

**结果**

| seed | v4 AUC | v3 AUC | Δ |
|---|---|---|---|
| 42 | 0.8707 | 0.8753 | −0.0046 |
| 123 | 0.8653 | 0.8755 | −0.0102 |
| 456 | 0.8444 | 0.8724 | −0.0281 |
| **mean** | **0.8601 ± 0.0113** | 0.8744 | **−0.0143** |

**失败原因（机制反噬）**：注意力确实变选择性了，但**选择性是 slide-invariant 的**。
因果检验里 matched / `disable_cross` / random-mismatch / within-class-mismatch
四者在 129 张 slide 上的 AUC 与逐 slide margin **到小数点后四位完全相同**
（seed 42：全部 0.8707）。也就是说：把 PR 换成另一张 slide 的 PR，甚至整条 PR 支路关掉，
预测一个字都不变。注意力学会了"在 slide 内部挑 patch"，但挑的依据与 PR 内容无关。

> v4 的价值只在于证明了：v3 的均匀注意力**不是** Q/K 共同方向造成的。诊断线索作废，
> 但改进方向也作废。

---

## 2. v5 — PR value memory 辅助监督

| | |
|---|---|
| **假设** | v3 的 PR 残差≈0，是因为 merged PR value memory（`V_tilde`）里没有类别信号、或者信号学不到 |
| **改动** | 在 v3 的 `V_tilde` 上接一个轻量辅助 ABMIL，训练期叠加 `β·CE(pr_value_logits, y)`；主路径推理结构不变 |
| **条件** | `β=0`（对照）与 `β=0.1`，各 3 seeds |
| **协议** | `fixed_split_internal_val`（216 train / 54 val，另留 129 dev-test） |

**结果**（dev-test 129 张）

| 条件 | dev-test mean AUC | val mean AUC | 辅助头 val AUC (42/123/456) |
|---|---|---|---|
| β=0 | 0.7999 ± 0.0434 | 0.9754 | 0.457 / 0.622 / 0.565 |
| β=0.1 | 0.7632 ± 0.0118 | 0.9768 | **0.879 / 0.753 / 0.861** |

配对 Δ（β=0.1 − β=0）：**−0.0367**（42: +0.0004，123: −0.0776，456: −0.0329）——3 选中 2 个变差。

**失败原因（排除法价值最大的一轮）**：辅助头本身把 val AUC 做到 0.75–0.88，
说明 **PR value memory 里确实有类别信号，而且能学到**。可是主路径没有变好，反而变差。
⇒ 瓶颈**不在 PR 值本身**，而在消费它的那条 **cross-attention 通路**。
这一轮之后，"PR 表示塌缩 / PR 没信号"这一类解释被正式排除。

---

## 3. v6 — HE / PR+cross 训练梯度分工

| | |
|---|---|
| **假设** | PR 支路被 HE 的梯度淹没，所以学不出有用的残差；把梯度按来源分工就能救 |
| **改动** | 前向仍是 v3 结构。`L = CE(HE logits, y) + CE(fused logits, y)`，两项权重均 1。参数分两组：A(HE CE 更新)=`patch_to_emb[0]`/`rrt_he`/`mil`；B(fused CE 更新)=`patch_to_emb[1]`/`rrt_ihc`/`cross_region_mod`；两组各自 `clip_grad_norm_(1.0)`，无全模型联合裁剪 |

**结果**

| seed | v6 AUC | 同期 v3 AUC | Δ |
|---|---|---|---|
| 42 | 0.8500 | 0.8753 | −0.0253 |
| 123 | 0.8737 | 0.8755 | −0.0018 |
| 456 | 0.8219 | 0.8724 | −0.0505 |
| **mean** | **0.8486 ± 0.0212** | 0.8744 ± 0.0014 | **−0.0259** |

3/3 seeds 全部变差，方差放大 15 倍。

**失败原因**：MIL 只被 HE 的 CE 训练，却要在 fused 输入上做推理——**训练/推理错配**。
fused 表示相对 HE 有残差偏移，而 MIL 的决策边界是按 HE 拟合的，残差在这里只会变成噪声。

---

## 4. v7 — 恢复 fused 梯度（v6 的修正版）

| | |
|---|---|
| **假设** | v6 的锅是"MIL 只见过 HE"；那就让同一个 MIL 同时吃两项 CE，且不做参数 detach |
| **改动** | `L = CE(M(H), y) + CE(M(F), y)`，同一 MIL、不 detach，主预测仍是 `M(F)`。三个不重叠参数组各自 `clip_grad_norm_(1.0)`：HE=`patch_to_emb[0]`/`rrt_he`；MIL=`mil`；PR2=`patch_to_emb[1]`/`rrt_ihc`/`cross_region_mod` |
| **变体 A** | `F = Stage2(H.detach(), P)` — HE 编码器只拿 HE 的 CE |
| **变体 B** | `F = Stage2(H, P)` — HE 编码器两项 CE 都拿 |
| **验收标准** | A/B 必须超过**同期 v3**（不能只以超过较弱的 v6 判成功） |
| | |

**结果**

| 变体 | mean AUC | Δ vs v3 | per-seed Δ |
|---|---|---|---|
| A | 0.8392 ± 0.0186 | **−0.0352** | −0.0110 / −0.0426 / −0.0520 |
| B | 0.8270 ± 0.0157 | **−0.0474** | −0.0263 / −0.0587 / −0.0573 |
| B − A | −0.0122 ± 0.0049 | | |

`init_hash` A==B per seed：`true`（结构一致，差异只来自训练）。均 3/3 seeds 变差。

**失败原因（结论性）**：v7 修好了 v6 指出的错配，**结果更差**。这说明问题不在"哪些参数
被哪项 CE 更新"，而在 **dual loss 这个目标本身**——在只有 270 张训练 slide 的设定下，
多一个 loss 项带来的优化干扰大于它提供的信息。**梯度隔离这条线到此终止。**

---

## 5. v8 — HE region query 直接读 PR patch memory

| | |
|---|---|
| **假设** | v3 的 PR 信息被 routing 加权池化压缩成 region 级就丢掉了；绕开压缩、让 HE 的 region query 直接读全部 PR patch token 就能保住信息 |
| **改动** | `route_norm_pr → attn_norm_pr` 之间的 routing 加权池化替换为 identity；K/V = Z_PR 的全部有效 patch token（K = 有效数，物理上不再压缩）；`phi_pr` 保留但不参与 forward |
| **模式** | `routed`（≡v3）/ `patch`，各 3 seeds |

**结果**

| 模式 | mean AUC | Δ vs v3 |
|---|---|---|
| routed | 0.8744 ± 0.0014 | 0.0000 |
| patch | 0.8529 ± 0.0200 | **−0.0215**（42: −0.0194，123: −0.0485，456: +0.0033） |

`routed` 模式与 v3 **逐 bit 等价**：`init_hash` 相同、129 张 slide 的 prob/margin 完全一致 —
这同时是对照组有效性的证明。

**模型内配对诊断**（seed 42）：`residual_auc_gain = 0.00026`，
`paired_pr_auc_gain ≈ 1.1e-16`，`residual_margin_delta ≈ 0.0093`。

**失败原因**：拆掉 PR routing 压缩后，PR 支路**依然是近似 no-op**（配对 ΔAUC 到 1e-16 量级），
而且绝对性能下降。⇒ **PR routing 的信息压缩不是瓶颈**——瓶颈在更上游：
这条 directed cross 通路本身就没有可用的对齐信号。

---

## 6. 五轮汇总

| 版本 | 改了哪一环 | 主指标 Δ vs 相同对照 | 判定 |
|---|---|---|---|
| v4 | attention 打分（Q/K 共同方向） | −0.0143 (vs v3, 3/3↓) | 机制成功、端到端失败 |
| v5 | PR value 监督（辅助头 β=0.1） | −0.0367 (vs β=0, 2/3↓) | 辅助头 0.75–0.88 AUC，主路径不涨 |
| v6 | HE / PR+cross 梯度分工 | −0.0259 (vs v3, 3/3↓) | 训练/推理错配 |
| v7 | dual loss + 同 MIL（A/B） | A −0.0352 / B −0.0474 (3/3↓) | dual loss 本身有害 |
| v8 | PR memory 压缩（patch 模式） | −0.0215 (vs v3, 2/3↓) | 压缩不是瓶颈 |

**逐轮排除掉的假设**：Q/K 共同方向 → PR value 无信号 → 梯度被淹没 / MIL 训练错配 →
dual loss → PR routing 压缩。**每一环单独换掉都不能让 PR 产生有效残差。**

**共同结论**：C16 的 HE↔PR 在 patch 级没有可被 directed cross-attention 利用的对应关系
（与第 7 轮 audit 的 derangement no-op 结论一致）。在这条路上继续做结构改动没有依据，
**该线关闭**。

**保留的部分**：v3 本身是这条线的**最好结果**，也是唯一被保留下来、并成为正式主方法
`he_aux_unified` 的 cross branch 的结构（见 `HE_AUX_UNIFIED_REPORT.md`）。

---

## 7. 清理范围

**已删除**（本轮）：

```
models/he_residual_cross_crmsa_v4.py, _v5.py, _v8.py        （v6/v7 无独立模型文件）
scripts/run_stage2_v4.py … run_stage2_v8.py
scripts/eval_v4.py … eval_v8.py
scripts/smoke_v4_train.py … smoke_v8_train.py
tests/test_he_residual_cross_v4.py … _v8.py
results/stage2_he_residual_cross_v4 … v8/                    （158 个受版本控制的文件）
models/mm_rrt_abmil.py, train.py 中 v4–v8 的 factory / forward / 专用 loss / 裁剪逻辑
scripts/_run_protocol_seed.py 中 v6/v7/v8 专属的结果记录块
.swp
```

被删除的 **34 个 `.pt` checkpoint 合计 2.41 GB**（v4 222 MB/3 个、v5 591 MB/8 个、
v6 518 MB/7 个、v7 588 MB/8 个、v8 592 MB/8 个），全部在 `.gitignore` 中，
**只存在于该服务器上**，删除后不可恢复。保留此文件作为它们唯一的文字记录。

**保留**（正式继承链，勿删）：

```
models/he_residual_cross_crmsa.py      base
models/he_residual_cross_crmsa_v2.py   cosine + value centering
models/he_residual_cross_crmsa_v3.py   dataset-prototype centering  ← 正式方法
models/he_aux_unified.py               HE + 任意辅助染色 + 可插拔 MIL
models/mm_rrt_abmil.py                 Stage-2 总装
models/mm_rrt_encoder.py, models/rmsa.py
models/mil_heads.py, models/abmil.py, models/mil_registry.py
results/stage2_he_residual_cross_v3/   历史 v3 结果与 checkpoint（未动）
```

Git 历史未重写；`*.pt` 本就在 `.gitignore` 中，被删除的 checkpoint 从未进入版本库。
