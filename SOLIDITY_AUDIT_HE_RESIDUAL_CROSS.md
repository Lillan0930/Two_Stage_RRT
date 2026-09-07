# Solidity Audit — `he_residual_cross`（定稿前审计）

**日期**: 2026-09-07
**对象**: `models/he_residual_cross_crmsa.py`（`HEResidualCrossCRMSA`）+ Train/Val-as-Test 协议结果
**审计产物目录**: `results/stage2_he_residual_cross_val_as_test/_audit/`
**审计脚本**: `scripts/solidity_audit.py`（本轮只读审计，未修改模型/协议/数据/特征）

---

## 0. 主结果（被审计对象）

**Evaluation protocol: Train / Val-as-Test**（val-as-test = C16 official test 129；用于每 epoch 验证、早停、best checkpoint 选择、最终报告）

| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |
|---|---|---|---|---|
| HE-only | 0.8344 | 0.8615 | 0.8143 | 0.8367 ± 0.0193 |
| Joint CR-MSA | 0.8505 | 0.8360 | 0.8449 | 0.8438 ± 0.0060 |
| HE Residual Cross | 0.8406 | 0.8908 | 0.8551 | **0.8622 ± 0.0211** |

ResidualCross − HE = **+0.0254**（+0.006 / +0.029 / +0.041，3/3 为正）。

---

## 1. 代码实现审计 —— 实现与设计一致 ✅

调用链逐项核对（`models/mm_rrt_abmil.py` forward `two_stage_region` 分支 + `models/he_residual_cross_crmsa.py`）：

| 要求 | 结论 | 证据 |
|---|---|---|
| HE/PR Stage1 独立 RRT，参数不共享 | ✅ | `rrt_he`/`rrt_ihc` 两个独立 `RRTEncoder` 实例；`model.rrt_he is model.rrt_ihc == False` |
| 两支从头训练、不加载旧 ckpt | ✅ | `initialization=random_from_scratch`，`pretrained_loaded=false`；`pretrained_he_ckpt=None`，correction-only 加载分支未触发（`use_correction_only=False`） |
| 两支未冻结 | ✅ | 6,321,579 参数全部 `requires_grad=True`（trainable == total） |
| Stage2 确实调用 `HEResidualCrossCRMSA` | ✅ | `stage2_type='he_residual_cross'` → `cross_region_mod = HEResidualCrossCRMSA(...)` |
| Q 只来自 HE | ✅ | `q = w_q(attn_norm_he(R_HE))` |
| K/V 只来自 PR | ✅ | `k = w_k(attn_norm_pr(R_PR))`, `v = w_v(attn_norm_pr(R_PR))` |
| 所有 HE slot 可访问所有有效 PR slot，无 same-slot 限制 | ✅ | attention `[B,h,Q,K]` 全连接，仅 `k_valid` mask，无 slot 对齐约束 |
| `phi_he`/`phi_pr` 独立 | ✅ | 均为独立 `nn.Parameter`；`m.phi_he is m.phi_pr == False` |
| Stage2 输出仅 `[B,N_HE,D]` | ✅ | `out = z_he + drop_path(0.1·Δ_HE)` |
| 无 PR token concat | ✅ | 输出无 concat，仅 HE patch 序列 |
| identity residual 基底 = 原始 `Z_HE` | ✅ | `out = z_he + …`，`z_he` 为 HE RRT 原始输出 |
| 无额外 out_norm / FFN / self-attn | ✅ | `no_out_norm=True`, `no_ffn=True`；模块无 post-residual 变换 |
| `residual_scale=0.1` 固定不参与优化 | ✅ | `register_buffer`，`requires_grad=False`，不在 `named_parameters()` |
| 无意外 detach | ✅ | 模块源码 `.detach(` 出现 0 次 |
| PR→routing→K/V→cross→Δ→ABMIL 梯度完整 | ✅ | 见 §6，无 `grad=None` 主路径参数 |

**参数统计**：

| 模型 | 总参数 | Stage2 参数 | HE 分支(proj+rrt_he) |
|---|---|---|---|
| HE-only | 2,763,051 | — | 2,499,624 |
| staining_msa | 6,317,995 | 1,054,208 | 2,499,624 |
| he_residual_cross | 6,321,579 | 1,057,792 | 2,499,624 |

HE 分支参数量在三模型中**完全一致**（2,499,624）。staining_msa 与 he_residual_cross 的 Stage2 差异 3,584 = 独立 `phi`/`route_norm`/`attn_norm` 的预期结构差异。

---

## 2. 配置公平性审计 —— 公平 ✅

对 9 个实际保存的 `config.json` 做逐字段 diff（`_audit/config_diff.json`）：

- **三模型之间**唯一差异均为**预期结构差异**：
  - `data.modalities` / `data.dir_mapping`：`["HE"]` vs `["HE","PR"]`
  - `model.stage2_type`：缺省 / `staining_msa` / `he_residual_cross`
  - `model.encoder_cfg` / `model.stage2_cfg`：仅双模态存在
  - `stage2_cfg` 在 he_residual_cross 多出 `residual_scale=0.1, disable_cross=False`
  - 其余 `training` / `model`(共享部分) / `data`(其余字段) / `data_split` **逐字节一致**
- **同条件跨 seed**：剥离 seed 相关字段后**零差异**（仅 seed 本身不同）。
- **effective LR = 1e-4** 确实作用于 HE RRT / PR RRT / Stage2 / ABMIL：`result.json` 的 `actual_optimizer_lrs` 六个活跃模块全部 `0.0001`（由 `create_optimizer_scheduler` 重建 optimizer 实读，非仅 YAML；统一 LR 分支）。
- **HE encoder 结构公平**：HE-only 的 `RRTEncoder` 与双模态 HE 分支结构参数一致（region_num=4 / n_heads=4 / epeg_k=9 / crmsa_k=3 / crmsa_heads=8 / drop_path=0.0）。

**结论**：无 `UNINTENDED DIFFERENCE`。

---

## 3. Checkpoint 与结果可复现性 —— 完全可复现 ✅

对全部 9 个 best checkpoint 重新纯 inference（重建 model + 重建 val dataset，sampling 确定性由 MD5 hash + `RandomState` 保证）：

| Model | Seed | Saved AUC | Recomputed AUC | Diff | PASS |
|---|---|---|---|---|---|
| he_only | 42 | 0.834439 | 0.834439 | 0.00e+00 | ✅ |
| he_only | 123 | 0.861480 | 0.861480 | 0.00e+00 | ✅ |
| he_only | 456 | 0.814286 | 0.814286 | 0.00e+00 | ✅ |
| staining_msa | 42 | 0.850510 | 0.850510 | 0.00e+00 | ✅ |
| staining_msa | 123 | 0.835969 | 0.835969 | 0.00e+00 | ✅ |
| staining_msa | 456 | 0.844898 | 0.844898 | 0.00e+00 | ✅ |
| he_residual_cross | 42 | 0.840561 | 0.840561 | 0.00e+00 | ✅ |
| he_residual_cross | 123 | 0.890816 | 0.890816 | 0.00e+00 | ✅ |
| he_residual_cross | 456 | 0.855102 | 0.855102 | 0.00e+00 | ✅ |

9/9 逐 seed 完全一致（浮点 0 误差）。`ckpt epoch == result.json best_epoch`（best checkpoint 而非 last；e.g. he_only seed42 ckpt_epoch=7=best_epoch）。无目录串用、无 checkpoint 路径串用、model seed 明确区分。

---

## 4. 因果验收 —— 模型预测不依赖匹配 PR ❌（核心负面发现）

只针对新训练的 3 个 he_residual_cross checkpoint。

### A. disable_cross（`Z_HE' = Z_HE` 严格绕过 Stage2）

| Seed | Full AUC | disable_cross AUC | Δ |
|---|---|---|---|
| 42 | 0.8406 | 0.8403 | +0.0003 |
| 123 | 0.8908 | **0.8916** | **−0.0008** |
| 456 | 0.8551 | **0.8602** | **−0.0051** |

**关闭 cross 不降低 AUC；3 个 seed 中 2 个反而提升。**

### B. residual_scale = 0 与 disable_cross 数值等价 ✅

`max_abs_logit_diff(disable_cross, residual_scale=0) = 0.00`（三者 seed 均如此）。数学等价性确认。

### C. Cross-slide PR replacement（derangement，π(i)≠i，5 个 replacement seed × 3 model seed）

| Model seed | Full | deranged AUC（5 个 replacement seed） | 结论 |
|---|---|---|---|
| 42 | 0.8406 | 0.8403 / 0.8406 / 0.8406 / 0.8406 / 0.8403 | 全部 ≈ Full（|Δ|≤0.0003） |
| 123 | 0.8908 | 0.8911 / 0.8908 / 0.8908 / 0.8911 / 0.8908 | 全部 ≈ Full（|Δ|≤0.0003） |
| 456 | 0.8551 | 0.8554 / 0.8551 / 0.8556 / 0.8554 / 0.8548 | 全部 ≈ Full（|Δ|≤0.0005） |

**把 PR 换成任意其他 slide（derangement，含跨 tumor/normal 交换）后，预测完全不变（15/15 次 Δ ≤ 0.0005）。**

> patch-count 一致的严格子集 AUC 有较大波动（~0.83–0.94），但这是**子集选择偏差**（每个 replacement seed 选出的子集不同、样本数 ~101–105/129），不是可比的诊断信号；主结论应以全 129 样本的 `auc_all` 为准。

### 因果结论

**Full model 并不稳定优于 mismatched-PR，也几乎不优于 disable_cross。** 三个 seed 一致地显示：当前模型的预测与「喂入哪个 PR slide」无关，且关闭 cross 不带来损失（2/3 反而更优）。**+0.0254 的提升无法归因于「匹配 PR 的 cross-attention」。**

---

## 5. 更新量检查 —— Δ 非退化、但 PR 内容无关

$$ \rho = \frac{\|0.1\cdot\Delta_{patch}\|_2}{\|Z_{HE}\|_2+\epsilon} $$

| Model seed | ρ mean | ρ std | ρ p10 | ρ p50 | ρ p90 | ρ max |
|---|---|---|---|---|---|---|
| 42 | 0.0836 | 0.0160 | 0.0701 | 0.0800 | 0.1026 | 0.1348 |
| 123 | 0.0797 | 0.0130 | 0.0686 | 0.0782 | 0.0948 | 0.1165 |
| 456 | 0.1347 | 0.0254 | 0.1069 | 0.1339 | 0.1705 | 0.1834 |

`‖Δ_patch‖` ~850–1450，`‖Z_HE‖` ~1070（所有 seed 中位 ~1120）；有效 HE routes ≈ 有效 PR routes ≈ 48（Stage2 内 region_num=4、crmsa_k=3，两支各 16×3 个 routing token）。

**Δ 是真实、非退化的小幅扰动（约 8–13% 相对幅度，非 0）**——但它与 §4 一致：这是一个**不随 PR 内容变化、且对分类无益（甚至轻微有害）**的扰动。即 cross 机制产生了可观的输出，但该输出没有携带「匹配 PR」的判别信息。

---

## 6. 梯度验收 —— 主路径梯度完整 ✅

用真实权重 + 合成有效输入（B=1, N=196, D=768）做 backward，逐模块 L2 梯度范数：

| 模块 | grad L2 | 参数数 |
|---|---|---|
| HE RRT (`rrt_he`) | 14.77 | 17 |
| PR RRT (`rrt_ihc`) | 0.955 | 17 |
| `phi_he` | 0.665 | 1 |
| `phi_pr` | 0.131 | 1 |
| `W_q_he` | 0.00276 | 2 |
| `W_k_pr` | 0.00271 | 2 |
| `W_v_pr` | 1.655 | 2 |
| `W_out` | 2.108 | 2 |
| ABMIL (`mil`) | 15.91 | 8 |

- 全部有限（无 NaN）；PR 主链（PR RRT / phi_pr / W_k_pr / W_v_pr / W_out）**整体非零**；Q/K/V/W_out 均有梯度；`grad=None` 的可训练参数为 **[]**。
- 观察：`W_q_he`（0.0028）/`W_k_pr`（0.0027）梯度极小，而 `W_v_pr`（1.65）/`W_out`（2.11）较大——与 §4/§5 一致：attention 权重（Q·K）几乎不被更新，交叉注意力退化为「内容无关」的 value 聚合。非失败，但属于与因果结论互证的旁证。

---

## 7. Paired prediction 稳健性 —— bootstrap CI 均包含 0（除 seed123 vs Joint）❌

基于已保存的逐 WSI predictions，paired bootstrap（5000 次，两模型共用同一批 bootstrap indices）：

| Comparison | Model seed | ΔAUC | 95% CI |
|---|---|---|---|
| ResidualCross − HE | 42 | +0.0062 | [−0.048, +0.059] |
| ResidualCross − HE | 123 | +0.0298 | [−0.033, +0.099] |
| ResidualCross − HE | 456 | +0.0408 | [−0.008, +0.095] |
| ResidualCross − Joint | 42 | −0.0093 | [−0.068, +0.050] |
| ResidualCross − Joint | 123 | +0.0553 | [+0.007, +0.107] |
| ResidualCross − Joint | 456 | +0.0103 | [−0.047, +0.069] |

**ResidualCross − HE 的 3 个 seed 95% CI 全部包含 0**；ResidualCross − Joint 仅 seed123 的 CI 不包含 0。即在当前 val-as-test 开发集上，+0.0254 的逐 seed 提升**不稳健**，不能包装为显著。

> 注意：official test 被用于 checkpoint selection，bootstrap CI 只描述当前 val-as-test 开发集上的稳定性，不作独立外部显著性证据。

---

## 8. seed123 是否单独拉高均值 —— 否 ✅

- **逐 seed 的 ΔAUC(ResidualCross−HE)**：+0.006(42) / +0.029(123) / **+0.041(456)**。拉高均值最多的是 **seed456**，不是 seed123。
- seed123 的**绝对** AUC 高（0.8908）是因为它的 **HE-only 基线也最高**（0.8615）——seed123 对三个模型都是「好 seed」，并非 RCX 特有。
- seed123 极端逐 WSI 变化**最少**（`|Δp|>0.2` 仅 10 例，seed42/456 各 22 例），且方向混合（上下翻转均有）。
- best_epoch=12 正常（42=9、456=6），训练日志正常（270/129、6.32M 参数、per-epoch sampling、persistent_workers=False 均符合协议），无 sampling/checkpoint 异常。

**seed123 不构成异常驱动。**

---

## 9. 最终评级

### 结论：**NOT SOLID**（作为「用匹配 PR 增强 HE 的论文主模型」而言）

**区分两类事实**：

**结构/协议/可复现层面全部 SOLID**：实现与数学设计逐项一致（§1）；三模型训练协议公平、无隐藏配置污染、effective LR=1e-4 实测作用于所有模块（§2）；9/9 checkpoint 逐 seed 精确复现（§3）；PR 主路径梯度完整（§6）。**没有代码 bug、没有数据泄漏、没有 checkpoint 串用。**

**科学主张层面 NOT SOLID**（命中 rubric C 的实质判据）：
1. **PR replacement 完全无影响**：15/15 次 derangement（含跨 tumor/normal 交换）预测不变（Δ≤0.0005）。
2. **关闭 cross 不降低 AUC**：disable_cross 在 3 seed 中 2 seed **反而提升**（+0.0003 / −0.0008 / −0.0051）。
3. **ResidualCross − HE 的 3 个 bootstrap 95% CI 全部包含 0**（§7）。

即：**+0.0254 的提升不是由「匹配 PR 的 cross-attention」产生的**。它来自：val-as-test 在 test 上选 checkpoint 的选择性抬高 + 不同随机初始化/不同训练动力学下的 HE encoder + ABMIL 差异，而非 PR 残差写回机制本身。这与之前 fixed-split 协议下的结论（rcx ≈ HE-only）方向一致——机制本身不贡献增量。

### 是否建议冻结结构进入正式消融？

**不建议以当前证据冻结为「PR-cross 主模型」。** 在把该结构作为主模型之前，必须先解决「cross 更新与 PR 内容解耦」这一实质问题（例如 attention 权重是否退化为均匀、PR routing token 是否塌缩为 slide 无关的常数、是否需要针对 cross 分支的监督/约束）。本轮按约定**不做任何模型修改、不调参、不重提特征**，仅报告事实。

---

## 10. 五个问题直接回答

1. **代码实现是否与数学设计一致？** 是。§1 逐项核对全部通过，无 detach、无共享 phi、Q/K/V 来源正确、输出仅 HE、residual_scale=0.1 固定 buffer、梯度完整。
2. **0.8622 vs 0.8367 是否公平、可复现？** 公平（§2 无 unintended diff，LR/HE 结构/数据全对齐）且完全可复现（§3，9/9 浮点 0 误差）。但注意该比较是「不同训练模型的比较」，且 test 被用于选 checkpoint（协议本身会抬高 AUC）。
3. **是否有直接因果证据表明预测确实使用了匹配 PR？** **没有**。证据相反：disable_cross 不降 AUC（2/3 反升），derangement 15/15 次预测不变。预测对「喂入哪个 PR」不敏感。
4. **+0.0254 是否被某个异常 seed 或配置问题主导？** 否。逐 seed Δ 为 +0.006/+0.029/+0.041（seed456 最大，非 seed123）；无配置污染、无 checkpoint 串用、无极端 WSI 主导（seed123 极端变化反而最少）。
5. **最终评级与建议？** **NOT SOLID（科学主张层面）**。实现/协议/可复现是 SOLID 的，但「PR cross-attention 带来提升」这一核心主张被因果检验证伪；不建议以当前证据冻结结构进入正式消融，需先解决 cross 更新与 PR 内容解耦的问题。
