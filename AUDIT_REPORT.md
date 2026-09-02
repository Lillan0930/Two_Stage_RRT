# Two-Stage R²T (HE+PR) 代码审计报告

审计范围：`models/`、`utils/metrics.py`、`train.py`、`data/`、`feature_extract/`
方法：静态代码走查 + 一次性 forward/backward 实测（`scripts/diag_audit.py`）+ 修复后冒烟（`scripts/smoke_test_audit.py`）。
结论优先级按「代码正确性 + 现有结构实际执行逻辑」排序，**不涉及模型重新设计**。

---

## 1. 总结

代码的**主链路（数据加载 → 双 RRT → CR-MSA → ABMIL → 2-logit CE → AUC）是正确且自洽的**：
两个 RRT encoder 都收到梯度，CR-MSA 确实执行了 HE↔PR 交互，损失/AUC 计算无误。现有 Two-stage
追平 HE 而不能稳定超越，**不是由某个会压制性能的隐藏 Bug 造成**，而是

1. 一个「死模块」`self.rrt_encoder`（MM_RRTEncoder，4.47M 参数，占 41%）被无条件创建但从不参与 forward —— 纯浪费（内存 + RNG 状态），不影响 AUC；
2. 权重初始化被重复执行多次（`initialize_weights` 内部再 `module.modules()` + 外部 `.apply()` 双重遍历）—— 冗余，不影响 AUC；
3. 二分类下 `sensitivity_macro == specificity_macro`（都等于 balanced accuracy）—— 报告口径问题，不影响 AUC；
4. **结构层面**：Stage2 `CrossStainingCRMSA` 是「HE/PR routing-token 的联合全注意力（joint self-attention）」，四块 A_HH/A_HP/A_PH/A_PP 全在场、无 mask，且 LayerNorm/phi/qkv 三处**跨模态共享**。这决定了 PR 只能以「与 HE 同语义空间、且与 HE 高度冗余的额外 token」的身份参与融合 —— 这正是「追平但难超越」的结构性原因（详见 §4 与 Q3/Q4）。

已按「最小改动」原则修复 3 处（§7），并通过 6 项冒烟测试（§6）。

---

## 2. Confirmed Bugs（A 类，已修复）

### A1. 权重初始化重复执行（双重遍历）

- **文件/函数**：`models/mm_rrt_encoder.py:40` `initialize_weights`；`models/abmil.py:151` 同名函数。
- **原代码**：
  ```python
  def initialize_weights(module):
      for m in module.modules():      # ← 内部递归遍历
          if isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight) ...
  ```
- **调用点**：`MM_RRTEncoder.__init__`（`self.apply(initialize_weights)`）、`RRTEncoder.__init__`（×2，need_init=True）、`ABMIL.__init__`（×2）、`mm_rrt_abmil.py:443`（`self.apply(initialize_weights)`）。`.apply()` **本身就会递归到每个子模块**，再叠加函数内部的 `module.modules()` 递归 → 越深层的 Linear/LayerNorm 被重复初始化越多次。
- **实测证据**（修复前 `diag_audit.py`）：`initialize_weights` 被调用 **109 + 11 = 120 次**，累计遍历 **482 + 29 = 511 个模块**。
- **修改**：函数改为**只初始化传入的单个 module**（去掉内部 `module.modules()` 循环）。`.apply()` 仍负责递归，于是每个参数**恰好初始化一次**。
- **为什么是 Bug**：这是「冗余 + RNG 状态被无谓消耗」的正确性瑕疵 —— 最后一次写覆盖前几次写，所以最终权重仍是 xavier 分布。**影响 AUC：≈0**（同分布重采样，无偏）。但会污染「给定 seed 的可复现 init」，且是死模块 4.47M 参数也被反复初始化的根因之一。

### A2. 死模块 `self.rrt_encoder`（MM_RRTEncoder）无条件创建

- **文件/函数**：`models/mm_rrt_abmil.py:160-184` `MM_RRT_ABMIL.__init__`。
- **原代码**：无论 `fusion_type` 是什么，都 `self.rrt_encoder = MM_RRTEncoder(need_init=True, ...)`。
- **问题**：在 `fusion_type == 'two_stage_region'`（以及 `two_stage_direct`）时，forward 走的是 `self.rrt_he` / `self.rrt_ihc` / `self.cross_region_mod` / `self.mil`，`self.rrt_encoder` **从不被调用**。
- **实测证据**：该死模块含 **38 个参数、4,472,659 个元素 = 总参数 41.4%**；backward 后这 38 个参数 `grad is None`（正是既有 verify 报告里的 `unused_grad_none_count: 38`）。
- **修改**：`fusion_type in ('two_stage_region', 'two_stage_direct')` 时 `self.rrt_encoder = None`，其余 fusion 类型照常创建；`get_model_info()` 增加 `rrt_encoder is not None` 守卫。
- **为什么是 Bug**：创建了从未使用的 4.47M 参数 —— 浪费显存、增加 init 时的 RNG 消耗、污染可复现性。**影响 AUC：≈0**（死参数从不参与 forward/backward），但属于必须清理的死代码。
- **修复效果**：总参数 10,790,654 → **6,317,995**；`grad=None` 参数 38 → **0**。其余 fusion 类型（`self_attention`/`instance_bag_expansion`）回归验证：`rrt_encoder` 仍正常创建。

### A3. 二分类下 Sensitivity/Specificity 恒等

- **文件/函数**：`utils/metrics.py:57-63` `calculate_metrics`。
- **原代码**：
  ```python
  metrics['sensitivity_macro'] = mean([sensitivity_class_0, sensitivity_class_1])
  metrics['specificity_macro'] = mean([specificity_class_0, specificity_class_1])
  ```
- **问题**：二分类时 `sensitivity_class_1 == specificity_class_0`、`specificity_class_1 == sensitivity_class_0`（互补），于是 `sensitivity_macro == specificity_macro == (sens_tumor+spec_tumor)/2 == balanced accuracy`。临床意义的「灵敏度/特异度（tumor 为阳性）」被这个 macro 掩盖。
- **修改**：`num_classes == 2` 时新增 `metrics['sensitivity'] = sensitivity_class_1`、`metrics['specificity'] = specificity_class_1`（tumor=1 为阳性）。
- **为什么是 Bug**：报告口径错误（headline 指标语义错误）。**影响 AUC：0**（AUC 仍走 `roc_auc_score(y_true, y_prob[:,1])`，正确）。per-class 值本就正确，`_run_protocol_seed.py` 里 `sensitivity_tumor` 也已取对。

---

## 3. Implementation Risks（B 类，未改 / 仅记录）

### B1. 特征提取 `rglob('*')` 未排序 → 潜在的 patch 行序不一致

- **文件**：`feature_extract/c16_feature_ctrans.py:93-96`
  ```python
  patch_files = [str(p) for p in Path(wsi_dir).rglob('*')
                 if p.suffix.lower() in {...} and p.stat().st_size > 0]
  ```
  `rglob` 遍历顺序由文件系统（inode/hash）决定，**不保证字典序**，且 HE（`C16_raw`）与 PR（`Results-PR`）是两个不同的目录树，行序可能不同。
- **下游契约**：`data/c16_multimodal_dataset.py:216-231` 要求 HE/PR **patch 数量严格相等**（否则 `ValueError`），且 `__getitem__` 用**同一组 `patch_indices`** 同时截断 HE 与 PR —— 代码注释明确假设「HE/PR patch 顺序完全相同」。
- **为什么只列为 Risk 而非 Bug**：CR-MSA 的跨模态交互发生在 **routing-token 层（region 摘要）**，对模态内 patch 顺序是置换不变的，因此 HE↔PR 的 patch 级行序错位**不会**让跨模态注意力「配对错组织」。且 verify 已测得全部 270+129 slide 的 `patch_count_mismatch=0`，说明 PR 是 1:1 由 HE 虚拟染色生成（数量与集合一致）。
- **残余隐患**：RRT 的 `region_partition` 把 1D token 序列 reshape 成 `sqrt(N)×sqrt(N)` 的 2D 网格，隐含「行序≈空间相邻」假设。未排序的 `rglob` 会让「region」变成任意 patch 分组 —— 但这同时作用于 HE-only baseline，是全局性限制，不解释「two-stage 追平而不超越」。
- **建议（低优先级）**：提取时 `sorted(...)`，或在 `.pt` 里同时保存 patch 文件名/坐标以做行序校验。

### B2. 随机种子/可复现性不完整

- **文件**：`train.py:63-70` `set_seed`。
- 只设了 `torch.manual_seed` / `torch.cuda.manual_seed_all` / `np.random.seed`，**没有 `random.seed()`**，且 `cudnn.benchmark=True`（非 deterministic）。注释「训练结果仍通过 manual_seed 保证可复现」**不准确**。
- **为什么列为 Risk**：数据采样层用 `hashlib.md5` 稳定哈希（`c16_multimodal_dataset.py:26-35`），**不受** Python hash/进程边界影响，因此 data 采样是可复现的；模型 init 用 `torch.manual_seed` 也可复现。残余非确定性来自 cuDNN benchmark 的 kernel 选择（backward 微小数值差异）与任何 `random` 模块调用。**影响 AUC：训练间抖动级别，非系统性偏差。**

### B3. `train.py:1417-1418` 仍用 `sensitivity_macro` 记录 best-val 敏感度/特异度

- `self.best_val_sensitivity = val_metrics.get('sensitivity_macro', 0.0)`（二分类下=balanced accuracy）。早停监控是 `val_auc`，故**不影响选模型**，仅日志口径。已在 `metrics.py` 提供 `sensitivity`/`specificity` 供后续改用。**影响 AUC：0。**

---

## 4. Structural Limitations（C 类，不改，仅说明）

### C1. Stage2 是「联合自注意力」，不是「有向跨模态注意力」

`CrossStainingCRMSA`（`models/cross_staining_crmsa.py:207-266`）实际执行：

```
z_he_n = norm(z_he);  z_pr_n = norm(z_pr)              # 共享 LayerNorm
routing_he = combine(z_he_n)   →  [K, nW_he, C]        # 共享 phi
routing_pr = combine(z_pr_n)   →  [K, nW_pr, C]
routing_joint = cat([routing_he, routing_pr], dim=1)   # [K, nW_he+nW_pr, C]
routing_joint = InnerAttention(routing_joint)          # 共享 qkv，q@k^T 无 mask
split → dispatch 回各自模态 → 残差 → out_norm
```

`InnerAttention`（`models/rmsa.py:120-153`）是**全注意力**（`attn = q @ k^T`，无任何 mask），因此联合区域集上的注意力矩阵可写成四块：

```
        HE regions   PR regions
HE      [ A_HH ]     [ A_HP ]
PR      [ A_PH ]     [ A_PP ]
```

四块**全部存在且可学习**：既有 HE↔PR 交互（A_HP/A_PH），也保留模态内交互（A_HH/A_PP）。所以：

- **Q3 的答案成立一半**：结构上确实「强制」了 HE/PR 在 routing 层交互（concat 后过同一个 attention）。
- 但它**不偏向**跨模态 —— 没有任何机制让 A_HP/A_PH 主导，也没有把 A_HH/A_PP 置零。模型可以「偷懒」靠 A_HH/A_PP（HE 自身已够用）就达到 HE 水平。

### C2. 三处跨模态参数共享 = 隐含「HE/PR 已语义对齐」假设

`CrossStainingCRMSA.__init__`（`models/cross_staining_crmsa.py:75-96`）：
- `self.norm = nn.LayerNorm(dim)` —— **HE/PR 共享**；
- `self.phi = nn.Parameter((dim, crmsa_k))` —— **HE/PR 共享**；
- `self.attn = InnerAttention(...)`（含 qkv/proj）—— **HE/PR 共享**；
- **无 modality embedding**，无任何「这是 HE 还是 PR」的可区分信号。

这隐含假设 HE 与 PR 的 Stage-1 输出已落在同一语义空间。而 PR 是 HE 的虚拟染色（高度冗余、信息量≈HE），于是：

> 模型最省力的解 = 把 PR 当成「另一批 HE-like token」喂给同一套共享参数。结果 PR 无法贡献 HE 之外的**独立**判别信息，只能追平 HE。

### C3. 融合发生在 region-routing 摘要层，而非 patch 级对应

跨模态信息只在 `routing`（每 region 的 K 个摘要 token）之间交换，再按**各自模态自己的 dispatch 权重**回写。patch 级、位置级（HE 第 i 个 patch ↔ PR 第 i 个 patch 同一组织位置）的互补信息在进入 routing 之前就被 softmax 摘要掉了。这也解释了为什么 §3-B1 的 patch 行序问题在功能上「不致命」—— 因为融合层本来就对 patch 对应不敏感。

> 结论：这三个结构限制合起来，就是「现有 two-stage 只能追平、不能稳定超越 HE」的**结构性**答案。它**不是代码 Bug**，而是当前 Stage2 融合机制的能力上限。

---

## 5. Full Forward Dataflow（实测，N=2500）

```
input HE [1,2500,768]                input PR [1,2500,768]
   │ patch_to_emb[0] (Linear+ReLU)      │ patch_to_emb[1] (Linear+ReLU)
   ▼                                      ▼
[1,2500,512]                         [1,2500,512]
   │ rrt_he (RRTEncoder, 4/9/3)          │ rrt_ihc (RRTEncoder, 8/15/5)
   ▼                                      ▼
z_he [1,2500,512]                    z_pr [1,2500,512]
   └─────────────── cross_region_mod (CrossStainingCRMSA) ────────────────┘
                    (norm → combine → cat routing → joint attn → dispatch → residual)
                                    ▼
                    z_final [1, 5000, 512]   (HE 前 2500，PR 后 2500)
                                    │ mil (AttentionMIL)
                                    ▼
                          logits [1, 2]  →  softmax →  y_prob[:,1] →  AUC
```

关键点：`self.rrt_encoder`（MM_RRTEncoder）**不在此数据流中**（修复前为死模块）。

---

## 6. Gradient Flow 表（实测，修复后，N=2500，CE loss）

| 模块 | 参数个数 | grad is None? | grad_norm | 相对更新量级 |
|---|---|---|---|---|
| `patch_to_emb[0]`（HE 投影） | 2 | 否 | 2.79 | ✓ 更新 |
| `patch_to_emb[1]`（PR 投影） | 2 | 否 | 2.93 | ✓ 更新 |
| `rrt_he`（HE RRT） | 17 | 否 | 1.24 | ✓ 更新 |
| `rrt_ihc`（PR RRT） | 17 | 否 | 1.55 | ✓ 更新 |
| `cross_region_mod`（Stage2 CR-MSA） | 9 | 否 | 2.87 | ✓ 更新 |
| `mil`（ABMIL） | 8 | 否 | 2.78 | ✓ 更新 |
| `rrt_encoder`（死模块） | 38 | **（修复前）是** | 0.0 | ✗ 从不更新 |

修复后：`grad is None` 的参数 = **0**；总参数 6,317,995（原 10,790,654）。

冒烟测试（`scripts/smoke_test_audit.py`）6 项全过：
1. 四变体（HE-only / PR-only / two-stage staining_msa / two-stage concat）均能构造；
2. forward shape 正确（`logits [1,2]`；staining_msa `[1,5000,512]`，concat 同维）；
3. backward 每个活跃模块 `grad_norm>0`；
4. optimizer.step 后每个活跃模块参数确实变化；
5. 固定 seed=42 两次构造 → forward 输出**逐位一致**；seed=42 vs 123 → 不同；
6. 二分类 metrics：`sensitivity=0.714 ≠ specificity=0.800`（`_macro` 仍=0.757=balanced acc）。

---

## 7. 修改文件

| 文件 | 行 | 改动 |
|---|---|---|
| `models/mm_rrt_encoder.py` | `initialize_weights` (40) | 单模块初始化（去内部 `module.modules()` 双重遍历） |
| `models/abmil.py` | `initialize_weights` (151) | 同上 |
| `models/mm_rrt_abmil.py` | 166-167 | two_stage_region/two_stage_direct 不创建死模块 `rrt_encoder` |
| `models/mm_rrt_abmil.py` | 923 | `get_model_info()` 增加 `rrt_encoder is not None` 守卫 |
| `utils/metrics.py` | 64-69 | 二分类新增 tumor-positive `sensitivity`/`specificity` |

新增诊断脚本（未纳入训练流程）：`scripts/diag_audit.py`、`scripts/smoke_test_audit.py`。

---

## 8. 未解决问题（刻意不改，符合 §13 禁令）

1. **Stage2 融合机制能力上限**（C1–C3）：联合自注意力 + 共享 norm/phi/qkv + 无 modality embedding。这是「追平而不超越」的根因，但属于「重新设计融合模型」范畴，本轮**不改**。
2. **patch 行序未排序**（B1）：建议后续 `sorted()` + 存坐标，本轮未动（避免重新提取特征）。
3. **`set_seed` 缺 `random.seed()` + benchmark=True**（B2）：可复现性为「训练间抖动级」，未强制 deterministic。
4. **`train.py` best-val 敏感度口径**（B3）：仍读 `sensitivity_macro`，仅日志影响。

---

## 9. 五个最终问题回答

**Q1. 现在还有没有某个 Bug 在压制性能（把 AUC 拉低）？**
没有。已确认的 3 个 Bug（A1 重复 init / A2 死模块 / A3 指标口径）**均不影响 AUC**（A1/A2 是冗余/死代码，最终权重分布不变；A3 只影响报告值）。loss/head/AUC 计算、两个 RRT 的梯度、CR-MSA 的交互，实测全部正确。

**Q2. 两个 RRT 是否都收到梯度？**
是。实测 `rrt_he` grad_norm=1.24、`rrt_ihc` grad_norm=1.55，且 optimizer.step 后参数确实更新。之前的「38 个 grad=None」参数**全部**来自死模块 `rrt_encoder`，与 `rrt_he`/`rrt_ihc` 无关。

**Q3. CR-MSA 是否真的强制了 HE-PR 交互？**
是（结构上）。`routing_joint = cat([routing_he, routing_pr])` 后过**同一个无 mask 的 InnerAttention**，HE routing token 与 PR routing token 之间确实存在可学习的注意力边（A_HP/A_PH）。但注意它是**双向、全连接、无偏向**的 joint attention，并非「只保留跨模态」的 directed cross-attention。

**Q4. CR-MSA 是否会主要学到模态内（intra-modality）而不是跨模态？**
**有可能，且这是当前最可疑的能力瓶颈。** 因为四块 A_HH/A_HP/A_PH/A_PP 全部在场、共享 norm/phi/qkv、无 modality embedding、PR 又与 HE 高度冗余。模型最省力的解就是「HE 主导 + PR 当 HE-like 附加 token」，从而只追平 HE。这与实测结果（two-stage unified-LR ≈ HE 0.8505 vs 0.8494）一致。

**Q5. 现在能否认为训练代码基本正确，剩下的问题是 Stage2 融合机制？**
**可以。** 训练/评估/数据主链路已逐项验证正确，3 个已确认 Bug 与 AUC 无关并已修复。剩余差距（追平而不超越）来自 Stage2 融合机制的结构性能力上限（C1–C3），而非代码错误。下一步若要突破，应聚焦 Stage2 融合机制本身（例如给跨模态一个更强的归纳偏置），但这属于 §13 明确禁止的「重新设计融合模型」，需另行立项。
