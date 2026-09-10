# HE-Anchored 统一模型（HE + 任意辅助染色 + 可插拔 MIL）—— 实现与验证报告

本轮只做**实现、测试与冒烟**：不训练完整模型、不做组合搜索、不声称性能提升。
历史 v3 路径与结果未被改动（见 §6 改动清单）。

---

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `models/he_aux_unified.py` | 统一模型：`HEAuxUnifiedModel` + 构建器 + v3 checkpoint 映射/加载 |
| `models/mil_heads.py` | 可插拔 MIL 头接口：`build_mil_head` / `MILHeadAdapter` / head adapter 注册表 |
| `train.py` | 新增 `stage2_type='he_aux_unified'` 分派 + `_create_he_aux_unified` + checkpoint 存 `model_config`；C16 数据集透传 `strict_modalities` |
| `data/c16_multimodal_dataset.py` | 新增**可选** `strict_modalities`（默认 `False`，历史行为不变） |
| `tests/test_he_aux_unified.py` | 6 组验证（回归 / 多模态 / EMA / MIL 解耦 / 配置与恢复） |
| `scripts/smoke_he_aux_unified.py` | 真数据冒烟：三个配置的前向/反向 + 缺失染色报错 |
| `configs/he_aux_unified/{he_only,he_pr,he_4aux}.json` + `README.md` | 三个配置示例 |

---

## 2. 接口

### 2.1 数据流

```
每染色特征 [B, N_m, input_dim]
  → patch_to_emb[stain] → dp                    独立投影（ModuleDict，按染色名）
  → rrt[stain]                                  独立 Stage-1 RRT（ModuleDict，按染色名）
  → 融合（下节）
  → tokens [B, N_HE, D]
  → mil(tokens=fused, mask=valid_HE)            可插拔 MIL 头
  → logits [B, num_classes]
```

### 2.2 模型接口

```python
from models.he_aux_unified import build_he_aux_unified, build_he_aux_unified_from_config

model = build_he_aux_unified(modality_list=['HE','ER','PR','HER2','Ki67'],
                             input_dim=768, mlp_dim=512, num_classes=2,
                             encoder_cfg={'PR': {...}}, stage2_cfg={'residual_scale': 0.1},
                             mil_cfg={'name': 'abmil', 'kwargs': {'hidden_dim': 256}})

model.fuse(features)                 # -> (fused, H, {stain: branch_output})
model.encode_he(features)            # -> H  [B, N_HE, D]（每次 forward 只算一次）
model(x)                             # -> (logits, Y_hat, attention, fusion_stats, aux_loss)
model.get_config()                   # -> 完整构建配置（存进 checkpoint）
model.get_modality_names()           # -> ['HE', ...]
```

- `x` 可以是 `{stain: [B,N,D]}`，或按 `stain_order` 排列的 list/tuple（等价，`modality_names` 可交叉校验）。
- `valid_masks={stain: [B,N] bool}` 可选；不传时走历史 MIL 前向路径。
- **配置了但没喂的染色会显式报错**（`ValueError` 列出 missing/unexpected），不静默跳过；mask 形状不符、未知染色名同样报错。
- `modality_list` 必须含 HE 且无重复；`data.modalities` 必须以 HE 开头。

### 2.3 MIL 接口（固定契约）

```python
head = build_mil_head(name=mil_cfg["name"], input_dim=feature_dim,
                      num_classes=num_classes, **mil_cfg.get("kwargs", {}))
result = head(tokens=fused, mask=valid_he)
# 必须：result["logits"] -> [B, num_classes]
# 可选：result["embedding"], result["attention"]
```

- MIL 名字先查 `MIL_REGISTRY`，再看有没有 head adapter；**未注册 / 未适配都显式报错**，不静默回退 ABMIL。
- adapter 只是把某个 MIL 包装成上面的契约，`ABMILHead` 直接复用被包装模块自己的 `attention` / `classifier`，**不新增池化、不新增分类器**，历史权重原样加载。
- `mask=None` 时 adapter 是纯透传（连 train 模式 dropout 的 RNG 顺序都不变），`mask!=None` 才走带 mask 的批量化路径（softmax 只在有效 token 上）。
- 主模型与训练循环只依赖 `logits`：`forward` 不假设任何 MIL 会返回 attention 或 slide embedding。
- 之后加 TransMIL 只需实现一个 adapter 并 `register_head_adapter`，RRT / 融合 / 训练主循环都不用改。

---

## 3. 融合公式

```python
H = he_encoder(he_features)                     # [B, N_HE, D]，每次 forward 只算一次
if not auxiliary_modalities:
    fused = H
else:
    branch_outputs = [cross_branch[m](H, encoded_features[m]) for m in auxiliary_modalities]
    fused = torch.stack(branch_outputs, dim=0).mean(dim=0)
```

即 **`H_fused = H + (residual_scale / M) · Σ_m Δ_m`**，`M` = 辅助染色个数。

要点：
- **每个分支的输出已经含 HE 残差**（`branch_m = H + residual_scale·Δ_m`，即 v3 的 `forward` 原样），
  所以是"分支输出求平均"，**不再加一次 H、不再乘一次 `residual_scale`**。
- 单辅助染色 ⇒ 平均退化成那一个分支 ⇒ **与历史 v3 完全一致**（§4 实测误差 0.0）。
- 无辅助染色 ⇒ 输出 `H` 本身，没有 cross 分支。
- 所有分支读**同一份未更新的 H**，并行计算，不串行改写 HE。
- `M` 由配置决定（染色子集），本轮不做 per-sample 缺失染色策略。
- 默认值与 v3 一致：`tau=0.2`、`residual_scale=0.1`、`prototype_momentum=0.99`、bias-free QKV。
  未引入新 gate / 辅助 loss / 蒸馏 / 梯度分工 / 额外归一化。

---

## 4. v3 回归误差

`tests/test_he_aux_unified.py` Test 1 —— 用历史 checkpoint
`results/stage2_he_residual_cross_v3/he_residual_cross_v3/seed42/ckpt/best_model.pt`
的**同一份权重、同一份 HE/PR 输入**，在 eval 下对比旧 `MM_RRT_ABMIL` 与新 `HEAuxUnifiedModel`：

| 对比项 | 最大绝对误差 |
|---|---|
| 融合输出 `fused`（B=1） | **0.000e+00** |
| ABMIL logits（B=1） | **0.000e+00** |
| ABMIL logits（B=3） | **0.000e+00** |
| ABMIL logits（**完全随机权重**，不加载 checkpoint） | **0.000e+00** |

- 63 个 legacy tensor **一一映射**到新状态字典（键集合完全相同，`strict=True` 加载通过）。
- 旧模型 logits `[[-4.5222, 3.6496]]`，新模型完全相同；`|logit gap| = 8.196`（非退化，不是"两边都是 0"）。
- 随机权重那行说明匹配的是**结构**，不只是加载进来的张量。

Test 2（HE-only 回归）：`fused == H`；HE-only 模型的 logits 与"HE+PR 模型的 `MIL(H)`"逐位相同；
把同样的权重灌进历史单模态 `MM_RRT_ABMIL`，logits 误差 **0.000e+00**。
同时验证 HE 编码器与 MIL 的初始化与辅助分支数量无关（HE-only / HE+PR / HE+4aux 三者共享模块权重逐位相同）。

---

## 5. 冒烟结果（真数据）

### 5.1 真数据前向/反向（`scripts/smoke_he_aux_unified.py`，C16，cuda:2）

| 配置 | 磁盘上染色文件数 | 数据集 | 参数量 | 辅助分支梯度（3 张真实 slide） |
|---|---|---|---|---|
| `he_only` | HE=399 | 270 slides / 1 模态 | 2,763,051 | —（无分支） |
| `he_pr` | HE=399, PR=399 | 270 slides / 2 模态 | 6,319,531 | PR: 14 tensor / 1,055,744 参数，‖g‖ = 1.49e-01, 2.97e+00, 3.53e+00 |
| `he_4aux` | HE/ER/PR/HER2/Ki67 各 399 | 270 slides / 5 模态 | 16,985,635 | ER 1.08e-01…, PR 9.46e-02…, HER2 9.80e-02…, Ki67 9.36e-02…（各 14 tensor / 1,055,744 参数） |

每个配置均验证：数据集的模态数 == 配置模态数、各染色 patch 数一致、每个辅助分支都被执行、
每个分支参数都在 optimizer 里且无重复、梯度存在且有限、`prototype_initialized=True`。
三个配置的 loss 均为有限值（如 he_pr: 0.0318 / 0.5412 / 0.2836）。

缺失染色检查（把最后一个染色指向不存在的目录）：
- `strict_modalities=True` → **`FileNotFoundError`**（列出染色名与路径）✔
- `strict_modalities=False` → 静默得到 **0 条样本**（这就是历史行为，也正是要避免的静默缩小）

### 5.2 端到端 `train.py`（`max_patches=256`，仅为打通流程）

```
python train.py --config results/smoke_he_aux_unified/he_pr/config.json
```

- he_pr（2 epoch，exit 0）：`Val AUC 0.5538 → 0.6166`，best model 正常保存；
  日志确认 `Params covered by the optimizer: 60 unique tensors (no duplicates)`、
  `Fusion: H + (residual_scale / M) * sum_m delta_m (residual_scale=0.1, tau=0.2, beta=0.99)`。
- he_4aux（1 epoch，exit 0）：`Val AUC 0.6750`（`Best val_auc: 0.6750`，`Done.`），best model 正常保存，
  `Params covered by the optimizer: 159 unique tensors (no duplicates)`。
- checkpoint 内含 `model_config`（`model_family` / `modality_list` / `aux_stains` / `encoder_cfg`
  + `encoder_cfg_resolved` + `encoder_cfg_source` / `stage2_cfg` / `fusion` / `mil_cfg` / `init_seed`），
  用同一份配置重建模型后 `load_state_dict(strict=True)` 通过（he_4aux：171 tensor，missing=0 / unexpected=0）；
  同一 checkpoint 也能被 `load_v3_checkpoint_into_unified` 检查式加载。

### 5.3 融合统计日志（本轮修掉的一处误导）

`train.py` 的统计打印是一长串 `elif <旧架构的键>` 分派，末尾 `else` 兜底打印
`alpha` / `delta_norm` / `att_norm`。这三个键属于 v1/v2 时代的 `fusion_stats`，统一模型用的是
`z_he_norm` / `branch_delta_norm` / `n_branches`，于是**落到兜底分支、打印 `.get(..., 0)` 的默认值**：

```
Fusion: α=0.0000 delta_norm=0.000 att_norm=0.000     ← 全是 0，看着像"残差塌成 0"
```

这是日志假象，不是测量结果（历史上 v2 确实塌成过 0，很容易误读）。已在兜底分支前加了
一条 `'branch_delta_norm' in gate_stats` 的分支，打印真实值：

```
Fusion(HE+1aux): |H|=22.628 |Δ| PR:0.6575
```

用两个已训练冒烟 checkpoint 在真实 val 数据上前向 3 张 slide（`max_patches=256`，eval 模式）实测的残差范数：

| 配置 | `\|H\|` | 每分支 `\|Δ_m\|` |
|---|---|---|
| he_pr | 22.62 | PR ≈ 0.51 – 0.58 |
| he_4aux | 22.64 | ER ≈ 1.05 – 1.17, PR ≈ 0.76 – 0.83, HER2 ≈ 0.79 – 0.85, Ki67 ≈ 0.81 – 0.87 |

即残差约占 HE 表征范数的 2–5%，**不是 0**。这四个分支彼此量级相近、互不共享。

> 注：以上 AUC 是 `max_patches=256`、1–2 epoch 的**流程冒烟**，不构成任何性能结论。

---

## 6. 改动清单与兼容性

**新增**：`models/mil_heads.py`、`models/he_aux_unified.py`、`tests/test_he_aux_unified.py`、
`scripts/smoke_he_aux_unified.py`、`configs/he_aux_unified/*`、本报告。

**改动**（都只做加法，历史路径不受影响）：
1. `train.py::create_model` 开头加了 `stage2_type == 'he_aux_unified'` 的分派；其余 `stage2_type` 走原 `MM_RRT_ABMIL` 分支，一行未动。
2. `train.py` 新增 `_create_he_aux_unified`（不改动既有方法）。
3. 两处 checkpoint 保存点加了 `model_config`，用 `hasattr(model, 'get_config')` 守卫 ⇒ 老模型 checkpoint 的键集合不变。
   另在训练的统计打印里新增一条 `'branch_delta_norm' in gate_stats` 分支（见 §5.3）；老架构的键不含它，
   原有分支顺序与行为一行未动。
4. `train.py` C16 数据集构造透传 `data.strict_modalities`，**缺省 `False`**。
5. `data/c16_multimodal_dataset.py` 新增可选参数 `strict_modalities`（默认 `False`）+ 两处 strict 检查；默认路径代码与之前逐行等价，历史实验结果不受影响。

**未改动**：`models/he_residual_cross_crmsa_v3.py`、`models/he_residual_cross_crmsa.py`、
`models/mm_rrt_abmil.py`、`models/abmil.py`、`models/mil_registry.py`、
`models/mm_rrt_encoder.py`、`models/rmsa.py`、270/129 val-as-test 协议与全部历史结果。

### 训练与初始化

- 仍是**单一最终 CE**、正常端到端反传、单一 Adam 参数组、**全模型统一 `clip_grad_norm_(1.0)`**（与 v3 相同）。
- 所有启用的分支 + MIL 参数都在 optimizer 里，且 `id()` 去重检查无重复参数。
- 染色遍历顺序固定为 `[HE] + 配置顺序`；每个子模块在**自己的 seeded RNG scope** 内构造并恢复调用者的 RNG 状态
  ⇒ 增删辅助分支不会改变 HE / MIL 的初始权重（Test 2 逐位验证）。
- 加载 v3 checkpoint 时做**显式参数映射 + 完整性/形状校验**，任何缺失、多余、形状不符、未映射的键都会
  `ValueError` 列出（不是掩盖问题的 `strict=False`）。这条校验在开发中确实抓到了一个编码器配置优先级 bug。

## 7. 已知限制 / 本轮未做

- 只支持**配置级**染色子集，不做 per-sample 缺失染色策略；某染色配置了就必须有（否则显式报错）。
- 保留 valid-token mask 通路，但不强制跨染色 patch 行级对应（C16 pipeline 目前不产生 mask，`mask=None` 是常规路径）。
- TransMIL 未实现（本轮只接线并验证 ABMIL）；接入方式见 §2.3。
- 未跑完整训练、未做 matched/mismatched 审计、未做染色组合搜索。
- `he_pr.json` 之外的两个示例配置尚未做过完整训练。
