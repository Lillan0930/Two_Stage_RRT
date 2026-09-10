# `he_aux_unified` 配置示例

HE 为主模态、任意辅助染色组合、可插拔 MIL 头的统一模型（`models/he_aux_unified.py`）的三个示例配置。

| 文件 | `data.modalities` | 说明 |
|---|---|---|
| `he_only.json` | `["HE"]` | 只有 HE。融合输出 = H，无 cross 分支。用于对照 HE-only 基线。 |
| `he_pr.json` | `["HE", "PR"]` | 与历史 v3 配置**逐字段一致**，只把 `stage2_type` 换成 `he_aux_unified`。单辅助染色 ⇒ 退化为原 v3。 |
| `he_4aux.json` | `["HE", "ER", "PR", "HER2", "Ki67"]` | HE + 4 个辅助染色，4 个独立 cross 分支并行读同一份 H。 |

运行方式与既有流程相同：

```bash
python train.py --config configs/he_aux_unified/he_pr.json --exp_name he_aux_he_pr
```

## 与历史 v3 配置的关系

`he_pr.json` 与 `results/stage2_he_residual_cross_v3/he_residual_cross_v3/seed42/config.json`
除 `stage2_type` 外完全一致（`encoder_cfg` HE/PR 两段、`stage2_cfg` 全部 14 个键、LR/协议/early stopping 都相同）。
在同一个 v3 checkpoint 下，新模型的融合输出与 ABMIL logits 与旧模型**逐位相同**（最大绝对误差 0.0，见 `tests/test_he_aux_unified.py` Test 1）。

## 两处刻意的写法差异

1. **`"weight_decay": 0.00001` 而不是 `1e-05`。**
   `train.py --config` 用 `yaml.safe_load` 读配置，而 YAML 1.1 不把 `1e-05` 解析成浮点数，
   会得到字符串 `'1e-05'`，随后 `optim.Adam(weight_decay=...)` 直接抛 `TypeError`。
   （历史的 `scripts/_run_protocol_seed.py` 走 `json.loads`，所以老 config.json 里的 `1e-05` 不受影响。）
   这些示例配置保证可以直接喂给 `train.py --config`。

2. **`"strict_modalities": true`。**
   `data.strict_modalities` 是新增的**可选**开关，默认 `false`（历史行为逐位不变）：
   - `false`（默认）：某个染色目录缺失时，`C16MultimodalDataset` 会静默丢掉所有样本（得到 0 条数据）；
     单张 slide 缺某个染色时静默丢弃该 slide 并打印一行 Warning。
   - `true`：上述两种情况都显式抛 `FileNotFoundError`，绝不静默缩小模态列表或数据集。

   由于本模型要求"配置了就必须有"，示例配置都打开它。历史实验配置不受影响。

## 其它可调项

- `model.mil_cfg`: 现代写法 `{"name": ..., "kwargs": {...}}`（`he_only.json` / `he_4aux.json` 使用）。
  也兼容旧的扁平写法 `mil_type` + `abmil_hidden_dim` + `dropout`（`he_pr.json` 使用，因为它要与 v3 逐字段对齐）。
  未注册的 MIL 名字会显式报错，不会静默回退到 abmil。
- `model.encoder_cfg`: 每个染色一个 Stage-1 RRT 配置。未列出的染色继承 HE 的解析结果，
  并记录在 checkpoint 的 `model_config.encoder_cfg_source` 里（值 `he_default`），不做任何参数搜索。
  `he_4aux.json` 显式写全了 5 个染色；把这 3 个非 PR 染色从 `encoder_cfg` 里删掉，行为完全一样。
- `model.init_seed`: 子模块初始化的基种子。共享模块（HE 编码器 / MIL）的初始化与辅助分支数量无关，
  增删辅助染色不会改变 HE/MIL 的初始权重。
- `training.aux_loss_weight` 必须为 `0`：统一模型只用最终单一 CE。

## 冒烟验证

```bash
python scripts/smoke_he_aux_unified.py --device cuda:2      # 真数据前向/反向 + 缺失染色报错
python tests/test_he_aux_unified.py                         # 6 组回归/接口测试
```
