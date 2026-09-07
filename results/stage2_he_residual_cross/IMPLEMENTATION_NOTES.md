# Stage2 改造 — `he_residual_cross` 实现说明

新 Stage2 模块 `models/he_residual_cross_crmsa.py::HEResidualCrossCRMSA`（`stage2_type='he_residual_cross'`）。
本文记录其**与旧 he_anchor / correction 的差异**，以及关键实现决策。

## 一、数学定义（任务书）

```
Stage 1（不变）:  HE → RRTEncoder → Z_HE [B,N_HE,D]
                  PR → RRTEncoder → Z_PR [B,N_PR,D]

Stage 2（本模块，有向 HE→PR cross-attn + HE 残差写回）:
  R_HE = Route(Z_HE)                 # 每 region 保留 crmsa_k 个 routing token
  R_PR = Route(Z_PR)
  Q    = W_q_he( attn_norm_he(R_HE) )
  K    = W_k_pr( attn_norm_pr(R_PR) )
  V    = W_v_pr( attn_norm_pr(R_PR) )
  Δ_rt = W_out( softmax(Q·Kᵀ/√head_dim + valid_mask) · V )   # [B, G_HE·k, D]
  Δ_HE = Dispatch_HE(Δ_rt)                                   # [B, N_HE, D]
  out  = Z_HE + residual_scale · Δ_HE                         # [B, N_HE, D] → ABMIL
```

要点（全部在代码中强制，测试覆盖）：

- **独立模态 routing**：`route_norm_he/pr`、`phi_he/pr` 是**独立实例**（非共享 Parameter/module），
  每 region 保留 `crmsa_k` 个 routing token，不会塌缩成单 token 平均池化。
- **全槽位 cross-attn**：Q 只来自 HE，K/V 只来自 PR；每个 HE 槽位 attend 所有有效 PR 槽位
  （无同槽位限制），按 `head_dim` 缩放（非 D），`assert D % heads == 0`。
- **HE 残差写回**：`delta_patch = Dispatch_HE(Δ_rt)`；`Z_HE' = Z_HE + residual_scale·Δ_HE`。
  残差基底**必须是原始 Z_HE**（恒等 skip）。禁止：把 HE patch 替换为 broadcast/pooled HE、
  残差后接 FFN/LayerNorm/self-attn、拼接 PR token、复制 HE、`MLP([HE, cross_out])` 作 delta；
  不新增 Stage2 out_norm。
- **padding/mask**：有效性来自长度或显式 mask（**非**“特征是否全零”推断）。combine 沿 patch 轴 P、
  dispatch 沿 slot k、`dispatch_weights_mm` 沿 patch 轴 min-max。空 region → routing 全零 + 槽位标无效。
  某样本无任何有效 PR routing → Δ=0，严格保留 HE。保留 raw logits（不原地写 -inf）。
- `residual_scale` 默认 0.1，**固定不学习**（`register_buffer`）；`disable_cross=True` 或
  `residual_scale==0` → 严格返回 Z_HE。

## 二、与旧 he_anchor（`CrossRegionReembedding`）的关键差异

| 维度 | 旧 `he_anchor` | 新 `he_residual_cross` |
|------|----------------|------------------------|
| 区域 token 提取 | **共享** `region_pool` MLP→softmax，**每 region 1 个 token**（平均池化式塌缩） | **独立** `phi_he/pr`，每 region **crmsa_k 个** routing token |
| Q 来源 | **可学习 embedding** `query_pos`（broadcast） | **来自 HE routing token**（`W_q_he∘attn_norm_he(R_HE)`） |
| K/V 来源 | `concat([R_HE, R_PR])`（**HE+PR 都进 K/V**） | **只来自 PR**（`W_k/v_pr∘attn_norm_pr(R_PR)`） |
| 输出方式 | 统一 token **broadcast 回 patch**（把 HE patch 整块替换成区域 token） | **逐 patch 残差写回** `Z_HE + scale·Δ_HE`（保留原始 HE 身份） |
| 残差基底 | 区域 token `r_he`（`r_he + drop_path(delta)`） | **原始 patch `Z_HE`**（恒等 skip） |
| 输出投影 | **zero-init**（init 时恒等） | **xavier_normal**（init 时残差非平凡） |
| 残差后 | `norm_out(r_he + delta)` 后接 **LayerNorm** | **无** out_norm / FFN / self-attn |

## 三、与旧 correction（`use_correction_only` / PR→HE correction）的差异

旧 correction 的机制是**用 PR 去 steering HE 的 routing**（PR 校正 HE 的区域 token 方向）。
历史结论（见 memory）：correction ≈ concat ≈ 0.74，比 HE-only 0.805 低约 0.066 ——
**“PR 去 steering HE routing 本身就是伤害”**。

新模块**刻意不做 PR→HE routing steering**：

- HE 的 routing token（`phi_he` / `route_norm_he`）完全由 HE 自己决定，PR 不参与 HE 的 routing 参数。
- PR 的信息只通过**有向 cross-attn** 进入：HE 主动 query PR，产出的是加在 HE patch 上的
  **小残差 Δ_HE**（`residual_scale=0.1`），而不是替换/校正 HE 自己的区域表征。
- 因此 HE 的判别主导权（identity skip `Z_HE`）始终保留，PR 只是“补充残差”，
  与“PR 去 steering HE”有本质区别。

## 四、关键实现决策（易错点）

1. **`residual_scale` 存为 buffer**（`register_buffer`），随 checkpoint 保存但不进 optimizer。
   与旧 he_anchor 的“zero-init out_proj + 可学习残差系数”不同：这里残差系数固定，
   out_proj 正常 xavier 初始化，二者不做双重 zero-init。
2. **combine/dispatch/min-max 全部只在有效 patch 上算**：combine 用 masked softmax（无效位置
   `-inf`），min/max 用 `masked_fill(±inf)` 后沿 patch 轴 reduce；空 region 用
   `nonempty_kp=[B,G,1,1]` 广播把 routing / dispatch_weights_mm 显式清零，避免
   `softmax(全 -inf)` → NaN。
3. **cross-attn 的 `-inf` 遮蔽用副本而非原地**：`attn.masked_fill(~k_valid, -inf)`（`masked_fill`
   返回新张量），raw logits 永不原地改。无有效 PR key 时 softmax 全 -inf → NaN，用
   `has_valid_key=[B,1,1,1]` 强制置零。
4. **输出投影后再次清零**：`out *= q_valid.unsqueeze(-1) * has_valid_key.view(B,1,1)`，
   防止 `w_out` 的 bias 在无效 HE query / 全空 PR 样本上“复活”非零 Δ。
5. **显式 mask 下无效 HE patch 的残差置零**：`delta_patch *= valid_he.unsqueeze(-1)`，
   使无效 patch 真正“缺席”（其输入值不会漏进非零 Δ），有效位置的输出对无效 patch 取值不变。
6. **`head_dim` 缩放 + 可整除断言**：`scale = head_dim**-0.5`（`head_dim = dim//num_heads`），
   `__init__` 断言 `dim % num_heads == 0`（512 % 8 == 0）。

## 五、验证

- 合成测试 `tests/test_he_residual_cross_crmsa.py`：**21/21 PASS**（8 条验收标准全覆盖）。
  - 其中 test4a 只校验**有效位置**输出不变（无效 patch 的输入值天然出现在其自身位置的恒等
    skip 中，全输出不变不可能成立），并新增 test4c 校验无效 HE patch 残差严格为零。
- 三条件 × 3 seed（42/123/456）实验见同目录 `README.md` / `summary.json`。

## 六、涉及文件

| 文件 | 动作 |
|------|------|
| `models/he_residual_cross_crmsa.py` | **新建** `HEResidualCrossCRMSA` |
| `models/mm_rrt_abmil.py` | 构造块 + forward 分支新增 `he_residual_cross`；`he_anchor` 显式化；未知 `stage2_type` 抛 `ValueError` |
| `tests/test_he_residual_cross_crmsa.py` | **新建** 8 条验收测试 |
| `scripts/run_stage2_he_residual_experiments.py` | **新建** 3 条件 × 3 seed driver |
| `scripts/_run_stage2_he_residual_seed.py` | **新建** 单 seed worker（训练 + 官方 Test 评估） |
