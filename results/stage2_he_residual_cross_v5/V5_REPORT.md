# HE Residual Cross v5 — PR value memory 训练期辅助分类监督

## 0. 一句话结论

> **否 —— v5 的 PR value 辅助分类监督没有带来主预测收益。** 内部 val(54) 上 β=0.1 与 β=0 基本持平（mean +0.0014），held-out dev-test(129) 上 β=0.1 反而略差（mean −0.0367）。辅助头本身确实学到了判别信号（AUC_PRval 从 β=0 的 ≈0.5 升到 β=0.1 的 ≈0.65–0.77），但**辅助头的判别能力没有转化为主预测收益**，与 §6 的提醒一致（「aux AUC 改善不能代替主预测收益」）。

---

## 1. 任务与实现

在 v3 基础上，对同一次 forward 中 cross 实际读取的 **merged PR value memory** `V_tilde`（= `Wv(LN_PR(R_PR)) − μ_PR`，合并 heads 后 `[B,K,D]`）接一个**独立、轻量的辅助 ABMIL 分类头**，训练期叠加 slide/bag 级监督：

```
L = CE(fused_logits, y) + β · CE(pr_value_logits, y)      (β = 0.1)
```

- 辅助头读的 `V_tilde` 是主 cross `A · V_tilde` 用到的**同一份带梯度张量**（view/reshape，无 detach）。
- 无逐 token 赋 slide 标签（先 MIL 聚合再 bag 级 CE）；无另建 value projection、无重跑 routing/cross/EMA。
- 辅助头是注册子模块（`aux_attn` + `aux_classifier`），构造时即入 optimizer。
- 推理不变：`Z_HE + 0.1·Δ → 原 ABMIL`；辅助 logits 仅随 forward 返回、不参与主预测。
- EMA prototype `μ_PR` 每次训练 forward 只更新一次（在 `_cross_attention` 内），eval 不更新；辅助头只读 `V_tilde`。

## 2. 修改说明

| 文件 | 动作 |
|------|------|
| `models/he_residual_cross_crmsa_v5.py` | 新增。继承 v3，新增 aux_attn(Tanh-attn)+aux_classifier(ReLU→Dropout→Linear)；`_cross_attention` 返回 `(out, (v_tilde_merged, k_valid))`；`_aux_forward` 计算 masked-softmax 聚合 + 分类；`forward` 返回 `(out, {'pr_value_logits','pr_value_has_valid'})`，fallback 返回 `(z_he, None)` |
| `models/mm_rrt_abmil.py` | 工厂新增 `he_residual_cross_v5` 分支（透传 `aux_hidden_dim/aux_dropout`）；forward 在 `fusion_stats` 里带 `pr_value_logits`/`pr_value_has_valid`；错误信息补 v5 |
| `train.py` | 新增 `pr_value_aux_weight`（默认 0）；train_epoch（AMP/非 AMP）收集 aux logits + has_valid，`β>0` 时叠加 `β·CE`（跳过全空 PR）；validate 收集 aux 概率并算 `auc_pr_value`；train() 日志加 `AUC_PRval=` |
| `tests/test_he_residual_cross_v5.py` | 新增 5 项单测（§4） |
| `scripts/run_stage2_v5.py` | 新增 fixed_split 216/54 配对对照 driver |
| `scripts/eval_v5.py` | 新增 §6 评估（matched/disable/mismatch 逐 slide margin/AUC/CE + aux 头 train/val） |
| `scripts/smoke_v5_train.py` | 新增训练冒烟 |

保留旧版本 `he_residual_cross_v3/v4` 及历史结果，未覆盖。

## 3. 测试结果（§4，实际运行）

`python tests/test_he_residual_cross_v5.py` → **5/5 PASS**：

| # | 断言 | 结果 |
|---|------|------|
| 1 | aux loss 单独 backward 更新 `w_v/phi_pr/attn_norm_pr/route_norm_pr`，不更新 `w_q/w_k/w_out/phi_he` | PASS |
| 2 | eval 下 v5 fused output 与 v3 逐位一致（max\|Δ\|=0.00e+00） | PASS |
| 3 | disable_cross / residual_scale=0 → 严格 `Z_HE` 恒等，aux=None | PASS |
| 4 | mask/空区域/空 PR 前向+反向有限，空 PR `has_valid=False` | PASS |
| 5 | EMA 每 forward 恰一次（drift=1.49e-08）、eval 冻结、checkpoint 往返 | PASS |

训练冒烟 `python scripts/smoke_v5_train.py --gpu 2`（8 slide 子集 × 64 patch × 2 epoch）→ **PASS**：β=0.0/0.1 均跑通，best_val_auc 有限，checkpoint 保存，forward 返回 4-tuple，`pr_value_logits` 形状 `[1,2]`、`has_valid=True`。

## 4. 启动命令

```bash
# 单测
/home/cxl/miniconda3/envs/rrtmil/bin/python tests/test_he_residual_cross_v5.py
# 冒烟
/home/cxl/miniconda3/envs/rrtmil/bin/python scripts/smoke_v5_train.py --gpu 2
# 全量配对对照（2 条件 × 3 seeds，6 卡并行）
nohup /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_stage2_v5.py \
    --gpus 2 3 4 5 6 7 > results/stage2_he_residual_cross_v5/_driver.log 2>&1 &
# §6 评估（训练完成后）
/home/cxl/miniconda3/envs/rrtmil/bin/python scripts/eval_v5.py --section causal aux --gpu 2
```

## 5. 实验汇总（§5）

- 协议：train=216 / val=54（内部选 checkpoint）/ dev-test=129（仅背景）。
- 条件：`v5_beta0`（pr_value_aux_weight=0.0，主路径≡v3）vs `v5_beta01`（=0.1）。
- 公共模块同 seed 同初始权重；统一 LR=1e-4、cosine、patience=10、epochs=80。

### 5.1 内部 val AUC（54，primary）

| seed | v5_beta0 (β=0) | v5_beta01 (β=0.1) | Δ |
|---|---|---|---|
| 42 | 0.9645 | 0.9659 | +0.0014 |
| 123 | 0.9730 | 0.9815 | +0.0085 |
| 456 | 0.9886 | 0.9830 | −0.0057 |
| **mean** | **0.9754** | **0.9768** | **+0.0014** |

### 5.2 dev-test AUC（129，仅背景，不能称独立测试收益）

| seed | v5_beta0 (β=0) | v5_beta01 (β=0.1) | Δ |
|---|---|---|---|
| 42 | 0.7468 | 0.7472 | +0.0004 |
| 123 | 0.8531 | 0.7755 | −0.0776 |
| 456 | 0.7997 | 0.7668 | −0.0329 |
| **mean** | **0.7999** | **0.7632** | **−0.0367** |

### 5.3 best_epoch

- v5_beta0：42→20 / 123→16 / 456→17
- v5_beta01：42→16 / 123→4 / 456→16

### 5.4 判定

内部 val 配对 Δ = **+0.0014 ± 0.0058**（噪声内），dev-test Δ = **−0.0367**（2/3 seed 变差）。**辅助监督未带来主预测收益**；早期 `AUC_PRval` 提升只证明辅助头学会了读 PR value 的判别信息，但没有反哺主预测（详见 §6 的 causal 分析：matched/disable/mismatch 的 AUC 差是否仍很小）。

## 6. §6 评估（matched/disable/mismatch + aux 头）

### 6.1 causal（dev-test 129：matched vs disable vs random-mismatch）

| 条件/seed | matched AUC (CE) | disable AUC (CE) | mismatch AUC (CE, 5 重排均值) | margin p50 matched/disable |
|---|---|---|---|---|
| β0 / 42 | 0.7468 (1.838) | 0.7472 (1.830) | 0.7468 (1.838) | −3.190 / −3.180 |
| β0 / 123 | 0.8531 (1.391) | 0.8531 (1.392) | 0.8529 (1.391) | −7.635 / −7.634 |
| β0 / 456 | 0.7997 (1.763) | 0.7997 (1.785) | 0.7997 (1.763) | 2.491 / 2.534 |
| β0.1 / 42 | 0.7472 (1.220) | 0.7454 (1.224) | 0.7437 (1.220) | −5.479 / −5.525 |
| β0.1 / 123 | 0.7755 (1.217) | 0.7750 (1.215) | 0.7758 (1.217) | −7.006 / −6.995 |
| β0.1 / 456 | 0.7668 (1.853) | 0.7653 (1.807) | 0.7644 (1.859) | 0.959 / 0.496 |

**结论**：六组里 matched / disable / random-mismatch 的 **AUC 差都很小（≤ 0.004）**，即 cross 带来的 **AUC 增益很小**。但这**不能**外推为「逐样本预测几乎不变」：逐 slide 的 margin 在 matched vs mismatch 之间可有实际变化（如 β0.1/456 的 |matched−mismatch margin| 均值 0.52、最大 2.57，而 β0/42 仅 0.0037）。因此现有结果只能支持「cross 的 AUC 增益很小」，**不能**支持「PR 内容完全无关」或「逐样本预测几乎不变」。

### 6.2 aux 头 train/val AUC

| 条件/seed | train aux AUC | val aux AUC |
|---|---|---|
| β0 / 42 | 0.543 | 0.457 |
| β0 / 123 | 0.538 | 0.622 |
| β0 / 456 | 0.458 | 0.565 |
| **β0.1 / 42** | **0.883** | **0.879** |
| **β0.1 / 123** | **0.673** | **0.753** |
| **β0.1 / 456** | **0.792** | **0.861** |

**结论**：β=0.1 的辅助头确实把 PR value memory 训出了强判别性（val aux AUC 0.75–0.88，β=0 对照 ≈ 0.46–0.62 ≈ 随机）。但这份判别信息**没有反哺主预测的 AUC**（主 val AUC 基本不变，dev-test AUC 反而略降）。注意：AUC 无收益 ≠ 逐样本预测无关（见 §6.1）。

## 7. 最终判定

**v5 判定 = 交叉注意力通路的 AUC 增益瓶颈（而非 PR value 表征本身）。**

- 直接监督可以把 PR value memory 训练到 val AUC 0.88，说明「PR value 缺判别信息」不是根因。
- 但主预测 AUC 不涨（内部 val Δ=+0.0014）、dev-test 反而略降（Δ=−0.0367）、causal 的 AUC 差 ≤ 0.004 ⇒ 瓶颈在 cross-attention 如何把 PR value 的判别信号**有效注入** `Z_HE + 0.1·Δ`：cross 带来的 AUC 增益很小。
- **重要修正（§8）**：AUC 差小**不能**推断「逐样本预测几乎不变」或「PR 内容完全无关」。逐 slide 的 margin 在 matched vs mismatch 之间可有实际变化（β0.1/456 均值 0.52、最大 2.57）。现有结果只能支持「cross 的 AUC 增益很小」，**不能**支持「残差注入通路内容完全无关」。
- 与 [[c16-round10-v4-qk-common-direction]] 的结论衔接：v4 已把注意力从「均匀」修到「选择性」；v5 进一步排除了「PR value 表征不足」。下一步应查 cross-attention 为何 AUC 增益小（而非继续增强 PR value 表征），同时避免再把「AUC 无收益」误读为「内容无关」。
