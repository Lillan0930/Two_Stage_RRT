# 第六轮诊断报告 — PR 相对 HE 的残差表征量化

> 项目：CAMELYON16（C16）WSI 二分类
> 路径：`/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT`
> 日期：2026-09-07
>
> **本轮唯一问题**：PR 中相对于 HE 无法解释的残差信息，具有多强的分类能力，以及加入 HE 后能否带来额外收益？
> **动作约束**：不修改 RRT / CR-MSA / ABMIL，不重新提取特征，不训练新的主模型；只做测试期 embedding 提取 + 线性 probe。

---

## 0. 协议与 checkpoint 选择

- **Model seed**：`42 / 123 / 456`（与 HE/PR RRT checkpoint 的训练 seed 一致）。
- **HE-only RRT**：`results/he_rrt_samplerfix_lr1e4/seed{seed}/ckpt/best_model.pt`
  （baseline encoder：`region_num=4 / epeg_k=9 / crmsa_k=3`）。
- **PR-only RRT**：`results/c16_test_as_val/seed{seed}/ckpt/best_model.pt`
  （tuned encoder：`region_num=8 / epeg_k=15 / crmsa_k=5`，**head/classifier 与 HE 对齐**：`n_heads=4 / drop_path=0 / abmil_hidden_dim=256`）。
  > 选 `c16_test_as_val` 而非 `pr_best_five_seed` 的原因：后者额外调了 `n_heads=8 / drop_path=0.1155 / abmil=384`，
  > 与 HE 的 head/classifier 容量不同，会把「表征内容差异」和「模型容量差异」混在一起。前者 head 结构完全一致，
  > 唯一区别是 RRT encoder 参数（region_num/epeg_k/crmsa_k）+ 模态内容，是最干净的 HE↔PR 残差对比。
- **Embedding**：ABMIL attention pooling 之后、classifier 之前的 slide-level 向量 `h ∈ R^512`
  （= `Z = softmax(A)·z`，与 `MM_RRT_ABMIL` 单模态 forward 完全一致）。
- **Split**：`fixed_split` 216 train / 54 val / 129 test（`c16_test_labels.csv`）。
- **`g`（HE→PR）**：Ridge（`alpha=1.0`，输入 `h_HE` 用 train 统计标准化），**仅用 train 拟合**，train/val/test 共用同一个 `g`。
- **Probe**：统一 `LogisticRegression`（`lbfgs`，`max_iter=5000`），输入用 train 统计标准化，`C` 仅在 val 上从
  `{1e-3,1e-2,1e-1,1,10,100,1000}` 选择，test 仅做最终评估。四种输入用**同一个分类器类型**（不是不同复杂度）。

**数值一致性校验**：对每个 seed 的 HE 模型，手动复现的单模态 forward 与 `model(x)` 的 logits 完全一致
（`max_abs = 0.000e+00`），且 `classifier(Z)` 与手动 logits 也一致（`max_abs = 0.000e+00`）——证明提取的 `Z` 就是模型内部真正的 slide-level embedding。

---

## 1. 主结果表

| Model seed | HE AUC | PR AUC | PR Residual AUC | HE+Residual AUC | Δ vs HE |
| ---------: | -----: | -----: | --------------: | --------------: | ------: |
| 42  | 0.8413 | 0.6957 | 0.6365 | 0.6571 | **−0.1842** |
| 123 | 0.8791 | 0.8250 | 0.4730 | 0.8069 | **−0.0722** |
| 456 | 0.7513 | 0.8031 | 0.6538 | 0.8038 | **+0.0526** |
| **Mean ± Std** | **0.8239 ± 0.0536** | **0.7746 ± 0.0565** | **0.5878 ± 0.0815** | **0.7560 ± 0.0699** | **−0.0679 ± 0.0967** |

其中 `Δ vs HE = AUC(HE+residual) − AUC(HE)`。3 个 seed 中 **2 个 Δ 为负、1 个接近 0 的微正**，均未出现稳定增益。

> 说明：这里的 "HE AUC / PR AUC" 是**线性 probe（LogReg）在固定 split 上的 AUC**，不是 checkpoint 自带 classifier 的 AUC，
> 两者 split 与分类器不同，不可直接对等；本轮四个 probe 使用同一套 split 与分类器，彼此可比。

---

## 2. 辅助量（test set 统计）

| 量 | seed 42 | seed 123 | seed 456 | Mean |
| --- | ------: | -------: | -------: | ---: |
| `‖r_PR‖ / ‖h_PR‖`（残差范数比） | 0.405 | 0.524 | 0.452 | **≈ 0.46** |
| `cos(h_HE, h_PR)` | +0.021 | −0.042 | −0.031 | **≈ 0.00** |
| `cos(h_HE, r_PR)` | +0.004 | −0.025 | −0.001 | **≈ 0.00** |
| `R²(HE→PR)`（train） | 0.977 | 0.938 | 0.989 | **≈ 0.97** |

**解读**：

- `cos(h_HE, h_PR) ≈ 0`：HE 与 PR 的 slide-level embedding 几乎**正交**——两个模态的 RRT 把同一 WSI 映射到了近似不相交的子空间。
- `cos(h_HE, r_PR) ≈ 0`：残差与 HE 近似正交（这是投影残差的应有性质）。
- `R²(HE→PR) ≈ 0.97` + `‖r‖/‖h_PR‖ ≈ 0.46`：Ridge 能解释 PR 的**绝大部分方差（Frobenius 意义上）**，但这是「固定线性映射把 HE 空间系统性地连到 PR 空间」，
  并不是说 `h_PR` 能被 `h_HE` 逐样本地贴近；逐样本仍近似正交（见上），残差向量长度约为 PR 的 46%。
- 综合：HE→PR 存在一条**很强的、固定的线性耦合**（解释了 97% 的方差），但这条耦合**不是**逐样本的「同向对齐」，
  而是把 HE 张成的子空间旋到 PR 的（近）正交子空间；剩下的 3% 方差 / 46% 范数残差里，才装着 PR 独有、HE 解释不了的部分。

---

## 3. 最终回答

### Q1：`r_PR` 单独的 AUC 是多少？

**`0.588 ± 0.082`（3 seed：0.6365 / 0.4730 / 0.6538）。**

高于随机（0.5），但显著低于 HE（0.824）和 PR（0.775），且跨 seed 波动很大（seed 123 的 0.473 几乎等于随机）。
即：HE 无法线性解释的那部分 PR 信息，**仍带有一点点判别力，但很弱、不稳定**。

### Q2：`HE + r_PR` 是否稳定优于 `HE`？

**否。** `AUC(HE+residual) = 0.756 ± 0.070`，低于 `AUC(HE) = 0.824 ± 0.054`，`Δ = −0.068 ± 0.097`。
3 个 seed 里 2 个 Δ 为负（−0.18、−0.07）、1 个接近 0 的微正（+0.05），**没有任何一个 seed 出现有意义的正增益**。
简单地把残差 concat 到 HE 上，不仅不能稳定提升，多数情况下反而**拉低** HE 的性能。

### Q3：当前 PR 增量更接近 A / B / C 哪一种？

**最接近 C（当前 representation 下残差信号较弱），并带一点 B 的痕迹。**

- **不是 A**：残差单独 AUC 只有 0.59，谈不上「明显判别能力」；且 concat 后不能补充 HE（Δ<0）。
- **B 只占很小一部分**：残差确实有微弱、高于随机的判别力（0.59>0.5），但「简单 HE+Residual 不能利用」这个说法成立的前提是「残差有可被利用的判别力」，而这里残差判别力本身就很弱，所以 B 不是主因。
- **主因是 C**：在**当前（未 coordinate-sorted）的 HE/PR RRT 表征下**，HE 之外那部分 PR 残差信息判别力弱、且与 HE 近乎正交但无法线性补充，
  简单 concat 反而稀释/干扰了 HE 的强表征。

> 结论一句话：**当前表征下，PR 相对 HE 的「不可解释残差」只携带很弱的、不稳定的分类信号，并且简单拼接会损害 HE 的强判别力。**
> 这为下一步改造 CR-MSA 提供了量化依据：不要指望「把 PR 残差硬加到 HE 上」能带来增量；若要让 PR 贡献增量，
> 问题不在「PR 是否含有效信息」，而在于当前表征里 PR 的独有信息无法以简单线性/拼接方式被 HE 利用。

---

## 附录：产物与复现

| 文件 | 用途 |
|------|------|
| `scripts/diag_pr_residual.py` | 提取 slide embedding → Ridge 残差 → 4 个线性 probe |
| `results/pr_residual.json` | 3 seed 完整结果 + mean/std |

复现：

```bash
/home/cxl/miniconda3/envs/rrtmil/bin/python scripts/diag_pr_residual.py --gpu 1 --seeds 42 123 456 --out results/pr_residual.json
```

**本轮未修改** RRT / CR-MSA / ABMIL / 训练结构 / split；未重新提取特征；未训练任何主模型。
