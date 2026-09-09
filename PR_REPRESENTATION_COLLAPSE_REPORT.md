# PR Representation Collapse Localization 报告

**日期**: 2026-09-07
**对象**: `he_residual_cross_v2` 已训练 checkpoint（seed 42/123/456），**不重训 / 不改模型 / 不调参**
**协议**: Train / Val-as-Test（129 WSIs）
**产物**: `results/stage2_he_residual_cross_v2/_eval/collapse_localization.json`
**脚本**: `scripts/diag_collapse_localization.py`（逐层截取+指标） / `scripts/gen_collapse_report.py`（本报告）

---

## 0. 结论

**PR token diversity 在**两处**塌缩，且都不在 PR-RRT（Stage1 编码器）本身：**

1. **幅度最大的塌缩 = `patch_to_emb` 嵌入层（L0→L1）**：pairwise cosine 0.777 → 0.919（**Δ=0.1417**，全流程最大），effective rank 180.0 → 57.7。这一步发生在 RRT **之前**，是 768→512 线性投影 + GELU 造成的特征同质化。

2. **对 cross-attention 致命的一处 = Stage2 routing（L3→L4）**：pairwise cosine 0.926 → 0.984（Δ=0.0583），**effective rank 53.0 → 3.3（~16× 塌缩）**。这一步把 48 个 routing token 压进 ~3.3 维子空间，直接导致 cross-attention 的 K/V 无可选择的信息。

**Stage1 RRT（L1→L2）几乎不塌缩**（Δpc=0.0067，Δeffrank=-4.7），**K/V（L5→L7）只增加微小塌缩**（K 0.993、V 0.988）。

**关键反向证据（排除 competitive routing 作为首选）**：routing 的 combine weight 槽间 cosine 仅 0.810、top-10 patch overlap 仅 0.390 —— **不同 slot 已经在选不同的 patch**，但 routing token 仍塌缩到 0.992（同 region）/ 0.984（跨 region）。这说明塌缩**不是** routing 缺少 slot 竞争，而是被聚合的 Z_PR patch 本身已高度同质（0.926），其根子在上游嵌入层。

**最终判断（对应任务 §8）**：既非纯粹的 Case A（Z_PR=0.926 未到 0.95、且 RRT 本身无责），也非可被 competitive routing 解决的 Case B（slot 竞争已存在）。是 **上游 PR 表征（嵌入投影）主导 + routing 放大** 的两步塌缩。

---

## 1. 核心表（3-seed mean，val-as-test）

**PR pipeline**

| Layer | Pairwise cosine ↓ | Token variance ↑ | Effective rank ↑ | Centroid cosine ↓ |
|---|---|---|---|---|
| PR Input (X_PR) | 0.777 | 0.005 | 180.0 | 0.941 |
| PR Embed (E_PR) | 0.919 | 0.001 | 57.7 | 0.982 |
| PR RRT output (Z_PR) | 0.926 | 0.039 | 53.0 | 0.983 |
| Route pre-LN | 0.927 | 0.039 | 52.5 | 0.983 |
| Routing token (R_PR) | 0.984 | 0.007 | 3.3 | 0.982 |
| Attn-LN | 0.989 | 0.007 | 3.3 | 0.982 |
| K (W_k) | 0.993 | 0.001 | 2.7 | 0.991 |
| V (W_v) | 0.988 | 0.001 | 3.3 | 0.980 |

**HE 对照**

| Layer | Pairwise cosine ↓ | Token variance ↑ | Effective rank ↑ | Centroid cosine ↓ |
|---|---|---|---|---|
| HE Z_HE (RRT output) | 0.940 | 0.059 | 31.8 | 0.898 |
| HE R_HE (routing token) | 0.951 | 0.025 | 5.2 | 0.903 |
| HE Attn-LN | 0.953 | 0.024 | 5.2 | 0.903 |
| HE Q (W_q) | 0.975 | 0.002 | 3.2 | 0.962 |

> 指标：pairwise cosine = slide 内 token 两两 cosine 的 mean（越接近 1 越塌缩）；
> token variance = mean_d Var_t(x[t,d])（越大越分散）；effective rank = 熵形式奇异值有效秩（越大子空间越丰富）；
> centroid cosine = 129 个 slide centroid 的两两 cosine（越低说明 slide 之间越不同）。

### 1.1 逐 seed pairwise cosine（PR）

| Seed | PR Input (X_PR) | PR Embed (E_PR) | PR RRT output (Z_PR) | Route pre-LN | Routing token (R_PR) | Attn-LN | K (W_k) | V (W_v) |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.777 | 0.899 | 0.943 | 0.943 | 0.974 | 0.978 | 0.984 | 0.972 |
| 123 | 0.778 | 0.909 | 0.908 | 0.908 | 0.989 | 0.994 | 0.997 | 0.993 |
| 456 | 0.778 | 0.949 | 0.927 | 0.928 | 0.989 | 0.994 | 0.999 | 0.998 |

### 1.2 逐 seed pairwise cosine（HE 对照）

| Seed | HE Z_HE (RRT output) | HE R_HE (routing token) | HE Attn-LN | HE Q (W_q) |
|---|---|---|---|---|
| 42 | 0.930 | 0.938 | 0.939 | 0.969 |
| 123 | 0.950 | 0.952 | 0.954 | 0.971 |
| 456 | 0.940 | 0.963 | 0.965 | 0.984 |

---

## 2. 塌缩跳变（相邻层 Δ，3-seed mean）

| Step (from → to) | Δ pairwise cosine | Δ effective rank | Δ token variance |
|---|---|---|---|
| PR Input (X_PR) → PR Embed (E_PR) | 0.1417 | -122.3 | -0.0037 |
| PR Embed (E_PR) → PR RRT output (Z_PR) | 0.0067 | -4.7 | 0.0377 |
| PR RRT output (Z_PR) → Route pre-LN | 0.0007 | -0.5 | -0.0001 |
| Route pre-LN → Routing token (R_PR) | 0.0577 | -49.2 | -0.0318 |
| Routing token (R_PR) → Attn-LN | 0.0046 | -0.0 | 0.0001 |
| Attn-LN → K (W_k) | 0.0045 | -0.7 | -0.0060 |
| K (W_k) → V (W_v) | -0.0055 | 0.6 | 0.0004 |

> Δ pairwise cosine 为正 = 该步增大同质性；Δ effective rank 为负 = 该步降低有效秩。

---

## 3. Stage2 routing 额外检查（3-seed）

| Seed | combine weight cosine (slot–slot) | combine weight overlap@topK | routing token cosine (same region) | routing token cosine (diff region) |
|---|---|---|---|---|
| 42 | 0.718 | 0.786 | 0.988 | 0.974 |
| 123 | 0.766 | 0.142 | 0.993 | 0.989 |
| 456 | 0.945 | 0.241 | 0.994 | 0.989 |

> combine weight cosine 高 ⇒ 同一 region 的 3 个 slot 选择几乎相同的 patches（routing 缺少 slot 竞争）。
> routing token cosine：same region = 同一 region 不同 slot 的 token 两两 cosine；diff region = 不同 region 之间。

---

## 4. 逐层补充指标（3-seed mean，mean ± std）

| Layer | pairwise cos (mean) | pairwise cos (median) | pairwise cos (p90) | token var | mean token std | eff. rank | centroid cos | centroid var |
|---|---|---|---|---|---|---|---|---|
| PR Input (X_PR) | 0.777±0.000 | 0.804 | 0.927 | 0.0048 | 0.148 | 180.0 | 0.941 | 0.0011 |
| PR Embed (E_PR) | 0.919±0.022 | 0.938 | 0.981 | 0.0011 | 0.086 | 57.7 | 0.982 | 0.0003 |
| PR RRT output (Z_PR) | 0.926±0.014 | 0.945 | 0.983 | 0.0388 | 0.723 | 53.0 | 0.983 | 0.0084 |
| Route pre-LN | 0.927±0.014 | 0.945 | 0.984 | 0.0387 | 0.724 | 52.5 | 0.983 | 0.0084 |
| Routing token (R_PR) | 0.984±0.007 | 0.979 | 0.998 | 0.0069 | 0.706 | 3.3 | 0.982 | 0.0101 |
| Attn-LN | 0.989±0.008 | 0.993 | 0.998 | 0.0069 | 0.713 | 3.3 | 0.982 | 0.0103 |
| K (W_k) | 0.993±0.007 | 0.996 | 0.999 | 0.0010 | 0.195 | 2.7 | 0.991 | 0.0011 |
| V (W_v) | 0.988±0.011 | 0.992 | 0.997 | 0.0014 | 0.232 | 3.3 | 0.980 | 0.0021 |
| HE Z_HE (RRT output) | 0.940±0.008 | 0.957 | 0.988 | 0.0591 | 0.994 | 31.8 | 0.898 | 0.0879 |
| HE R_HE (routing token) | 0.951±0.010 | 0.956 | 0.992 | 0.0248 | 0.725 | 5.2 | 0.903 | 0.0496 |
| HE Attn-LN | 0.953±0.011 | 0.969 | 0.992 | 0.0239 | 0.693 | 5.2 | 0.903 | 0.0480 |
| HE Q (W_q) | 0.975±0.007 | 0.987 | 0.997 | 0.0020 | 0.200 | 3.2 | 0.962 | 0.0038 |

---

## 5. 五个问题直接回答

**1. PR Stage1 `Z_PR` 是否已经塌缩？**  Z_PR pairwise cosine = 0.926（< 0.95），effective rank = 53.0/512。**否（未达 0.95 阈值），但已明显高于输入 0.777。**注意：塌缩发生在 Stage1 **之前**的嵌入层（0.777→0.919），RRT 本身只 +0.0067。

**2. Stage2 routing 是否显著降低 PR token diversity？**  Δ pairwise cosine (Z_PR→R_PR) = 0.0583，Δ effective rank = 53.0→3.3（~16×）。**是，routing 是 cross-attention 输入塌缩（rank≈3.3）的直接位置**——但 routing 的 slot 竞争机制并未失效（overlap 0.390），塌缩主要继承自上游同质的 Z_PR patch。

**3. `attn_norm / W_k / W_v` 是否进一步造成 collapse？**  R_PR→K = 0.0091，R_PR→V = 0.0036。**否，K/V 只增加微小（甚至反向）变化**，collapse 在 routing 已经完成（0.984）。

**4. PR 是否仍保留明显的 cross-slide slide-specific centroid information？**  X_PR centroid cosine = 0.941，Z_PR = 0.983，R_PR = 0.982（均很高）。**否——连输入层的 centroid 都很接近（0.941），PR 几乎不携带 slide-specific 全局信息**，接近 dataset-level common prior（排除 Case D）。

**5. 下一步最小修改应该落在：**

- **首选 A（PR 表征 / 嵌入投影）**：最大 diversity loss 在 `patch_to_emb`（0.777→0.919，Δ=0.1417），且 Stage1 RRT 几乎不塌缩——瓶颈在上游特征投影导致的 PR patch 同质化。
- **不首选 B（competitive routing）**：routing slot 已选不同 patch（overlap 0.390、combine cos 0.810），token 仍塌缩到 0.992——说明竞争并非瓶颈，改 routing 大概率无效。
- C（K/V projection）与 D（保留 slide-specific global component）**均可排除**：K/V 增collapse 微小，且 centroid 无显著 slide-specific 差异可保留。

---

## 6. 与 HE 对照（判断 collapse 是否 PR 特有）

- HE `Z_HE` pairwise cosine = 0.940（PR Z_PR = 0.926），effective rank = 31.8（PR = 53.0）。HE 的 Stage1 输出甚至**更**同质（cosine 更高、rank 更低），说明「Stage1 输出高度同质」是 RRT 的**共性**，不是 PR 独有。
- HE `R_HE`→`Q_HE`：0.951→0.975（Δ=0.0233），与 PR routing→K 的 +0.0091 同量级。routing 使 HE/PR 都进一步塌缩，但 PR 侧因值 centering 而残差归零（v2 已知结论），HE 侧因作为 identity 主路不受影响。

---

## 7. 判断依据（实测 3-seed mean）

- PR pipeline pairwise cosine：X_PR 0.777 → E_PR 0.919 → Z_PR 0.926 → R_PR 0.984 → K 0.993 → V 0.988
- PR pipeline effective rank：X_PR 180.0 → E_PR 57.7 → Z_PR 53.0 → R_PR 3.3 → K 2.7 → V 3.3
- routing slot：combine cos 0.810，overlap@topK 0.390，routing token cos（same/diff）0.992/0.984
- cross-slide centroid cosine：X_PR 0.941，Z_PR 0.983，R_PR 0.982

> 判定规则（任务 §8）：Case A = Z_PR > 0.95 且有效秩低；Case B = Z_PR 正常但 R_PR > 0.95；Case C = R_PR 有 diversity 但 K/V ≈ 0.99；Case D = within-slide 相似但 cross-slide centroid 差异明显。
> 实测：Z_PR=0.926（<0.95，非 A）；R_PR=0.984（>0.95，形式上 B）；K/V 仅 +0.009/+0.004（非 C）；centroid cosine 0.983（高，非 D）。但「最大塌缩在嵌入层」「slot 竞争已存在」两点使得单纯的 competitive routing（B）不是正确的最小修改——正确的落点是上游 PR 表征（A 方向）。

