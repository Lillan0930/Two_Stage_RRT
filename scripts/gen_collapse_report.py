#!/usr/bin/env python3
"""Generate PR_REPRESENTATION_COLLAPSE_REPORT.md from collapse_localization.json."""
import json
from pathlib import Path

import numpy as np

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
JSON = PROJECT / "results" / "stage2_he_residual_cross_v2" / "_eval" / "collapse_localization.json"
OUT = PROJECT / "PR_REPRESENTATION_COLLAPSE_REPORT.md"
SEEDS = [42, 123, 456]

PR_LAYERS = [
    ("X_PR",           "PR Input (X_PR)",        "L0"),
    ("E_PR",           "PR Embed (E_PR)",        "L1"),
    ("Z_PR",           "PR RRT output (Z_PR)",   "L2"),
    ("route_preln_pr", "Route pre-LN",           "L3"),
    ("R_PR",           "Routing token (R_PR)",   "L4"),
    ("attn_ln_pr",     "Attn-LN",                "L5"),
    ("K_PR",           "K (W_k)",                "L6"),
    ("V_PR",           "V (W_v)",                "L7"),
]
HE_LAYERS = [
    ("Z_HE",        "HE Z_HE (RRT output)",    "H2"),
    ("R_HE",        "HE R_HE (routing token)", "H4"),
    ("attn_ln_he",  "HE Attn-LN",              "H5"),
    ("Q_HE",        "HE Q (W_q)",              "HQ"),
]


def load():
    return json.loads(JSON.read_text())


def mseeds(results, key, metric):
    return float(np.mean([results[str(s)]["layers"][key][metric]["mean"] for s in SEEDS]))


def mseeds_cc(results, key):
    return float(np.mean([results[str(s)]["layers"][key]["cross_slide_centroid_cosine"]
                          for s in SEEDS]))


def mseeds_slot(results, key):
    return float(np.mean([results[str(s)]["routing_slot"][key]["mean"] for s in SEEDS]))


def fmt(v, nd=3):
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def build_core_table(results, layer_specs):
    rows = ["| Layer | Pairwise cosine ↓ | Token variance ↑ | "
            "Effective rank ↑ | Centroid cosine ↓ |",
            "|---|---|---|---|---|"]
    for key, name, _t in layer_specs:
        rows.append(f"| {name} | {fmt(mseeds(results, key, 'pairwise_cosine_mean'))} "
                    f"| {fmt(mseeds(results, key, 'token_variance'))} "
                    f"| {fmt(mseeds(results, key, 'effective_rank'), 1)} "
                    f"| {fmt(mseeds_cc(results, key))} |")
    return "\n".join(rows)


def per_seed_table(results, layer_specs):
    rows = ["| Seed | " + " | ".join(n for _k, n, _t in layer_specs) + " |",
            "|---|" + "|".join("---" for _ in layer_specs) + "|"]
    for s in SEEDS:
        cells = [fmt(results[str(s)]["layers"][k]["pairwise_cosine_mean"]["mean"])
                 for k, _n, _t in layer_specs]
        rows.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def collapse_jumps(results):
    rows = ["| Step (from → to) | Δ pairwise cosine | Δ effective rank | Δ token variance |",
            "|---|---|---|---|"]
    for (k1, n1, _), (k2, n2, _) in zip(PR_LAYERS, PR_LAYERS[1:]):
        d_pc = mseeds(results, k2, "pairwise_cosine_mean") - mseeds(results, k1, "pairwise_cosine_mean")
        d_er = mseeds(results, k2, "effective_rank") - mseeds(results, k1, "effective_rank")
        d_tv = mseeds(results, k2, "token_variance") - mseeds(results, k1, "token_variance")
        rows.append(f"| {n1} → {n2} | {fmt(d_pc, 4)} | {fmt(d_er, 1)} | {fmt(d_tv, 4)} |")
    return "\n".join(rows)


def routing_slot_table(results):
    keys = [("combine_weight_cos_same_region", "combine weight cosine (slot–slot)"),
            ("combine_weight_overlap_topK", "combine weight overlap@topK"),
            ("routing_token_cos_same_region", "routing token cosine (same region)"),
            ("routing_token_cos_diff_region", "routing token cosine (diff region)")]
    rows = ["| Seed | " + " | ".join(n for _k, n in keys) + " |",
            "|---|" + "|".join("---" for _ in keys) + "|"]
    for s in SEEDS:
        cells = [fmt(results[str(s)]["routing_slot"][k]["mean"]) for k, _n in keys]
        rows.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    results = load()

    # key 3-seed numbers
    x_pc = mseeds(results, "X_PR", "pairwise_cosine_mean")
    e_pc = mseeds(results, "E_PR", "pairwise_cosine_mean")
    z_pc = mseeds(results, "Z_PR", "pairwise_cosine_mean")
    r_pc = mseeds(results, "R_PR", "pairwise_cosine_mean")
    k_pc = mseeds(results, "K_PR", "pairwise_cosine_mean")
    v_pc = mseeds(results, "V_PR", "pairwise_cosine_mean")
    x_er = mseeds(results, "X_PR", "effective_rank")
    e_er = mseeds(results, "E_PR", "effective_rank")
    z_er = mseeds(results, "Z_PR", "effective_rank")
    r_er = mseeds(results, "R_PR", "effective_rank")
    k_er = mseeds(results, "K_PR", "effective_rank")
    v_er = mseeds(results, "V_PR", "effective_rank")
    z_cc = mseeds_cc(results, "Z_PR")
    r_cc = mseeds_cc(results, "R_PR")
    x_cc = mseeds_cc(results, "X_PR")
    cw_cos = mseeds_slot(results, "combine_weight_cos_same_region")
    cw_over = mseeds_slot(results, "combine_weight_overlap_topK")
    rt_same = mseeds_slot(results, "routing_token_cos_same_region")
    rt_diff = mseeds_slot(results, "routing_token_cos_diff_region")

    # HE control
    zhe_pc = mseeds(results, "Z_HE", "pairwise_cosine_mean")
    rhe_pc = mseeds(results, "R_HE", "pairwise_cosine_mean")
    qhe_pc = mseeds(results, "Q_HE", "pairwise_cosine_mean")
    zhe_er = mseeds(results, "Z_HE", "effective_rank")
    rhe_er = mseeds(results, "R_HE", "effective_rank")
    qhe_er = mseeds(results, "Q_HE", "effective_rank")

    L = []
    L.append("# PR Representation Collapse Localization 报告")
    L.append("")
    L.append("**日期**: 2026-09-07")
    L.append("**对象**: `he_residual_cross_v2` 已训练 checkpoint（seed 42/123/456），**不重训 / 不改模型 / 不调参**")
    L.append("**协议**: Train / Val-as-Test（129 WSIs）")
    L.append("**产物**: `results/stage2_he_residual_cross_v2/_eval/collapse_localization.json`")
    L.append("**脚本**: `scripts/diag_collapse_localization.py`（逐层截取+指标） / `scripts/gen_collapse_report.py`（本报告）")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 0. 结论")
    L.append("")
    L.append("**PR token diversity 在**两处**塌缩，且都不在 PR-RRT（Stage1 编码器）本身：**")
    L.append("")
    L.append(f"1. **幅度最大的塌缩 = `patch_to_emb` 嵌入层（L0→L1）**：pairwise cosine {fmt(x_pc)} → {fmt(e_pc)}"
             f"（**Δ={fmt(e_pc - x_pc, 4)}**，全流程最大），effective rank {fmt(x_er, 1)} → {fmt(e_er, 1)}。"
             f"这一步发生在 RRT **之前**，是 768→512 线性投影 + GELU 造成的特征同质化。")
    L.append("")
    L.append(f"2. **对 cross-attention 致命的一处 = Stage2 routing（L3→L4）**：pairwise cosine {fmt(z_pc)} → {fmt(r_pc)}"
             f"（Δ={fmt(r_pc - z_pc, 4)}），**effective rank {fmt(z_er, 1)} → {fmt(r_er, 1)}（~16× 塌缩）**。"
             f"这一步把 48 个 routing token 压进 ~{fmt(r_er, 1)} 维子空间，直接导致 cross-attention 的 K/V 无可选择的信息。")
    L.append("")
    L.append(f"**Stage1 RRT（L1→L2）几乎不塌缩**（Δpc={fmt(z_pc - e_pc, 4)}，Δeffrank={fmt(z_er - e_er, 1)}），"
              f"**K/V（L5→L7）只增加微小塌缩**（K {fmt(k_pc)}、V {fmt(v_pc)}）。")
    L.append("")
    L.append(f"**关键反向证据（排除 competitive routing 作为首选）**：routing 的 combine weight 槽间 cosine 仅 "
             f"{fmt(cw_cos)}、top-10 patch overlap 仅 {fmt(cw_over)} —— **不同 slot 已经在选不同的 patch**，"
             f"但 routing token 仍塌缩到 {fmt(rt_same)}（同 region）/ {fmt(rt_diff)}（跨 region）。"
             f"这说明塌缩**不是** routing 缺少 slot 竞争，而是被聚合的 Z_PR patch 本身已高度同质（{fmt(z_pc)}），"
             f"其根子在上游嵌入层。")
    L.append("")
    L.append(f"**最终判断（对应任务 §8）**：既非纯粹的 Case A（Z_PR={fmt(z_pc)} 未到 0.95、且 RRT 本身无责），"
              f"也非可被 competitive routing 解决的 Case B（slot 竞争已存在）。是 **上游 PR 表征（嵌入投影）主导 + "
              f"routing 放大** 的两步塌缩。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 核心表（3-seed mean，val-as-test）")
    L.append("")
    L.append("**PR pipeline**")
    L.append("")
    L.append(build_core_table(results, PR_LAYERS))
    L.append("")
    L.append("**HE 对照**")
    L.append("")
    L.append(build_core_table(results, HE_LAYERS))
    L.append("")
    L.append("> 指标：pairwise cosine = slide 内 token 两两 cosine 的 mean（越接近 1 越塌缩）；")
    L.append("> token variance = mean_d Var_t(x[t,d])（越大越分散）；effective rank = 熵形式奇异值有效秩（越大子空间越丰富）；")
    L.append("> centroid cosine = 129 个 slide centroid 的两两 cosine（越低说明 slide 之间越不同）。")
    L.append("")
    L.append("### 1.1 逐 seed pairwise cosine（PR）")
    L.append("")
    L.append(per_seed_table(results, PR_LAYERS))
    L.append("")
    L.append("### 1.2 逐 seed pairwise cosine（HE 对照）")
    L.append("")
    L.append(per_seed_table(results, HE_LAYERS))
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. 塌缩跳变（相邻层 Δ，3-seed mean）")
    L.append("")
    L.append(collapse_jumps(results))
    L.append("")
    L.append("> Δ pairwise cosine 为正 = 该步增大同质性；Δ effective rank 为负 = 该步降低有效秩。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. Stage2 routing 额外检查（3-seed）")
    L.append("")
    L.append(routing_slot_table(results))
    L.append("")
    L.append("> combine weight cosine 高 ⇒ 同一 region 的 3 个 slot 选择几乎相同的 patches（routing 缺少 slot 竞争）。")
    L.append("> routing token cosine：same region = 同一 region 不同 slot 的 token 两两 cosine；diff region = 不同 region 之间。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. 逐层补充指标（3-seed mean，mean ± std）")
    L.append("")
    L.append("| Layer | pairwise cos (mean) | pairwise cos (median) | pairwise cos (p90) | token var | mean token std | eff. rank | centroid cos | centroid var |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key, name, _t in PR_LAYERS + HE_LAYERS:
        d = {}
        for metric in ["pairwise_cosine_mean", "pairwise_cosine_median",
                       "pairwise_cosine_p90", "token_variance", "mean_token_std",
                       "effective_rank"]:
            vals = [results[str(s)]["layers"][key][metric]["mean"] for s in SEEDS]
            d[metric] = (float(np.mean(vals)), float(np.std(vals)))
        cc = float(np.mean([results[str(s)]["layers"][key]["cross_slide_centroid_cosine"] for s in SEEDS]))
        cv = float(np.mean([results[str(s)]["layers"][key]["cross_slide_centroid_variance"] for s in SEEDS]))
        L.append(f"| {name} | {fmt(d['pairwise_cosine_mean'][0])}±{fmt(d['pairwise_cosine_mean'][1], 3)} "
                 f"| {fmt(d['pairwise_cosine_median'][0])} | {fmt(d['pairwise_cosine_p90'][0])} "
                 f"| {fmt(d['token_variance'][0], 4)} | {fmt(d['mean_token_std'][0])} "
                 f"| {fmt(d['effective_rank'][0], 1)} | {fmt(cc)} | {fmt(cv, 4)} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. 五个问题直接回答")
    L.append("")
    L.append(f"**1. PR Stage1 `Z_PR` 是否已经塌缩？**  Z_PR pairwise cosine = {fmt(z_pc)}（< 0.95），"
             f"effective rank = {fmt(z_er, 1)}/512。**否（未达 0.95 阈值），但已明显高于输入 {fmt(x_pc)}。**"
             f"注意：塌缩发生在 Stage1 **之前**的嵌入层（{fmt(x_pc)}→{fmt(e_pc)}），RRT 本身只 +{fmt(z_pc - e_pc, 4)}。")
    L.append("")
    L.append(f"**2. Stage2 routing 是否显著降低 PR token diversity？**  Δ pairwise cosine (Z_PR→R_PR) = "
             f"{fmt(r_pc - z_pc, 4)}，Δ effective rank = {fmt(z_er, 1)}→{fmt(r_er, 1)}（~16×）。"
             f"**是，routing 是 cross-attention 输入塌缩（rank≈{fmt(r_er, 1)}）的直接位置**——但 routing 的 slot 竞争机制"
             f"并未失效（overlap {fmt(cw_over)}），塌缩主要继承自上游同质的 Z_PR patch。")
    L.append("")
    L.append(f"**3. `attn_norm / W_k / W_v` 是否进一步造成 collapse？**  R_PR→K = "
             f"{fmt(k_pc - r_pc, 4)}，R_PR→V = {fmt(v_pc - r_pc, 4)}。**否，K/V 只增加微小（甚至反向）变化**，"
             f"collapse 在 routing 已经完成（{fmt(r_pc)}）。")
    L.append("")
    L.append(f"**4. PR 是否仍保留明显的 cross-slide slide-specific centroid information？**  "
             f"X_PR centroid cosine = {fmt(x_cc)}，Z_PR = {fmt(z_cc)}，R_PR = {fmt(r_cc)}（均很高）。"
             f"**否——连输入层的 centroid 都很接近（{fmt(x_cc)}），PR 几乎不携带 slide-specific 全局信息**，"
             f"接近 dataset-level common prior（排除 Case D）。")
    L.append("")
    L.append("**5. 下一步最小修改应该落在：**")
    L.append("")
    L.append(f"- **首选 A（PR 表征 / 嵌入投影）**：最大 diversity loss 在 `patch_to_emb`（{fmt(x_pc)}→{fmt(e_pc)}，"
             f"Δ={fmt(e_pc - x_pc, 4)}），且 Stage1 RRT 几乎不塌缩——瓶颈在上游特征投影导致的 PR patch 同质化。")
    L.append(f"- **不首选 B（competitive routing）**：routing slot 已选不同 patch（overlap {fmt(cw_over)}、"
             f"combine cos {fmt(cw_cos)}），token 仍塌缩到 {fmt(rt_same)}——说明竞争并非瓶颈，改 routing 大概率无效。")
    L.append("- C（K/V projection）与 D（保留 slide-specific global component）**均可排除**：K/V 增collapse 微小，"
             "且 centroid 无显著 slide-specific 差异可保留。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. 与 HE 对照（判断 collapse 是否 PR 特有）")
    L.append("")
    L.append(f"- HE `Z_HE` pairwise cosine = {fmt(zhe_pc)}（PR Z_PR = {fmt(z_pc)}），effective rank = {fmt(zhe_er, 1)}"
             f"（PR = {fmt(z_er, 1)}）。HE 的 Stage1 输出甚至**更**同质（cosine 更高、rank 更低），"
             f"说明「Stage1 输出高度同质」是 RRT 的**共性**，不是 PR 独有。")
    L.append(f"- HE `R_HE`→`Q_HE`：{fmt(rhe_pc)}→{fmt(qhe_pc)}（Δ={fmt(qhe_pc - rhe_pc, 4)}），"
             f"与 PR routing→K 的 +{fmt(k_pc - r_pc, 4)} 同量级。routing 使 HE/PR 都进一步塌缩，"
             f"但 PR 侧因值 centering 而残差归零（v2 已知结论），HE 侧因作为 identity 主路不受影响。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. 判断依据（实测 3-seed mean）")
    L.append("")
    L.append(f"- PR pipeline pairwise cosine：X_PR {fmt(x_pc)} → E_PR {fmt(e_pc)} → Z_PR {fmt(z_pc)} "
             f"→ R_PR {fmt(r_pc)} → K {fmt(k_pc)} → V {fmt(v_pc)}")
    L.append(f"- PR pipeline effective rank：X_PR {fmt(x_er, 1)} → E_PR {fmt(e_er, 1)} → Z_PR {fmt(z_er, 1)} "
             f"→ R_PR {fmt(r_er, 1)} → K {fmt(k_er, 1)} → V {fmt(v_er, 1)}")
    L.append(f"- routing slot：combine cos {fmt(cw_cos)}，overlap@topK {fmt(cw_over)}，"
             f"routing token cos（same/diff）{fmt(rt_same)}/{fmt(rt_diff)}")
    L.append(f"- cross-slide centroid cosine：X_PR {fmt(x_cc)}，Z_PR {fmt(z_cc)}，R_PR {fmt(r_cc)}")
    L.append("")
    L.append("> 判定规则（任务 §8）：Case A = Z_PR > 0.95 且有效秩低；Case B = Z_PR 正常但 R_PR > 0.95；"
              "Case C = R_PR 有 diversity 但 K/V ≈ 0.99；Case D = within-slide 相似但 cross-slide centroid 差异明显。")
    L.append("> 实测：Z_PR={:.3f}（<0.95，非 A）；R_PR={:.3f}（>0.95，形式上 B）；K/V 仅 +{:.3f}/+{:.3f}（非 C）；"
              "centroid cosine {:.3f}（高，非 D）。但「最大塌缩在嵌入层」「slot 竞争已存在」两点使得单纯的 competitive routing（B）"
              "不是正确的最小修改——正确的落点是上游 PR 表征（A 方向）。".format(z_pc, r_pc, k_pc - r_pc, v_pc - r_pc, z_cc))
    L.append("")

    OUT.write_text("\n".join(L) + "\n")
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
