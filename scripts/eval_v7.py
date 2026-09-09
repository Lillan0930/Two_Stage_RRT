#!/usr/bin/env python3
"""HE Residual Cross v7 — post-training evaluation（§6/§7）。

对每个 variant（A/B）× seed 的 best checkpoint，在 129 上记录：
  normal       — 正常前向 M(F)：AUC / CE / 逐 slide prob+margin
  disable_cross— 同 checkpoint 置 disable_cross → M(H)：AUC / CE / 逐 slide
  replaced_pr  — 同 checkpoint 随机跨 slide 替换 PR（5 个替换 seed 的 derangement）：
                 AUC / CE / 逐 slide
并输出**同模型内**的配对：
  residual 是否改善预测   = normal vs disable_cross（同一 checkpoint，逐 slide Δmargin）
  正确配对 PR 是否有帮助  = normal vs replaced_pr（同一 checkpoint）

§7 提醒：cross 贡献只能用**同一 checkpoint**的 normal-vs-disable / normal-vs-replaced
来度量；不同模型（如 v7 vs v3）之间的 margin 差不是 cross 贡献。性能判定只认
A/B 是否超过同期 v3 的主预测 AUC。

输出落在 results/stage2_he_residual_cross_v7/_eval/：
  main.json / disable_cross.json / replaced_pr.json / paired_inmodel.json

用法:
  python scripts/eval_v7.py --gpu 7 [--section main disable_cross replaced_pr]
"""
import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v7"
EVAL_DIR = RESULTS_ROOT / "_eval"
VARIANTS = ["A", "B"]
SEEDS = [42, 123, 456]
REPLACEMENT_SEEDS = [0, 1, 2, 3, 4]
DEV_TEST_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_test_labels.csv")


def build_model(cfg):
    from models.mm_rrt_abmil import MM_RRT_ABMIL
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return MM_RRT_ABMIL(
        num_modalities=len(cfg["data"]["modalities"]),
        modality_list=cfg["data"]["modalities"],
        input_dim=data_cfg["input_dim"],
        mlp_dim=model_cfg.get("mlp_dim", 512),
        num_classes=data_cfg["num_classes"],
        dropout=model_cfg.get("dropout", 0.25),
        region_num=model_cfg.get("region_num", 8),
        n_layers=model_cfg.get("n_layers", 2),
        n_heads=model_cfg.get("n_heads", 8),
        drop_path=model_cfg.get("drop_path", 0.0),
        trans_dropout=model_cfg.get("trans_dropout", 0.1),
        epeg=model_cfg.get("epeg", True),
        epeg_k=model_cfg.get("epeg_k", 15),
        crmsa_k=model_cfg.get("crmsa_k", 3),
        cr_msa=model_cfg.get("cr_msa", True),
        all_shortcut=model_cfg.get("all_shortcut", False),
        crmsa_heads=model_cfg.get("crmsa_heads", 8),
        crmsa_mlp=model_cfg.get("crmsa_mlp", False),
        fusion_type=model_cfg.get("fusion_type", "self_attention"),
        fusion_stage=model_cfg.get("fusion_stage", "middle"),
        fusion_kwargs={},
        stage2_type=model_cfg.get("stage2_type", "staining_msa"),
        use_gated_fusion=model_cfg.get("use_gated_fusion", False),
        use_per_layer_fusion=model_cfg.get("use_per_layer_fusion", True),
        use_logit_fusion=model_cfg.get("use_logit_fusion", False),
        fixed_beta=model_cfg.get("fixed_beta", None),
        fixed_alpha=model_cfg.get("fixed_alpha", None),
        use_consistency_fusion=model_cfg.get("use_consistency_fusion", False),
        use_arlc_fusion=model_cfg.get("use_arlc_fusion", False),
        use_correction_only=model_cfg.get("use_correction_only", False),
        use_logit_attn=model_cfg.get("use_logit_attn", False),
        pretrained_he_ckpt=model_cfg.get("pretrained_he_ckpt", None),
        alpha_mode=model_cfg.get("alpha_mode", "feature"),
        use_lowrank_correction=model_cfg.get("use_lowrank_correction", False),
        use_srp_fusion=model_cfg.get("use_srp_fusion", False),
        srp_beta=model_cfg.get("srp_beta", 0.1),
        srp_mode=model_cfg.get("srp_mode", "residual"),
        use_shared_rrt=model_cfg.get("use_shared_rrt", False),
        shared_rrt_alpha=model_cfg.get("shared_rrt_alpha", 0.02),
        use_partial_align=model_cfg.get("use_partial_align", False),
        use_mclc=model_cfg.get("use_mclc", False),
        freeze_mclc=model_cfg.get("freeze_mclc", False),
        he_only=model_cfg.get("he_only", False),
        encoder_cfg=model_cfg.get("encoder_cfg", None),
        stage2_cfg=model_cfg.get("stage2_cfg", None),
        mil_type=model_cfg.get("mil_type", "abmil"),
        abmil_hidden_dim=model_cfg.get("abmil_hidden_dim", 128),
        use_gated=model_cfg.get("use_gated", False),
    )


def load_checkpoint_model(seed_dir, device):
    cfg = json.loads((seed_dir / "config.json").read_text())
    ckpt = torch.load(seed_dir / "ckpt" / "best_model.pt",
                      map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return cfg, model, ckpt


def build_dataset(cfg, label_file):
    from train import build_feature_dirs
    from data.c16_multimodal_dataset import C16MultimodalDataset
    data_cfg = cfg["data"]
    feature_dirs = build_feature_dirs(
        data_cfg["feature_base_dir"], data_cfg["modalities"],
        data_cfg.get("dir_mapping", None))
    ds = C16MultimodalDataset(
        feature_dirs=feature_dirs, label_file=label_file,
        max_patches=data_cfg.get("max_patches", 2500), preload=False, verbose=False,
        sampling=data_cfg.get("sampling", "random"),
        sample_seed=data_cfg.get("sample_seed", 0), per_epoch=False)
    return ds


def _load_features(ds):
    n = len(ds)
    labels = [smp["label"] for smp in ds.samples]
    slide_ids = [smp["slide_id"] for smp in ds.samples]
    he_list = [ds[i]["features"]["HE"] for i in range(n)]
    pr_list = [ds[i]["features"]["PR"] for i in range(n)]
    return slide_ids, labels, he_list, pr_list


@torch.inference_mode()
def _forward(model, he, pr, device):
    he = he.to(device).unsqueeze(0)
    pr = pr.to(device).unsqueeze(0)
    out = model([he, pr])
    return out[0].float().cpu()


def _metrics(labels, logits):
    labels = np.asarray(labels)
    probs = torch.softmax(logits, dim=-1)
    p1 = probs[:, 1].numpy()
    margin = (logits[:, 1] - logits[:, 0]).numpy()
    log_p = torch.log_softmax(logits, dim=-1)
    ce = -log_p[np.arange(len(labels)), labels].numpy()
    auc = float(roc_auc_score(labels, p1))
    return {"probs": p1.tolist(), "margins": margin.tolist(), "ce": ce.tolist(),
            "auc": auc, "mean_ce": float(np.mean(ce))}


def _run_model(model, he_list, pr_list, device, disable_cross=False):
    if disable_cross:
        model.cross_region_mod.disable_cross = True
    n = len(he_list)
    logits = []
    for i in range(n):
        logits.append(_forward(model, he_list[i], pr_list[i], device))
    if disable_cross:
        model.cross_region_mod.disable_cross = False
    return torch.cat(logits, dim=0)


def _derangement(n, rng):
    perm = rng.permutation(n)
    fixed = [i for i in range(n) if perm[i] == i]
    while fixed:
        for i in fixed:
            j = int(rng.randint(0, n))
            if j != i:
                perm[i], perm[j] = perm[j], perm[i]
        fixed = [i for i in range(n) if perm[i] == i]
    return perm


def _percentiles(a):
    a = np.asarray(a, dtype=float)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)),
            "p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "min": float(np.min(a)), "max": float(np.max(a))}


def section_main(device):
    out = {}
    for v in VARIANTS:
        out[v] = {}
        for s in SEEDS:
            seed_dir = RESULTS_ROOT / v / f"seed{s}"
            if not (seed_dir / "ckpt" / "best_model.pt").exists():
                print(f"[skip] {v}/seed{s} no checkpoint", flush=True)
                continue
            cfg, model, _ = load_checkpoint_model(seed_dir, device)
            ds = build_dataset(cfg, DEV_TEST_LABEL_FILE)
            slide_ids, labels, he_list, pr_list = _load_features(ds)
            logits = _run_model(model, he_list, pr_list, device)
            m = _metrics(np.asarray(labels), logits)
            per_slide = {str(sid): {"label": int(labels[i]),
                                    "prob": m["probs"][i],
                                    "margin": m["margins"][i],
                                    "ce": m["ce"][i]}
                         for i, sid in enumerate(slide_ids)}
            out[v][str(s)] = {"auc": m["auc"], "mean_ce": m["mean_ce"],
                              "per_slide": per_slide}
            print(f"[main] v7{v}/seed{s}: auc={m['auc']:.4f} mean_ce={m['mean_ce']:.4f}",
                  flush=True)
    (EVAL_DIR / "main.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def section_disable_cross(device):
    out = {}
    for v in VARIANTS:
        out[v] = {}
        for s in SEEDS:
            seed_dir = RESULTS_ROOT / v / f"seed{s}"
            if not (seed_dir / "ckpt" / "best_model.pt").exists():
                print(f"[skip] {v}/seed{s} no checkpoint", flush=True)
                continue
            cfg, model, _ = load_checkpoint_model(seed_dir, device)
            ds = build_dataset(cfg, DEV_TEST_LABEL_FILE)
            slide_ids, labels, he_list, pr_list = _load_features(ds)
            logits = _run_model(model, he_list, pr_list, device, disable_cross=True)
            m = _metrics(np.asarray(labels), logits)
            per_slide = {str(sid): {"label": int(labels[i]),
                                    "prob": m["probs"][i],
                                    "margin": m["margins"][i]}
                         for i, sid in enumerate(slide_ids)}
            out[v][str(s)] = {"disable_cross_auc": m["auc"], "mean_ce": m["mean_ce"],
                              "per_slide": per_slide}
            print(f"[disable_cross] v7{v}/seed{s}: disable_auc={m['auc']:.4f} "
                  f"mean_ce={m['mean_ce']:.4f}", flush=True)
    (EVAL_DIR / "disable_cross.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def section_replaced_pr(device):
    out = {}
    for v in VARIANTS:
        out[v] = {}
        for s in SEEDS:
            seed_dir = RESULTS_ROOT / v / f"seed{s}"
            if not (seed_dir / "ckpt" / "best_model.pt").exists():
                print(f"[skip] {v}/seed{s} no checkpoint", flush=True)
                continue
            cfg, model, _ = load_checkpoint_model(seed_dir, device)
            ds = build_dataset(cfg, DEV_TEST_LABEL_FILE)
            slide_ids, labels, he_list, pr_list = _load_features(ds)
            n = len(labels)
            labels_np = np.asarray(labels)
            out[v][str(s)] = {}
            for rs in REPLACEMENT_SEEDS:
                rng = np.random.RandomState(rs)
                perm = _derangement(n, rng)
                logits = []
                for i in range(n):
                    logits.append(_forward(model, he_list[i], pr_list[perm[i]], device))
                logits = torch.cat(logits, dim=0)
                m = _metrics(labels_np, logits)
                per_slide = {str(sid): {"label": int(labels[i]),
                                        "prob": m["probs"][i],
                                        "margin": m["margins"][i]}
                             for i, sid in enumerate(slide_ids)}
                out[v][str(s)][str(rs)] = {"auc": m["auc"], "mean_ce": m["mean_ce"],
                                           "per_slide": per_slide}
            aucs = [out[v][str(s)][str(rs)]["auc"] for rs in REPLACEMENT_SEEDS]
            print(f"[replaced_pr] v7{v}/seed{s}: mean_auc={np.mean(aucs):.4f} "
                  f"(per-seed {[f'{a:.4f}' for a in aucs]})", flush=True)
    (EVAL_DIR / "replaced_pr.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def section_paired_inmodel(device):
    """同一 checkpoint 内的配对：normal vs disable_cross（残差贡献）、
    normal vs replaced_pr（正确配对 PR 的帮助）。均不跨模型比较。"""
    main = json.loads((EVAL_DIR / "main.json").read_text())
    dc = json.loads((EVAL_DIR / "disable_cross.json").read_text())
    rp = json.loads((EVAL_DIR / "replaced_pr.json").read_text())
    out = {}
    for v in VARIANTS:
        out[v] = {}
        for s in SEEDS:
            if str(s) not in main.get(v, {}) or str(s) not in dc.get(v, {}) \
                    or str(s) not in rp.get(v, {}):
                continue
            mv, dv = main[v][str(s)], dc[v][str(s)]
            common = sorted(set(mv["per_slide"]) & set(dv["per_slide"]))
            # residual contribution: normal - disable_cross (per-slide)
            res_margins, res_probs = [], []
            for sid in common:
                res_margins.append(mv["per_slide"][sid]["margin"] - dv["per_slide"][sid]["margin"])
                res_probs.append(mv["per_slide"][sid]["prob"] - dv["per_slide"][sid]["prob"])
            # paired-PR help: normal - mean(replaced_pr) (per-slide, averaged over 5 replacements)
            pr_margins, pr_probs = [], []
            for sid in common:
                dm = [mv["per_slide"][sid]["margin"] - rp[v][str(s)][str(rs)]["per_slide"][sid]["margin"]
                      for rs in REPLACEMENT_SEEDS]
                dp = [mv["per_slide"][sid]["prob"] - rp[v][str(s)][str(rs)]["per_slide"][sid]["prob"]
                      for rs in REPLACEMENT_SEEDS]
                pr_margins.append(float(np.mean(dm)))
                pr_probs.append(float(np.mean(dp)))
            rp_aucs = [rp[v][str(s)][str(rs)]["auc"] for rs in REPLACEMENT_SEEDS]
            out[v][str(s)] = {
                "auc_normal": mv["auc"],
                "auc_disable_cross": dv["disable_cross_auc"],
                "auc_replaced_pr_mean": float(np.mean(rp_aucs)),
                "residual_auc_gain": mv["auc"] - dv["disable_cross_auc"],
                "paired_pr_auc_gain": mv["auc"] - float(np.mean(rp_aucs)),
                "residual_margin_delta": _percentiles(res_margins),
                "residual_prob_delta": _percentiles(res_probs),
                "paired_pr_margin_delta": _percentiles(pr_margins),
                "paired_pr_prob_delta": _percentiles(pr_probs),
                "n_slides": len(common),
            }
            print(f"[paired] v7{v}/seed{s}: auc_normal={mv['auc']:.4f} "
                  f"residual_gain(normal-disable)={out[v][str(s)]['residual_auc_gain']:+.4f} "
                  f"paired_pr_gain(normal-replaced)={out[v][str(s)]['paired_pr_auc_gain']:+.4f}",
                  flush=True)
    (EVAL_DIR / "paired_inmodel.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", nargs="+",
                    default=["main", "disable_cross", "replaced_pr", "paired_inmodel"])
    ap.add_argument("--gpu", type=int, default=7)
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    sections = {"main": section_main, "disable_cross": section_disable_cross,
                "replaced_pr": section_replaced_pr, "paired_inmodel": section_paired_inmodel}
    for name in args.section:
        if name not in sections:
            print(f"[skip] unknown section {name}", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        sections[name](device)
    print("\nEVAL DONE", flush=True)


if __name__ == "__main__":
    main()
