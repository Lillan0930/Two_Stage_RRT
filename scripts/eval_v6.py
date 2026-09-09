#!/usr/bin/env python3
"""HE Residual Cross v6 — post-training evaluation（§6/§7 的逐样本 + disable_cross 审计）。

产出（results/stage2_he_residual_cross_v6/_eval/）：
  main.json          — 每 condition/seed 的主预测 AUC（129）+ 逐 slide prob/margin/ce。
  disable_cross.json — 每 condition/seed 用**同一 checkpoint** 置 disable_cross=True 后
                       的 HE 路径 AUC（v6 与 v3 均支持 disable_cross → M(Z_HE)）。
  paired.json        — 同 seed 下 v6−v3 的逐 slide margin 变化（§7「per-sample margin
                       change」）+ AUC 差。

要点（§7 接受标准）：**只看主预测 AUC 收益**，不把 disable_cross AUC / attention 形状 /
margin 分布当作替代收益。disable_cross 只用于「cross 贡献方向」的审计。

用法:
  python scripts/eval_v6.py --gpu 7 [--section main disable_cross paired]
"""
import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v6"
EVAL_DIR = RESULTS_ROOT / "_eval"
CONDITIONS = ["v3", "v6"]
SEEDS = [42, 123, 456]
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


def section_main(device):
    out = {}
    for cond in CONDITIONS:
        out[cond] = {}
        for s in SEEDS:
            seed_dir = RESULTS_ROOT / cond / f"seed{s}"
            if not (seed_dir / "ckpt" / "best_model.pt").exists():
                print(f"[skip] {cond}/seed{s} no checkpoint", flush=True)
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
            out[cond][str(s)] = {"auc": m["auc"], "mean_ce": m["mean_ce"],
                                 "per_slide": per_slide}
            print(f"[main] {cond}/seed{s}: auc={m['auc']:.4f} mean_ce={m['mean_ce']:.4f}",
                  flush=True)
    (EVAL_DIR / "main.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def section_disable_cross(device):
    out = {}
    for cond in CONDITIONS:
        out[cond] = {}
        for s in SEEDS:
            seed_dir = RESULTS_ROOT / cond / f"seed{s}"
            if not (seed_dir / "ckpt" / "best_model.pt").exists():
                print(f"[skip] {cond}/seed{s} no checkpoint", flush=True)
                continue
            cfg, model, _ = load_checkpoint_model(seed_dir, device)
            ds = build_dataset(cfg, DEV_TEST_LABEL_FILE)
            _, labels, he_list, pr_list = _load_features(ds)
            logits = _run_model(model, he_list, pr_list, device, disable_cross=True)
            m = _metrics(np.asarray(labels), logits)
            out[cond][str(s)] = {"disable_cross_auc": m["auc"], "mean_ce": m["mean_ce"]}
            print(f"[disable_cross] {cond}/seed{s}: disable_auc={m['auc']:.4f}",
                  flush=True)
    (EVAL_DIR / "disable_cross.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def section_paired(device):
    """同 seed 下 v6−v3 的逐 slide margin / prob 变化（§7）。"""
    main = json.loads((EVAL_DIR / "main.json").read_text())
    out = {}
    for s in SEEDS:
        if "v3" not in main or "v6" not in main:
            continue
        if str(s) not in main["v3"] or str(s) not in main["v6"]:
            continue
        v3 = main["v3"][str(s)]["per_slide"]
        v6 = main["v6"][str(s)]["per_slide"]
        common = sorted(set(v3) & set(v6))
        per_slide = {}
        margin_deltas, prob_deltas = [], []
        for sid in common:
            dm = v6[sid]["margin"] - v3[sid]["margin"]
            dp = v6[sid]["prob"] - v3[sid]["prob"]
            per_slide[sid] = {"label": v3[sid]["label"],
                              "margin_v3": v3[sid]["margin"],
                              "margin_v6": v6[sid]["margin"],
                              "margin_delta": dm,
                              "prob_v3": v3[sid]["prob"],
                              "prob_v6": v6[sid]["prob"],
                              "prob_delta": dp}
            margin_deltas.append(dm)
            prob_deltas.append(dp)
        margin_deltas = np.asarray(margin_deltas)
        prob_deltas = np.asarray(prob_deltas)
        out[str(s)] = {
            "auc_v3": main["v3"][str(s)]["auc"],
            "auc_v6": main["v6"][str(s)]["auc"],
            "auc_delta": main["v6"][str(s)]["auc"] - main["v3"][str(s)]["auc"],
            "margin_delta": {
                "mean": float(np.mean(margin_deltas)), "std": float(np.std(margin_deltas)),
                "mean_abs": float(np.mean(np.abs(margin_deltas))),
                "p10": float(np.percentile(margin_deltas, 10)),
                "p50": float(np.percentile(margin_deltas, 50)),
                "p90": float(np.percentile(margin_deltas, 90)),
            },
            "prob_delta": {
                "mean": float(np.mean(prob_deltas)), "std": float(np.std(prob_deltas)),
                "mean_abs": float(np.mean(np.abs(prob_deltas))),
            },
            "n_slides": len(common),
            "per_slide": per_slide,
        }
        print(f"[paired] seed{s}: auc_delta={out[str(s)]['auc_delta']:+.4f} "
              f"margin_delta mean={out[str(s)]['margin_delta']['mean']:+.4f} "
              f"mean_abs={out[str(s)]['margin_delta']['mean_abs']:.4f}", flush=True)
    (EVAL_DIR / "paired.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", nargs="+", default=["main", "disable_cross", "paired"])
    ap.add_argument("--gpu", type=int, default=7)
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    sections = {"main": section_main, "disable_cross": section_disable_cross,
                "paired": section_paired}
    for name in args.section:
        if name not in sections:
            print(f"[skip] unknown section {name}", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        sections[name](device)
    print("\nEVAL DONE", flush=True)


if __name__ == "__main__":
    main()
