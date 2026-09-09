#!/usr/bin/env python3
"""HE Residual Cross v5 — post-training evaluation (§6).

Sections (run with --section, or all by default):
  causal   matched / disable_cross / random cross-slide PR mismatch(5) on the
           dev-test 129 set → per-slide logit-margin + probability + AUC + CE.
           (129 集为 v5 的 held-out 背景集；v5 从未在训练/早停中见它。)
  aux      aux head 的 train(216) / val(54) AUC（含 main AUC 参照）。

核心提醒（§6）：
  * AUC 相同不能推断 logits / attention 不变 —— 因此本节额外保存逐 slide 的
    margin / 概率 / CE 分布，而不是只看 AUC。
  * aux AUC / 熵 / cosine 的改善不能代替主预测收益。

输出落在 results/stage2_he_residual_cross_v5/_eval/。
"""
import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v5"
EVAL_DIR = RESULTS_ROOT / "_eval"
CONDITIONS = ["v5_beta0", "v5_beta01"]
SEEDS = [42, 123, 456]
REPLACEMENT_SEEDS = [0, 1, 2, 3, 4]
DEV_TEST_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_test_labels.csv")


def build_model(cfg):
    from models.mm_rrt_abmil import MM_RRT_ABMIL
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    model = MM_RRT_ABMIL(
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
    return model


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
    return out[0].float().cpu(), out[3] if len(out) > 3 else None


def _metrics(labels, logits):
    """logits [N, C] → prob / margin / auc / ce (per-slide + aggregate)."""
    labels = np.asarray(labels)
    probs = torch.softmax(logits, dim=-1)
    p1 = probs[:, 1].numpy()
    margin = (logits[:, 1] - logits[:, 0]).numpy()
    # cross-entropy per slide: -log p_true
    log_p = torch.log_softmax(logits, dim=-1)
    ce = -log_p[np.arange(len(labels)), labels].numpy()
    auc = float(roc_auc_score(labels, p1))
    return {
        "probs": p1.tolist(),
        "margins": margin.tolist(),
        "ce": ce.tolist(),
        "auc": auc,
        "mean_ce": float(np.mean(ce)),
    }


def _percentiles(a):
    a = np.asarray(a, dtype=float)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)),
            "p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "min": float(np.min(a)), "max": float(np.max(a))}


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


def section_causal(device):
    out = {}
    for cond in CONDITIONS:
        out[cond] = {}
        for s in SEEDS:
            seed_dir = RESULTS_ROOT / cond / f"seed{s}"
            if not (seed_dir / "ckpt" / "best_model.pt").exists():
                print(f"[skip] {cond}/seed{s} no checkpoint", flush=True)
                continue
            cfg, model, ckpt = load_checkpoint_model(seed_dir, device)
            ds = build_dataset(cfg, DEV_TEST_LABEL_FILE)
            slide_ids, labels, he_list, pr_list = _load_features(ds)
            n = len(labels)
            labels_np = np.asarray(labels)

            # matched
            logits_all = []
            for i in range(n):
                lg, _ = _forward(model, he_list[i], pr_list[i], device)
                logits_all.append(lg)
            logits_all = torch.cat(logits_all, dim=0)
            matched = _metrics(labels_np, logits_all)

            # disable_cross
            model.cross_region_mod.disable_cross = True
            logits_dc = []
            for i in range(n):
                lg, _ = _forward(model, he_list[i], pr_list[i], device)
                logits_dc.append(lg)
            logits_dc = torch.cat(logits_dc, dim=0)
            disable = _metrics(labels_np, logits_dc)
            model.cross_region_mod.disable_cross = False

            # random cross-slide PR mismatch (5 replacement seeds)
            mismatch = {}
            for rs in REPLACEMENT_SEEDS:
                rng = np.random.RandomState(rs)
                perm = _derangement(n, rng)
                logits_mm = []
                for i in range(n):
                    lg, _ = _forward(model, he_list[i], pr_list[perm[i]], device)
                    logits_mm.append(lg)
                logits_mm = torch.cat(logits_mm, dim=0)
                mismatch[str(rs)] = _metrics(labels_np, logits_mm)

            # per-slide matched-vs-disable-vs-mismatch margin/prob/CE
            per_slide = {}
            for i, sid in enumerate(slide_ids):
                per_slide[str(sid)] = {
                    "label": int(labels[i]),
                    "matched_prob": matched["probs"][i],
                    "matched_margin": matched["margins"][i],
                    "matched_ce": matched["ce"][i],
                    "disable_prob": disable["probs"][i],
                    "disable_margin": disable["margins"][i],
                    "disable_ce": disable["ce"][i],
                }
            for rs in REPLACEMENT_SEEDS:
                mm = mismatch[str(rs)]
                for i, sid in enumerate(slide_ids):
                    per_slide[str(sid)][f"mismatch{rs}_prob"] = mm["probs"][i]
                    per_slide[str(sid)][f"mismatch{rs}_margin"] = mm["margins"][i]

            out[cond][str(s)] = {
                "matched": {k: matched[k] for k in ("auc", "mean_ce")},
                "disable": {k: disable[k] for k in ("auc", "mean_ce")},
                "mismatch": {str(rs): {k: mismatch[str(rs)][k]
                                       for k in ("auc", "mean_ce")}
                             for rs in REPLACEMENT_SEEDS},
                "matched_minus_disable_auc": matched["auc"] - disable["auc"],
                "matched_minus_mismatch_auc_mean": float(
                    np.mean([mismatch[str(rs)]["auc"] for rs in REPLACEMENT_SEEDS])),
                "margin_pct": {
                    "matched": _percentiles(matched["margins"]),
                    "disable": _percentiles(disable["margins"]),
                },
                "per_slide": per_slide,
            }
            print(f"[causal] {cond}/seed{s}: matched_auc={matched['auc']:.4f} "
                  f"disable={disable['auc']:.4f} "
                  f"Δ={matched['auc']-disable['auc']:+.4f} "
                  f"mm_mean={out[cond][str(s)]['matched_minus_mismatch_auc_mean']:.4f}",
                  flush=True)

    (EVAL_DIR / "causal.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def section_aux(device):
    out = {}
    for cond in CONDITIONS:
        out[cond] = {}
        for s in SEEDS:
            seed_dir = RESULTS_ROOT / cond / f"seed{s}"
            if not (seed_dir / "ckpt" / "best_model.pt").exists():
                print(f"[skip] {cond}/seed{s} no checkpoint", flush=True)
                continue
            cfg, model, ckpt = load_checkpoint_model(seed_dir, device)
            rec = {}
            for split, label_file in [
                ("train", cfg["data"]["train_label_file"]),
                ("val", cfg["data"]["val_label_file"]),
            ]:
                ds = build_dataset(cfg, label_file)
                slide_ids, labels, he_list, pr_list = _load_features(ds)
                n = len(labels)
                labels_np = np.asarray(labels)
                main_logits, aux_logits, aux_valid = [], [], []
                for i in range(n):
                    lg, fs = _forward(model, he_list[i], pr_list[i], device)
                    main_logits.append(lg)
                    if fs is not None and fs.get("pr_value_logits") is not None:
                        aux_logits.append(fs["pr_value_logits"].float().cpu())
                        aux_valid.append(bool(fs["pr_value_has_valid"].item()))
                    else:
                        aux_logits.append(None)
                        aux_valid.append(False)
                main_logits = torch.cat(main_logits, dim=0)
                main_auc = float(roc_auc_score(
                    labels_np, torch.softmax(main_logits, -1)[:, 1].numpy()))

                aux_auc = None
                if any(v for v in aux_valid):
                    valid_idx = [i for i in range(n) if aux_valid[i]]
                    aux_lg = torch.cat([aux_logits[i] for i in valid_idx], dim=0)
                    aux_lb = labels_np[valid_idx]
                    aux_auc = float(roc_auc_score(
                        aux_lb, torch.softmax(aux_lg, -1)[:, 1].numpy()))
                rec[split] = {"main_auc": main_auc, "aux_auc": aux_auc,
                              "n_valid_pr": int(sum(aux_valid))}
            out[cond][str(s)] = rec
            print(f"[aux] {cond}/seed{s}: "
                  f"train main={rec['train']['main_auc']:.4f} aux={rec['train']['aux_auc']} | "
                  f"val main={rec['val']['main_auc']:.4f} aux={rec['val']['aux_auc']}",
                  flush=True)
    (EVAL_DIR / "aux.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", nargs="+", default=["causal", "aux"])
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    sections = {"causal": section_causal, "aux": section_aux}
    for name in args.section:
        if name not in sections:
            print(f"[skip] unknown section {name}", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        sections[name](device)
    print("\nEVAL DONE", flush=True)


if __name__ == "__main__":
    main()
