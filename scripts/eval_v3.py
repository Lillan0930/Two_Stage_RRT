#!/usr/bin/env python3
"""HE Residual Cross v3 — post-training evaluation.

Sections (run with --section, or all by default):
  repro        re-run pure inference on 3 v3 checkpoints, compare saved AUC
  causal       matched / random-mismatch(5) / within-class-mismatch(5) / disable
  diagnostics  7 per-slide diagnostics: entropy_norm, score_std, value_diversity,
               selective_ratio, mu_norm (||μ_PR||), r_slide, r_region, plus rho
               (= ||0.1·Δ||/||z_he||).
  summary      main result table (v3 vs HE-only/v1/v2 + causal) → _eval/summary.json

Outputs land in results/stage2_he_residual_cross_v3/_eval/.
"""
import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v3"
EVAL_DIR = RESULTS_ROOT / "_eval"
V2_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v2"
V1_ROOT = PROJECT / "results" / "stage2_he_residual_cross_val_as_test"
SEEDS = [42, 123, 456]
REPLACEMENT_SEEDS = [0, 1, 2, 3, 4]


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


def build_val_dataset(cfg):
    from train import build_feature_dirs
    from data.c16_multimodal_dataset import C16MultimodalDataset
    data_cfg = cfg["data"]
    feature_dirs = build_feature_dirs(
        data_cfg["feature_base_dir"], data_cfg["modalities"],
        data_cfg.get("dir_mapping", None))
    ds = C16MultimodalDataset(
        feature_dirs=feature_dirs, label_file=data_cfg["val_label_file"],
        max_patches=data_cfg.get("max_patches", 2500), preload=False, verbose=False,
        sampling=data_cfg.get("sampling", "random"),
        sample_seed=data_cfg.get("sample_seed", 0), per_epoch=False)
    return ds


@torch.inference_mode()
def run_inference(model, he_list, pr_list, device):
    probs = []
    for i in range(len(he_list)):
        he = he_list[i].to(device).unsqueeze(0)
        if pr_list is None:
            out = model([he])
        else:
            pr = pr_list[i].to(device).unsqueeze(0)
            out = model([he, pr])
        logits = out[0]
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def auc_from_probs(labels, probs):
    return float(roc_auc_score(labels, probs))


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


def within_class_derangement(labels, rng):
    labels = np.asarray(labels)
    n = len(labels)
    perm = np.arange(n)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) <= 1:
            continue
        sub = _derangement(len(idx), rng)
        perm[idx] = idx[sub]
    return perm


@torch.inference_mode()
def compute_v3_diag(model, he, pr):
    m = model.cross_region_mod
    z_he = model.rrt_he(model.patch_to_emb[0](model.dp(he)))
    z_ihc = model.rrt_ihc(model.patch_to_emb[1](model.dp(pr)))
    if z_he.dim() == 2:
        z_he = z_he.unsqueeze(0)
    if z_ihc.dim() == 2:
        z_ihc = z_ihc.unsqueeze(0)
    d = m.diagnose(z_he, z_ihc)
    return d


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def section_repro(device):
    out = {}
    for s in SEEDS:
        seed_dir = RESULTS_ROOT / "he_residual_cross_v3" / f"seed{s}"
        cfg, model, ckpt = load_checkpoint_model(seed_dir, device)
        ds = build_val_dataset(cfg)
        n = len(ds)
        labels = [smp["label"] for smp in ds.samples]
        he_list = [ds[i]["features"]["HE"] for i in range(n)]
        pr_list = [ds[i]["features"]["PR"] for i in range(n)]
        probs = run_inference(model, he_list, pr_list, device)
        auc = auc_from_probs(labels, probs)
        saved = json.loads((seed_dir / "result.json").read_text())["auc"]
        out[s] = {"saved_auc": saved, "recomputed_auc": auc,
                  "diff": auc - saved}
    (EVAL_DIR / "repro.json").write_text(json.dumps(out, indent=2) + "\n")
    for s in SEEDS:
        print(f"seed{s}: saved={out[s]['saved_auc']:.6f} recomputed={out[s]['recomputed_auc']:.6f} "
              f"diff={out[s]['diff']:.2e}")
    return out


def section_causal(device):
    out = {}
    for s in SEEDS:
        seed_dir = RESULTS_ROOT / "he_residual_cross_v3" / f"seed{s}"
        cfg, model, ckpt = load_checkpoint_model(seed_dir, device)
        ds = build_val_dataset(cfg)
        n = len(ds)
        labels = [smp["label"] for smp in ds.samples]
        he_list = [ds[i]["features"]["HE"] for i in range(n)]
        pr_list = [ds[i]["features"]["PR"] for i in range(n)]

        # matched
        full_probs = run_inference(model, he_list, pr_list, device)
        full_auc = auc_from_probs(labels, full_probs)

        # disable_cross
        model.cross_region_mod.disable_cross = True
        dc_probs = run_inference(model, he_list, pr_list, device)
        dc_auc = auc_from_probs(labels, dc_probs)
        model.cross_region_mod.disable_cross = False

        # random mismatched (derangement over all n)
        rand_mm = {}
        for rs in REPLACEMENT_SEEDS:
            rng = np.random.RandomState(rs)
            perm = _derangement(n, rng)
            pr_swapped = [pr_list[perm[i]] for i in range(n)]
            der_probs = run_inference(model, he_list, pr_swapped, device)
            rand_mm[rs] = {"replacement_seed": rs,
                           "auc_all": auc_from_probs(labels, der_probs)}
        # within-class mismatched (derangement within tumor / within normal)
        wc_mm = {}
        for rs in REPLACEMENT_SEEDS:
            rng = np.random.RandomState(rs)
            perm = within_class_derangement(labels, rng)
            pr_swapped = [pr_list[perm[i]] for i in range(n)]
            der_probs = run_inference(model, he_list, pr_swapped, device)
            wc_mm[rs] = {"replacement_seed": rs,
                         "auc_all": auc_from_probs(labels, der_probs)}

        out[s] = {
            "matched_auc": full_auc,
            "disable_cross_auc": dc_auc,
            "matched_minus_disable": full_auc - dc_auc,
            "random_mismatch": rand_mm,
            "within_class_mismatch": wc_mm,
        }
    (EVAL_DIR / "causal.json").write_text(json.dumps(out, indent=2) + "\n")
    for s in SEEDS:
        r = out[s]
        print(f"seed{s}: matched={r['matched_auc']:.4f} disable={r['disable_cross_auc']:.4f} "
              f"(Δ={r['matched_minus_disable']:+.4f})")
        for rs in REPLACEMENT_SEEDS:
            rm = r["random_mismatch"][rs]["auc_all"]
            wm = r["within_class_mismatch"][rs]["auc_all"]
            print(f"  repl_seed{rs}: random_mm={rm:.4f} within_class_mm={wm:.4f}")
    return out


def section_diagnostics(device):
    out = {}
    for s in SEEDS:
        seed_dir = RESULTS_ROOT / "he_residual_cross_v3" / f"seed{s}"
        cfg, model, ckpt = load_checkpoint_model(seed_dir, device)
        ds = build_val_dataset(cfg)
        n = len(ds)
        keys = ['entropy_norm', 'score_std', 'value_diversity', 'selective_ratio',
                'mu_norm', 'r_slide', 'r_region']
        acc = {k: [] for k in keys}
        rho = []
        for i in range(n):
            he = ds[i]["features"]["HE"].to(device).unsqueeze(0)
            pr = ds[i]["features"]["PR"].to(device).unsqueeze(0)
            d = compute_v3_diag(model, he, pr)
            for k in keys:
                acc[k].append(float(d[k]))
            scaled = 0.1 * d['delta_patch']
            rho.append((scaled.norm().item() / (d['z_he'].norm().item() + 1e-8)))
        out[s] = {k: _percentiles(acc[k]) for k in keys}
        out[s]['rho'] = _percentiles(rho)
    (EVAL_DIR / "diagnostics.json").write_text(json.dumps(out, indent=2) + "\n")
    for s in SEEDS:
        for k in keys:
            print(f"seed{s} {k}: {out[s][k]}")
        print(f"seed{s} rho: {out[s]['rho']}")
    return out


def section_summary(device):
    """Aggregate repro + causal + diagnostics + reused baselines → _eval/summary.json."""
    repro = json.loads((EVAL_DIR / "repro.json").read_text())
    causal = json.loads((EVAL_DIR / "causal.json").read_text())
    diag = json.loads((EVAL_DIR / "diagnostics.json").read_text())

    def _mean_std(a):
        vals = list(a.values())
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    v3_auc = {str(s): repro[str(s)]["recomputed_auc"] for s in SEEDS}

    # reused baselines
    v1_summary = json.loads((V1_ROOT / "summary.json").read_text())
    v2_summary = json.loads((V2_ROOT / "summary.json").read_text())
    he_only_auc = v1_summary["conditions"]["he_only"]["per_seed_auc"]
    v1_auc = v1_summary["conditions"]["he_residual_cross"]["per_seed_auc"]
    v2_auc = v2_summary["condition"]["per_seed_auc"]

    def _delta(t, b):
        common = sorted(set(t) & set(b))
        ds = [t[s] - b[s] for s in common]
        return {"per_seed": {s: round(t[s] - b[s], 6) for s in common},
                "mean": float(np.mean(ds)) if ds else None,
                "std": float(np.std(ds)) if ds else None}

    # causal aggregate: mean matched, mean disable, mean random_mm, mean wc_mm
    matched = {str(s): causal[str(s)]["matched_auc"] for s in SEEDS}
    disable = {str(s): causal[str(s)]["disable_cross_auc"] for s in SEEDS}
    rand_mm = {}
    wc_mm = {}
    for rs in REPLACEMENT_SEEDS:
        rand_mm[str(rs)] = {str(s): causal[str(s)]["random_mismatch"][str(rs)]["auc_all"]
                            for s in SEEDS}
        wc_mm[str(rs)] = {str(s): causal[str(s)]["within_class_mismatch"][str(rs)]["auc_all"]
                          for s in SEEDS}

    summary = {
        "task": "stage2_he_residual_cross_v3_eval",
        "evaluation_protocol": "Train / Val-as-Test",
        "v3_per_seed_auc": v3_auc,
        "v3_mean_auc": _mean_std(v3_auc)[0],
        "v3_std_auc": _mean_std(v3_auc)[1],
        "reused_baselines": {
            "he_only": he_only_auc, "v1": v1_auc, "v2": v2_auc,
        },
        "paired_delta_auc": {
            "v3_minus_he": _delta(v3_auc, he_only_auc),
            "v3_minus_v1": _delta(v3_auc, v1_auc),
            "v3_minus_v2": _delta(v3_auc, v2_auc),
        },
        "causal": {
            "matched_auc": matched,
            "disable_cross_auc": disable,
            "matched_minus_disable_mean": _delta(matched, disable)["mean"],
            "random_mismatch_auc": rand_mm,
            "within_class_mismatch_auc": wc_mm,
            "random_mm_mean_over_repl": {
                str(rs): _mean_std(rand_mm[str(rs)])[0] for rs in REPLACEMENT_SEEDS},
            "wc_mm_mean_over_repl": {
                str(rs): _mean_std(wc_mm[str(rs)])[0] for rs in REPLACEMENT_SEEDS},
        },
        "diagnostics": diag,
    }
    (EVAL_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2)[:4000])
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", nargs="+",
                    default=["repro", "causal", "diagnostics", "summary"])
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    sections = {
        "repro": section_repro,
        "causal": section_causal,
        "diagnostics": section_diagnostics,
        "summary": section_summary,
    }
    for name in args.section:
        if name not in sections:
            print(f"[skip] unknown section {name}")
            continue
        print(f"\n=== {name} ===", flush=True)
        sections[name](device)
    print("\nEVAL DONE", flush=True)


if __name__ == "__main__":
    main()
