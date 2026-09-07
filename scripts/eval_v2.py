#!/usr/bin/env python3
"""HE Residual Cross v2 — post-training evaluation.

Sections (run with --section, or all by default):
  repro        re-run pure inference on 3 v2 checkpoints, compare saved AUC
  causal       matched / random-mismatch(5) / within-class-mismatch(5) / disable
  diagnostics  4 per-slide diagnostics (entropy_norm, score_std, value_diversity,
               selective_ratio) over val-as-test + v1 entropy for comparison

Outputs land in results/stage2_he_residual_cross_v2/_eval/.
"""
import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v2"
EVAL_DIR = RESULTS_ROOT / "_eval"
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


# ---------------------------------------------------------------------------
# v2 diagnostics (uses HEResidualCrossCRMSAv2.diagnose)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def compute_v2_diag(model, he, pr):
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
# v1 attention stats (reimplement v1's raw softmax(QK^T/sqrt(d)) for comparison)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def compute_v1_attn_stats(model, he, pr):
    m = model.cross_region_mod
    z_he = model.rrt_he(model.patch_to_emb[0](model.dp(he)))
    z_ihc = model.rrt_ihc(model.patch_to_emb[1](model.dp(pr)))
    if z_he.dim() == 2:
        z_he = z_he.unsqueeze(0)
    if z_ihc.dim() == 2:
        z_ihc = z_ihc.unsqueeze(0)
    B = z_he.shape[0]
    routing_he, _d, _d2, vhs, _h, _w, _a, _r = m._route(z_he, m.phi_he, m.route_norm_he, None)
    routing_pr, _d, _d2, vps, _h, _w, _a, _r = m._route(z_ihc, m.phi_pr, m.route_norm_pr, None)
    r_he = routing_he.reshape(B, -1, m.dim)
    r_pr = routing_pr.reshape(B, -1, m.dim)
    q_valid = vhs.reshape(B, -1)
    k_valid = vps.reshape(B, -1)
    Q = r_he.shape[1]
    K = r_pr.shape[1]
    h = m.num_heads
    d = m.head_dim
    r_he_n = m.attn_norm_he(r_he)
    r_pr_n = m.attn_norm_pr(r_pr)
    q = m.w_q(r_he_n).view(B, Q, h, d).transpose(1, 2)
    k = m.w_k(r_pr_n).view(B, K, h, d).transpose(1, 2)
    v_full = m.w_v(r_pr_n)
    score = (q @ k.transpose(-2, -1)) * (d ** -0.5)          # [B,h,Q,K]
    attn = score.masked_fill(~k_valid.view(B, 1, 1, K), float('-inf'))
    has_key = k_valid.any(dim=-1)
    A = torch.softmax(attn, dim=-1)
    A = torch.where(has_key.view(B, 1, 1, 1), A, torch.zeros_like(A))

    count = k_valid.float().sum(-1).clamp(min=1.0)
    # entropy_norm
    H = -(A * torch.log(A + 1e-12)).sum(dim=-1)
    log_K = torch.log(count.clamp(min=2.0))
    entropy_norm = (H / log_K.view(B, 1, 1)).mean(dim=(1, 2))
    # score_std (raw score, v1 scale)
    sm = score.masked_fill(~k_valid.view(B, 1, 1, K), 0.0)
    s_mean = sm.sum(-1) / count.view(B, 1, 1)
    d2 = ((score - s_mean.unsqueeze(-1)) ** 2).masked_fill(~k_valid.view(B, 1, 1, K), 0.0)
    score_std = (d2.sum(-1) / count.view(B, 1, 1)).clamp(min=0).sqrt().mean(dim=(1, 2))
    # value_diversity (mean pairwise cosine over valid tokens)
    vn = torch.nn.functional.normalize(
        v_full * k_valid.float().unsqueeze(-1), dim=-1)
    s = vn.sum(dim=1)
    value_diversity = ((s ** 2).sum(-1) - count) / (count * (count - 1)).clamp(min=1.0)
    return {
        'entropy_norm': float(entropy_norm.squeeze(0).item()),
        'score_std': float(score_std.squeeze(0).item()),
        'value_diversity': float(value_diversity.squeeze(0).item()),
    }


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def section_repro(device):
    out = {}
    for s in SEEDS:
        seed_dir = RESULTS_ROOT / "he_residual_cross_v2" / f"seed{s}"
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
        seed_dir = RESULTS_ROOT / "he_residual_cross_v2" / f"seed{s}"
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
        seed_dir = RESULTS_ROOT / "he_residual_cross_v2" / f"seed{s}"
        cfg, model, ckpt = load_checkpoint_model(seed_dir, device)
        ds = build_val_dataset(cfg)
        n = len(ds)
        ents, sstd, vdiv, sratio, rho = [], [], [], [], []
        for i in range(n):
            he = ds[i]["features"]["HE"].to(device).unsqueeze(0)
            pr = ds[i]["features"]["PR"].to(device).unsqueeze(0)
            d = compute_v2_diag(model, he, pr)
            ents.append(float(d['entropy_norm']))
            sstd.append(float(d['score_std']))
            vdiv.append(float(d['value_diversity']))
            sratio.append(float(d['selective_ratio']))
            scaled = 0.1 * d['delta_patch']
            rho.append((scaled.norm().item() / (d['z_he'].norm().item() + 1e-8)))
        out[s] = {
            "entropy_norm": _percentiles(ents),
            "score_std": _percentiles(sstd),
            "value_diversity": _percentiles(vdiv),
            "selective_ratio": _percentiles(sratio),
            "rho": _percentiles(rho),
        }
    (EVAL_DIR / "diagnostics.json").write_text(json.dumps(out, indent=2) + "\n")
    for s in SEEDS:
        print(f"seed{s} entropy_norm: {out[s]['entropy_norm']}")
        print(f"seed{s} score_std: {out[s]['score_std']}")
        print(f"seed{s} value_diversity: {out[s]['value_diversity']}")
        print(f"seed{s} selective_ratio: {out[s]['selective_ratio']}")
        print(f"seed{s} rho: {out[s]['rho']}")
    return out


def section_v1_entropy(device):
    """Recompute v1's attention entropy (and score_std/value_diversity) for the
    'entropy became more selective?' comparison question."""
    out = {}
    for s in SEEDS:
        seed_dir = V1_ROOT / "he_residual_cross" / f"seed{s}"
        cfg, model, ckpt = load_checkpoint_model(seed_dir, device)
        ds = build_val_dataset(cfg)
        n = len(ds)
        ents, sstd, vdiv = [], [], []
        for i in range(n):
            he = ds[i]["features"]["HE"].to(device).unsqueeze(0)
            pr = ds[i]["features"]["PR"].to(device).unsqueeze(0)
            d = compute_v1_attn_stats(model, he, pr)
            ents.append(d['entropy_norm'])
            sstd.append(d['score_std'])
            vdiv.append(d['value_diversity'])
        out[s] = {
            "entropy_norm": _percentiles(ents),
            "score_std": _percentiles(sstd),
            "value_diversity": _percentiles(vdiv),
        }
    (EVAL_DIR / "v1_attention.json").write_text(json.dumps(out, indent=2) + "\n")
    for s in SEEDS:
        print(f"v1 seed{s} entropy_norm: {out[s]['entropy_norm']}")
        print(f"v1 seed{s} score_std: {out[s]['score_std']}")
        print(f"v1 seed{s} value_diversity: {out[s]['value_diversity']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", nargs="+", default=["repro", "causal", "diagnostics", "v1_entropy"])
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    sections = {
        "repro": section_repro,
        "causal": section_causal,
        "diagnostics": section_diagnostics,
        "v1_entropy": section_v1_entropy,
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
