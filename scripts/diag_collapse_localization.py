#!/usr/bin/env python3
"""PR representation collapse localization — v2 checkpoints, no retraining.

Goal: locate *which layer* makes the PR token representation highly homogeneous
(the known v2 symptom: attention entropy ≈ 1, V-token pairwise cosine ≈ 0.976–0.999,
matched PR ≈ mismatched PR).  We do NOT retrain, modify the model, or change any
hyper-parameter.  We only run the trained v2 checkpoints (seeds 42/123/456) over
the val-as-test set (129 WSIs) and measure, at each PR pipeline stage, how much
token diversity survives.

PR pipeline layers (per slide):
    L0  X_PR            = raw PR patch features            [N, 768]
    L1  E_PR            = patch_to_emb[1] (Linear+GELU)   [N, 512]
    L2  Z_PR            = rrt_ihc(E_PR)                   [N, 512]   <- Stage1 output
    L3  route_norm_pr   = route_norm_pr(Z_PR)             [N, 512]
    L4  R_PR            = routing tokens                  [G*k, 512]
    L5  attn_norm_pr    = attn_norm_pr(R_PR)              [G*k, 512]
    L6  K_PR            = w_k(attn_norm_pr(R_PR))         [G*k, 512]
    L7  V_PR            = w_v(attn_norm_pr(R_PR))         [G*k, 512]

HE controls (same metrics, key layers):
    H2  Z_HE            = rrt_he(E_HE)
    H4  R_HE            = HE routing tokens
    H5  attn_norm_he    = attn_norm_he(R_HE)
    HQ  Q_HE            = w_q(attn_norm_he(R_HE))

Per-layer metrics (per slide, then aggregated over 129 slides):
    A  pairwise cosine    mean / median / p90 (over token pairs)
    B  token variance     mean_d Var_t(x[t,d]);  mean_token_std = mean_t std_d(x[t,:])
    C  effective rank     entropy-based over singular values
    D  cross-slide centroid diversity (mean pairwise cosine + centroid variance)

Stage2 routing extra checks:
    combine_weights slot-slot patch-selection similarity (cos + overlap@topK)
    routing token pairwise cosine (same region vs different region)

Output: results/stage2_he_residual_cross_v2/_eval/collapse_localization.json
"""
import os, sys, json, argparse, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from models.rmsa import region_partition

RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v2"
EVAL_DIR = RESULTS_ROOT / "_eval"
SEEDS = [42, 123, 456]
OUT_JSON = EVAL_DIR / "collapse_localization.json"


# ---------------------------------------------------------------------------
# model / data loading (identical to eval_v2.py)
# ---------------------------------------------------------------------------
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
    return cfg, model


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


# ---------------------------------------------------------------------------
# metric primitives
# ---------------------------------------------------------------------------
def pairwise_cosine_stats(X):
    """mean / median / p90 of pairwise cosine over token pairs of X [T, D]."""
    T = X.shape[0]
    Xn = F.normalize(X, dim=-1, eps=1e-8)
    s = Xn.sum(0)
    mean = ((s @ s) - T) / (T * (T - 1))
    if T <= 4096:  # full exact upper-triangle for median/p90
        G = Xn @ Xn.t()
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=X.device), 1)
        vals = G[mask]
        q = torch.quantile(vals, torch.tensor([0.5, 0.9], device=X.device))
        return float(mean.item()), float(q[0].item()), float(q[1].item())
    else:
        return float(mean.item()), float('nan'), float('nan')


def token_variance_stats(X):
    """mean_d Var_t(x[t,d]);  mean_token_std = mean_t std_d(x[t,:])."""
    v = X.var(dim=0).mean().item()             # mean over dims of per-dim variance
    s = X.std(dim=1).mean().item()             # mean over tokens of per-token std
    return float(v), float(s)


def effective_rank(X):
    """entropy-based effective rank over singular values of X [T, D]."""
    G = X.t() @ X                              # [D, D]
    eig = torch.linalg.eigvalsh(G)             # ascending, real
    eig = torch.clamp(eig, min=0.0)
    s = torch.sqrt(eig)                        # singular values ascending
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    H = -(p * torch.log(p)).sum()
    return float(torch.exp(H).item())


# ---------------------------------------------------------------------------
# faithful re-implementation of _route that ALSO returns combine_weights
# (verify in smoke test that `routing` matches the module's own `_route`)
# ---------------------------------------------------------------------------
def route_full(m, z, phi, route_norm, valid=None):
    B, L, D = z.shape
    z_n = route_norm(z)                                    # [B, N, D]
    x_padded, H, W, add_length, region_size = m._pad(z_n)
    x_grid = x_padded.view(B, H, W, D)
    x_regions = region_partition(x_grid, region_size)
    G = (H // region_size) * (W // region_size)
    P = region_size * region_size
    x_regions = x_regions.view(B, G, region_size, region_size, D)
    x_regions = x_regions.view(B, G, P, D)

    if valid is None:
        valid_full = torch.ones(B, L, device=z.device, dtype=torch.bool)
    else:
        valid_full = valid.to(device=z.device, dtype=torch.bool).reshape(B, L)
    if add_length > 0:
        pad_valid = torch.zeros(B, add_length, device=z.device, dtype=torch.bool)
        valid_full = torch.cat([valid_full, pad_valid], dim=1)
    valid_grid = valid_full.view(B, H, W)
    valid_regions = m._partition_mask(valid_grid, region_size)   # [B, G, P] bool

    if m.crmsa_mlp:
        logits = phi(x_regions).transpose(-1, -2)
    else:
        logits = torch.einsum('bgpd, dk -> bgkp', x_regions, phi)  # [B, G, k, P]
    k = logits.shape[2]

    nonempty = valid_regions.any(dim=-1)                          # [B, G]
    nonempty_kp = nonempty.unsqueeze(-1).unsqueeze(-1)
    logits_masked = logits.masked_fill(~valid_regions.unsqueeze(2), float('-inf'))
    combine_weights = torch.softmax(logits_masked, dim=-1)        # [B, G, k, P]
    combine_weights = torch.where(
        nonempty_kp, combine_weights, torch.zeros_like(combine_weights))

    routing = torch.einsum('bgpd, bgkp -> bgkd', x_regions, combine_weights)
    routing = routing * nonempty_kp

    return routing, combine_weights, valid_regions, G, P, k, region_size


# ---------------------------------------------------------------------------
# extract one slide's PR/HE layer representations
# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract_slide(model, he, pr):
    """he/pr: [1, N, 768] on device. Returns dict of [T, D] tensors (CPU float)."""
    m = model.cross_region_mod
    E_HE = model.dp(model.patch_to_emb[0](he))           # [1, N, 512]
    E_PR = model.dp(model.patch_to_emb[1](pr))
    Z_HE = model.rrt_he(E_HE)
    Z_PR = model.rrt_ihc(E_PR)

    # PR Stage2 layers
    route_preln_pr = m.route_norm_pr(Z_PR)               # [1, N, 512]  (L3)
    routing_pr, cw_pr, valid_pr_reg, G_pr, P_pr, k_pr, rs_pr = \
        route_full(m, Z_PR, m.phi_pr, m.route_norm_pr, None)
    R_PR = routing_pr.reshape(1, -1, m.dim)              # [1, G*k, 512] (L4)
    attn_ln_pr = m.attn_norm_pr(R_PR)                    # (L5)
    K_PR = m.w_k(attn_ln_pr)                             # (L6)
    V_PR = m.w_v(attn_ln_pr)                             # (L7)

    # HE Stage2 controls
    routing_he, cw_he, valid_he_reg, G_he, P_he, k_he, rs_he = \
        route_full(m, Z_HE, m.phi_he, m.route_norm_he, None)
    R_HE = routing_he.reshape(1, -1, m.dim)              # (H4)
    attn_ln_he = m.attn_norm_he(R_HE)                    # (H5)
    Q_HE = m.w_q(attn_ln_he)                             # (HQ)

    return {
        # PR pipeline
        'L0': he.new_zeros(()),  # placeholder; filled from pr raw below
        'X_PR': pr.squeeze(0).float(),
        'E_PR': E_PR.squeeze(0).float(),
        'Z_PR': Z_PR.squeeze(0).float(),
        'route_preln_pr': route_preln_pr.squeeze(0).float(),
        'R_PR': R_PR.squeeze(0).float(),
        'attn_ln_pr': attn_ln_pr.squeeze(0).float(),
        'K_PR': K_PR.squeeze(0).float(),
        'V_PR': V_PR.squeeze(0).float(),
        # HE controls
        'Z_HE': Z_HE.squeeze(0).float(),
        'R_HE': R_HE.squeeze(0).float(),
        'attn_ln_he': attn_ln_he.squeeze(0).float(),
        'Q_HE': Q_HE.squeeze(0).float(),
        # routing extra: combine_weights (valid-region-masked, CPU float)
        'cw_pr': cw_pr.squeeze(0).float(),               # [G, k, P]
        'cw_he': cw_he.squeeze(0).float(),
        'valid_pr_reg': valid_pr_reg.squeeze(0),         # [G, P] bool
        'valid_he_reg': valid_he_reg.squeeze(0),
        'G': G_pr, 'k': k_pr, 'P': P_pr,
    }


# ---------------------------------------------------------------------------
# routing slot analysis
# ---------------------------------------------------------------------------
def routing_slot_analysis(cw, R, G, k, P, topK=10):
    """cw: [G,k,P] combine_weights; R: [G*k, D] routing tokens.
    Returns dict of mean slot-slot combine-weight cosine, overlap@topK,
    routing-token cosine (same region vs different region)."""
    cw = cw.to(torch.float32)
    cw_flat = cw.reshape(G * k, P)                       # [G*k, P]
    # combine-weight cosine between slot pairs within the same region
    k_eff = min(topK, P)
    cos_pairs = []
    overlap = []
    for g in range(G):
        w = cw[g]                                        # [k, P]
        for a in range(k):
            for b in range(a + 1, k):
                wa, wb = w[a], w[b]
                ca = (wa @ wb) / (wa.norm() * wb.norm() + 1e-12)
                cos_pairs.append(ca.item())
                topa = set(torch.topk(wa, k_eff).indices.tolist())
                topb = set(torch.topk(wb, k_eff).indices.tolist())
                overlap.append(len(topa & topb) / k_eff)
    # routing-token cosine: same region vs different region
    Rn = F.normalize(R.reshape(G, k, -1), dim=-1)        # [G, k, D]
    same = []
    diff = []
    for g in range(G):
        for a in range(k):
            for b in range(a + 1, k):
                same.append((Rn[g, a] @ Rn[g, b]).item())
    for g1 in range(G):
        for g2 in range(g1 + 1, G):
            for a in range(k):
                for b in range(k):
                    diff.append((Rn[g1, a] @ Rn[g2, b]).item())
    return {
        'combine_weight_cos_same_region': float(np.mean(cos_pairs)),
        'combine_weight_overlap_topK': float(np.mean(overlap)),
        'routing_token_cos_same_region': float(np.mean(same)),
        'routing_token_cos_diff_region': float(np.mean(diff)),
    }


# ---------------------------------------------------------------------------
# per-layer metric collection
# ---------------------------------------------------------------------------
def pct(a):
    a = np.asarray(a, dtype=float)
    return {"mean": float(np.mean(a)), "std": float(np.std(a)),
            "p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90))}


def run_seed(seed, device, topK=10):
    seed_dir = RESULTS_ROOT / "he_residual_cross_v2" / f"seed{seed}"
    cfg, model = load_checkpoint_model(seed_dir, device)
    ds = build_val_dataset(cfg)
    n = len(ds)
    labels = [smp["label"] for smp in ds.samples]

    # layer name -> list of per-slide metrics
    layers = ['X_PR', 'E_PR', 'Z_PR', 'route_preln_pr', 'R_PR',
              'attn_ln_pr', 'K_PR', 'V_PR',
              'Z_HE', 'R_HE', 'attn_ln_he', 'Q_HE']
    accum = {ly: dict(pc_mean=[], pc_med=[], pc_p90=[], var=[], tok_std=[],
                      effrank=[], centroid=[]) for ly in layers}
    slot_stats = []

    for i in range(n):
        he = ds[i]["features"]["HE"].to(device).unsqueeze(0)
        pr = ds[i]["features"]["PR"].to(device).unsqueeze(0)
        r = extract_slide(model, he, pr)

        for ly in layers:
            X = r[ly]                                    # [T, D]
            pc_mean, pc_med, pc_p90 = pairwise_cosine_stats(X)
            var, tok_std = token_variance_stats(X)
            er = effective_rank(X)
            accum[ly]['pc_mean'].append(pc_mean)
            accum[ly]['pc_med'].append(pc_med)
            accum[ly]['pc_p90'].append(pc_p90)
            accum[ly]['var'].append(var)
            accum[ly]['tok_std'].append(tok_std)
            accum[ly]['effrank'].append(er)
            accum[ly]['centroid'].append(X.mean(0).cpu().numpy())

        slot_stats.append(routing_slot_analysis(
            r['cw_pr'], r['R_PR'], r['G'], r['k'], r['P'], topK=topK))
        if (i + 1) % 25 == 0:
            print(f"  seed{seed}: {i+1}/{n} slides done", flush=True)

    # aggregate per-layer per-slide metrics + cross-slide centroid diversity
    out = {}
    for ly in layers:
        C = np.stack(accum[ly]['centroid'])              # [129, D]
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        S = Cn.sum(0)
        cent_cos = (float((S @ S) - C.shape[0]) / (C.shape[0] * (C.shape[0] - 1)))
        cent_var = float(np.var(C, axis=0).mean())
        out[ly] = {
            'pairwise_cosine_mean': pct(accum[ly]['pc_mean']),
            'pairwise_cosine_median': pct(accum[ly]['pc_med']),
            'pairwise_cosine_p90': pct(accum[ly]['pc_p90']),
            'token_variance': pct(accum[ly]['var']),
            'mean_token_std': pct(accum[ly]['tok_std']),
            'effective_rank': pct(accum[ly]['effrank']),
            'cross_slide_centroid_cosine': cent_cos,
            'cross_slide_centroid_variance': cent_var,
        }

    # aggregate routing slot analysis over slides
    slot_agg = {}
    for key in slot_stats[0].keys():
        slot_agg[key] = pct([s[key] for s in slot_stats])

    return {'seed': seed, 'n_slides': n, 'layers': out, 'routing_slot': slot_agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--topK", type=int, default=10)
    ap.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    results = {}
    for s in args.seed:
        print(f"=== seed {s} ===", flush=True)
        results[str(s)] = run_seed(s, device, topK=args.topK)

    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWROTE {OUT_JSON}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
