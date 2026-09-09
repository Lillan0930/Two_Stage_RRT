#!/usr/bin/env python3
"""v3 pre-implementation confirmations (NO retraining, NO model change).

Two quick checks on the trained v2 checkpoints (seeds 42/123/456):

  A. PR centroid residual test — for each PR slide compute the Stage2 value
     centroid c_i = mean_t V_{i,t} (over valid routing tokens).  Using ONLY the
     train set compute mu_train = mean_i c_i, then delta_i = c_i - mu_train and
     report, on val-as-test:
         ||delta_i|| / ||c_i||
         pairwise cosine(c_i, c_j)
         pairwise cosine(delta_i, delta_j)
         variance(delta_i)
         effective rank of the slide-level delta matrix [N_val, D]
     The key question: after subtracting the train-set common centroid, do
     different PR slides re-gain visible slide-specific diversity (i.e. does
     cos(delta_i, delta_j) drop well below cos(c_i, c_j))?

  B. HE X->Embed control — measure HE raw X_HE -> patch_to_emb -> E_HE vs PR
     X_PR -> E_PR with identical metrics (pairwise cosine, token variance,
     effective rank), to judge whether the 768->512 Linear+GELU diversity
     compression is shared by HE and PR (if so, do NOT touch patch_to_emb).

Output: results/stage2_he_residual_cross_v2/_eval/v3_prototype_centroid.json
"""
import os, sys, json, argparse
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
OUT_JSON = EVAL_DIR / "v3_prototype_centroid.json"


# ---------------------------------------------------------------------------
# model / data loading (identical to eval_v2.py / diag_collapse_localization.py)
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


# ---------------------------------------------------------------------------
# metric primitives
# ---------------------------------------------------------------------------
def pairwise_cosine(X):
    """mean pairwise cosine over rows of X [T, D] (gram trick)."""
    T = X.shape[0]
    if T < 2:
        return 0.0
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn.sum(0)
    return float(((S @ S) - T) / (T * (T - 1)))


def token_variance(X):
    """mean_d Var_t(x[t,d])."""
    return float(np.var(X, axis=0).mean())


def effective_rank_np(X):
    """entropy-based effective rank over singular values of X [T, D]."""
    G = X.T @ X
    eig = np.linalg.eigvalsh(G)
    eig = np.clip(eig, 0.0, None)
    s = np.sqrt(eig)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def route_full(m, z, phi, route_norm, valid=None):
    """faithful _route re-implementation returning routing + valid_slots [G,k]."""
    B, L, D = z.shape
    z_n = route_norm(z)
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
    valid_regions = m._partition_mask(valid_grid, region_size)

    logits = torch.einsum('bgpd, dk -> bgkp', x_regions, phi)
    k = logits.shape[2]
    nonempty = valid_regions.any(dim=-1)                       # [B, G]
    nonempty_kp = nonempty.unsqueeze(-1).unsqueeze(-1)
    logits_masked = logits.masked_fill(~valid_regions.unsqueeze(2), float('-inf'))
    combine_weights = torch.softmax(logits_masked, dim=-1)
    combine_weights = torch.where(
        nonempty_kp, combine_weights, torch.zeros_like(combine_weights))
    routing = torch.einsum('bgpd, bgkp -> bgkd', x_regions, combine_weights)
    routing = routing * nonempty_kp
    valid_slots = nonempty.unsqueeze(-1).expand(B, G, k).contiguous()  # [B, G, k]
    return routing, valid_slots, G, k


@torch.inference_mode()
def extract(model, he, pr):
    """Return V_PR / R_PR centroids (valid-masked) + X/E for both modalities."""
    m = model.cross_region_mod
    E_HE = model.dp(model.patch_to_emb[0](he))
    E_PR = model.dp(model.patch_to_emb[1](pr))
    Z_HE = model.rrt_he(E_HE)
    Z_PR = model.rrt_ihc(E_PR)

    routing_pr, valid_slots_pr, G, k = route_full(
        m, Z_PR, m.phi_pr, m.route_norm_pr, None)
    R_PR = routing_pr.reshape(1, -1, m.dim)                    # [1, G*k, D]
    attn_ln_pr = m.attn_norm_pr(R_PR)
    V_PR = m.w_v(attn_ln_pr)                                   # [1, G*k, D]
    vmask = valid_slots_pr.reshape(1, -1).float()              # [1, G*k]

    routing_he, valid_slots_he, _, _ = route_full(
        m, Z_HE, m.phi_he, m.route_norm_he, None)
    R_HE = routing_he.reshape(1, -1, m.dim)

    out = {
        'c_vpr': (V_PR.squeeze(0) * vmask.squeeze(0).unsqueeze(1)).sum(0)
                 / vmask.sum().clamp(min=1.0),
        'c_rpr': (R_PR.squeeze(0) * vmask.squeeze(0).unsqueeze(1)).sum(0)
                 / vmask.sum().clamp(min=1.0),
        'X_HE': he.squeeze(0).float(),
        'E_HE': E_HE.squeeze(0).float(),
        'X_PR': pr.squeeze(0).float(),
        'E_PR': E_PR.squeeze(0).float(),
    }
    return {kk: vv.cpu() for kk, vv in out.items()}


def centroid_residual_metrics(centroids, mu_train):
    """Given [N, D] slide centroids and [D] train prototype, return metric dict."""
    C = centroids                                     # [N, D]
    delta = C - mu_train                              # [N, D] (mu_train [1, D] broadcasts)
    cn = np.linalg.norm(C, axis=1, keepdims=True)
    dn = np.linalg.norm(delta, axis=1, keepdims=True)
    ratio = dn / (cn + 1e-8)                          # [N, 1]
    return {
        'n': int(C.shape[0]),
        'cos_c': pairwise_cosine(C),
        'cos_delta': pairwise_cosine(delta),
        'ratio_mean': float(np.mean(ratio)),
        'ratio_median': float(np.median(ratio)),
        'delta_variance': token_variance(delta),       # mean_d Var_i(delta[d])
        'delta_norm_mean': float(np.mean(dn)),
        'delta_effective_rank': effective_rank_np(delta),
        'c_norm_mean': float(np.mean(cn)),
    }


def run_seed(seed, device):
    seed_dir = RESULTS_ROOT / "he_residual_cross_v2" / f"seed{seed}"
    cfg, model = load_checkpoint_model(seed_dir, device)

    train_ds = build_dataset(cfg, str(PROJECT / "data/C16_labels/c16_train_labels.csv"))
    val_ds = build_dataset(cfg, cfg["data"]["val_label_file"])

    # ── train-set prototype ──
    train_c_vpr = []
    train_c_rpr = []
    for i in range(len(train_ds)):
        he = train_ds[i]["features"]["HE"].to(device).unsqueeze(0)
        pr = train_ds[i]["features"]["PR"].to(device).unsqueeze(0)
        r = extract(model, he, pr)
        train_c_vpr.append(r['c_vpr'].numpy())
        train_c_rpr.append(r['c_rpr'].numpy())
        if (i + 1) % 90 == 0:
            print(f"  seed{seed}: train {i+1}/{len(train_ds)}", flush=True)
    train_c_vpr = np.stack(train_c_vpr)                 # [270, D]
    train_c_rpr = np.stack(train_c_rpr)
    mu_vpr = train_c_vpr.mean(0, keepdims=True)         # [1, D]
    mu_rpr = train_c_rpr.mean(0, keepdims=True)

    # ── val-as-test residual metrics ──
    val_c_vpr, val_c_rpr = [], []
    xhe, ehe, xpr, epr = [], [], [], []
    for i in range(len(val_ds)):
        he = val_ds[i]["features"]["HE"].to(device).unsqueeze(0)
        pr = val_ds[i]["features"]["PR"].to(device).unsqueeze(0)
        r = extract(model, he, pr)
        val_c_vpr.append(r['c_vpr'].numpy())
        val_c_rpr.append(r['c_rpr'].numpy())
        xhe.append(r['X_HE'].numpy()); ehe.append(r['E_HE'].numpy())
        xpr.append(r['X_PR'].numpy()); epr.append(r['E_PR'].numpy())
        if (i + 1) % 50 == 0:
            print(f"  seed{seed}: val {i+1}/{len(val_ds)}", flush=True)
    val_c_vpr = np.stack(val_c_vpr)
    val_c_rpr = np.stack(val_c_rpr)

    def emb_metrics(X_list):
        """aggregate pairwise cosine / token var / eff rank over slides."""
        pc, tv, er = [], [], []
        for X in X_list:
            pc.append(pairwise_cosine(X))
            tv.append(token_variance(X))
            er.append(effective_rank_np(X))
        return {
            'pairwise_cosine_mean': float(np.mean(pc)),
            'pairwise_cosine_std': float(np.std(pc)),
            'token_variance_mean': float(np.mean(tv)),
            'effective_rank_mean': float(np.mean(er)),
        }

    return {
        'seed': seed,
        'n_train': int(train_c_vpr.shape[0]),
        'n_val': int(val_c_vpr.shape[0]),
        'prototype_magnitude_vpr': float(np.linalg.norm(mu_vpr)),
        'prototype_magnitude_rpr': float(np.linalg.norm(mu_rpr)),
        'val_vpr': centroid_residual_metrics(val_c_vpr, mu_vpr),
        'val_rpr': centroid_residual_metrics(val_c_rpr, mu_rpr),
        'embed_control': {
            'HE_X': emb_metrics(xhe),
            'HE_E': emb_metrics(ehe),
            'PR_X': emb_metrics(xpr),
            'PR_E': emb_metrics(epr),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")

    results = {}
    for s in args.seed:
        print(f"=== seed {s} ===", flush=True)
        results[str(s)] = run_seed(s, device)

    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWROTE {OUT_JSON}", flush=True)
    for s in args.seed:
        r = results[str(s)]
        v = r['val_vpr']
        e = r['embed_control']
        print(f"seed{s}: cos_c={v['cos_c']:.4f} cos_delta={v['cos_delta']:.4f} "
              f"ratio_mean={v['ratio_mean']:.3f} delta_er={v['delta_effective_rank']:.1f}")
        print(f"   embed: HE_X pc={e['HE_X']['pairwise_cosine_mean']:.3f} -> HE_E "
              f"pc={e['HE_E']['pairwise_cosine_mean']:.3f} | PR_X pc={e['PR_X']['pairwise_cosine_mean']:.3f} "
              f"-> PR_E pc={e['PR_E']['pairwise_cosine_mean']:.3f}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
