#!/usr/bin/env python3
"""Round 3 — causal ablation diagnosis (PR shuffle + cross-block ablation).

No training, no model-structure change, no weight modification.  Uses an
existing trained two-stage (staining_msa) checkpoint and only adds test-time
diagnostic switches:

  Task 1 (PR shuffle):  keep HE_i fixed, replace PR_i with PR_{pi(i)} (a
      derangement over the test set).  ≥5 shuffle seeds.  Breaks slide-specific
      HE–PR pairing while keeping the multiset of PR tokens unchanged.

  Task 2 (cross-block ablation):  mask routing-token attention logits *before*
      softmax via a forward_pre_hook on the Stage-2 InnerAttention softmax.
        - Full     : no mask
        - No-Cross : mask A_HP (HE q -> PR k) AND A_PH (PR q -> HE k)
        - No HE<-PR: mask A_HP only (cut PR -> HE-side representation)
      Softmax renormalizes over the surviving keys (per query row).

  Numerical equivalence:  with ablation=None the forward MUST equal the
      un-hooked forward exactly (max_abs_diff == 0).

Usage:
  python scripts/diag_causal_ablation.py \
      --config results/twostage_r4_noepeg_samplerfix_unified_lr1e4/seed42/config.json \
      --checkpoint results/twostage_r4_noepeg_samplerfix_unified_lr1e4/seed42/ckpt/best_model.pt \
      --gpu 0 --shuffle-seeds 5 --out results/causal_ablation_seed42.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))


def build_model(cfg):
    from models.mm_rrt_abmil import MM_RRT_ABMIL
    mc, dc = cfg["model"], cfg["data"]
    return MM_RRT_ABMIL(
        num_modalities=len(dc["modalities"]), modality_list=dc["modalities"],
        input_dim=dc["input_dim"], mlp_dim=mc.get("mlp_dim", 512),
        num_classes=dc["num_classes"], dropout=mc.get("dropout", 0.25),
        region_num=mc.get("region_num", 8), n_layers=mc.get("n_layers", 2),
        n_heads=mc.get("n_heads", 8), drop_path=mc.get("drop_path", 0.0),
        trans_dropout=mc.get("trans_dropout", 0.1), epeg=mc.get("epeg", True),
        epeg_k=mc.get("epeg_k", 15), crmsa_k=mc.get("crmsa_k", 3),
        cr_msa=mc.get("cr_msa", True), all_shortcut=mc.get("all_shortcut", False),
        crmsa_heads=mc.get("crmsa_heads", 8), crmsa_mlp=mc.get("crmsa_mlp", False),
        fusion_type=mc.get("fusion_type", "self_attention"),
        fusion_stage=mc.get("fusion_stage", "middle"), fusion_kwargs={},
        stage2_type=mc.get("stage2_type", "staining_msa"),
        use_gated_fusion=mc.get("use_gated_fusion", False),
        use_per_layer_fusion=mc.get("use_per_layer_fusion", True),
        use_logit_fusion=mc.get("use_logit_fusion", False),
        use_consistency_fusion=mc.get("use_consistency_fusion", False),
        use_arlc_fusion=mc.get("use_arlc_fusion", False),
        use_correction_only=mc.get("use_correction_only", False),
        use_logit_attn=mc.get("use_logit_attn", False),
        pretrained_he_ckpt=mc.get("pretrained_he_ckpt", None),
        alpha_mode=mc.get("alpha_mode", "feature"),
        use_lowrank_correction=mc.get("use_lowrank_correction", False),
        use_srp_fusion=mc.get("use_srp_fusion", False),
        srp_beta=mc.get("srp_beta", 0.1), srp_mode=mc.get("srp_mode", "residual"),
        use_shared_rrt=mc.get("use_shared_rrt", False),
        shared_rrt_alpha=mc.get("shared_rrt_alpha", 0.02),
        use_partial_align=mc.get("use_partial_align", False),
        use_mclc=mc.get("use_mclc", False), freeze_mclc=mc.get("freeze_mclc", False),
        he_only=mc.get("he_only", False),
        encoder_cfg=mc.get("encoder_cfg", None), stage2_cfg=mc.get("stage2_cfg", None),
        mil_type=mc.get("mil_type", "abmil"),
        abmil_hidden_dim=mc.get("abmil_hidden_dim", 128),
        use_gated=mc.get("use_gated", False),
    )


def load_checkpoint(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    sd = {k: v for k, v in sd.items() if not k.startswith("rrt_encoder.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("rrt_encoder.")]
    if missing:
        print(f"[warn] missing keys: {missing}")
    return ck.get("epoch") if isinstance(ck, dict) else None


def install_ablation_hook(model):
    """forward_pre_hook on the Stage-2 routing-token softmax.  No-op when
    model.cross_region_mod._ablation is None."""
    cross_mod = model.cross_region_mod
    softmax_mod = cross_mod.attn.softmax

    def prehook(module, inputs):
        logits = inputs[0]                      # [crmsa_k, n_heads, N, N]
        ab = getattr(cross_mod, "_ablation", None)
        if ab is None:
            return None
        N = logits.shape[-1]
        split = N // 2                          # HE regions first (region_num=4 -> 16)
        assert split * 2 == N, f"unexpected routing N={N} (expected even)"
        mask = torch.zeros_like(logits, dtype=torch.bool)
        if ab == "no_cross":
            mask[:, :, :split, split:] = True   # A_HP
            mask[:, :, split:, :split] = True   # A_PH
        elif ab == "no_hp":
            mask[:, :, :split, split:] = True   # A_HP only
        else:
            raise ValueError(f"unknown ablation: {ab}")
        logits.masked_fill_(mask, float("-inf"))
        return None

    softmax_mod.register_forward_pre_hook(prehook)
    return prehook


def derangement(rng, n):
    """Random permutation of 0..n-1 with no fixed points."""
    if n < 2:
        return np.arange(n)
    perm = np.arange(n)
    while True:
        rng.shuffle(perm)
        if (perm == np.arange(n)).sum() == 0:
            return perm.copy()


def forward_one(model, he, pr, device):
    feats = [he.unsqueeze(0).to(device), pr.unsqueeze(0).to(device)]
    out = model(feats)
    logits = out[0].float().cpu()
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shuffle-seeds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = json.loads(Path(args.config).read_text())
    dc = cfg["data"]
    from train import build_feature_dirs
    from data.c16_multimodal_dataset import C16MultimodalDataset

    model = build_model(cfg).to(device)
    epoch = load_checkpoint(model, args.checkpoint)
    model.eval()

    # ---- load test slides (features kept on CPU) ----
    feature_dirs = build_feature_dirs(dc["feature_base_dir"], dc["modalities"],
                                      dc.get("dir_mapping", None))
    ds = C16MultimodalDataset(
        feature_dirs=feature_dirs, label_file=dc["val_label_file"],
        max_patches=dc.get("max_patches", 2500), preload=False, verbose=False,
        sampling=dc.get("sampling", "random"), sample_seed=dc.get("sample_seed", 0),
        per_epoch=False,
    )
    mods = dc["modalities"]
    slides = []
    for i in range(len(ds)):
        item = ds[i]
        slides.append({
            "slide_id": item["slide_id"],
            "label": int(item["label"]),
            "he": item["features"][mods[0]].float(),
            "pr": item["features"][mods[1]].float(),
            "n_he": item["features"][mods[0]].shape[0],
            "n_pr": item["features"][mods[1]].shape[0],
        })
    labels = np.array([s["label"] for s in slides])
    n = len(slides)
    print(f"==> {n} test slides, checkpoint epoch={epoch}, device={device}")

    # ---- numerical equivalence: un-hooked vs hooked(ablation=None) ----
    he0, pr0 = slides[0]["he"], slides[0]["pr"]
    with torch.inference_mode():
        logits_orig = forward_one(model, he0, pr0, device)
    install_ablation_hook(model)
    model.cross_region_mod._ablation = None
    with torch.inference_mode():
        logits_hooked = forward_one(model, he0, pr0, device)
    max_diff = float((logits_orig - logits_hooked).abs().max())
    print(f"==> numerical equivalence: max_abs_diff(orig vs hooked[None]) = {max_diff:.3e}")
    assert max_diff < 1e-6, f"NOT equivalent: {max_diff}"

    def run_pass(ablation=None, perm=None):
        """Return prob_pos [n] for the given ablation / PR permutation."""
        model.cross_region_mod._ablation = ablation
        probs = np.zeros(n, dtype=np.float64)
        with torch.inference_mode():
            for i in range(n):
                he = slides[i]["he"]
                pr_idx = i if perm is None else int(perm[i])
                pr = slides[pr_idx]["pr"]
                logits = forward_one(model, he, pr, device)
                probs[i] = torch.softmax(logits, dim=1)[0, 1].item()
        return probs

    # ---- Task 2: cross-block ablation ----
    full_prob = run_pass(None, None)
    full_auc = roc_auc_score(labels, full_prob)
    no_cross_prob = run_pass("no_cross", None)
    no_cross_auc = roc_auc_score(labels, no_cross_prob)
    no_hp_prob = run_pass("no_hp", None)
    no_hp_auc = roc_auc_score(labels, no_hp_prob)

    # ---- Task 1: PR shuffle (≥5 seeds) ----
    shuffled_aucs = []
    shuffled_probs = []
    for s in range(args.shuffle_seeds):
        rng = np.random.RandomState(1000 + s)
        perm = derangement(rng, n)
        prob = run_pass(None, perm)
        auc = roc_auc_score(labels, prob)
        shuffled_aucs.append(auc)
        shuffled_probs.append(prob)
        print(f"   shuffle seed {s}: AUC = {auc:.4f}")
    shuffled_aucs = np.array(shuffled_aucs)

    # ---- uniform-2500 subset (clean count-preserved shuffle) ----
    uni_idx = np.array([i for i in range(n) if slides[i]["n_he"] == 2500])
    uni_full_auc = roc_auc_score(labels[uni_idx], full_prob[uni_idx])
    # recompute shuffle restricted to uniform subset (count-preserving derangement)
    uni_shuffled_aucs = []
    for s in range(args.shuffle_seeds):
        rng = np.random.RandomState(2000 + s)
        sub_perm = derangement(rng, len(uni_idx))
        model.cross_region_mod._ablation = None
        probs = np.zeros(len(uni_idx), dtype=np.float64)
        with torch.inference_mode():
            for k, i in enumerate(uni_idx):
                he = slides[i]["he"]
                pr = slides[int(uni_idx[sub_perm[k]])]["pr"]
                logits = forward_one(model, he, pr, device)
                probs[k] = torch.softmax(logits, dim=1)[0, 1].item()
        uni_shuffled_aucs.append(roc_auc_score(labels[uni_idx], probs))
    uni_shuffled_aucs = np.array(uni_shuffled_aucs)

    results = {
        "n_slides": n, "n_uniform2500": int(len(uni_idx)),
        "numerical_equiv_max_abs_diff": max_diff,
        "full_auc": float(full_auc),
        "no_cross_auc": float(no_cross_auc),
        "no_hp_auc": float(no_hp_auc),
        "d_no_cross": float(no_cross_auc - full_auc),
        "d_no_hp": float(no_hp_auc - full_auc),
        "shuffle_aucs": shuffled_aucs.tolist(),
        "shuffle_auc_mean": float(shuffled_aucs.mean()),
        "shuffle_auc_std": float(shuffled_aucs.std()),
        "d_shuffle_mean": float(shuffled_aucs.mean() - full_auc),
        "uniform2500_full_auc": float(uni_full_auc),
        "uniform2500_shuffle_auc_mean": float(uni_shuffled_aucs.mean()),
        "uniform2500_shuffle_auc_std": float(uni_shuffled_aucs.std()),
        "uniform2500_d_shuffle_mean": float(uni_shuffled_aucs.mean() - uni_full_auc),
    }

    print("\n" + "=" * 72)
    print("Cross-block ablation (Task 2)")
    print("=" * 72)
    print(f"  Full      : {full_auc:.4f}")
    print(f"  No-Cross  : {no_cross_auc:.4f}   (Δ = {results['d_no_cross']:+.4f})")
    print(f"  No HE<-PR : {no_hp_auc:.4f}   (Δ = {results['d_no_hp']:+.4f})")

    print("\n" + "=" * 72)
    print("PR shuffle (Task 1)")
    print("=" * 72)
    print(f"  Full AUC        : {full_auc:.4f}")
    print(f"  Shuffled AUC    : {results['shuffle_auc_mean']:.4f} ± {results['shuffle_auc_std']:.4f} "
          f"(Δ = {results['d_shuffle_mean']:+.4f})")
    print(f"  per-shuffle     : " + "  ".join(f"{a:.4f}" for a in shuffled_aucs))
    print(f"  [uniform-2500]  Full {uni_full_auc:.4f} → Shuffled "
          f"{results['uniform2500_shuffle_auc_mean']:.4f} ± {results['uniform2500_shuffle_auc_std']:.4f} "
          f"(Δ = {results['uniform2500_d_shuffle_mean']:+.4f})")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    print("DONE")


if __name__ == "__main__":
    main()
