#!/usr/bin/env python3
"""Round 5 — intra-slide token-order sensitivity of single-modality RRT.

Question: how much does a HE-only / PR-only RRT (two_stage_region, num_modalities=1)
depend on the *current* patch-token order?

Method (test-time only, no training, no weight change, no re-extraction):
  For each WSI, load its tile feature X ∈ R^{N×768} (exactly as the eval pipeline
  does: max_patches=2500 random sampling, sorted indices, per_epoch=False).
    - Original AUC  : forward(X)
    - Shuffled AUC  : forward(X[perm])  with perm a random permutation of 0..N-1
                      drawn *per slide* (no cross-slide exchange; values/labels/count
                      unchanged).  ≥5 shuffle seeds.
  Numerical equivalence: forward(X[identity]) MUST equal forward(X) exactly
  (max_abs_diff ≈ 0), proving the shuffle machinery itself introduces no change.

Usage:
  python scripts/diag_order_sensitivity.py \
      --config results/he_rrt_samplerfix_lr1e4/seed42/config.json \
      --checkpoint results/he_rrt_samplerfix_lr1e4/seed42/ckpt/best_model.pt \
      --gpu 0 --shuffle-seeds 5 --out results/order_sens_he_seed42.json
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))

DEFAULT_TEST_LABEL = str(PROJECT / "data/C16_labels/c16_test_labels.csv")


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
    # drop unused multi-modality MM_RRTEncoder params (present in these ckpts but
    # never used in the num_modalities==1 forward path)
    sd = {k: v for k, v in sd.items() if not k.startswith("rrt_encoder.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("rrt_encoder.")]
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys (ignored): {unexpected}")
    return ck.get("epoch") if isinstance(ck, dict) else None


def forward_one(model, x, device):
    feats = [x.unsqueeze(0).to(device)]
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

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = json.loads(Path(args.config).read_text())
    dc = cfg["data"]
    from train import build_feature_dirs
    from data.c16_multimodal_dataset import C16MultimodalDataset

    model = build_model(cfg).to(device)
    epoch = load_checkpoint(model, args.checkpoint)
    model.eval()

    label_file = dc.get("val_label_file", DEFAULT_TEST_LABEL)
    feature_dirs = build_feature_dirs(dc["feature_base_dir"], dc["modalities"],
                                      dc.get("dir_mapping", None))
    ds = C16MultimodalDataset(
        feature_dirs=feature_dirs, label_file=label_file,
        max_patches=dc.get("max_patches", 2500), preload=False, verbose=False,
        sampling=dc.get("sampling", "random"), sample_seed=dc.get("sample_seed", 0),
        per_epoch=False,
    )
    mod = dc["modalities"][0]

    slides = []
    for i in range(len(ds)):
        item = ds[i]
        x = item["features"][mod].float()  # [N, 768]
        slides.append({
            "slide_id": item["slide_id"],
            "label": int(item["label"]),
            "x": x,
            "n": int(x.shape[0]),
        })
    labels = np.array([s["label"] for s in slides])
    n = len(slides)
    print(f"==> {n} test slides | modality={mod} | checkpoint epoch={epoch} | device={device}")

    # ---- original (unshuffled) forward ----
    orig_prob = np.zeros(n, dtype=np.float64)
    with torch.inference_mode():
        for i in range(n):
            logits = forward_one(model, slides[i]["x"], device)
            orig_prob[i] = torch.softmax(logits, dim=1)[0, 1].item()
    orig_auc = roc_auc_score(labels, orig_prob)

    # ---- numerical equivalence: identity permutation == original ----
    x0 = slides[0]["x"]
    with torch.inference_mode():
        logits_orig = forward_one(model, x0, device)
        logits_ident = forward_one(model, x0[torch.arange(x0.shape[0])], device)
    max_diff = float((logits_orig - logits_ident).abs().max())
    print(f"==> numerical equivalence (identity perm vs original): max_abs_diff = {max_diff:.3e}")
    assert max_diff < 1e-6, f"NOT equivalent: {max_diff}"

    # ---- intra-slide shuffle (≥5 seeds) ----
    shuffled_aucs = []
    for s in range(args.shuffle_seeds):
        rng = np.random.RandomState(1000 + s)
        prob = np.zeros(n, dtype=np.float64)
        with torch.inference_mode():
            for i in range(n):
                x = slides[i]["x"]
                perm = rng.permutation(x.shape[0])
                logits = forward_one(model, x[perm], device)
                prob[i] = torch.softmax(logits, dim=1)[0, 1].item()
        auc = roc_auc_score(labels, prob)
        shuffled_aucs.append(auc)
        print(f"   shuffle seed {s}: AUC = {auc:.4f}   (Δ = {auc - orig_auc:+.4f})")
    shuffled_aucs = np.array(shuffled_aucs)

    results = {
        "modality": mod,
        "n_slides": n,
        "n_tokens_per_slide": [s["n"] for s in slides],
        "numerical_equiv_max_abs_diff": max_diff,
        "original_auc": float(orig_auc),
        "shuffled_aucs": [float(a) for a in shuffled_aucs],
        "shuffled_auc_mean": float(shuffled_aucs.mean()),
        "shuffled_auc_std": float(shuffled_aucs.std()),
        "delta_mean": float(shuffled_aucs.mean() - orig_auc),
        "delta_min": float(shuffled_aucs.min() - orig_auc),
        "delta_max": float(shuffled_aucs.max() - orig_auc),
    }

    print("\n" + "=" * 72)
    print(f"INTRA-SLIDE SHUFFLE — {mod} (epoch {epoch})")
    print("=" * 72)
    print(f"  Original AUC        : {orig_auc:.4f}")
    print(f"  Shuffled AUC mean±std: {shuffled_aucs.mean():.4f} ± {shuffled_aucs.std():.4f}")
    print(f"  Δ (shuffled-orig)   : {results['delta_mean']:+.4f}")
    print(f"  per-shuffle         : " + "  ".join(f"{a:.4f}" for a in shuffled_aucs))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    print("DONE")


if __name__ == "__main__":
    main()
