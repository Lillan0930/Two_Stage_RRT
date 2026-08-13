#!/usr/bin/env python3
"""
Diagnostic 3 — Stage2 cross-staining update magnitude.

Loads the trained Two-stage CR-MSA checkpoint (results/c16_twostage_crmsa/seed42),
runs Stage-1 (per-modality RRTEncoder) + the Stage-2 internals by hand, and
reports ‖delta‖ / ‖z‖ for HE and PR across the Official Test slides.

If ‖delta‖ ≫ ‖z‖ the residual overwrites the Stage-1 features (bad); if ≪ 1 the
fusion is a near no-op (≈ plain concat).  ~O(1) is a normal residual update.

Usage:
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/diag_stage2_magnitude.py
"""
import os, sys, json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from train import build_feature_dirs
from models.mm_rrt_abmil import MM_RRT_ABMIL
from data.c16_multimodal_dataset import C16MultimodalDataset, c16_multimodal_collate_fn

CKPT = PROJECT / "results/c16_twostage_crmsa/seed42/ckpt/best_model.pt"
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
MODALITIES = ["HE", "PR"]
DIR_MAP = {"HE": "C16_HE_features", "PR": "C16_PR_features"}


def fro(x):
    return float(x.reshape(x.shape[0], -1).norm(dim=1).mean())


def main():
    ckpt = torch.load(str(CKPT), map_location="cuda:0", weights_only=False)
    mc = ckpt["config"]["model"]

    model = MM_RRT_ABMIL(
        num_modalities=2, input_dim=768, num_classes=2,
        mlp_dim=mc.get("mlp_dim", 512), region_num=mc.get("region_num", 4),
        n_layers=mc.get("n_layers", 2), n_heads=mc.get("n_heads", 4),
        drop_path=mc.get("drop_path", 0.0), trans_dropout=mc.get("trans_dropout", 0.1),
        epeg=mc.get("epeg", True), epeg_k=mc.get("epeg_k", 9),
        crmsa_k=mc.get("crmsa_k", 3), cr_msa=mc.get("cr_msa", True),
        all_shortcut=mc.get("all_shortcut", True),
        crmsa_heads=mc.get("crmsa_heads", 8), crmsa_mlp=mc.get("crmsa_mlp", False),
        fusion_type=mc.get("fusion_type", "two_stage_region"),
        stage2_type=mc.get("stage2_type", "staining_msa"),
        abmil_hidden_dim=mc.get("abmil_hidden_dim", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.cuda().eval()

    assert model.stage2_type == 'staining_msa', f"unexpected stage2_type={model.stage2_type}"
    mod = model.cross_region_mod

    feature_dirs = build_feature_dirs(FEATURE_BASE, MODALITIES, DIR_MAP)
    ds = C16MultimodalDataset(
        feature_dirs=feature_dirs,
        label_file=str(PROJECT / "data/C16_labels/c16_test_labels.csv"),
        max_patches=2500, preload=False, verbose=False,
        sampling="random", sample_seed=42, per_epoch=False,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    collate_fn=c16_multimodal_collate_fn, num_workers=0)

    rows = []
    with torch.no_grad():
        for b, batch in enumerate(dl):
            feats = [torch.stack(m).cuda() for m in batch["features"]]
            # projection (eval → dropout is identity)
            he_emb = model.patch_to_emb[0](feats[0])
            pr_emb = model.patch_to_emb[1](feats[1])
            # Stage 1
            z_he = model.rrt_he(he_emb)
            z_pr = model.rrt_ihc(pr_emb)
            if z_he.dim() == 2: z_he = z_he.unsqueeze(0)
            if z_pr.dim() == 2: z_pr = z_pr.unsqueeze(0)

            # Stage 2 internals (replicate CrossStainingCRMSA.forward)
            z_he_n = mod.norm(z_he)
            z_pr_n = mod.norm(z_pr)
            r_he = mod._combine(z_he_n)
            r_pr = mod._combine(z_pr_n)
            routing = torch.cat([r_he[0], r_pr[0]], dim=1)
            routing = mod.attn(routing)
            n_he = r_he[0].shape[1]
            delta_he = mod._dispatch(routing[:, :n_he], r_he[1], r_he[2], r_he[3], r_he[4], r_he[5], r_he[6])
            delta_pr = mod._dispatch(routing[:, n_he:], r_pr[1], r_pr[2], r_pr[3], r_pr[4], r_pr[5], r_pr[6])

            zhe_n, zpr_n = fro(z_he), fro(z_pr)
            dhe_n, dpr_n = fro(delta_he), fro(delta_pr)
            rows.append({
                "slide": batch["slide_ids"][0],
                "z_he": zhe_n, "z_pr": zpr_n,
                "delta_he": dhe_n, "delta_pr": dpr_n,
                "ratio_he": dhe_n / (zhe_n + 1e-8),
                "ratio_pr": dpr_n / (zpr_n + 1e-8),
            })
            if b < 3:
                print(f"[{batch['slide_ids'][0]}] "
                      f"‖z_he‖={zhe_n:.3f} ‖z_pr‖={zpr_n:.3f} | "
                      f"‖δ_he‖={dhe_n:.3f} ‖δ_pr‖={dpr_n:.3f} | "
                      f"δ/z_he={dhe_n/(zhe_n+1e-8):.3f} δ/z_pr={dpr_n/(zpr_n+1e-8):.3f}")

    r_he = np.array([r["ratio_he"] for r in rows])
    r_pr = np.array([r["ratio_pr"] for r in rows])
    z_he = np.array([r["z_he"] for r in rows])
    z_pr = np.array([r["z_pr"] for r in rows])

    def summ(name, v):
        print(f"{name:>14}: mean={v.mean():.3f} median={np.median(v):.3f} "
              f"min={v.min():.3f} max={v.max():.3f}")

    print("\n===== Stage2 delta magnitude (Official Test, n=%d slides) =====" % len(rows))
    summ("‖z_he‖", z_he)
    summ("‖z_pr‖", z_pr)
    summ("δ/z_he", r_he)
    summ("δ/z_pr", r_pr)


if __name__ == "__main__":
    main()
