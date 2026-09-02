#!/usr/bin/env python3
"""Task 3 — CR-MSA attention-matrix + ABMIL modality-attention diagnosis.

Diagnostic-only: registers read-only forward hooks on the trained two-stage
(staining_msa) checkpoint and reports the *actual* attention statistics of the
CrossStainingCRMSA stage.  Does NOT change any model math or weights.

Captured per slide (batch_size==1, eval mode):
  - CR-MSA routing-token attention map  A  [crmsa_k, n_heads, R, R]
      where R = n_he_regions + n_pr_regions (region_num^2 per side).
      The 4 blocks split at `split`:
          A_HH = A[:, :, :split, :split]      (HE query -> HE key)
          A_HP = A[:, :, :split, split:]      (HE query -> PR key)
          A_PH = A[:, :, split:, :split]      (PR query -> HE key)
          A_PP = A[:, :, split:, split:]      (PR query -> PR key)
  - residual delta  ||delta_joint|| / ||concat(z_he, z_pr)||   (Stage-2 change)
  - net change      ||z_final - concat|| / ||concat||
  - ABMIL token attention split over HE vs PR tokens

Statistics (all derived from the above, no preset answers):
  1. query-normalized 4-block attention mass
  2. key-count-normalized enrichment (mass / uniform-expectation; >1 = more than
     uniform, i.e. a cross-modal preference)
  3. per-head (n_heads) block masses
  4. grouped by normal/tumor and correct/incorrect
  5. Stage-2 residual delta ratio
  6. final ABMIL HE/PR attention share

Usage:
  python scripts/diag_crmsa_attention.py --config <cfg.json> --checkpoint <best_model.pt> --gpu 1
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

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
    # Drop dead 'rrt_encoder.' params (legacy module removed in two_stage_region).
    sd = {k: v for k, v in sd.items() if not k.startswith("rrt_encoder.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("rrt_encoder.")]
    if missing:
        print(f"[warn] missing keys: {missing}")
    return ck.get("epoch") if isinstance(ck, dict) else None


def block_mass(A, split):
    """A: [K, H, R, R] post-softmax.  Return 4-block query-normalized masses [K,H]."""
    HH = A[:, :, :split, :split].sum(-1).mean(-1)          # [K,H]
    HP = A[:, :, :split, split:].sum(-1).mean(-1)
    PH = A[:, :, split:, :split].sum(-1).mean(-1)
    PP = A[:, :, split:, split:].sum(-1).mean(-1)
    return HH, HP, PH, PP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["train", "test", "both"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os = __import__("os")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch  # noqa
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = json.loads(Path(args.config).read_text())
    from train import build_feature_dirs
    from data.c16_multimodal_dataset import C16MultimodalDataset

    model = build_model(cfg).to(device)
    epoch = load_checkpoint(model, args.checkpoint)
    model.eval()

    stage2_cfg = cfg["model"].get("stage2_cfg", {})
    region_num = stage2_cfg.get("region_num", cfg["model"].get("region_num", 8))
    split = region_num * region_num          # routing regions per modality
    n_heads = stage2_cfg.get("crmsa_heads", cfg["model"].get("crmsa_heads", 8))

    # ---- read-only capture hooks ----
    captured = {}

    def attn_hook(mod, inp, out):
        # out = post-softmax attention [K, n_heads, R, R]; inp[0] = pre-softmax
        captured["attn"] = out.detach().clone()
        captured["attn_raw"] = inp[0].detach().clone()

    def residual_hook(mod, inp, out):
        # inp[0] = delta_joint before DropPath/Identity
        captured["delta_joint"] = inp[0].detach().clone()

    def module_hook(mod, inp, out):
        z_list = inp[0]                      # [z_he, z_pr]
        captured["z_he"] = z_list[0].detach().clone()
        captured["z_pr"] = z_list[1].detach().clone()
        captured["z_final"] = out.detach().clone()

    model.cross_region_mod.attn.softmax.register_forward_hook(attn_hook)
    model.cross_region_mod.drop_path.register_forward_hook(residual_hook)
    model.cross_region_mod.register_forward_hook(module_hook)

    # ---- dataset ----
    dc = cfg["data"]
    feature_dirs = build_feature_dirs(dc["feature_base_dir"], dc["modalities"],
                                      dc.get("dir_mapping", None))
    label_file = dc["val_label_file"] if args.split in ("test",) else dc["train_label_file"]
    ds = C16MultimodalDataset(
        feature_dirs=feature_dirs, label_file=label_file,
        max_patches=dc.get("max_patches", 2500), preload=False, verbose=False,
        sampling=dc.get("sampling", "random"), sample_seed=dc.get("sample_seed", 0),
        per_epoch=False,
    )

    rows = []
    print(f"==> {len(ds)} slides ({args.split}), checkpoint epoch={epoch}, "
          f"region_num={region_num}, split={split}, heads={n_heads}")
    with torch.inference_mode():
        for i in range(len(ds)):
            item = ds[i]
            slide_id = item["slide_id"]; label = int(item["label"])
            feats = [item["features"][m].to(device).unsqueeze(0) for m in dc["modalities"]]
            out = model(feats)
            logits = out[0]
            abmil_attn = out[2]               # list of [N] tensors (one per batch item)

            prob = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
            pred = int(logits.argmax(dim=1).item())

            A = captured["attn"]              # [K, H, R, R]
            HH, HP, PH, PP = block_mass(A, split)
            K = A.shape[0]
            he_share = HH + HP                # per [K,H]: fraction of HE query to HE (==1) sanity
            # query-normalized 4-block mass (averaged over K routing channels + heads)
            mass = {b: float(v.mean()) for b, v in
                    zip(["HH", "HP", "PH", "PP"], [HH, HP, PH, PP])}
            # enrichment vs uniform: each block's expected share = (keys in block)/R = 0.5
            n_he_reg = split; n_pr_reg = split; R = A.shape[-1]
            enrich = {
                "HH": mass["HH"] / (n_he_reg / R),
                "HP": mass["HP"] / (n_pr_reg / R),
                "PH": mass["PH"] / (n_he_reg / R),
                "PP": mass["PP"] / (n_pr_reg / R),
            }

            # residual / net change (joint + per-modality)
            z_he_c = captured["z_he"]; z_pr_c = captured["z_pr"]
            concat = torch.cat([z_he_c, z_pr_c], dim=1)
            delta = captured["delta_joint"]
            zf = captured["z_final"]
            n_he_tok = z_he_c.shape[1]
            residual_ratio = float((delta.norm() / (concat.norm() + 1e-8)).item())
            net_ratio = float(((zf - concat).norm() / (concat.norm() + 1e-8)).item())
            he_net = float(((zf[:, :n_he_tok] - z_he_c).norm() / (z_he_c.norm() + 1e-8)).item())
            pr_net = float(((zf[:, n_he_tok:] - z_pr_c).norm() / (z_pr_c.norm() + 1e-8)).item())
            he_resid = float((delta[:, :n_he_tok].norm() / (z_he_c.norm() + 1e-8)).item())
            pr_resid = float((delta[:, n_he_tok:].norm() / (z_pr_c.norm() + 1e-8)).item())

            # ABMIL HE/PR token attention share
            if abmil_attn is not None:
                if isinstance(abmil_attn, (list, tuple)):
                    a = abmil_attn[0]
                else:
                    a = abmil_attn
                a = a.reshape(-1).float()
                n_he_tok = captured["z_he"].shape[1]
                abmil_he = float(a[:n_he_tok].sum() / (a.sum() + 1e-8))
                abmil_pr = float(a[n_he_tok:].sum() / (a.sum() + 1e-8))
            else:
                abmil_he = abmil_pr = float("nan")

            # per-head masses [H] -> store as lists
            per_head = {
                "HH": HH.mean(0).tolist(), "HP": HP.mean(0).tolist(),
                "PH": PH.mean(0).tolist(), "PP": PP.mean(0).tolist(),
            }

            rows.append({
                "slide_id": slide_id, "label": label, "pred": pred,
                "correct": int(pred == label),
                "prob_pos": float(prob[1]),
                "mass": mass, "enrichment": enrich,
                "per_head": per_head,
                "residual_ratio": residual_ratio, "net_ratio": net_ratio,
                "he_net": he_net, "pr_net": pr_net,
                "he_resid": he_resid, "pr_resid": pr_resid,
                "abmil_he": abmil_he, "abmil_pr": abmil_pr,
                "n_he_tok": int(captured["z_he"].shape[1]),
                "n_pr_tok": int(captured["z_pr"].shape[1]),
            })

    # ---- aggregation helpers ----
    def mean_of(sel, key):
        vals = [r[key] for r in sel]
        return float(np.mean(vals)) if vals else float("nan")

    def mean_mass(sel, block):
        return float(np.mean([r["mass"][block] for r in sel]))

    def mean_enrich(sel, block):
        return float(np.mean([r["enrichment"][block] for r in sel]))

    groups = {
        "all": rows,
        "normal": [r for r in rows if r["label"] == 0],
        "tumor": [r for r in rows if r["label"] == 1],
        "correct": [r for r in rows if r["correct"] == 1],
        "incorrect": [r for r in rows if r["correct"] == 0],
    }

    summary = {"per_slide": rows, "groups": {}}
    for gname, sel in groups.items():
        summary["groups"][gname] = {
            "n": len(sel),
            "mass": {b: mean_mass(sel, b) for b in ["HH", "HP", "PH", "PP"]},
            "enrichment": {b: mean_enrich(sel, b) for b in ["HH", "HP", "PH", "PP"]},
            "residual_ratio": mean_of(sel, "residual_ratio"),
            "net_ratio": mean_of(sel, "net_ratio"),
            "he_net": mean_of(sel, "he_net"),
            "pr_net": mean_of(sel, "pr_net"),
            "abmil_he": mean_of(sel, "abmil_he"),
            "abmil_pr": mean_of(sel, "abmil_pr"),
            "acc": float(np.mean([r["correct"] for r in sel])) if sel else float("nan"),
        }

    # per-head aggregate (over all slides): mean over slides of per-head masses
    per_head_agg = {b: [] for b in ["HH", "HP", "PH", "PP"]}
    for r in rows:
        for b in per_head_agg:
            per_head_agg[b].append(r["per_head"][b])
    summary["per_head"] = {b: np.mean(per_head_agg[b], axis=0).tolist()
                           for b in per_head_agg}

    print("\n" + "=" * 78)
    print("CR-MSA 4-block attention (query-normalized mass | enrichment vs uniform)")
    print("=" * 78)
    print(f"{'group':12s} {'n':>4s}  {'HH':>8s} {'HP':>8s} {'PH':>8s} {'PP':>8s}  "
          f"{'eHH':>6s} {'eHP':>6s} {'ePH':>6s} {'ePP':>6s}  {'resid':>6s} {'abmilHE':>8s}")
    for gname, g in summary["groups"].items():
        m, e = g["mass"], g["enrichment"]
        print(f"{gname:12s} {g['n']:4d}  {m['HH']:8.4f} {m['HP']:8.4f} "
              f"{m['PH']:8.4f} {m['PP']:8.4f}  {e['HH']:6.2f} {e['HP']:6.2f} "
              f"{e['PH']:6.2f} {e['PP']:6.2f}  {g['residual_ratio']:6.3f} "
              f"{g['abmil_he']:8.3f}")

    print("\nper-head mass (mean over slides), block = HH / HP / PH / PP:")
    hdr = "head: " + "  ".join(f"{h:>7d}" for h in range(n_heads))
    print(hdr)
    for b in ["HH", "HP", "PH", "PP"]:
        vals = summary["per_head"][b]
        print(f"  {b}: " + "  ".join(f"{v:7.4f}" for v in vals))

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}")
    print("DONE")


if __name__ == "__main__":
    main()
