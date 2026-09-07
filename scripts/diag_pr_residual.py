#!/usr/bin/env python3
"""Round 6 — quantify PR's residual information relative to HE.

Question: how discriminative is the part of PR that HE cannot (linearly) explain,
and does adding it to HE bring extra benefit?

Method (test-time only, no RRT/CR-MSA/ABMIL change, no re-extraction, no new main model):
  1. Load HE-only RRT and PR-only RRT checkpoints (same model seed).
  2. Extract slide-level embedding h ∈ R^512 for each slide = ABMIL attention-pooled
     vector Z (after pooling, before classifier), via the exact single-modality forward.
  3. On TRAIN only, fit g: h_PR ≈ Ridge(h_HE)  (alpha fixed; input standardized on train).
  4. residual  r_PR = h_PR - g(h_HE)   (same train-fitted g for train/val/test).
  5. Four identical linear probes (LogisticRegression, C chosen on VAL only):
        A. h_HE      B. h_PR      C. r_PR      D. concat(h_HE, r_PR)
  6. Report test AUC per probe + Δ = AUC(HE+residual) - AUC(HE).
  7. Aux on TEST: ||r_PR||/||h_PR||, cos(h_HE,h_PR), cos(h_HE,r_PR).

Splits (project convention, fixed 216/54 + 129 test):
  train = data/C16_labels/fixed_split/train.csv  (216)
  val   = data/C16_labels/fixed_split/val.csv    (54)
  test  = data/C16_labels/c16_test_labels.csv    (129)

Usage:
  python scripts/diag_pr_residual.py --gpu 1 --out results/pr_residual.json
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))

HE_CFG = "results/he_rrt_samplerfix_lr1e4/seed{seed}/config.json"
HE_CKPT = "results/he_rrt_samplerfix_lr1e4/seed{seed}/ckpt/best_model.pt"
PR_CFG = "results/c16_test_as_val/seed{seed}/config.json"
PR_CKPT = "results/c16_test_as_val/seed{seed}/ckpt/best_model.pt"

TRAIN_LABEL = PROJECT / "data/C16_labels/fixed_split/train.csv"
VAL_LABEL = PROJECT / "data/C16_labels/fixed_split/val.csv"
TEST_LABEL = PROJECT / "data/C16_labels/c16_test_labels.csv"

RIDGE_ALPHA = 1.0
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]


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
    model.load_state_dict(sd, strict=False)
    return ck.get("epoch") if isinstance(ck, dict) else None


def forward_embed(model, x, device):
    """Return (Z, logits) for single-modality model: Z = ABMIL attention-pooled
    slide embedding [512], logits [2].  Mirrors MM_RRT_ABMIL.forward num_modalities==1."""
    with torch.inference_mode():
        xb = x.unsqueeze(0).to(device)
        x_emb = model.patch_to_emb[0](xb)
        x_emb = model.dp(x_emb)
        z = model.rrt_he(x_emb)
        if len(z.shape) == 2:
            z = z.unsqueeze(0)
        mil_result = model.mil(z)
        A = mil_result['attention'][0]          # [N]
        logits = mil_result['logits']           # [1, 2]
        Z = A.unsqueeze(0) @ z.squeeze(0)       # [1, 512]
    return Z.squeeze(0).float().cpu(), logits.float().cpu()


def load_mod_features(slide_id, feat_dir, max_patches=2500, sample_seed=0):
    cat = slide_id.split('_')[0]
    p = Path(feat_dir) / cat / f"{slide_id}.pt"
    t = torch.load(str(p), map_location='cpu', weights_only=True)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    total = t.shape[0]
    if max_patches and total > max_patches:
        from data.c16_multimodal_dataset import stable_slide_seed
        rng = np.random.RandomState(stable_slide_seed(slide_id, sample_seed, 0))
        idx = rng.choice(total, max_patches, replace=False)
        idx.sort()
        t = t[idx]
    return t.float()


def read_labels(path):
    out = []
    for line in Path(path).read_text().splitlines()[1:]:
        sid, lbl = line.split(',')
        out.append((sid.strip(), int(lbl.strip())))
    return out


def fit_probe(Xtr, ytr, Xva, yva):
    """LogisticRegression with C chosen on val. Returns (best_C, val_auc, probe)."""
    best_C, best_auc, best_probe = None, -1.0, None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=5000, solver='lbfgs')
        clf.fit(Xtr, ytr)
        va_auc = roc_auc_score(yva, clf.predict_proba(Xva)[:, 1])
        if va_auc > best_auc:
            best_C, best_auc, best_probe = C, va_auc, clf
    return best_C, best_auc, best_probe


def cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a * b).sum(1)


def run_seed(seed, device):
    he_cfg = json.loads(Path(HE_CFG.format(seed=seed)).read_text())
    pr_cfg = json.loads(Path(PR_CFG.format(seed=seed)).read_text())

    he_model = build_model(he_cfg).to(device); he_model.eval()
    pr_model = build_model(pr_cfg).to(device); pr_model.eval()
    load_checkpoint(he_model, str(PROJECT / HE_CKPT.format(seed=seed)))
    load_checkpoint(pr_model, str(PROJECT / PR_CKPT.format(seed=seed)))

    base = he_cfg["data"]["feature_base_dir"]
    he_dir = f"{base}/C16_HE_features"
    pr_dir = f"{base}/C16_PR_features"
    sample_seed = he_cfg["data"].get("sample_seed", 0)

    # --- numerical equivalence: manual embed vs full model logits ---
    # use one slide's HE feature as a check
    check_slide = "normal_001"
    xc = load_mod_features(check_slide, he_dir, sample_seed=sample_seed)
    Zc, logits_manual = forward_embed(he_model, xc, device)
    out = he_model([xc.unsqueeze(0).to(device)])
    logits_model = out[0].float().cpu()
    max_diff = float((logits_model - logits_manual).abs().max())
    # also: classifier(Z) must equal manual logits
    with torch.inference_mode():
        logits_from_z = he_model.mil.classifier(Zc.unsqueeze(0).to(device)).float().cpu()
    max_diff_z = float((logits_from_z - logits_manual).abs().max())
    print(f"  [seed {seed}] embed equiv: logits(manual vs model) max_abs={max_diff:.3e} "
          f"| classifier(Z) vs manual logits max_abs={max_diff_z:.3e}")

    # --- collect embeddings for all slides ---
    label_map = {}
    split_of = {}   # slide_id -> 'train'/'val'/'test'
    for path, split in [(TRAIN_LABEL, 'train'), (VAL_LABEL, 'val'), (TEST_LABEL, 'test')]:
        for sid, lbl in read_labels(path):
            label_map[sid] = lbl
            split_of[sid] = split
    all_slides = sorted(split_of.keys())

    H_HE, H_PR, labels, splits = [], [], [], []
    for sid in all_slides:
        x_he = load_mod_features(sid, he_dir, sample_seed=sample_seed)
        x_pr = load_mod_features(sid, pr_dir, sample_seed=sample_seed)
        z_he, _ = forward_embed(he_model, x_he, device)
        z_pr, _ = forward_embed(pr_model, x_pr, device)
        H_HE.append(z_he.numpy()); H_PR.append(z_pr.numpy())
        labels.append(label_map[sid])
        splits.append(split_of[sid])
    H_HE = np.stack(H_HE).astype(np.float64)
    H_PR = np.stack(H_PR).astype(np.float64)
    labels = np.array(labels)
    splits = np.array(splits)

    tr = splits == 'train'; va = splits == 'val'; te = splits == 'test'

    # --- Ridge g: h_PR ≈ g(h_HE), fit on TRAIN only ---
    scaler_he = StandardScaler().fit(H_HE[tr])
    Xtr_std = scaler_he.transform(H_HE[tr])
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(Xtr_std, H_PR[tr])
    H_PR_hat_all = ridge.predict(scaler_he.transform(H_HE))   # same g for all
    R_PR = H_PR - H_PR_hat_all                                  # residual

    # --- probes (identical classifier, C on val); also record full C-sweep for diagnosis ---
    def probe_on(name, feat):
        scaler = StandardScaler().fit(feat[tr])
        Xtr, Xva, Xte = scaler.transform(feat[tr]), scaler.transform(feat[va]), scaler.transform(feat[te])
        C, va_auc, clf = fit_probe(Xtr, labels[tr], Xva, labels[va])
        te_auc = roc_auc_score(labels[te], clf.predict_proba(Xte)[:, 1])
        # full sweep (diagnostic only — see whether a different C recovers HE on the concat)
        sweep = []
        for c in C_GRID:
            clfc = LogisticRegression(C=c, max_iter=5000, solver='lbfgs').fit(Xtr, labels[tr])
            sweep.append({"C": c,
                          "val_auc": roc_auc_score(labels[va], clfc.predict_proba(Xva)[:, 1]),
                          "test_auc": roc_auc_score(labels[te], clfc.predict_proba(Xte)[:, 1])})
        return te_auc, va_auc, C, sweep

    he_te, he_va, he_C, he_sweep = probe_on('HE', H_HE)
    pr_te, pr_va, pr_C, pr_sweep = probe_on('PR', H_PR)
    rp_te, rp_va, rp_C, rp_sweep = probe_on('residual', R_PR)
    cat = np.concatenate([H_HE, R_PR], axis=1)
    hr_te, hr_va, hr_C, hr_sweep = probe_on('HE+residual', cat)
    # auxiliary: concat(HE, FULL PR) — since r_PR = PR - g(HE) is a linear reparam of
    # [HE, PR], a linear probe should behave the same; used to disambiguate probe artifacts.
    cat_full = np.concatenate([H_HE, H_PR], axis=1)
    hf_te, hf_va, hf_C, hf_sweep = probe_on('HE+PR(full)', cat_full)

    # --- aux on test ---
    norm_ratio = (np.linalg.norm(R_PR[te], axis=1) / (np.linalg.norm(H_PR[te], axis=1) + 1e-12)).mean()
    cos_he_pr = cos(H_HE[te], H_PR[te]).mean()
    cos_he_rp = cos(H_HE[te], R_PR[te]).mean()
    # fraction of PR variance explained by g (train, R^2-ish)
    ss_res = ((H_PR[tr] - H_PR_hat_all[tr]) ** 2).sum()
    ss_tot = ((H_PR[tr] - H_PR[tr].mean(0)) ** 2).sum()
    r2_train = 1 - ss_res / ss_tot

    return {
        "seed": seed,
        "embed_equiv_max_abs": max_diff, "embed_equiv_classifier_z": max_diff_z,
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "HE_auc": he_te, "HE_val_auc": he_va, "HE_C": he_C,
        "PR_auc": pr_te, "PR_val_auc": pr_va, "PR_C": pr_C,
        "residual_auc": rp_te, "residual_val_auc": rp_va, "residual_C": rp_C,
        "HE_residual_auc": hr_te, "HE_residual_val_auc": hr_va, "HE_residual_C": hr_C,
        "delta_vs_HE": hr_te - he_te,
        "residual_norm_ratio": float(norm_ratio),
        "cos_he_pr": float(cos_he_pr),
        "cos_he_residual": float(cos_he_rp),
        "r2_he_to_pr_train": float(r2_train),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs='*', default=[42, 123, 456])
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results = []
    for seed in args.seeds:
        r = run_seed(seed, device)
        results.append(r)
        print(f"  [seed {seed}] HE={r['HE_auc']:.4f} PR={r['PR_auc']:.4f} "
              f"res={r['residual_auc']:.4f} HE+res={r['HE_residual_auc']:.4f} "
              f"Δ={r['delta_vs_HE']:+.4f} | norm_ratio={r['residual_norm_ratio']:.3f} "
              f"cos(HE,PR)={r['cos_he_pr']:.3f} cos(HE,res)={r['cos_he_residual']:.3f} "
              f"R²(HE→PR)={r['r2_he_to_pr_train']:.3f}")

    # summary table
    print("\n" + "=" * 90)
    print(f"{'seed':>6} {'HE':>8} {'PR':>8} {'PR res':>8} {'HE+res':>8} {'Δ vs HE':>8} "
          f"{'||r||/||PR||':>12} {'cos(HE,PR)':>12} {'cos(HE,r)':>12} {'R² HE→PR':>10}")
    print("=" * 90)
    for r in results:
        print(f"{r['seed']:>6} {r['HE_auc']:>8.4f} {r['PR_auc']:>8.4f} {r['residual_auc']:>8.4f} "
              f"{r['HE_residual_auc']:>8.4f} {r['delta_vs_HE']:>+8.4f} "
              f"{r['residual_norm_ratio']:>12.3f} {r['cos_he_pr']:>12.3f} "
              f"{r['cos_he_residual']:>12.3f} {r['r2_he_to_pr_train']:>10.3f}")

    he = np.array([r['HE_auc'] for r in results])
    pr = np.array([r['PR_auc'] for r in results])
    res = np.array([r['residual_auc'] for r in results])
    hres = np.array([r['HE_residual_auc'] for r in results])
    dlt = np.array([r['delta_vs_HE'] for r in results])
    print("-" * 90)
    print(f"{'MEAN±STD':>6} {he.mean():>8.4f} {pr.mean():>8.4f} {res.mean():>8.4f} "
          f"{hres.mean():>8.4f} {dlt.mean():>+8.4f}")
    print(f"{'':>6} ±{he.std():.4f} ±{pr.std():.4f} ±{res.std():.4f} "
          f"±{hres.std():.4f} ±{dlt.std():.4f}")

    payload = {"ridge_alpha": RIDGE_ALPHA, "C_grid": C_GRID, "seeds": results,
               "mean": {"HE": float(he.mean()), "PR": float(pr.mean()),
                        "residual": float(res.mean()), "HE_residual": float(hres.mean()),
                        "delta": float(dlt.mean())},
               "std": {"HE": float(he.std()), "PR": float(pr.std()),
                       "residual": float(res.std()), "HE_residual": float(hres.std()),
                       "delta": float(dlt.std())}}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    print("DONE")


if __name__ == "__main__":
    main()
