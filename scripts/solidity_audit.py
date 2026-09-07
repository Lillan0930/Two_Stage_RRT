#!/usr/bin/env python3
"""Solidity audit for he_residual_cross (finalization check, no model changes).

Sections (run with --section):
  config        config-fairness diff across 9 runs
  arch          param counts + architecture invariants
  repro         re-run pure inference on 9 best checkpoints, compare saved AUC
  causal        disable_cross / residual_scale=0 / cross-slide PR derangement
  magnitude     rho + delta/Z norms over val-as-test per model seed
  grad          backward pass, per-module gradient norms
  bootstrap     paired bootstrap CI on saved per-WSI predictions
  seed123       seed123 outlier diagnosis

Outputs land in results/stage2_he_residual_cross_val_as_test/_audit/.
"""
import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_val_as_test"
AUDIT_DIR = RESULTS_ROOT / "_audit"
CONDITIONS = ["he_only", "staining_msa", "he_residual_cross"]
SEEDS = [42, 123, 456]

from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# model building (mirrors train.py::create_model)
# ---------------------------------------------------------------------------
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
    """he_list/pr_list: lists of tensors [N,768] aligned per slide (len = n_slides)."""
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


# ---------------------------------------------------------------------------
# config fairness diff
# ---------------------------------------------------------------------------
def section_config():
    out = {"runs": {}, "diff": {}}
    for c in CONDITIONS:
        out["runs"][c] = {}
        for s in SEEDS:
            out["runs"][c][s] = json.loads(
                (RESULTS_ROOT / c / f"seed{s}" / "config.json").read_text())

    def _strip(cfg):
        # remove fields that are intended to differ (seed) or are path/echo fields
        d = json.loads(json.dumps(cfg))
        for k in ["seeds", "output", "protocol"]:
            d.pop(k, None)
        d["environment"].pop("seed", None)
        d["data"].pop("sample_seed", None)
        return d

    # compare across conditions at a fixed seed (42)
    base = _strip(out["runs"]["he_only"][42])
    for c in ["staining_msa", "he_residual_cross"]:
        other = _strip(out["runs"][c][42])
        diffs = []
        keys = sorted(set(list(base.keys())) | set(list(other.keys())))
        for k in keys:
            if k not in base or k not in other:
                diffs.append({"key": k, "he_only": base.get(k, "<missing>"),
                              c: other.get(k, "<missing>")})
            elif base[k] != other[k]:
                diffs.append({"key": k, "he_only": base[k], c: other[k]})
        out["diff"][f"{c}_vs_he_only@seed42"] = diffs

    # staining_msa vs he_residual_cross
    a, b = _strip(out["runs"]["staining_msa"][42]), _strip(out["runs"]["he_residual_cross"][42])
    diffs = []
    for k in sorted(set(list(a.keys())) | set(list(b.keys()))):
        if k not in a or k not in b:
            diffs.append({"key": k, "staining_msa": a.get(k, "<missing>"),
                          "he_residual_cross": b.get(k, "<missing>")})
        elif a[k] != b[k]:
            diffs.append({"key": k, "staining_msa": a[k], "he_residual_cross": b[k]})
    out["diff"]["he_residual_cross_vs_staining_msa@seed42"] = diffs

    # across seeds within a condition: only seed-bearing fields should differ
    for c in CONDITIONS:
        cfgs = [_strip(out["runs"][c][s]) for s in SEEDS]
        seed_diffs = []
        for i in range(1, len(cfgs)):
            if cfgs[i] != cfgs[0]:
                seed_diffs.append(f"seed{SEEDS[i]} != seed{SEEDS[0]}")
        out["diff"][f"{c}_across_seeds"] = seed_diffs

    (AUDIT_DIR / "config_diff.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out["diff"], indent=2, default=str))
    return out


# ---------------------------------------------------------------------------
# architecture / params
# ---------------------------------------------------------------------------
def _params_per_prefix(model):
    counts = {}
    for name, p in model.named_parameters():
        prefix = name.split(".")[0]
        if prefix not in ("patch_to_emb",):
            prefix = name
        counts[name] = p.numel()
    # aggregate by module family
    fam = {}
    for name, p in model.named_parameters():
        if name.startswith("patch_to_emb.0"):
            f = "patch_to_emb.0 (HE proj)"
        elif name.startswith("patch_to_emb.1"):
            f = "patch_to_emb.1 (PR proj)"
        elif name.startswith("rrt_he."):
            f = "rrt_he (HE RRT)"
        elif name.startswith("rrt_ihc."):
            f = "rrt_ihc (PR RRT)"
        elif name.startswith("cross_region_mod."):
            f = "cross_region_mod (Stage2)"
        elif name.startswith("mil."):
            f = "mil (ABMIL)"
        else:
            f = name
        fam[f] = fam.get(f, 0) + p.numel()
    return fam


def section_arch():
    out = {}
    for c in CONDITIONS:
        seed_dir = RESULTS_ROOT / c / "seed42"
        cfg, model, ckpt = load_checkpoint_model(seed_dir, "cpu")
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fam = _params_per_prefix(model)
        out[c] = {"total": total, "trainable": trainable, "by_family": fam}
        if c == "he_residual_cross":
            m = model.cross_region_mod
            out[c]["invariants"] = {
                "rrt_he is rrt_ihc": model.rrt_he is model.rrt_ihc,
                "phi_he is phi_pr": m.phi_he is m.phi_pr,
                "route_norm_he is route_norm_pr": m.route_norm_he is m.route_norm_pr,
                "attn_norm_he is attn_norm_pr": m.attn_norm_he is m.attn_norm_pr,
                "residual_scale_is_buffer": "residual_scale" in dict(model.named_buffers()),
                "residual_scale_value": float(m.residual_scale),
                "residual_scale_in_params": any("residual_scale" in n for n, _ in model.named_parameters()),
                "disable_cross": bool(m.disable_cross),
                "drop_path_identity": isinstance(m.drop_path, torch.nn.Identity),
                "stage2_type": getattr(model, "stage2_type", None),
            }
            # Q/K/V projections exist
            out[c]["invariants"]["has_w_q_k_v_out"] = all(
                hasattr(m, a) for a in ["w_q", "w_k", "w_v", "w_out"])
            # no post-residual norm/ffn
            out[c]["invariants"]["no_out_norm"] = not hasattr(m, "out_norm")
            out[c]["invariants"]["no_ffn"] = not hasattr(m, "ffn") or m.__dict__.get("ffn", False) is False
    (AUDIT_DIR / "arch.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------
def section_repro(device):
    rows = []
    for c in CONDITIONS:
        for s in SEEDS:
            seed_dir = RESULTS_ROOT / c / f"seed{s}"
            cfg, model, ckpt = load_checkpoint_model(seed_dir, device)
            ds = build_val_dataset(cfg)
            labels = [smp["label"] for smp in ds.samples]
            he_list = [ds[i]["features"][cfg["data"]["modalities"][0]] for i in range(len(ds))]
            pr_list = None
            if len(cfg["data"]["modalities"]) > 1:
                pr_list = [ds[i]["features"]["PR"] for i in range(len(ds))]
            probs = run_inference(model, he_list, pr_list, device)
            saved = json.loads((seed_dir / "result.json").read_text())["auc"]
            recomputed = auc_from_probs(labels, probs)
            rows.append({
                "model": c, "seed": s, "saved_auc": saved,
                "recomputed_auc": recomputed,
                "diff": recomputed - saved,
                "pass": abs(recomputed - saved) < 1e-6,
                "checkpoint_epoch": ckpt.get("epoch"),
                "best_epoch_from_result": json.loads((seed_dir / "result.json").read_text())["best_epoch"],
            })
    (AUDIT_DIR / "repro.json").write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(f"{r['model']:20s} seed{r['seed']:>3d} saved={r['saved_auc']:.6f} "
              f"recomputed={r['recomputed_auc']:.6f} diff={r['diff']:+.2e} "
              f"PASS={r['pass']} ckpt_epoch={r['checkpoint_epoch']}")
    return rows


# ---------------------------------------------------------------------------
# magnitude helper: capture delta_patch / z_he for a slide
# ---------------------------------------------------------------------------
def compute_delta_stats(model, he, pr):
    """Return (delta_patch, z_he, n_valid_he_routes, n_valid_pr_routes)."""
    from models.mm_rrt_abmil import MM_RRT_ABMIL  # noqa
    m = model.cross_region_mod
    with torch.inference_mode():
        z_he = model.rrt_he(model.patch_to_emb[0](model.dp(he)))
        z_ihc = model.rrt_ihc(model.patch_to_emb[1](model.dp(pr)))
        if z_he.dim() == 2:
            z_he = z_he.unsqueeze(0)
        if z_ihc.dim() == 2:
            z_ihc = z_ihc.unsqueeze(0)
        B = z_he.shape[0]
        routing_he, dmm_he, dw_he, vhs, H_he, W_he, add_he, rs_he = \
            m._route(z_he, m.phi_he, m.route_norm_he, None)
        routing_pr, dmm_pr, dw_pr, vps, H_pr, W_pr, add_pr, rs_pr = \
            m._route(z_ihc, m.phi_pr, m.route_norm_pr, None)
        r_he = routing_he.reshape(B, -1, m.dim)
        r_pr = routing_pr.reshape(B, -1, m.dim)
        q_valid = vhs.reshape(B, -1)
        k_valid = vps.reshape(B, -1)
        delta_routing = m._cross_attention(r_he, r_pr, q_valid, k_valid)
        delta_routing = delta_routing.view(B, -1, m.crmsa_k, m.dim)
        delta_patch = m._dispatch(delta_routing, dmm_he, dw_he, rs_he, H_he, W_he, add_he)
        n_he_routes = int(q_valid.sum().item())
        n_pr_routes = int(k_valid.sum().item())
    return delta_patch.squeeze(0), z_he.squeeze(0), n_he_routes, n_pr_routes


def section_magnitude(device):
    out = {}
    for s in SEEDS:
        seed_dir = RESULTS_ROOT / "he_residual_cross" / f"seed{s}"
        cfg, model, ckpt = load_checkpoint_model(seed_dir, device)
        ds = build_val_dataset(cfg)
        rhos, dnorms, znorms, nhe, npr = [], [], [], [], []
        for i in range(len(ds)):
            he = ds[i]["features"]["HE"].to(device).unsqueeze(0)
            pr = ds[i]["features"]["PR"].to(device).unsqueeze(0)
            dp, zh, nhr, npr_ = compute_delta_stats(model, he, pr)
            scaled = 0.1 * dp
            rho = scaled.norm().item() / (zh.norm().item() + 1e-8)
            rhos.append(rho)
            dnorms.append(dp.norm().item())
            znorms.append(zh.norm().item())
            nhe.append(nhr)
            npr.append(npr_)
        rhos = np.array(rhos)
        out[s] = {
            "rho": _percentiles(rhos),
            "delta_norm": _percentiles(np.array(dnorms)),
            "z_he_norm": _percentiles(np.array(znorms)),
            "valid_he_routes": _percentiles(np.array(nhe, dtype=float)),
            "valid_pr_routes": _percentiles(np.array(npr, dtype=float)),
        }
    (AUDIT_DIR / "magnitude.json").write_text(json.dumps(out, indent=2) + "\n")
    for s in SEEDS:
        print(f"seed{s} rho: {out[s]['rho']}")
        print(f"seed{s} delta_norm: {out[s]['delta_norm']}")
        print(f"seed{s} z_norm: {out[s]['z_he_norm']}")
        print(f"seed{s} valid_he_routes: {out[s]['valid_he_routes']}")
        print(f"seed{s} valid_pr_routes: {out[s]['valid_pr_routes']}")
    return out


def _percentiles(a):
    return {"mean": float(np.mean(a)), "std": float(np.std(a)),
            "p10": float(np.percentile(a, 10)), "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "min": float(np.min(a)), "max": float(np.max(a))}


# ---------------------------------------------------------------------------
# causal
# ---------------------------------------------------------------------------
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
    REPLACEMENT_SEEDS = [0, 1, 2, 3, 4]
    out = {}
    for s in SEEDS:
        seed_dir = RESULTS_ROOT / "he_residual_cross" / f"seed{s}"
        cfg, model, ckpt = load_checkpoint_model(seed_dir, device)
        ds = build_val_dataset(cfg)
        n = len(ds)
        labels = [smp["label"] for smp in ds.samples]
        he_list = [ds[i]["features"]["HE"] for i in range(n)]
        pr_list = [ds[i]["features"]["PR"] for i in range(n)]

        # Full
        full_probs = run_inference(model, he_list, pr_list, device)
        full_auc = auc_from_probs(labels, full_probs)

        # disable_cross
        model.cross_region_mod.disable_cross = True
        dc_probs = run_inference(model, he_list, pr_list, device)
        dc_auc = auc_from_probs(labels, dc_probs)
        model.cross_region_mod.disable_cross = False

        # residual_scale = 0
        model.cross_region_mod.residual_scale.fill_(0.0)
        rs0_probs = run_inference(model, he_list, pr_list, device)
        rs0_auc = auc_from_probs(labels, rs0_probs)
        model.cross_region_mod.residual_scale.fill_(0.1)
        def _logit(p):
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return np.log(p / (1 - p))
        max_logit_diff = float(np.max(np.abs(_logit(dc_probs) - _logit(rs0_probs))))

        # derangement
        deranged = {}
        for rs in REPLACEMENT_SEEDS:
            rng = np.random.RandomState(rs)
            perm = _derangement(n, rng)
            pr_swapped = [pr_list[perm[i]] for i in range(n)]
            # patch-count consistent subset
            consistent_idx = [i for i in range(n)
                              if pr_swapped[i].shape[0] == pr_list[i].shape[0]]
            der_probs = run_inference(model, he_list, pr_swapped, device)
            der_auc = auc_from_probs(labels, der_probs)
            der_auc_consistent = auc_from_probs(
                [labels[i] for i in consistent_idx],
                np.concatenate([der_probs[i:i + 1] for i in consistent_idx]))
            deranged[rs] = {
                "replacement_seed": rs,
                "auc_all": der_auc,
                "auc_patchcount_consistent": der_auc_consistent,
                "n_consistent": len(consistent_idx),
                "perm": perm.tolist(),
            }

        out[s] = {
            "full_auc": full_auc,
            "disable_cross_auc": dc_auc,
            "residual_scale0_auc": rs0_auc,
            "max_abs_logit_diff_dc_vs_rs0": max_logit_diff,
            "full_minus_disable_cross": full_auc - dc_auc,
            "derangement": deranged,
        }
    (AUDIT_DIR / "causal.json").write_text(json.dumps(out, indent=2) + "\n")
    for s in SEEDS:
        r = out[s]
        print(f"seed{s}: full={r['full_auc']:.4f} disable_cross={r['disable_cross_auc']:.4f} "
              f"rs0={r['residual_scale0_auc']:.4f} max_logit_diff={r['max_abs_logit_diff_dc_vs_rs0']:.2e}")
        for rs in REPLACEMENT_SEEDS:
            d = r["derangement"][rs]
            print(f"  repl_seed{rs}: auc_all={d['auc_all']:.4f} "
                  f"auc_consistent({d['n_consistent']})={d['auc_patchcount_consistent']:.4f}")
    return out


# ---------------------------------------------------------------------------
# gradient
# ---------------------------------------------------------------------------
def section_grad(device):
    seed_dir = RESULTS_ROOT / "he_residual_cross" / "seed42"
    cfg, model, ckpt = load_checkpoint_model(seed_dir, device)
    model.train()
    # synthetic small valid bag (all valid)
    B, N, D = 1, 196, 768
    he = torch.randn(B, N, D, device=device)
    pr = torch.randn(B, N, D, device=device)
    out = model([he, pr])
    logits = out[0]
    loss = logits.sum()
    model.zero_grad()
    loss.backward()

    groups = {
        "HE RRT (rrt_he)": "rrt_he.",
        "PR RRT (rrt_ihc)": "rrt_ihc.",
        "phi_he": "cross_region_mod.phi_he",
        "phi_pr": "cross_region_mod.phi_pr",
        "W_q_he (w_q)": "cross_region_mod.w_q",
        "W_k_pr (w_k)": "cross_region_mod.w_k",
        "W_v_pr (w_v)": "cross_region_mod.w_v",
        "W_out": "cross_region_mod.w_out",
        "ABMIL (mil)": "mil.",
    }
    grad_norms = {}
    none_grads = []
    for gname, prefix in groups.items():
        tot = 0.0
        n_params = 0
        for name, p in model.named_parameters():
            if name.startswith(prefix):
                if p.grad is None:
                    none_grads.append(name)
                else:
                    tot += float((p.grad ** 2).sum().item())
                    n_params += 1
        grad_norms[gname] = {"l2_norm": float(np.sqrt(tot)), "n_params_with_grad": n_params}
    # all main-path params with grad=None
    all_none = [n for n, p in model.named_parameters() if p.grad is None and p.requires_grad]
    out = {"grad_norms": grad_norms, "params_with_grad_none": all_none}
    (AUDIT_DIR / "grad.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(grad_norms, indent=2))
    print("grad=None trainable params:", all_none)
    return out


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def section_bootstrap():
    def load_preds(c, s):
        p = RESULTS_ROOT / c / f"seed{s}" / "test_predictions.csv"
        lines = (p.read_text().strip().split("\n"))[1:]
        slide, label, prob = [], [], []
        for ln in lines:
            parts = ln.split(",")
            slide.append(parts[0])
            label.append(int(parts[1]))
            prob.append(float(parts[2]))
        order = np.argsort(slide)
        slide = np.array(slide)[order]
        label = np.array(label)[order]
        prob = np.array(prob)[order]
        return slide, label, prob

    def bootstrap_delta_auc(labels, pA, pB, n_boot=5000, seed=0):
        rng = np.random.RandomState(seed)
        n = len(labels)
        deltas = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            la, lb = labels[idx], labels[idx]
            # guard single-class resample
            if len(np.unique(la)) < 2 or len(np.unique(lb)) < 2:
                continue
            a = roc_auc_score(la, pA[idx])
            b = roc_auc_score(lb, pB[idx])
            deltas.append(a - b)
        deltas = np.array(deltas)
        return {
            "mean_delta": float(np.mean(deltas)),
            "ci_low": float(np.percentile(deltas, 2.5)),
            "ci_high": float(np.percentile(deltas, 97.5)),
            "n_valid": len(deltas),
        }

    out = {}
    for s in SEEDS:
        _, lbl, p_he = load_preds("he_only", s)
        _, _, p_joint = load_preds("staining_msa", s)
        _, _, p_rcx = load_preds("he_residual_cross", s)
        # verify slide alignment
        s_he, _, _ = load_preds("he_only", s)
        s_joint, _, _ = load_preds("staining_msa", s)
        s_rcx, _, _ = load_preds("he_residual_cross", s)
        assert np.array_equal(s_he, s_joint) and np.array_equal(s_he, s_rcx), "slide misalignment"
        out[s] = {
            "rcx_vs_he": bootstrap_delta_auc(lbl, p_rcx, p_he),
            "rcx_vs_joint": bootstrap_delta_auc(lbl, p_rcx, p_joint),
        }
    (AUDIT_DIR / "bootstrap.json").write_text(json.dumps(out, indent=2) + "\n")
    for s in SEEDS:
        print(f"seed{s} rcx-vs-he:  {out[s]['rcx_vs_he']}")
        print(f"seed{s} rcx-vs-joint: {out[s]['rcx_vs_joint']}")
    return out


# ---------------------------------------------------------------------------
# seed123 diagnosis
# ---------------------------------------------------------------------------
def section_seed123():
    out = {}
    # best epochs + val AUC progression from logs
    for c in CONDITIONS:
        out[c] = {}
        for s in SEEDS:
            r = json.loads((RESULTS_ROOT / c / f"seed{s}" / "result.json").read_text())
            log = (RESULTS_ROOT / c / f"seed{s}" / "logs" / "run.log").read_text()
            epochs = []
            for line in log.splitlines():
                if "Epoch" in line and "Val" in line:
                    epochs.append(line.strip())
            out[c][s] = {"best_epoch": r["best_epoch"], "auc": r["auc"],
                         "n_epoch_lines": len(epochs)}
    # per-WSI prediction deltas (rcx - he) for seed123 vs other seeds
    def load_prob(c, s):
        p = RESULTS_ROOT / c / f"seed{s}" / "test_predictions.csv"
        lines = (p.read_text().strip().split("\n"))[1:]
        d = {}
        for ln in lines:
            parts = ln.split(",")
            d[parts[0]] = (int(parts[1]), float(parts[2]))
        return d
    out["rcx_minus_he_per_wsi"] = {}
    for s in SEEDS:
        he = load_prob("he_only", s)
        rcx = load_prob("he_residual_cross", s)
        deltas = []
        for sid in sorted(he.keys()):
            if sid in rcx:
                deltas.append((sid, he[sid][1], rcx[sid][1], rcx[sid][1] - he[sid][1]))
        deltas.sort(key=lambda x: -abs(x[3]))
        out["rcx_minus_he_per_wsi"][s] = {
            "top5_abs_delta": [(sid, round(he_, 4), round(rcx_, 4), round(d, 4))
                               for sid, he_, rcx_, d in deltas[:5]],
            "n_large_gt_0.2": sum(1 for _, _, _, d in deltas if abs(d) > 0.2),
        }
    (AUDIT_DIR / "seed123.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2, default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True,
                    choices=["config", "arch", "repro", "causal", "magnitude",
                             "grad", "bootstrap", "seed123", "all"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")

    if args.section in ("config", "all"):
        section_config()
    if args.section in ("arch", "all"):
        section_arch()
    if args.section in ("repro", "all"):
        section_repro(device)
    if args.section in ("causal", "all"):
        section_causal(device)
    if args.section in ("magnitude", "all"):
        section_magnitude(device)
    if args.section in ("grad", "all"):
        section_grad(device)
    if args.section in ("bootstrap", "all"):
        section_bootstrap()
    if args.section in ("seed123", "all"):
        section_seed123()


if __name__ == "__main__":
    main()
