#!/usr/bin/env python3
"""Smoke test after audit fixes (section 15).

Test 1: construct 4 variants — HE-only / PR-only / two-stage staining_msa /
        two-stage concat
Test 2: forward shapes
Test 3: backward — every active module receives gradient
Test 4: optimizer.step — every active module's params actually update
Test 5: fixed-seed stability — same seed → identical forward logits
Test 6: metrics — binary sensitivity != specificity (tumor-positive)

Usage: python scripts/smoke_test_audit.py
"""
import os, sys, json, math
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from models.mm_rrt_abmil import MM_RRT_ABMIL
from utils.metrics import calculate_metrics

STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4, "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": True,
}

BASE = dict(input_dim=768, mlp_dim=512, num_classes=2, dropout=0.25,
            region_num=4, n_layers=2, n_heads=4, drop_path=0.0, trans_dropout=0.1,
            epeg=True, epeg_k=9, crmsa_k=3, cr_msa=True, all_shortcut=True,
            crmsa_heads=8, crmsa_mlp=False, fusion_type="two_stage_region",
            fusion_stage="middle", use_gated_fusion=False, abmil_hidden_dim=256,
            use_mclc=False, aggregate_modalities=True)


def build(num_modalities, modality_list, stage2_type="staining_msa", encoder_cfg=None, stage2_cfg=None):
    kw = dict(BASE)
    kw.update(num_modalities=num_modalities, modality_list=modality_list)
    if num_modalities > 1:
        kw["stage2_type"] = stage2_type
        kw["encoder_cfg"] = encoder_cfg
        kw["stage2_cfg"] = stage2_cfg
    return MM_RRT_ABMIL(**kw)


def grad_norm(mod):
    return math.sqrt(sum((p.grad.norm().item() ** 2) for p in mod.parameters() if p.grad is not None))


def active_modules(model):
    mods = {"he_projection": model.patch_to_emb[0],
            "he_rrt": model.rrt_he,
            "abmil": model.mil}
    if model.num_modalities > 1:
        mods["pr_projection"] = model.patch_to_emb[1]
        mods["pr_rrt"] = model.rrt_ihc
        if model.cross_region_mod is not None:
            mods["stage2"] = model.cross_region_mod
    return mods


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N = 64

    variants = {
        "HE-only": build(1, ["HE"]),
        "PR-only": build(1, ["PR"]),
        "two-stage staining_msa": build(2, ["HE", "PR"], "staining_msa",
                                         STAGE1_ENCODER_CFG, STAGE2_CFG),
        "two-stage concat": build(2, ["HE", "PR"], "concat",
                                  STAGE1_ENCODER_CFG, STAGE2_CFG),
    }

    print("=" * 72)
    print("[Test 1+2] construct + forward shapes")
    for name, m in variants.items():
        m = m.to(device)
        m.train()
        if m.num_modalities == 1:
            x = [torch.randn(1, N, 768, device=device)]
        else:
            x = [torch.randn(1, N, 768, device=device) for _ in range(2)]
        out = m(x)
        logits = out[0]
        nparams = sum(p.numel() for p in m.parameters())
        print(f"  {name:24s} logits={tuple(logits.shape)}  params={nparams:,}")

    print("=" * 72)
    print("[Test 3] backward — every active module receives gradient")
    for name, m in variants.items():
        m.zero_grad()
        if m.num_modalities == 1:
            x = [torch.randn(1, N, 768, device=device)]
        else:
            x = [torch.randn(1, N, 768, device=device) for _ in range(2)]
        logits = m(x)[0]
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([1], device=device))
        loss.backward()
        gs = {k: grad_norm(v) for k, v in active_modules(m).items()}
        all_pos = all(g > 0 for g in gs.values())
        print(f"  {name:24s} all_grad>0={all_pos}  { {k: round(v,3) for k,v in gs.items()} }")

    print("=" * 72)
    print("[Test 4] optimizer.step — every active module's params update")
    for name, m in variants.items():
        m.zero_grad()
        if m.num_modalities == 1:
            x = [torch.randn(1, N, 768, device=device)]
        else:
            x = [torch.randn(1, N, 768, device=device) for _ in range(2)]
        logits = m(x)[0]
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([1], device=device))
        loss.backward()
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        # snapshot one representative weight per active module
        reps = {}
        for k, mod in active_modules(m).items():
            reps[k] = next(mod.parameters()).detach().clone()
        opt.step()
        updated = {}
        for k, mod in active_modules(m).items():
            updated[k] = not torch.equal(reps[k], next(mod.parameters()))
        all_updated = all(updated.values())
        print(f"  {name:24s} all_updated={all_updated}")

    print("=" * 72)
    print("[Test 5] fixed-seed stability (two-stage staining_msa)")
    def make_and_fwd(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = build(2, ["HE", "PR"], "staining_msa", STAGE1_ENCODER_CFG, STAGE2_CFG).to(device)
        m.eval()
        x = [torch.randn(1, N, 768, device=device) for _ in range(2)]
        with torch.no_grad():
            return m(x)[0].detach().cpu()
    a = make_and_fwd(42)
    b = make_and_fwd(42)
    c = make_and_fwd(123)
    print(f"  same seed (42 vs 42) identical: {torch.allclose(a, b)}")
    print(f"  diff seed (42 vs 123) different: {not torch.allclose(a, c)}")

    print("=" * 72)
    print("[Test 6] metrics — binary sensitivity != specificity")
    y_true = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
    y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]
    y_prob = np.array([[1 - 0.1, 0.1], [1 - 0.2, 0.2], [1 - 0.7, 0.7],
                       [1 - 0.4, 0.4], [1 - 0.8, 0.8], [1 - 0.9, 0.9],
                       [1 - 0.6, 0.6], [1 - 0.3, 0.3], [1 - 0.1, 0.1],
                       [1 - 0.5, 0.5], [1 - 0.85, 0.85], [1 - 0.2, 0.2]])
    m = calculate_metrics(y_true, y_pred, y_prob, num_classes=2)
    for k in ["sensitivity", "specificity", "sensitivity_class_1", "specificity_class_1",
              "sensitivity_macro", "specificity_macro", "auc"]:
        print(f"  {k:24s} = {m[k]}")
    print(f"  sensitivity != specificity : {abs(m['sensitivity'] - m['specificity']) > 1e-9}")

    print("=" * 72)
    print("SMOKE TEST DONE")


if __name__ == "__main__":
    main()
