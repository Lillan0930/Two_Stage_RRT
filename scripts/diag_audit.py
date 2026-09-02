#!/usr/bin/env python3
"""Audit diagnostic: init passes, dead module, grad=None, shapes, gradient flow.

Runs one forward+backward on the exact two_stage_region/staining_msa model config
(HE/PR), using synthetic [1, N, 768] features. Reports:

  A. initialize_weights call count + total module visits (double-iteration proof)
  B. dead module (self.rrt_encoder) param count + names
  C. forward tensor shapes at every stage
  D. per-module gradient norms (rrt_he / rrt_ihc / cross_region_mod / mil)
  E. full list of params with grad=None (the "38")

Usage: python scripts/diag_audit.py
"""
import os, sys, json, math
from pathlib import Path

import torch
import torch.nn as nn

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

# ---- patch initialize_weights BEFORE importing model modules ----
import models.mm_rrt_encoder as mm_enc
import models.abmil as abmil

_call_count = {"mm_rrt_encoder": 0, "abmil": 0}
_visit_count = {"mm_rrt_encoder": 0, "abmil": 0}


def _make_wrapper(fn, key):
    def wrapper(module):
        _call_count[key] += 1
        _visit_count[key] += sum(1 for _ in module.modules())
        return fn(module)
    return wrapper


mm_enc.initialize_weights = _make_wrapper(mm_enc.initialize_weights, "mm_rrt_encoder")
abmil.initialize_weights = _make_wrapper(abmil.initialize_weights, "abmil")

# Now import the model (mm_rrt_abmil does `from models.mm_rrt_encoder import initialize_weights`,
# which now resolves to our wrapper object)
from models.mm_rrt_abmil import MM_RRT_ABMIL

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


def build_model():
    return MM_RRT_ABMIL(
        num_modalities=2, modality_list=["HE", "PR"],
        input_dim=768, mlp_dim=512, num_classes=2,
        dropout=0.25, region_num=4, n_layers=2, n_heads=4,
        drop_path=0.0, trans_dropout=0.1, epeg=True, epeg_k=9,
        crmsa_k=3, cr_msa=True, all_shortcut=True,
        crmsa_heads=8, crmsa_mlp=False,
        fusion_type="two_stage_region", fusion_stage="middle",
        stage2_type="staining_msa",
        use_gated_fusion=False, abmil_hidden_dim=256,
        use_mclc=False, aggregate_modalities=True,
        encoder_cfg=STAGE1_ENCODER_CFG, stage2_cfg=STAGE2_CFG,
    )


def grad_norm(mod):
    total = 0.0
    for p in mod.parameters():
        if p.grad is not None:
            total += (p.grad.norm().item() ** 2)
    return math.sqrt(total)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 72)
    print("[A] initialize_weights call count / module visits")
    print(f"    mm_rrt_encoder.initialize_weights: calls={_call_count['mm_rrt_encoder']} "
          f"visits={_visit_count['mm_rrt_encoder']}")
    print(f"    abmil.initialize_weights:         calls={_call_count['abmil']} "
          f"visits={_visit_count['abmil']}")
    print(f"    total module re-init events: {_visit_count['mm_rrt_encoder'] + _visit_count['abmil']}")
    print(f"    total model params: {total_params:,}  trainable: {trainable:,}")

    print("=" * 72)
    print("[B] dead module self.rrt_encoder (MM_RRTEncoder)")
    if model.rrt_encoder is None:
        print("    rrt_encoder is None — dead module REMOVED (fix applied)")
        dead = []
        dead_params = 0
    else:
        dead = [n for n, _ in model.named_parameters() if n.startswith("rrt_encoder.")]
        dead_params = sum(model.get_parameter(n).numel() for n in dead)
        print(f"    param count: {len(dead)}  numel: {dead_params:,}")
        print(f"    sample names: {dead[:5]} ... {dead[-2:]}")

    print("=" * 72)
    print("[C] forward tensor shapes (synthetic [1, 2500, 768])")
    N = 2500
    he = torch.randn(1, N, 768, device=device)
    pr = torch.randn(1, N, 768, device=device)
    model = model.to(device)
    model.train()

    # instrument forward stages by monkeypatching the submodules with hooks
    shapes = {}

    def hook_factory(key):
        def hook(m, inp, out):
            if isinstance(out, dict):
                out = out['logits']
            elif isinstance(out, tuple):
                out = out[0]
            shapes[key] = tuple(out.shape)
        return hook

    model.rrt_he.register_forward_hook(hook_factory("rrt_he_out"))
    model.rrt_ihc.register_forward_hook(hook_factory("rrt_ihc_out"))
    model.cross_region_mod.register_forward_hook(hook_factory("cross_region_mod_out"))
    model.mil.register_forward_hook(hook_factory("mil_out"))
    model.patch_to_emb[0].register_forward_hook(hook_factory("patch_to_emb_0_out"))
    model.patch_to_emb[1].register_forward_hook(hook_factory("patch_to_emb_1_out"))

    out = model([he, pr])
    logits = out[0]
    print(f"    input he: {tuple(he.shape)}  pr: {tuple(pr.shape)}")
    for k in ["patch_to_emb_0_out", "patch_to_emb_1_out", "rrt_he_out", "rrt_ihc_out",
              "cross_region_mod_out", "mil_out"]:
        print(f"    {k}: {shapes.get(k)}")
    print(f"    logits: {tuple(logits.shape)}")

    print("=" * 72)
    print("[D] gradient flow (backward on CE loss)")
    label = torch.tensor([1], device=device)
    loss = nn.CrossEntropyLoss()(logits, label)
    loss.backward()

    active = {
        "he_projection": model.patch_to_emb[0],
        "pr_projection": model.patch_to_emb[1],
        "he_rrt": model.rrt_he,
        "pr_rrt": model.rrt_ihc,
        "stage2": model.cross_region_mod,
        "abmil": model.mil,
    }
    if model.rrt_encoder is not None:
        active["DEAD_rrt_encoder"] = model.rrt_encoder
    for name, mod in active.items():
        gn = grad_norm(mod)
        n_params = sum(1 for _ in mod.parameters())
        print(f"    {name:22s} grad_norm={gn:.6f}  (params={n_params})")

    print("=" * 72)
    print("[E] params with grad=None")
    unused = [(n, p.numel()) for n, p in model.named_parameters() if p.grad is None]
    print(f"    total grad=None: {len(unused)}")
    for n, num in unused:
        print(f"      {n}  ({num})")

    print("=" * 72)
    print("DONE")


if __name__ == "__main__":
    main()
