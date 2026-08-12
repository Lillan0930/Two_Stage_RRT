#!/usr/bin/env python3
"""
Verify the new official-style CrossStainingCRMSA (Stage 2).

Checks:
  1. Tensor shapes: Stage1 HE/PR [B,N,512] → routing tokens [B, 2*K*crmsa_k, 512]
     → output HE [B,N,512], PR [B,N,512] → final [B, 2N, 512]
  2. HE and PR with DIFFERENT patch counts run without error
  3. Current experiment params: region_num=4, crmsa_k=3, crmsa_heads=8
  4. Gradients flow (module is trainable)

Usage:
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/verify_cross_staining_crmsa.py
"""
import os, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

import torch
from models.cross_staining_crmsa import CrossStainingCRMSA


def main():
    dim = 512
    region_num = 4
    crmsa_k = 3
    crmsa_heads = 8
    K = region_num ** 2          # 16 regions per staining
    total_tokens = 2 * K * crmsa_k  # 96

    mod = CrossStainingCRMSA(
        dim=dim, num_heads=crmsa_heads, region_num=region_num, crmsa_k=crmsa_k,
        drop_out=0.1, drop_path=0.0, epeg=True, epeg_k=9,
    )
    print(f"region_num={region_num}  crmsa_k={crmsa_k}  crmsa_heads={crmsa_heads}")
    print(f"K={K} regions/staining, routing tokens total = 2*K*crmsa_k = {total_tokens}\n")

    # ── Case 1: same patch count ──
    B, N = 2, 100
    z_he = torch.randn(B, N, dim)
    z_pr = torch.randn(B, N, dim)
    print("[Same N]")
    print(f"  Stage1 HE: {tuple(z_he.shape)}   PR: {tuple(z_pr.shape)}")
    out = mod([z_he, z_pr])
    print(f"  Stage2 final (concat): {tuple(out.shape)}")
    assert out.shape == (B, 2 * N, dim), out.shape

    # ── Case 2: different patch count ──
    N_he, N_pr = 100, 144
    z_he = torch.randn(B, N_he, dim)
    z_pr = torch.randn(B, N_pr, dim)
    print("\n[Different N]")
    print(f"  Stage1 HE: {tuple(z_he.shape)}   PR: {tuple(z_pr.shape)}")
    out = mod([z_he, z_pr])
    print(f"  Stage2 final (concat): {tuple(out.shape)}")
    assert out.shape == (B, N_he + N_pr, dim), out.shape

    # ── Routing-token shape check (logical view [B, 2*K*crmsa_k, D]) ──
    print("\n[Routing tokens]")
    with torch.no_grad():
        z_he_n = mod.norm(z_he)
        z_pr_n = mod.norm(z_pr)
        r_he = mod._combine(z_he_n)[0]            # [crmsa_k, nW*B, C]
        r_pr = mod._combine(z_pr_n)[0]
        routing = torch.cat([r_he, r_pr], dim=1)  # [crmsa_k, 2*nW*B, C]
        logical = routing.permute(1, 0, 2).reshape(B, total_tokens, dim)
        print(f"  region routing tokens: {tuple(logical.shape)}")
        assert logical.shape == (B, total_tokens, dim), logical.shape

    # ── Gradients ──
    print("\n[Gradient flow]")
    mod2 = CrossStainingCRMSA(
        dim=dim, num_heads=crmsa_heads, region_num=region_num, crmsa_k=crmsa_k,
        drop_out=0.1, drop_path=0.0, epeg=True, epeg_k=9)
    z_he = torch.randn(B, N_he, dim, requires_grad=True)
    z_pr = torch.randn(B, N_pr, dim, requires_grad=True)
    out = mod2([z_he, z_pr])
    out.sum().backward()
    assert mod2.phi.grad is not None and mod2.phi.grad.abs().sum().item() > 0
    assert mod2.attn.qkv.weight.grad is not None
    print("  phi + qkv gradients populated — trainable")

    print("\nALL VERIFICATION CHECKS PASSED")


if __name__ == "__main__":
    main()
