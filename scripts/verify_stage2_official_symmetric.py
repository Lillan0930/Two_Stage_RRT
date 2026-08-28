#!/usr/bin/env python3
"""Verify the symmetric dual-modality Cross-Staining CR-MSA (Stage 2).

Checks (no training, CPU):
  1. EPEG is off                 ->  module.attn.pe is None
  2. routing-token shapes         ->  B=1, N=2500, region_num=8, crmsa_k=3
                                      routing_he == routing_pr == (3, 64, 512)
                                      routing_joint == (3, 128, 512)
  3. output shape (symmetric)     ->  (1, N_HE + N_PR, 512) == (1, 5000, 512)
  4. output shape (asymmetric)    ->  N_HE=2300 / N_PR=2500 -> (1, 4800, 512)
  5. gradients + finiteness       ->  both modalities receive non-zero grad;
                                      phi.grad / attn.qkv.weight.grad non-None
  6. single-modality kernel match ->  max |delta_stage2 - CrossRegionAttention|
                                      (phi + InnerAttention weights copied)

Run:  python scripts/verify_stage2_official_symmetric.py
"""
import sys
from pathlib import Path

import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))

from models.cross_staining_crmsa import CrossStainingCRMSA
from models.rmsa import CrossRegionAttention


def main():
    torch.manual_seed(0)
    D = 512
    module = CrossStainingCRMSA(
        dim=D, num_heads=8, region_num=8, crmsa_k=3,
        drop_out=0.0, drop_path=0.0, epeg=False, epeg_k=15,
        crmsa_mlp=False, ffn=False, qkv_bias=True,
    )
    module.eval()

    print("=" * 70)
    print("Verify symmetric CrossStainingCRMSA (Stage 2 official-style)")
    print("=" * 70)

    # ---- Check 1: EPEG off ----
    print("\n[1] EPEG off -> module.attn.pe is None")
    assert module.attn.pe is None, "EPEG must be off (attn.pe must be None)"
    print("    OK: module.attn.pe is None")

    # ---- Check 2: routing-token shapes ----
    print("\n[2] routing-token shapes (B=1, N=2500, region_num=8, crmsa_k=3)")
    z_he = torch.randn(1, 2500, D)
    z_pr = torch.randn(1, 2500, D)
    routing_he, _, _, _, _, _, _ = module._combine(module.norm(z_he))
    routing_pr, _, _, _, _, _, _ = module._combine(module.norm(z_pr))
    routing_joint = torch.cat([routing_he, routing_pr], dim=1)
    print(f"    routing_he.shape    = {tuple(routing_he.shape)}")
    print(f"    routing_pr.shape    = {tuple(routing_pr.shape)}")
    print(f"    routing_joint.shape = {tuple(routing_joint.shape)}")
    assert routing_he.shape == (3, 64, 512), routing_he.shape
    assert routing_pr.shape == (3, 64, 512), routing_pr.shape
    assert routing_joint.shape == (3, 128, 512), routing_joint.shape
    print("    OK")

    # ---- Check 3: output shape (symmetric) ----
    print("\n[3] output shape (N_HE=N_PR=2500)")
    with torch.no_grad():
        out = module([z_he, z_pr])
    print(f"    output.shape = {tuple(out.shape)}")
    assert out.shape == (1, 5000, 512), out.shape
    print("    OK")

    # ---- Check 4: output shape (asymmetric) ----
    print("\n[4] output shape (N_HE=2300, N_PR=2500)")
    z_he2 = torch.randn(1, 2300, D)
    z_pr2 = torch.randn(1, 2500, D)
    with torch.no_grad():
        out2 = module([z_he2, z_pr2])
    print(f"    output.shape = {tuple(out2.shape)}")
    assert out2.shape == (1, 4800, 512), out2.shape
    print("    OK")

    # ---- Check 5: gradients + finiteness ----
    print("\n[5] gradients (both modalities) + finite output")
    z_he3 = torch.randn(1, 2500, D, requires_grad=True)
    z_pr3 = torch.randn(1, 2500, D, requires_grad=True)
    out3 = module([z_he3, z_pr3])
    assert torch.isfinite(out3).all(), "output has non-finite values"
    out3.sum().backward()
    assert module.phi.grad is not None, "phi.grad is None"
    assert module.attn.qkv.weight.grad is not None, "attn.qkv.weight.grad is None"
    assert z_he3.grad is not None and z_pr3.grad is not None, "input grad missing"
    assert torch.isfinite(z_he3.grad).all() and torch.isfinite(z_pr3.grad).all()
    assert z_he3.grad.norm() > 0, "HE grad is zero"
    assert z_pr3.grad.norm() > 0, "PR grad is zero"
    print(f"    phi.grad norm        = {module.phi.grad.norm().item():.4e}")
    print(f"    attn.qkv.weight.grad = {module.attn.qkv.weight.grad.norm().item():.4e}")
    print(f"    HE grad norm         = {z_he3.grad.norm().item():.4e}")
    print(f"    PR grad norm         = {z_pr3.grad.norm().item():.4e}")
    print("    OK")

    # ---- Check 6: single-modality kernel vs official ----
    print("\n[6] single-modality kernel vs official CrossRegionAttention")
    ref = CrossRegionAttention(
        dim=D, num_heads=8, region_num=8, crmsa_k=3,
        drop=0.0, attn_drop=0.0, epeg=False, qkv_bias=True,
    )
    ref.eval()
    # copy phi + InnerAttention weights
    ref.phi.data.copy_(module.phi.data)
    ref.attn.load_state_dict(module.attn.state_dict())

    x = torch.randn(1, 2500, D)
    x_n = module.norm(x)
    routing, dmm, dw, rs, H, W, add = module._combine(x_n)
    routing = module.attn(routing)
    delta_stage2_single = module._dispatch(routing, dmm, dw, rs, H, W, add)

    with torch.no_grad():
        delta_reference = ref(x_n)

    max_abs_err = (delta_stage2_single - delta_reference).abs().max().item()
    print(f"    delta_stage2_single.shape = {tuple(delta_stage2_single.shape)}")
    print(f"    delta_reference.shape      = {tuple(delta_reference.shape)}")
    print(f"    max |delta_stage2 - delta_ref| = {max_abs_err:.3e}")
    assert torch.allclose(delta_stage2_single, delta_reference, atol=1e-6, rtol=1e-5), \
        f"kernel mismatch: max abs err = {max_abs_err:.3e}"
    print("    OK")

    print("\n" + "=" * 70)
    print(f"ALL CHECKS PASSED  (max abs err vs official kernel = {max_abs_err:.3e})")
    print("=" * 70)


if __name__ == "__main__":
    main()
