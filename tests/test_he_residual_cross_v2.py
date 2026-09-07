#!/usr/bin/env python3
"""Implementation tests for HEResidualCrossCRMSAv2 (5 tests).

  Test 1 — HE fallback / identity + bias-free projections
  Test 2 — uniform attention cancellation (constant slide ⇒ centered O ≈ 0)
  Test 3 — PR content sensitivity (changing PR changes the delta)
  Test 4 — mask / empty-PR handling (identity + invalid-HE zeroing, finite)
  Test 5 — gradient (finite, full main path alive, PR path non-zero)

Run:  python tests/test_he_residual_cross_v2.py
"""
import os, sys, math
from pathlib import Path

import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from models.he_residual_cross_crmsa_v2 import HEResidualCrossCRMSAv2

DIM = 64
HEADS = 4
REGION_NUM = 2
CRMSA_K = 2
N = 16
TAU = 0.2


def make_mod(**kw):
    base = dict(dim=DIM, num_heads=HEADS, region_num=REGION_NUM, crmsa_k=CRMSA_K,
                drop_out=0.1, drop_path=0.0, qkv_bias=True, tau=TAU)
    base.update(kw)
    return HEResidualCrossCRMSAv2(**base)


def rand_z(seed, b=1, n=N, d=DIM):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, d, generator=g)


def test1_fallback_and_bias_free():
    m = make_mod()
    assert m.w_q.bias is None and m.w_k.bias is None, "w_q/w_k must be bias-free"
    assert m.w_v.bias is None and m.w_out.bias is None, "w_v/w_out must be bias-free"
    assert abs(m.tau - TAU) < 1e-9, f"tau must be fixed {TAU}, got {m.tau}"

    z_he = rand_z(1)
    z_pr = rand_z(2)
    out = m(z_he, z_pr)
    assert out.shape == z_he.shape, "output must be [B, N_HE, D]"

    m.disable_cross = True
    assert torch.equal(m(z_he, z_pr), z_he), "disable_cross must be identity"
    m.disable_cross = False

    m.residual_scale = torch.tensor(0.0)
    assert torch.equal(m(z_he, z_pr), z_he), "residual_scale==0 must be identity"
    m.residual_scale = torch.tensor(0.1)
    return "bias-free + identity paths OK"


def test2_uniform_attention_cancellation():
    m = make_mod().eval()
    with torch.no_grad():
        c_he = torch.randn(DIM)
        c_pr = torch.randn(DIM)
        z_he = c_he.unsqueeze(0).expand(1, N, DIM)
        z_pr = c_pr.unsqueeze(0).expand(1, N, DIM)
        diag = m.diagnose(z_he, z_pr)
        rel = diag['delta_patch'].norm() / (diag['z_he'].norm() + 1e-8)
    assert rel < 1e-3, f"uniform attention should cancel the residual, rel={rel}"
    return f"uniform cancellation: ||Δ||/||Z||={rel:.2e}"


def test3_pr_content_sensitivity():
    m = make_mod().eval()
    with torch.no_grad():
        z_he = rand_z(3)
        z_pr_a = rand_z(4)
        z_pr_b = rand_z(5)
        da = m.diagnose(z_he, z_pr_a)['delta_patch']
        db = m.diagnose(z_he, z_pr_b)['delta_patch']
        diff = (da - db).norm()
    assert diff > 1e-6, "delta must depend on PR content"
    return f"PR content sensitivity: ||Δa−Δb||={diff:.4f}"


def test4_mask_and_empty_pr():
    m = make_mod().eval()
    with torch.no_grad():
        z_he = rand_z(6)
        z_pr = rand_z(7)

        # all PR invalid → strict identity (no NaN)
        valid_pr = torch.zeros(1, N, dtype=torch.bool)
        out = m(z_he, z_pr, valid_pr=valid_pr)
        assert torch.equal(out, z_he), "all-invalid PR must yield identity"

        # partial masks → finite + invalid HE slots zeroed
        valid_he = torch.ones(1, N, dtype=torch.bool)
        valid_he[0, :4] = False
        valid_pr2 = torch.ones(1, N, dtype=torch.bool)
        valid_pr2[0, :6] = False
        d = m.diagnose(z_he, z_pr, valid_he=valid_he, valid_pr=valid_pr2)
        assert torch.isfinite(d['delta_patch']).all(), "delta must be finite"
        assert d['delta_patch'][0, :4].abs().max() < 1e-6, \
            "invalid HE slots must be zeroed"
    return "mask + empty-PR OK"


def test5_gradient():
    torch.manual_seed(0)
    m = make_mod()
    z_he = rand_z(8)
    z_pr = rand_z(9)
    out = m(z_he, z_pr)
    out.sum().backward()

    grads = {}
    for name, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[name] = p.grad.norm().item()

    assert grads, "no gradients collected"
    assert all(math.isfinite(v) for v in grads.values()), "non-finite gradient"
    for key in ['w_q.weight', 'w_k.weight', 'w_v.weight', 'w_out.weight',
                'phi_pr', 'phi_he', 'attn_norm_pr.weight']:
        assert key in grads, f"missing grad for {key}"
    assert grads['w_v.weight'] > 0 and grads['w_k.weight'] > 0 and \
        grads['phi_pr'] > 0, "PR value/attention path must receive gradient"
    return (f"gradients finite & alive: w_q={grads['w_q.weight']:.2e} "
            f"w_k={grads['w_k.weight']:.2e} w_v={grads['w_v.weight']:.2e} "
            f"phi_pr={grads['phi_pr']:.2e}")


def main():
    torch.manual_seed(0)
    tests = [
        ("Test 1 — HE fallback + bias-free", test1_fallback_and_bias_free),
        ("Test 2 — uniform-attention cancellation", test2_uniform_attention_cancellation),
        ("Test 3 — PR content sensitivity", test3_pr_content_sensitivity),
        ("Test 4 — mask / empty-PR", test4_mask_and_empty_pr),
        ("Test 5 — gradient", test5_gradient),
    ]
    passed = 0
    for name, fn in tests:
        try:
            msg = fn()
            print(f"[PASS] {name} — {msg}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name} — {e}")
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {name} — {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
