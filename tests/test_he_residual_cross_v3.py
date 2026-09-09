#!/usr/bin/env python3
"""Implementation tests for HEResidualCrossCRMSAv3 (6 tests).

  Test 1 — prototype EMA updates in train, frozen in eval (no val/test leakage)
  Test 2 — dataset-common removed, slide-specific preserved (prototype centering)
  Test 3 — matched PR content sensitivity on the FINAL enhanced HE (full forward)
  Test 4 — fallback: disable_cross / residual_scale==0 → strict identity
  Test 5 — mask / empty-PR / NaN safety (v2 parity)
  Test 6 — gradient: main path alive, prototype buffer has NO grad / not a param

Run:  python tests/test_he_residual_cross_v3.py
"""
import os, sys, math
from pathlib import Path

import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from models.he_residual_cross_crmsa_v3 import HEResidualCrossCRMSAv3

DIM = 64
HEADS = 4
REGION_NUM = 2
CRMSA_K = 2
N = 16
TAU = 0.2
BETA = 0.99


def make_mod(**kw):
    base = dict(dim=DIM, num_heads=HEADS, region_num=REGION_NUM, crmsa_k=CRMSA_K,
                drop_out=0.1, drop_path=0.0, qkv_bias=True, tau=TAU,
                prototype_momentum=BETA)
    base.update(kw)
    return HEResidualCrossCRMSAv3(**base)


def rand_z(seed, b=1, n=N, d=DIM):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, d, generator=g)


def reference_value_mean(m, z_pr):
    """[1, h, d] = mean over valid routing tokens of w_v(attn_norm(R))."""
    routing, dmm, dw, valid_slots, H, W, add, rs = \
        m._route(z_pr, m.phi_pr, m.route_norm_pr, None)
    r_pr = routing.reshape(1, -1, m.dim)                 # [1, K, D]
    v_full = m.w_v(m.attn_norm_pr(r_pr))                 # [1, K, D]
    v = v_full.view(1, -1, m.num_heads, m.head_dim).transpose(1, 2)  # [1,h,K,d]
    return v.mean(dim=2)                                  # [1,h,d]


def test1_prototype_train_vs_eval():
    m = make_mod()                                       # train mode by default
    z_he_a, z_pr_a = rand_z(1), rand_z(2)
    z_he_b, z_pr_b = rand_z(3), rand_z(4)

    # first train forward initializes the prototype
    m(z_he_a, z_pr_a)
    assert bool(m.prototype_initialized.item()) is True, "prototype must initialize"
    mu_first = m.mu_pr.clone()

    # second train forward on a DIFFERENT slide must move the prototype
    m(z_he_b, z_pr_b)
    assert not torch.equal(mu_first, m.mu_pr), "prototype must update during train"

    # eval forward must NOT mutate the prototype buffer
    m.eval()
    mu_before = m.mu_pr.clone()
    with torch.no_grad():
        m(z_he_a, z_pr_a)
    assert torch.equal(mu_before, m.mu_pr), "prototype must be frozen in eval"
    return "prototype updates in train only, frozen in eval"


def test2_prototype_centering_semantics():
    m = make_mod().eval()
    with torch.no_grad():
        # constant slides: V = C (dataset-common) with no slide-specific component
        C = torch.randn(1, 1, DIM) * 3.0
        z_pr = C.expand(1, N, DIM)                       # exactly constant slide
        z_he = rand_z(30)
        mi = reference_value_mean(m, z_pr)               # ≈ mean value = W_v(attn_norm(C))

        # (a) prototype == slide mean -> common fully removed -> delta ≈ 0
        m.mu_pr.copy_(mi)
        d_rm = m.diagnose(z_he, z_pr)['delta_patch']

        # (b) prototype == 0 -> slide-global component kept -> delta non-zero
        m.mu_pr.zero_()
        d_keep = m.diagnose(z_he, z_pr)['delta_patch']

        # (c) a DIFFERENT constant slide differs after centering (slide-specific kept)
        C2 = torch.randn(1, 1, DIM) * 3.0
        z_pr_b = C2.expand(1, N, DIM)
        d_b = m.diagnose(z_he, z_pr_b)['delta_patch']

    assert d_rm.norm() < 1e-3, f"prototype==mean should cancel, got {d_rm.norm():.2e}"
    assert d_keep.norm() > 1e-3, f"zero prototype keeps slide-global, got {d_keep.norm():.2e}"
    assert (d_keep - d_b).norm() > 1e-3, "two slides must differ after centering"
    return (f"common removed (|Δ_rm|={d_rm.norm():.2e}), slide-global kept "
            f"(|Δ_keep|={d_keep.norm():.3f}), slide diff={ (d_keep - d_b).norm():.3f}")


def test3_pr_content_sensitivity():
    m = make_mod().eval()
    with torch.no_grad():
        z_he = rand_z(31)
        z_pr_a = rand_z(32)
        z_pr_b = rand_z(33)
        out_a = m(z_he, z_pr_a)                          # FINAL enhanced HE
        out_b = m(z_he, z_pr_b)
        diff = (out_a - out_b).norm()
    assert diff > 1e-5, "final enhanced HE must depend on PR content"
    return f"final-output content sensitivity: ||out(PR_A)−out(PR_B)||={diff:.4f}"


def test4_fallback():
    m = make_mod()
    assert m.w_q.bias is None and m.w_k.bias is None and \
        m.w_v.bias is None and m.w_out.bias is None, "all cross projections bias-free"
    assert abs(m.tau - TAU) < 1e-9 and abs(m.beta - BETA) < 1e-9, "tau/beta fixed"
    z_he, z_pr = rand_z(40), rand_z(41)
    out = m(z_he, z_pr)
    assert out.shape == z_he.shape
    m.disable_cross = True
    assert torch.equal(m(z_he, z_pr), z_he), "disable_cross must be identity"
    m.disable_cross = False
    m.residual_scale = torch.tensor(0.0)
    assert torch.equal(m(z_he, z_pr), z_he), "residual_scale==0 must be identity"
    m.residual_scale = torch.tensor(0.1)
    return "bias-free + identity paths OK"


def test5_mask_and_empty_pr():
    m = make_mod().eval()
    with torch.no_grad():
        z_he = rand_z(50)
        z_pr = rand_z(51)
        valid_pr = torch.zeros(1, N, dtype=torch.bool)
        out = m(z_he, z_pr, valid_pr=valid_pr)
        assert torch.equal(out, z_he), "all-invalid PR must yield identity"
        valid_he = torch.ones(1, N, dtype=torch.bool)
        valid_he[0, :4] = False
        valid_pr2 = torch.ones(1, N, dtype=torch.bool)
        valid_pr2[0, :6] = False
        d = m.diagnose(z_he, z_pr, valid_he=valid_he, valid_pr=valid_pr2)
        assert torch.isfinite(d['delta_patch']).all(), "delta must be finite"
        assert d['delta_patch'][0, :4].abs().max() < 1e-6, "invalid HE slots zeroed"
        for kk in ['entropy_norm', 'score_std', 'value_diversity', 'selective_ratio',
                   'mu_norm', 'r_slide', 'r_region']:
            assert torch.isfinite(d[kk]).all(), f"{kk} must be finite"
    return "mask + empty-PR + diagnostics OK"


def test6_gradient():
    torch.manual_seed(0)
    m = make_mod()                                       # train mode → exercises EMA too
    z_he, z_pr = rand_z(60), rand_z(61)
    out = m(z_he, z_pr)
    out.sum().backward()

    # prototype is a buffer, NOT a parameter, and carries no grad
    assert 'mu_pr' not in dict(m.named_parameters()), "mu_pr must be a buffer"
    assert 'prototype_initialized' not in dict(m.named_parameters())
    assert m.mu_pr.grad is None and m.prototype_initialized.grad is None, \
        "prototype buffer must have no grad"

    grads = {name: p.grad.norm().item()
             for name, p in m.named_parameters()
             if p.requires_grad and p.grad is not None}
    assert grads and all(math.isfinite(v) for v in grads.values()), "non-finite grad"
    for key in ['w_q.weight', 'w_k.weight', 'w_v.weight', 'w_out.weight',
                'phi_pr', 'phi_he', 'attn_norm_pr.weight']:
        assert key in grads, f"missing grad for {key}"
    assert grads['w_v.weight'] > 0 and grads['w_k.weight'] > 0 and \
        grads['phi_pr'] > 0, "PR value/attention path must receive gradient"
    return (f"grads alive: w_q={grads['w_q.weight']:.2e} w_k={grads['w_k.weight']:.2e} "
            f"w_v={grads['w_v.weight']:.2e} phi_pr={grads['phi_pr']:.2e} | mu_pr grad=None")


def main():
    torch.manual_seed(0)
    tests = [
        ("Test 1 — prototype train/eval gating", test1_prototype_train_vs_eval),
        ("Test 2 — dataset-common removed, slide-specific kept", test2_prototype_centering_semantics),
        ("Test 3 — final-output PR content sensitivity", test3_pr_content_sensitivity),
        ("Test 4 — fallback (disable_cross / residual_scale=0)", test4_fallback),
        ("Test 5 — mask / empty-PR / diagnostics", test5_mask_and_empty_pr),
        ("Test 6 — gradient (prototype no-grad, main path alive)", test6_gradient),
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
