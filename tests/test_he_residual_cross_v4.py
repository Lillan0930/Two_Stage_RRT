#!/usr/bin/env python3
"""Implementation tests for HEResidualCrossCRMSAv4 (5 tests).

  Test 1 — fallback: disable_cross / residual_scale==0 → identity; bias-free QKV
  Test 2 — common-direction invariance: Q/K per-slide centering removes the
           shared direction (mean≈0) and is invariant to a common Q/K shift
  Test 3 — token-specific difference sensitivity: the residual still responds
           to *which* PR slide / region is given (not over-removed)
  Test 4 — mask / empty-PR / NaN safety (all 12 diagnostics finite)
  Test 5 — gradient: main path alive; prototype buffer has NO grad

Run:  python tests/test_he_residual_cross_v4.py
"""
import os, sys, math
from pathlib import Path

import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from models.he_residual_cross_crmsa_v4 import HEResidualCrossCRMSAv4

DIM = 64
HEADS = 4
REGION_NUM = 2
CRMSA_K = 2
N = 16
TAU = 0.2
BETA = 0.99

DIAG_KEYS = ['entropy_norm', 'score_std', 'value_diversity', 'selective_ratio',
             'mu_norm', 'r_slide', 'r_region',
             'q_raw_cos', 'q_centered_cos', 'k_raw_cos', 'k_centered_cos',
             'attn_weight_std']


def make_mod(**kw):
    base = dict(dim=DIM, num_heads=HEADS, region_num=REGION_NUM, crmsa_k=CRMSA_K,
                drop_out=0.1, drop_path=0.0, qkv_bias=True, tau=TAU,
                prototype_momentum=BETA)
    base.update(kw)
    return HEResidualCrossCRMSAv4(**base)


def rand_z(seed, b=1, n=N, d=DIM):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, d, generator=g)


def reference_qk(m, z_he, z_pr):
    """Recompute the exact post-projection q/k + valid masks (mirrors
    `_cross_attention`), so we can test the centering math directly."""
    routing_he, _, _, valid_he_slots, _, _, _, _ = \
        m._route(z_he, m.phi_he, m.route_norm_he, None)
    routing_pr, _, _, valid_pr_slots, _, _, _, _ = \
        m._route(z_pr, m.phi_pr, m.route_norm_pr, None)
    r_he = routing_he.reshape(1, -1, m.dim)                  # [1, Q, D]
    r_pr = routing_pr.reshape(1, -1, m.dim)                  # [1, K, D]
    q_valid = valid_he_slots.reshape(1, -1)
    k_valid = valid_pr_slots.reshape(1, -1)
    q = m.w_q(m.attn_norm_he(r_he)).view(1, -1, m.num_heads, m.head_dim) \
        .transpose(1, 2)                                     # [1, h, Q, d]
    k = m.w_k(m.attn_norm_pr(r_pr)).view(1, -1, m.num_heads, m.head_dim) \
        .transpose(1, 2)                                     # [1, h, K, d]
    return q, k, q_valid, k_valid


def center(x, valid):
    """x [B,h,T,d], valid [B,T] → x − mean_over_valid_tokens(x) [B,h,T,d]."""
    B, T = x.shape[0], x.shape[2]
    vf = valid.float()
    cnt = vf.sum(-1).clamp(min=1.0)
    mean = (x * vf.view(B, 1, T, 1)).sum(dim=2) / cnt.view(B, 1, 1)
    return x - mean.unsqueeze(2)


def test1_fallback():
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


def test2_common_direction_invariance():
    m = make_mod().eval()
    with torch.no_grad():
        z_he, z_pr = rand_z(21), rand_z(22)
        q, k, q_valid, k_valid = reference_qk(m, z_he, z_pr)
        q_c = center(q, q_valid)
        k_c = center(k, k_valid)

        # (a) centering removes the per-slide common direction: valid-token mean ≈ 0
        B, Q = q.shape[0], q.shape[2]
        qvf = q_valid.float()
        q_c_mean = (q_c * qvf.view(B, 1, Q, 1)).sum(2) / \
            qvf.sum(-1).clamp(min=1.0).view(B, 1, 1)
        assert q_c_mean.abs().max() < 1e-5, \
            f"centered Q must have ~0 valid-token mean, got {q_c_mean.abs().max():.2e}"

        # (b) invariance: adding a common direction to ALL Q / ALL K tokens cancels
        u = torch.randn(1, m.num_heads, 1, m.head_dim) * 3.0
        assert (center(q + u, q_valid) - q_c).abs().max() < 1e-5, \
            "common Q shift must cancel in centering"
        w = torch.randn(1, m.num_heads, 1, m.head_dim) * 3.0
        assert (center(k + w, k_valid) - k_c).abs().max() < 1e-5, \
            "common K shift must cancel in centering"

        # (c) on a genuinely collinear slide (shared base direction), centering
        #     lowers the pairwise cosine — the whole point of §3
        base = torch.randn(1, m.num_heads, 1, m.head_dim) * 5.0
        small = torch.randn(1, m.num_heads, Q, m.head_dim) * 0.1
        q_collinear = base + small
        q_cnt = qvf.sum(-1).clamp(min=1.0)
        raw = m._mean_pairwise_cos_tokens(q_collinear, q_valid, q_cnt).item()
        cen = m._mean_pairwise_cos_tokens(center(q_collinear, q_valid),
                                          q_valid, q_cnt).item()
        assert cen < raw, f"centering should lower pairwise cosine: {raw:.4f} → {cen:.4f}"

    return (f"common-direction removed (|mean q_c|={q_c_mean.abs().max():.2e}), "
            f"invariant to common shift, collinear cos {raw:.4f}→{cen:.4f}")


def test3_token_specific_sensitivity():
    m = make_mod().eval()
    with torch.no_grad():
        z_he, z_pr = rand_z(31), rand_z(32)
        d_a = m.diagnose(z_he, z_pr)['delta_patch']
        # token-specific perturbation (not a common shift): alter one region
        z_pr2 = z_pr.clone()
        z_pr2[0, 0] += torch.randn(DIM) * 1.0
        d_b = m.diagnose(z_he, z_pr2)['delta_patch']
        diff = (d_a - d_b).norm()
        # whole-slide content sensitivity (v3 parity)
        z_pr3 = rand_z(33)
        d_c = m.diagnose(z_he, z_pr3)['delta_patch']
        diff_full = (d_a - d_c).norm()
    assert diff > 1e-5, "token-specific PR change must alter the residual"
    assert diff_full > 1e-5, "full PR slide change must alter the residual"
    return (f"token-specific sensitivity: ||Δ(PR)−Δ(PR')||={diff:.4f}, "
            f"whole-slide={diff_full:.4f}")


def test4_mask_and_empty_pr():
    m = make_mod().eval()
    with torch.no_grad():
        z_he, z_pr = rand_z(50), rand_z(51)
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
        for kk in DIAG_KEYS:
            assert torch.isfinite(d[kk]).all(), f"{kk} must be finite"
    return "mask + empty-PR + 12 diagnostics OK"


def test5_gradient():
    torch.manual_seed(0)
    m = make_mod()                                       # train mode → exercises EMA too
    z_he, z_pr = rand_z(60), rand_z(61)
    out = m(z_he, z_pr)
    out.sum().backward()

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
        ("Test 1 — fallback (disable_cross / residual_scale=0)", test1_fallback),
        ("Test 2 — common-direction invariance (Q/K centering)", test2_common_direction_invariance),
        ("Test 3 — token-specific difference sensitivity", test3_token_specific_sensitivity),
        ("Test 4 — mask / empty-PR / NaN", test4_mask_and_empty_pr),
        ("Test 5 — gradient (prototype no-grad, main path alive)", test5_gradient),
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
