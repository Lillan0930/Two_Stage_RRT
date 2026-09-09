#!/usr/bin/env python3
"""Implementation tests for HEResidualCrossCRMSAv5 (5 tests).

  Test 1 — aux loss alone backprops into PR value path (w_v / phi_pr /
           attn_norm_pr) but NOT HE path (w_q / w_k / w_out / phi_he)
  Test 2 — shared weights + prototype, eval mode: v5 fused output == v3 (β=0)
  Test 3 — fallback: disable_cross / residual_scale==0 → strict Z_HE identity,
           aux dict is None
  Test 4 — mask / empty-region / empty-PR forward + backward all finite,
           empty PR flagged pr_value_has_valid=False
  Test 5 — EMA updates exactly once per training forward (aux does NOT re-update)
           + checkpoint save/restore of mu_pr & prototype_initialized

Run:  python tests/test_he_residual_cross_v5.py
"""
import os, sys, math
from pathlib import Path

import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from models.he_residual_cross_crmsa_v3 import HEResidualCrossCRMSAv3
from models.he_residual_cross_crmsa_v5 import HEResidualCrossCRMSAv5

DIM = 64
HEADS = 4
REGION_NUM = 2
CRMSA_K = 2
N = 16
TAU = 0.2
BETA = 0.99
NUM_CLASSES = 2


def make_mod(**kw):
    base = dict(dim=DIM, num_heads=HEADS, region_num=REGION_NUM, crmsa_k=CRMSA_K,
                drop_out=0.1, drop_path=0.0, qkv_bias=True, tau=TAU,
                prototype_momentum=BETA, num_classes=NUM_CLASSES)
    base.update(kw)
    return HEResidualCrossCRMSAv5(**base)


def make_v3(**kw):
    base = dict(dim=DIM, num_heads=HEADS, region_num=REGION_NUM, crmsa_k=CRMSA_K,
                drop_out=0.1, drop_path=0.0, qkv_bias=True, tau=TAU,
                prototype_momentum=BETA)
    base.update(kw)
    return HEResidualCrossCRMSAv3(**base)


def rand_z(seed, b=1, n=N, d=DIM):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, d, generator=g)


def reference_value_mean(m, z_pr):
    """[1, h, d] = mean over ALL routing tokens of w_v(attn_norm(R)) (full valid)."""
    routing, dmm, dw, valid_slots, H, W, add, rs = \
        m._route(z_pr, m.phi_pr, m.route_norm_pr, None)
    r_pr = routing.reshape(1, -1, m.dim)                 # [1, K, D]
    v_full = m.w_v(m.attn_norm_pr(r_pr))                 # [1, K, D]
    v = v_full.view(1, -1, m.num_heads, m.head_dim).transpose(1, 2)  # [1,h,K,d]
    return v.mean(dim=2)                                  # [1,h,d]


def test1_aux_loss_updates_pr_value_only():
    torch.manual_seed(0)
    m = make_mod()                                        # train mode
    z_he, z_pr = rand_z(1), rand_z(2)
    out, aux = m(z_he, z_pr)
    assert aux is not None and 'pr_value_logits' in aux

    # Backward on the AUXILIARY logits ALONE (never touching the main `out`).
    aux['pr_value_logits'].sum().backward()

    # PR value path must receive gradient (aux head reads the shared V_tilde).
    for key in ['w_v.weight', 'phi_pr', 'attn_norm_pr.weight',
                'route_norm_pr.weight']:
        p = dict(m.named_parameters())[key]
        assert p.grad is not None and p.grad.norm().item() > 0, \
            f"aux loss must update {key}"
    # aux head itself gets gradient
    for key in ['aux_attn.0.weight', 'aux_attn.2.weight',
                'aux_classifier.0.weight', 'aux_classifier.3.weight']:
        p = dict(m.named_parameters())[key]
        assert p.grad is not None and p.grad.norm().item() > 0, \
            f"aux head {key} must receive gradient"

    # HE / attention-score path must NOT be updated by the aux loss.
    for key in ['w_q.weight', 'w_k.weight', 'w_out.weight',
                'phi_he', 'attn_norm_he.weight', 'route_norm_he.weight']:
        p = dict(m.named_parameters())[key]
        assert p.grad is None, f"aux loss must NOT update {key} (got grad)"

    return ("aux→w_v/phi_pr/attn_norm_pr OK; w_q/w_k/w_out/phi_he untouched")


def test2_fused_output_matches_v3_eval():
    m3 = make_v3().eval()
    m5 = make_mod().eval()
    # non-trivial prototype so centering is exercised
    with torch.no_grad():
        m3.mu_pr.copy_(torch.randn_like(m3.mu_pr) * 2.0)
        m3.prototype_initialized.fill_(1.0)
    # copy ALL v3 common weights + buffers into v5 (aux params keep v5 init)
    m5.load_state_dict(m3.state_dict(), strict=False)

    z_he, z_pr = rand_z(30), rand_z(31)
    with torch.no_grad():
        out3 = m3(z_he, z_pr)
        out5, aux = m5(z_he, z_pr)
    diff = (out3 - out5).abs().max().item()
    assert diff < 1e-6, f"v5 fused output must equal v3 (β=0), max|Δ|={diff:.2e}"
    assert aux is not None and 'pr_value_logits' in aux, "aux must still be produced"
    return f"v5 fused output == v3 (max|Δ|={diff:.2e}); aux present but not fused"


def test3_fallback_identity():
    m = make_mod()
    assert m.w_q.bias is None and m.w_k.bias is None and \
        m.w_v.bias is None and m.w_out.bias is None, "all cross projections bias-free"
    z_he, z_pr = rand_z(40), rand_z(41)
    out, aux = m(z_he, z_pr)
    assert out.shape == z_he.shape and aux is not None
    m.disable_cross = True
    out_dc, aux_dc = m(z_he, z_pr)
    assert torch.equal(out_dc, z_he), "disable_cross must be identity"
    assert aux_dc is None, "disable_cross must yield no aux dict"
    m.disable_cross = False
    m.residual_scale = torch.tensor(0.0)
    out_rs, aux_rs = m(z_he, z_pr)
    assert torch.equal(out_rs, z_he), "residual_scale==0 must be identity"
    assert aux_rs is None, "residual_scale==0 must yield no aux dict"
    m.residual_scale = torch.tensor(0.1)
    return "bias-free + identity paths OK (aux=None on fallback)"


def test4_mask_empty_forward_backward_finite():
    m = make_mod().eval()
    with torch.no_grad():
        z_he, z_pr = rand_z(50), rand_z(51)

        # (a) empty PR → identity + has_valid=False + finite aux logits
        valid_pr = torch.zeros(1, N, dtype=torch.bool)
        out, aux = m(z_he, z_pr, valid_pr=valid_pr)
        assert torch.equal(out, z_he), "all-invalid PR must yield identity"
        assert aux is not None
        assert bool(aux['pr_value_has_valid'].item()) is False, \
            "empty PR must be flagged has_valid=False"
        assert torch.isfinite(aux['pr_value_logits']).all(), "empty-PR aux finite"

        # (b) partial masks → finite forward
        valid_he = torch.ones(1, N, dtype=torch.bool); valid_he[0, :4] = False
        valid_pr2 = torch.ones(1, N, dtype=torch.bool); valid_pr2[0, :6] = False
        out2, aux2 = m(z_he, z_pr, valid_he=valid_he, valid_pr=valid_pr2)
        assert torch.isfinite(out2).all()
        assert bool(aux2['pr_value_has_valid'].item()) is True
        assert torch.isfinite(aux2['pr_value_logits']).all()

    # (c) backward through main + aux is finite
    m.train()
    m.zero_grad()
    out, aux = m(z_he, z_pr, valid_he=valid_he, valid_pr=valid_pr2)
    (out.sum() + aux['pr_value_logits'].sum()).backward()
    for name, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    return "mask + empty-region + empty-PR forward/backward finite"


def test5_ema_once_per_forward_and_checkpoint():
    torch.manual_seed(0)
    m = make_mod()                                        # train mode
    z_pr_a, z_he = rand_z(60), rand_z(61)
    z_pr_b = rand_z(62)

    # Forward 1: prototype initialized to m_i(a)
    m(z_he, z_pr_a)
    assert bool(m.prototype_initialized.item()) is True
    mu_a = m.mu_pr.clone()
    mi_a = reference_value_mean(m, z_pr_a)
    assert (mu_a - mi_a).abs().max() < 1e-5, "first forward copies m_i"

    # Forward 2: exactly ONE EMA step mu <- beta*mu + (1-beta)*m_i(b)
    m(z_he, z_pr_b)
    mi_b = reference_value_mean(m, z_pr_b)
    expected = BETA * mu_a + (1.0 - BETA) * mi_b
    got = m.mu_pr
    assert (got - expected).abs().max() < 1e-5, \
        f"EMA must update exactly once/forward, got drift {(got-expected).abs().max():.2e}"

    # eval forward must NOT update
    m.eval()
    mu_before = m.mu_pr.clone()
    with torch.no_grad():
        m(z_he, z_pr_a)
    assert torch.equal(mu_before, m.mu_pr), "eval must freeze mu_pr"

    # checkpoint round-trip: buffers + params + output preserved
    m.train()
    sd = m.state_dict()
    m2 = make_mod()
    m2.load_state_dict(sd)
    assert torch.equal(m2.mu_pr, m.mu_pr), "mu_pr must restore"
    assert torch.equal(m2.prototype_initialized, m.prototype_initialized)
    assert 'mu_pr' in sd and 'prototype_initialized' in sd, "buffers in state_dict"
    m.eval(); m2.eval()
    with torch.no_grad():
        o1, _ = m(z_he, z_pr_b)
        o2, _ = m2(z_he, z_pr_b)
    assert torch.equal(o1, o2), "checkpoint restore must reproduce output"
    drift = (got - expected).abs().max().item()
    return (f"EMA once/forward (|drift|={drift:.2e}), "
            "eval frozen, checkpoint round-trip OK")


def main():
    torch.manual_seed(0)
    tests = [
        ("Test 1 — aux loss updates PR value path only", test1_aux_loss_updates_pr_value_only),
        ("Test 2 — v5 fused output == v3 (β=0)", test2_fused_output_matches_v3_eval),
        ("Test 3 — fallback (disable_cross / residual_scale=0)", test3_fallback_identity),
        ("Test 4 — mask / empty-PR / backward finite", test4_mask_empty_forward_backward_finite),
        ("Test 5 — EMA once/forward + checkpoint", test5_ema_once_per_forward_and_checkpoint),
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
