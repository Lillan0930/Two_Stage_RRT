#!/usr/bin/env python3
"""Acceptance tests for `models/he_residual_cross_crmsa.HEResidualCrossCRMSA`.

Covers the 8 acceptance criteria from the Stage2 改造任务书:

  1.  HE exact fallback (torch.equal) when disable_cross or residual_scale == 0
  2.  output shape / order [B, N_HE, D], N_HE != N_PR
  3.  slot renumbering invariance (phi column permutation)
  4.  mask correctness (invalid-patch invariance + valid-zero-patch used)
  5.  empty-region numerical stability (no NaN; all-PR-invalid ⇒ identity)
  6.  batch isolation
  7.  valid non-zero gradients (residual_scale is a buffer, no grad)
  8.  integration + init + unknown stage2_type raises

Run:
  /home/cxl/miniconda3/envs/rrtmil/bin/python tests/test_he_residual_cross_crmsa.py
"""
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from models.he_residual_cross_crmsa import HEResidualCrossCRMSA

# deterministic
torch.manual_seed(0)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def make_mod(**over):
    kw = dict(dim=64, num_heads=4, region_num=2, crmsa_k=3,
              drop_out=0.0, drop_path=0.0, epeg=False, crmsa_mlp=False,
              ffn=False, qkv_bias=True, residual_scale=0.1,
              disable_cross=False)
    kw.update(over)
    return HEResidualCrossCRMSA(**kw)


# ---------------------------------------------------------------------------
# Test 1 — HE exact fallback
# ---------------------------------------------------------------------------
def test1():
    B, N_he, N_pr, D = 1, 16, 25, 64
    z_he = torch.randn(B, N_he, D)
    z_pr = torch.randn(B, N_pr, D)

    m_disable = make_mod(disable_cross=True)
    m_zero = make_mod(residual_scale=0.0)
    m_disable.eval()
    m_zero.eval()

    with torch.no_grad():
        out_disable = m_disable(z_he, z_pr)
        out_zero = m_zero(z_he, z_pr)

    check("1a disable_cross → torch.equal(z_he)",
          torch.equal(out_disable, z_he))
    check("1b residual_scale=0 → torch.equal(z_he)",
          torch.equal(out_zero, z_he))


# ---------------------------------------------------------------------------
# Test 2 — shape / order, N_HE != N_PR
# ---------------------------------------------------------------------------
def test2():
    B, N_he, N_pr, D = 2, 16, 25, 64
    z_he = torch.randn(B, N_he, D)
    z_pr = torch.randn(B, N_pr, D)

    m = make_mod()
    m.eval()
    out = m(z_he, z_pr)

    check("2a output shape == [B, N_HE, D]", out.shape == (B, N_he, D),
          f"got {tuple(out.shape)}")
    check("2b output finite", torch.isfinite(out).all())
    check("2c residual is non-trivial (out != z_he)",
          not torch.equal(out, z_he))
    # order: patch i maps to position i (region_reverse is exact inverse of
    # region_partition). Verify the round-trip primitive the module relies on.
    from models.rmsa import region_partition, region_reverse
    H = W = 4
    rs = 2
    x = torch.randn(B, H, W, D)
    regs = region_partition(x, rs).view(B, -1, rs, rs, D)
    back = region_reverse(regs.view(B * (H // rs) * (W // rs), rs, rs, D),
                          rs, H, W).view(B, H, W, D)
    check("2d region round-trip exact", torch.equal(back, x))


# ---------------------------------------------------------------------------
# Test 3 — slot renumbering invariance (phi column permutation)
# ---------------------------------------------------------------------------
def test3():
    B, N_he, N_pr, D = 1, 16, 25, 64
    z_he = torch.randn(B, N_he, D)
    z_pr = torch.randn(B, N_pr, D)

    m_a = make_mod()
    m_a.eval()
    m_b = copy.deepcopy(m_a)

    perm_he = torch.randperm(m_a.crmsa_k)
    perm_pr = torch.randperm(m_a.crmsa_k)
    with torch.no_grad():
        m_b.phi_he.data.copy_(m_b.phi_he.data[:, perm_he])
        m_b.phi_pr.data.copy_(m_b.phi_pr.data[:, perm_pr])

    with torch.no_grad():
        out_a = m_a(z_he, z_pr)
        out_b = m_b(z_he, z_pr)

    check("3 slot renumbering invariance (phi column perm)",
          torch.allclose(out_a, out_b, atol=1e-5),
          f"max_abs={(out_a - out_b).abs().max().item():.3e}")


# ---------------------------------------------------------------------------
# Test 4 — mask correctness
# ---------------------------------------------------------------------------
def test4():
    B, N_he, N_pr, D = 1, 16, 25, 64
    z_he = torch.randn(B, N_he, D)
    z_pr = torch.randn(B, N_pr, D)

    valid_he = torch.ones(B, N_he, dtype=torch.bool)
    valid_pr = torch.ones(B, N_pr, dtype=torch.bool)
    valid_he[0, 3] = False
    valid_he[0, 10] = False
    valid_pr[0, 0] = False
    valid_pr[0, 20] = False

    m = make_mod()
    m.eval()

    with torch.no_grad():
        out1 = m(z_he, z_pr, valid_he=valid_he, valid_pr=valid_pr)

        # change finite VALUES of invalid patches → output must be unchanged
        z_he2 = z_he.clone()
        z_pr2 = z_pr.clone()
        z_he2[0, 3] = torch.randn(D)
        z_he2[0, 10] = torch.randn(D)
        z_pr2[0, 0] = torch.randn(D)
        z_pr2[0, 20] = torch.randn(D)
        out2 = m(z_he2, z_pr2, valid_he=valid_he, valid_pr=valid_pr)

    check("4a invalid-patch values do not affect valid-position output",
          torch.allclose(out1[valid_he], out2[valid_he], atol=1e-5),
          f"max_abs={(out1[valid_he] - out2[valid_he]).abs().max().item():.3e}")

    # invalid HE patches get zero residual (identity), independent of their value
    check("4c invalid HE patch gets zero residual (identity)",
          torch.allclose(out1[~valid_he], z_he[~valid_he], atol=1e-5))

    # 4b — a valid all-zero patch must be USED (its value affects output),
    # i.e. validity is from mask/length, not from "all-zero features".
    z_he3 = z_he.clone()
    z_he3[0, 5] = 0.0  # within length, valid by default → treated as valid
    with torch.no_grad():
        out_base = m(z_he, z_pr)          # patch 5 non-zero
        out_zero = m(z_he3, z_pr)         # patch 5 zero, still valid
    check("4b valid-zero patch is used (value affects output)",
          not torch.allclose(out_base, out_zero, atol=1e-5))


# ---------------------------------------------------------------------------
# Test 5 — empty-region numerical stability
# ---------------------------------------------------------------------------
def test5():
    B, N_he, N_pr, D = 1, 16, 25, 64
    z_he = torch.randn(B, N_he, D)
    z_pr = torch.randn(B, N_pr, D)

    # 5a — no valid PR token at all ⇒ strictly keep HE (identity)
    valid_pr_none = torch.zeros(B, N_pr, dtype=torch.bool)
    m = make_mod()
    m.eval()
    with torch.no_grad():
        out_none = m(z_he, z_pr, valid_pr=valid_pr_none)
    check("5a all-PR-invalid → torch.equal(z_he)",
          torch.equal(out_none, z_he))

    # 5b — some HE/PR regions fully invalid → no NaN
    valid_he = torch.ones(B, N_he, dtype=torch.bool)
    valid_pr = torch.ones(B, N_pr, dtype=torch.bool)
    # region 0 of HE = patches 0..3 (region_size=2 → P=4); invalidate all of it
    valid_he[0, 0:4] = False
    # region 3 of PR = patches 24..35 (padded to 36); invalidate the real ones
    valid_pr[0, 24:25] = False
    with torch.no_grad():
        out_partial = m(z_he, z_pr, valid_he=valid_he, valid_pr=valid_pr)
    check("5b partial-empty regions → finite output",
          torch.isfinite(out_partial).all())


# ---------------------------------------------------------------------------
# Test 6 — batch isolation
# ---------------------------------------------------------------------------
def test6():
    B, N_he, N_pr, D = 2, 16, 25, 64
    z_he = torch.randn(B, N_he, D)
    z_pr = torch.randn(B, N_pr, D)

    m = make_mod()
    m.eval()
    with torch.no_grad():
        out_batch = m(z_he, z_pr)
        out0 = m(z_he[0:1], z_pr[0:1])
        out1 = m(z_he[1:2], z_pr[1:2])

    check("6 batch isolation (batch[0] == single[0])",
          torch.allclose(out_batch[0:1], out0, atol=1e-5))
    check("6 batch isolation (batch[1] == single[1])",
          torch.allclose(out_batch[1:2], out1, atol=1e-5))


# ---------------------------------------------------------------------------
# Test 7 — valid non-zero gradients
# ---------------------------------------------------------------------------
def test7():
    B, N_he, N_pr, D = 1, 16, 25, 64
    z_he = torch.randn(B, N_he, D)
    z_pr = torch.randn(B, N_pr, D)

    m = make_mod()
    m.train()
    out = m(z_he, z_pr)
    out.sum().backward()

    learnable = [n for n, p in m.named_parameters() if p.requires_grad]
    bad = [n for n in learnable
           if m.get_parameter(n).grad is None
           or m.get_parameter(n).grad.norm() <= 0]
    check("7 all learnable params have non-zero grad",
          len(bad) == 0, f"bad={bad}")

    # residual_scale is a buffer → not in parameters, no grad
    is_param = 'residual_scale' in dict(m.named_parameters())
    is_buffer = 'residual_scale' in dict(m.named_buffers())
    check("7b residual_scale is a buffer (not a parameter)",
          (not is_param) and is_buffer)


# ---------------------------------------------------------------------------
# Test 8 — integration + init + unknown stage2_type
# ---------------------------------------------------------------------------
def test8():
    from models.mm_rrt_abmil import MM_RRT_ABMIL
    from models.he_residual_cross_crmsa import HEResidualCrossCRMSA

    base = dict(num_modalities=2, modality_list=['HE', 'PR'],
                input_dim=128, mlp_dim=128, num_classes=2,
                dropout=0.0, region_num=4, n_layers=2, n_heads=4,
                drop_path=0.0, trans_dropout=0.1, epeg=True, epeg_k=9,
                crmsa_k=3, cr_msa=True, all_shortcut=True,
                crmsa_heads=4, crmsa_mlp=False,
                fusion_type='two_stage_region',
                stage2_type='he_residual_cross',
                encoder_cfg={
                    'HE': {'region_num': 4, 'epeg_k': 9, 'crmsa_k': 3,
                           'n_heads': 4, 'drop_path': 0.0},
                    'PR': {'region_num': 4, 'epeg_k': 9, 'crmsa_k': 3,
                           'n_heads': 4, 'drop_path': 0.0},
                },
                stage2_cfg={'region_num': 4, 'crmsa_heads': 4, 'crmsa_k': 3,
                            'drop_out': 0.0, 'drop_path': 0.0, 'epeg': False,
                            'crmsa_mlp': False, 'ffn': False, 'qkv_bias': True,
                            'residual_scale': 0.1, 'disable_cross': False},
                abmil_hidden_dim=64)

    model = MM_RRT_ABMIL(**base)
    check("8a cross_region_mod is HEResidualCrossCRMSA",
          isinstance(model.cross_region_mod, HEResidualCrossCRMSA))

    # forward produces [B, num_classes] logits + stage2 fusion_stats
    model.eval()
    N = 256
    x_he = torch.randn(1, N, 128)
    x_pr = torch.randn(1, N, 128)
    with torch.no_grad():
        logits, _, _, fusion_stats = model([x_he, x_pr])
    check("8b forward logits shape [1, 2]",
          tuple(logits.shape) == (1, 2), f"got {tuple(logits.shape)}")
    check("8c fusion_stats stage2 == 'he_residual_cross'",
          fusion_stats.get('stage2') == 'he_residual_cross')

    # Q/K/V/out weights are non-zero (normal init, no zero-init out-proj)
    m2 = make_mod()
    wq = m2.w_q.weight
    wo = m2.w_out.weight
    check("8d Q/out projections non-zero init",
          wq.abs().max().item() > 0 and wo.abs().max().item() > 0)

    # unknown stage2_type must raise
    try:
        MM_RRT_ABMIL(**{**base, 'stage2_type': 'bogus_type'})
        check("8e unknown stage2_type raises ValueError", False)
    except ValueError:
        check("8e unknown stage2_type raises ValueError", True)


def main():
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()

    print("\n" + "=" * 60)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"TOTAL: {len(RESULTS)} checks, {n_fail} failed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAIL: {name}  {detail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
