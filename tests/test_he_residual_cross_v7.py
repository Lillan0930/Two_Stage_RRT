#!/usr/bin/env python3
"""Implementation tests for the v7 "restore fused gradient" mechanism (6 tests).

  Test 1 — Variant A (F = Stage2(H.detach(), P)) gradient flow:
           HE CE → HE encoder + MIL only (PR/Stage2 out of graph);
           fused CE → MIL + PR/Stage2, NOT HE encoder.
  Test 2 — Variant B (F = Stage2(H, P)) gradient flow:
           HE CE → HE encoder + MIL only;
           fused CE → HE encoder + MIL + PR/Stage2 (no v6-style isolation).
  Test 3 — eval forward (normal dispatch) matches v3 with the same weights
           (for both A and B).
  Test 4 — disable_cross → strict M(Z_HE) identity; empty-PR fallback finite.
  Test 5 — three-param-group coverage: HE / MIL / PR+Stage2 cover every
           trainable parameter exactly once (no overlap, no leftover).
  Test 6 — full two-loss backward finite for A and B; all three groups
           receive gradient.

Note (same as v6): the RRT encoder's CR-MSA (`rrt_*.cr_msa.*`) has a
pre-existing near-zero gradient w.r.t. its own params (present in v1–v6),
so presence/absence of grad (`.grad is None`) is asserted, not grad > 0.

Run:  python tests/test_he_residual_cross_v7.py
"""
import os, sys
from pathlib import Path

import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from models.mm_rrt_abmil import MM_RRT_ABMIL
from models.he_residual_cross_crmsa_v3 import HEResidualCrossCRMSAv3

INPUT_DIM = 32
MLP_DIM = 64
NUM_CLASSES = 2
N = 16
HEADS = 4
CRMSA_HEADS = 8
REGION_NUM = 2
CRMSA_K = 2


def make_model(detach_he=False, disable_cross=False, encoder_epeg=False):
    """Build a v7 model. detach_he=True → variant A, False → variant B."""
    model_cfg = dict(
        num_modalities=2,
        modality_list=["HE", "PR"],
        input_dim=INPUT_DIM,
        mlp_dim=MLP_DIM,
        num_classes=NUM_CLASSES,
        dropout=0.25,
        region_num=REGION_NUM, n_layers=1, n_heads=HEADS,
        drop_path=0.0, trans_dropout=0.1,
        epeg=encoder_epeg, epeg_k=15, crmsa_k=CRMSA_K,
        cr_msa=True, all_shortcut=True, crmsa_heads=CRMSA_HEADS,
        crmsa_mlp=False,
        fusion_type="two_stage_region", fusion_stage="middle",
        use_gated_fusion=False, abmil_hidden_dim=32,
        use_mclc=False, aggregate_modalities=True,
        stage2_type="he_residual_cross_v7",
        mil_type="abmil",
        encoder_cfg={
            "HE": {"region_num": REGION_NUM, "epeg_k": 15, "crmsa_k": CRMSA_K,
                   "n_heads": HEADS, "drop_path": 0.0},
            "PR": {"region_num": REGION_NUM, "epeg_k": 15, "crmsa_k": CRMSA_K,
                   "n_heads": HEADS, "drop_path": 0.0},
        },
        stage2_cfg={
            "region_num": REGION_NUM, "crmsa_heads": CRMSA_HEADS,
            "crmsa_k": CRMSA_K, "drop_out": 0.1, "drop_path": 0.0,
            "epeg": False, "epeg_k": 15, "crmsa_mlp": False, "ffn": False,
            "qkv_bias": False, "temperature": 0.2, "residual_scale": 0.1,
            "disable_cross": disable_cross, "prototype_momentum": 0.99,
            "v7_fused_he_detach": detach_he,
        },
    )
    return MM_RRT_ABMIL(**model_cfg)


def rand_x(seed, n=N, d=INPUT_DIM):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n, d, generator=g)


def group_of(model, name):
    # three groups (§3): HE / MIL / PR2 (patch_to_emb[1]/rrt_ihc/cross)
    HE = ('patch_to_emb.0.', 'rrt_he.')
    MIL = ('mil.',)
    PR2 = ('patch_to_emb.1.', 'rrt_ihc.', 'cross_region_mod.')
    if name.startswith(HE):
        return 'HE'
    if name.startswith(MIL):
        return 'MIL'
    if name.startswith(PR2):
        return 'PR2'
    return '?'


def _group_norm_sum(model, grp):
    return sum(float(p.grad.norm().item()) if p.grad is not None else 0.0
               for n, p in model.named_parameters()
               if group_of(model, n) == grp)


def test1_variant_a_gradient_flow():
    torch.manual_seed(0)
    m = make_model(detach_he=True).train()
    he, pr = rand_x(1), rand_x(2)

    # (a) HE CE only
    m.zero_grad()
    out = m.forward_v7([he, pr])
    out['he_logits'].sum().backward()
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        grp = group_of(m, n)
        if grp in ('HE', 'MIL'):
            assert p.grad is not None, f"HE CE must reach {grp} {n}"
        else:
            assert p.grad is None, f"HE CE must NOT reach {grp} {n}"
    assert _group_norm_sum(m, 'HE') > 0 and _group_norm_sum(m, 'MIL') > 0
    assert _group_norm_sum(m, 'PR2') == 0

    # (b) fused CE only — A: MUST NOT update HE encoder, MUST update MIL + PR2
    m.zero_grad()
    out2 = m.forward_v7([he, pr])
    out2['fused_logits'].sum().backward()
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        grp = group_of(m, n)
        if grp == 'HE':
            assert p.grad is None, f"A fused CE must NOT update HE {n}"
        else:
            assert p.grad is not None, f"A fused CE must update {grp} {n}"
    assert _group_norm_sum(m, 'HE') == 0
    assert _group_norm_sum(m, 'MIL') > 0 and _group_norm_sum(m, 'PR2') > 0

    return ("A: HE CE→{HE,MIL}; fused CE→{MIL,PR2}, NOT HE")


def test2_variant_b_gradient_flow():
    torch.manual_seed(0)
    m = make_model(detach_he=False).train()
    he, pr = rand_x(3), rand_x(4)

    # (a) HE CE only — B: HE + MIL only
    m.zero_grad()
    out = m.forward_v7([he, pr])
    out['he_logits'].sum().backward()
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        grp = group_of(m, n)
        if grp in ('HE', 'MIL'):
            assert p.grad is not None, f"HE CE must reach {grp} {n}"
        else:
            assert p.grad is None, f"HE CE must NOT reach {grp} {n}"
    assert _group_norm_sum(m, 'HE') > 0 and _group_norm_sum(m, 'MIL') > 0
    assert _group_norm_sum(m, 'PR2') == 0

    # (b) fused CE only — B: HE + MIL + PR2 ALL updated (no isolation)
    m.zero_grad()
    out2 = m.forward_v7([he, pr])
    out2['fused_logits'].sum().backward()
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        grp = group_of(m, n)
        assert p.grad is not None, f"B fused CE must update {grp} {n}"
    assert _group_norm_sum(m, 'HE') > 0
    assert _group_norm_sum(m, 'MIL') > 0
    assert _group_norm_sum(m, 'PR2') > 0

    return ("B: HE CE→{HE,MIL}; fused CE→{HE,MIL,PR2}")


def test3_eval_matches_v3():
    # v3 model as reference
    torch.manual_seed(0)
    m3 = MM_RRT_ABMIL(
        num_modalities=2, modality_list=["HE", "PR"], input_dim=INPUT_DIM,
        mlp_dim=MLP_DIM, num_classes=NUM_CLASSES, dropout=0.25,
        region_num=REGION_NUM, n_layers=1, n_heads=HEADS, drop_path=0.0,
        trans_dropout=0.1, epeg=False, epeg_k=15, crmsa_k=CRMSA_K,
        cr_msa=True, all_shortcut=True, crmsa_heads=CRMSA_HEADS, crmsa_mlp=False,
        fusion_type="two_stage_region", fusion_stage="middle",
        use_gated_fusion=False, abmil_hidden_dim=32,
        use_mclc=False, aggregate_modalities=True,
        stage2_type="he_residual_cross_v3", mil_type="abmil",
        encoder_cfg={
            "HE": {"region_num": REGION_NUM, "epeg_k": 15, "crmsa_k": CRMSA_K,
                   "n_heads": HEADS, "drop_path": 0.0},
            "PR": {"region_num": REGION_NUM, "epeg_k": 15, "crmsa_k": CRMSA_K,
                   "n_heads": HEADS, "drop_path": 0.0},
        },
        stage2_cfg={
            "region_num": REGION_NUM, "crmsa_heads": CRMSA_HEADS,
            "crmsa_k": CRMSA_K, "drop_out": 0.1, "drop_path": 0.0,
            "epeg": False, "epeg_k": 15, "crmsa_mlp": False, "ffn": False,
            "qkv_bias": False, "temperature": 0.2, "residual_scale": 0.1,
            "disable_cross": False, "prototype_momentum": 0.99,
        },
    ).eval()

    he, pr = rand_x(20), rand_x(21)
    with torch.no_grad():
        o3 = m3([he, pr])[0]

    for detach_he, tag in ((True, 'A'), (False, 'B')):
        torch.manual_seed(0)
        m7 = make_model(detach_he=detach_he).eval()
        m7.load_state_dict(m3.state_dict(), strict=True)  # identical architecture
        with torch.no_grad():
            o7 = m7([he, pr])[0]
        diff = (o3 - o7).abs().max().item()
        assert diff < 1e-6, f"v7{tag} eval forward must equal v3, max|Δ|={diff:.2e}"
    return f"v7 A/B eval forward == v3 (max|Δ|={diff:.2e})"


def test4_disable_cross_and_empty_pr():
    torch.manual_seed(0)
    m = make_model(detach_he=False, disable_cross=True).eval()
    he, pr = rand_x(30), rand_x(31)
    with torch.no_grad():
        logits = m([he, pr])[0]
        H = m.rrt_he(m.dp(m.patch_to_emb[0](he)))
        if H.dim() == 2:
            H = H.unsqueeze(0)
        ref = m.mil(H)['logits']
    assert (logits - ref).abs().max() < 1e-6, "disable_cross must give M(Z_HE)"

    cm = HEResidualCrossCRMSAv3(dim=MLP_DIM, num_heads=CRMSA_HEADS,
                                region_num=REGION_NUM, crmsa_k=CRMSA_K,
                                drop_out=0.1, drop_path=0.0, qkv_bias=True,
                                tau=0.2, prototype_momentum=0.99).eval()
    z_he = torch.randn(1, N, MLP_DIM)
    z_pr = torch.randn(1, N, MLP_DIM)
    valid_pr = torch.zeros(1, N, dtype=torch.bool)
    with torch.no_grad():
        out = cm(z_he, z_pr, valid_pr=valid_pr)
    assert torch.equal(out, z_he), "empty PR must yield identity"
    assert torch.isfinite(out).all()
    return "disable_cross → M(Z_HE); empty-PR → identity + finite"


def test5_three_group_coverage():
    m = make_model(detach_he=False)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names, "no trainable params?"
    groups = {}
    for n in names:
        grp = group_of(m, n)
        assert grp != '?', f"param {n} not covered by any of the three groups"
        groups[n] = grp
    # each group non-empty
    for grp in ('HE', 'MIL', 'PR2'):
        assert any(g == grp for g in groups.values()), f"group {grp} empty"
    # union covers all (no leftover) and groups are disjoint by construction
    assert set(groups.values()) == {'HE', 'MIL', 'PR2'}
    return (f"all {len(names)} trainable params covered by HE/MIL/PR2 "
            f"({sum(g=='HE' for g in groups.values())}/"
            f"{sum(g=='MIL' for g in groups.values())}/"
            f"{sum(g=='PR2' for g in groups.values())})")


def test6_full_two_loss_backward_finite():
    y = torch.tensor([1])
    for detach_he, tag in ((True, 'A'), (False, 'B')):
        torch.manual_seed(0)
        m = make_model(detach_he=detach_he).train()
        he, pr = rand_x(40), rand_x(41)
        m.zero_grad()
        out = m.forward_v7([he, pr])
        loss = torch.nn.functional.cross_entropy(out['he_logits'], y) + \
            torch.nn.functional.cross_entropy(out['fused_logits'], y)
        loss.backward()
        n_grad = 0
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            assert p.grad is not None, f"{tag} {n} has no grad"
            assert torch.isfinite(p.grad).all(), f"{tag} non-finite grad {n}"
            n_grad += 1
        assert n_grad > 0
        for grp in ('HE', 'MIL', 'PR2'):
            assert _group_norm_sum(m, grp) > 0, f"{tag} group {grp} has no grad"
        # per-group clip works without touching other groups' grads
        for grp_name, g in [('HE', [p for n, p in m.named_parameters()
                                    if group_of(m, n) == 'HE']),
                            ('MIL', [p for n, p in m.named_parameters()
                                     if group_of(m, n) == 'MIL']),
                            ('PR2', [p for n, p in m.named_parameters()
                                     if group_of(m, n) == 'PR2'])]:
            torch.nn.utils.clip_grad_norm_(g, max_norm=1.0)
    return "A/B two-loss backward finite; three groups receive grad; clip per-group OK"


def main():
    torch.manual_seed(0)
    tests = [
        ("Test 1 — variant A gradient flow", test1_variant_a_gradient_flow),
        ("Test 2 — variant B gradient flow", test2_variant_b_gradient_flow),
        ("Test 3 — eval forward == v3 (A and B)", test3_eval_matches_v3),
        ("Test 4 — disable_cross + empty-PR fallback", test4_disable_cross_and_empty_pr),
        ("Test 5 — three-group coverage", test5_three_group_coverage),
        ("Test 6 — full two-loss backward finite + per-group clip",
         test6_full_two_loss_backward_finite),
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
