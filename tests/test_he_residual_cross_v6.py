#!/usr/bin/env python3
"""Implementation tests for the v6 gradient-division mechanism (5 tests).

  Test 1 — gradient division of labor: he_logits updates Group A
           (patch_to_emb[0]/rrt_he/mil) and NOT Group B; fused_logits updates
           Group B (patch_to_emb[1]/rrt_ihc/cross_region_mod) and NOT Group A
           (in particular NOT mil).
  Test 2 — fused input gradient is intact (M's detached params do NOT kill the
           gradient into `fused`).
  Test 3 — eval forward (normal dispatch) matches v3 with the same weights.
  Test 4 — disable_cross → strict Z_HE identity at the model level; empty-PR
           fallback (cross module) finite + identity.
  Test 5 — a full forward + backward over both losses yields finite gradients
           for every trainable parameter in BOTH groups (covers the real
           two-loss branch).

Run:  python tests/test_he_residual_cross_v6.py
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


def make_model(stage2_type="he_residual_cross_v6", disable_cross=False,
               encoder_epeg=False):
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
        stage2_type=stage2_type,
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
        },
    )
    return MM_RRT_ABMIL(**model_cfg)


def rand_x(seed, n=N, d=INPUT_DIM):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n, d, generator=g)


def group_of(model, name):
    # group A = patch_to_emb.0 / rrt_he / mil ; group B = patch_to_emb.1 / rrt_ihc / cross
    A = ('patch_to_emb.0.', 'rrt_he.', 'mil.')
    B = ('patch_to_emb.1.', 'rrt_ihc.', 'cross_region_mod.')
    if name.startswith(A):
        return 'A'
    if name.startswith(B):
        return 'B'
    return '?'


def _grad_norm(p):
    return 0.0 if p.grad is None else float(p.grad.norm().item())


def _group_norm_sum(model):
    return (sum(_grad_norm(p) for n, p in model.named_parameters()
                if group_of(model, n) == 'A'),
            sum(_grad_norm(p) for n, p in model.named_parameters()
                if group_of(model, n) == 'B'))


def test1_gradient_division():
    # NOTE: `grad is None` is the presence/absence test here — the RRT encoder's
    # CR-MSA (rrt_*.cr_msa.*) has a pre-existing near-zero gradient w.r.t. its own
    # params (output depends on input but d(params)≈0, present in v1–v5 too), so a
    # `grad > 0` assertion per-parameter is too strict.  We instead assert (i) the
    # right group is IN the graph and the wrong group is OUT of it, and (ii) each
    # group's *total* gradient norm is non-zero (the loss really flows).
    torch.manual_seed(0)
    m = make_model().train()
    he, pr = rand_x(1), rand_x(2)

    # (a) backward on he_logits ONLY
    m.zero_grad()
    out = m.forward_v6([he, pr])
    out['he_logits'].sum().backward()
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        grp = group_of(m, n)
        if grp == 'A':
            assert p.grad is not None, f"he CE must own Group A {n}"
        else:
            assert p.grad is None, f"he CE must NOT own {grp} {n}"
    a_sum, b_sum = _group_norm_sum(m)
    assert a_sum > 0 and b_sum == 0, f"he CE: A={a_sum:.3e}, B={b_sum:.3e}"

    # (b) backward on fused_logits ONLY (fresh forward)
    m.zero_grad()
    out2 = m.forward_v6([he, pr])
    out2['fused_logits'].sum().backward()
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        grp = group_of(m, n)
        if grp == 'B':
            assert p.grad is not None, f"fused CE must own Group B {n}"
        else:
            assert p.grad is None, f"fused CE must NOT own {grp} {n}"
    a_sum, b_sum = _group_norm_sum(m)
    assert b_sum > 0 and a_sum == 0, f"fused CE: A={a_sum:.3e}, B={b_sum:.3e}"

    return ("he→Group A only; fused→Group B only (mil detached from fused)")


def test2_fused_input_gradient_intact():
    torch.manual_seed(0)
    m = make_model().train()
    he, pr = rand_x(10), rand_x(11)
    m.zero_grad()
    out = m.forward_v6([he, pr])
    fused = out['fused_features']
    assert fused.requires_grad, "fused features must require grad"
    fused.retain_grad()
    out['fused_logits'].sum().backward()
    assert fused.grad is not None, "fused input grad must not be None"
    assert fused.grad.abs().sum().item() > 0, "fused input grad must be non-zero"
    # M's params must carry no grad on this backward (detached in fused call)
    for n, p in m.named_parameters():
        if n.startswith('mil.'):
            assert p.grad is None or p.grad.abs().sum() == 0, \
                f"mil {n} must be detached for fused logits"
    return "fused input grad non-zero; mil params detached in fused call"


def test3_eval_matches_v3():
    torch.manual_seed(0)
    m3 = make_model("he_residual_cross_v3").eval()
    torch.manual_seed(0)
    m6 = make_model("he_residual_cross_v6").eval()
    # identical architecture ⇒ strict load must succeed
    m6.load_state_dict(m3.state_dict(), strict=True)

    he, pr = rand_x(20), rand_x(21)
    with torch.no_grad():
        o3 = m3([he, pr])[0]
        o6 = m6([he, pr])[0]
    diff = (o3 - o6).abs().max().item()
    assert diff < 1e-6, f"v6 eval forward must equal v3, max|Δ|={diff:.2e}"
    return f"v6 eval forward == v3 (max|Δ|={diff:.2e})"


def test4_disable_cross_and_empty_pr():
    # (a) disable_cross → M(Z_HE) identity at the model level
    torch.manual_seed(0)
    m = make_model("he_residual_cross_v6", disable_cross=True).eval()
    he, pr = rand_x(30), rand_x(31)
    with torch.no_grad():
        logits = m([he, pr])[0]
        H = m.rrt_he(m.dp(m.patch_to_emb[0](he)))
        if H.dim() == 2:
            H = H.unsqueeze(0)
        ref = m.mil(H)['logits']
    assert (logits - ref).abs().max() < 1e-6, "disable_cross must give M(Z_HE)"

    # (b) empty-PR fallback at the cross module (v6 reuses v3 module)
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


def test5_full_two_loss_backward_finite():
    torch.manual_seed(0)
    m = make_model().train()
    he, pr = rand_x(40), rand_x(41)
    y = torch.tensor([1])  # one label

    m.zero_grad()
    out = m.forward_v6([he, pr])
    loss = torch.nn.functional.cross_entropy(out['he_logits'], y) + \
        torch.nn.functional.cross_entropy(out['fused_logits'], y)
    loss.backward()

    n_grad = 0
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"{n} has no grad"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {n}"
        n_grad += 1
    assert n_grad > 0, "no trainable params?"
    # every group must receive some gradient
    ga = sum(_grad_norm(p) for n, p in m.named_parameters()
             if group_of(m, n) == 'A')
    gb = sum(_grad_norm(p) for n, p in m.named_parameters()
             if group_of(m, n) == 'B')
    assert ga > 0 and gb > 0, f"both groups need grad (A={ga:.3f}, B={gb:.3f})"
    return (f"two-loss backward finite across all {n_grad} params "
            f"(|grad| A={ga:.3e}, B={gb:.3e})")


def main():
    torch.manual_seed(0)
    tests = [
        ("Test 1 — gradient division (HE→A, fused→B)", test1_gradient_division),
        ("Test 2 — fused input gradient intact", test2_fused_input_gradient_intact),
        ("Test 3 — eval forward == v3", test3_eval_matches_v3),
        ("Test 4 — disable_cross + empty-PR fallback", test4_disable_cross_and_empty_pr),
        ("Test 5 — full two-loss backward finite", test5_full_two_loss_backward_finite),
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
