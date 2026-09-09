#!/usr/bin/env python3
"""Implementation tests for v8 "HE region queries read PR patch memory" (6 tests).

  Test 1 — patch mode K length: the actual K passed to Wk equals the VALID PR
           patch count (e.g. 50), NOT the routed slot count (48).
  Test 2 — patch mode gradient flow: main CE reaches PR-RRT, Wk/Wv, HE queries
           (Wq + HE routing) and Wout; phi_pr stays unused (grad None, recorded).
  Test 3 — disable_cross / empty-PR strict fallback: identity output, and
           empty-PR backward is finite.
  Test 4 — routed mode keeps v3 behavior: same weights → bit-identical forward
           (module level and model level); state-dict keys identical.
  Test 5 — routed vs patch share the SAME parameter set: same-seed init
           produces identical state dicts (incl. phi_pr kept in patch mode).
  Test 6 — EMA prototype centering: patch mode updates μ_PR once per TRAINING
           forward over valid PR patch value tokens; eval forward frozen.

Run:  python tests/test_he_residual_cross_v8.py
"""
import os, sys
from pathlib import Path

import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from models.mm_rrt_abmil import MM_RRT_ABMIL
from models.he_residual_cross_crmsa_v3 import HEResidualCrossCRMSAv3
from models.he_residual_cross_crmsa_v8 import HEResidualCrossCRMSAv8

INPUT_DIM = 32
MLP_DIM = 64
NUM_CLASSES = 2
HEADS = 4
CRMSA_HEADS = 8
REGION_NUM = 2
CRMSA_K = 2
N_HE = 40
N_PR = 60
N_PR_VALID = 50


def make_model(stage2_type, pr_memory_mode='patch', disable_cross=False):
    model_cfg = dict(
        num_modalities=2,
        modality_list=["HE", "PR"],
        input_dim=INPUT_DIM,
        mlp_dim=MLP_DIM,
        num_classes=NUM_CLASSES,
        dropout=0.25,
        region_num=REGION_NUM, n_layers=1, n_heads=HEADS,
        drop_path=0.0, trans_dropout=0.1,
        epeg=False, epeg_k=15, crmsa_k=CRMSA_K,
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
            "pr_memory_mode": pr_memory_mode,
        },
    )
    return MM_RRT_ABMIL(**model_cfg)


def rand_x(seed, n, d):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n, d, generator=g)


def test1_patch_k_length():
    cm = HEResidualCrossCRMSAv8(
        dim=MLP_DIM, num_heads=CRMSA_HEADS, region_num=REGION_NUM,
        crmsa_k=CRMSA_K, drop_out=0.1, drop_path=0.0, qkv_bias=True,
        tau=0.2, prototype_momentum=0.99, pr_memory_mode='patch').eval()
    z_he = torch.randn(1, N_HE, MLP_DIM)
    z_pr = torch.randn(1, N_PR, MLP_DIM)
    valid_pr = torch.zeros(1, N_PR, dtype=torch.bool)
    valid_pr[:, :N_PR_VALID] = True
    captured = {}

    def _hook(m, inp, out):
        captured['K'] = inp[0].shape[1]      # must return None (else replaces out)

    handle = cm.w_k.register_forward_hook(_hook)
    with torch.no_grad():
        out = cm(z_he, z_pr, valid_pr=valid_pr)
    handle.remove()
    assert captured['K'] == N_PR_VALID, \
        f"patch-mode K must equal valid PR patch count {N_PR_VALID}, " \
        f"got {captured['K']} (routed would be {REGION_NUM**2 * CRMSA_K})"
    assert out.shape == (1, N_HE, MLP_DIM)
    return f"patch-mode K = {captured['K']} == valid PR patch count (not 48-style)"


def test2_patch_gradient_flow():
    torch.manual_seed(0)
    m = make_model('he_residual_cross_v8', pr_memory_mode='patch').train()
    he, pr = rand_x(1, N_HE, INPUT_DIM), rand_x(2, N_PR, INPUT_DIM)
    out = m([he, pr])
    loss = torch.nn.functional.cross_entropy(out[0], torch.tensor([1]))
    loss.backward()

    cm = m.cross_region_mod
    # main CE must reach Wk/Wv/Wout and HE queries (Wq + HE routing)
    for pname in ('w_k.weight', 'w_v.weight', 'w_out.weight', 'w_q.weight'):
        p = cm.get_parameter(pname)
        assert p.grad is not None, f"{pname} must receive main CE gradient"
    assert cm.phi_he.grad is not None, "HE routing phi must receive gradient"
    assert cm.route_norm_pr.weight.grad is not None, \
        "PR path route_norm_pr must receive gradient"

    # PR-RRT gets gradient (sum over its params; note the pre-existing near-zero
    # CR-MSA param grads, other layers carry the signal)
    pr_rrt_grad = sum(float(p.grad.norm().item()) if p.grad is not None else 0.0
                      for n, p in m.named_parameters()
                      if n.startswith('rrt_ihc.'))
    assert pr_rrt_grad > 0, "main CE must reach the PR-RRT encoder"
    he_rrt_grad = sum(float(p.grad.norm().item()) if p.grad is not None else 0.0
                      for n, p in m.named_parameters()
                      if n.startswith('rrt_he.'))
    assert he_rrt_grad > 0, "main CE must reach the HE-RRT (full end-to-end)"

    # phi_pr is kept for init compatibility but unused in patch mode — recorded
    assert cm.phi_pr.grad is None, "patch-mode forward must not use phi_pr"
    return ("CE→PR-RRT/Wk/Wv/HE-queries/Wout; phi_pr unused (grad None, recorded)")


def test3_disable_and_empty_pr():
    # (a) disable_cross → M(Z_HE) identity at the model level (patch mode)
    torch.manual_seed(0)
    m = make_model('he_residual_cross_v8', pr_memory_mode='patch',
                   disable_cross=True).eval()
    he, pr = rand_x(30, N_HE, INPUT_DIM), rand_x(31, N_PR, INPUT_DIM)
    with torch.no_grad():
        logits = m([he, pr])[0]
        H = m.rrt_he(m.dp(m.patch_to_emb[0](he)))
        if H.dim() == 2:
            H = H.unsqueeze(0)
        ref = m.mil(H)['logits']
    assert (logits - ref).abs().max() < 1e-6, "disable_cross must give M(Z_HE)"

    # (b) empty PR → strict identity + finite backward (patch mode)
    cm = HEResidualCrossCRMSAv8(
        dim=MLP_DIM, num_heads=CRMSA_HEADS, region_num=REGION_NUM,
        crmsa_k=CRMSA_K, drop_out=0.1, drop_path=0.0, qkv_bias=True,
        tau=0.2, prototype_momentum=0.99, pr_memory_mode='patch').train()
    z_he = torch.randn(1, N_HE, MLP_DIM, requires_grad=True)
    z_pr = torch.randn(1, N_PR, MLP_DIM, requires_grad=True)
    valid_pr = torch.zeros(1, N_PR, dtype=torch.bool)
    out = cm(z_he, z_pr, valid_pr=valid_pr)
    assert torch.equal(out, z_he), "empty PR must yield strict identity"
    out.sum().backward()
    assert torch.isfinite(z_he.grad).all(), "empty-PR backward must be finite"
    assert z_pr.grad is None or torch.isfinite(z_pr.grad).all(), \
        "empty-PR backward must be finite on PR path"
    return "disable_cross → M(Z_HE); empty PR → strict identity + finite backward"


def test4_routed_matches_v3():
    # module level: same state dict → bit-identical output
    torch.manual_seed(0)
    v3 = HEResidualCrossCRMSAv3(dim=MLP_DIM, num_heads=CRMSA_HEADS,
                                region_num=REGION_NUM, crmsa_k=CRMSA_K,
                                drop_out=0.1, drop_path=0.0, qkv_bias=True,
                                tau=0.2, prototype_momentum=0.99).eval()
    torch.manual_seed(0)
    v8r = HEResidualCrossCRMSAv8(dim=MLP_DIM, num_heads=CRMSA_HEADS,
                                 region_num=REGION_NUM, crmsa_k=CRMSA_K,
                                 drop_out=0.1, drop_path=0.0, qkv_bias=True,
                                 tau=0.2, prototype_momentum=0.99,
                                 pr_memory_mode='routed').eval()
    assert set(v3.state_dict().keys()) == set(v8r.state_dict().keys()), \
        "routed-mode state-dict keys must equal v3"
    v8r.load_state_dict(v3.state_dict(), strict=True)
    z_he = torch.randn(1, N_HE, MLP_DIM)
    z_pr = torch.randn(1, N_PR, MLP_DIM)
    with torch.no_grad():
        o3 = v3(z_he, z_pr)
        o8 = v8r(z_he, z_pr)
    assert (o3 - o8).abs().max() < 1e-6, "routed-mode forward must equal v3"

    # model level: eval dispatch of stage2_type v8-routed == v3
    torch.manual_seed(0)
    m3 = make_model('he_residual_cross_v3').eval()
    torch.manual_seed(0)
    m8 = make_model('he_residual_cross_v8', pr_memory_mode='routed').eval()
    m8.load_state_dict(m3.state_dict(), strict=True)
    he, pr = rand_x(20, N_HE, INPUT_DIM), rand_x(21, N_PR, INPUT_DIM)
    with torch.no_grad():
        l3 = m3([he, pr])[0]
        l8 = m8([he, pr])[0]
    assert (l3 - l8).abs().max() < 1e-6, "v8-routed model eval must equal v3"
    return "routed mode ≡ v3 bit-identical (module + model level)"


def test5_same_seed_identical_init():
    torch.manual_seed(0)
    mr = make_model('he_residual_cross_v8', pr_memory_mode='routed')
    torch.manual_seed(0)
    mp = make_model('he_residual_cross_v8', pr_memory_mode='patch')
    sd_r, sd_p = mr.state_dict(), mp.state_dict()
    assert set(sd_r.keys()) == set(sd_p.keys()), "param sets must be identical"
    for k in sd_r:
        assert torch.equal(sd_r[k], sd_p[k]), f"init mismatch at {k}"
    assert 'cross_region_mod.phi_pr' in sd_p, \
        "phi_pr must be kept in patch mode for init/checkpoint compatibility"
    return "routed/patch same-seed init identical (phi_pr kept in patch mode)"


def test6_ema_patch_mode():
    cm = HEResidualCrossCRMSAv8(
        dim=MLP_DIM, num_heads=CRMSA_HEADS, region_num=REGION_NUM,
        crmsa_k=CRMSA_K, drop_out=0.1, drop_path=0.0, qkv_bias=True,
        tau=0.2, prototype_momentum=0.99, pr_memory_mode='patch').train()
    z_he = torch.randn(1, N_HE, MLP_DIM)
    z_pr = torch.randn(1, N_PR, MLP_DIM)
    valid_pr = torch.ones(1, N_PR, dtype=torch.bool)

    mu0 = cm.mu_pr.clone()
    _ = cm(z_he, z_pr, valid_pr=valid_pr)          # one training forward
    assert not torch.equal(cm.mu_pr, mu0), \
        "training forward must update μ_PR (over valid PR patch value tokens)"
    assert cm.prototype_initialized.item() == 1.0

    cm.eval()
    mu1 = cm.mu_pr.clone()
    with torch.no_grad():
        _ = cm(z_he, z_pr, valid_pr=valid_pr)      # eval forward
    assert torch.equal(cm.mu_pr, mu1), "eval forward must NOT update μ_PR"
    return "patch-mode EMA: train-update / eval-frozen (β=0.99, patch-token mean)"


def main():
    torch.manual_seed(0)
    tests = [
        ("Test 1 — patch K length == valid PR patch count", test1_patch_k_length),
        ("Test 2 — patch gradient flow (PR-RRT/Wk/Wv/HE-queries/Wout)",
         test2_patch_gradient_flow),
        ("Test 3 — disable_cross + empty-PR strict fallback",
         test3_disable_and_empty_pr),
        ("Test 4 — routed mode keeps v3 behavior", test4_routed_matches_v3),
        ("Test 5 — routed/patch same-seed identical init", test5_same_seed_identical_init),
        ("Test 6 — EMA prototype (patch mode)", test6_ema_patch_mode),
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
