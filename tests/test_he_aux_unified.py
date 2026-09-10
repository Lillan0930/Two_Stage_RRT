#!/usr/bin/env python3
"""Verification tests for the HE-anchored unified model (`models/he_aux_unified.py`).

  Test 1 — dual regression: legacy v3 vs unified on identical HE/PR weights,
           identical input, eval → fused features AND ABMIL logits identical
           (max abs error reported)
  Test 2 — HE-only regression: unified HE-only == single-modality path with the
           same projection/RRT/MIL weights, and HE/MIL init is independent of
           how many auxiliary branches exist
  Test 3 — multi-modal entry: 1 / 2 / 4 auxiliary stains; every encoder and
           cross branch executes; HE encoded exactly once; every parameter is in
           the optimizer with no duplicates; gradients exist and are finite
  Test 4 — EMA independence: per-stain prototype buffers are distinct objects,
           updated in train mode only for fed stains, frozen in eval
  Test 5 — MIL decoupling: a logits-only dummy head drives the interface and a
           full training step; the main model never touches ABMIL internals
  Test 6 — config & restore: stain name mapping, missing/mismatched input errors,
           token masks, checkpoint save/reload, v3 mapping completeness checks
  Test 7 — model_config-only restore: `config_schema_version` round-trip, rebuild
           from the checkpoint's own config with strict=True, reproducible init
           hash, aux-count-independent initialisation
  Test 8 — feature caching & mask policy: `return_features` returns the tensors
           the forward actually used, all-True/None masks accepted, any-False
           mask refused
  Test 9 — seeds & output dirs: explicit-vs-inherited seed resolution, empty
           output dirs resolving under the repo root, optimizer coverage check

Run:  python tests/test_he_aux_unified.py
      python tests/test_he_aux_unified.py --v3-ckpt /path/to/best_model.pt --device cuda:2

Paths are resolved from this file's location (repo root), never hardcoded — the
v3 checkpoint used by Test 1 is a CLI argument and Test 1 skips with an explicit
message when it is absent.
"""
import os, sys, copy, math, shutil, tempfile, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from models.he_aux_unified import (
    HEAuxUnifiedModel, build_he_aux_unified, build_he_aux_unified_from_config,
    map_v3_state_dict_to_unified, load_v3_checkpoint_into_unified,
    HISTORICAL_V3_ENCODER_CFG, HISTORICAL_V3_STAGE2_CFG,
)
from models.mil_heads import (
    build_mil_head, register_head_adapter, MILHeadAdapter, available_heads,
)
from models.mil_registry import register_mil

DEFAULT_V3_CKPT = (PROJECT / "results" / "stage2_he_residual_cross_v3"
                   / "he_residual_cross_v3" / "seed42" / "ckpt" / "best_model.pt")
#: overridable via `--v3-ckpt`; Test 1 skips (it does not fail) when missing
V3_CKPT = DEFAULT_V3_CKPT

INPUT_DIM = 768
MLP_DIM = 512
NUM_CLASSES = 2
N_HE = 289          # not a perfect square → exercises region padding
N_AUX = 256

# Small geometry for the multi-stain tests (keeps them fast).  The *cross
# branches* still use the historical stage2 geometry (crmsa_heads=8,
# region_num=4, crmsa_k=3) — only the Stage-1 encoders are shrunk here.
# NB: epeg_k must be odd — the EPEG conv uses k//2 padding, so an even kernel
# would grow the attention map by one (the historical configs use 9 / 15).
SMALL = dict(input_dim=64, mlp_dim=64, num_classes=2, dropout=0.1,
             region_num=2, n_layers=2, n_heads=4, epeg=True, epeg_k=5,
             crmsa_k=3, crmsa_heads=4, trans_dropout=0.0)
SMALL_ENC = {'region_num': 2, 'epeg_k': 5, 'crmsa_k': 3, 'n_heads': 4,
             'drop_path': 0.0}
SMALL_STRUCT = {k: SMALL[k] for k in
                ('region_num', 'n_layers', 'n_heads', 'epeg', 'epeg_k',
                 'crmsa_k', 'crmsa_heads', 'trans_dropout')}


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _inputs(stains, b=1, seed=100):
    return {s: _rand(b, N_HE if s == 'HE' else N_AUX, INPUT_DIM, seed=seed + i)
            for i, s in enumerate(stains)}


def _small_inputs(stains, b=1, seed=100):
    return {s: _rand(b, 36 if s == 'HE' else 25, 64, seed=seed + i)
            for i, s in enumerate(stains)}


def _small_model(mods, **extra):
    """Unified model with HE *and* every stain pinned to the small geometry."""
    enc = {m: dict(SMALL_ENC) for m in mods}
    return build_he_aux_unified(mods, encoder_cfg=enc, **{**SMALL, **extra})


def _backward_report(model, loss):
    """Backward `loss`, then summarize which params have finite grads."""
    loss.backward()
    have, missing, nonfinite = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all():
            nonfinite.append(name)
        else:
            have.append(name)
    return have, missing, nonfinite


def _count_forward_calls(model):
    """Hook every Stage-1 encoder and every cross branch; return the counter."""
    calls, hooks = {}, []
    for name, mod in model.rrt.items():
        key = ('rrt', name)
        calls[key] = 0
        hooks.append(mod.register_forward_hook(
            lambda *a, k=key: calls.__setitem__(k, calls[k] + 1)))
    for name, mod in model.cross_branches.items():
        key = ('cross', name)
        calls[key] = 0
        hooks.append(mod.register_forward_hook(
            lambda *a, k=key: calls.__setitem__(k, calls[k] + 1)))
    return calls, hooks


# ---------------------------------------------------------------------------
# legacy v3 model (exactly as scripts/run_stage2_v3.py builds it)
# ---------------------------------------------------------------------------
def build_legacy_v3(model_cfg, data_cfg):
    from models.mm_rrt_abmil import MM_RRT_ABMIL
    mc, dc = model_cfg, data_cfg
    return MM_RRT_ABMIL(
        num_modalities=len(dc['modalities']), modality_list=dc['modalities'],
        input_dim=dc['input_dim'], mlp_dim=mc['mlp_dim'],
        num_classes=dc['num_classes'], dropout=mc['dropout'],
        region_num=mc['region_num'], n_layers=mc['n_layers'],
        n_heads=mc['n_heads'], drop_path=mc['drop_path'],
        trans_dropout=mc['trans_dropout'], epeg=mc['epeg'], epeg_k=mc['epeg_k'],
        crmsa_k=mc['crmsa_k'], cr_msa=mc['cr_msa'],
        all_shortcut=mc['all_shortcut'], crmsa_heads=mc['crmsa_heads'],
        crmsa_mlp=mc['crmsa_mlp'], fusion_type=mc['fusion_type'],
        fusion_stage=mc['fusion_stage'], stage2_type=mc['stage2_type'],
        encoder_cfg=mc['encoder_cfg'], stage2_cfg=mc['stage2_cfg'],
        mil_type=mc['mil_type'], abmil_hidden_dim=mc['abmil_hidden_dim'],
    )


def legacy_fused_and_logits(legacy, x_he, x_pr):
    """Fused HE tokens + ABMIL logits from the legacy two-stage forward."""
    with torch.no_grad():
        he_emb = legacy.dp(legacy.patch_to_emb[0](x_he))
        pr_emb = legacy.dp(legacy.patch_to_emb[1](x_pr))
        z_he = legacy.rrt_he(he_emb)
        z_pr = legacy.rrt_ihc(pr_emb)
        fused = legacy.cross_region_mod(z_he, z_pr)
        logits = legacy.mil(fused)['logits']
    return fused, logits


# ---------------------------------------------------------------------------
# Test 1 — dual regression against legacy v3
# ---------------------------------------------------------------------------
def test1_dual_regression():
    if not V3_CKPT.exists():
        return f"SKIP (no v3 checkpoint at {V3_CKPT})"

    ckpt = torch.load(V3_CKPT, map_location='cpu', weights_only=False)
    mc, dc = ckpt['config']['model'], ckpt['config']['data']
    old_sd = ckpt['model_state_dict']

    legacy = build_legacy_v3(mc, dc)
    legacy.load_state_dict(old_sd)
    legacy.eval()

    unified = build_he_aux_unified_from_config(mc, dc)
    report = load_v3_checkpoint_into_unified(unified, str(V3_CKPT), verbose=False)
    unified.eval()

    # the mapping must be a complete bijection legacy keys ↔ unified keys
    mapped, unmapped = map_v3_state_dict_to_unified(old_sd, 'HE', 'PR')
    assert not unmapped, f"unmapped legacy keys: {unmapped}"
    assert set(mapped) == set(unified.state_dict()), (
        f"key sets differ: missing="
        f"{sorted(set(unified.state_dict()) - set(mapped))[:4]}, extra="
        f"{sorted(set(mapped) - set(unified.state_dict()))[:4]}")
    assert report['n_mapped'] == len(old_sd)

    x_he = _rand(1, N_HE, INPUT_DIM, seed=7)
    x_pr = _rand(1, N_AUX, INPUT_DIM, seed=8)

    fused_leg, logits_leg = legacy_fused_and_logits(legacy, x_he, x_pr)
    with torch.no_grad():
        fused_uni, _, _ = unified.fuse({'HE': x_he, 'PR': x_pr})
        logits_uni = unified([x_he, x_pr])[0]

    err_fused = (fused_leg - fused_uni).abs().max().item()
    err_logits = (logits_leg - logits_uni).abs().max().item()
    assert err_fused == 0.0, f"fused features differ: max|Δ|={err_fused:.3e}"
    assert err_logits == 0.0, f"ABMIL logits differ: max|Δ|={err_logits:.3e}"

    # multi-slide batch
    x_he_b = _rand(3, N_HE, INPUT_DIM, seed=11)
    x_pr_b = _rand(3, N_AUX, INPUT_DIM, seed=12)
    with torch.no_grad():
        l_leg = legacy([x_he_b, x_pr_b])[0]
        l_uni = unified([x_he_b, x_pr_b])[0]
    err_batch = (l_leg - l_uni).abs().max().item()
    assert err_batch == 0.0, f"batched logits differ {err_batch:.3e}"

    # fully random weights (no checkpoint at all) — proves it is the *structure*
    # that matches, not merely the loaded tensors
    torch.manual_seed(3)
    legacy2 = build_legacy_v3(mc, dc).eval()
    unified2 = build_he_aux_unified_from_config(mc, dc).eval()
    m2, u2 = map_v3_state_dict_to_unified(legacy2.state_dict(), 'HE', 'PR')
    assert not u2, f"unmapped keys: {u2}"
    unified2.load_state_dict(m2, strict=True)
    _, logits_leg2 = legacy_fused_and_logits(legacy2, x_he, x_pr)
    with torch.no_grad():
        logits_uni2 = unified2([x_he, x_pr])[0]
    err_rand = (logits_leg2 - logits_uni2).abs().max().item()
    assert err_rand == 0.0, f"random-weight logits mismatch {err_rand:.3e}"

    # sanity: the trained checkpoint is not degenerate
    gap = (logits_leg[:, 0] - logits_leg[:, 1]).abs().max().item()
    assert gap > 1e-3, f"legacy logits are (near-)tied: gap={gap:.3e}"

    return (f"max|Δ| fused = {err_fused:.3e}, logits = {err_logits:.3e} "
            f"(trained ckpt, B=1); logits = {err_batch:.3e} (B=3); "
            f"random-weight logits = {err_rand:.3e}; "
            f"{report['n_mapped']} tensors mapped bijectively; "
            f"legacy |logit gap| = {gap:.3f}")


# ---------------------------------------------------------------------------
# Test 2 — HE-only regression + init independence
# ---------------------------------------------------------------------------
def test2_he_only_regression():
    he_only = _small_model(['HE']).eval()
    he_pr = _small_model(['HE', 'PR']).eval()
    full = _small_model(['HE', 'ER', 'PR', 'HER2', 'Ki67']).eval()

    # (a) shared-module init must not depend on the number of auxiliary branches
    sd_he = he_only.state_dict()
    for other, label in ((he_pr, 'HE+PR'), (full, 'HE+4aux')):
        sd_o = other.state_dict()
        for k, v in sd_he.items():
            assert k in sd_o, f"{label}: missing key {k}"
            assert torch.equal(v, sd_o[k]), f"{label}: HE/MIL init changed at {k}"
    assert torch.equal(he_only.mil.module.classifier[0].weight,
                       full.mil.module.classifier[0].weight), \
        "MIL classifier init must be independent of auxiliary branches"

    # (b) HE-only forward == H, and == MIL(H) of a model that also has branches
    x = _small_inputs(['HE'], seed=21)['HE']
    with torch.no_grad():
        logits_only = he_only([x])[0]
        H = he_pr.encode_he({'HE': x,
                             'PR': _small_inputs(['PR'], seed=31)['PR']})
        logits_ref = he_pr.mil(tokens=H, mask=None)['logits']
        fused_only, H_only, branches = he_only.fuse({'HE': x})
    assert branches == {}, "HE-only model must have no cross branches"
    assert torch.equal(fused_only, H_only), "HE-only fused output must be H"
    assert torch.equal(logits_only, logits_ref), \
        "HE-only logits must equal MIL(H) of the HE+PR model"

    # (c) cross-model: same weights into a legacy single-modality MM_RRT_ABMIL
    from models.mm_rrt_abmil import MM_RRT_ABMIL
    legacy1 = MM_RRT_ABMIL(
        num_modalities=1, modality_list=['HE'], input_dim=SMALL['input_dim'],
        mlp_dim=SMALL['mlp_dim'], num_classes=SMALL['num_classes'],
        dropout=SMALL['dropout'], region_num=SMALL_ENC['region_num'],
        n_layers=SMALL['n_layers'], n_heads=SMALL_ENC['n_heads'],
        trans_dropout=SMALL['trans_dropout'], epeg=SMALL['epeg'],
        epeg_k=SMALL_ENC['epeg_k'], crmsa_k=SMALL_ENC['crmsa_k'],
        crmsa_heads=SMALL['crmsa_heads'], crmsa_mlp=False,
        all_shortcut=True, drop_path=SMALL_ENC['drop_path'],
        fusion_type='two_stage_region', mil_type='abmil',
        abmil_hidden_dim=256).eval()

    sd_only = he_only.state_dict()
    for k, v in legacy1.state_dict().items():
        if k.startswith('patch_to_emb.0.'):
            src = 'patch_to_emb.HE.' + k[len('patch_to_emb.0.'):]
        elif k.startswith('rrt_he.'):
            src = 'rrt.HE.' + k[len('rrt_he.'):]
        elif k.startswith('mil.'):
            src = 'mil.module.' + k[len('mil.'):]
        else:
            raise AssertionError(f"unexpected legacy single-modality key {k}")
        assert src in sd_only, f"missing {src}"
        v.copy_(sd_only[src])

    with torch.no_grad():
        logits_legacy1 = legacy1([x])[0]
    err = (logits_legacy1 - logits_only).abs().max().item()
    assert err == 0.0, f"HE-only vs single-modality path differ: {err:.3e}"

    return (f"HE-only == MIL(H) exactly; legacy single-modality Δ = {err:.3e}; "
            f"HE/MIL init identical across HE-only / HE+PR / HE+4aux")


# ---------------------------------------------------------------------------
# Test 3 — multi-modal entry (1 / 2 / 4 auxiliary stains)
# ---------------------------------------------------------------------------
def test3_multi_modal_entry():
    cases = [
        (['HE', 'PR'], 'PR'),
        (['HE', 'ER', 'PR'], 'ER,PR'),
        (['HE', 'ER', 'PR', 'HER2', 'Ki67'], 'ER,PR,HER2,Ki67'),
    ]
    msgs = []
    for mods, label in cases:
        model = _small_model(mods)
        aux = [m for m in mods if m != 'HE']
        feats = _small_inputs(mods, seed=41)
        named = dict(model.named_parameters())

        # ── every encoder / cross branch runs exactly once ──
        calls, hooks = _count_forward_calls(model)
        model.eval()
        with torch.no_grad():
            logits = model(feats)[0]
        for h in hooks:
            h.remove()

        assert logits.shape == (1, NUM_CLASSES), f"{label}: bad logits shape"
        assert calls[('rrt', 'HE')] == 1, \
            f"{label}: HE must be encoded exactly once, " \
            f"got {calls[('rrt', 'HE')]}"
        assert ('cross', 'HE') not in calls, "HE must not have a cross branch"
        for s in aux:
            assert calls[('rrt', s)] == 1, \
                f"{label}: rrt[{s}] ran {calls.get(('rrt', s))}×"
            assert calls[('cross', s)] == 1, \
                f"{label}: cross[{s}] ran {calls.get(('cross', s))}×"

        # ── optimizer coverage: every param, no duplicates ──
        params = list(model.parameters())
        assert all(p.requires_grad for p in params), f"{label}: frozen params"
        ids = [id(p) for p in params]
        assert len(ids) == len(set(ids)), f"{label}: duplicate parameter objects"
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        opt_ids = {id(p) for g in opt.param_groups for p in g['params']}
        assert opt_ids == set(ids), \
            f"{label}: optimizer covers {len(opt_ids)}/{len(ids)} tensors"
        for s in aux:
            assert any(n.startswith(f'cross_branches.{s}.') for n in named), \
                f"{label}: no parameters for cross branch {s}"

        # ── gradients exist and are finite ──
        model.train()
        loss = F.cross_entropy(model(feats)[0], torch.tensor([1]))
        have, missing, nonfinite = _backward_report(model, loss)
        assert not nonfinite, f"{label}: non-finite grads {nonfinite[:3]}"
        assert not missing, f"{label}: params without grad {missing[:3]}"
        for s in aux:
            assert any(n.startswith(f'cross_branches.{s}.') for n in have), \
                f"{label}: no grad flowed into cross branch {s}"
        assert any(n.startswith('rrt.HE.') for n in have), f"{label}: no HE grad"
        assert any(n.startswith('patch_to_emb.HE.') for n in have), \
            f"{label}: no HE projection grad"
        assert any(n.startswith('mil.') for n in have), f"{label}: no MIL grad"

        # ── structural checks in eval (dropout off ⇒ deterministic) ──
        model.eval()
        with torch.no_grad():
            H_before = model.encode_he(feats)
            fused, H_after, branches = model.fuse(feats)
            assert torch.equal(H_before, H_after), \
                f"{label}: the HE representation was modified by a branch"
            assert set(branches) == set(aux), f"{label}: wrong branch set"

            # fused == mean of complete branch outputs (no extra H, no 2nd scale)
            mean_branch = torch.stack([branches[s] for s in aux]).mean(0)
            assert torch.equal(fused, mean_branch), \
                f"{label}: fusion formula broken"
            manual = H_after + sum(branches[s] - H_after for s in aux) / len(aux)
            assert torch.allclose(fused, manual, atol=1e-5), \
                f"{label}: H + (1/M)·Σ(branch_m − H) mismatch: " \
                f"{(fused - manual).abs().max().item():.3e}"
            if len(aux) == 1:
                assert torch.equal(fused, branches[aux[0]]), \
                    f"{label}: single aux must degenerate to the v3 branch"

        # ── the exact H + (residual_scale/M)·Σ Δ_m identity ──
        # A second, weight-identical model whose branches use residual_scale=1.0
        # exposes Δ_m directly as `branch_m(H, Z_m) - H`; this pins down both the
        # 1/M factor and the fact that residual_scale is applied exactly once.
        m1 = _small_model(mods, stage2_cfg={**model.stage2_cfg,
                                            'residual_scale': 1.0}).eval()
        # identical parameters (the two models are the same model, only the
        # non-learnable residual_scale buffer differs)
        p_a, p_b = dict(model.named_parameters()), dict(m1.named_parameters())
        assert set(p_a) == set(p_b)
        for k in p_a:
            assert torch.equal(p_a[k], p_b[k]), f"{label}: weights differ at {k}"
        # the EMA prototypes are buffers, already moved by the train-mode step
        # above — carry them over so the two models see the same centering
        for k, v in model.state_dict().items():
            if k in p_a or k.endswith('residual_scale'):
                continue
            m1.state_dict()[k].copy_(v)
        with torch.no_grad():
            fused1, H1, br1 = m1.fuse(feats)
            scale = model.stage2_cfg['residual_scale']
            deltas = {s: br1[s] - H1 for s in aux}
            formula = H1 + (scale / len(aux)) * sum(deltas[s] for s in aux)
        assert torch.allclose(fused, formula, atol=1e-5), \
            f"{label}: formal fusion identity violated by " \
            f"{(fused - formula).abs().max().item():.3e}"
        for s in aux:
            assert deltas[s].abs().max() > 0, f"{label}: branch {s} residual is 0"

            # every auxiliary branch actually influences the prediction
            base = model(feats)[0]
            for s in aux:
                pert = dict(feats)
                pert[s] = pert[s] + 1.0
                d = (model(pert)[0] - base).abs().max().item()
                assert d > 1e-6, f"{label}: auxiliary stain {s} has no effect"

        msgs.append(f"{len(aux)}aux({label}): {len(have)} grads finite, "
                    f"{len(ids)} params in optimizer, all branches live")

    return ' | '.join(msgs)


# ---------------------------------------------------------------------------
# Test 4 — EMA prototype independence
# ---------------------------------------------------------------------------
def test4_ema_independence():
    mods = ['HE', 'ER', 'PR']
    model = _small_model(mods)

    # distinct buffer objects per auxiliary stain
    bufs = dict(model.named_buffers())
    names = [n for n in bufs if n.endswith('mu_pr')]
    flags = [n.replace('mu_pr', 'prototype_initialized') for n in names]
    assert len(names) == len(model.aux_stains), \
        f"expected one mu_pr per aux stain, got {names}"
    assert all(f in bufs for f in flags), f"missing flags {flags}"
    assert len({id(bufs[n]) for n in names}) == len(names), "mu_pr buffers shared"
    assert len({id(bufs[n]) for n in flags}) == len(names), "flag buffers shared"
    assert 'HE' not in model.cross_branches, "HE needs no cross branch"

    feats = _small_inputs(mods, seed=51)
    model.train()
    model(feats)
    for s in mods[1:]:
        assert bool(model.cross_branches[s].prototype_initialized.item()), \
            f"{s}: prototype must initialize on the first train forward"
    mu1 = {s: model.cross_branches[s].mu_pr.clone() for s in mods[1:]}

    model(_small_inputs(mods, seed=61))
    moved = [s for s in mods[1:]
             if not torch.equal(mu1[s], model.cross_branches[s].mu_pr)]
    assert sorted(moved) == sorted(mods[1:]), \
        f"every branch must update its own prototype; moved={moved}"
    assert not torch.equal(model.cross_branches['ER'].mu_pr,
                           model.cross_branches['PR'].mu_pr), \
        "different stains must learn different prototypes"

    model.eval()
    frozen = {s: model.cross_branches[s].mu_pr.clone() for s in mods[1:]}
    with torch.no_grad():
        for _ in range(3):
            model(_small_inputs(mods, seed=71))
    for s in mods[1:]:
        assert torch.equal(frozen[s], model.cross_branches[s].mu_pr), \
            f"{s}: prototype must be frozen in eval"

    # buffers are checkpointed but never optimized
    assert not any('mu_pr' in n for n in dict(model.named_parameters())), \
        "mu_pr must not be a parameter"
    in_opt = {id(p) for g in torch.optim.Adam(model.parameters(),
                                              lr=1e-3).param_groups
              for p in g['params']}
    assert not any(id(bufs[n]) in in_opt for n in names + flags), \
        "prototype buffers must not be in the optimizer"

    # single-aux models: the only branch is the one that gets fed
    for s, seed in (('ER', 81), ('PR', 91)):
        m1 = _small_model(['HE', s]).train()
        m1(_small_inputs(['HE', s], seed=seed))
        assert bool(m1.cross_branches[s].prototype_initialized.item())

    return (f"{len(names)} independent prototypes "
            f"({', '.join(n.split('.')[1] for n in names)}); all updated in "
            f"train, all frozen in eval; ER != PR; none in the optimizer")


# ---------------------------------------------------------------------------
# Test 5 — MIL pluggability / decoupling from ABMIL internals
# ---------------------------------------------------------------------------
class _DummyMIL(nn.Module):
    """Logits-only MIL: mean-pool over tokens → linear.

    Deliberately has no `attention`, no `classifier` and no `input_dim` — the
    main model must not depend on any of them.
    """

    def __init__(self, input_dim=64, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return {'logits': self.proj(x.mean(dim=1))}


class _DummyHead(MILHeadAdapter):
    def forward(self, tokens, mask=None):
        if mask is not None:
            x = tokens * mask.unsqueeze(-1).to(tokens.dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(tokens.dtype)
            x = x.sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
        else:
            x = tokens
        return {'logits': self.module(x)['logits']}     # logits ONLY


class _NoAdapterMIL(nn.Module):
    def __init__(self, input_dim=64, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return {'logits': self.proj(x.mean(dim=1))}


register_mil('dummy_logits_only')(_DummyMIL)
register_head_adapter('dummy_logits_only', _DummyHead)
register_mil('dummy_no_adapter')(_NoAdapterMIL)


def test5_mil_decoupling():
    # (a) unknown MIL name → explicit error, never a silent ABMIL fallback
    try:
        build_mil_head('not_a_mil', input_dim=64, num_classes=2)
        raise AssertionError("unknown MIL name must raise")
    except ValueError as e:
        assert 'not_a_mil' in str(e) and 'abmil' in str(e)

    # (b) registered MIL without an adapter → explicit error naming the fix
    try:
        build_mil_head('dummy_no_adapter', input_dim=64, num_classes=2)
        raise AssertionError("adapter-less MIL must raise")
    except ValueError as e:
        assert 'head adapter' in str(e), str(e)

    # (c) the unified model trains with a logits-only head
    model = _small_model(
        ['HE', 'PR'],
        mil_cfg={'name': 'dummy_logits_only',
                 'kwargs': {'input_dim': 64, 'num_classes': 2}})
    assert isinstance(model.mil, _DummyHead)
    assert not hasattr(model.mil, 'attention'), "must not expose ABMIL internals"
    assert not hasattr(model.mil, 'classifier')

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    feats = _small_inputs(['HE', 'PR'], seed=101)
    y = torch.tensor([1])
    losses = []
    for _ in range(5):
        opt.zero_grad()
        logits = model(feats)[0]
        assert logits.shape == (1, 2)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(math.isfinite(v) for v in losses), f"non-finite loss {losses}"
    assert losses[-1] < losses[0], f"training step did not reduce loss {losses}"

    # (d) the fusion path does not touch the MIL at all
    model.eval()
    with torch.no_grad():
        fused, H, branches = model.fuse(feats)
    assert fused.shape == H.shape and set(branches) == {'PR'}
    calls = []
    model.mil.register_forward_hook(lambda *a: calls.append(1))
    with torch.no_grad():
        model.fuse(feats)
    assert not calls, "fuse() must not invoke the MIL head"

    # (e) a richer head still satisfies the same contract
    head = build_mil_head('abmil', input_dim=64, num_classes=2,
                          hidden_dim=16, dropout_rate=0.0).eval()
    with torch.no_grad():
        r = head(tokens=torch.randn(2, 7, 64), mask=None)
    assert set(r) >= {'logits'} and r['logits'].shape == (2, 2)

    return (f"dummy logits-only head: loss {losses[0]:.4f} → {losses[-1]:.4f} "
            f"over 5 steps; unknown/unadapted MIL names raise; "
            f"heads with adapters = {available_heads()}")


# ---------------------------------------------------------------------------
# Test 6 — config, masks, errors, checkpoint round-trip
# ---------------------------------------------------------------------------
def test6_config_and_restore():
    mods = ['HE', 'ER', 'PR']
    model = _small_model(mods)
    assert model.get_modality_names() == mods
    assert model.aux_stains == ['ER', 'PR']

    feats = _small_inputs(mods, seed=111)
    model.eval()

    # (a) dict input and list input in canonical order → same result
    with torch.no_grad():
        a = model(feats)[0]
        b = model([feats[m] for m in mods])[0]
        c = model([feats[m] for m in mods], modality_names=mods)[0]
    assert torch.equal(a, b) and torch.equal(a, c), "list/dict input mismatch"

    # (b) explicit errors — never a silent skip or dataset shrink
    bad = [
        ({m: feats[m] for m in ['HE', 'PR']}, 'missing stain'),
        ({**feats, 'XY': feats['HE']}, 'unexpected stain'),
        (feats['HE'], 'bare tensor on a multi-stain model'),
    ]
    for inp, label in bad:
        try:
            model(inp)
            raise AssertionError(f"{label}: expected an error")
        except (ValueError, TypeError):
            pass
    try:
        model([feats['HE'], feats['PR'], feats['ER']],
              modality_names=['HE', 'PR', 'ER'])
        raise AssertionError("wrong modality_names order must raise")
    except ValueError:
        pass

    for bad_mods in (['PR', 'ER'], ['HE', 'HE'], []):
        try:
            _small_model(bad_mods)
            raise AssertionError(f"modality_list {bad_mods} must raise")
        except ValueError:
            pass

    # (c) encoder config: HE/PR historical by default, others inherit HE
    hist = build_he_aux_unified(['HE', 'PR'], **SMALL)   # no explicit encoder_cfg
    assert hist.encoder_cfg_resolved['PR'] == HISTORICAL_V3_ENCODER_CFG['PR'], \
        "PR must keep its historical v3 encoder config by default"
    assert hist.encoder_cfg_source['PR'] == 'historical_v3'

    plain = build_he_aux_unified(['HE', 'ER', 'PR'], **SMALL)
    assert plain.encoder_cfg_source['ER'] == 'he_default', \
        "unconfigured stains must be recorded as inheriting the HE config"
    assert plain.encoder_cfg_resolved['ER'] == \
        plain.encoder_cfg_resolved['HE'], "ER must inherit HE's encoder config"
    assert plain.encoder_cfg_resolved['ER'] != HISTORICAL_V3_ENCODER_CFG['PR'], \
        "ER must not silently pick up PR's geometry"
    assert plain.encoder_cfg_resolved['PR'] == HISTORICAL_V3_ENCODER_CFG['PR']
    assert plain.stage2_cfg == HISTORICAL_V3_STAGE2_CFG, \
        "default stage2 cfg must equal the historical v3 config"
    assert (plain.stage2_cfg['temperature'], plain.stage2_cfg['residual_scale'],
            plain.stage2_cfg['prototype_momentum']) == (0.2, 0.1, 0.99)

    # (d) token mask: all-valid == no mask; invalid tokens are ignored
    with torch.no_grad():
        t = _rand(4, 13, 64, seed=121)
        head = build_mil_head('abmil', input_dim=64, num_classes=2,
                              hidden_dim=16, dropout_rate=0.0).eval()
        no_mask = head(tokens=t, mask=None)['logits']
        all_true = head(tokens=t,
                        mask=torch.ones(4, 13, dtype=torch.bool))['logits']
        assert torch.allclose(no_mask, all_true, atol=1e-5), \
            "an all-True mask must be a numerical no-op"

        m = torch.ones(4, 13, dtype=torch.bool)
        m[:, 7:] = False
        masked = head(tokens=t, mask=m)['logits']
        t2 = t.clone()
        t2[:, 7:] = 1e4                      # garbage in the invalid slots
        masked2 = head(tokens=t2, mask=m)['logits']
        assert torch.allclose(masked, masked2, atol=1e-5), \
            "invalid tokens must not influence the masked head"
        assert not torch.allclose(masked, no_mask, atol=1e-5), \
            "the mask must actually change the pooling"

        mm = model([feats['HE'], feats['ER'], feats['PR']],
                   valid_masks={'HE': torch.ones(
                       1, feats['HE'].shape[1], dtype=torch.bool)})
        assert mm[0].shape == (1, 2)

    for bad_masks in ({'HE': torch.ones(1, 3, dtype=torch.bool)},
                      {'XX': torch.ones(1, 36, dtype=torch.bool)}):
        try:
            model(feats, valid_masks=bad_masks)
            raise AssertionError(f"valid_masks {list(bad_masks)} must raise")
        except ValueError:
            pass

    # (e) checkpoint round-trip: config + weights → identical outputs
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = copy.deepcopy(model.get_config())
    assert cfg['model_family'] == 'he_aux_unified'
    assert cfg['modality_list'] == mods and cfg['aux_stains'] == ['ER', 'PR']
    assert cfg['mil_cfg'] == {'name': 'abmil',
                              'kwargs': {'hidden_dim': 256, 'dropout_rate': 0.25}}
    assert cfg['encoder_cfg_resolved']['PR'] == SMALL_ENC
    assert cfg['fusion']['formula'].startswith('H + (residual_scale / M)')
    assert cfg['fusion']['branch_order'] == ['ER', 'PR']

    restored = build_he_aux_unified(
        cfg['modality_list'], input_dim=cfg['input_dim'], mlp_dim=cfg['mlp_dim'],
        num_classes=cfg['num_classes'], dropout=cfg['dropout'], act=cfg['act'],
        encoder_cfg=cfg['encoder_cfg'], stage2_cfg=cfg['stage2_cfg'],
        mil_cfg=cfg['mil_cfg'], init_seed=cfg['init_seed'], **SMALL_STRUCT)
    restored.load_state_dict(sd, strict=True)
    restored.eval()
    with torch.no_grad():
        r1 = restored([feats[m] for m in mods])[0]
    assert torch.equal(a, r1), "checkpoint round-trip changed the logits"

    # (f) the v3 key mapping reports its problems instead of hiding them
    target = model.state_dict()
    assert 'rrt.HE.norm.weight' in target
    mapped, unmapped = map_v3_state_dict_to_unified(
        {'rrt_he.norm.weight': target['rrt.HE.norm.weight'],
         'bogus.key': torch.zeros(1)}, 'HE', 'PR')
    assert unmapped == ['bogus.key'], f"unmapped keys not reported: {unmapped}"
    assert 'rrt.HE.norm.weight' in mapped

    if V3_CKPT.exists():
        import tempfile
        ckpt = torch.load(V3_CKPT, map_location='cpu', weights_only=False)
        truncated = dict(ckpt['model_state_dict'])
        truncated.pop('rrt_he.norm.weight')          # one parameter short
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save({'model_state_dict': truncated,
                        'modalities': ckpt.get('modalities', ['HE', 'PR'])},
                       f.name)
            tmp = f.name
        try:
            uni_bad = build_he_aux_unified_from_config(
                ckpt['config']['model'], ckpt['config']['data'])
            try:
                load_v3_checkpoint_into_unified(uni_bad, tmp, verbose=False)
                raise AssertionError("incomplete checkpoint must be rejected")
            except ValueError as e:
                assert 'not covered' in str(e), str(e)
        finally:
            os.unlink(tmp)

    # (g) config-driven build requires HE first
    try:
        build_he_aux_unified_from_config(
            {'stage2_type': 'he_aux_unified'}, {'modalities': ['PR', 'HE']})
        raise AssertionError("HE must be first in data.modalities")
    except ValueError as e:
        assert 'first' in str(e)

    return ("dict/list input agree; missing/extra/mismatched stains and mask "
            "shapes raise; PR keeps the historical encoder cfg, ER inherits HE; "
            "checkpoint round-trip exact; truncated v3 checkpoint rejected")


# ---------------------------------------------------------------------------
def test7_model_config_restore():
    """Test 7 — restore from the checkpoint's own `model_config` + init hash.

    * `get_config()` carries every construction argument and a schema version
    * a model can be rebuilt from `model_config` ALONE (no data block, no
      training config) and loaded with `strict=True`
    * the same config + init_seed reproduces the identical initial weights, and
      adding auxiliary branches does not shift HE/MIL initialisation
    * an unknown / missing `config_schema_version` is refused, not guessed at
    """
    from models.he_aux_unified import (
        build_he_aux_unified_from_model_config, initialization_hash,
        MODEL_CONFIG_SCHEMA_VERSION, _CONSTRUCTION_KEYS,
    )

    m = build_he_aux_unified(
        modality_list=['HE', 'ER', 'PR'], init_seed=11, **SMALL)
    cfg = m.get_config()

    assert cfg['config_schema_version'] == MODEL_CONFIG_SCHEMA_VERSION
    missing_keys = [k for k in _CONSTRUCTION_KEYS if k not in cfg]
    assert not missing_keys, f"get_config() omits construction keys {missing_keys}"

    # rebuild from model_config alone, strict load
    m2 = build_he_aux_unified_from_model_config(cfg)
    m2.load_state_dict(m.state_dict(), strict=True)
    assert initialization_hash(m) == initialization_hash(m2), \
        "same model_config + init_seed must reproduce identical init weights"

    # a second independent build is bit-identical too (hash is meaningful)
    m3 = build_he_aux_unified_from_model_config(cfg)
    assert initialization_hash(m3) == initialization_hash(m), "init not reproducible"

    # init must not depend on the number of auxiliary branches
    base = dict(SMALL, init_seed=11)
    a = build_he_aux_unified(modality_list=['HE'], **base)
    b = build_he_aux_unified(
        modality_list=['HE', 'ER', 'PR', 'HER2', 'Ki67'], **base)
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    shared = [k for k in pa if k in pb]
    assert shared, "no shared parameter names between HE-only and HE+4aux"
    bad = [k for k in shared if not torch.equal(pa[k], pb[k])]
    assert not bad, f"adding aux branches changed shared init weights: {bad[:3]}"

    # schema version is enforced
    for bad_cfg, why in (
            ({k: v for k, v in cfg.items() if k != 'config_schema_version'},
             'missing'),
            (dict(cfg, config_schema_version=99), 'unknown')):
        try:
            build_he_aux_unified_from_model_config(bad_cfg)
            raise AssertionError(f"{why} config_schema_version must be refused")
        except ValueError:
            pass
    try:
        build_he_aux_unified_from_model_config(dict(cfg, model_family='other'))
        raise AssertionError("foreign model_family must be refused")
    except ValueError:
        pass

    # round-trip through a real checkpoint on disk
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'ckpt.pt')
        torch.save({'model_state_dict': m.state_dict(),
                    'model_config': cfg,
                    'resolved_seeds': {'run_seed': 11},
                    'init_hash': initialization_hash(m)}, path)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        r = build_he_aux_unified_from_model_config(ck['model_config'])
        r.load_state_dict(ck['model_state_dict'], strict=True)
        assert initialization_hash(r) == ck['init_hash']
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return (f"config_schema_version={MODEL_CONFIG_SCHEMA_VERSION}; rebuilt from "
            f"model_config alone + strict=True; init hash reproducible; "
            f"{len(shared)} shared tensors identical across aux counts; "
            f"missing/unknown version refused")


# ---------------------------------------------------------------------------
def test8_feature_cache_and_masks():
    """Test 8 — `return_features` reuses the forward's own Stage-1 tensors.

    In train mode dropout would draw a *fresh* mask on a second projection call,
    so the reported features would not be the ones the forward used.  The check
    is exact: re-running the Stage-1 RRT on the returned `embedded_features`
    must reproduce the returned `encoded_features`.
    """
    m = build_he_aux_unified(modality_list=['HE', 'PR'], init_seed=5, **SMALL)
    m.train()
    x = _small_inputs(['HE', 'PR'])
    out = m(x, return_features=True)

    emb = out['embedded_features']
    enc = out['encoded_features']
    assert len(emb) == 2 and len(enc) == 2
    for i, stain in enumerate(m.stain_order):
        recomputed = m._as_3d(m.rrt[stain](emb[i]))
        assert torch.equal(recomputed, enc[i]), (
            f"{stain}: encoded_features is not the RRT of the returned "
            f"embedded_features — Stage-1 was recomputed (fresh dropout)")

    # and the returned HE embedding must be the one the fusion actually used
    _, H, _ = m.fuse(x, encoded={'HE': enc[0], 'PR': enc[1]})
    assert torch.equal(H, enc[0])

    # ── mask policy: all-True / None are fine, any False raises ──
    n_he = x['HE'].shape[1]
    m(x, valid_masks={'HE': torch.ones(1, n_he, dtype=torch.bool)})
    m(x, valid_masks=None)

    partial = torch.ones(1, n_he, dtype=torch.bool)
    partial[0, 3] = False
    try:
        m(x, valid_masks={'HE': partial})
        raise AssertionError("a mask containing False must be refused")
    except ValueError as e:
        assert 'Stage-1' in str(e), str(e)
    # the same refusal applies to an auxiliary stain's mask
    n_pr = x['PR'].shape[1]
    p_pr = torch.ones(1, n_pr, dtype=torch.bool)
    p_pr[0, 0] = False
    try:
        m(x, valid_masks={'PR': p_pr})
        raise AssertionError("partial aux mask must be refused")
    except ValueError as e:
        assert 'Stage-1' in str(e), str(e)

    return ("encoded_features == rrt(embedded_features) exactly (no dropout "
            "recompute); all-True/None masks accepted; any-False mask refused "
            "for both HE and aux stains")


# ---------------------------------------------------------------------------
def test9_seed_and_output_resolution():
    """Test 9 — run/model_init/sampling seed resolution and output dirs."""
    import train as T

    # explicit values win; all three are recorded with their provenance
    cfg = {
        'environment': {'seed': 7},
        'model': {'stage2_type': 'he_aux_unified', 'init_seed': 21},
        'data': {'modalities': ['HE', 'ER', 'PR'], 'sample_seed': 33},
        'output': {'save_dir': '', 'log_dir': '', 'img_dir': ''},
    }
    run, init, samp = T.resolve_seeds(cfg)
    assert (run, init, samp) == (7, 21, 33), (run, init, samp)
    rs = cfg['resolved_seeds']
    assert rs['model_init_seed_source'] == 'explicit'
    assert rs['sampling_seed_source'] == 'explicit'
    # written back so every downstream reader sees the resolved value
    assert cfg['model']['init_seed'] == 21 and cfg['data']['sample_seed'] == 33

    # absent values inherit run_seed
    cfg2 = {
        'environment': {'seed': 7},
        'model': {'stage2_type': 'he_aux_unified'},
        'data': {'modalities': ['HE']},
        'output': {},
    }
    run2, init2, samp2 = T.resolve_seeds(cfg2)
    assert (run2, init2, samp2) == (7, 7, 7), (run2, init2, samp2)
    assert cfg2['resolved_seeds']['model_init_seed_source'] == 'run_seed'
    assert cfg2['resolved_seeds']['sampling_seed_source'] == 'run_seed'

    # empty output dirs resolve under the repo root, keyed by family/combination/seed
    out = T.resolve_output_dirs(cfg)
    assert T.modality_slug(['HE', 'ER', 'PR']) == 'he_er_pr'
    expected = T.REPO_ROOT / 'results' / 'he_aux_unified' / 'he_er_pr' / 'seed7'
    assert Path(out['save_dir']) == expected, out['save_dir']
    assert Path(out['log_dir']) == expected / 'logs'
    assert Path(out['img_dir']) == expected / 'img'

    # an explicitly configured directory is left exactly as written
    cfg3 = {'environment': {'seed': 1}, 'model': {}, 'data': {},
            'output': {'save_dir': '/tmp/explicit', 'log_dir': '/tmp/explicit/l',
                       'img_dir': '/tmp/explicit/i'}}
    out3 = T.resolve_output_dirs(cfg3)
    assert out3['save_dir'] == '/tmp/explicit', out3['save_dir']

    # optimizer coverage check: missing and duplicated params both raise
    model = build_he_aux_unified(modality_list=['HE', 'PR'], init_seed=3, **SMALL)
    params = [p for p in model.parameters() if p.requires_grad]

    class _Rec:
        def __init__(self, logger): self.logger = logger
    rec = _Rec(T.logging.getLogger('test9'))
    T.Trainer.verify_optimizer_coverage(rec, model,
                                        torch.optim.Adam(params, lr=1e-4))
    try:                       # one parameter dropped → must raise
        T.Trainer.verify_optimizer_coverage(
            rec, model, torch.optim.Adam(params[1:], lr=1e-4))
        raise AssertionError("an uncovered trainable param must raise")
    except RuntimeError as e:
        assert 'never update' in str(e), str(e)
    try:                       # one parameter listed twice → must raise
        # `Adam([...])` itself refuses a param in two groups, so build a valid
        # optimizer and duplicate the entry afterwards (what a hand-edited
        # `param_groups`, or an appended group, would produce).
        dup = torch.optim.Adam(params, lr=1e-4)
        dup.param_groups[0]['params'] = (list(dup.param_groups[0]['params'])
                                         + [params[0]])
        T.Trainer.verify_optimizer_coverage(rec, model, dup)
        raise AssertionError("a duplicated param must raise")
    except RuntimeError as e:
        assert 'more than one' in str(e), str(e)

    return ("explicit seeds win / absent inherit run_seed; empty output dirs → "
            "results/he_aux_unified/he_er_pr/seed7/{,logs,img}; explicit dirs "
            "untouched; optimizer coverage catches missing + duplicated params")


# ---------------------------------------------------------------------------
def main(argv=None):
    global V3_CKPT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v3-ckpt', type=Path, default=DEFAULT_V3_CKPT,
                    help='legacy v3 checkpoint used by Test 1 (dual regression)')
    ap.add_argument('--device', default='cpu',
                    help="device for the tests (default 'cpu')")
    args = ap.parse_args(argv)

    V3_CKPT = Path(args.v3_ckpt)
    torch.manual_seed(0)
    print(f"repo root : {PROJECT}")
    print(f"v3 ckpt   : {V3_CKPT} ({'present' if V3_CKPT.exists() else 'MISSING'})")
    print(f"device    : {args.device}\n")

    tests = [
        ("Test 1 — dual regression vs legacy v3", test1_dual_regression),
        ("Test 2 — HE-only regression + init independence",
         test2_he_only_regression),
        ("Test 3 — multi-modal entry (1/2/4 aux)", test3_multi_modal_entry),
        ("Test 4 — EMA prototype independence", test4_ema_independence),
        ("Test 5 — MIL pluggability / decoupling", test5_mil_decoupling),
        ("Test 6 — config, masks, errors, restore", test6_config_and_restore),
        ("Test 7 — model_config-only restore + init hash", test7_model_config_restore),
        ("Test 8 — cached return_features + mask policy", test8_feature_cache_and_masks),
        ("Test 9 — seed/output resolution + optimizer coverage",
         test9_seed_and_output_resolution),
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
            import traceback
            print(f"[ERROR] {name} — {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
