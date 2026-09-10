"""
HE-anchored unified model — HE primary + **any** subset of auxiliary stains,
with a pluggable MIL prediction head.

This is the "official" generalisation of the `he_residual_cross_v3` two-stage
model.  The pre-existing v3 path (`MM_RRT_ABMIL` with
`stage2_type='he_residual_cross_v3'`) is left untouched and keeps working; this
module is a *new, explicitly named* model path that reuses the same v3 building
blocks.

    Stage 1:  per-stain projection → per-stain RRT encoder
              H  = he_encoder(proj_HE(x_HE))                  [B, N_HE, D]  (computed ONCE)
              Z_m = rrt_m(proj_m(x_m))                        [B, N_m,  D]  for each aux stain m

    Stage 2:  one independent v3 cross branch per aux stain, each reading the
              SAME un-updated H (they are parallel, never serial):

              branch_m(H, Z_m) = H + residual_scale · Δ_m            (v3 forward verbatim)

              H_fused = mean_m branch_m(H, Z_m)
                      = H + (residual_scale / M) · Σ_m Δ_m           (M = #aux stains)

              with no aux stain:  H_fused = H

    Stage 3:  MIL head (pluggable, `models/mil_heads.py`)

              logits = build_mil_head(...)(tokens=H_fused, mask=valid_HE)['logits']

Three properties this file is built around:

1. **Every branch output already contains the HE residual.**  We therefore sum the
   branch outputs and divide by M — we must NOT add H again and must NOT apply
   residual_scale a second time.  With a single auxiliary stain the mean of one
   branch is that branch, i.e. exactly the historical v3 forward.

2. **Name-keyed modules.**  Projections, RRT encoders and cross branches live in
   `nn.ModuleDict`s keyed by stain name — adding a stain is a config change, not
   a `num_modalities` bump, and there is no "only the first two inputs are used"
   restriction anywhere in the forward.

3. **Order-independent, reproducible init.**  Each submodule is constructed inside
   its own seeded RNG scope (`_seeded_scope`), keyed by `<role>/<stain>`.  The
   caller's RNG stream is restored afterwards, so adding an auxiliary branch can
   never change the initial weights of the HE branch or the MIL head.

Checkpoint compatibility: `load_v3_checkpoint_into_unified()` performs an
explicit, checked key mapping (never a blanket `strict=False`), so existing v3
checkpoints load into this model with a full completeness/shape report.

Historical v3 hyper-parameters are the defaults here (`HISTORICAL_V3_STAGE2_CFG`,
`HISTORICAL_V3_ENCODER_CFG`): cosine cross-attention τ=0.2, residual_scale=0.1,
prototype_momentum=0.99, bias-free QKV.  No new gate, auxiliary loss,
distillation, gradient division or extra normalisation is introduced.
"""

from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.he_residual_cross_crmsa_v3 import HEResidualCrossCRMSAv3
from models.mil_heads import build_mil_head, mil_head_config
from models.mm_rrt_encoder import RRTEncoder, initialize_weights


HE_STAIN = 'HE'
MODEL_FAMILY = 'he_aux_unified'

#: Bumped whenever the *meaning* of a `get_config()` entry changes.  Written
#: into every checkpoint's `model_config`; restoring refuses an unknown version
#: rather than guessing at an older layout.
MODEL_CONFIG_SCHEMA_VERSION = 1

#: The subset of `get_config()` that `HEAuxUnifiedModel.__init__` accepts — the
#: only keys `build_he_aux_unified_from_model_config` forwards.  Everything else
#: in `get_config()` is derived metadata and is deliberately not replayed.
_CONSTRUCTION_KEYS = (
    'modality_list', 'he_stain', 'input_dim', 'mlp_dim', 'num_classes',
    'dropout', 'act', 'encoder_cfg', 'stage2_cfg', 'mil_cfg', 'init_seed',
    'region_num', 'n_layers', 'n_heads', 'drop_path', 'trans_dropout',
    'epeg', 'epeg_k', 'crmsa_k', 'cr_msa', 'all_shortcut', 'crmsa_mlp',
    'crmsa_heads',
)

#: Construction keys without which a model cannot be rebuilt (the rest fall back
#: to the class defaults).
_REQUIRED_CONSTRUCTION_KEYS = ('modality_list', 'input_dim', 'mlp_dim',
                               'num_classes')

#: Stage-1 encoder config used by the historical v3 runs
#: (`scripts/run_stage2_v3.py::STAGE1_ENCODER_CFG`).  HE and PR keep these; any
#: other stain without an explicit `encoder_cfg` entry uses the **HE** entry
#: (recorded per-stain in `encoder_cfg_resolved` — no parameter search).
HISTORICAL_V3_ENCODER_CFG = {
    'HE': {'region_num': 4, 'epeg_k': 9, 'crmsa_k': 3, 'n_heads': 4,
           'drop_path': 0.0},
    'PR': {'region_num': 8, 'epeg_k': 15, 'crmsa_k': 5, 'n_heads': 8,
           'drop_path': 0.11554210024949738},
}

#: Stage-2 cross-branch config used by the historical v3 runs
#: (`scripts/run_stage2_v3.py::STAGE2_CFG`).  Shared by every auxiliary branch.
HISTORICAL_V3_STAGE2_CFG = {
    'region_num': 4, 'crmsa_heads': 8, 'crmsa_k': 3, 'drop_out': 0.1,
    'drop_path': 0.0, 'epeg': False, 'epeg_k': 15, 'crmsa_mlp': False,
    'ffn': False, 'qkv_bias': False, 'temperature': 0.2,
    'residual_scale': 0.1, 'disable_cross': False, 'prototype_momentum': 0.99,
}

#: MIL config used by the historical v3 runs (`abmil_hidden_dim=256`,
#: `dropout=0.25`).
HISTORICAL_V3_MIL_CFG = {
    'name': 'abmil',
    'kwargs': {'hidden_dim': 256, 'dropout_rate': 0.25},
}

def _derive_seed(base_seed: int, key: str) -> int:
    """Deterministic sub-seed for a module key (stable across processes)."""
    h = hashlib.md5(f'{base_seed}|{key}'.encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1)


@contextmanager
def _seeded_scope(seed: int):
    """Run a construction under `seed`, restoring the caller's RNG state after.

    Restoring (rather than merely re-seeding) is what makes module init
    order-independent: an extra auxiliary branch consumes RNG only inside its
    own scope and leaves no trace for the modules built after it.

    `torch.manual_seed` seeds *every* device, so the CUDA generator has to be
    saved/restored alongside the CPU one — otherwise constructing a submodule
    silently shifts the CUDA stream and a later `.to(device)` / stochastic op
    on GPU stops being reproducible.
    """
    cpu_state = torch.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() else None)
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def initialization_hash(model: nn.Module) -> str:
    """Stable sha256 over a model's *current* parameters and buffers.

    Called right after construction to record the initial weights, so two runs
    claiming the same seed can be checked for identical initialisation.
    """
    h = hashlib.sha256()
    sd = model.state_dict()
    for name in sorted(sd.keys()):
        h.update(name.encode('utf-8'))
        h.update(sd[name].detach().cpu().numpy().tobytes())
    return h.hexdigest()


class HEAuxUnifiedModel(nn.Module):
    """HE + any subset of auxiliary stains → fused HE tokens → pluggable MIL head.

    Args:
        modality_list: stain names, e.g. ``['HE']``, ``['HE','PR']`` or
            ``['HE','ER','PR','HER2','Ki67']``.  HE must be present; all other
            names are auxiliary stains.  Traversal order is canonicalised to
            ``[HE] + aux in the given order`` and never changes afterwards.
        input_dim:     patch-feature dimension (768 for CTransPath C16 features)
        mlp_dim:       internal feature dimension D (512)
        num_classes:   classifier output width
        dropout:       dropout on the projected per-stain features (0.25 for v3)
        act:           activation after the projection ('relu' for v3)
        encoder_cfg:   per-stain Stage-1 RRT config; missing stains fall back to
                       HE's resolved config (HE/PR default to the historical v3
                       values, see `HISTORICAL_V3_ENCODER_CFG`)
        stage2_cfg:    v3 cross-branch config, shared by all aux branches
        mil_cfg:       ``{'name': ..., 'kwargs': {...}}`` for `build_mil_head`
        init_seed:     base seed for the deterministic per-module init
        he_stain:      name of the primary stain (default 'HE')
    """

    def __init__(self, modality_list=('HE', 'PR'), input_dim=768, mlp_dim=512,
                 num_classes=2, dropout=0.25, act='relu',
                 encoder_cfg=None, stage2_cfg=None, mil_cfg=None,
                 init_seed=1234, he_stain=HE_STAIN,
                 # model-level Stage-1 / Stage-2 structure (v3 defaults)
                 region_num=4, n_layers=2, n_heads=4, drop_path=0.0,
                 trans_dropout=0.1, epeg=True, epeg_k=9, crmsa_k=3,
                 cr_msa=True, all_shortcut=True, crmsa_mlp=False,
                 crmsa_heads=8, **kwargs):
        super().__init__()

        modality_list = [str(m) for m in (modality_list or [])]
        if not modality_list:
            raise ValueError("modality_list must name at least the primary stain")
        if len(set(modality_list)) != len(modality_list):
            raise ValueError(f"modality_list has duplicates: {modality_list}")
        if he_stain not in modality_list:
            raise ValueError(
                f"modality_list {modality_list} must contain the primary stain "
                f"{he_stain!r}; HE is the primary modality of this model.")

        # Canonical traversal order: HE first, auxiliary stains in config order.
        self.he_stain = he_stain
        self.aux_stains = [m for m in modality_list if m != he_stain]
        self.stain_order = [he_stain] + self.aux_stains
        self.modality_list = list(self.stain_order)
        self.num_modalities = len(self.stain_order)

        self.input_dim = int(input_dim)
        self.mlp_dim = int(mlp_dim)
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)
        self.act = act
        self.init_seed = int(init_seed)

        # model-level structure defaults
        self.region_num = region_num
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.drop_path = drop_path
        self.trans_dropout = trans_dropout
        self.epeg = epeg
        self.epeg_k = epeg_k
        self.crmsa_k = crmsa_k
        self.cr_msa = cr_msa
        self.all_shortcut = all_shortcut
        self.crmsa_mlp = crmsa_mlp
        self.crmsa_heads = crmsa_heads

        # ── configs ────────────────────────────────────────────────────────
        self.encoder_cfg = copy.deepcopy(encoder_cfg) if encoder_cfg else {}
        self.encoder_cfg_resolved = self._resolve_encoder_cfg()
        self.encoder_cfg_source = {}
        for m in self.stain_order:
            if m in self.encoder_cfg:
                self.encoder_cfg_source[m] = 'explicit'
            elif m in HISTORICAL_V3_ENCODER_CFG:
                self.encoder_cfg_source[m] = 'historical_v3'
            elif m == he_stain:
                self.encoder_cfg_source[m] = 'model_default'
            else:
                self.encoder_cfg_source[m] = 'he_default'
        self.stage2_cfg = dict(HISTORICAL_V3_STAGE2_CFG)
        if stage2_cfg:
            self.stage2_cfg.update(stage2_cfg)
        self.mil_cfg = copy.deepcopy(mil_cfg) if mil_cfg else copy.deepcopy(
            HISTORICAL_V3_MIL_CFG)

        # ── per-stain projection (independent weights) ─────────────────────
        self.patch_to_emb = nn.ModuleDict()
        for stain in self.stain_order:
            self.patch_to_emb[stain] = self._build(
                f'patch_to_emb/{stain}', self._make_projection)

        self.dp = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

        # ── per-stain Stage-1 RRT encoder (independent weights) ────────────
        self.rrt = nn.ModuleDict()
        for stain in self.stain_order:
            self.rrt[stain] = self._build(
                f'rrt/{stain}', lambda s=stain: self._make_rrt(s))

        # ── one independent v3 cross branch per auxiliary stain ────────────
        # (independent parameters AND independent EMA prototypes)
        self.cross_branches = nn.ModuleDict()
        for stain in self.aux_stains:
            self.cross_branches[stain] = self._build(
                f'cross_branches/{stain}', self._make_cross_branch)

        # ── pluggable MIL head ─────────────────────────────────────────────
        self.mil_type = self.mil_cfg['name']
        self.mil = self._build(
            'mil', lambda: build_mil_head(**mil_head_config(
                self.mil_cfg, input_dim=self.mlp_dim,
                num_classes=self.num_classes)))

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _resolve_encoder_cfg(self) -> Dict[str, dict]:
        """Per-stain Stage-1 encoder config, explicitly resolved.

        Precedence (later wins):
            1. model-level defaults
            2. the HE stain's resolved config — what an *unconfigured* stain
               inherits ("explicitly recorded HE encoder default")
            3. the historical v3 entry for that stain (so PR keeps its own
               historical geometry instead of inheriting HE's)
            4. an explicit `encoder_cfg[stain]` entry

        Nothing is searched; every stain's resolved values are recorded in
        `encoder_cfg_resolved` (and stored in the checkpoint).
        """
        defaults = {
            'region_num': self.region_num,
            'n_heads': self.n_heads,
            'epeg_k': self.epeg_k,
            'crmsa_k': self.crmsa_k,
            'drop_path': self.drop_path,
        }
        base = dict(defaults)
        base.update(HISTORICAL_V3_ENCODER_CFG.get(self.he_stain, {}))
        base.update(copy.deepcopy(self.encoder_cfg.get(self.he_stain, {})))

        resolved = {self.he_stain: dict(base)}
        for stain in self.aux_stains:
            cfg = dict(base)                                    # 1+2: inherit HE
            cfg.update(HISTORICAL_V3_ENCODER_CFG.get(stain, {}))  # 3: historical
            cfg.update(copy.deepcopy(self.encoder_cfg.get(stain, {})))  # 4: explicit
            resolved[stain] = cfg
        return resolved

    def _build(self, key: str, ctor):
        """Construct one submodule under its own deterministic RNG scope."""
        with _seeded_scope(_derive_seed(self.init_seed, key)):
            return ctor()

    def _make_projection(self):
        layers = [nn.Linear(self.input_dim, self.mlp_dim)]
        if str(self.act).lower() == 'relu':
            layers.append(nn.ReLU())
        elif str(self.act).lower() == 'gelu':
            layers.append(nn.GELU())
        proj = nn.Sequential(*layers)
        proj.apply(initialize_weights)
        return proj

    def _make_rrt(self, stain: str):
        cfg = self.encoder_cfg_resolved[stain]
        return RRTEncoder(
            mlp_dim=self.mlp_dim,
            region_num=cfg['region_num'], n_layers=self.n_layers,
            n_heads=cfg['n_heads'], drop_path=cfg['drop_path'],
            drop_out=self.trans_dropout, epeg=self.epeg, epeg_k=cfg['epeg_k'],
            crmsa_k=cfg['crmsa_k'], cr_msa=self.cr_msa,
            all_shortcut=self.all_shortcut, crmsa_mlp=self.crmsa_mlp,
            crmsa_heads=self.crmsa_heads,
            need_init=True,
        )

    def _make_cross_branch(self):
        """One independent v3 cross branch (independent params + EMA buffer)."""
        c = self.stage2_cfg
        return HEResidualCrossCRMSAv3(
            dim=self.mlp_dim,
            num_heads=c.get('crmsa_heads', 8),
            region_num=c.get('region_num', 4),
            crmsa_k=c.get('crmsa_k', 3),
            drop_out=c.get('drop_out', 0.1),
            drop_path=c.get('drop_path', 0.0),
            epeg=c.get('epeg', False),
            epeg_k=c.get('epeg_k', 15),
            crmsa_mlp=c.get('crmsa_mlp', False),
            ffn=c.get('ffn', False),
            qkv_bias=c.get('qkv_bias', False),
            residual_scale=c.get('residual_scale', 0.1),
            disable_cross=c.get('disable_cross', False),
            tau=c.get('temperature', 0.2),
            prototype_momentum=c.get('prototype_momentum', 0.99),
        )

    # ------------------------------------------------------------------
    # input handling
    # ------------------------------------------------------------------
    def _as_3d(self, z: torch.Tensor) -> torch.Tensor:
        return z.unsqueeze(0) if z.dim() == 2 else z

    def _normalize_inputs(self, x, modality_names=None) -> Dict[str, torch.Tensor]:
        """Accept a dict {stain: [B,N,D]}, or a list/tuple in `stain_order`.

        A list is only accepted in the canonical order (HE first, then the
        auxiliary stains as configured).  Missing / unexpected stains raise —
        the model never silently drops a modality.
        """
        expected = self.stain_order
        if isinstance(x, dict):
            given = set(x)
            missing = [m for m in expected if m not in given]
            extra = sorted(given - set(expected))
            if missing or extra:
                raise ValueError(
                    f"input stain mismatch: missing {missing}, unexpected {extra}; "
                    f"model expects {expected}")
            feats = {m: self._as_3d(x[m]) for m in expected}
        elif isinstance(x, (list, tuple)):
            if modality_names is not None:
                names = [str(n) for n in modality_names]
                if names != expected:
                    raise ValueError(
                        f"modality_names {names} do not match the model's stain "
                        f"order {expected}")
            if len(x) != len(expected):
                raise ValueError(
                    f"expected {len(expected)} stain tensors ({expected}), "
                    f"got {len(x)}")
            feats = {m: self._as_3d(t) for m, t in zip(expected, x)}
        elif torch.is_tensor(x):
            if len(expected) != 1:
                raise ValueError(
                    f"a bare tensor input is only valid for a single-stain model, "
                    f"but this model expects {expected}")
            feats = {expected[0]: self._as_3d(x)}
        else:
            raise TypeError(
                f"unsupported input type {type(x).__name__}; expected dict, "
                f"list/tuple or tensor")

        for m, t in feats.items():
            if t.dim() != 3:
                raise ValueError(
                    f"stain {m!r}: expected [B, N, D], got {tuple(t.shape)}")
            if t.shape[-1] != self.input_dim:
                raise ValueError(
                    f"stain {m!r}: feature dim {t.shape[-1]} != input_dim "
                    f"{self.input_dim}")
        return feats

    def _normalize_masks(self, valid_masks, feats):
        """dict {stain: [B,N] bool} or None → dict with only present entries.

        Only all-True masks (or `None`) are accepted.  Stage-1 RRT has no
        masking, so a mask with any `False` would be honoured by the Stage-2
        cross branch while the encoder still attended over the padding — i.e.
        it would be *silently half-applied*.  Rather than claim end-to-end
        mask support we refuse it loudly.
        """
        if valid_masks is None:
            return {}
        if not isinstance(valid_masks, dict):
            raise TypeError(
                f"valid_masks must be a dict {{stain: [B,N] bool}} or None, got "
                f"{type(valid_masks).__name__}")
        unknown = sorted(set(valid_masks) - set(self.stain_order))
        if unknown:
            raise ValueError(
                f"valid_masks has unknown stains {unknown}; model expects "
                f"{self.stain_order}")
        out = {}
        for stain, m in valid_masks.items():
            if m is None:
                continue
            m = m.to(device=feats[stain].device, dtype=torch.bool)
            if m.dim() == 1:
                m = m.unsqueeze(0)
            if m.shape != feats[stain].shape[:2]:
                raise ValueError(
                    f"valid mask for {stain!r} has shape {tuple(m.shape)}, "
                    f"expected {tuple(feats[stain].shape[:2])}")
            n_invalid = int((~m).sum().item())
            if n_invalid:
                raise ValueError(
                    f"valid mask for {stain!r} marks {n_invalid}/"
                    f"{m.numel()} token(s) invalid. Stage-1 RRT does not "
                    f"support partially-invalid bags yet, so such a mask would "
                    f"be applied by the Stage-2 cross branch but silently "
                    f"ignored by the encoder. Pass None (or an all-True mask) "
                    f"until Stage-1 masking is implemented.")
            out[stain] = m
        return out

    # ------------------------------------------------------------------
    # core computation
    # ------------------------------------------------------------------
    def encode(self, features: Dict[str, torch.Tensor]):
        """Stage-1 for every stain, evaluated exactly once per forward.

        Returns `(projected, encoded)`: `projected[m]` is the post-projection,
        post-dropout patch embedding and `encoded[m]` the Stage-1 RRT output.

        Anything that wants the intermediate representations (`return_features`)
        reads them back from here instead of re-running the projection: in train
        mode a second call would draw a *fresh* dropout mask and report tensors
        the forward never actually used.
        """
        projected = {m: self.dp(self.patch_to_emb[m](features[m]))
                     for m in self.stain_order}
        encoded = {m: self._as_3d(self.rrt[m](projected[m]))
                   for m in self.stain_order}
        return projected, encoded

    def encode_he(self, features: Dict[str, torch.Tensor],
                  encoded: Optional[Dict[str, torch.Tensor]] = None
                  ) -> torch.Tensor:
        """H = he_encoder(proj_HE(x_HE)) — computed once per forward.

        Pass `encoded` (from `encode`) to reuse an already-computed Stage-1
        result rather than running the HE branch a second time.
        """
        if encoded is not None:
            return encoded[self.he_stain]
        he_emb = self.dp(self.patch_to_emb[self.he_stain](features[self.he_stain]))
        return self._as_3d(self.rrt[self.he_stain](he_emb))

    def fuse(self, features: Dict[str, torch.Tensor], valid=None,
             encoded: Optional[Dict[str, torch.Tensor]] = None):
        """Fuse HE with every auxiliary stain.

        Returns:
            (fused, H, branch_outputs) where `fused = H` when there is no
            auxiliary stain, else `mean_m branch_m(H, Z_m)`; each
            `branch_m(H, Z_m) = H + residual_scale · Δ_m` already contains the
            HE residual.

        `encoded` may carry the Stage-1 results from `encode()` so a caller that
        already ran Stage-1 does not run it twice.
        """
        valid = valid or {}
        if encoded is None:
            _, encoded = self.encode(features)
        H = encoded[self.he_stain]                         # [B, N_HE, D] — once

        if not self.aux_stains:
            return H, H, {}

        vh = valid.get(self.he_stain)
        branch_outputs = {}
        for stain in self.aux_stains:
            # Every branch reads the SAME H object: branches are parallel and
            # never mutate the HE representation.
            branch_outputs[stain] = self.cross_branches[stain](
                H, encoded[stain], valid_he=vh, valid_pr=valid.get(stain))

        stacked = torch.stack([branch_outputs[s] for s in self.aux_stains], dim=0)
        # Mean of complete branch outputs (each = H + s·Δ_m) ⇒ H + (s/M)·Σ Δ_m.
        # No extra H, no second residual_scale.  M == 1 ⇒ identical to that branch.
        fused = stacked.mean(dim=0)
        return fused, H, branch_outputs

    # ------------------------------------------------------------------
    # forward — mirrors the MM_RRT_ABMIL tuple contract used by the trainer
    # ------------------------------------------------------------------
    def forward(self, x, valid_masks=None, modality_names=None,
                return_features=False, return_modality_attns=False):
        """Args:
            x: dict {stain: [B,N,input_dim]} or list/tuple in `stain_order`.
            valid_masks: optional dict {stain: [B,N] bool} of valid patch tokens.
            modality_names: optional names for a list input (cross-checked).
            return_features: return the rich dict instead of the tuple.

        Returns (mirrors `MM_RRT_ABMIL`):
            (logits, Y_hat, attention, fusion_stats, aux_loss)
        with `attention` / `fusion_stats` entries optional (the trainer must not
        assume any particular MIL returns attention).
        """
        feats = self._normalize_inputs(x, modality_names=modality_names)
        valid = self._normalize_masks(valid_masks, feats)

        # Stage 1 runs exactly once; the fused output and every `return_features`
        # entry are read off these tensors.
        projected, encoded = self.encode(feats)
        fused, H, branch_outputs = self.fuse(feats, valid, encoded=encoded)
        vh = valid.get(self.he_stain)

        # tokens + validity mask → pluggable MIL head
        mil_result = self.mil(tokens=fused, mask=vh)
        logits = mil_result['logits']
        Y_hat = torch.argmax(logits, dim=-1)
        attention = mil_result.get('attention')      # may be None for some MILs

        with torch.no_grad():
            fusion_stats = {
                'model_family': MODEL_FAMILY,
                'he_stain': self.he_stain,
                'aux_stains': list(self.aux_stains),
                'n_branches': len(self.aux_stains),
                'z_he_norm': float(H.detach().norm(dim=-1).mean()),
                'branch_delta_norm': {
                    s: float((branch_outputs[s].detach() - H.detach())
                             .norm(dim=-1).mean())
                    for s in self.aux_stains
                },
            }

        if return_features:
            return {
                'logits': logits,
                'prediction': Y_hat,
                'attention': attention,
                'fused_features': fused,
                # cached Stage-1 tensors from THIS forward — never recomputed
                'embedded_features': [projected[m] for m in self.stain_order],
                'encoded_features': [encoded[m] for m in self.stain_order],
                'fusion_stats': fusion_stats,
            }
        return logits, Y_hat, attention, fusion_stats, torch.tensor(
            0.0, device=logits.device)

    # ------------------------------------------------------------------
    # config / checkpoint metadata
    # ------------------------------------------------------------------
    def get_config(self) -> dict:
        """Full construction config — stored in every checkpoint.

        Contains every argument `HEAuxUnifiedModel.__init__` accepts (see
        `_CONSTRUCTION_KEYS`), so `build_he_aux_unified_from_model_config` can
        rebuild the model from a checkpoint alone.  The remaining entries
        (`*_resolved`, `*_source`, `fusion`) are derived metadata, recorded so a
        restored run can be audited without re-deriving them.
        """
        return {
            'config_schema_version': MODEL_CONFIG_SCHEMA_VERSION,
            'model_family': MODEL_FAMILY,
            # ── construction arguments ─────────────────────────────────────
            'modality_list': list(self.stain_order),
            'he_stain': self.he_stain,
            'input_dim': self.input_dim,
            'mlp_dim': self.mlp_dim,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'act': self.act,
            'encoder_cfg': copy.deepcopy(self.encoder_cfg),
            'stage2_cfg': copy.deepcopy(self.stage2_cfg),
            'mil_cfg': copy.deepcopy(self.mil_cfg),
            'init_seed': self.init_seed,
            'region_num': self.region_num,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'drop_path': self.drop_path,
            'trans_dropout': self.trans_dropout,
            'epeg': self.epeg,
            'epeg_k': self.epeg_k,
            'crmsa_k': self.crmsa_k,
            'cr_msa': self.cr_msa,
            'all_shortcut': self.all_shortcut,
            'crmsa_mlp': self.crmsa_mlp,
            'crmsa_heads': self.crmsa_heads,
            # ── derived metadata ───────────────────────────────────────────
            'aux_stains': list(self.aux_stains),
            'encoder_cfg_resolved': copy.deepcopy(self.encoder_cfg_resolved),
            'encoder_cfg_source': dict(self.encoder_cfg_source),
            'fusion': {
                'formula': 'H + (residual_scale / M) * sum_m delta_m',
                'residual_scale': float(self.stage2_cfg.get('residual_scale', 0.1)),
                'tau': float(self.stage2_cfg.get('temperature', 0.2)),
                'prototype_momentum': float(
                    self.stage2_cfg.get('prototype_momentum', 0.99)),
                'branch_order': list(self.aux_stains),
            },
        }

    def get_modality_names(self) -> List[str]:
        return list(self.stain_order)

    def get_model_info(self) -> dict:
        return {
            'model_family': MODEL_FAMILY,
            'num_modalities': self.num_modalities,
            'modality_list': list(self.stain_order),
            'aux_stains': list(self.aux_stains),
            'input_dim': self.input_dim,
            'mlp_dim': self.mlp_dim,
            'num_classes': self.num_classes,
            'mil_type': self.mil_type,
            'fusion_type': 'he_aux_unified',
        }


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def build_he_aux_unified(modality_list, input_dim=768, mlp_dim=512,
                         num_classes=2, dropout=0.25, act='relu',
                         encoder_cfg=None, stage2_cfg=None, mil_cfg=None,
                         init_seed=1234, **kwargs) -> HEAuxUnifiedModel:
    """Build the unified model from explicit arguments (see the class docstring)."""
    return HEAuxUnifiedModel(
        modality_list=modality_list, input_dim=input_dim, mlp_dim=mlp_dim,
        num_classes=num_classes, dropout=dropout, act=act,
        encoder_cfg=encoder_cfg, stage2_cfg=stage2_cfg, mil_cfg=mil_cfg,
        init_seed=init_seed, **kwargs)


def build_he_aux_unified_from_config(model_cfg: dict, data_cfg: dict,
                                     **overrides) -> HEAuxUnifiedModel:
    """Build from a trainer config (`model` + `data` blocks).

    Reads `data.modalities` (HE must be listed first — this is what makes the
    positional list input of `validate()`/`train_epoch()` unambiguous),
    `model.encoder_cfg`, `model.stage2_cfg` and the MIL block.  The MIL block is
    either the modern `model.mil_cfg = {'name', 'kwargs'}` or the legacy flat
    `mil_type` + `abmil_hidden_dim` + `dropout` triple (auto-converted, so
    existing v3-style configs work unchanged).
    """
    modalities = [str(m) for m in data_cfg['modalities']]
    if not modalities:
        raise ValueError("data.modalities must not be empty")
    if modalities[0] != HE_STAIN:
        raise ValueError(
            f"data.modalities must list {HE_STAIN!r} first (got {modalities}); "
            f"positional inputs from the dataset are matched against this order.")

    mil_cfg = model_cfg.get('mil_cfg')
    if mil_cfg is None:
        kwargs = {
            'hidden_dim': model_cfg.get('abmil_hidden_dim', 256),
            'dropout_rate': model_cfg.get('dropout', 0.25),
        }
        mil_cfg = {'name': model_cfg.get('mil_type', 'abmil'), 'kwargs': kwargs}

    cfg = dict(
        modality_list=modalities,
        input_dim=data_cfg.get('input_dim', 768),
        mlp_dim=model_cfg.get('mlp_dim', 512),
        num_classes=data_cfg.get('num_classes', 2),
        dropout=model_cfg.get('dropout', 0.25),
        act=model_cfg.get('act', 'relu'),
        encoder_cfg=model_cfg.get('encoder_cfg'),
        stage2_cfg=model_cfg.get('stage2_cfg'),
        mil_cfg=mil_cfg,
        init_seed=model_cfg.get('init_seed', 1234),
    )
    for key in ('region_num', 'n_layers', 'n_heads', 'drop_path', 'trans_dropout',
                'epeg', 'epeg_k', 'crmsa_k', 'cr_msa', 'all_shortcut',
                'crmsa_mlp', 'crmsa_heads'):
        if key in model_cfg:
            cfg[key] = model_cfg[key]
    cfg.update(overrides)
    return build_he_aux_unified(**cfg)


def build_he_aux_unified_from_model_config(model_config: dict,
                                           **overrides) -> HEAuxUnifiedModel:
    """Rebuild the model from a checkpoint's `model_config` **alone**.

    This is the supported restore path: no training config, no `data` block, no
    command-line flags.  The stored `config_schema_version` is checked first, so
    an unknown/older layout fails loudly instead of being reinterpreted under
    today's key meanings.

    Pair it with `load_state_dict(ckpt['model_state_dict'], strict=True)` — the
    point of rebuilding from the same dict the model wrote is that `strict=True`
    can then be used without surprises.
    """
    if not isinstance(model_config, dict):
        raise TypeError(
            f"model_config must be a dict, got {type(model_config).__name__}")

    version = model_config.get('config_schema_version')
    if version is None:
        raise ValueError(
            "checkpoint 'model_config' has no 'config_schema_version' — it "
            "predates schema versioning and cannot be restored automatically. "
            f"Re-export it with a model that writes version "
            f"{MODEL_CONFIG_SCHEMA_VERSION}.")
    if version != MODEL_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported model_config schema version {version!r}; this build "
            f"understands version {MODEL_CONFIG_SCHEMA_VERSION} only.")

    family = model_config.get('model_family')
    if family is not None and family != MODEL_FAMILY:
        raise ValueError(
            f"model_config is for model_family {family!r}, not {MODEL_FAMILY!r}")

    kwargs = {k: copy.deepcopy(model_config[k])
              for k in _CONSTRUCTION_KEYS if k in model_config}
    missing = [k for k in _REQUIRED_CONSTRUCTION_KEYS if k not in kwargs]
    if missing:
        raise ValueError(
            f"model_config is missing required construction key(s) {missing}")
    kwargs.update(overrides)
    return build_he_aux_unified(**kwargs)


# ---------------------------------------------------------------------------
# v3 checkpoint → unified model
# ---------------------------------------------------------------------------
#: Explicit legacy-key → unified-key mapping for the historical v3 two-stage
#: model (`MM_RRT_ABMIL`, HE + one IHC stain).  Legacy prefixes are unique, so
#: the first match wins.  `mil.` gains the adapter's `module.` segment because
#: the head is wrapped by a `MILHeadAdapter`.
#: Legacy prefix → (target submodule path template, stain slot)
_V3_PREFIX_TO_TARGET = {
    'patch_to_emb.0.': ('patch_to_emb.{stain}.', 'he'),
    'patch_to_emb.1.': ('patch_to_emb.{stain}.', 'aux'),
    'rrt_he.': ('rrt.{stain}.', 'he'),
    'rrt_ihc.': ('rrt.{stain}.', 'aux'),
    'cross_region_mod.': ('cross_branches.{stain}.', 'aux'),
    'mil.': ('mil.module.', None),
}


def map_v3_state_dict_to_unified(state_dict: dict, he_stain: str = HE_STAIN,
                                 aux_stain: str = 'PR'):
    """Map a legacy v3 state dict onto unified-model keys.

    Returns `(mapped, unmapped_keys)`.  Purely a key rename — every tensor is
    carried over unchanged, so the caller can verify completeness and shapes
    before loading (no blanket `strict=False`).
    """
    mapped, unmapped = {}, []
    for key, value in state_dict.items():
        for prefix, (target, slot) in _V3_PREFIX_TO_TARGET.items():
            if key.startswith(prefix):
                stain = he_stain if slot == 'he' else (
                    aux_stain if slot == 'aux' else None)
                target_prefix = target.format(stain=stain) if stain else target
                mapped[target_prefix + key[len(prefix):]] = value
                break
        else:
            unmapped.append(key)
    return mapped, unmapped


def checkpoint_aux_stain(ckpt: dict, default: str = 'PR') -> str:
    """Infer the auxiliary stain name of a v3 checkpoint from its metadata."""
    modalities = ckpt.get('modalities') or []
    if len(modalities) >= 2:
        return str(modalities[1])
    cfg_mods = (ckpt.get('config') or {}).get('data', {}).get('modalities') or []
    if len(cfg_mods) >= 2:
        return str(cfg_mods[1])
    return default


def load_v3_checkpoint_into_unified(model: HEAuxUnifiedModel, ckpt_path: str,
                                    he_stain: str = HE_STAIN,
                                    aux_stain: Optional[str] = None,
                                    map_location='cpu', verbose: bool = True):
    """Load a legacy v3 checkpoint into a unified model with explicit checks.

    Every legacy key must map to exactly one unified key of the same shape and
    every unified parameter must be covered; otherwise a `ValueError` listing
    the offenders is raised (a missing/extra parameter is never swallowed).

    Returns a report dict: `{'mapped': n, 'aux_stain': ..., 'ckpt_epoch': ...}`.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if 'model_state_dict' not in ckpt:
        raise ValueError(f"{ckpt_path}: no 'model_state_dict' entry")

    aux_stain = aux_stain or checkpoint_aux_stain(ckpt)
    old_sd = ckpt['model_state_dict']
    mapped, unmapped = map_v3_state_dict_to_unified(old_sd, he_stain, aux_stain)

    if aux_stain not in model.aux_stains:
        raise ValueError(
            f"checkpoint auxiliary stain {aux_stain!r} is not an auxiliary stain "
            f"of this model (aux_stains={model.aux_stains})")

    target = model.state_dict()
    missing = sorted(k for k in target if k not in mapped)
    unexpected = sorted(k for k in mapped if k not in target)
    shape_mismatch = sorted(
        (k, tuple(mapped[k].shape), tuple(target[k].shape))
        for k in mapped if k in target and mapped[k].shape != target[k].shape)

    problems = []
    if unmapped:
        problems.append(f"{len(unmapped)} unmapped legacy key(s): {unmapped[:5]}")
    if missing:
        problems.append(f"{len(missing)} unified param(s) not covered: {missing[:5]}")
    if unexpected:
        problems.append(f"{len(unexpected)} mapped key(s) not in model: {unexpected[:5]}")
    if shape_mismatch:
        problems.append(f"{len(shape_mismatch)} shape mismatch(es): {shape_mismatch[:3]}")
    if problems:
        raise ValueError(
            f"Cannot load v3 checkpoint {ckpt_path} into the unified model "
            f"(aux_stain={aux_stain!r}): " + '; '.join(problems))

    model.load_state_dict(mapped, strict=True)
    report = {
        'path': str(ckpt_path),
        'aux_stain': aux_stain,
        'he_stain': he_stain,
        'n_mapped': len(mapped),
        'ckpt_epoch': ckpt.get('epoch'),
        'ckpt_modalities': ckpt.get('modalities'),
    }
    if verbose:
        print(f"[ckpt] v3 → unified: {report['n_mapped']} tensors mapped "
              f"(HE + {aux_stain}), epoch={report['ckpt_epoch']}")
    return report
