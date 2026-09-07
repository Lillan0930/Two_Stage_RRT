"""
HE-residual cross-CR-MSA — Stage 2 of Two-stage R²T, PR-guided HE residual.

A *directed* alternative to the symmetric `CrossStainingCRMSA`.  Instead of
treating HE and PR as one joint region set and dispatching both back to their
own patches (then concatenating), this module keeps the HE patch sequence
[Z_HE] as the *only* output and writes a residual `Δ_HE` onto it, where `Δ_HE`
is produced by a full-slot HE→PR cross-attention over each modality's own
routing tokens.

    Stage 1 (unchanged):
        HE → RRTEncoder → Z_HE   [B, N_HE, D]
        PR → RRTEncoder → Z_PR   [B, N_PR, D]

    Stage 2 (this module) — directed HE→PR cross-attention + HE residual:
        R_HE = Route(Z_HE)            routing tokens, [B, G_HE·k, D]
        R_PR = Route(Z_PR)            routing tokens, [B, G_PR·k, D]
        Q    = W_q_he( attn_norm_he(R_HE) )
        K    = W_k_pr( attn_norm_pr(R_PR) )
        V    = W_v_pr( attn_norm_pr(R_PR) )
        Δ_rt = W_out( softmax(Q·Kᵀ/√head_dim + mask) · V )     # [B, G_HE·k, D]
        Δ_HE = Dispatch_HE(Δ_rt)                               # [B, N_HE, D]
        out  = Z_HE + residual_scale · Δ_HE                    # [B, N_HE, D]  → ABMIL

Key properties (all enforced here, tested in tests/):
  * **Independent modality routing** — `route_norm_he/pr` and `phi_he/pr` are
    separate `nn.Module`/`nn.Parameter` instances (never shared), each region
    keeps `crmsa_k` routing tokens (no single average-pool collapse).
  * **Full-slot cross-attention** — Q comes only from HE, K/V only from PR;
    every HE slot attends every valid PR slot (no same-slot restriction), scaled
    by `head_dim` (not full `D`).
  * **HE patch residual write-back** — the residual base is the *original* Z_HE
    (identity skip); no FFN / LayerNorm / self-attention after the residual, no
    concat of PR tokens, no broadcast/pooled-HE replacement, no `MLP([HE, x])`.
  * **Padding / mask** — validity comes from length or an explicit mask (never
    inferred from "all-zero features"); combine & dispatch min-max run only over
    valid patches; empty regions yield zero routing + invalid slots; raw logits
    are never mutated in place to `-inf`; a sample with no valid PR tokens gets a
    zero Δ (strictly keeps HE); invalid queries are re-zeroed after the output
    projection so projection bias cannot resurrect a non-zero Δ.
  * `residual_scale` is a fixed (non-learned) buffer; `disable_cross=True` or
    `residual_scale == 0` returns Z_HE **unchanged** (identity).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from models.rmsa import region_partition, region_reverse


class HEResidualCrossCRMSA(nn.Module):
    """Directed HE→PR cross-attention CR-MSA writing a residual onto HE only.

    Args:
        dim:           feature dimension (512)
        num_heads:     cross-attention heads (stage2 `crmsa_heads`); D must be
                       divisible by num_heads
        region_num:    regions per side; G = region_num² per modality (default 4)
        crmsa_k:       routing tokens per region (default 3)
        drop_out:      attention / projection dropout
        drop_path:     stochastic depth on the residual (default 0.0 → identity)
        epeg:          unused, accepted for config compatibility (kept False)
        epeg_k:        unused, accepted for config compatibility
        crmsa_mlp:     whether phi is a learned MLP (else a [dim, crmsa_k] param)
        ffn:           unused, accepted for config compatibility (always False)
        qkv_bias:      bias on Q/K/V/output projections
        residual_scale: fixed scalar applied to Δ_HE (buffer, default 0.1)
        disable_cross:  if True, forward returns Z_HE unchanged
        region_size / min_region_num / min_region_ratio: forwarded to `_pad`
    """

    def __init__(self, dim=512, num_heads=8, region_num=4, crmsa_k=3,
                 drop_out=0.1, drop_path=0.0, epeg=False, epeg_k=15,
                 crmsa_mlp=False, ffn=False, ffn_act='gelu', mlp_ratio=4.,
                 region_size=0, min_region_num=0, min_region_ratio=0,
                 qkv_bias=True, residual_scale=0.1, disable_cross=False,
                 **kwargs):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.region_num = region_num
        self.region_size = region_size if region_size > 0 else None
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio
        self.crmsa_k = crmsa_k
        self.crmsa_mlp = crmsa_mlp
        self.disable_cross = disable_cross

        # Fixed, non-learned residual coefficient (buffer → checkpointed, not
        # optimized).  residual_scale == 0 ⇒ strict identity on Z_HE.
        self.register_buffer('residual_scale', torch.tensor(float(residual_scale)))

        # ── Independent per-modality pre-LN (before combine) ──
        self.route_norm_he = nn.LayerNorm(dim)
        self.route_norm_pr = nn.LayerNorm(dim)

        # ── Independent per-modality routing phi ──
        if crmsa_mlp:
            self.phi_he = nn.Sequential(
                nn.Linear(dim, dim // 4, bias=False),
                nn.Tanh(),
                nn.Linear(dim // 4, crmsa_k, bias=False),
            )
            self.phi_pr = nn.Sequential(
                nn.Linear(dim, dim // 4, bias=False),
                nn.Tanh(),
                nn.Linear(dim // 4, crmsa_k, bias=False),
            )
        else:
            self.phi_he = nn.Parameter(torch.empty((dim, crmsa_k)))
            self.phi_pr = nn.Parameter(torch.empty((dim, crmsa_k)))
            nn.init.kaiming_uniform_(self.phi_he, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.phi_pr, a=math.sqrt(5))

        # ── Independent per-modality attention pre-LN (on routing tokens) ──
        self.attn_norm_he = nn.LayerNorm(dim)
        self.attn_norm_pr = nn.LayerNorm(dim)

        # ── Directed cross-attention projections ──
        # Q only from HE; K/V only from PR; normal non-zero init (no zero-init
        # out-proj, no dual zero-init with the residual coefficient).
        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_out = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(drop_out)
        self.proj_drop = nn.Dropout(drop_out)

        # Stochastic depth on the residual (identity for drop_path == 0).
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in (self.w_q, self.w_k, self.w_v, self.w_out):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------------
    # Padding — mirrors official CrossRegionAttention.padding
    # ------------------------------------------------------------------
    def _pad(self, x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H + _n, W + _n
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H + _n, W + _n
            region_size = int(H // self.region_num)

        add_length = H * W - L

        # if padding much, give up region attention (only for ablation)
        if (add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

        return x, H, W, add_length, region_size

    @staticmethod
    def _partition_mask(valid_grid, region_size):
        """[B, H, W] bool → [B, G, P] bool (same region ordering as region_partition)."""
        B, H, W = valid_grid.shape
        v = valid_grid.view(B, H // region_size, region_size,
                            W // region_size, region_size)
        v = v.permute(0, 1, 3, 2, 4).contiguous()
        v = v.view(B, (H // region_size) * (W // region_size),
                   region_size * region_size)
        return v

    def _route(self, z, phi, route_norm, valid):
        """Pre-LN → pad → partition → routing tokens + dispatch weights.

        Returns:
            routing:             [B, G, k, D]
            dispatch_weights_mm: [B, G, k, P]
            dispatch_weights:    [B, G, k, P]
            valid_slots:         [B, G, k] bool (False for empty regions)
            H, W, add_length, region_size
        """
        B, L, D = z.shape

        # Pre-LN on the real patch sequence (before padding), so zero-padding is
        # never fed through LayerNorm (LayerNorm(0) == learned bias after init).
        z_n = route_norm(z)                                    # [B, N, D]

        # pad + partition
        x_padded, H, W, add_length, region_size = self._pad(z_n)   # [B, H*W, D]
        x_grid = x_padded.view(B, H, W, D)
        x_regions = region_partition(x_grid, region_size)          # [B*G, rs, rs, D]
        G = (H // region_size) * (W // region_size)
        P = region_size * region_size
        x_regions = x_regions.view(B, G, region_size, region_size, D)
        x_regions = x_regions.view(B, G, P, D)                      # [B, G, P, D]

        # validity mask: real patches (from length) + optional explicit mask;
        # module-added padding is always invalid.
        if valid is None:
            valid_full = torch.ones(B, L, device=z.device, dtype=torch.bool)
        else:
            valid_full = valid.to(device=z.device, dtype=torch.bool).reshape(B, L)
        if add_length > 0:
            pad_valid = torch.zeros(B, add_length, device=z.device, dtype=torch.bool)
            valid_full = torch.cat([valid_full, pad_valid], dim=1)
        valid_grid = valid_full.view(B, H, W)
        valid_regions = self._partition_mask(valid_grid, region_size)   # [B, G, P] bool

        # routing logits: patch → crmsa_k (raw logits preserved, never in-place -inf)
        if self.crmsa_mlp:
            logits = phi(x_regions).transpose(-1, -2)          # [B, G, k, P]
        else:
            logits = torch.einsum('bgpd, dk -> bgkp', x_regions, phi)   # [B, G, k, P]
        k = logits.shape[2]

        valid_p = valid_regions                                   # [B, G, P]
        nonempty = valid_p.any(dim=-1)                            # [B, G] bool
        nonempty_kp = nonempty.unsqueeze(-1).unsqueeze(-1)        # [B, G, 1, 1]

        # combine: masked softmax over P (invalid patches excluded).  Empty
        # regions → zero weights (avoids softmax over all -inf → NaN).
        logits_masked = logits.masked_fill(~valid_p.unsqueeze(2), float('-inf'))
        combine_weights = torch.softmax(logits_masked, dim=-1)    # [B, G, k, P]
        combine_weights = torch.where(
            nonempty_kp, combine_weights, torch.zeros_like(combine_weights))

        routing = torch.einsum('bgpd, bgkp -> bgkd', x_regions, combine_weights)  # [B,G,k,D]
        routing = routing * nonempty_kp                           # exact zero for empty

        # dispatch: softmax over slots k (no mask — k is always a valid axis).
        dispatch_weights = torch.softmax(logits, dim=2)           # [B, G, k, P]

        # dispatch min-max over valid patches only (raw logits untouched).
        logits_max = logits.masked_fill(
            ~valid_p.unsqueeze(2), float('-inf')).max(dim=-1, keepdim=True).values
        logits_min = logits.masked_fill(
            ~valid_p.unsqueeze(2), float('inf')).min(dim=-1, keepdim=True).values
        dispatch_weights_mm = (logits - logits_min) / \
            (logits_max - logits_min + 1e-8)                     # [B, G, k, P]
        # empty regions would produce NaN (max=-inf / min=+inf) → zero them.
        dispatch_weights_mm = torch.where(
            nonempty_kp, dispatch_weights_mm,
            torch.zeros_like(dispatch_weights_mm))

        valid_slots = nonempty.unsqueeze(-1).expand(B, G, k).contiguous()  # [B, G, k]

        return routing, dispatch_weights_mm, dispatch_weights, valid_slots, \
            H, W, add_length, region_size

    def _dispatch(self, delta_routing, dispatch_weights_mm, dispatch_weights,
                  region_size, H, W, add_length):
        """[B, G, k, D] routing deltas → [B, N, D] patch deltas (official reverse)."""
        B, G, k, D = delta_routing.shape
        P = dispatch_weights_mm.shape[-1]

        delta_regions = torch.einsum(
            'bgkd, bgkp -> bgpd', delta_routing,
            dispatch_weights_mm * dispatch_weights)              # [B, G, P, D]

        delta_regions = delta_regions.view(B, G, region_size, region_size, D)
        delta_regions = delta_regions.view(B * G, region_size, region_size, D)
        x = region_reverse(delta_regions, region_size, H, W)      # [B, H, W, D]
        x = x.view(B, H * W, D)
        if add_length > 0:
            x = x[:, :-add_length]
        return x

    def _cross_attention(self, r_he, r_pr, q_valid, k_valid):
        """Full-slot directed cross-attention.

        Args:
            r_he: [B, Q, D] HE routing tokens (Q = G_HE·k)
            r_pr: [B, K, D] PR routing tokens (K = G_PR·k)
            q_valid: [B, Q] bool
            k_valid: [B, K] bool

        Returns:
            delta_routing: [B, Q, D] (invalid HE queries / no-valid-PR samples → 0)
        """
        B, Q, D = r_he.shape
        K = r_pr.shape[1]
        h = self.num_heads
        d = self.head_dim

        r_he_n = self.attn_norm_he(r_he)                          # [B, Q, D]
        r_pr_n = self.attn_norm_pr(r_pr)                          # [B, K, D]
        q = self.w_q(r_he_n)                                      # [B, Q, D]
        k = self.w_k(r_pr_n)                                      # [B, K, D]
        v = self.w_v(r_pr_n)                                      # [B, K, D]

        q = q.view(B, Q, h, d).transpose(1, 2)                    # [B, h, Q, d]
        k = k.view(B, K, h, d).transpose(1, 2)                    # [B, h, K, d]
        v = v.view(B, K, h, d).transpose(1, 2)                    # [B, h, K, d]

        scale = d ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale                  # [B, h, Q, K]

        # mask invalid PR keys
        attn = attn.masked_fill(
            ~k_valid.view(B, 1, 1, K), float('-inf'))

        has_valid_key = k_valid.any(dim=-1)                       # [B] bool
        attn_weights = torch.softmax(attn, dim=-1)                # [B, h, Q, K]
        # no valid PR key → softmax over all -inf → NaN → force to zero
        attn_weights = torch.where(
            has_valid_key.view(B, 1, 1, 1), attn_weights,
            torch.zeros_like(attn_weights))
        attn_weights = self.attn_drop(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).contiguous()     # [B, Q, h, d]
        out = out.view(B, Q, D)
        out = self.proj_drop(self.w_out(out))                     # [B, Q, D]

        # Re-zero invalid HE queries and no-valid-PR samples after the output
        # projection, so projection bias cannot regenerate a non-zero Δ.
        out = out * q_valid.float().unsqueeze(-1)
        out = out * has_valid_key.float().view(B, 1, 1)

        return out

    def forward(self, z_he, z_pr, valid_he=None, valid_pr=None):
        """Directed HE→PR cross-attention + HE residual write-back.

        Args:
            z_he: [B, N_HE, D]
            z_pr: [B, N_PR, D]
            valid_he: optional [B, N_HE] bool
            valid_pr: optional [B, N_PR] bool

        Returns:
            [B, N_HE, D] — Z_HE + residual_scale · Δ_HE (order-preserving)
        """
        if self.disable_cross or float(self.residual_scale) == 0.0:
            return z_he

        if z_he.dim() == 2:
            z_he = z_he.unsqueeze(0)
            z_pr = z_pr.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False
        B = z_he.shape[0]

        # Independent per-modality routing.
        routing_he, dmm_he, dw_he, valid_he_slots, H_he, W_he, add_he, rs_he = \
            self._route(z_he, self.phi_he, self.route_norm_he, valid_he)
        routing_pr, dmm_pr, dw_pr, valid_pr_slots, H_pr, W_pr, add_pr, rs_pr = \
            self._route(z_pr, self.phi_pr, self.route_norm_pr, valid_pr)

        # Flatten slots for cross-attention: [B, G·k, D].
        r_he = routing_he.reshape(B, -1, self.dim)
        r_pr = routing_pr.reshape(B, -1, self.dim)
        q_valid = valid_he_slots.reshape(B, -1)
        k_valid = valid_pr_slots.reshape(B, -1)

        delta_routing = self._cross_attention(r_he, r_pr, q_valid, k_valid)  # [B, Q, D]
        delta_routing = delta_routing.view(B, -1, self.crmsa_k, self.dim)    # [B, G_HE, k, D]

        delta_patch = self._dispatch(delta_routing, dmm_he, dw_he,
                                     rs_he, H_he, W_he, add_he)              # [B, N_HE, D]

        # Explicitly-invalid HE patches are truly absent: zero their residual so
        # their input values never leak into a non-zero Δ (identity at those slots).
        if valid_he is not None:
            vh = valid_he.to(device=z_he.device, dtype=z_he.dtype) \
                         .reshape(B, z_he.shape[1])
            delta_patch = delta_patch * vh.unsqueeze(-1)

        out = z_he + self.drop_path(self.residual_scale * delta_patch)

        if squeezed:
            out = out.squeeze(0)
        return out
