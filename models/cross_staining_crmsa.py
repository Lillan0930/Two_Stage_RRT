"""
Cross-Staining CR-MSA TransLayer — official RRT CR-MSA extended to dual-staining.

Stage 2 of Two-stage R²T.  Faithfully mirrors the official RRT-MIL
`CrossRegionAttention` (models/rmsa.py), the *only* change being how the input
regions are organized: HE and PR regions are treated as one unified region set,
so the combine → cross-region attention → dispatch pipeline runs jointly over
both stainings.

    Stage 1 (unchanged):
        HE → RRTEncoder → Z_HE   [B, N_HE, D]
        PR → RRTEncoder → Z_PR   [B, N_PR, D]

    Stage 2 (this module):
        Z_HE, Z_PR
          → pad + region_partition (official)   → 16 HE regions + 16 PR regions
          → phi(·) logits → combine_weights / dispatch_weights (official)
          → routing tokens (crmsa_k per region) → concat [HE|PR] region set
          → InnerAttention over 32 regions (official, no MultiheadAttention)
          → split back → dispatch_weights_mm / dispatch_weights → region_reverse
          → Z_HE_out [B, N_HE, D], Z_PR_out [B, N_PR, D]
          → LayerNorm → concat([Z_HE_out, Z_PR_out], dim=1)   [B, N_HE+N_PR, D]

No HE anchor, no fusion gate, no alpha correction, no modality weighting,
no cross-attention query/key/value, no new fusion loss.  The combine-attention-
dispatch mechanism is preserved verbatim from the official CR-MSA; the residual
is the official TransLayer residual (x = x + DropPath(z)) — no all_shortcut,
because this Stage 2 is CR-MSA only (there is no preceding R-MSA to shortcut).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath

from models.rmsa import InnerAttention, Mlp, region_partition, region_reverse


class CrossStainingCRMSA(nn.Module):
    """Official-style cross-staining CR-MSA TransLayer.

    Args:
        dim:          feature dimension (512)
        num_heads:    attention heads for the routing-token MSA (crmsa_heads)
        region_num:   number of regions per side; K = region_num² per staining
        crmsa_k:      routing tokens per region (official `crmsa_k`)
        drop_out:     attention / projection dropout
        drop_path:    stochastic depth on the residual
        epeg:         whether InnerAttention uses EPEG (official)
        epeg_k:       EPEG kernel size
        crmsa_mlp:    whether phi is a learned MLP (else a [dim, crmsa_k] param)
        ffn:          whether to append the official FFN (default False, matches
                      current RRT config)
    """

    def __init__(self, dim=512, num_heads=8, region_num=4, crmsa_k=3,
                 drop_out=0.1, drop_path=0.0, epeg=True, epeg_k=9,
                 crmsa_mlp=False, ffn=False, ffn_act='gelu', mlp_ratio=4.,
                 region_size=0, min_region_num=0, min_region_ratio=0,
                 qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.region_num = region_num
        self.region_size = region_size if region_size > 0 else None
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        # Pre-norm (official TransLayer) + optional FFN norm
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim) if ffn else nn.Identity()

        # Routing-token MSA — official InnerAttention (NOT nn.MultiheadAttention)
        self.attn = InnerAttention(
            dim, num_heads=num_heads, head_dim=dim // num_heads,
            qkv_bias=qkv_bias, attn_drop=drop_out, proj_drop=drop_out,
            epeg=epeg, epeg_k=epeg_k)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # phi: patch → crmsa_k routing-token logits (shared across HE & PR)
        self.crmsa_mlp = crmsa_mlp
        if crmsa_mlp:
            self.phi = nn.Sequential(
                nn.Linear(self.dim, self.dim // 4, bias=False),
                nn.Tanh(),
                nn.Linear(self.dim // 4, crmsa_k, bias=False),
            )
        else:
            self.phi = nn.Parameter(torch.empty((self.dim, crmsa_k),))
            nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

        # Optional FFN (matches official TransLayer)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = (Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                        act_layer=act_layer, drop=drop_out)
                    if ffn else nn.Identity())

        # Final norm (matches RRTEncoder's trailing LayerNorm)
        self.out_norm = nn.LayerNorm(dim)

    # ------------------------------------------------------------------
    # Padding — verbatim from official CrossRegionAttention.padding
    # ------------------------------------------------------------------
    def _pad(self, x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H + _n, W + _n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H + _n, W + _n
            region_size = int(H // self.region_num)
            region_num = self.region_num

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

        return x, H, W, add_length, region_num, region_size

    # ------------------------------------------------------------------
    # Combine — verbatim from official CrossRegionAttention (first half)
    # ------------------------------------------------------------------
    def _combine(self, x):
        """pad → region_partition → phi → combine/dispatch weights → routing tokens.

        Returns:
            routing:             [crmsa_k, nW*B, C]   (transposed for InnerAttention)
            dispatch_weights_mm: [nW*B, crmsa_k, rs²]
            dispatch_weights:    [nW*B, crmsa_k, rs²]
            region_size, H, W, add_length
        """
        B, L, C = x.shape
        x, H, W, add_length, region_num, region_size = self._pad(x)
        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)                 # nW*B, rs, rs, C
        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, rs², C

        # CR-MSA combine: patch → routing-token logits
        if self.crmsa_mlp:
            logits = self.phi(x_regions).transpose(1, 2)             # nW*B, crmsa_k, rs²
        else:
            logits = torch.einsum("w p c, c n -> w p n", x_regions, self.phi).transpose(1, 2)

        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)

        logits_min, _ = logits.min(dim=-1)
        logits_max, _ = logits.max(dim=-1)
        dispatch_weights_mm = (logits - logits_min.unsqueeze(-1)) / \
            (logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)

        routing = torch.einsum(
            "w p c, w n p -> w n p c", x_regions, combine_weights).sum(dim=-2).transpose(0, 1)
        # crmsa_k, nW*B, C

        return routing, dispatch_weights_mm, dispatch_weights, region_size, H, W, add_length

    # ------------------------------------------------------------------
    # Dispatch — verbatim from official CrossRegionAttention (second half)
    # ------------------------------------------------------------------
    def _dispatch(self, routing, dispatch_weights_mm, dispatch_weights,
                  region_size, H, W, add_length):
        """routing tokens → patch features (official region_reverse, no broadcast)."""
        attn_regions = routing.transpose(0, 1)                       # nW*B, crmsa_k, C

        attn_regions = torch.einsum(
            "w n c, w n p -> w n p c", attn_regions, dispatch_weights_mm)
        # nW*B, crmsa_k, rs², C
        attn_regions = torch.einsum(
            "w n p c, w n p -> w n p c", attn_regions, dispatch_weights).sum(dim=1)
        # nW*B, rs², C

        C = attn_regions.shape[-1]
        attn_regions = attn_regions.view(-1, region_size, region_size, C)
        x = region_reverse(attn_regions, region_size, H, W)          # B, H, W, C
        x = x.view(x.shape[0], H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, z_list):
        """Cross-staining CR-MSA TransLayer.

        Args:
            z_list: [z_he, z_pr], each [B, N, D] (Stage-1 RRTEncoder outputs)

        Returns:
            [B, N_he + N_pr, D]: fused patch features (HE then PR concatenated)
        """
        z_he, z_pr = z_list

        # Pre-norm (official TransLayer)
        z_he_n = self.norm(z_he)
        z_pr_n = self.norm(z_pr)

        # Combine: pad + partition + routing tokens for each staining
        routing_he, dmm_he, dw_he, rs_he, H_he, W_he, add_he = self._combine(z_he_n)
        routing_pr, dmm_pr, dw_pr, rs_pr, H_pr, W_pr, add_pr = self._combine(z_pr_n)

        # Unified region set: concat routing tokens along the region axis
        routing = torch.cat([routing_he, routing_pr], dim=1)          # crmsa_k, 2·nW·B, C

        # Cross-staining region attention (official InnerAttention)
        routing = self.attn(routing)

        # Split back into HE / PR routing tokens
        n_he = routing_he.shape[1]
        routing_he = routing[:, :n_he]
        routing_pr = routing[:, n_he:]

        # Dispatch reconstruction (official, no broadcast)
        out_he = self._dispatch(routing_he, dmm_he, dw_he, rs_he, H_he, W_he, add_he)
        out_pr = self._dispatch(routing_pr, dmm_pr, dw_pr, rs_pr, H_pr, W_pr, add_pr)

        # Residual (official TransLayer): x = x + DropPath(z)
        z_he = z_he + self.drop_path(out_he)
        z_pr = z_pr + self.drop_path(out_pr)

        # Optional FFN
        if self.ffn:
            z_he = z_he + self.drop_path(self.mlp(self.norm2(z_he)))
            z_pr = z_pr + self.drop_path(self.mlp(self.norm2(z_pr)))

        # Final LayerNorm, then concatenate for ABMIL
        z_he = self.out_norm(z_he)
        z_pr = self.out_norm(z_pr)
        return torch.cat([z_he, z_pr], dim=1)
