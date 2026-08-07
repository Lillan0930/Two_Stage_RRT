"""
MR²T v2 — Direct Cross-Modality Region Attention.

Key difference from v1:
- NO learnable lambda gate
- Cross attention output IS the new representation
- F = R_HE + CrossAttn(Q=R_HE, K=R_IHC, V=R_IHC)
- Residual design, identity initialization via zero-init output projection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalityRegionAttentionV2(nn.Module):
    """
    Direct cross-modality region attention. No gate, no lambda.

    HE region tokens: Query
    IHC region tokens: Key / Value

    Output = R_HE + CrossAttn(Q_HE, K_IHC, V_IHC)
    """

    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q from HE
        self.w_q = nn.Linear(dim, dim, bias=False)
        # K, V from IHC
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)

        # Zero-init output projection → identity at initialization
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Region token pooling
        self.region_pool = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.w_q, self.w_k, self.w_v]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        for m in self.region_pool:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def extract_region_tokens(self, z_grid, region_size):
        """Extract region tokens via learned attention pooling."""
        B, H, W, D = z_grid.shape
        nH = H // region_size
        nW = W // region_size
        K = nH * nW

        z_regions = (
            z_grid
            .view(B, nH, region_size, nW, region_size, D)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, K, region_size * region_size, D)
        )

        z_flat = z_regions.view(B * K, region_size * region_size, D)
        attn = self.region_pool(z_flat)
        attn = F.softmax(attn, dim=1)
        tokens = (z_flat * attn).sum(dim=1)
        return tokens.view(B, K, D)

    def forward(self, he_tokens, ihc_tokens):
        """
        R_HE_new = CrossAttn(Q=HE, K=IHC, V=IHC)
        return R_HE + R_HE_new
        """
        B, K, D = he_tokens.shape

        # Multi-head cross attention
        q = self.w_q(he_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(ihc_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(ihc_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, K, D)
        delta = self.out_proj(out)  # zero-init → delta=0 at init

        # Residual: R_HE_new = R_HE + delta
        return he_tokens + delta, attn_weights


def broadcast_region_to_patch(fused_tokens, grid_shape, region_size):
    """Broadcast region tokens back to patch level."""
    B, K, D = fused_tokens.shape
    H, W = grid_shape
    nH = H // region_size
    nW = W // region_size
    tokens_grid = fused_tokens.view(B, nH, nW, D)
    tokens_expanded = (
        tokens_grid
        .unsqueeze(2).unsqueeze(4)
        .expand(B, nH, region_size, nW, region_size, D)
        .contiguous()
        .view(B, H, W, D)
    )
    return tokens_expanded.view(B, H * W, D)
