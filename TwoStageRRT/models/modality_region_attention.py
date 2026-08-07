"""
Modality-aware Region R²T — Cross-Modality Region Attention.

HE → R-MSA → HE region tokens ┐
                               ├→ Cross-Modality Region Attention → Region Fusion → CR-MSA
IHC → R-MSA → IHC region tokens┘

Key principle: interaction happens at the region semantic level, NOT patch level.
HE region tokens act as Query; IHC region tokens act as Key/Value.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalityRegionAttention(nn.Module):
    """
    Cross-Modality Attention at the region representation level.

    HE region tokens: Query
    IHC region tokens: Key / Value

    Args:
        dim: feature dimension (e.g., 512)
        num_heads: attention heads (default 8)
        init_lambda: initial fusion weight (default 0.0 → HE-only at init)
    """

    def __init__(self, dim=512, num_heads=8, init_lambda=0.0,
                 fixed_lambda=None, no_cross_attention=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.no_cross_attention = no_cross_attention

        # Q from HE, K/V from IHC (only used if no_cross_attention=False)
        if not no_cross_attention:
            self.w_q = nn.Linear(dim, dim, bias=False)
            self.w_k = nn.Linear(dim, dim, bias=False)
            self.w_v = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim)

        # Region token pooling: learned attention over patches within each region
        self.region_pool = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

        # Fusion lambda: HE-anchor + lambda * IHC
        self.fixed_lambda = fixed_lambda
        if fixed_lambda is not None:
            self.register_buffer('lambda_fusion', torch.tensor(fixed_lambda))
        else:
            self.lambda_fusion = nn.Parameter(torch.tensor(init_lambda))

        self._init_weights()

    def _init_weights(self):
        for m in [self.w_q, self.w_k, self.w_v, self.out_proj]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        for m in self.region_pool:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def extract_region_tokens(self, z_grid, region_size):
        """
        Extract one region token per region via learned attention pooling.

        Args:
            z_grid: [B, H, W, D]  (square grid, padded)
            region_size: int

        Returns:
            region_tokens: [B, K, D]  where K = (H/region_size) * (W/region_size)
        """
        B, H, W, D = z_grid.shape
        nH = H // region_size
        nW = W // region_size
        K = nH * nW

        # [B, K, region_size*region_size, D]
        z_regions = (
            z_grid
            .view(B, nH, region_size, nW, region_size, D)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, K, region_size * region_size, D)
        )

        # Pool: learned attention over patches in region
        z_flat = z_regions.view(B * K, region_size * region_size, D)        # [B*K, P, D]
        attn = self.region_pool(z_flat)                                      # [B*K, P, 1]
        attn = F.softmax(attn, dim=1)                                        # [B*K, P, 1]
        tokens = (z_flat * attn).sum(dim=1)                                  # [B*K, D]
        tokens = tokens.view(B, K, D)                                        # [B, K, D]

        return tokens

    def _multi_head_attention(self, q, k, v):
        """Standard multi-head scaled dot-product attention."""
        B, K, D = q.shape
        q = self.w_q(q).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(k).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(v).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale        # [B, H, K, K]
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = attn_weights @ v                                       # [B, H, K, hd]
        out = out.transpose(1, 2).contiguous().view(B, K, D)        # [B, K, D]
        out = self.out_proj(out)

        return out, attn_weights

    def forward(self, he_tokens, ihc_tokens, return_diagnostics=False):
        """
        Args:
            he_tokens:  [B, K, D]  HE region tokens
            ihc_tokens: [B, K, D]  IHC region tokens

        Returns:
            fused_tokens: [B, K, D]  fused region tokens for CR-MSA
            diagnostics: dict (optional)
        """
        lam = self.lambda_fusion if self.fixed_lambda is not None else torch.tanh(self.lambda_fusion)

        if self.no_cross_attention:
            # Simple region-level fusion: F = HE + lambda * IHC (no cross attention)
            fused_tokens = he_tokens + lam * ihc_tokens
            attn_weights = None
            enhanced_he = he_tokens
            enhanced_ihc = ihc_tokens
        else:
            # HE query IHC: HE region attn over IHC regions
            enhanced_he, attn_weights = self._multi_head_attention(
                he_tokens, ihc_tokens, ihc_tokens
            )

            # IHC query HE: IHC region attn over HE regions (reverse)
            enhanced_ihc, _ = self._multi_head_attention(
                ihc_tokens, he_tokens, he_tokens
            )

            # HE-anchor fusion with lambda
            fused_tokens = enhanced_he + lam * enhanced_ihc

        if return_diagnostics:
            diag = {
                'lambda': lam.item() if not isinstance(lam, float) else lam,
                'he_token_norm': he_tokens.norm(dim=-1).mean().item(),
                'ihc_token_norm': ihc_tokens.norm(dim=-1).mean().item(),
                'fused_token_norm': fused_tokens.norm(dim=-1).mean().item(),
            }
            if attn_weights is not None:
                diag['cross_attn_mean'] = attn_weights.mean().item()
            return fused_tokens, diag

        return fused_tokens


def broadcast_region_to_patch(fused_tokens, grid_shape, region_size):
    """
    Broadcast fused region tokens back to patch level.

    Each patch in a region receives the same fused region token.

    Args:
        fused_tokens: [B, K, D]  fused region tokens
        grid_shape: (H, W) of the padded grid
        region_size: int

    Returns:
        patch_features: [B, H*W, D]
    """
    B, K, D = fused_tokens.shape
    H, W = grid_shape
    nH = H // region_size
    nW = W // region_size

    # [B, K, D] → [B, nH, nW, D]
    tokens_grid = fused_tokens.view(B, nH, nW, D)

    # Expand each region token to region_size × region_size patches
    # [B, nH, nW, D] → [B, nH, 1, nW, 1, D] → [B, nH, region_size, nW, region_size, D]
    tokens_expanded = (
        tokens_grid
        .unsqueeze(2).unsqueeze(4)  # [B, nH, 1, nW, 1, D]
        .expand(B, nH, region_size, nW, region_size, D)
        .contiguous()
        .view(B, H, W, D)
    )

    return tokens_expanded.view(B, H * W, D)
