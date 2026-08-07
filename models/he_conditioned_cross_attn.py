"""
HE-Conditioned Cross Region Attention (v3).

HE queries IHC — like a Transformer decoder.
Gate is controlled by HE, deciding per-region whether to accept IHC info.
Output: only updated HE representation (no IHC output).

Key insight:
  g = σ(MLP(R_HE))       ← HE decides if it needs IHC
  ΔR = CrossAttn(Q_HE, K_IHC, V_IHC)
  R_HE' = R_HE + g · ΔR  ← gated residual

Zero-init → identity at initialization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HEConditionedCrossAttention(nn.Module):
    """
    HE-conditioned cross-modality region attention.

    HE region tokens: Query (and gate controller)
    IHC region tokens: Key / Value (memory)

    Output: updated HE region tokens only.
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

        # Zero-init output projection → ΔR=0 at init
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # HE-controlled per-region gate: g = σ(MLP(R_HE))
        self.gate_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
        )

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
        Args:
            he_tokens:  [B, K, D]  HE region tokens
            ihc_tokens: [B, K, D]  IHC region tokens

        Returns:
            updated_he: [B, K, D]  updated HE region tokens
        """
        B, K, D = he_tokens.shape

        # ── Cross attention: HE queries IHC ──
        q = self.w_q(he_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(ihc_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(ihc_tokens).view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, K, D)
        delta = self.out_proj(out)  # zero-init → delta=0 at init

        # ── HE-controlled per-region gate ──
        gate = torch.sigmoid(self.gate_net(he_tokens))  # [B, K, 1]
        # Gate starts ≈ 0.5 at init (before training)

        # ── Gated residual ──
        updated_he = he_tokens + gate * delta

        return updated_he, attn_weights, gate


def broadcast_region_to_patch_v3(tokens, grid_shape, region_size):
    """Broadcast region tokens back to patch level."""
    B, K, D = tokens.shape
    H, W = grid_shape
    nH = H // region_size
    nW = W // region_size
    tokens_grid = tokens.view(B, nH, nW, D)
    expanded = (
        tokens_grid
        .unsqueeze(2).unsqueeze(4)
        .expand(B, nH, region_size, nW, region_size, D)
        .contiguous()
        .view(B, H, W, D)
    )
    return expanded.view(B, H * W, D)
