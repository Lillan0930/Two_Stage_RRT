"""
Cross-Modality Region Re-embedding (Stage 2 of Two-stage R²T).

Input:  Z_HE' [B,N,D] and Z_IHC' [B,N,D] — both already R²T re-embedded
Output: Z_final [B,N,D] — unified patch-level representation for ABMIL

Architecture:
  1. Extract region tokens from each modality → R_HE [B,K,D], R_IHC [B,K,D]
  2. Cross-attention between region token sets → R_joint [B,K,D]
  3. Broadcast R_joint back to patches → Z_final [B,N,D]

Key: K region tokens output, NOT 2K. The cross-attention produces a SINGLE
     unified region representation incorporating both modalities.
"""

import math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from models.rmsa import CrossRegionAttention, region_partition, region_reverse


class CrossRegionReembedding(nn.Module):
    """
    Stage 2: Cross-Modality Region Re-embedding.

    Takes two R²T outputs, extracts their region tokens, runs cross-attention
    between them, and produces a unified region representation.

    Args:
        dim: feature dimension (512)
        crmsa_k: CR-MSA routing dimension
        crmsa_heads: number of attention heads
        region_num: number of regions per side
        epeg_k: EPEG kernel size
    """

    def __init__(self, dim=512, crmsa_k=5, crmsa_heads=8,
                 region_num=4, epeg=True, epeg_k=21, drop_out=0.1):
        super().__init__()
        self.dim = dim
        self.region_num = region_num
        self.crmsa_k = crmsa_k

        # Region token pooling (shared across modalities)
        self.region_pool = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

        # Cross-attention: HE region tokens attend to a learned query
        # Produces K output region tokens (not 2K)
        self.num_heads = crmsa_heads
        self.head_dim = dim // crmsa_heads
        self.scale = self.head_dim ** -0.5

        # Learned query embeddings — K queries that attend over all 2K region tokens
        self.register_buffer('query_pos', torch.zeros(1, 1, dim))
        self.w_q = nn.Linear(dim, dim, bias=False)   # query from learnable embedding
        self.w_k = nn.Linear(dim, dim, bias=False)   # key from concat region tokens
        self.w_v = nn.Linear(dim, dim, bias=False)   # value from concat region tokens

        # Zero-init output projection → identity at initialization
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Residual connection
        self.norm_out = nn.LayerNorm(dim)

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

    def extract_region_tokens(self, z_grid, H, W, region_size):
        """Extract K region tokens from patch features [B, H*W, D]."""
        B, N, D = z_grid.shape
        x = z_grid.view(B, H, W, D)
        nH = H // region_size
        nW = W // region_size
        K = nH * nW

        regions = (
            x.view(B, nH, region_size, nW, region_size, D)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, K, region_size * region_size, D)
        )
        z_flat = regions.view(B * K, region_size * region_size, D)
        attn = self.region_pool(z_flat)
        attn = F.softmax(attn, dim=1)
        tokens = (z_flat * attn).sum(dim=1)
        return tokens.view(B, K, D)

    def broadcast_tokens(self, tokens, H, W, region_size, N_orig):
        """Broadcast [B, K, D] back to [B, N_orig, D]."""
        B, K, D = tokens.shape
        nH = H // region_size
        nW = W // region_size
        t = tokens.view(B, nH, nW, D)
        t = t.unsqueeze(2).unsqueeze(4)
        t = t.expand(B, nH, region_size, nW, region_size, D)
        t = t.contiguous().view(B, H, W, D)
        return t.view(B, H * W, D)[:, :N_orig, :]

    def pad_to_grid(self, z):
        """Pad [B,N,D] to square grid divisible by region_num."""
        B, N, D = z.shape
        grid_size = int(np.ceil(np.sqrt(N)))
        pad_len = grid_size * grid_size - N
        if pad_len > 0:
            z = F.pad(z, (0, 0, 0, pad_len), value=0)
        H = W = grid_size
        z_grid = z.view(B, H, W, D)
        pad_h = (self.region_num - H % self.region_num) % self.region_num
        pad_w = (self.region_num - W % self.region_num) % self.region_num
        if pad_h > 0 or pad_w > 0:
            z_grid = F.pad(z_grid, (0, 0, 0, pad_w, 0, pad_h), value=0)
            H, W = H + pad_h, W + pad_w
        return z_grid, H, W

    def forward(self, z_he, z_ihc):
        """
        Args:
            z_he:  [B, N, D]  HE R²T output
            z_ihc: [B, N, D]  IHC R²T output

        Returns:
            z_final: [B, N, D]  unified re-embedded patch features
        """
        B, N_orig, D = z_he.shape

        # 1. Pad and extract region tokens from each modality
        he_grid, H, W = self.pad_to_grid(z_he)
        ihc_grid, _, _ = self.pad_to_grid(z_ihc)
        region_size = H // self.region_num

        r_he = self.extract_region_tokens(
            he_grid.view(B, H * W, D), H, W, region_size)    # [B, K, D]
        r_ihc = self.extract_region_tokens(
            ihc_grid.view(B, H * W, D), H, W, region_size)   # [B, K, D]

        # 2. Concat all region tokens for cross-attention key/value pool
        K = r_he.shape[1]  # number of regions
        all_tokens = torch.cat([r_he, r_ihc], dim=1)          # [B, 2K, D]

        # 3. Learned queries attend over all 2K region tokens → K output tokens
        q = self.w_q(self.query_pos.expand(B, K, -1))         # [B, K, D]
        k = self.w_k(all_tokens)                               # [B, 2K, D]
        v = self.w_v(all_tokens)                               # [B, 2K, D]

        q = q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 2 * K, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 2 * K, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, K, D)
        delta = self.out_proj(out)                             # zero-init → residual=0

        # 4. Residual with HE region tokens
        r_joint = self.norm_out(r_he + delta)                  # [B, K, D]

        # 5. Broadcast back to patches
        z_final = self.broadcast_tokens(r_joint, H, W, region_size, N_orig)

        return z_final
