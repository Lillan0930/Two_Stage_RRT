"""
Cross-Modality Region Re-embedding (Stage 2 of Two-stage R²T).

Input:  Z_HE' [B,N,D] and Z_IHC' [B,N,D] — both already R²T re-embedded
Output: Z_final [B,N,D] — unified patch-level representation for ABMIL

Architecture:
  1. Extract region tokens from each modality → R_HE [B,K,D], R_IHC [B,K,D]
  2. Pre-Norm → Learned-query cross-attention over all 2K region tokens
     → DropPath → Residual with HE → Post-Norm → R_joint [B,K,D]
  3. Broadcast R_joint back to patches → Z_final [B,N,D]

Key: K region tokens output, NOT 2K. The cross-attention produces a SINGLE
     unified region representation incorporating both modalities.

The attention block follows the TransLayer pattern:
    Norm(KV) → CrossAttn(Q_learned, KV) → DropPath → +R_HE(residual) → Norm
"""

import math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath


class CrossRegionReembedding(nn.Module):
    """
    Stage 2: Cross-Modality Region Re-embedding.

    Takes two R²T outputs, extracts their region tokens, runs cross-attention
    between them, and produces a unified region representation.

    The cross-attention block now follows the official TransLayer wrapping pattern:
      Norm(KV) → CrossAttn → DropPath → +R_HE → Norm

    Args:
        dim: feature dimension (512)
        crmsa_k: CR-MSA routing dimension (used for compatibility; actual
                 cross-attn uses crmsa_heads)
        crmsa_heads: number of attention heads
        region_num: number of regions per side
        epeg: kept for API compatibility (not used in learned-query cross-attn)
        epeg_k: kept for API compatibility
        drop_out: attention dropout rate
        drop_path: DropPath rate (stochastic depth)
    """

    def __init__(self, dim=512, crmsa_k=5, crmsa_heads=8,
                 region_num=4, epeg=True, epeg_k=21, drop_out=0.1,
                 drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.region_num = region_num
        self.crmsa_k = crmsa_k

        # ── Region token pooling (shared across modalities) ──
        self.region_pool = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

        # ── Pre-attention norms ──
        self.norm_kv = nn.LayerNorm(dim)   # normalise region tokens before K/V projection

        # ── Cross-attention parameters ──
        self.num_heads = crmsa_heads
        self.head_dim = dim // crmsa_heads
        self.scale = self.head_dim ** -0.5

        # Learned query embeddings — K queries that attend over all 2K region tokens
        self.query_pos = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query_pos, std=0.02)
        self.norm_q = nn.LayerNorm(dim)    # normalise learned query before Q projection

        self.w_q = nn.Linear(dim, dim, bias=False)   # query from learnable embedding
        self.w_k = nn.Linear(dim, dim, bias=False)   # key   from (normed) region tokens
        self.w_v = nn.Linear(dim, dim, bias=False)   # value from (normed) region tokens

        # ── Attention dropout ──
        self.attn_drop = nn.Dropout(drop_out) if drop_out > 0 else nn.Identity()

        # ── Output projection (zero-init → identity at initialization) ──
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # ── DropPath (stochastic depth) on the attention residual ──
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # ── Post-residual LayerNorm ──
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

    # ------------------------------------------------------------------
    # Region token extraction / broadcast  (unchanged logic)
    # ------------------------------------------------------------------

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
        # 与 pad_to_grid 的两段 padding 互逆: 先裁回原方阵 [:gs,:gs], 再裁前 N 个
        gs = int(np.ceil(np.sqrt(N_orig)))
        t = t[:, :gs, :gs, :]
        return t.reshape(B, gs * gs, D)[:, :N_orig, :]

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z_he, z_ihc):
        """
        Args:
            z_he:  [B, N, D]  HE R²T output
            z_ihc: [B, N, D]  IHC R²T output

        Returns:
            z_final: [B, N, D]  unified re-embedded patch features
        """
        B, N_orig, D = z_he.shape

        # ── 1. Pad and extract region tokens ──
        he_grid, H_he, W_he = self.pad_to_grid(z_he)
        ihc_grid, H_ihc, W_ihc = self.pad_to_grid(z_ihc)

        # Both modalities may have different patch counts → unify to max H, W
        H = max(H_he, H_ihc)
        W = max(W_he, W_ihc)

        if H_he < H or W_he < W:
            he_grid = F.pad(he_grid, (0, 0, 0, W - W_he, 0, H - H_he))
        if H_ihc < H or W_ihc < W:
            ihc_grid = F.pad(ihc_grid, (0, 0, 0, W - W_ihc, 0, H - H_ihc))

        region_size = H // self.region_num

        r_he = self.extract_region_tokens(
            he_grid.view(B, H * W, D), H, W, region_size)     # [B, K, D]
        r_ihc = self.extract_region_tokens(
            ihc_grid.view(B, H * W, D), H, W, region_size)    # [B, K, D]

        K = r_he.shape[1]
        all_tokens = torch.cat([r_he, r_ihc], dim=1)           # [B, 2K, D]

        # ── 2. Pre-Norm ──
        kv_in = self.norm_kv(all_tokens)                        # [B, 2K, D]
        q_in = self.norm_q(self.query_pos.expand(B, K, -1))     # [B, K, D]

        # ── 3. Cross-attention: learned Q attends over KV from all region tokens ──
        q = self.w_q(q_in)                                      # [B, K, D]
        k = self.w_k(kv_in)                                     # [B, 2K, D]
        v = self.w_v(kv_in)                                     # [B, 2K, D]

        q = q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)       # [B, H, K, d]
        k = k.view(B, 2 * K, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, 2K, d]
        v = v.view(B, 2 * K, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, 2K, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                            # [B, H, K, d]
        out = out.transpose(1, 2).contiguous().view(B, K, D)      # [B, K, D]

        # ── 4. Output projection + DropPath + Residual + Post-Norm ──
        delta = self.out_proj(out)                                # zero-init → 0 at init
        r_joint = self.norm_out(r_he + self.drop_path(delta))     # [B, K, D]

        # ── 5. Broadcast back to patches ──
        z_final = self.broadcast_tokens(r_joint, H, W, region_size, N_orig)

        return z_final
