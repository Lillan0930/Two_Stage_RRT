"""
Joint Region-Modality Attention for R²T.

Extends R²T from single-modality region re-embedding to multi-modality
joint region-modality re-embedding.

Flow:
  [B, M, N, D] → per-modality R-MSA → region tokens [B, M, R, D]
  → Add modality embedding → flatten [B, M*R, D]
  → Joint CR-MSA → mean-pool M → [B, R, D]
  → Broadcast to patches → [B, N, D]
"""

import math, torch, torch.nn as nn, torch.nn.functional as F
from models.rmsa import CrossRegionAttention, region_partition, region_reverse
import numpy as np


class JointRegionModalityProcessor(nn.Module):
    """
    Joint Region-Modality CR-MSA with learnable modality embeddings.

    Args:
        dim: feature dimension (512)
        num_modalities: M (2 for HE+IHC)
        crmsa_k: CR-MSA routing dimension
        crmsa_heads: number of attention heads for CR-MSA
        region_num: number of regions per side
        epeg: use EPEG
        epeg_k: EPEG kernel size
    """

    def __init__(self, dim=512, num_modalities=2,
                 crmsa_k=5, crmsa_heads=8,
                 region_num=4, epeg=True, epeg_k=21,
                 drop_out=0.1, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.region_num = region_num

        # Learnable modality embedding
        self.mod_embed = nn.Parameter(torch.zeros(1, num_modalities, 1, dim))
        nn.init.trunc_normal_(self.mod_embed, std=0.02)

        # Region token pooling (shared across modalities)
        self.region_pool = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

        # Joint CR-MSA — treats all M*R tokens as one set
        self.cr_msa = CrossRegionAttention(
            dim=dim, num_heads=crmsa_heads, drop=drop_out,
            region_num=region_num, head_dim=dim // crmsa_heads,
            epeg=epeg, epeg_k=epeg_k,
            crmsa_k=crmsa_k, crmsa_mlp=False, **kwargs
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def extract_region_tokens(self, x, H, W, region_size):
        """x: [B, N, D] → region tokens [B, K, D]"""
        B, N, D = x.shape
        x_grid = x.view(B, H, W, D)
        nH, nW = H // region_size, W // region_size
        K = nH * nW
        regions = x_grid.view(B, nH, region_size, nW, region_size, D)
        regions = regions.permute(0, 1, 3, 2, 4, 5).contiguous()
        regions = regions.view(B, K, region_size * region_size, D)
        z_flat = regions.view(B * K, region_size * region_size, D)
        attn = self.region_pool(z_flat)
        attn = F.softmax(attn, dim=1)
        tokens = (z_flat * attn).sum(dim=1)
        return tokens.view(B, K, D)

    def broadcast_tokens(self, tokens, H, W, region_size, N_orig):
        """tokens: [B, K, D] → broadcast to patches [B, N_orig, D]"""
        B, K, D = tokens.shape
        nH, nW = H // region_size, W // region_size
        t = tokens.view(B, nH, nW, D)
        t = t.unsqueeze(2).unsqueeze(4)
        t = t.expand(B, nH, region_size, nW, region_size, D)
        t = t.contiguous().view(B, H, W, D)
        return t.view(B, H * W, D)[:, :N_orig, :]

    def pad_to_grid(self, x_list):
        """Pad each modality to square grid divisible by region_num."""
        B, N, D = x_list[0].shape
        grid_size = int(np.ceil(np.sqrt(N)))
        pad_len = grid_size * grid_size - N

        padded = []
        H = W = grid_size
        for x in x_list:
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len), value=0)
            x = x.view(B, grid_size, grid_size, D)
            # Pad to be divisible by region_num
            pad_h = (self.region_num - grid_size % self.region_num) % self.region_num
            pad_w = (self.region_num - grid_size % self.region_num) % self.region_num
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h), value=0)
                H = grid_size + pad_h
                W = grid_size + pad_w
            padded.append(x)
        return padded, N, H, W

    def forward(self, x_list):
        """
        Args:
            x_list: list of [B, N, D] (one per modality, HE first)

        Returns:
            fused: [B, N, D] unified HE representation for ABMIL
        """
        M = len(x_list)
        B, N_orig, D = x_list[0].shape

        # 1. Pad to square grid
        padded, N_pad, H, W = self.pad_to_grid(x_list)
        region_size = H // self.region_num

        # 2. Extract region tokens per modality → [B, M, K, D]
        all_tokens = []
        for x in padded:
            tok = self.extract_region_tokens(x.view(B, H * W, D), H, W, region_size)
            all_tokens.append(tok)
        tokens = torch.stack(all_tokens, dim=1)  # [B, M, K, D]

        # 3. Add modality embedding
        tokens = tokens + self.mod_embed[:, :M, :, :]

        # 4. Flatten to [B, M*K, D]
        B, M_actual, K, D4 = tokens.shape
        tokens_flat = tokens.view(B, M_actual * K, D4)

        # 5. Joint CR-MSA
        tokens_out = self.cr_msa(tokens_flat)  # [B, M*K, D]

        # 6. Reshape back, mean-pool M → [B, K, D]
        tokens_out = tokens_out.view(B, M_actual, K, D4)
        unified = tokens_out.mean(dim=1)  # [B, K, D]

        # 7. Broadcast to patches → [B, N_orig, D]
        patches = self.broadcast_tokens(unified, H, W, region_size, N_orig)

        return patches
