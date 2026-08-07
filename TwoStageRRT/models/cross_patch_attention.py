"""
Cross Patch Modality Attention — Patch-aligned Cross Modality Interaction.

HE and IHC come from the same patch → no N×N global attention needed.
Each HE patch only receives complementary info from its corresponding IHC patch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossPatchModalityAttention(nn.Module):
    """
    Patch-aligned Cross Modality Attention.

    Input:
        he:  [B, N, D]
        ihc: [B, N, D]

    Output:
        enhanced_he:  [B, N, D]
        enhanced_ihc: [B, N, D]
    """

    def __init__(self, dim=512, reduction=4, init_alpha=0.0):
        super().__init__()

        # HE <- IHC
        self.ihc_to_he = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )

        # IHC <- HE
        self.he_to_ihc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU()
        )

        # Learnable residual scale
        self.alpha_he = nn.Parameter(torch.tensor(init_alpha))
        self.alpha_ihc = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, he, ihc):
        """Cross modality residual refinement."""

        # IHC provides complementary residual
        delta_he = self.ihc_to_he(ihc)

        # HE updates IHC
        delta_ihc = self.he_to_ihc(he)

        enhanced_he = he + torch.tanh(self.alpha_he) * delta_he
        enhanced_ihc = ihc + torch.tanh(self.alpha_ihc) * delta_ihc

        return enhanced_he, enhanced_ihc


class ConcatPatchFusion(nn.Module):
    """
    Patch-level Concat Fusion: HE ← Linear([HE; IHC]).

    Replaces the residual-gated cross_patch with a direct concat + projection.
    No learnable gate — IHC features are always fully injected into HE.

    Input:
        he:  [B, N, D]
        ihc: [B, N, D]

    Output:
        fused_he: [B, N, D]
    """

    def __init__(self, dim=512):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.proj = nn.Linear(2 * dim, dim)

    def forward(self, he, ihc):
        x = torch.cat([he, ihc], dim=-1)   # [B, N, 2*D]
        x = self.norm(x)
        fused_he = self.proj(x)            # [B, N, D]
        return fused_he
