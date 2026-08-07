"""Minimal stub for gated modality fusion."""
import torch.nn as nn


class GatedModalityFusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, modalities=None):
        return x


def create_gated_fusion(*args, **kwargs):
    return GatedModalityFusion(*args, **kwargs)
