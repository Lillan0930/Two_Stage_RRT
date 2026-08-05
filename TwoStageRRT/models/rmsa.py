import torch
import torch.nn as nn
import numpy as np
import math


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def region_partition(x, region_size):
    """
    Args:
        x: (B, H, W, C)
        region_size (int): region size
    Returns:
        regions: (num_regions*B, region_size, region_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
    return regions


def region_reverse(regions, region_size, H, W):
    """
    Args:
        regions: (num_regions*B, region_size, region_size, C)
        region_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class InnerAttention(nn.Module):
    """Inner attention with EPEG (Enhanced Positional Encoding)"""
    def __init__(self, dim, head_dim=None, num_heads=8, qkv_bias=True, qk_scale=None, 
                 attn_drop=0., proj_drop=0., epeg=True, epeg_k=15):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # EPEG: Enhanced Positional Encoding
        if epeg:
            padding = epeg_k // 2
            self.pe = nn.Conv2d(num_heads, num_heads, (epeg_k, 1), 
                               padding=(padding, 0), groups=num_heads, bias=True)
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_regions*B, N, C)
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Apply EPEG to attention map
        if self.pe is not None:
            pe = self.pe(attn)
            attn = attn + pe

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads * self.head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RegionAttention(nn.Module):
    """Region-based Multi-head Self-Attention (R-MSA)"""
    def __init__(self, dim, head_dim=None, num_heads=8, region_size=0, qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., region_num=8, epeg=False, 
                 min_region_num=0, min_region_ratio=0., **kwargs):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio
        
        self.attn = InnerAttention(
            dim, head_dim=head_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
            proj_drop=drop, epeg=epeg, epeg_k=kwargs.get('epeg_k', 15))

    def padding(self, x):
        """Pad features to fit region partitioning"""
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

        # If padding is too much, use minimal region size
        if (add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
            
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)
        
        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape
        
        # Padding
        x, H, W, add_length, region_num, region_size = self.padding(x)

        # Reshape to 2D
        x = x.view(B, H, W, C)

        # Partition into regions
        x_regions = region_partition(x, region_size)
        x_regions = x_regions.view(-1, region_size * region_size, C)

        # R-MSA
        attn_regions = self.attn(x_regions)

        # Merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)
        x = region_reverse(attn_regions, region_size, H, W)

        # Reshape back
        x = x.view(B, H * W, C)

        # Remove padding
        if add_length > 0:
            x = x[:, :-add_length]

        return x


class CrossRegionAttention(nn.Module):
    """Cross-Region Multi-head Self-Attention (CR-MSA)"""
    def __init__(self, dim, head_dim=None, num_heads=8, region_size=0, qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., region_num=8, epeg=False,
                 min_region_num=0, min_region_ratio=0., crmsa_k=3, crmsa_mlp=False, **kwargs):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio
        
        self.attn = InnerAttention(
            dim, head_dim=head_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, epeg=epeg, epeg_k=kwargs.get('epeg_k', 15))

        self.crmsa_mlp = crmsa_mlp
        if crmsa_mlp:
            self.phi = nn.Sequential(
                nn.Linear(self.dim, self.dim // 4, bias=False),
                nn.Tanh(),
                nn.Linear(self.dim // 4, crmsa_k, bias=False)
            )
        else:
            self.phi = nn.Parameter(torch.empty((self.dim, crmsa_k)))
            nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def padding(self, x):
        """Same as RegionAttention"""
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

        if (add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
            
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)
        
        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape
        
        # Padding
        x, H, W, add_length, region_num, region_size = self.padding(x)

        # Reshape to 2D
        x = x.view(B, H, W, C)

        # Partition into regions
        x_regions = region_partition(x, region_size)
        x_regions = x_regions.view(-1, region_size * region_size, C)

        # CR-MSA: Cross-Region attention
        if self.crmsa_mlp:
            logits = self.phi(x_regions).transpose(1, 2)
        else:
            logits = torch.einsum("w p c, c n -> w p n", x_regions, self.phi).transpose(1, 2)

        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)

        logits_min, _ = logits.min(dim=-1)
        logits_max, _ = logits.max(dim=-1)
        dispatch_weights_mm = (logits - logits_min.unsqueeze(-1)) / (logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)

        # Combine regions
        attn_regions = torch.einsum("w p c, w n p -> w n p c", x_regions, combine_weights).sum(dim=-2).transpose(0, 1)

        # Apply attention
        attn_regions = self.attn(attn_regions).transpose(0, 1)

        # Dispatch back
        attn_regions = torch.einsum("w n c, w n p -> w n p c", attn_regions, dispatch_weights_mm)
        attn_regions = torch.einsum("w n p c, w n p -> w n p c", attn_regions, dispatch_weights).sum(dim=1)

        # Merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)
        x = region_reverse(attn_regions, region_size, H, W)

        # Reshape back
        x = x.view(B, H * W, C)

        # Remove padding
        if add_length > 0:
            x = x[:, :-add_length]

        return x
