"""
ModalityAttention: Multi-Modal Fusion Module for MM-RRT-Light

实现MM-RRT-Light论文中的Step 3: Modality Self-Attention
在R-MSA之后、CR-MSA之前进行模态间的交互融合。

输入: X ∈ R^{B × I × N × D}
    B: batch size
    I: number of patches
    N: number of modalities
    D: feature dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ModalityAttention(nn.Module):
    """
    模态自注意力模块 (Modality Self-Attention)
    
    对每个region，在所有模态之间进行注意力计算，实现模态融合。
    复杂度: O(N^2 * D), N是模态数量(通常2-5)，计算开销很小。
    """
    def __init__(self, dim, num_modalities, num_heads=4, qkv_bias=True,
                 attn_drop=0., proj_drop=0., use_mean_pooling=True,
                 fixed_weights=None):
        """
        Args:
            dim: 特征维度
            num_modalities: 模态数量 (2, 3, 4, 5)
            num_heads: 注意力头数
            qkv_bias: 是否使用bias
            attn_drop: attention dropout
            proj_drop: projection dropout
            use_mean_pooling: 是否使用mean pooling聚合多模态特征
            fixed_weights: 固定模态权重，如[1.0, 0.0]表示第一个模态权重为1，第二个为0
        """
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_mean_pooling = use_mean_pooling
        self.fixed_weights = fixed_weights
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Modality importance weights (可学习的模态重要性)
        if fixed_weights is not None:
            # 使用固定的模态权重，不作为可学习参数
            self.register_buffer('modality_weights', torch.tensor(fixed_weights, dtype=torch.float32))
        else:
            self.modality_weights = nn.Parameter(torch.ones(num_modalities))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, region_tokens):
        """
        Args:
            region_tokens: [B, num_regions, N, D]
                B: batch size
                num_regions: region数量
                N: 模态数量
                D: 特征维度
        
        Returns:
            fused_tokens: [B, num_regions, D] 融合后的region表示
            attn_weights: [B, num_regions, N, N] 模态间注意力权重 (用于可视化)
        """
        B, num_regions, N, D = region_tokens.shape
        assert N == self.num_modalities, f"Expected {self.num_modalities} modalities, got {N}"
        
        # 重塑为 [B * num_regions, N, D] 以便进行注意力计算
        x = region_tokens.reshape(B * num_regions, N, D)
        
        # QKV计算
        qkv = self.qkv(x).reshape(B * num_regions, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*R, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B*R, heads, N, N]
        
        # 应用softmax得到模态间注意力权重
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn.clone()  # 保存用于可视化
        attn = self.attn_drop(attn)
        
        # 加权聚合
        x = (attn @ v).transpose(1, 2).reshape(B * num_regions, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 重塑回 [B, num_regions, N, D]
        x = x.reshape(B, num_regions, N, D)
        
        # 使用模态重要性权重进行聚合
        modality_weights = F.softmax(self.modality_weights, dim=0)  # [N]
        
        if self.use_mean_pooling:
            # Mean pooling with learned weights
            fused = (x * modality_weights.view(1, 1, N, 1)).sum(dim=2)  # [B, num_regions, D]
        else:
            # Concatenate all modality features
            fused = x.reshape(B, num_regions, -1)  # [B, num_regions, N*D]
            # 投影回D维度
            fused = F.linear(fused, torch.eye(D, N*D, device=fused.device))
        
        # 重塑注意力权重用于可视化 [B, num_regions, N, N]
        attn_weights = attn_weights.mean(dim=1).reshape(B, num_regions, N, N)
        
        return fused, attn_weights


class ModalityCrossAttention(nn.Module):
    """
    模态交叉注意力 (Modality Cross-Attention)
    
    允许一个主模态(query)与其他辅助模态(keys/values)进行交互。
    适用于有一个主要模态(RAW)和其他辅助模态的情况。
    """
    def __init__(self, dim, num_modalities, query_modality_idx=0, 
                 num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim: 特征维度
            num_modalities: 模态数量
            query_modality_idx: 哪个模态作为query (默认0，即第一个模态)
            num_heads: 注意力头数
        """
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.query_modality_idx = query_modality_idx
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate projections for query (one modality) and key/value (all modalities)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, region_tokens):
        """
        Args:
            region_tokens: [B, num_regions, N, D]
        
        Returns:
            fused_tokens: [B, num_regions, D]
            attn_weights: [B, num_regions, 1, N-1] (不含query modality自身)
        """
        B, num_regions, N, D = region_tokens.shape
        
        # 重塑为 [B * num_regions, N, D]
        x = region_tokens.reshape(B * num_regions, N, D)
        
        # Query: 只使用指定模态
        q = x[:, self.query_modality_idx:self.query_modality_idx+1, :]  # [B*R, 1, D]
        q = self.q_proj(q).reshape(B * num_regions, 1, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B*R, heads, 1, head_dim]
        
        # Key/Value: 使用所有模态
        kv = self.kv_proj(x).reshape(B * num_regions, N, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B*R, heads, N, head_dim]
        k, v = kv[0], kv[1]
        
        # 计算注意力
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B*R, heads, 1, N]
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn.clone()
        attn = self.attn_drop(attn)
        
        # 加权聚合
        x = (attn @ v).transpose(1, 2).reshape(B * num_regions, 1, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 重塑回 [B, num_regions, D]
        x = x.reshape(B, num_regions, D)
        
        # 重塑注意力权重 [B, num_regions, 1, N]
        attn_weights = attn_weights.mean(dim=1).reshape(B, num_regions, 1, N)
        
        return x, attn_weights


class ModalityFusionType:
    """模态融合类型枚举"""
    SELF_ATTENTION = 'self_attention'      # 所有模态互相attention
    CROSS_ATTENTION = 'cross_attention'    # 一个主模态attention其他
    CONCAT_MLP = 'concat_mlp'              # 拼接后MLP
    SUM = 'sum'                            # 简单相加
    WEIGHTED_SUM = 'weighted_sum'          # 加权求和


def create_modality_fusion(fusion_type, dim, num_modalities, **kwargs):
    """
    工厂函数: 创建不同类型的模态融合模块
    
    Args:
        fusion_type: 融合类型 (见 ModalityFusionType)
        dim: 特征维度
        num_modalities: 模态数量
        **kwargs: 额外参数
    
    Returns:
        对应的融合模块
    """
    if fusion_type == ModalityFusionType.SELF_ATTENTION:
        return ModalityAttention(dim, num_modalities, **kwargs)
    elif fusion_type == ModalityFusionType.CROSS_ATTENTION:
        return ModalityCrossAttention(dim, num_modalities, **kwargs)
    elif fusion_type == ModalityFusionType.CONCAT_MLP:
        return ConcatMLPFusion(dim, num_modalities, **kwargs)
    elif fusion_type == ModalityFusionType.WEIGHTED_SUM:
        return WeightedSumFusion(dim, num_modalities, **kwargs)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


class ConcatMLPFusion(nn.Module):
    """拼接 + MLP融合"""
    def __init__(self, dim, num_modalities, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        
        if hidden_dim is None:
            hidden_dim = dim * num_modalities // 2
        
        self.mlp = nn.Sequential(
            nn.Linear(dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, region_tokens):
        """
        Args:
            region_tokens: [B, num_regions, N, D]
        Returns:
            fused_tokens: [B, num_regions, D]
            None: (无注意力权重)
        """
        B, num_regions, N, D = region_tokens.shape
        x = region_tokens.reshape(B, num_regions, N * D)
        fused = self.mlp(x)
        return fused, None


class WeightedSumFusion(nn.Module):
    """可学习的加权求和融合"""
    def __init__(self, dim, num_modalities, temperature=1.0):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.temperature = temperature
        
        # 可学习的模态权重
        self.modality_weights = nn.Parameter(torch.ones(num_modalities))
    
    def forward(self, region_tokens):
        """
        Args:
            region_tokens: [B, num_regions, N, D]
        Returns:
            fused_tokens: [B, num_regions, D]
            weights: [N] 模态权重
        """
        # 计算softmax权重
        weights = F.softmax(self.modality_weights / self.temperature, dim=0)
        
        # 加权求和 [B, num_regions, D]
        fused = (region_tokens * weights.view(1, 1, -1, 1)).sum(dim=2)
        
        return fused, weights


if __name__ == "__main__":
    # 测试代码
    print("Testing ModalityAttention...")
    
    B, num_regions, N, D = 2, 16, 3, 512
    region_tokens = torch.randn(B, num_regions, N, D)
    
    # Test self-attention
    fusion = ModalityAttention(D, N, num_heads=4)
    fused, attn = fusion(region_tokens)
    print(f"Input shape: {region_tokens.shape}")
    print(f"Fused shape: {fused.shape}")
    print(f"Attention shape: {attn.shape}")
    
    # Test cross-attention
    print("\nTesting ModalityCrossAttention...")
    cross_fusion = ModalityCrossAttention(D, N, query_modality_idx=0)
    fused_cross, attn_cross = cross_fusion(region_tokens)
    print(f"Fused shape: {fused_cross.shape}")
    print(f"Attention shape: {attn_cross.shape}")
    
    print("\nAll tests passed!")
