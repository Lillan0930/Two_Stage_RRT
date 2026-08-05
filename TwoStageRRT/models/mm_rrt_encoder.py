"""
MM-RRT-Light Encoder: Multi-Modal Region Relation Transformer

基于MM-RRT-Light论文实现的多模态RRT编码器。

核心流程:
1. Intra-Modality Spatial Modeling (R-MSA): 每个模态独立进行区域自注意力
2. Region Token Construction: 构建每个模态的region tokens
3. Modality Self-Attention: 模态间交互融合 (核心创新)
4. Cross-Region Modeling (CR-MSA): 跨region注意力
5. Region-to-Patch Broadcast: 将融合信息广播回patch

输入格式:
    - 单模态: [B, N, D] -> 保持与原RRT兼容
    - 多模态: list of [B, N, D] with length M (模态数)
    - 或 [B, M, N, D] (已堆叠的多模态特征)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional
import sys
import os

# 添加父目录到路径以导入rmsa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.rmsa import RegionAttention, CrossRegionAttention, Mlp, region_partition, region_reverse
from models.modality_attention import ModalityAttention, ModalityFusionType, create_modality_fusion
from models.gated_modality_fusion import GatedModalityFusion, create_gated_fusion
from models.cross_patch_attention import CrossPatchModalityAttention, ConcatPatchFusion
from models.modality_region_attention import CrossModalityRegionAttention, broadcast_region_to_patch
from models.modality_region_attention_v2 import CrossModalityRegionAttentionV2
from models.he_conditioned_cross_attn import HEConditionedCrossAttention, broadcast_region_to_patch_v3
# joint_region_modality not needed for TwoStageRRT


def initialize_weights(module):
    """Initialize weights for the module"""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class ModalityTransLayer(nn.Module):
    """
    多模态Transformer Layer
    
    每个模态独立进行R-MSA，然后可选地进行模态融合
    """
    def __init__(self, dim, num_modalities, head=8, drop_out=0.1,
                 drop_path=0., ffn=False, ffn_act='gelu', mlp_ratio=4.,
                 region_num=8, epeg=True, region_size=0,
                 min_region_num=0, min_region_ratio=0,
                 qkv_bias=True, epeg_k=15, enable_modality_fusion=False,
                 fusion_type='self_attention', fusion_kwargs=None,
                 use_gated_fusion=False, **kwargs):
        super().__init__()
        
        self.num_modalities = num_modalities
        self.dim = dim
        self.region_num = region_num
        self.enable_modality_fusion = enable_modality_fusion and num_modalities > 1
        self._fusion_type = fusion_type  # saved for forward() dispatch

        # 可学习的融合权重（每个模态独立，sigmoid 约束在 0~1）
        # 初始化接近 0.1，让模型自己学习最优权重
        self.fusion_weight = nn.Parameter(torch.ones(num_modalities) * (-2.2))  # sigmoid(-2.2) ≈ 0.1

        # ── Cross Patch Modality Attention ──
        self.enable_cross_patch = (
            fusion_type == 'cross_patch_attention'
            and num_modalities > 1
        )
        if self.enable_cross_patch:
            self.cross_patch_attention = CrossPatchModalityAttention(dim=dim)
        else:
            self.cross_patch_attention = None

        # ── Concat Patch Fusion ──
        self.enable_concat_fusion = (
            fusion_type == 'concat_fusion'
            and num_modalities > 1
        )
        if self.enable_concat_fusion:
            self.concat_fusion = ConcatPatchFusion(dim=dim)
        else:
            self.concat_fusion = None

        # ── Modality-aware Region R²T ──
        self.enable_modality_region = (
            fusion_type == 'modality_region_attention'
            and num_modalities > 1
        )
        if self.enable_modality_region:
            fixed_lam = kwargs.get('modality_region_fixed_lambda', None)
            no_cm = kwargs.get('modality_region_no_cm', False)
            self.modality_region_attn = CrossModalityRegionAttention(
                dim=dim, num_heads=head, init_lambda=0.0,
                fixed_lambda=fixed_lam, no_cross_attention=no_cm,
            )
        else:
            self.modality_region_attn = None

        # ── MR²T v2: Direct Cross-Modality Region Attention (no gate) ──
        self.enable_modality_region_v2 = (
            fusion_type == 'modality_region_v2'
            and num_modalities > 1
        )
        if self.enable_modality_region_v2:
            self.modality_region_attn_v2 = CrossModalityRegionAttentionV2(
                dim=dim, num_heads=head,
            )
        else:
            self.modality_region_attn_v2 = None

        # ── HE-Conditioned Cross Attention (v3) ──
        self.enable_he_conditioned = (
            fusion_type == 'he_conditioned_cross_attn'
            and num_modalities > 1
        )
        if self.enable_he_conditioned:
            self.he_conditioned_attn = HEConditionedCrossAttention(
                dim=dim, num_heads=head,
            )
        else:
            self.he_conditioned_attn = None

        # ── Joint Region-Modality ──
        self.enable_joint_region = (
            fusion_type == 'joint_region_modality'
            and num_modalities > 1
        )
        if self.enable_joint_region:
            crmsa_k = kwargs.get('crmsa_k', 5)
            epeg_k = kwargs.get('epeg_k', 21)
            self.joint_region_processor = JointRegionModalityProcessor(
                dim=dim, num_modalities=num_modalities,
                crmsa_k=crmsa_k, crmsa_heads=head,
                region_num=region_num, epeg=True, epeg_k=epeg_k,
                drop_out=drop_out if isinstance(drop_out, float) else 0.1,
            )
        else:
            self.joint_region_processor = None

        # Region token 聚合：可学习注意力（类似 CR-MSA 的 combine stage）
        # 每个模态独立学习如何聚合 region 内的 patches
        if self.enable_modality_fusion:
            self.region_pool_attn = nn.Sequential(
                nn.Linear(dim, max(dim // 4, 32)),
                nn.Tanh(),
                nn.Linear(max(dim // 4, 32), 1)
            )
        else:
            self.region_pool_attn = None
        
        # 每个模态独立的R-MSA
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_modalities)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(dim) if ffn else nn.Identity() 
                                      for _ in range(num_modalities)])
        
        self.attns = nn.ModuleList([
            RegionAttention(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=region_num,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                **kwargs
            ) for _ in range(num_modalities)
        ])
        
        self.drop_path = nn.Identity()
        
        # FFN
        self.ffn = ffn
        if ffn:
            act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlps = nn.ModuleList([
                Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                    act_layer=act_layer, drop=drop_out)
                for _ in range(num_modalities)
            ])
        else:
            self.mlps = nn.ModuleList([nn.Identity() for _ in range(num_modalities)])
        
        # 模态融合模块 (在R-MSA之后)
        if self.enable_modality_fusion:
            fusion_kwargs = fusion_kwargs or {}
            
            # 检查是否使用新的Gated Fusion
            if use_gated_fusion or fusion_type.startswith('gated'):
                self.use_gated_fusion = True
                self.modality_fusion = create_gated_fusion(
                    fusion_type=fusion_type if fusion_type.startswith('gated') else 'gated_full',
                    dim=dim,
                    num_modalities=num_modalities,
                    **fusion_kwargs
                )
            elif fusion_type == 'clean_anchor':
                from clean_fusion import CleanAnchorFusion
                self.use_gated_fusion = False
                self.modality_fusion = CleanAnchorFusion(
                    dim=dim,
                    num_modalities=num_modalities,
                    **fusion_kwargs
                )
            else:
                self.use_gated_fusion = False
                self.modality_fusion = create_modality_fusion(
                    fusion_type=fusion_type,
                    dim=dim,
                    num_modalities=num_modalities,
                    **fusion_kwargs
                )
            
            # 融合后的特征转换
            self.fusion_norm = nn.LayerNorm(dim)
            self.fusion_proj = nn.Linear(dim, dim)
        else:
            self.modality_fusion = None
            self.use_gated_fusion = False
    
    def forward(self, x_list, return_attn=False):
        """
        Args:
            x_list: list of [B, N, D] with length num_modalities
        Returns:
            output_list: list of [B, N, D] 
            fusion_info: dict with fusion information
        """
        # 允许少模态（logit fusion 模式只传 HE）
        if len(x_list) != self.num_modalities:
            pass  # 不做断言, 静默处理
        
        # Step 1: R-MSA for each modality independently
        z_list = []
        for i, (x, norm, attn, norm2, mlp) in enumerate(
            zip(x_list, self.norms, self.attns, self.norms2, self.mlps)):
            # R-MSA
            z = attn(norm(x))
            x = x + self.drop_path(z)
            
            # FFN
            if self.ffn:
                x = x + self.drop_path(mlp(norm2(x)))
            
            z_list.append(x)
        
        fusion_info = {}

        # ══════════════════════════════════════════════════════════
        # Step 2a: Patch-level Cross Modality Attention
        # ══════════════════════════════════════════════════════════
        if self.enable_cross_patch:
            he = z_list[0]
            ihc = z_list[1]
            he, ihc = self.cross_patch_attention(he, ihc)
            z_list[0] = he
            z_list[1] = ihc
            fusion_info['cross_patch'] = True

        # ══════════════════════════════════════════════════════════
        # Step 2a': Patch-level Concat Fusion
        # ══════════════════════════════════════════════════════════
        if self.enable_concat_fusion:
            he = z_list[0]
            ihc = z_list[1]
            fused_he = self.concat_fusion(he, ihc)
            z_list[0] = fused_he
            z_list[1] = fused_he
            fusion_info['concat_fusion'] = True

        # ══════════════════════════════════════════════════════════
        # Step 2a'': Modality-aware Region R²T
        # ══════════════════════════════════════════════════════════
        if self.enable_modality_region and self.modality_region_attn is not None:
            mod_attn = self.modality_region_attn

            # Pad HE and IHC to square grids (same padding for both)
            B, N, D = z_list[0].shape
            grid_size = int(np.ceil(np.sqrt(N)))
            pad_len = grid_size * grid_size - N

            def pad_to_grid(z):
                if pad_len > 0:
                    return F.pad(z, (0, 0, 0, pad_len), mode='constant', value=0)
                return z

            he_pad = pad_to_grid(z_list[0]).view(B, grid_size, grid_size, D)
            ihc_pad = pad_to_grid(z_list[1]).view(B, grid_size, grid_size, D)

            # Use same region_size as the encoder's region_num
            # Need to pad grid to be divisible by region_num
            region_num = self.region_num
            pad_h = (region_num - grid_size % region_num) % region_num
            pad_w = (region_num - grid_size % region_num) % region_num
            if pad_h > 0 or pad_w > 0:
                he_pad = F.pad(he_pad, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
                ihc_pad = F.pad(ihc_pad, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
                H, W = grid_size + pad_h, grid_size + pad_w
            else:
                H, W = grid_size, grid_size

            region_size = H // region_num

            # Extract region tokens from HE and IHC
            he_tokens = mod_attn.extract_region_tokens(he_pad, region_size)     # [B, K, D]
            ihc_tokens = mod_attn.extract_region_tokens(ihc_pad, region_size)   # [B, K, D]

            # Cross-modality region attention + fusion
            fused_tokens, diagnostics = mod_attn(he_tokens, ihc_tokens, return_diagnostics=True)

            # Broadcast back to patch level
            fused_patches = broadcast_region_to_patch(
                fused_tokens, (H, W), region_size
            )
            # Remove padding to get back to original N patches
            fused_patches = fused_patches[:, :N, :]

            # Replace HE with fused representation
            z_list[0] = fused_patches
            z_list[1] = fused_patches  # IHC also uses fused for CR-MSA compat

            fusion_info['modality_region'] = True
            fusion_info['diagnostics'] = diagnostics

        # ══════════════════════════════════════════════════════════
        # Step 2a''': MR²T v2 — Direct Cross Attention (no gate)
        # ══════════════════════════════════════════════════════════
        if self.enable_modality_region_v2 and self.modality_region_attn_v2 is not None:
            mod_attn = self.modality_region_attn_v2

            B, N, D = z_list[0].shape
            grid_size = int(np.ceil(np.sqrt(N)))
            pad_len = grid_size * grid_size - N

            def pad_to_grid(z):
                if pad_len > 0:
                    return F.pad(z, (0, 0, 0, pad_len), mode='constant', value=0)
                return z

            he_pad = pad_to_grid(z_list[0]).view(B, grid_size, grid_size, D)
            ihc_pad = pad_to_grid(z_list[1]).view(B, grid_size, grid_size, D)

            region_num = self.region_num
            pad_h = (region_num - grid_size % region_num) % region_num
            pad_w = (region_num - grid_size % region_num) % region_num
            if pad_h > 0 or pad_w > 0:
                he_pad = F.pad(he_pad, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
                ihc_pad = F.pad(ihc_pad, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
                H, W = grid_size + pad_h, grid_size + pad_w
            else:
                H, W = grid_size, grid_size

            region_size = H // region_num

            he_tokens = mod_attn.extract_region_tokens(he_pad, region_size)
            ihc_tokens = mod_attn.extract_region_tokens(ihc_pad, region_size)

            # Direct cross attention: R_new = R_HE + CrossAttn(Q_HE, K_IHC, V_IHC)
            updated_tokens, _ = mod_attn(he_tokens, ihc_tokens)

            fused_patches = broadcast_region_to_patch(updated_tokens, (H, W), region_size)
            fused_patches = fused_patches[:, :N, :]

            z_list[0] = fused_patches
            z_list[1] = fused_patches
            fusion_info['modality_region_v2'] = True

        # ══════════════════════════════════════════════════════════
        # Step 2a'''': HE-Conditioned Cross Attention (v3)
        # ══════════════════════════════════════════════════════════
        if self.enable_he_conditioned and self.he_conditioned_attn is not None:
            mod_attn = self.he_conditioned_attn

            B, N, D = z_list[0].shape
            grid_size = int(np.ceil(np.sqrt(N)))
            pad_len = grid_size * grid_size - N

            def pad_to_grid(z):
                if pad_len > 0:
                    return F.pad(z, (0, 0, 0, pad_len), mode='constant', value=0)
                return z

            he_pad = pad_to_grid(z_list[0]).view(B, grid_size, grid_size, D)
            ihc_pad = pad_to_grid(z_list[1]).view(B, grid_size, grid_size, D)

            pad_h = (self.region_num - grid_size % self.region_num) % self.region_num
            pad_w = (self.region_num - grid_size % self.region_num) % self.region_num
            if pad_h > 0 or pad_w > 0:
                he_pad = F.pad(he_pad, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
                ihc_pad = F.pad(ihc_pad, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
                H, W = grid_size + pad_h, grid_size + pad_w
            else:
                H, W = grid_size, grid_size

            region_size = H // self.region_num
            he_tokens = mod_attn.extract_region_tokens(he_pad, region_size)
            ihc_tokens = mod_attn.extract_region_tokens(ihc_pad, region_size)

            # HE-conditioned cross attention: HE queries IHC, HE controls gate
            updated_tokens, _, gate = mod_attn(he_tokens, ihc_tokens)

            fused_patches = broadcast_region_to_patch_v3(updated_tokens, (H, W), region_size)
            fused_patches = fused_patches[:, :N, :]

            z_list[0] = fused_patches
            z_list[1] = fused_patches
            fusion_info['he_conditioned'] = True
            fusion_info['gate_mean'] = gate.mean().item()

        # Step 2b: Region-level Modality Fusion (if enabled)
        # 采用原始RRT论文的region划分方式：将网格划分为region_num×region_num个区域，
        # 每个区域内的patch做自注意力（跨模态），而非全局平均池化。
        # NOTE: cross_patch 和 region fusion 互斥，cross_patch 优先
        skip_region_fusion = any([
            self.enable_cross_patch, self.enable_concat_fusion,
            self.enable_modality_region, self.enable_modality_region_v2,
            self.enable_he_conditioned, self.enable_joint_region,
        ])
        if self.enable_modality_fusion and self.modality_fusion is not None and not skip_region_fusion:
            try:
                z_stacked = torch.stack(z_list, dim=1)  # [B, M, N, D]
                B, M, N, D = z_stacked.shape

                # ── 1. 将 N 个 patch 排列为方形网格 ──
                grid_size = int(np.ceil(np.sqrt(N)))
                if grid_size * grid_size > N:
                    padding = grid_size * grid_size - N
                    z_grid = F.pad(
                        z_stacked.reshape(B, M, N, D),
                        (0, 0, 0, padding), mode='constant', value=0
                    )
                else:
                    z_grid = z_stacked.reshape(B, M, N, D)
                z_grid = z_grid.reshape(B, M, grid_size, grid_size, D)  # [B, M, H, W, D]

                # ── 2. 合并 B×M，用 region_partition 划分区域 ──
                z_merged = z_grid.reshape(B * M, grid_size, grid_size, D)  # [B*M, H, W, D]

                # 补齐网格使其能被 region_num 整除
                pad_h = (self.region_num - grid_size % self.region_num) % self.region_num
                pad_w = (self.region_num - grid_size % self.region_num) % self.region_num
                if pad_h > 0 or pad_w > 0:
                    z_merged = F.pad(z_merged, (0, 0, 0, pad_w, 0, pad_h),
                                     mode='constant', value=0)
                    H, W = grid_size + pad_h, grid_size + pad_w
                else:
                    H, W = grid_size, grid_size

                # 每个区域的边长（patch数）
                region_side = H // self.region_num  # patches per region per dim

                # region_partition: [B*M, nW, region_side*region_side, D]
                regions = region_partition(z_merged, region_side)
                # region_partition 返回 [B*M*nW, region_side, region_side, D]
                # 其中 nW = (H//region_side) * (W//region_side)
                nW = (H // region_side) * (W // region_side)  # = region_num^2
                regions = regions.reshape(B, M, nW, region_side * region_side, D)

                # ── 3. 每个区域内，用可学习注意力池化为 region token ──
                # (类 CR-MSA combine stage: learned weights, 而非 mean pooling)
                regions_pool = regions.reshape(B * M * nW, region_side * region_side, D)
                attn_scores = self.region_pool_attn(regions_pool)  # [B*M*nW, R*R, 1]
                attn_weights = F.softmax(attn_scores, dim=1)       # [B*M*nW, R*R, 1]
                region_tokens = (regions_pool * attn_weights).sum(dim=1)  # [B*M*nW, D]
                region_tokens = region_tokens.reshape(B, M, nW, D)  # [B, M, nW, D]

                # 转置为模态融合模块需要的格式: [B, nW, M, D]
                region_tokens = region_tokens.permute(0, 2, 1, 3)  # [B, nW, M, D]

                # ── 4. 模态融合（在 M 维度上做 attention）──
                if self.use_gated_fusion:
                    fused_regions, fusion_details = self.modality_fusion(
                        region_tokens, return_details=True
                    )
                    fusion_info.update(fusion_details)
                    fusion_stats = self.modality_fusion.get_fusion_stats()
                    fusion_info['fusion_stats'] = fusion_stats
                elif hasattr(self, '_fusion_type') and isinstance(self._fusion_type, str) and self._fusion_type == 'clean_anchor':
                    fused_regions = self.modality_fusion(region_tokens)
                else:
                    fused_regions, attn_weights = self.modality_fusion(region_tokens)
                    fusion_info['attn_weights'] = attn_weights

                fusion_info['region_tokens'] = region_tokens
                # fused_regions: [B, nW, D]

                # ── 5. 广播回 patch 级别，region 感知 ──
                # 用 region_reverse 将每个 region 的融合特征回传到对应 patch
                # (类似原始 RRT 的 CR-MSA dispatch，而非简单跨 region 平均)
                fusion_weights = torch.sigmoid(self.fusion_weight)  # [M], bounded in (0, 1)

                # fused_regions: [B, nW, D] → 展开为每个 region 所有 patch
                fused_expand = fused_regions.unsqueeze(2)  # [B, nW, 1, D]
                fused_expand = fused_expand.expand(B, nW, region_side * region_side, D)
                fused_expand = fused_expand.reshape(B * nW, region_side, region_side, D)
                # region_reverse: [B*nW, R, R, D] → [B, H, W, D]
                fused_grid = region_reverse(fused_expand, region_side, H, W)
                fused_grid = fused_grid.reshape(B, H * W, D)

                # 去掉 padding，只保留原始的 N 个 patch
                fused_signal = fused_grid[:, :N, :]  # [B, N, D]

                for i in range(M):
                    z_list[i] = z_list[i] + fusion_weights[i] * fused_signal

            except Exception as e:
                print(f"Modality fusion skipped due to error: {e}")
                pass

        return z_list, fusion_info


class MM_RRTEncoder(nn.Module):
    """
    多模态Region Relation Transformer Encoder
    
    支持灵活配置:
    - num_modalities: 模态数量 (1-5)
    - fusion_stage: 在哪个阶段进行模态融合 ('early', 'middle', 'late')
    - fusion_type: 融合类型 ('self_attention', 'cross_attention', 'concat_mlp', 'weighted_sum')
    
    工作流程:
    1. 输入: M个模态的特征列表 [B, N, D] * M
    2. R-MSA layers: 每个模态独立进行spatial modeling
    3. Modality Fusion: 在指定层进行模态交互
    4. CR-MSA: 跨region attention
    5. 输出: M个模态的融合特征 [B, N, D] * M 或聚合后的特征 [B, N, D]
    """
    def __init__(self, mlp_dim=512, num_modalities=2,
                 fusion_stage='middle', fusion_type='self_attention',
                 fusion_kwargs=None,
                 pos_pos=0, pos='none', peg_k=7,
                 region_num=8, drop_out=0.1, n_layers=2,
                 n_heads=8, drop_path=0., ffn=False, ffn_act='gelu',
                 mlp_ratio=4., epeg=True, epeg_k=15,
                 region_size=0, min_region_num=0, min_region_ratio=0,
                 qkv_bias=True, peg_bias=True, peg_1d=False,
                 cr_msa=True, crmsa_k=3, all_shortcut=False,
                 crmsa_mlp=False, crmsa_heads=8, need_init=True,
                 return_modality_features=False,  # 是否返回所有模态的特征
                 aggregate_modalities=True,  # 是否聚合多模态特征
                 use_gated_fusion=False,  # 是否使用Gated Fusion
                 use_per_layer_fusion=True,  # 是否在每层R-MSA后做模态融合
                 use_cross_patch_anchor=True,  # cross_patch后直接取HE作为输出
                 **kwargs):
        """
        Args:
            mlp_dim: 特征维度
            num_modalities: 模态数量 (1-5)
            fusion_stage: 'early', 'middle', 'late'
            fusion_type: 融合类型
            fusion_kwargs: 融合模块的额外参数
            return_modality_features: 是否返回每个模态的独立特征
            aggregate_modalities: 是否将多模态聚合成一个特征
            use_gated_fusion: 是否使用Gated Fusion (Gate + Residual + Dropout)
            其他参数与原始RRT相同
        """
        super(MM_RRTEncoder, self).__init__()
        
        self.final_dim = mlp_dim
        self.num_modalities = num_modalities
        self.fusion_stage = fusion_stage
        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = all_shortcut
        self.return_modality_features = return_modality_features
        self.aggregate_modalities = aggregate_modalities and num_modalities > 1
        self.use_per_layer_fusion = use_per_layer_fusion and num_modalities > 1
        self.use_cross_patch_anchor = use_cross_patch_anchor and (fusion_type in ('cross_patch_attention', 'concat_fusion', 'modality_region_attention'))
        
        # 确定在哪一层进行融合
        if fusion_stage == 'early':
            fusion_layers = [0]
        elif fusion_stage == 'middle':
            fusion_layers = [n_layers // 2]
        elif fusion_stage == 'late':
            fusion_layers = [n_layers - 1]
        elif fusion_stage == 'all':
            fusion_layers = list(range(n_layers))
        else:
            fusion_layers = []
        
        # R-MSA layers (多模态版本)
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            enable_fusion = (i in fusion_layers) and self.use_per_layer_fusion
            self.layers.append(ModalityTransLayer(
                dim=mlp_dim,
                num_modalities=num_modalities,
                head=n_heads,
                drop_out=drop_out,
                drop_path=drop_path,
                ffn=ffn,
                ffn_act=ffn_act,
                mlp_ratio=mlp_ratio,
                region_num=region_num,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                enable_modality_fusion=enable_fusion,
                fusion_type=fusion_type,
                fusion_kwargs=fusion_kwargs,
                use_gated_fusion=use_gated_fusion,
                **kwargs
            ))
        
        # CR-MSA: always ModuleList for consistent init (use [0] when aggregating)
        if cr_msa:
            self.cr_msa = nn.ModuleList([
                CrossRegionAttention(
                    dim=mlp_dim, num_heads=crmsa_heads, drop=drop_out,
                    region_num=region_num, head_dim=mlp_dim // crmsa_heads,
                    epeg=epeg, region_size=region_size,
                    min_region_num=min_region_num, min_region_ratio=min_region_ratio,
                    qkv_bias=qkv_bias, crmsa_k=crmsa_k, crmsa_mlp=crmsa_mlp, **kwargs
                ) for _ in range(max(1, num_modalities))
            ])
        else:
            self.cr_msa = None
        
        # 正交 IHC 修正: IHC 只补 HE 没有的方向, 不重复 HE 已有信息
        self.fixed_alpha = kwargs.get('fixed_alpha', None)  # None=可学习, float=固定值
        if self.aggregate_modalities:
            dim = mlp_dim
            # IHC 投影: 轻量级, 对齐到 HE 特征空间
            self.ihc_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
            # 全局缩放: α 初始 ≈ 0.018, 可用 fixed_alpha 固定
            if self.fixed_alpha is None:
                self.alpha_logit = nn.Parameter(torch.tensor(-4.0))
            else:
                self.register_buffer('alpha_logit', torch.tensor(0.0))  # placeholder
        
        self.pos_pos = pos_pos
        
        if need_init:
            self.apply(initialize_weights)
    
    def forward(self, x):
        """
        Args:
            x: 
                - 单模态: [B, N, C] 或 [N, C]
                - 多模态: list of [B, N, C] with length num_modalities
                - 或 [B, M, N, C] (已堆叠)
        
        Returns:
            如果 aggregate_modalities=True: [B, N, C]
            否则: list of [B, N, C] with length num_modalities
        """
        # 处理输入格式
        x_list, shape_info = self._process_input(x)
        shape_len = shape_info['shape_len']
        
        # 保存shortcut
        if self.all_shortcut:
            x_shortcut = [xi.clone() for xi in x_list]
        
        # R-MSA layers with modality fusion
        fusion_info_list = []
        for layer in self.layers:
            x_list, fusion_info = layer(x_list)
            if fusion_info:
                fusion_info_list.append(fusion_info)
        
        gate_stats = None

        # ── Joint Region-Modality CR-MSA (replaces regular CR-MSA) ──
        use_joint = any(getattr(layer, 'enable_joint_region', False) for layer in self.layers)
        if use_joint:
            processor = None
            for layer in self.layers:
                if getattr(layer, 'enable_joint_region', False) and layer.joint_region_processor is not None:
                    processor = layer.joint_region_processor
                    break
            if processor is not None:
                fused = processor(x_list)
                x_list = [fused]
                gate_stats = {'joint_region_modality': True}

        # CR-MSA (skip if joint_region already handled)
        if self.cr_msa is not None and not use_joint:
            if self.aggregate_modalities:
                # 先聚合，再进行CR-MSA
                x_agg, gate_stats = self._aggregate_modalities(x_list, return_gate_stats=True)
                x_agg = self.cr_msa[0](x_agg)
                x_list = [x_agg]
            else:
                # 对每个模态独立进行CR-MSA
                for i, cr_msa in enumerate(self.cr_msa):
                    x_list[i] = cr_msa(x_list[i])
        
        # Shortcut
        if self.all_shortcut:
            x_list = [x + s for x, s in zip(x_list, x_shortcut)]
        
        # Norm
        x_list = [self.norm(x) for x in x_list]
        
        # 恢复原始形状
        x_list = self._restore_shape(x_list, shape_info)
        
        # 返回格式
        if self.aggregate_modalities or self.num_modalities == 1:
            result = x_list[0] if len(x_list) == 1 else x_list
            return (result, gate_stats) if gate_stats is not None else result
        elif self.return_modality_features:
            return x_list
        else:
            # 默认返回聚合特征
            return self._aggregate_modalities(x_list) if len(x_list) > 1 else x_list[0]
    
    def _process_input(self, x):
        """处理输入，统一转换为list格式"""
        shape_len = 3
        B = None
        
        if isinstance(x, list):
            # 已经是list格式
            x_list = x
            if len(x_list[0].shape) == 2:  # [N, C]
                x_list = [xi.unsqueeze(0) for xi in x_list]
                shape_len = 2
            B = x_list[0].shape[0]
        elif len(x.shape) == 4:  # [B, M, N, C]
            B, M, N, C = x.shape
            x_list = [x[:, i, :, :] for i in range(M)]
            shape_len = 4
        elif len(x.shape) == 3:  # [B, N, C] - 单模态
            x_list = [x]
            B = x.shape[0]
        elif len(x.shape) == 2:  # [N, C] - 单模态，无batch
            x_list = [x.unsqueeze(0)]
            shape_len = 2
            B = 1
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        return x_list, {'shape_len': shape_len, 'B': B}
    
    def _restore_shape(self, x_list, shape_info):
        """恢复原始形状"""
        shape_len = shape_info['shape_len']
        
        if shape_len == 2:
            x_list = [xi.squeeze(0) for xi in x_list]
        elif shape_len == 4:
            # 不需要处理，保持list格式
            pass
        
        return x_list
    
    def _aggregate_modalities(self, x_list, return_gate_stats=False):
        """
        模态聚合。

        默认: 正交 IHC 修正 (IHC 只补 HE 没有的方向)
        当 use_cross_patch_anchor=True: 直接取 cross_patch 增强后的 HE 作为输出
        """
        if len(x_list) == 1:
            if return_gate_stats:
                return x_list[0], None
            return x_list[0]

        # ── Cross Patch Anchor: HE already enhanced by cross_patch, just return it ──
        if self.use_cross_patch_anchor:
            he = x_list[0]
            if return_gate_stats:
                return he, {'cross_patch_anchor': True}
            return he

        # ── Default: Orthogonal IHC Correction ──
        he = x_list[0]                      # [B, N, D]  HE-RRT output
        ihc_list = x_list[1:]               # IHC patch embeddings
        ihc = torch.stack(ihc_list, dim=1).mean(dim=1)  # [B, N, D]  mean pool

        # Project IHC to align with HE feature space
        ihc = self.ihc_proj(ihc)            # [B, N, D]

        # Orthogonal decomposition: IHC = parallel + orth
        # parallel = projection of IHC onto HE unit direction (redundant info)
        # orth     = IHC - parallel (complementary info)
        he_unit = F.normalize(he, dim=-1)                           # [B, N, D]
        coeff = (ihc * he_unit).sum(dim=-1, keepdim=True)           # [B, N, 1]
        ihc_parallel = coeff * he_unit                              # [B, N, D]
        ihc_orth = ihc - ihc_parallel                               # [B, N, D]

        # α scales the orthogonal correction
        if self.fixed_alpha is not None:
            alpha = self.fixed_alpha
        else:
            alpha = torch.sigmoid(self.alpha_logit)  # init ≈ 0.018

        # Uncertainty mask: HE 越不确定的地方, 越需要 IHC 补充
        conf = he.norm(dim=-1, keepdim=True)          # [B, N, 1] 特征范数 ≈ 置信度
        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-6)  # [0, 1]
        uncertainty = 1 - conf                         # [0, 1] 不确定度

        fused = he + alpha * uncertainty * ihc_orth

        if return_gate_stats:
            # Verify: cos(he, ihc_orth) should be ≈ 0
            cos_after = F.cosine_similarity(he, ihc_orth, dim=-1).mean().item()
            stats = {
                'alpha': alpha.item() if hasattr(alpha, 'item') else alpha,
                'orth_norm': ihc_orth.norm(dim=-1).mean().item(),
                'parallel_norm': ihc_parallel.norm(dim=-1).mean().item(),
                'cos_after': cos_after,
            }
            return fused, stats

        return fused


# 便捷创建函数
def create_mm_rrt_encoder(num_modalities=2, fusion_type='self_attention', 
                          fusion_stage='middle', **kwargs):
    """
    创建多模态RRT编码器的便捷函数
    
    Args:
        num_modalities: 模态数量
        fusion_type: 融合类型
        fusion_stage: 融合阶段
        **kwargs: 其他RRT参数
    
    Returns:
        MM_RRTEncoder实例
    """
    return MM_RRTEncoder(
        num_modalities=num_modalities,
        fusion_type=fusion_type,
        fusion_stage=fusion_stage,
        **kwargs
    )


if __name__ == "__main__":
    print("Testing MM_RRTEncoder...")
    
    # Test 1: 单模态 (向后兼容)
    print("\n1. Testing single modality...")
    encoder_single = MM_RRTEncoder(mlp_dim=512, num_modalities=1, n_layers=2)
    x_single = torch.randn(2, 100, 512)  # [B, N, C]
    out_single = encoder_single(x_single)
    print(f"Input: {x_single.shape}, Output: {out_single.shape}")
    
    # Test 2: 双模态
    print("\n2. Testing two modalities...")
    encoder_dual = MM_RRTEncoder(
        mlp_dim=512, 
        num_modalities=2, 
        n_layers=2,
        fusion_type='self_attention',
        aggregate_modalities=True
    )
    x_dual = [torch.randn(2, 100, 512) for _ in range(2)]
    out_dual = encoder_dual(x_dual)
    print(f"Input: {[x.shape for x in x_dual]}, Output: {out_dual.shape}")
    
    # Test 3: 五模态
    print("\n3. Testing five modalities...")
    encoder_five = MM_RRTEncoder(
        mlp_dim=512,
        num_modalities=5,
        n_layers=2,
        fusion_type='self_attention',
        aggregate_modalities=True
    )
    x_five = [torch.randn(2, 100, 512) for _ in range(5)]
    out_five = encoder_five(x_five)
    print(f"Input: {[x.shape for x in x_five]}, Output: {out_five.shape}")
    
    print("\nAll tests passed!")
