"""
MM-RRT-Light + ABMIL: Multi-Modal Region Relation Transformer with Attention MIL

多模态RRT与ABMIL的结合模型。

支持灵活配置:
- num_modalities: 模态数量 (1-5)
- modality_list: 使用的模态名称列表 (e.g., ['RAW', 'ER'])
- fusion_type: 模态融合类型
- fusion_stage: 融合阶段

数据流:
1. 输入: M个模态的patch特征
2. 特征投影: 每个模态独立投影
3. MM-RRT Encoder: 多模态RRT编码 (包含模态融合)
4. ABMIL: 注意力聚合 + 分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import warnings
from typing import List, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入RMSA相关模块
try:
    from rmsa import Mlp
except ImportError:
    # 如果rmsa不存在，定义简单的Mlp
    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, 
                     act_layer=nn.GELU, drop=0.):
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

from models.mm_rrt_encoder import MM_RRTEncoder, initialize_weights


# AttentionMIL 和 GatedAttentionMIL 已提取到 models/abmil.py
# 通过 MIL_REGISTRY 动态加载


class MM_RRT_ABMIL(nn.Module):
    """
    多模态RRT + ABMIL 模型
    
    特点:
    - 支持2-5个模态的灵活配置
    - 支持多种融合策略
    - 向后兼容单模态
    
    配置示例:
    # 双模态 (RAW + ER)
    model = MM_RRT_ABMIL(
        num_modalities=2,
        modality_list=['RAW', 'ER'],
        fusion_type='self_attention',
        fusion_stage='middle'
    )
    
    # 五模态 (RAW + ER + PR + HER2 + KI67)
    model = MM_RRT_ABMIL(
        num_modalities=5,
        modality_list=['RAW', 'ER', 'PR', 'HER2', 'KI67'],
        fusion_type='self_attention',
        fusion_stage='middle'
    )
    """
    def __init__(self, 
                 # 模态配置
                 num_modalities=2,
                 modality_list=None,  # e.g., ['RAW', 'ER']
                 
                 # 特征维度
                 input_dim=768,  # CTransPath特征维度
                 mlp_dim=512,
                 num_classes=4,
                 
                 # 投影层
                 dropout=0.5,
                 act='relu',
                 
                 # RRT Encoder 参数
                 region_num=8,
                 n_layers=2,
                 n_heads=8,
                 drop_path=0.,
                 trans_dropout=0.1,
                 epeg=True,
                 epeg_k=15,
                 crmsa_k=3,
                 cr_msa=True,
                 all_shortcut=False,
                 crmsa_mlp=False,
                 crmsa_heads=8,
                 
                 # 模态融合参数
                 fusion_type='self_attention',  # 'self_attention', 'cross_attention', etc.
                 fusion_stage='middle',  # 'early', 'middle', 'late'
                 fusion_kwargs=None,
                 use_gated_fusion=False,  # 是否使用Gated Fusion (Gate + Residual + Dropout)
                 use_per_layer_fusion=True,  # 是否在每层R-MSA后做模态融合 (False=仅末端残差融合)
                 use_logit_fusion=False,    # 是否使用决策级融合 (β*HE + (1-β)*IHC at logit level)
                 
                 # MIL 参数
                 mil_type='abmil',     # MIL模型类型: 'abmil', 'gated_abmil', 'transmil'
                 abmil_hidden_dim=128,
                 use_gated=False,      # [deprecated] 用 mil_type='gated_abmil' 替代
                 
                 # 其他
                 **kwargs):
        super(MM_RRT_ABMIL, self).__init__()
        
        self.num_modalities = num_modalities
        self.modality_list = modality_list or [f'MOD_{i}' for i in range(num_modalities)]
        self.input_dim = input_dim
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.cross_region = None  # created lazily for two_stage_region

        # 检查模态列表长度
        assert len(self.modality_list) == num_modalities, \
            f"modality_list length ({len(self.modality_list)}) must match num_modalities ({num_modalities})"
        
        # 为每个模态创建独立的特征投影层
        self.patch_to_emb = nn.ModuleList()
        for i in range(num_modalities):
            layers = [nn.Linear(input_dim, mlp_dim)]
            if act.lower() == 'relu':
                layers += [nn.ReLU()]
            elif act.lower() == 'gelu':
                layers += [nn.GELU()]
            self.patch_to_emb.append(nn.Sequential(*layers))
        
        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        # MM-RRT Encoder
        encoder_extra = {}
        for k in ['aggregate_modalities', 'use_cross_patch_anchor']:
            if k in kwargs:
                encoder_extra[k] = kwargs.pop(k)
        self.rrt_encoder = MM_RRTEncoder(
            mlp_dim=mlp_dim,
            num_modalities=num_modalities,
            fusion_type=fusion_type,
            fusion_stage=fusion_stage,
            fusion_kwargs=fusion_kwargs,
            use_gated_fusion=use_gated_fusion,
            use_per_layer_fusion=use_per_layer_fusion,
            region_num=region_num,
            n_layers=n_layers,
            n_heads=n_heads,
            drop_path=drop_path,
            drop_out=trans_dropout,
            epeg=epeg,
            epeg_k=epeg_k,
            crmsa_k=crmsa_k,
            cr_msa=cr_msa,
            all_shortcut=all_shortcut,
            crmsa_mlp=crmsa_mlp,
            crmsa_heads=crmsa_heads,
            aggregate_modalities=encoder_extra.get('aggregate_modalities', True),
            use_cross_patch_anchor=encoder_extra.get('use_cross_patch_anchor', True),
            need_init=True,
            **kwargs
        )
        
        # MIL — 通过注册表动态加载（可插拔）
        from models.mil_registry import MIL_REGISTRY

        # 向后兼容：use_gated → mil_type
        resolved_mil_type = mil_type
        if use_gated and mil_type == 'abmil':
            resolved_mil_type = 'gated_abmil'
            warnings.warn(
                "参数 'use_gated=True' 已弃用，请改用 mil_type='gated_abmil'。"
                "未来版本将移除 use_gated 参数。",
                DeprecationWarning, stacklevel=2
            )

        self.mil_type = resolved_mil_type
        self.mil = MIL_REGISTRY.create(
            resolved_mil_type,
            input_dim=mlp_dim,
            hidden_dim=abmil_hidden_dim,
            num_classes=num_classes,
            dropout_rate=dropout,
        )

        # ── Official RRTEncoder for single-modality Stage-1 (two_stage_region etc.) ──
        self.rrt_he = None   # official RRTEncoder for HE
        self.rrt_ihc = None  # official RRTEncoder for IHC (always independent)

        # ── Single-modality HE-only: official RRTEncoder ──
        # Uses the same TransLayer-wrapped R²T as two_stage_region HE branch,
        # ensuring a fair HE-only baseline (not MM_RRTEncoder with bare CR-MSA).
        if num_modalities == 1:
            from models.mm_rrt_encoder import RRTEncoder
            self.rrt_he = RRTEncoder(
                mlp_dim=mlp_dim, region_num=region_num, n_layers=n_layers,
                n_heads=n_heads, drop_path=drop_path, drop_out=trans_dropout,
                epeg=epeg, epeg_k=epeg_k, crmsa_k=crmsa_k,
                cr_msa=cr_msa, all_shortcut=all_shortcut,
                crmsa_mlp=crmsa_mlp, crmsa_heads=crmsa_heads,
                need_init=True,
            )

        # Cross-region re-embedding (for two_stage_region)
        self.cross_region_mod = None
        if fusion_type == 'two_stage_region' and num_modalities > 1:
            from models.mm_rrt_encoder import RRTEncoder
            # Stage 2 模块选择:
            #   'staining_msa' (默认) — "染色即区域" 跨染色 MSA 对称融合
            #   'he_anchor'            — 旧版 HE 锚定 cross-attn (消融对照)
            self.stage2_type = kwargs.get('stage2_type', 'staining_msa')
            if self.stage2_type == 'staining_msa':
                from models.cross_staining_crmsa import CrossStainingCRMSA
                self.cross_region_mod = CrossStainingCRMSA(
                    dim=mlp_dim, num_heads=crmsa_heads, region_num=region_num,
                    crmsa_k=crmsa_k, drop_out=trans_dropout if isinstance(trans_dropout, (int, float)) else 0.1,
                    drop_path=drop_path, epeg=epeg, epeg_k=epeg_k, crmsa_mlp=crmsa_mlp,
                )
            else:
                from models.cross_region_reembedding import CrossRegionReembedding
                self.cross_region_mod = CrossRegionReembedding(
                    dim=mlp_dim, crmsa_k=crmsa_k, crmsa_heads=n_heads,
                    region_num=region_num, epeg=True, epeg_k=epeg_k,
                    drop_out=trans_dropout if isinstance(trans_dropout, (int, float)) else 0.1,
                    drop_path=drop_path,
                )
            # HE encoder (official R²T, independent weights)
            self.rrt_he = RRTEncoder(
                mlp_dim=mlp_dim, region_num=region_num, n_layers=n_layers,
                n_heads=n_heads, drop_path=drop_path, drop_out=trans_dropout,
                epeg=epeg, epeg_k=epeg_k, crmsa_k=crmsa_k,
                cr_msa=cr_msa, all_shortcut=all_shortcut,
                crmsa_mlp=crmsa_mlp, crmsa_heads=crmsa_heads,
                need_init=True,
            )
            # IHC encoder (official R²T, INDEPENDENT weights — never shared with HE)
            self.rrt_ihc = RRTEncoder(
                mlp_dim=mlp_dim, region_num=region_num, n_layers=n_layers,
                n_heads=n_heads, drop_path=drop_path, drop_out=trans_dropout,
                epeg=epeg, epeg_k=epeg_k, crmsa_k=crmsa_k,
                cr_msa=cr_msa, all_shortcut=all_shortcut,
                crmsa_mlp=crmsa_mlp, crmsa_heads=crmsa_heads,
                need_init=True,
            )
        # Direct cross-region (no Stage 1 R²T) — for ablation
        self.direct_cross_region = (fusion_type == 'two_stage_direct' and num_modalities > 1)
        if self.direct_cross_region:
            from models.cross_region_reembedding import CrossRegionReembedding
            self.cross_region_mod = CrossRegionReembedding(
                dim=mlp_dim, crmsa_k=crmsa_k, crmsa_heads=n_heads,
                region_num=region_num, epeg=True, epeg_k=epeg_k,
                drop_out=trans_dropout if isinstance(trans_dropout, (int, float)) else 0.1,
            )
        
        # Logit-level fusion: HE (full RRT) + IHC (simple MLP) at decision level
        self.use_logit_fusion = (use_logit_fusion or kwargs.get('use_logit_fusion', False)) and num_modalities > 1
        self.fixed_beta = kwargs.get('fixed_beta', None)  # None=训练β, 浮点数=固定β
        if self.use_logit_fusion:
            # IHC 简单路径: mean pool patches → MLP → classifier
            self.ihc_mlp = nn.Sequential(
                nn.Linear(mlp_dim, mlp_dim // 2),
                nn.LayerNorm(mlp_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_dim // 2, num_classes),
            )
            # β: HE vs IHC 权重 — None=learnable, float=fixed
            if self.fixed_beta is None:
                self.beta_logit = nn.Parameter(torch.tensor(0.0))  # learnable
            else:
                self.register_buffer('beta_logit', torch.tensor(0.0))  # placeholder

        # Consistency-guided fusion: reliability = 1 - variance(IHC_logits)
        self.use_consistency_fusion = kwargs.get('use_consistency_fusion', False) and num_modalities > 1
        if self.use_consistency_fusion:
            # One simple head per IHC modality: mean pool → MLP → logit
            self.ihc_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(mlp_dim, mlp_dim // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(mlp_dim // 4, num_classes),
                ) for _ in range(num_modalities - 1)  # skip HE
            ])
            # β range: 0.6-0.9, controlled by reliability
            self.register_buffer('beta_base', torch.tensor(0.9))
            self.register_buffer('beta_range', torch.tensor(0.3))

        # Multi-Correction: per-IHC CorrectionNet, tanh→mean, α=0.1 (warmup 5 epochs)
        self.use_correction_only = kwargs.get('use_correction_only', False) and num_modalities > 1
        if self.use_correction_only:
            dim = mlp_dim
            num_ihc = num_modalities - 1
            self.correction_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 1),
                ) for _ in range(num_ihc)
            ])
            self.correction_alpha = 0.0  # warmup: first 5 epochs α=0
            self.alpha_mode = kwargs.get('alpha_mode', 'feature')  # 'feature' or 'confidence'
            if self.alpha_mode == 'feature':
                self.alpha_net = nn.Sequential(
                    nn.Linear(dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Sigmoid())

        # MCLC: Manifold-Constrained Logit Correction (Sinkhorn mixing matrix)
        self.use_mclc = kwargs.get('use_mclc', False) and num_modalities > 1
        self.freeze_mclc = kwargs.get('freeze_mclc', False)
        self.he_only = kwargs.get('he_only', False)  # HE-only baseline (same arch, no PR)
        if self.use_mclc:
            # mclc not needed
            mclc_init_diag = kwargs.get('mclc_init_diag', 2.0)
            self.logit_mixer = ManifoldLogitCorrection(
                num_modalities=num_modalities, num_classes=num_classes,
                init_diag=mclc_init_diag, frozen=self.freeze_mclc)
            # IHC logit heads: one per auxiliary modality (skip HE)
            num_ihc = num_modalities - 1
            self.ihc_logit_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(mlp_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes))
                for _ in range(num_ihc)
            ])

        # Partial Alignment: PR shared → HE shared subspace, PR private retained
        self.use_partial_align = kwargs.get('use_partial_align', False) and num_modalities > 1
        if self.use_partial_align:
            from .partial_align import PartialAlignmentFusion
            self.pa_fusion = PartialAlignmentFusion(dim=mlp_dim)

        # Shared RRT: HE+IHC through same encoder weights → shared semantic space
        self.use_shared_rrt = kwargs.get('use_shared_rrt', False) and num_modalities > 1
        self.shared_rrt_alpha = kwargs.get('shared_rrt_alpha', 0.02)

        # SRP Residual Fusion: P + β·g·Δ, IHC as small residual on HE prototype
        self.use_srp_fusion = kwargs.get('use_srp_fusion', False) and num_modalities > 1
        self.srp_beta = kwargs.get('srp_beta', 0.1)
        if self.use_srp_fusion:
            srp_mode = kwargs.get('srp_mode', 'residual')
            if srp_mode == 'scale':
                from .srp_fusion import SRPScaleFusion
                self.srp_fusion = SRPScaleFusion(dim=mlp_dim, beta=self.srp_beta)
            else:
                from .srp_fusion import SRPResidualFusion
                self.srp_fusion = SRPResidualFusion(dim=mlp_dim, beta=self.srp_beta)

        # Low-rank Feature Correction: IHC → constrained residual on HE features
        self.use_lowrank_correction = kwargs.get('use_lowrank_correction', False) and num_modalities > 1
        if self.use_lowrank_correction:
            dim = mlp_dim
            # IHC projection to HE space
            self.ihc_proj_lr = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
            # Bottleneck: 512+512 → 64 → 512
            self.lr_bottleneck_down = nn.Linear(dim * 2, 64)
            self.lr_bottleneck_up   = nn.Linear(64, dim)
            self.lr_norm = nn.LayerNorm(dim)

        # Logit Query Attention: HE-conditioned attention over multi-modal logits
        self.use_logit_attn = kwargs.get('use_logit_attn', False) and num_modalities > 1
        if self.use_logit_attn:
            dim = mlp_dim
            num_ihc = num_modalities - 1
            # IHC heads: each outputs [B, 2] logits
            self.ihc_logit_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, num_classes),
                ) for _ in range(num_ihc)
            ])
            # HE feature → Query
            self.query_proj = nn.Linear(dim, num_classes)
            # Logits → Key
            self.key_proj = nn.Linear(num_classes, num_classes, bias=False)
            # Temperature
            self.temperature = nn.Parameter(torch.tensor(1.0))

        # ARLC: Adaptive Reliability-guided Logit Correction
        self.use_arlc_fusion = kwargs.get('use_arlc_fusion', False) and num_modalities > 1
        if self.use_arlc_fusion:
            dim = mlp_dim
            num_ihc = num_modalities - 1
            # IHC Aggregator: learned attention over IHC modalities
            self.ihc_teacher_score = nn.Linear(dim, 1)
            # ReliabilityNet: he.detach() ⊕ ihc → r∈[0,1]
            self.reliability_net = nn.Sequential(
                nn.Linear(dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
            # CorrectionNet: ihc only → raw_δ (scalar)
            self.correction_net = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
            )

        self.apply(initialize_weights)

        # Load pretrained single-HE checkpoint for HE branch (correction-only mode)
        pretrained_ckpt = kwargs.get('pretrained_he_ckpt', None)
        if pretrained_ckpt and (self.use_correction_only or self.use_logit_attn):
            pretrained = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)
            pretrained_sd = pretrained['model_state_dict']
            own_sd = self.state_dict()
            copied = 0
            for k, v in own_sd.items():
                if k in pretrained_sd and v.shape == pretrained_sd[k].shape:
                    own_sd[k] = pretrained_sd[k]
                    copied += 1
                elif 'cr_msa.0.' in k or 'layers.0.' in k or 'patch_to_emb.0.' in k:
                    alt_k = k.replace('cr_msa.0.','cr_msa.').replace('layers.0.','layers.').replace('patch_to_emb.0.','patch_to_emb.')
                    if alt_k in pretrained_sd and v.shape == pretrained_sd[alt_k].shape:
                        own_sd[k] = pretrained_sd[alt_k]
                        copied += 1
            self.load_state_dict(own_sd)
            print(f'[Correction] Loaded {copied} HE params from pretrained checkpoint')
    
    def forward(self, x, return_features=False, return_modality_attns=False):
        """
        Args:
            x: 
                - 单模态: [B, N, input_dim] 或 [N, input_dim]
                - 多模态: list of [B, N, input_dim] with length num_modalities
            return_features: 是否返回中间特征
            return_modality_attns: 是否返回模态注意力权重
        
        Returns:
            logits: 分类logits
            Y_hat: 预测标签
            A: 注意力权重 (可选)
            features: 中间特征 (如果return_features=True)
        """
        # 处理输入格式
        if not isinstance(x, list):
            if self.num_modalities == 1:
                x = [x]
            else:
                raise ValueError(f"Expected list of {self.num_modalities} modality features")
        
        # 允许少模态（HE-only 评估或 logit fusion）
        gate_stats = None  # initialize
        
        # 确保每个模态有batch维度
        x_list = []
        for xi in x:
            if len(xi.shape) == 2:
                xi = xi.unsqueeze(0)
            x_list.append(xi)
        
        # 1. 特征投影 (每个模态独立)
        x_emb_list = []
        for i, xi in enumerate(x_list):
            x_emb = self.patch_to_emb[i](xi)
            x_emb = self.dp(x_emb)
            x_emb_list.append(x_emb)

        # ── Single-modality shortcut: official RRTEncoder → MIL ──
        if self.num_modalities == 1:
            z = self.rrt_he(x_emb_list[0])
            if len(z.shape) == 2:
                z = z.unsqueeze(0)
            mil_result = self.mil(z)
            return (mil_result['logits'],
                    torch.argmax(mil_result['logits'], dim=-1),
                    mil_result.get('attention', None), None)

        # ══════════════════════════════════════════════════════════
        # Instance Bag Expansion: each modality independently through
        # the same R²T encoder, then concat instances before MIL.
        # No fusion, no cross attention, no gate.
        # ══════════════════════════════════════════════════════════
        # ══════════════════════════════════════════════════════════
        # Two-stage Modality-aware R²T:
        #   HE → R²T → Z_HE,  IHC → R²T → Z_IHC
        #   CrossRegionReembedding(Z_HE, Z_IHC) → Z_final [B,N,D]
        #   ABMIL(Z_final) → logits
        # ══════════════════════════════════════════════════════════
        # ═══ Two-stage Region R²T (official RRTEncoder per modality) ═══
        if self.fusion_type == 'two_stage_region' and len(x_emb_list) > 1:
            # Stage 1: official R²T encoder for each modality (no list wrapping)
            z_he = self.rrt_he(x_emb_list[0])
            z_ihc = self.rrt_ihc(x_emb_list[1])
            if len(z_he.shape) == 2: z_he = z_he.unsqueeze(0)
            if len(z_ihc.shape) == 2: z_ihc = z_ihc.unsqueeze(0)

            # Stage 2: cross-staining fusion
            if self.stage2_type == 'staining_msa':
                # official-style cross-staining CR-MSA, output concatenated
                # along patch dim [B, N_he+N_ihc, D]
                z_final = self.cross_region_mod([z_he, z_ihc])
                fusion_stats = {'two_stage_region': True, 'stage2': 'staining_msa'}
            else:
                # 旧版 HE 锚定 cross-attn (消融对照)
                z_final = self.cross_region_mod(z_he, z_ihc)
                fusion_stats = {'two_stage_region': True, 'stage2': 'he_anchor'}

            # Stage 3: ABMIL
            mil_result = self.mil(z_final)
            return (mil_result['logits'], torch.argmax(mil_result['logits'], dim=-1),
                    mil_result.get('attention', None), fusion_stats)

        # ═══ Direct cross-region (no Stage 1 R²T) — ablation ═══
        if self.direct_cross_region and len(x_emb_list) > 1:
            # Skip R²T, feed projected features directly to cross-region
            z_final = self.cross_region_mod(x_emb_list[0], x_emb_list[1])
            mil_result = self.mil(z_final)
            return (mil_result['logits'], torch.argmax(mil_result['logits'], dim=-1),
                    mil_result.get('attention', None), {'two_stage_direct': True})

        if self.fusion_type == 'instance_bag_expansion' and len(x_emb_list) > 1:
            outputs = []
            for x_emb in x_emb_list:
                z = self.rrt_encoder([x_emb])
                if isinstance(z, tuple):
                    z = z[0]
                if len(z.shape) == 2:
                    z = z.unsqueeze(0)
                outputs.append(z)
            # Concat along instance dim: [B, N, D] + [B, N, D] → [B, 2N, D]
            z = torch.cat(outputs, dim=1)
            mil_result = self.mil(z)
            logits = mil_result['logits']
            Y_hat = torch.argmax(logits, dim=-1)
            A = mil_result.get('attention', None)
            return (logits, Y_hat, A, {'instance_bag_expansion': True, 'modalities': len(x_emb_list)})

        # 2. MCLC: Sinkhorn logit mixing (N modalities)
        aux_loss = torch.tensor(0.0)
        fusion_stats = None
        if self.use_mclc and len(x_emb_list) > 1:
            # HE path
            encoder_out = self.rrt_encoder([x_emb_list[0]])
            x_fused = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out
            mil_result = self.mil(x_fused)
            logits_he = mil_result['logits']
            A = mil_result.get('attention', None)

            if self.he_only:
                # HE-only baseline: same architecture, no IHC contribution
                logits = logits_he
                Y_hat = torch.argmax(logits, dim=-1)
                fusion_stats = {'logits_he': logits_he}
            else:
                # IHC paths: one head per auxiliary modality
                num_ihc = len(x_emb_list) - 1
                logits_ihc_list = []
                for i in range(num_ihc):
                    ihc_feat = x_emb_list[i + 1].mean(dim=1)
                    logits_ihc = self.ihc_logit_heads[i](ihc_feat)
                    logits_ihc_list.append(logits_ihc)
                # Sinkhorn mixing: HE + all IHCs
                logits, W = self.logit_mixer([logits_he] + logits_ihc_list)
                Y_hat = torch.argmax(logits, dim=-1)
                fusion_stats = {
                    'logits_he': logits_he,
                    'logits_ihc_list': logits_ihc_list,
                    'W': W,
                }
                # Backward-compat: expose first IHC as logits_pr
                if num_ihc >= 1:
                    fusion_stats['logits_pr'] = logits_ihc_list[0]

        # 3. Partial Alignment Feature Reconstruction
        elif self.use_partial_align and len(x_emb_list) > 1:
            # HE through RRT
            he_out = self.rrt_encoder([x_emb_list[0]])
            H = he_out[0] if isinstance(he_out, tuple) else he_out

            # PR through patch_to_emb only (no RRT)
            P = x_emb_list[1]

            # Partial alignment fusion → raw tensors
            x_fused, P_s, P_p, H_s, delta = self.pa_fusion(H, P)
            fusion_stats = {'P_s': P_s, 'P_p': P_p, 'H_s': H_s, 'delta': delta}
            aux_loss = None  # computed in train.py

            mil_result = self.mil(x_fused)
            logits = mil_result['logits']
            Y_hat = mil_result['Y_hat']
            A = mil_result.get('attention', None)

        elif self.use_shared_rrt and len(x_emb_list) > 1:
            # HE through shared encoder
            he_out = self.rrt_encoder([x_emb_list[0]])
            h_he = he_out[0] if isinstance(he_out, tuple) else he_out

            # PR through same shared encoder
            pr_out = self.rrt_encoder([x_emb_list[1]])
            h_pr = pr_out[0] if isinstance(pr_out, tuple) else pr_out

            # Simple addition in shared semantic space
            x_fused = h_he + self.shared_rrt_alpha * h_pr

            mil_result = self.mil(x_fused)
            logits = mil_result['logits']
            Y_hat = mil_result['Y_hat']
            A = mil_result.get('attention', None)
            fusion_stats = {
                'he_norm': h_he.norm(dim=-1).mean().item(),
                'pr_norm': h_pr.norm(dim=-1).mean().item(),
            }

        # 3. SRP Residual Fusion: P + β·g·Δ
        if not self.use_mclc and not self.use_partial_align:
            fusion_stats = None
        if self.use_srp_fusion and len(x_emb_list) > 1:
            # HE: through encoder → prototype P [B,N,D]
            encoder_out = self.rrt_encoder([x_emb_list[0]])
            P = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out

            # IHC: simple projection to match dim
            M = x_emb_list[1]  # [B,N,D] already projected by patch_to_emb

            # SRP fusion
            Z, gate, delta = self.srp_fusion(P, M)

            # MIL on fused
            mil_result = self.mil(Z)
            logits = mil_result['logits']
            Y_hat = mil_result['Y_hat']
            A = mil_result.get('attention', None)
            x_fused = Z
            fusion_stats = {
                'gate_mean': gate.mean().item() if gate is not None else 0,
                'gate_max': gate.max().item() if gate is not None else 0,
                'delta_norm': delta.norm(dim=-1).mean().item() if delta is not None else 0,
                'P_norm': P.norm(dim=-1).mean().item(),
            }

        # 3. Low-rank Feature Correction: h_fused = h_HE + 0.05*tanh(Bottleneck([h_HE.detach, IHC_proj]))
        if not self.use_mclc and not self.use_partial_align:
            fusion_stats = None
        if self.use_lowrank_correction and len(x_emb_list) > 1:
            # HE: single modality through full RRT encoder → clean HE features
            encoder_out = self.rrt_encoder([x_emb_list[0]])
            x_fused = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out
            h_he = x_fused  # [B,N,D]

            # IHC: project patch embeddings
            h_ihc = self.ihc_proj_lr(x_emb_list[1])  # [B,N,D]

            # Low-rank bottleneck: concat → 64 → 512
            concat = torch.cat([h_he.detach(), h_ihc], dim=-1)  # [B,N,2D]
            delta_h = self.lr_bottleneck_up(torch.relu(self.lr_bottleneck_down(concat)))
            delta_h = self.lr_norm(delta_h)
            delta_h = 0.05 * torch.tanh(delta_h)

            x_fused = h_he + delta_h

            mil_result = self.mil(x_fused)
            logits = mil_result['logits']
            Y_hat = mil_result['Y_hat']
            A = mil_result.get('attention', None)
            fusion_stats = {'delta_h_norm': delta_h.norm(dim=-1).mean().item()}

        elif self.use_logit_attn and len(x_emb_list) > 1:
            # HE path: full RRT + MIL → logits_he
            encoder_out = self.rrt_encoder([x_emb_list[0]])
            x_fused = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out
            mil_result = self.mil(x_fused)
            logits_he = mil_result['logits']           # [B, 2]
            he_feat = x_fused.mean(dim=1)               # [B, D]

            # IHC logits: each IHC → simple head → logits [B, 2]
            ihc_logits = [logits_he]
            for i, head in enumerate(self.ihc_logit_heads):
                ihc_feat = x_emb_list[i+1].mean(dim=1)
                ihc_logits.append(head(ihc_feat))
            logits_stack = torch.stack(ihc_logits, dim=1)  # [B, N, 2]

            # Query: HE feature → Q
            query = self.query_proj(he_feat)               # [B, 2]
            # Key: logits → K
            keys = self.key_proj(logits_stack)             # [B, N, 2]
            # Score: Q · K^T
            score = (query.unsqueeze(1) * keys).sum(-1)    # [B, N]
            weight = torch.softmax(score / self.temperature, dim=1)  # [B, N]

            # Fused logits
            logits = (weight.unsqueeze(-1) * logits_stack).sum(dim=1)  # [B, 2]
            Y_hat = torch.argmax(logits, dim=-1)
            A = mil_result.get('attention', None)
            fusion_stats = {
                'weight_mean': weight.mean(dim=0).cpu(),    # [N] avg attention
                'temperature': self.temperature.item(),
            }

        # 3. Multi-Correction
        elif self.use_correction_only and len(x_emb_list) > 1:
            encoder_out = self.rrt_encoder([x_emb_list[0]])
            x_fused = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out
            mil_result = self.mil(x_fused)
            logits_he = mil_result['logits']

            # Each IHC → its own CorrectionNet → tanh
            delta_list = []
            for i in range(len(self.correction_nets)):
                ihc_feat = x_emb_list[i+1].mean(dim=1)           # [B, D]
                raw = self.correction_nets[i](ihc_feat)           # [B, 1]
                delta_list.append(torch.tanh(raw))                # ∈ (-1, 1)

            delta = torch.stack(delta_list, dim=0).mean(dim=0)    # [B, 1]  avg over IHCs
            # Dynamic alpha
            if self.alpha_mode == 'confidence':
                p_he = torch.softmax(logits_he, dim=-1)
                conf = p_he.max(dim=-1).values.unsqueeze(-1)
                dyn_alpha = self.correction_alpha * (1.0 - conf)
            else:
                he_pooled = x_fused.mean(dim=1)
                dyn_alpha = self.correction_alpha * self.alpha_net(he_pooled)
            logits = logits_he + dyn_alpha * torch.cat([-delta, delta], dim=-1)
            Y_hat = torch.argmax(logits, dim=-1)
            A = mil_result.get('attention', None)
            fusion_stats = {'delta': delta.mean().item(), 'n_ihc': len(self.correction_nets)}

        elif self.use_arlc_fusion and len(x_emb_list) > 1:
            # HE path: full RRT + MIL → logits_he
            encoder_out = self.rrt_encoder([x_emb_list[0]])
            x_fused = encoder_out[0] if isinstance(encoder_out, tuple) else encoder_out  # for return_features
            if isinstance(encoder_out, tuple):
                he_feat, _ = encoder_out  # [B, N, D]
            else:
                he_feat = encoder_out
            mil_result = self.mil(he_feat)
            logits_he = mil_result['logits']  # [B, 2]
            he_pooled = he_feat.mean(dim=1)   # [B, D] slide-level HE

            # IHC path: learned attention aggregation → ihc_feat
            ihc_list = [x_emb_list[i].mean(dim=1) for i in range(1, len(x_emb_list))]
            ihc_stack = torch.stack(ihc_list, dim=1)                    # [B, num_ihc, D]
            scores = self.ihc_teacher_score(ihc_stack).squeeze(-1)      # [B, num_ihc]
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)       # [B, num_ihc, 1]
            ihc_feat = (ihc_stack * weights).sum(dim=1)                 # [B, D]

            # ReliabilityNet: he.detach() ⊕ ihc → r∈[0,1]
            r_input = torch.cat([he_pooled.detach(), ihc_feat], dim=-1)  # [B, 2D]
            r = self.reliability_net(r_input)                             # [B, 1]

            # CorrectionNet: ihc only → raw_δ (scalar), capped at ±0.2
            raw_delta = self.correction_net(ihc_feat)                     # [B, 1]
            delta = 0.2 * torch.tanh(raw_delta)                           # [B, 1] ∈ [-0.2, 0.2]

            # Logit correction: logits = logits_he + r * [-delta, +delta]
            correction = r * torch.cat([-delta, delta], dim=-1)          # [B, 2]
            logits = logits_he + correction
            Y_hat = torch.argmax(logits, dim=-1)
            A = mil_result.get('attention', None)
            fusion_stats = {
                'r': r.mean().item(),
                'delta': delta.mean().item(),
                'logits_he': logits_he,
                'logits_corrected': logits,
            }

        elif self.use_consistency_fusion and len(x_emb_list) > 1:
            # HE path: full RRT + MIL
            encoder_out = self.rrt_encoder([x_emb_list[0]])
            if isinstance(encoder_out, tuple):
                x_fused, _ = encoder_out
            else:
                x_fused = encoder_out
            mil_result = self.mil(x_fused)
            logits_he = mil_result['logits']

            # IHC heads: each IHC → simple MLP → logit
            ihc_logits = []
            for i in range(len(self.ihc_heads)):
                ihc_feat = x_emb_list[i+1].mean(dim=1)  # [B, D]
                ihc_logits.append(self.ihc_heads[i](ihc_feat))  # [B, 2]
            ihc_logits = torch.stack(ihc_logits, dim=1)  # [B, num_ihc, 2]

            # IHC consistency → reliability
            p_ihc = torch.sigmoid(ihc_logits)  # [B, num_ihc, 2]
            reliability = 1.0 - p_ihc[:, :, 1].var(dim=1, keepdim=True)  # [B, 1]
            reliability = torch.clamp(reliability, 0.0, 1.0)

            # Dynamic beta: IHC可靠 → beta↓ (多信IHC), IHC不可靠 → beta↑ (多信HE)
            beta = self.beta_base - self.beta_range * reliability.mean()

            # Teacher: mean of IHC logits
            logits_teacher = ihc_logits.mean(dim=1)  # [B, 2]

            # Fused: HE + teacher weighted by beta
            logits = beta * logits_he + (1 - beta) * logits_teacher
            Y_hat = torch.argmax(logits, dim=-1)
            A = mil_result.get('attention', None)
            fusion_stats = {
                'beta': beta.item(),
                'reliability': reliability.mean().item(),
                'logits_he': logits_he,
                'logits_teacher': logits_teacher,
                'p_ihc': p_ihc,
            }

        elif self.use_logit_fusion and len(x_emb_list) > 1:
            # HE path: ONLY HE through full RRT encoder + MIL (no IHC contamination)
            encoder_out = self.rrt_encoder([x_emb_list[0]])  # single modality
            if isinstance(encoder_out, tuple):
                x_fused, _ = encoder_out
            else:
                x_fused = encoder_out
            mil_result = self.mil(x_fused)
            logits_he = mil_result['logits']

            # IHC path: simple mean pool → MLP → logits_ihc
            ihc_feats = [x_emb_list[i].mean(dim=1) for i in range(1, len(x_emb_list))]
            ihc = torch.stack(ihc_feats, dim=1).mean(dim=1)  # [B, D]
            logits_ihc = self.ihc_mlp(ihc)                    # [B, 2]

            # β-weighted logit fusion
            if self.fixed_beta is not None:
                beta = self.fixed_beta
            else:
                beta = torch.sigmoid(self.beta_logit)         # learnable ≈ 0.5
            logits = beta * logits_he + (1 - beta) * logits_ihc
            Y_hat = torch.argmax(logits, dim=-1)
            A = mil_result.get('attention', None)
            fusion_stats = {
                'beta': beta if isinstance(beta, float) else beta.item(),
                'logits_he': logits_he,    # for per-path AUC
                'logits_ihc': logits_ihc,  # for per-path AUC
            }
        elif not self.use_mclc and not self.use_partial_align and not self.use_shared_rrt and not self.use_srp_fusion:
            # Standard path: all modalities through RRT → MIL (only when no other fusion is active)
            encoder_out = self.rrt_encoder(x_emb_list)
            if isinstance(encoder_out, tuple):
                x_fused, gate_stats = encoder_out
            else:
                x_fused = encoder_out
            fusion_stats = gate_stats

            mil_result = self.mil(x_fused)
            logits = mil_result['logits']
            Y_hat = mil_result['Y_hat']
            A = mil_result.get('attention', None)

        if return_features:
            return {
                'logits': logits,
                'prediction': Y_hat,
                'attention': A,
                'fused_features': x_fused,
                'embedded_features': x_emb_list,
                'fusion_stats': fusion_stats,
            }
        else:
            if return_modality_attns:
                return logits, Y_hat, A, None  # modality_atns待实现
            return logits, Y_hat, A, fusion_stats, aux_loss
    
    def get_modality_names(self):
        """获取模态名称列表"""
        return self.modality_list
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'num_modalities': self.num_modalities,
            'modality_list': self.modality_list,
            'input_dim': self.input_dim,
            'mlp_dim': self.mlp_dim,
            'num_classes': self.num_classes,
            'mil_type': self.mil_type,
            'fusion_type': self.rrt_encoder.layers[0].modality_fusion.__class__.__name__ \
                if self.num_modalities > 1 and hasattr(self.rrt_encoder.layers[0], 'modality_fusion') \
                and self.rrt_encoder.layers[0].modality_fusion else 'None'
        }


# 预配置模型工厂
def create_mm_rrt_abmil(num_modalities=2, modality_list=None,
                        fusion_type='self_attention', model_size='medium',
                        mil_type='abmil', use_gated=False, use_gated_fusion=False, **kwargs):
    """
    创建预配置的多模态RRT+ABMIL模型
    
    Args:
        num_modalities: 模态数量 (1-5)
        modality_list: 模态名称列表
        fusion_type: 融合类型
        model_size: 'small', 'medium', 'large'
        use_gated: 是否使用门控注意力(ABMIL)
        use_gated_fusion: 是否使用Gated Fusion (Gate + Residual + Dropout)
        **kwargs: 额外参数
    
    Returns:
        MM_RRT_ABMIL实例
    """
    configs = {
        'small': {
            'mlp_dim': 256,
            'region_num': 4,
            'n_layers': 1,
            'n_heads': 4,
            'abmil_hidden_dim': 64
        },
        'medium': {
            'mlp_dim': 512,
            'region_num': 8,
            'n_layers': 2,
            'n_heads': 8,
            'abmil_hidden_dim': 128
        },
        'large': {
            'mlp_dim': 768,
            'region_num': 16,
            'n_layers': 3,
            'n_heads': 12,
            'abmil_hidden_dim': 256
        }
    }
    
    config = configs.get(model_size, configs['medium'])
    
    if modality_list is None:
        # 默认模态名称
        default_names = ['RAW', 'ER', 'PR', 'HER2', 'KI67']
        modality_list = default_names[:num_modalities]
    
    return MM_RRT_ABMIL(
        num_modalities=num_modalities,
        modality_list=modality_list,
        fusion_type=fusion_type,
        mil_type=mil_type,
        use_gated=use_gated,
        use_gated_fusion=use_gated_fusion,
        **config,
        **kwargs
    )


# 针对不同模态数量的便捷函数
def create_dual_modal_rrt_abmil(modality_list=None, **kwargs):
    """创建双模态RRT+ABMIL"""
    if modality_list is None:
        modality_list = ['RAW', 'ER']
    return create_mm_rrt_abmil(num_modalities=2, modality_list=modality_list, **kwargs)


def create_triple_modal_rrt_abmil(modality_list=None, **kwargs):
    """创建三模态RRT+ABMIL"""
    if modality_list is None:
        modality_list = ['RAW', 'ER', 'PR']
    return create_mm_rrt_abmil(num_modalities=3, modality_list=modality_list, **kwargs)


def create_quad_modal_rrt_abmil(modality_list=None, **kwargs):
    """创建四模态RRT+ABMIL"""
    if modality_list is None:
        modality_list = ['RAW', 'ER', 'PR', 'HER2']
    return create_mm_rrt_abmil(num_modalities=4, modality_list=modality_list, **kwargs)


def create_penta_modal_rrt_abmil(modality_list=None, **kwargs):
    """创建五模态RRT+ABMIL"""
    if modality_list is None:
        modality_list = ['RAW', 'ER', 'PR', 'HER2', 'KI67']
    return create_mm_rrt_abmil(num_modalities=5, modality_list=modality_list, **kwargs)


if __name__ == "__main__":
    print("Testing MM_RRT_ABMIL...")
    
    # Test 1: 单模态 (向后兼容)
    print("\n1. Testing single modality...")
    model_single = MM_RRT_ABMIL(num_modalities=1, modality_list=['RAW'])
    x_single = torch.randn(2, 100, 768)
    logits, Y_hat, A = model_single(x_single)
    print(f"Input: {x_single.shape}")
    print(f"Output logits: {logits.shape}, predictions: {Y_hat.shape}")
    print(f"Model info: {model_single.get_model_info()}")
    
    # Test 2: 双模态
    print("\n2. Testing dual modality (RAW + ER)...")
    model_dual = create_dual_modal_rrt_abmil(fusion_type='self_attention')
    x_dual = [torch.randn(2, 100, 768), torch.randn(2, 100, 768)]
    logits, Y_hat, A = model_dual(x_dual)
    print(f"Input: {[x.shape for x in x_dual]}")
    print(f"Output logits: {logits.shape}, predictions: {Y_hat.shape}")
    print(f"Model info: {model_dual.get_model_info()}")
    
    # Test 3: 五模态
    print("\n3. Testing penta modality...")
    model_penta = create_penta_modal_rrt_abmil(fusion_type='self_attention')
    x_penta = [torch.randn(2, 100, 768) for _ in range(5)]
    logits, Y_hat, A = model_penta(x_penta)
    print(f"Input: {[x.shape for x in x_penta]}")
    print(f"Output logits: {logits.shape}, predictions: {Y_hat.shape}")
    print(f"Model info: {model_penta.get_model_info()}")
    
    print("\nAll tests passed!")
