"""
Cross-Staining Region MSA —— "染色即区域" 多染色融合模块（Two-stage R²T 的 Stage 2）。

思想:
    原始 R²T 的 CR-MSA 把一张切片的 K 个 region token 互相融合 (S(R))。
    本模块把它推广到染色维度: 把 n 种染色各自的 region token 视为
    "虚拟超级切片" 上的不同区域, 用同一个 MSA 完成跨染色融合。

等价性说明:
    计算上无需真的把 n 种染色平铺成 2D 巨网格。"平铺" 的计算内核只是
    "按染色分组 + 组内建模 + 组间融合"; 纯注意力对 token 的排布方式
    (1D 拼接 / 2D 平铺 / 3D 堆叠) 置换等变。因此直接在堆叠的 region
    token 集合 [B, nK, D] 上做 MSA, 与超级切片布局逐位等价。
    Stage 1 中每种染色独立过 R²T, 即对应超级切片上的组内 R-MSA。

数据流 (染色 m 的 patch 数为 N_m, region 数 K = region_num²):
    1. Z_m [B,N_m,D] → pad 成方形网格 → 切 K 个 region → 注意力池化
       → R_m [B,K,D]                       (区域摘要, 对应 CR-MSA 的 combine)
    2. R = concat(R_1..R_n) [B, nK, D]
    3. R̂ = R + DropPath( MSA( LN(R) ) )    (跨染色融合, 对应 CR-MSA 的 S(R))
       —— 纯 padding 产生的区域 token 会被 mask, 不吸收也不分发信息
    4. 拆回各染色, 广播回 patch 级, 以零初始化门控残差注入:
       z'_m = z_m + g · Broadcast(R̂_m),  g 初始为 0
       (训练起点等价于 "直接拼接所有染色" 基线, 融合能力随训练逐渐打开)
    5. 返回 concat(z'_1..z'_n) [B, ΣN_m, D] → 交给下游 ABMIL
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class CrossStainingRegionMSA(nn.Module):
    """
    "染色即区域" 跨染色融合模块。

    Args:
        dim:        特征维度 (512)
        num_heads:  跨染色 MSA 的注意力头数
        region_num: 每种染色切分的 region 边数, K = region_num²
        drop_out:   注意力 dropout
        drop_path:  融合残差上的随机深度
    """

    def __init__(self, dim=512, num_heads=8, region_num=4,
                 drop_out=0.1, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.region_num = region_num

        # 区域摘要池化 (所有染色共享): 区域内 patch → 1 个 region token
        self.region_pool = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

        # 跨染色融合: 标准 pre-LN MSA block (对应 CR-MSA 的 S(R) 步骤)
        self.norm_msa = nn.LayerNorm(dim)
        self.msa = nn.MultiheadAttention(
            dim, num_heads, dropout=drop_out, batch_first=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # 零初始化融合门控 (ReZero 风格): g=0 时整个模块为恒等映射
        self.fusion_gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # 网格工具: pad / 区域摘要 / 广播
    # ------------------------------------------------------------------

    def _pad_to_grid(self, z):
        """[B,N,D] → [B,H,W,D]: 先补零到 ⌈√N⌉² 方阵, 再补到 region_num 的倍数。"""
        B, N, D = z.shape
        gs = int(np.ceil(np.sqrt(N)))
        if gs * gs > N:
            z = F.pad(z, (0, 0, 0, gs * gs - N))
        grid = z.view(B, gs, gs, D)
        pad = (-gs) % self.region_num
        if pad:
            grid = F.pad(grid, (0, 0, 0, pad, 0, pad))
        return grid, gs

    def _extract_region_tokens(self, grid, gs, N):
        """[B,H,W,D] → region tokens [B,K,D] 和有效区域掩码 [K]。

        有效 = 区域内至少含一个真实 patch (r<gs 且 c<gs 且 r*gs+c<N);
        纯 padding 区域为 False, 后续 MSA 中会被屏蔽。
        """
        B, H, W, D = grid.shape
        G = self.region_num
        rs = H // G                            # region 边长
        K = G * G

        regions = (grid.view(B, G, rs, G, rs, D)
                       .permute(0, 1, 3, 2, 4, 5)
                       .reshape(B * K, rs * rs, D))
        attn = self.region_pool(regions).softmax(dim=1)      # [B*K, rs², 1]
        tokens = (regions * attn).sum(dim=1).view(B, K, D)

        rows = torch.arange(H, device=grid.device).view(H, 1)
        cols = torch.arange(W, device=grid.device).view(1, W)
        real_cell = (rows < gs) & (cols < gs) & ((rows * gs + cols) < N)
        valid = (real_cell.view(G, rs, G, rs)
                          .permute(0, 2, 1, 3)
                          .reshape(K, rs * rs)
                          .any(dim=1))                        # [K]
        return tokens, valid

    def _broadcast_tokens(self, tokens, H, W, N):
        """[B,K,D] → [B,N,D]: 每个 patch 取回所属 region 的 token, 裁掉 padding。"""
        B, K, D = tokens.shape
        G = self.region_num
        rs = H // G
        t = tokens.view(B, G, 1, G, 1, D).expand(B, G, rs, G, rs, D)
        return t.reshape(B, H * W, D)[:, :N]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z_list):
        """
        Args:
            z_list: list of [B, N_m, D], 每种染色的 R²T 输出 (长度 = 染色数 n)

        Returns:
            [B, ΣN_m, D]: 融合后的 patch 特征 (按染色拼接), 供下游 MIL
        """
        # 1. 每种染色: patch → region tokens (附有效区域掩码)
        tokens_list, valid_list, shapes = [], [], []
        for z in z_list:
            N = z.shape[1]
            grid, gs = self._pad_to_grid(z)
            tokens, valid = self._extract_region_tokens(grid, gs, N)
            tokens_list.append(tokens)
            valid_list.append(valid)
            shapes.append((N, grid.shape[1], grid.shape[2]))

        # 2. 拼接所有染色的 region token: "超级切片" 的完整区域集合
        R = torch.cat(tokens_list, dim=1)                    # [B, nK, D]
        valid = torch.cat(valid_list, dim=0)                 # [nK]

        # 3. 跨染色 MSA 融合 (幻影区域作为 key 被屏蔽)
        B = R.shape[0]
        key_padding_mask = ~valid.unsqueeze(0).expand(B, -1)  # [B, nK]
        R_norm = self.norm_msa(R)
        fused, _ = self.msa(R_norm, R_norm, R_norm,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)
        R = R + self.drop_path(fused)                        # [B, nK, D]

        # 4. 拆回各染色 → 广播回 patch → 门控残差注入
        K = self.region_num ** 2
        out_list = []
        start = 0
        for z, (N, H, W) in zip(z_list, shapes):
            t = R[:, start:start + K]                        # 该染色的融合 token
            start += K
            z = z + self.fusion_gate * self._broadcast_tokens(t, H, W, N)
            out_list.append(z)

        # 5. 按染色拼接, 交给下游 MIL
        return torch.cat(out_list, dim=1)                    # [B, ΣN_m, D]
