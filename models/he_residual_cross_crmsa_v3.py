"""
HE-residual cross-CR-MSA **v3** — dataset-prototype PR value centering.

Inherits everything from `HEResidualCrossCRMSAv2` (cosine cross-attention τ=0.2,
bias-free QKV, HE identity residual, independent routing, full-slot directed
HE→PR cross, padding/mask handling, `disable_cross`/`residual_scale==0`
identity).  The single core change vs v2:

    v2 centering (per slide):   V'_{i,t} = V_{i,t} − mean_t(V_{i,t})
    v3 centering (prototype):   V'_{i,t} = V_{i,t} − μ_PR

where μ_PR is an **EMA running prototype of the per-slide mean value**, learned
ONLY from the training set PR (never updated on val/test).  If the value
decomposes as  V_{i,t} = C_dataset + D_i + E_{i,t}, then

    V'_{i,t} ≈ D_i + E_{i,t}

i.e. v3 removes the dataset-common PR component while *keeping* the slide-
specific global (D_i) and region-specific (E_{i,t}) information that v2's
per-slide centering deleted.

Key implementation properties:
  * `mu_pr` is a registered buffer [1, num_heads, head_dim] — **not** a
    nn.Parameter, **not** in the optimizer, checkpointed automatically.
  * `prototype_initialized` is a scalar buffer flag; the first valid train PR
    slide sets μ_PR = m_i, thereafter μ_PR ← β·μ_PR + (1−β)·m_i.
  * EMA update runs ONLY when `self.training` is True, under `torch.no_grad()`,
    on `m_i.detach()` (the buffer carries no gradient).  `V − μ_PR` still lets
    V backprop normally.
  * β (prototype momentum) = 0.99, fixed (not searched).  No data leakage:
    val/test forwards never mutate the buffer.
  * Extra diagnostics (observational, no loss): prototype magnitude `mu_norm`,
    slide-global residual ratio `r_slide`, region residual ratio `r_region`,
    alongside v2's `entropy_norm` / `score_std` / `value_diversity` /
    `selective_ratio`.
"""

import torch
import torch.nn.functional as F

from models.he_residual_cross_crmsa_v2 import HEResidualCrossCRMSAv2


class HEResidualCrossCRMSAv3(HEResidualCrossCRMSAv2):
    """v3: v2's cosine/bias-free cross, but dataset-prototype value centering.

    Args match `HEResidualCrossCRMSAv2`; `prototype_momentum` is the EMA decay
    β (default 0.99).  Everything else (routing, cross structure, residual) is
    inherited unchanged.
    """

    def __init__(self, dim=512, num_heads=8, region_num=4, crmsa_k=3,
                 drop_out=0.1, drop_path=0.0, epeg=False, epeg_k=15,
                 crmsa_mlp=False, ffn=False, ffn_act='gelu', mlp_ratio=4.,
                 region_size=0, min_region_num=0, min_region_ratio=0,
                 qkv_bias=True, residual_scale=0.1, disable_cross=False,
                 tau=0.2, prototype_momentum=0.99, **kwargs):
        super().__init__(dim=dim, num_heads=num_heads, region_num=region_num,
                         crmsa_k=crmsa_k, drop_out=drop_out, drop_path=drop_path,
                         epeg=epeg, epeg_k=epeg_k, crmsa_mlp=crmsa_mlp, ffn=ffn,
                         ffn_act=ffn_act, mlp_ratio=mlp_ratio,
                         region_size=region_size, min_region_num=min_region_num,
                         min_region_ratio=min_region_ratio,
                         qkv_bias=False, residual_scale=residual_scale,
                         disable_cross=disable_cross, tau=tau)

        self.beta = float(prototype_momentum)

        # Dataset-level PR value prototype [1, num_heads, head_dim], EMA-learned
        # from train PR only.  Buffer ⇒ checkpointed, never optimized.
        self.register_buffer(
            'mu_pr', torch.zeros(1, self.num_heads, self.head_dim))
        self.register_buffer('prototype_initialized', torch.zeros(1))

    # ------------------------------------------------------------------
    # Cross-attention (v3): cosine + dataset-prototype value centering.
    # ------------------------------------------------------------------
    def _cross_attention(self, r_he, r_pr, q_valid, k_valid, return_diag=False):
        B, Q, D = r_he.shape
        K = r_pr.shape[1]
        h = self.num_heads
        d = self.head_dim

        r_he_n = self.attn_norm_he(r_he)
        r_pr_n = self.attn_norm_pr(r_pr)
        q = self.w_q(r_he_n)                                      # [B, Q, D] bias-free
        k = self.w_k(r_pr_n)                                      # [B, K, D]
        v_full = self.w_v(r_pr_n)                                 # [B, K, D]

        q = q.view(B, Q, h, d).transpose(1, 2)                    # [B, h, Q, d]
        k = k.view(B, K, h, d).transpose(1, 2)                    # [B, h, K, d]
        v = v_full.view(B, K, h, d).transpose(1, 2)               # [B, h, K, d]

        # ── (1) cosine cross-attention with fixed temperature ──
        eps = 1e-6
        q_hat = q / (q.norm(dim=-1, keepdim=True) + eps)
        k_hat = k / (k.norm(dim=-1, keepdim=True) + eps)
        cos = q_hat @ k_hat.transpose(-2, -1)                     # [B, h, Q, K]
        score = cos / self.tau                                    # [B, h, Q, K]

        attn = score.masked_fill(~k_valid.view(B, 1, 1, K), float('-inf'))
        has_valid_key = k_valid.any(dim=-1)                       # [B]
        attn_weights = torch.softmax(attn, dim=-1)
        attn_weights = torch.where(
            has_valid_key.view(B, 1, 1, 1), attn_weights,
            torch.zeros_like(attn_weights))

        k_valid_f = k_valid.float()                               # [B, K]
        count = k_valid_f.sum(-1).clamp(min=1.0)                  # [B]

        # per-slide per-head mean over valid tokens (feeds EMA update + r_slide)
        v_mean = (v * k_valid_f.view(B, 1, K, 1)).sum(dim=2) / count.view(B, 1, 1)  # [B,h,d]

        # ── (2) dataset-prototype EMA update (train only, no-grad, detached) ──
        if self.training:
            with torch.no_grad():
                if has_valid_key.any():
                    m_i = v_mean[has_valid_key].mean(dim=0, keepdim=True)  # [1,h,d]
                    if not bool(self.prototype_initialized.item()):
                        self.mu_pr.copy_(m_i)
                        self.prototype_initialized.fill_(1.0)
                    else:
                        self.mu_pr.mul_(self.beta).add_(m_i, alpha=1.0 - self.beta)

        # ── (3) center value by the dataset prototype (NOT per-slide mean) ──
        v_tilde = v - self.mu_pr.unsqueeze(2)                     # [B,h,K,d] - [1,h,1,d]

        diag = None
        if return_diag:
            diag = self._diagnostics(attn_weights, score, v, v_tilde, v_full,
                                     k_valid, count, v_mean)

        attn_weights = self.attn_drop(attn_weights)
        out = (attn_weights @ v_tilde).transpose(1, 2).contiguous()  # [B, Q, h, d]
        out = out.view(B, Q, D)
        out = self.proj_drop(self.w_out(out))                     # [B, Q, D]

        out = out * q_valid.float().unsqueeze(-1)
        out = out * has_valid_key.float().view(B, 1, 1)

        if return_diag:
            return out, diag
        return out

    # ------------------------------------------------------------------
    # Diagnostics = v2's four + prototype magnitude / residual ratios.
    # ------------------------------------------------------------------
    def _diagnostics(self, attn_weights, score, v, v_tilde, v_full, k_valid,
                     count, v_mean):
        B = attn_weights.shape[0]
        valid_mask = k_valid

        # 1. normalized attention entropy
        A = attn_weights
        H = -(A * torch.log(A + 1e-12)).sum(dim=-1)
        log_K = torch.log(count.clamp(min=2.0))
        entropy_norm = (H / log_K.view(B, 1, 1)).mean(dim=(1, 2))

        # 2. std of attention score over valid keys
        score_masked = score.masked_fill(~valid_mask.view(B, 1, 1, -1), 0.0)
        score_mean = score_masked.sum(-1) / count.view(B, 1, 1)
        diff2 = ((score - score_mean.unsqueeze(-1)) ** 2) \
            .masked_fill(~valid_mask.view(B, 1, 1, -1), 0.0)
        score_std = (diff2.sum(-1) / count.view(B, 1, 1)).clamp(min=0).sqrt() \
            .mean(dim=(1, 2))

        # 3. mean pairwise cosine of PR routing values (full D) over valid tokens
        vn = F.normalize(v_full * valid_mask.float().unsqueeze(-1), dim=-1)
        s = vn.sum(dim=1)
        pairwise_sum = (s ** 2).sum(-1) - count
        value_diversity = pairwise_sum / (count * (count - 1)).clamp(min=1.0)

        # 4. selective-component ratio r = ||A·Ṽ|| / (||A·V|| + ε)
        O_centered = attn_weights @ v_tilde
        O_full = attn_weights @ v
        selective_ratio = O_centered.reshape(B, -1).norm(dim=-1) / \
            (O_full.reshape(B, -1).norm(dim=-1) + 1e-8)

        # 5. prototype magnitude ||μ_PR|| (same across slides)
        mu_norm = self.mu_pr.norm().expand(B)

        # 6. slide-global residual ratio r_slide = ||mean(V_i) − μ_PR|| / ||mean(V_i)||
        mu_pr_full = self.mu_pr.reshape(1, 1, self.dim)           # [1,1,D]
        v_mean_full = (v_full * valid_mask.float().unsqueeze(-1)).sum(dim=1) \
            / count.view(B, 1)                                    # [B,D]
        r_slide = (v_mean_full - mu_pr_full.squeeze(1)).norm(dim=-1) / \
            (v_mean_full.norm(dim=-1) + 1e-8)

        # 7. region residual ratio r_region = mean_t ||V_{i,t} − μ_PR|| / ||V_{i,t}||
        vcent = v_full - mu_pr_full                               # [B,K,D]
        r_num = (vcent * valid_mask.float().unsqueeze(-1)).norm(dim=-1)   # [B,K]
        r_den = (v_full * valid_mask.float().unsqueeze(-1)).norm(dim=-1) + 1e-8
        r_region = (r_num / r_den * valid_mask.float()).sum(dim=1) / count  # [B]

        return {
            'entropy_norm': entropy_norm,
            'score_std': score_std,
            'value_diversity': value_diversity,
            'selective_ratio': selective_ratio,
            'mu_norm': mu_norm,
            'r_slide': r_slide,
            'r_region': r_region,
        }

    # ------------------------------------------------------------------
    # Diagnostics entrypoint (returns ALL diagnostic scalars, generically).
    # ------------------------------------------------------------------
    def diagnose(self, z_he, z_pr, valid_he=None, valid_pr=None):
        if z_he.dim() == 2:
            z_he = z_he.unsqueeze(0)
            z_pr = z_pr.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False
        B = z_he.shape[0]

        routing_he, dmm_he, dw_he, valid_he_slots, H_he, W_he, add_he, rs_he = \
            self._route(z_he, self.phi_he, self.route_norm_he, valid_he)
        routing_pr, dmm_pr, dw_pr, valid_pr_slots, H_pr, W_pr, add_pr, rs_pr = \
            self._route(z_pr, self.phi_pr, self.route_norm_pr, valid_pr)

        r_he = routing_he.reshape(B, -1, self.dim)
        r_pr = routing_pr.reshape(B, -1, self.dim)
        q_valid = valid_he_slots.reshape(B, -1)
        k_valid = valid_pr_slots.reshape(B, -1)

        delta_routing, diag = self._cross_attention(
            r_he, r_pr, q_valid, k_valid, return_diag=True)
        delta_routing = delta_routing.view(B, -1, self.crmsa_k, self.dim)

        delta_patch = self._dispatch(delta_routing, dmm_he, dw_he,
                                     rs_he, H_he, W_he, add_he)
        if valid_he is not None:
            vh = valid_he.to(device=z_he.device, dtype=z_he.dtype) \
                         .reshape(B, z_he.shape[1])
            delta_patch = delta_patch * vh.unsqueeze(-1)

        result = {
            'delta_patch': delta_patch.squeeze(0) if squeezed else delta_patch,
            'z_he': z_he.squeeze(0) if squeezed else z_he,
        }
        for kk, vv in diag.items():
            result[kk] = vv.squeeze(0) if squeezed else vv
        return result
