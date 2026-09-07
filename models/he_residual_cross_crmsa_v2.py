"""
HE-residual cross-CR-MSA **v2** — directed HE→PR cross-attention with three
targeted fixes aimed at making the residual actually *depend on matched PR*.

Inherits everything from `HEResidualCrossCRMSA` (independent routing, full-slot
directed cross, HE-only residual write-back, padding/mask handling, identity on
`disable_cross`/`residual_scale==0`) and overrides only the cross-attention
computation plus the projection biases.  The three v2 changes:

    1. Cosine cross-attention  — Q and K are L2-normalized per head, then the
       similarity is scaled by a *fixed* temperature τ=0.2 (not learnable, not
       searched):
           Q̂ = Q/||Q||₂+ε,  K̂ = K/||K||₂+ε,  S = Q̂·K̂ᵀ/τ,  A = softmax(S)
       This removes the magnitude of the projected Q/K from the attention logits,
       so attention is driven purely by *direction*.

    2. PR value centering (core)  — for each slide and each head, the mean value
       over *valid* PR routing tokens is subtracted before the output is formed:
           μ_V = (1/|valid|) Σ_{j∈valid} V_j,   Ṽ_j = V_j − μ_V,
           O_i  = Σ_j A_ij Ṽ_j
       Key property: uniform attention ⇒ O ≈ 0, so the model *cannot* exploit a
       constant "average PR vector" — it must select specific PR tokens to produce
       a non-zero residual.

    3. Bias-free cross projections  — `W_q`, `W_k`, `W_v`, `W_out` all `bias=False`,
       so a zero-centered output (O≈0) maps to a zero residual (W_out(0)=0) instead
       of a learned constant offset.

Everything else (two independent RRTs, independent HE/PR routing, Q=HE, K/V=PR,
HE identity residual, residual_scale=0.1, ABMIL, CE-only, Train/Val-as-Test
protocol) is unchanged.  No contrastive / orthogonal / reconstruction /
matched-mismatched loss, no auxiliary PR classifier, no entropy regularization,
no gate, no FFN, no bidirectional cross, no new self-attention, no modality
dropout, no residual-scale search.

The module additionally exposes `diagnose()` which returns, alongside the patch
delta, four per-slide diagnostic scalars used in the v2 report:
    entropy_norm      normalized attention entropy H_norm ∈ [0, 1]
    score_std         std of the attention score S = cos/τ over valid keys
    value_diversity   mean pairwise cosine of PR routing values over valid tokens
    selective_ratio   ||A·Ṽ|| / (||A·V|| + ε)  (0 ⇒ uniform attention)
"""

import math
import torch
import torch.nn.functional as F

from models.he_residual_cross_crmsa import HEResidualCrossCRMSA


class HEResidualCrossCRMSAv2(HEResidualCrossCRMSA):
    """v2: cosine cross-attention (τ fixed) + PR value centering + bias-free QKV.

    Args match `HEResidualCrossCRMSA`; `tau` is the fixed temperature (default
    0.2).  `qkv_bias` is accepted for config compatibility but is ignored — v2
    always constructs bias-free cross projections.
    """

    def __init__(self, dim=512, num_heads=8, region_num=4, crmsa_k=3,
                 drop_out=0.1, drop_path=0.0, epeg=False, epeg_k=15,
                 crmsa_mlp=False, ffn=False, ffn_act='gelu', mlp_ratio=4.,
                 region_size=0, min_region_num=0, min_region_ratio=0,
                 qkv_bias=True, residual_scale=0.1, disable_cross=False,
                 tau=0.2, **kwargs):
        # Force bias-free projections (v2 core property); qkv_bias is ignored.
        super().__init__(dim=dim, num_heads=num_heads, region_num=region_num,
                         crmsa_k=crmsa_k, drop_out=drop_out, drop_path=drop_path,
                         epeg=epeg, epeg_k=epeg_k, crmsa_mlp=crmsa_mlp, ffn=ffn,
                         ffn_act=ffn_act, mlp_ratio=mlp_ratio,
                         region_size=region_size, min_region_num=min_region_num,
                         min_region_ratio=min_region_ratio,
                         qkv_bias=False, residual_scale=residual_scale,
                         disable_cross=disable_cross)
        self.tau = float(tau)

    # ------------------------------------------------------------------
    # Cross-attention (v2): cosine + temperature + PR value centering.
    # ------------------------------------------------------------------
    def _cross_attention(self, r_he, r_pr, q_valid, k_valid, return_diag=False):
        """Directed HE→PR cross-attention, v2.

        Returns `delta_routing` [B, Q, D]; if `return_diag`, also returns a dict
        of per-sample diagnostic scalars (each [B]).
        """
        B, Q, D = r_he.shape
        K = r_pr.shape[1]
        h = self.num_heads
        d = self.head_dim

        r_he_n = self.attn_norm_he(r_he)                          # [B, Q, D]
        r_pr_n = self.attn_norm_pr(r_pr)                          # [B, K, D]
        q = self.w_q(r_he_n)                                      # [B, Q, D] bias-free
        k = self.w_k(r_pr_n)                                      # [B, K, D]
        v_full = self.w_v(r_pr_n)                                 # [B, K, D]

        q = q.view(B, Q, h, d).transpose(1, 2)                    # [B, h, Q, d]
        k = k.view(B, K, h, d).transpose(1, 2)                    # [B, h, K, d]
        v = v_full.view(B, K, h, d).transpose(1, 2)               # [B, h, K, d]

        # ── (1) cosine cross-attention with fixed temperature ──
        eps = 1e-6
        q_hat = q / (q.norm(dim=-1, keepdim=True) + eps)          # [B, h, Q, d]
        k_hat = k / (k.norm(dim=-1, keepdim=True) + eps)          # [B, h, K, d]
        cos = q_hat @ k_hat.transpose(-2, -1)                     # [B, h, Q, K] ∈ [-1,1]
        score = cos / self.tau                                    # [B, h, Q, K]

        attn = score.masked_fill(~k_valid.view(B, 1, 1, K), float('-inf'))
        has_valid_key = k_valid.any(dim=-1)                       # [B]
        attn_weights = torch.softmax(attn, dim=-1)                # [B, h, Q, K]
        attn_weights = torch.where(
            has_valid_key.view(B, 1, 1, 1), attn_weights,
            torch.zeros_like(attn_weights))

        # ── (2) PR value centering: per slide, per head, over valid PR keys ──
        k_valid_f = k_valid.float()                               # [B, K]
        count = k_valid_f.sum(-1).clamp(min=1.0)                  # [B]
        v_mean = (v * k_valid_f.view(B, 1, K, 1)).sum(dim=2) / count.view(B, 1, 1)  # [B,h,d]
        v_tilde = v - v_mean.unsqueeze(2)                         # [B, h, K, d]

        diag = None
        if return_diag:
            diag = self._diagnostics(attn_weights, score, v, v_tilde, v_full,
                                     k_valid, count)

        attn_weights = self.attn_drop(attn_weights)

        out = (attn_weights @ v_tilde).transpose(1, 2).contiguous()  # [B, Q, h, d]
        out = out.view(B, Q, D)
        out = self.proj_drop(self.w_out(out))                     # [B, Q, D] bias-free

        # Re-zero invalid HE queries and no-valid-PR samples (bias-free ⇒ O=0 ⇒ 0).
        out = out * q_valid.float().unsqueeze(-1)
        out = out * has_valid_key.float().view(B, 1, 1)

        if return_diag:
            return out, diag
        return out

    # ------------------------------------------------------------------
    # Four per-slide diagnostic scalars (observational only, no loss).
    # ------------------------------------------------------------------
    @staticmethod
    def _diagnostics(attn_weights, score, v, v_tilde, v_full, k_valid, count):
        B = attn_weights.shape[0]
        valid_mask = k_valid                                         # [B, K]

        # 1. normalized attention entropy (over valid keys; mean over heads+queries)
        A = attn_weights                                             # [B,h,Q,K]
        H = -(A * torch.log(A + 1e-12)).sum(dim=-1)                 # [B,h,Q]
        log_K = torch.log(count.clamp(min=2.0))                     # [B]
        entropy_norm = (H / log_K.view(B, 1, 1)).mean(dim=(1, 2))   # [B]

        # 2. std of the attention score S = cos/τ over valid keys
        score_masked = score.masked_fill(~valid_mask.view(B, 1, 1, -1), 0.0)
        score_mean = score_masked.sum(-1) / count.view(B, 1, 1)     # [B,h,Q]
        diff2 = ((score - score_mean.unsqueeze(-1)) ** 2) \
            .masked_fill(~valid_mask.view(B, 1, 1, -1), 0.0)
        score_std = (diff2.sum(-1) / count.view(B, 1, 1)).clamp(min=0).sqrt() \
            .mean(dim=(1, 2))                                       # [B]

        # 3. mean pairwise cosine of PR routing values (full D) over valid tokens
        vn = F.normalize(v_full * valid_mask.float().unsqueeze(-1), dim=-1)  # [B,K,D]
        s = vn.sum(dim=1)                                           # [B,D]
        pairwise_sum = (s ** 2).sum(-1) - count                     # [B]
        value_diversity = pairwise_sum / (count * (count - 1)).clamp(min=1.0)

        # 4. selective-component ratio r = ||A·Ṽ|| / (||A·V|| + ε)
        O_centered = attn_weights @ v_tilde                          # [B,h,Q,d]
        O_full = attn_weights @ v                                    # [B,h,Q,d]
        num = O_centered.reshape(B, -1).norm(dim=-1)                 # [B]
        den = O_full.reshape(B, -1).norm(dim=-1) + 1e-8              # [B]
        selective_ratio = num / den                                  # [B]

        return {
            'entropy_norm': entropy_norm,
            'score_std': score_std,
            'value_diversity': value_diversity,
            'selective_ratio': selective_ratio,
        }

    # ------------------------------------------------------------------
    # Diagnostics entrypoint (mirrors forward but collects per-slide stats).
    # ------------------------------------------------------------------
    def diagnose(self, z_he, z_pr, valid_he=None, valid_pr=None):
        """Compute the patch delta and the 4 per-slide diagnostics (B=1 friendly).

        Returns a dict with `delta_patch`, `z_he` and scalar tensors
        `entropy_norm`, `score_std`, `value_diversity`, `selective_ratio` (each
        [B]).  Ignores `disable_cross`/`residual_scale==0` (always runs the cross)
        so it can be used to characterise the learned attention even when the
        forward path is identity.
        """
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
            r_he, r_pr, q_valid, k_valid, return_diag=True)         # [B, Q, D]
        delta_routing = delta_routing.view(B, -1, self.crmsa_k, self.dim)

        delta_patch = self._dispatch(delta_routing, dmm_he, dw_he,
                                     rs_he, H_he, W_he, add_he)     # [B, N_HE, D]
        if valid_he is not None:
            vh = valid_he.to(device=z_he.device, dtype=z_he.dtype) \
                         .reshape(B, z_he.shape[1])
            delta_patch = delta_patch * vh.unsqueeze(-1)

        if squeezed:
            return {
                'delta_patch': delta_patch.squeeze(0),
                'z_he': z_he.squeeze(0),
                'entropy_norm': diag['entropy_norm'].squeeze(0),
                'score_std': diag['score_std'].squeeze(0),
                'value_diversity': diag['value_diversity'].squeeze(0),
                'selective_ratio': diag['selective_ratio'].squeeze(0),
            }
        return {
            'delta_patch': delta_patch,
            'z_he': z_he,
            **{k: v for k, v in diag.items()},
        }
