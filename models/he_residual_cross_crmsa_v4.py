"""
HE-residual cross-CR-MSA **v4** — Q/K slide-internal common-direction removal.

Inherits everything from `HEResidualCrossCRMSAv3` (cosine cross-attention τ=0.2,
bias-free QKV, HE identity residual, independent routing, full-slot directed
HE→PR cross, dataset-prototype PR **value** centering V' = V − μ_PR).  The
single core change vs v3 is confined to the **attention score**:

    v3:  A = softmax( Q̂ K̂ᵀ / τ ),       Q̂ = Q/‖Q‖,  K̂ = K/‖K‖
    v4:  A = softmax( Q̂_c K̂_cᵀ / τ ),    Q̂_c = Q_c/‖Q_c‖,  K̂_c = K_c/‖K_c‖
         with Q_c = Q − Q̄,  K_c = K − K̄

where Q̄ / K̄ are the **per-slide, per-head means over valid routing tokens**
(Q̄ = mean over valid HE routes of Q, K̄ = mean over valid PR routes of K).
This removes the strong common direction shared by all routing tokens of a
slide, so the QK matching now compares *relative* region/slot offsets instead
of a slide-global direction that dominates the dot product and makes attention
near-uniform (the v3 failure mode).

V is **unchanged** from v3:  V'_{i,t} = V_{i,t} − μ_PR  (dataset prototype).
The centering of Q/K is applied ONLY to the attention score, never to V.

Numerical safety:
  * Q̄ / K̄ are means over VALID tokens only (invalid tokens excluded from the
    mean; they are masked out of the attention anyway and zeroed at the output).
  * count.clamp(min=1.0) guards the zero-valid-token mean; when a slide has no
    valid HE or PR routing token the cross output is forced to 0 (identity
    residual → returns original HE), identical to v3.
  * ε = 1e-6 in the L2-normalization denominator keeps near-zero-norm Q_c/K_c
    finite (no NaN).

Extra diagnostics (observational, no loss): Q raw pairwise cosine, Q centered
pairwise cosine, K raw pairwise cosine, K centered pairwise cosine (all mean
over heads), and the std of attention weights across PR keys — alongside all of
v3's diagnostics.
"""

import torch
import torch.nn.functional as F

from models.he_residual_cross_crmsa_v3 import HEResidualCrossCRMSAv3


class HEResidualCrossCRMSAv4(HEResidualCrossCRMSAv3):
    """v4: v3's dataset-prototype value centering + Q/K common-direction removal.

    Args match `HEResidualCrossCRMSAv3`; no new hyperparameters are introduced
    (τ, residual_scale, prototype_momentum all inherited).  The only behavioral
    delta is the per-slide-per-head Q/K mean removal inside `_cross_attention`.
    """

    @staticmethod
    def _mean_pairwise_cos_tokens(x, valid_mask, count):
        """Mean pairwise cosine over the valid tokens of x [B,h,T,d] → [B].

        Gram trick: for unit vectors x̂_t, Σ_{i≠j} x̂_i·x̂_j = ‖Σ_t x̂_t‖² − T_valid,
        with T_valid·(T_valid−1) ordered pairs.  Averaged over heads.
        """
        B = x.shape[0]
        T = x.shape[2]
        xn = F.normalize(x * valid_mask.float().view(B, 1, T, 1), dim=-1)
        s = xn.sum(dim=2)                                    # [B,h,d]
        pairwise = ((s ** 2).sum(-1) - count.view(B, 1)) / \
            (count * (count - 1)).clamp(min=1.0).view(B, 1)  # [B,h]
        return pairwise.mean(dim=1)                          # [B]

    # ------------------------------------------------------------------
    # Cross-attention (v4): v3 V-centering + Q/K common-direction removal.
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

        q_valid_f = q_valid.float()                               # [B, Q]
        k_valid_f = k_valid.float()                               # [B, K]
        q_count = q_valid_f.sum(-1).clamp(min=1.0)                # [B]
        k_count = k_valid_f.sum(-1).clamp(min=1.0)                # [B]

        # ── (1) per-slide per-head Q/K common-direction removal ──
        # Means over VALID routing tokens only (invalid tokens excluded).
        q_mean = (q * q_valid_f.view(B, 1, Q, 1)).sum(dim=2) / q_count.view(B, 1, 1)  # [B,h,d]
        k_mean = (k * k_valid_f.view(B, 1, K, 1)).sum(dim=2) / k_count.view(B, 1, 1)  # [B,h,d]
        q_c = q - q_mean.unsqueeze(2)                             # [B,h,Q,d]
        k_c = k - k_mean.unsqueeze(2)                             # [B,h,K,d]

        # ── (2) cosine cross-attention on CENTERED Q/K, fixed temperature ──
        eps = 1e-6
        q_hat = q_c / (q_c.norm(dim=-1, keepdim=True) + eps)
        k_hat = k_c / (k_c.norm(dim=-1, keepdim=True) + eps)
        cos = q_hat @ k_hat.transpose(-2, -1)                     # [B, h, Q, K]
        score = cos / self.tau                                    # [B, h, Q, K]

        attn = score.masked_fill(~k_valid.view(B, 1, 1, K), float('-inf'))
        has_valid_key = k_valid.any(dim=-1)                       # [B]
        attn_weights = torch.softmax(attn, dim=-1)
        attn_weights = torch.where(
            has_valid_key.view(B, 1, 1, 1), attn_weights,
            torch.zeros_like(attn_weights))

        # per-slide per-head mean over valid tokens (feeds EMA update + r_slide)
        v_mean = (v * k_valid_f.view(B, 1, K, 1)).sum(dim=2) / k_count.view(B, 1, 1)  # [B,h,d]

        # ── (3) dataset-prototype EMA update (train only, no-grad, detached) ──
        if self.training:
            with torch.no_grad():
                if has_valid_key.any():
                    m_i = v_mean[has_valid_key].mean(dim=0, keepdim=True)  # [1,h,d]
                    if not bool(self.prototype_initialized.item()):
                        self.mu_pr.copy_(m_i)
                        self.prototype_initialized.fill_(1.0)
                    else:
                        self.mu_pr.mul_(self.beta).add_(m_i, alpha=1.0 - self.beta)

        # ── (4) center value by the dataset prototype (v3, unchanged) ──
        v_tilde = v - self.mu_pr.unsqueeze(2)                     # [B,h,K,d] - [1,h,1,d]

        diag = None
        if return_diag:
            diag = self._diagnostics(attn_weights, score, v, v_tilde, v_full,
                                     q, k, q_c, k_c, q_valid, k_valid,
                                     q_count, k_count, v_mean)

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
    # Diagnostics = v3's seven + Q/K raw/centered cosine + attention std.
    # ------------------------------------------------------------------
    def _diagnostics(self, attn_weights, score, v, v_tilde, v_full,
                     q, k, q_c, k_c, q_valid, k_valid, q_count, k_count, v_mean):
        B = attn_weights.shape[0]
        valid_mask = k_valid
        count = k_count

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

        # 8. Q/K raw vs centered mean pairwise cosine (per-head, mean over heads)
        q_raw_cos = self._mean_pairwise_cos_tokens(q, q_valid, q_count)
        q_centered_cos = self._mean_pairwise_cos_tokens(q_c, q_valid, q_count)
        k_raw_cos = self._mean_pairwise_cos_tokens(k, k_valid, k_count)
        k_centered_cos = self._mean_pairwise_cos_tokens(k_c, k_valid, k_count)

        # 9. std of attention weight across PR keys (uniform → ~0, selective → ↑)
        attn_weight_std = attn_weights.std(dim=-1).mean(dim=(1, 2))  # [B]

        return {
            'entropy_norm': entropy_norm,
            'score_std': score_std,
            'value_diversity': value_diversity,
            'selective_ratio': selective_ratio,
            'mu_norm': mu_norm,
            'r_slide': r_slide,
            'r_region': r_region,
            'q_raw_cos': q_raw_cos,
            'q_centered_cos': q_centered_cos,
            'k_raw_cos': k_raw_cos,
            'k_centered_cos': k_centered_cos,
            'attn_weight_std': attn_weight_std,
        }
