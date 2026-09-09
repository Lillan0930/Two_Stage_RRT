"""
HE-residual cross-CR-MSA **v5** — training-time auxiliary classification
supervision on the *actual* PR value memory (merged `V_tilde`).

Inherits everything from `HEResidualCrossCRMSAv3` (cosine τ=0.2 cross-attention,
bias-free QKV, dataset-prototype PR value centering `V'=V−μ_PR`, HE identity
residual, independent routing, full-slot directed HE→PR cross, padding/mask
handling, `disable_cross`/`residual_scale==0` identity, EMA prototype μ_PR).

The **single core addition** vs v3 is a lightweight auxiliary ABMIL head that
reads, on the *same* forward pass and from the *same* gradient-carrying value
tensor, the merged-heads value memory

    V_tilde_merged = merge_heads(V − μ_PR)                     [B, K, D]

i.e. the exact `V_tilde = V − μ_PR` that the main cross-attention multiplies by
`A` (`A · V_tilde`), merged over heads.  It applies a small Tanh-attention MIL
over the valid PR routing tokens only, then a small classifier, producing
`pr_value_logits` [B, C] that receive slide/bag-level supervision during
training:

    L = CE(fused_logits, y) + β·CE(pr_value_logits, y)         (β = 0.1)

Design constraints (see §1–§3 of the task spec):

  * **No detach** — `V_tilde_merged` carries gradient back to `w_v`,
    `attn_norm_pr`, `phi_pr` and the PR RRT, so the aux loss *does* update the
    PR value path.  No separate value projection is built; the head reads the
    very `v_full = w_v(attn_norm_pr(r_pr))` used by the main path.
  * **No per-token slide labelling** — the aux MIL aggregates first, then a
    slide/bag-level CE is applied to the aggregated logits.
  * **Aux head is a registered submodule** — `self.aux_attn` +
    `self.aux_classifier` — so `MM_RRT_ABMIL.apply(initialize_weights)` and
    `optimizer` pick them up automatically.  The EMA prototype update still runs
    exactly once per *training* forward (in `_cross_attention`), never in eval,
    and the aux head merely reads `V_tilde` (never re-runs routing / cross / EMA).
  * **Inference unchanged** — `forward` returns the fused `Z_HE + 0.1·Δ` exactly
    as v3; the aux logits are returned *alongside* (in `aux_dict`) purely for the
    trainer / evaluator, never added to the final prediction.
  * **Padding never enters the aux MIL** — invalid PR tokens are masked to −inf
    before the aux softmax, and fully-empty PR slides are flagged via
    `pr_value_has_valid` so the trainer can skip their auxiliary CE.
"""

import torch
import torch.nn as nn

from models.he_residual_cross_crmsa_v3 import HEResidualCrossCRMSAv3


class HEResidualCrossCRMSAv5(HEResidualCrossCRMSAv3):
    """v5: v3's cross + training-time auxiliary ABMIL on merged PR value memory.

    Args match `HEResidualCrossCRMSAv3` plus:
        num_classes:    auxiliary classifier output dim (default 2).
        aux_hidden_dim: hidden dim of the lightweight aux attention/classifier
                        MLP (default 64).
        aux_dropout:    dropout of the aux classifier (default 0.1).
    """

    def __init__(self, dim=512, num_heads=8, region_num=4, crmsa_k=3,
                 drop_out=0.1, drop_path=0.0, epeg=False, epeg_k=15,
                 crmsa_mlp=False, ffn=False, ffn_act='gelu', mlp_ratio=4.,
                 region_size=0, min_region_num=0, min_region_ratio=0,
                 qkv_bias=True, residual_scale=0.1, disable_cross=False,
                 tau=0.2, prototype_momentum=0.99,
                 num_classes=2, aux_hidden_dim=64, aux_dropout=0.1, **kwargs):
        super().__init__(dim=dim, num_heads=num_heads, region_num=region_num,
                         crmsa_k=crmsa_k, drop_out=drop_out, drop_path=drop_path,
                         epeg=epeg, epeg_k=epeg_k, crmsa_mlp=crmsa_mlp, ffn=ffn,
                         ffn_act=ffn_act, mlp_ratio=mlp_ratio,
                         region_size=region_size, min_region_num=min_region_num,
                         min_region_ratio=min_region_ratio,
                         qkv_bias=False, residual_scale=residual_scale,
                         disable_cross=disable_cross, tau=tau,
                         prototype_momentum=prototype_momentum)
        self.num_classes = num_classes

        # ── Lightweight auxiliary ABMIL head over the merged PR value memory ──
        # aux_attn: V_tilde_merged [B,K,D] → score [B,K,1] (Tanh-attention)
        # aux_classifier: pooled [B,D] → logits [B,C]
        self.aux_attn = nn.Sequential(
            nn.Linear(dim, aux_hidden_dim),
            nn.Tanh(),
            nn.Linear(aux_hidden_dim, 1),
        )
        self.aux_classifier = nn.Sequential(
            nn.Linear(dim, aux_hidden_dim),
            nn.ReLU(),
            nn.Dropout(aux_dropout),
            nn.Linear(aux_hidden_dim, num_classes),
        )
        # Match the rest of the model's xavier init (the factory also re-inits
        # via initialize_weights; this keeps standalone use well-behaved).
        for _mod in (self.aux_attn, self.aux_classifier):
            for _layer in _mod:
                if isinstance(_layer, nn.Linear):
                    nn.init.xavier_normal_(_layer.weight)
                    if _layer.bias is not None:
                        nn.init.constant_(_layer.bias, 0.0)

    # ------------------------------------------------------------------
    # Cross-attention (v5): identical math to v3, but also returns the
    # merged-heads value memory + key validity for the aux head.
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
        # Runs exactly once per training forward; the aux head does NOT re-run
        # routing/cross/EMA — it only reads v_tilde below.
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
        v_tilde = v - self.mu_pr.unsqueeze(2)                     # [B,h,K,d]

        # Merge heads → the exact V_tilde read by the main cross, [B,K,D].
        # This is the SAME gradient-carrying tensor (view/reshape only), so the
        # aux head backprops into w_v / attn_norm_pr / phi_pr / PR RRT.
        v_tilde_merged = v_tilde.transpose(1, 2).reshape(B, K, D)

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

        aux_input = (v_tilde_merged, k_valid)
        if return_diag:
            return out, aux_input, diag
        return out, aux_input

    # ------------------------------------------------------------------
    # Auxiliary ABMIL over the merged PR value memory (train-time only use).
    # ------------------------------------------------------------------
    def _aux_forward(self, v_tilde_merged, k_valid):
        B = v_tilde_merged.shape[0]
        has_valid = k_valid.any(dim=-1)                           # [B]

        attn_logits = self.aux_attn(v_tilde_merged)               # [B, K, 1]
        attn_logits = attn_logits.masked_fill(
            ~k_valid.unsqueeze(-1), float('-inf'))
        attn = torch.softmax(attn_logits, dim=1)                  # [B, K, 1]
        attn = torch.where(
            has_valid.view(B, 1, 1), attn, torch.zeros_like(attn))

        pooled = (attn * v_tilde_merged).sum(dim=1)               # [B, D]
        logits = self.aux_classifier(pooled)                      # [B, C]
        return logits, has_valid

    # ------------------------------------------------------------------
    # forward: v3 output + auxiliary logits (returned alongside, not fused).
    # ------------------------------------------------------------------
    def forward(self, z_he, z_pr, valid_he=None, valid_pr=None):
        # identity fallback (identical to v3 / base)
        if self.disable_cross or float(self.residual_scale) == 0.0:
            return z_he, None

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

        delta_routing, aux_input = self._cross_attention(
            r_he, r_pr, q_valid, k_valid)
        delta_routing = delta_routing.view(B, -1, self.crmsa_k, self.dim)

        delta_patch = self._dispatch(delta_routing, dmm_he, dw_he,
                                     rs_he, H_he, W_he, add_he)
        if valid_he is not None:
            vh = valid_he.to(device=z_he.device, dtype=z_he.dtype) \
                         .reshape(B, z_he.shape[1])
            delta_patch = delta_patch * vh.unsqueeze(-1)

        out = z_he + self.drop_path(self.residual_scale * delta_patch)
        if squeezed:
            out = out.squeeze(0)

        v_tilde_merged, k_valid_aux = aux_input
        pr_value_logits, pr_value_has_valid = self._aux_forward(
            v_tilde_merged, k_valid_aux)

        return out, {
            'pr_value_logits': pr_value_logits,      # [B, C]
            'pr_value_has_valid': pr_value_has_valid,  # [B] bool
        }

    # ------------------------------------------------------------------
    # Diagnostics entrypoint: identical to v3 (7 scalars) — ignores aux head.
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

        delta_routing, _aux_input, diag = self._cross_attention(
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
