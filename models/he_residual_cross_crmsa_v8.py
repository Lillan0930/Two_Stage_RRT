"""
HE-residual cross-CR-MSA **v8** — "HE region queries 读取 PR patch memory" 结构对照。

Inherits `HEResidualCrossCRMSAv3` (cosine τ=0.2, bias-free QKV, HE identity
residual, independent routing, EMA dataset-prototype value centering).  Two
memory modes, selected at construction, SAME parameter set / state-dict keys:

  * `routed` (control) — identical to v3: PR goes route_norm_pr → routing
    weighted pooling (phi_pr, G_PR·k=48 slots) → attn_norm_pr → Wk/Wv.
  * `patch`   (ablation) — the ONLY change: the routing weighted pooling between
    route_norm_pr and attn_norm_pr is replaced by **identity**, so K/V read ALL
    valid Z_PR patch tokens (K = N_PR, e.g. up to 2500) from the SAME forward,
    skipping Stage-2 PR combine/routing compression.

Preserved in both modes: HE routing (region queries Q, Wq, q_valid), cosine
attention with τ=0.2, Wk/Wv/Wout, HE dispatch, alpha (residual_scale)=0.1,
ABMIL, EMA prototype centering (β=0.99).  In patch mode:

  * PR path = route_norm_pr → identity → attn_norm_pr → Wk/Wv (no new params,
    no gate, no extra head, no positional bias, no new loss);
  * EMA statistic object = mean of the ACTUAL valid PR patch value tokens
    (v_mean over K=N_PR valid keys), train-only, once per training forward;
  * `phi_pr` is kept in the state dict for init/checkpoint compatibility but
    is NOT used by the patch-mode forward (recorded as unused);
  * unequal Q/K lengths, padding, empty PR (strict Z_HE identity) are handled
    by the inherited masked cosine attention (k_valid per patch token).
"""

import torch

from models.he_residual_cross_crmsa_v3 import HEResidualCrossCRMSAv3


class HEResidualCrossCRMSAv8(HEResidualCrossCRMSAv3):
    """v8: v3 + PR memory mode ('routed' == v3 | 'patch' == no PR routing
    pooling, K/V over all valid PR patch tokens)."""

    def __init__(self, dim=512, num_heads=8, region_num=4, crmsa_k=3,
                 drop_out=0.1, drop_path=0.0, epeg=False, epeg_k=15,
                 crmsa_mlp=False, ffn=False, ffn_act='gelu', mlp_ratio=4.,
                 region_size=0, min_region_num=0, min_region_ratio=0,
                 qkv_bias=True, residual_scale=0.1, disable_cross=False,
                 tau=0.2, prototype_momentum=0.99, pr_memory_mode='patch',
                 **kwargs):
        super().__init__(dim=dim, num_heads=num_heads, region_num=region_num,
                         crmsa_k=crmsa_k, drop_out=drop_out, drop_path=drop_path,
                         epeg=epeg, epeg_k=epeg_k, crmsa_mlp=crmsa_mlp, ffn=ffn,
                         ffn_act=ffn_act, mlp_ratio=mlp_ratio,
                         region_size=region_size, min_region_num=min_region_num,
                         min_region_ratio=min_region_ratio,
                         qkv_bias=False, residual_scale=residual_scale,
                         disable_cross=disable_cross, tau=tau,
                         prototype_momentum=prototype_momentum)
        assert pr_memory_mode in ('routed', 'patch'), \
            f"unknown pr_memory_mode {pr_memory_mode!r}"
        self.pr_memory_mode = pr_memory_mode

    # ------------------------------------------------------------------
    # Forward — routed mode is EXACTLY v3; patch mode swaps the PR routing
    # pooling for identity (route_norm_pr applied to all Z_PR tokens).
    # ------------------------------------------------------------------
    def forward(self, z_he, z_pr, valid_he=None, valid_pr=None):
        if self.disable_cross or float(self.residual_scale) == 0.0:
            return z_he
        if self.pr_memory_mode == 'routed':
            return super().forward(z_he, z_pr, valid_he=valid_he,
                                   valid_pr=valid_pr)

        # ── patch mode ──
        if z_he.dim() == 2:
            z_he = z_he.unsqueeze(0)
            z_pr = z_pr.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False
        B = z_he.shape[0]
        N_PR = z_pr.shape[1]

        # HE side unchanged: routing → region queries.
        routing_he, dmm_he, dw_he, valid_he_slots, H_he, W_he, add_he, rs_he = \
            self._route(z_he, self.phi_he, self.route_norm_he, valid_he)
        r_he = routing_he.reshape(B, -1, self.dim)
        q_valid = valid_he_slots.reshape(B, -1)

        # PR side: route_norm_pr → identity (NO routing pooling; phi_pr unused
        # in this mode) → _cross_attention applies attn_norm_pr → Wk/Wv over
        # ALL valid PR patch tokens.  The keys are physically compacted to the
        # valid tokens, so the actual K equals the valid PR patch count (§6).
        z_pr_n = self.route_norm_pr(z_pr)                       # [B, N_PR, D]
        if valid_pr is None:
            k_valid = torch.ones(B, N_PR, device=z_pr.device, dtype=torch.bool)
        else:
            k_valid = valid_pr.to(device=z_pr.device,
                                  dtype=torch.bool).reshape(B, N_PR)

        if not bool(k_valid.any()):
            # empty PR → strict Z_HE identity (no cross, no EMA update).
            return z_he.squeeze(0) if squeezed else z_he

        if B == 1:
            idx = k_valid[0].nonzero(as_tuple=False).squeeze(-1)
            r_pr = z_pr_n[:, idx, :]                            # [1, K_valid, D]
            k_valid = torch.ones(1, idx.numel(), device=z_pr.device,
                                 dtype=torch.bool)
        else:
            # Batched unequal lengths are not used by this pipeline (batch=1);
            # fall back to the masked (non-compacted) form for safety.
            r_pr = z_pr_n

        delta_routing = self._cross_attention(
            r_he, r_pr, q_valid, k_valid)                       # [B, Q, D]
        delta_routing = delta_routing.view(B, -1, self.crmsa_k, self.dim)

        delta_patch = self._dispatch(delta_routing, dmm_he, dw_he,
                                     rs_he, H_he, W_he, add_he)  # [B, N_HE, D]
        if valid_he is not None:
            vh = valid_he.to(device=z_he.device, dtype=z_he.dtype) \
                         .reshape(B, z_he.shape[1])
            delta_patch = delta_patch * vh.unsqueeze(-1)

        out = z_he + self.drop_path(self.residual_scale * delta_patch)
        if squeezed:
            out = out.squeeze(0)
        return out

    # ------------------------------------------------------------------
    # Diagnostics — patch-mode mirror of v3.diagnose (PR side without routing).
    # ------------------------------------------------------------------
    def diagnose(self, z_he, z_pr, valid_he=None, valid_pr=None):
        if self.pr_memory_mode == 'routed':
            return super().diagnose(z_he, z_pr, valid_he=valid_he,
                                    valid_pr=valid_pr)
        if z_he.dim() == 2:
            z_he = z_he.unsqueeze(0)
            z_pr = z_pr.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False
        B = z_he.shape[0]
        N_PR = z_pr.shape[1]

        routing_he, dmm_he, dw_he, valid_he_slots, H_he, W_he, add_he, rs_he = \
            self._route(z_he, self.phi_he, self.route_norm_he, valid_he)
        r_he = routing_he.reshape(B, -1, self.dim)
        q_valid = valid_he_slots.reshape(B, -1)

        z_pr_n = self.route_norm_pr(z_pr)
        if valid_pr is None:
            k_valid = torch.ones(B, N_PR, device=z_pr.device, dtype=torch.bool)
        else:
            k_valid = valid_pr.to(device=z_pr.device,
                                  dtype=torch.bool).reshape(B, N_PR)

        if not bool(k_valid.any()):
            return {
                'delta_patch': torch.zeros_like(z_he).squeeze(0) if squeezed
                else torch.zeros_like(z_he),
                'z_he': z_he.squeeze(0) if squeezed else z_he,
                'k_len': 0,
            }

        if B == 1:
            idx = k_valid[0].nonzero(as_tuple=False).squeeze(-1)
            r_pr = z_pr_n[:, idx, :]
            k_valid = torch.ones(1, idx.numel(), device=z_pr.device,
                                 dtype=torch.bool)
        else:
            r_pr = z_pr_n

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
            'k_len': int(k_valid.sum().item()) if B == 1 else None,
        }
        for kk, vv in diag.items():
            result[kk] = vv.squeeze(0) if squeezed else vv
        return result
