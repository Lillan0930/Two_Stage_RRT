"""
Pluggable MIL prediction heads — the unified head interface of the HE-anchored
multi-stain model (`models/he_aux_unified.py`).

Contract (fixed — the fusion module and the training loop depend only on this):

    head = build_mil_head(
        name=mil_cfg["name"],
        input_dim=feature_dim,
        num_classes=num_classes,
        **mil_cfg.get("kwargs", {}),
    )

    result = head(tokens=fused, mask=valid_he)
    # required: result["logits"]      -> [B, num_classes]
    # optional: result["embedding"], result["attention"]

Design notes
------------
* `MIL_REGISTRY` (`models/mil_registry.py`) stays the single source of truth for
  *which* MILs exist.  `build_mil_head` looks the name up there and raises on an
  unknown name — it never silently falls back to ABMIL.
* Every registered MIL is wrapped in a **head adapter**.  The adapter is the only
  place that knows a specific MIL's internals; `MILHeadAdapter` normalizes the
  return value to the contract above.  Adding TransMIL later therefore means
  *writing one adapter and registering it* — no change to the RRT encoders, the
  cross-stain fusion, or the training main loop.
* `mask=None` ⇒ the adapter is a pure pass-through to the wrapped module's own
  `forward`, so legacy numerics (including train-mode dropout RNG order) are
  preserved bit-for-bit.  Passing a mask takes the adapter's masked path.
* The adapters add **no** pooling and **no** classifier of their own: they reuse
  the wrapped module's `attention` / `classifier` submodules verbatim, so
  historical MIL weights load unchanged and there is exactly one pooling +
  one classifier per head.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mil_registry import MIL_REGISTRY


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _check_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Validate the token tensor; accept [B, N, D] only (batch dim explicit)."""
    if not torch.is_tensor(tokens):
        raise TypeError(
            f"MIL head expects a tensor of tokens, got {type(tokens).__name__}")
    if tokens.dim() == 2:
        tokens = tokens.unsqueeze(0)
    if tokens.dim() != 3:
        raise ValueError(
            f"MIL head expects tokens of shape [B, N, D], got {tuple(tokens.shape)}")
    return tokens


def _check_mask(mask: Optional[torch.Tensor], tokens: torch.Tensor):
    """Validate / normalize the token-validity mask to [B, N] bool (or None)."""
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        raise TypeError(f"mask must be a tensor or None, got {type(mask).__name__}")
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.dim() != 2:
        raise ValueError(
            f"mask must be [B, N] (or [N]), got {tuple(mask.shape)}")
    if mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match tokens "
            f"{tuple(tokens.shape[:2])}")
    return mask.to(device=tokens.device, dtype=torch.bool)


def _masked_softmax_attention(attn_logits: torch.Tensor, mask: torch.Tensor):
    """[B, N] logits + [B, N] bool → attention weights [B, N] over valid tokens.

    An all-True mask is numerically a no-op (masked_fill with an all-False mask
    returns the input unchanged), so this reduces to plain softmax-over-N.
    Samples with no valid token get all-zero attention instead of NaN.
    """
    attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
    has_valid = mask.any(dim=1, keepdim=True)                     # [B, 1]
    attn = torch.softmax(attn_logits, dim=1)
    return torch.where(has_valid, attn, torch.zeros_like(attn))


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------
class MILHeadAdapter(nn.Module):
    """Base class for MIL head adapters.

    Subclasses wrap one registered MIL module (`self.module`) and implement
    `forward(tokens, mask=None) -> dict` with at least a `logits` entry.
    """

    def __init__(self, module: nn.Module, name: str):
        super().__init__()
        self.name = name
        self.module = module

    def forward(self, tokens, mask=None):  # pragma: no cover - interface only
        raise NotImplementedError(
            f"MIL head adapter for {self.name!r} must implement forward(tokens, mask)")

    def _legacy_pass_through(self, tokens):
        """Unmasked path: delegate to the wrapped module's own forward.

        Keeps the historical computation (and, in train mode, its dropout RNG
        draw order) exactly as before the adapter existed.
        """
        out = self.module(tokens)
        if not isinstance(out, dict) or 'logits' not in out:
            raise TypeError(
                f"{type(self.module).__name__}.forward must return a dict with a "
                f"'logits' entry, got {type(out).__name__}")
        return {
            'logits': out['logits'],
            'attention': out.get('attention'),
            'embedding': out.get('embedding'),
            'Y_prob': out.get('Y_prob'),
            'Y_hat': out.get('Y_hat'),
        }

    def extra_repr(self):
        return f"name={self.name!r}, module={type(self.module).__name__}"


class ABMILHead(MILHeadAdapter):
    """Adapter for `abmil` (`models/abmil.py::AttentionMIL`).

    Legacy path (mask=None) is the wrapped module's own forward.
    Masked path reproduces the same computation batched, with the attention
    softmax restricted to valid tokens:

        A = softmax(mask(attention(x)))          # over valid tokens only
        Z = Σ_n A_n · x_n                        # invalid tokens contribute 0
        logits = classifier(Z)

    Same `attention` / `classifier` submodules as the legacy module — no extra
    pooling or classifier is introduced.
    """

    def forward(self, tokens, mask=None):
        if mask is None:
            return self._legacy_pass_through(tokens)

        x = _check_tokens(tokens)
        mask = _check_mask(mask, x)

        attn_logits = self.module.attention(x).squeeze(-1)         # [B, N]
        attn = _masked_softmax_attention(attn_logits, mask)         # [B, N]

        x_valid = x * mask.unsqueeze(-1).to(x.dtype)                # zero invalid
        z = torch.einsum('bn,bnd->bd', attn, x_valid)               # [B, D]
        logits = self.module.classifier(z)                          # [B, C]

        return {
            'logits': logits,
            'attention': attn,
            'embedding': z,
            'Y_prob': F.softmax(logits, dim=-1),
            'Y_hat': torch.argmax(logits, dim=-1),
        }


class GatedABMILHead(MILHeadAdapter):
    """Adapter for `gated_abmil` (`models/abmil.py::GatedAttentionMIL`).

    Identical contract; the attention logits use the gated
    `w(V(x) ⊙ U(x))` form of the wrapped module.
    """

    def forward(self, tokens, mask=None):
        if mask is None:
            return self._legacy_pass_through(tokens)

        x = _check_tokens(tokens)
        mask = _check_mask(mask, x)

        a_v = self.module.attention_V(x)
        a_u = self.module.attention_U(x)
        attn_logits = self.module.attention_w(a_v * a_u).squeeze(-1)  # [B, N]
        attn = _masked_softmax_attention(attn_logits, mask)

        x_valid = x * mask.unsqueeze(-1).to(x.dtype)
        z = torch.einsum('bn,bnd->bd', attn, x_valid)
        logits = self.module.classifier(z)

        return {
            'logits': logits,
            'attention': attn,
            'embedding': z,
            'Y_prob': F.softmax(logits, dim=-1),
            'Y_hat': torch.argmax(logits, dim=-1),
        }


# name → adapter class.  A MIL registered in MIL_REGISTRY without an entry here
# is an explicit error (never a silent fallback).
_HEAD_ADAPTERS = {
    'abmil': ABMILHead,
    'gated_abmil': GatedABMILHead,
}


def register_head_adapter(name: str, adapter_cls: type):
    """Register the head adapter for a MIL name.

    Called next to the MIL's own `@register_mil(...)` when adding a new MIL
    (e.g. TransMIL): implement `MILHeadAdapter` and register it here.  Nothing
    in the RRT encoders, the fusion path or the training loop changes.
    """
    if not (isinstance(adapter_cls, type) and issubclass(adapter_cls, MILHeadAdapter)):
        raise TypeError(
            f"head adapter for {name!r} must subclass MILHeadAdapter, "
            f"got {adapter_cls!r}")
    _HEAD_ADAPTERS[name] = adapter_cls


def available_heads() -> list:
    """Names that can be built right now (registered MIL ∩ has an adapter)."""
    return sorted(n for n in MIL_REGISTRY.list_available() if n in _HEAD_ADAPTERS)


def build_mil_head(name: str, input_dim: int, num_classes: int, **kwargs):
    """Build a MIL head with the unified `(tokens, mask)` interface.

    Args:
        name:        registered MIL name (registered in `MIL_REGISTRY` **and**
                     having a head adapter here) — unknown names raise.
        input_dim:   token feature dimension (mlp_dim of the fusion module)
        num_classes: number of output classes
        **kwargs:    forwarded verbatim to the MIL constructor
                     (e.g. `hidden_dim`, `dropout_rate` for `abmil`)

    Raises:
        ValueError: unknown MIL name, or a registered MIL without an adapter.
        TypeError:  constructor arity mismatch (unknown kwarg for that MIL).
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"mil_cfg['name'] must be a non-empty string, got {name!r}")

    if not MIL_REGISTRY.is_registered(name):
        raise ValueError(
            f"Unknown MIL type {name!r}. Registered MILs: "
            f"{MIL_REGISTRY.list_available()}. (No silent fallback to 'abmil'.)")

    if name not in _HEAD_ADAPTERS:
        raise ValueError(
            f"MIL {name!r} is registered but has no head adapter. Add one with "
            f"`register_head_adapter({name!r}, YourAdapter)` in models/mil_heads.py. "
            f"MILs with adapters: {available_heads()}")

    mil_cls = MIL_REGISTRY.get(name)
    try:
        module = mil_cls(input_dim=input_dim, num_classes=num_classes, **kwargs)
    except TypeError as e:
        raise TypeError(
            f"Cannot build MIL {name!r}(input_dim={input_dim}, "
            f"num_classes={num_classes}, **{sorted(kwargs)}): {e}") from e

    return _HEAD_ADAPTERS[name](module, name)


def mil_head_config(mil_cfg: dict, input_dim: int, num_classes: int) -> dict:
    """Normalize a `mil_cfg` block into the exact kwargs `build_mil_head` uses.

    Accepted form (checkpoint / config round-trippable)::

        {"name": "abmil", "kwargs": {"hidden_dim": 256, "dropout_rate": 0.25}}
    """
    if not isinstance(mil_cfg, dict) or 'name' not in mil_cfg:
        raise ValueError(
            f"mil_cfg must be a dict with a 'name' key, got {mil_cfg!r}")
    kwargs = dict(mil_cfg.get('kwargs') or {})
    return {
        'name': mil_cfg['name'],
        'input_dim': int(input_dim),
        'num_classes': int(num_classes),
        **kwargs,
    }
