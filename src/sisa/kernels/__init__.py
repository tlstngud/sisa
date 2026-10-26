"""Triton kernels of SISA (NVIDIA GPUs, sm_80 or newer, Triton >= 3.3).

The submodules import Triton, so they are loaded lazily: ``import sisa.kernels`` works without Triton, and the
attributes below are resolved on first access. Check availability with :func:`sisa.backends.triton_status`.

``sisa_attention_triton(q, k, v, c, b, g, lam, fgate=None) -> o``  (:mod:`sisa.kernels.attention`)
    Exact SISA attention, forward and backward. q, k, v: ``(B, H, L, 64)`` bf16/fp16/fp32 (any strides with a
    contiguous last dimension); c, b: ``(B, H, L, 32)`` rotated C̃/B̃; g: ``(B, H, L)`` fp32 non-increasing cumulative
    log-decay; lam: ``(H,)`` fp32; fgate: ``(B, H, L)`` fp32 cumulative log-sigmoid forget gate or None.
    Returns ``(B, H, L, 64)`` in ``q.dtype`` (stored in ``(B, L, H, 64)`` memory layout, so
    ``o.transpose(1, 2).reshape(B, L, H * 64)`` is a view).

``sisa_fused(proj, z, lam, n_head) -> y``  (:mod:`sisa.kernels.fused_prep`)
    Fused layer core. proj: ``(B, L, NPAD)`` bf16/fp16 output of one linear layer whose weight is the row
    concatenation ``[W_Q, W_K, W_V, W_B, W_C, theta_proj, alpha_proj, zero padding]`` (see ``fused_layout`` and
    ``concat_fused_weights``); z: ``(B, L, H)`` fp32 forget logits; lam: ``(H,)`` fp32. Returns
    ``y`` of shape ``(B, L, H * 64)`` in ``proj.dtype`` (heads concatenated, ready for ``W_O``).

``fused_layout(n_head) -> dict``, ``concat_fused_weights(...) -> (W_cat, b_cat)``  (:mod:`sisa.kernels.fused_prep`)
    Column offsets of ``proj`` (keys ``Q, K, V, B, C, TH, A, NTOT, NPAD``) and a helper building ``W_cat, b_cat``.

``linear_cross_entropy(hidden, weight, labels, ignore_index=-100) -> loss``  (:mod:`sisa.kernels.linear_ce`)
    Chunked tied LM head + mean cross-entropy without materialising the logits (labels are not shifted here).
"""
from __future__ import annotations

import importlib

__all__ = ["sisa_attention_triton", "sisa_fused", "fused_layout", "concat_fused_weights", "linear_cross_entropy"]

_LAZY = {
    "sisa_attention_triton": "attention",
    "sisa_fused": "fused_prep",
    "fused_layout": "fused_prep",
    "concat_fused_weights": "fused_prep",
    "linear_cross_entropy": "linear_ce",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{mod}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
