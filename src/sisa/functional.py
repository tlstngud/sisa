"""Functional form of SISA attention.

For each head the score of query ``i`` and key ``j <= i`` is::

    s_ij = q_i . k_j / sqrt(d_head)                      content term (no positional encoding)
         + lam_h * exp(g_i - g_j) * (c_i . b_j)          SSM term, c and b already rotated
         + (F_i - F_j)                                   forget-gate term

and ``o_i = sum_j softmax_j(s_ij) v_j``. Here ``g`` is the non-increasing
cumulative log-decay ``cumsum(-softplus(a))``, ``F`` the cumulative
log-sigmoid forget gate ``cumsum(logsigmoid(z))``, and ``c``, ``b`` are the
SSM read/write vectors rotated pairwise by the cumulative phase
``Phi = cumsum(theta)`` (see :func:`rotate_pairs`).

Shapes used throughout:

* ``q, k, v``: ``(B, H, L, d_head)``
* ``c, b``: ``(B, H, L, d_state)``, rotated
* ``g``: ``(B, H, L)`` float32
* ``lam``: ``(H,)`` float32, ``softplus(lambda_raw)``
* ``fgate``: ``(B, H, L)`` float32 or ``None``
"""

from __future__ import annotations

import contextlib
import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

__all__ = ["rotate_pairs", "sisa_attention_torch", "sisa_attention"]


def rotate_pairs(z: Tensor, phi: Tensor) -> Tensor:
    """Rotate interleaved channel pairs ``(0, 1), (2, 3), ...`` of ``z`` by ``phi``.

    Args:
        z: ``(..., d_state)``.
        phi: ``(..., d_state // 2)`` angles in radians.
    """
    z_even, z_odd = z[..., 0::2], z[..., 1::2]
    cos, sin = torch.cos(phi), torch.sin(phi)
    out = torch.stack([z_even * cos - z_odd * sin, z_even * sin + z_odd * cos], dim=-1)
    return out.flatten(-2)


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    """float32, or float64 when the inputs already are float64."""
    return torch.promote_types(dtype, torch.float32)


# Device types with autocast, resolved once at import so that the check below stays traceable by
# torch.compile (``torch.amp.is_autocast_available`` is not).
_AUTOCAST_DEVICE_TYPES = frozenset(
    d for d in ("cuda", "cpu", "xpu", "hpu", "mps") if torch.amp.is_autocast_available(d))


def _autocast_off(device_type: str):
    """A context that disables autocast on ``device_type`` (a no-op where autocast does not exist)."""
    if device_type in _AUTOCAST_DEVICE_TYPES:
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


def _attend_rows(q: Tensor, k: Tensor, v: Tensor, c: Tensor, b: Tensor,
                 g_q: Tensor, g_k: Tensor, lam: Tensor,
                 f_q: Tensor | None, f_k: Tensor | None, q_start: int) -> Tensor:
    """Exact attention of ``M`` query rows against ``N`` keys, in the inputs' dtype.

    Query row ``r`` sits at absolute position ``q_start + r``; key ``j`` at
    position ``j``. Keys after a query's position are masked. Autocast is
    disabled inside, so the matmuls run in the inputs' dtype (the callers
    pass float32 or float64) even when the caller has autocast enabled.

    Args:
        q, c: ``(B, H, M, .)`` queries and their rotated SSM read vectors.
        k, v, b: ``(B, H, N, .)`` keys, values and rotated SSM write vectors.
        g_q, f_q: ``(B, H, M)`` cumulative decay / forget values of the queries.
        g_k, f_k: ``(B, H, N)`` the same for the keys.
        lam: ``(H,)``.
    """
    with _autocast_off(q.device.type):
        m, n = q.shape[-2], k.shape[-2]
        scale = 1.0 / math.sqrt(q.shape[-1])
        pos_q = torch.arange(q_start, q_start + m, device=q.device)
        pos_k = torch.arange(n, device=q.device)
        causal = pos_k.unsqueeze(0) <= pos_q.unsqueeze(1)                 # (M, N)

        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        # Mask the decay exponent before exp: for j > i, g_i - g_j >= 0 grows with
        # the distance and would overflow to inf (and inf * 0 = nan).
        delta = (g_q.unsqueeze(-1) - g_k.unsqueeze(-2)).masked_fill(~causal, 0.0)
        ssm = torch.matmul(c, b.transpose(-1, -2))
        scores = scores + lam.view(1, -1, 1, 1) * torch.exp(delta) * ssm
        if f_q is not None:
            scores = scores + (f_q.unsqueeze(-1) - f_k.unsqueeze(-2))
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v)


def sisa_attention_torch(q: Tensor, k: Tensor, v: Tensor, c: Tensor, b: Tensor, g: Tensor,
                         lam: Tensor, fgate: Tensor | None = None, block_q: int = 256) -> Tensor:
    """Exact SISA attention in plain PyTorch (any device, any ``d_head``, any even ``d_state``).

    All inputs are cast to float32 (float64 inputs stay float64); scores, the
    softmax and ``P @ V`` are computed in that precision, also under autocast
    (which is disabled inside), and the output is cast back to ``q.dtype``.
    Queries are processed in blocks of ``block_q``
    rows, each against all keys up to the end of its block, so the score
    matrix held at once is ``(B, H, block_q, L)``. When gradients are needed,
    each block is checkpointed so the backward pass keeps the same bound.

    Returns:
        ``(B, H, L, d_head)`` in ``q.dtype``.
    """
    if q.dim() != 4:
        raise ValueError(f"q must be (B, H, L, d_head), got shape {tuple(q.shape)}")
    bsz, n_head, seq_len, _ = q.shape
    if k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError(f"q, k, v shapes do not match: {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")
    if c.shape[:3] != q.shape[:3] or b.shape != c.shape or c.shape[-1] % 2 != 0:
        raise ValueError(f"c and b must be (B, H, L, d_state) with even d_state, got {tuple(c.shape)}, {tuple(b.shape)}")
    if g.shape != q.shape[:3] or (fgate is not None and fgate.shape != g.shape):
        raise ValueError("g and fgate must be (B, H, L)")
    if lam.shape != (n_head,):
        raise ValueError(f"lam must be ({n_head},), got {tuple(lam.shape)}")
    if block_q <= 0:
        raise ValueError("block_q must be positive")

    out_dtype = q.dtype
    wd = _work_dtype(q.dtype)
    q, k, v, c, b = (t.to(wd) for t in (q, k, v, c, b))
    g, lam = g.to(wd), lam.to(wd)
    fgate = None if fgate is None else fgate.to(wd)

    use_ckpt = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (q, k, v, c, b, g, lam, fgate))
    outs = []
    for start in range(0, seq_len, block_q):
        end = min(start + block_q, seq_len)
        args = (q[:, :, start:end], k[:, :, :end], v[:, :, :end], c[:, :, start:end], b[:, :, :end],
                g[:, :, start:end], g[:, :, :end], lam,
                None if fgate is None else fgate[:, :, start:end],
                None if fgate is None else fgate[:, :, :end], start)
        if use_ckpt and seq_len > block_q:
            outs.append(checkpoint(_attend_rows, *args, use_reentrant=False))
        else:
            outs.append(_attend_rows(*args))
    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=2)
    return out.to(out_dtype)


def _triton_problem(q: Tensor, k: Tensor, v: Tensor, c: Tensor, g: Tensor,
                    fgate: Tensor | None) -> str | None:
    """Why the Triton attention kernel cannot take these inputs, or ``None``."""
    from .config import TRITON_D_HEAD, TRITON_D_STATE

    if q.device.type != "cuda":
        return f"inputs are on {q.device.type}, the Triton kernels need CUDA"
    if q.shape[-1] != TRITON_D_HEAD or c.shape[-1] != TRITON_D_STATE:
        return (f"the Triton kernels need d_head={TRITON_D_HEAD} and d_state={TRITON_D_STATE}, "
                f"got {q.shape[-1]} and {c.shape[-1]}")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"unsupported dtype {q.dtype} (bf16, fp16 or fp32)"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return f"q, k and v must share one dtype, got {q.dtype}, {k.dtype}, {v.dtype}"
    if g.dtype != torch.float32 or (fgate is not None and fgate.dtype != torch.float32):
        return (f"g and fgate must be float32, got {g.dtype} and "
                f"{None if fgate is None else fgate.dtype}")
    from . import backends

    ok, why = backends.triton_status(q.device)
    return None if ok else why


def sisa_attention(q: Tensor, k: Tensor, v: Tensor, c: Tensor, b: Tensor, g: Tensor, lam: Tensor,
                   fgate: Tensor | None = None, impl: str = "auto", block_q: int = 256) -> Tensor:
    """SISA attention with a selectable backend.

    Args:
        impl: ``"torch"`` (:func:`sisa_attention_torch`), ``"triton"`` (the
            fused Triton kernel; CUDA, ``d_head`` 64, ``d_state`` 32) or
            ``"auto"`` (Triton when possible, else torch). Forcing
            ``"triton"`` on inputs it cannot take raises ``ValueError``.
            ``"triton_fused"`` fuses the projections and therefore exists
            only at the layer level (:class:`sisa.SISAAttention`).
        block_q: Query block size of the torch backend.

    Returns:
        ``(B, H, L, d_head)`` in ``q.dtype``.
    """
    if impl == "auto":
        why = _triton_problem(q, k, v, c, g, fgate)
        impl = "triton" if why is None else "torch"
        if why is not None and q.device.type == "cuda":
            from . import backends

            key = (q.device.index, why)
            if key not in backends._WARNED:
                backends._WARNED.add(key)
                warnings.warn(f"SISA: using the torch attention implementation on CUDA because {why}",
                              stacklevel=2)
    if impl == "torch":
        return sisa_attention_torch(q, k, v, c, b, g, lam, fgate, block_q=block_q)
    if impl == "triton":
        why = _triton_problem(q, k, v, c, g, fgate)
        if why is not None:
            raise ValueError(f"impl='triton' is not available: {why}")
        from .kernels.attention import sisa_attention_triton

        return sisa_attention_triton(q, k, v, c, b, g, lam, fgate)
    if impl == "triton_fused":
        raise ValueError("impl='triton_fused' fuses the input projections and is only available through "
                         "SISAAttention / SISAConfig.attn_impl; use impl='triton' for prepared inputs")
    raise ValueError(f"unknown impl {impl!r}; expected 'auto', 'torch' or 'triton'")
