"""Chunked linear + cross-entropy for a tied LM head, without materialising the full logits.

``linear_cross_entropy(hidden, weight, labels)`` returns the same scalar loss as
``F.cross_entropy(F.linear(hidden, weight).float().view(-1, V), labels.view(-1), ignore_index=-100)`` (mean over
non-ignored rows) and gradients for ``hidden`` and ``weight``. The approach (gradients computed during the forward pass,
chunk by chunk) follows Liger-Kernel's fused linear cross-entropy.

How it works
------------
One autograd function. Forward walks the tokens in chunks (by default balanced chunks under a 1 GiB logits buffer)
and computes the gradients right away:

  1. ``logits_c = x_c @ W_padᵀ`` (cuBLAS, compute-dtype output). W is padded with zero rows to a multiple of 128
     (50,277 -> 50,304), which keeps cuBLAS on aligned kernels; the padding is never stored in a parameter.
  2. ``_lmce_row_kernel`` (Triton, one program per row, in place over the chunk buffer):
       pass 1: lane-wise online max / sum-exp over the V real columns (padded columns masked to -inf) -> lse, loss row
       pass 2: G = (exp(logit - lse) - onehot) · (g / N_valid), rounded once to the compute dtype and written over
               the logits (padded columns 0)
  3. ``dx_c = G_c @ W_pad`` (fp32 output).
  4. ``dW += G_cᵀ @ x_c`` over all padded rows, fp32 accumulation across chunks.

Backward multiplies the stored dx and dW by the incoming gradient (two passes). With ``grad_scale=g`` promised up front,
G is pre-scaled in forward and backward only performs a device-side check of the promise.

Numerics
--------
 - Logits: same formula as autocast (compute-dtype inputs, fp32 accumulation, RNE output). The loss matches the
   eager formula to about one fp32 ulp.
 - lse / loss: fp32 lane-wise online reduction (a different order than torch), tl.exp / tl.log.
 - G is rounded to the compute dtype at the same point as autocast. Without ``grad_scale`` the 1/N factor is applied
   in the kernel and the incoming gradient g afterwards in fp32; when 1/g is a power of two this is bit-identical to
   pre-scaling, otherwise the two differ by one rounding of G (same accuracy).
 - dx and dW keep fp32 GEMM outputs (one rounding fewer than the eager path).
 - NaN/inf propagate like the eager formula (loss NaN): the online max uses propagate_nan=ALL, otherwise a NaN in a
   logit column could be dropped silently.
 - Deterministic (no atomics).

Requirements: CUDA tensors and a bf16/fp16 compute dtype (autocast on, or low-precision tensors). Otherwise the eager
formula is used. ``torch.mm(..., out_dtype=)`` (torch >= 2.8) gives fp32 GEMM outputs; older torch falls back to
compute-dtype outputs with fp32 accumulation. No host synchronisation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = ["linear_cross_entropy", "reference_linear_cross_entropy"]

_HAS_OUT_DTYPE = hasattr(torch.ops.aten.mm, "dtype") and hasattr(torch.ops.aten.addmm, "dtype_out")

# Row-kernel launch configuration (vocabulary block, warps) and the logits-buffer budget per chunk.
_BLOCK_V = 8192
_WARPS = 16
_CHUNK_MB = 1024
_VPAD = 128


@triton.jit
def _lmce_row_kernel(L, stride_l, LAB, LOSS, INV_N, gscale, V, Vp, ignore_index,
                     COMPUTE_GRAD: tl.constexpr, BLOCK_V: tl.constexpr):
    """One program = one token row of the (T, Vp) logits buffer (bf16/fp16). Writes loss_row (fp32) and, if
    COMPUTE_GRAD, overwrites the row with G = (softmax - onehot)·(inv_n·gscale) in the buffer dtype (padded cols 0)."""
    row = tl.program_id(0).to(tl.int64)
    ptr = L + row * stride_l
    lab = tl.load(LAB + row)
    valid = lab != ignore_index
    safe_lab = tl.where(valid, lab, 0)
    offs = tl.arange(0, BLOCK_V)
    # pass 1: lane-wise online max / sum-exp over the V real columns (fp32), one cross-lane reduction at the end
    # (no block-wide reduction + barrier per iteration).
    mv = tl.full((BLOCK_V,), float('-inf'), tl.float32)
    sv = tl.zeros((BLOCK_V,), tl.float32)
    for start in range(0, V, BLOCK_V):
        cols = start + offs
        x = tl.load(ptr + cols, mask=cols < V, other=float('-inf')).to(tl.float32)
        # propagate_nan=ALL: a NaN logit must poison its lane (maxnum would drop it while mv is still -inf and the
        # -inf guard below would then zero the lane -> finite loss from NaN logits).
        mn = tl.maximum(mv, x, propagate_nan=tl.PropagateNan.ALL)
        # lanes that have only seen masked columns stay at (-inf, 0) (exp(-inf - -inf) would be NaN)
        sv = tl.where(mn == float('-inf'), 0.0, sv * tl.exp(mv - mn) + tl.exp(x - mn))
        mv = mn
    m = tl.max(mv, axis=0)
    s = tl.sum(tl.where(mv == float('-inf'), 0.0, sv * tl.exp(mv - m)), axis=0)
    lse = m + tl.log(s)
    tgt = tl.load(ptr + safe_lab).to(tl.float32)
    tl.store(LOSS + row, tl.where(valid, lse - tgt, 0.0))
    if COMPUTE_GRAD:
        # pass 2 (the re-read mostly hits L2): G = (p - onehot)·scale, rounded once to the buffer dtype
        scale = tl.where(valid, tl.load(INV_N) * gscale, 0.0)
        for start in range(0, Vp, BLOCK_V):
            cols = start + offs
            x = tl.load(ptr + cols, mask=cols < V, other=float('-inf')).to(tl.float32)
            g = (tl.exp(x - lse) - tl.where(cols == safe_lab, 1.0, 0.0)) * scale
            tl.store(ptr + cols, g.to(L.dtype.element_ty), mask=cols < Vp)


def _launch_rows(lg, lab, loss_rows, inv_n, V, ignore_index, compute_grad, gscale=1.0):
    T, Vp = lg.shape
    if T == 0:
        return
    _lmce_row_kernel[(T,)](lg, lg.stride(0), lab, loss_rows, inv_n, float(gscale), V, Vp, ignore_index,
                           COMPUTE_GRAD=compute_grad, BLOCK_V=_BLOCK_V, num_warps=_WARPS)


def _compute_dtype(x):
    try:
        if torch.is_autocast_enabled(x.device.type):
            return torch.get_autocast_dtype(x.device.type)
    except (TypeError, RuntimeError):
        pass
    return x.dtype


def _pick_chunk(N, Vp, chunk_tokens, elt):
    """Token-chunk size: explicit, or balanced chunks under a logits-buffer budget (1 GiB), rounded up to a multiple
    of 128 rows (whole GEMM tiles)."""
    if chunk_tokens:
        return max(1, min(N, int(chunk_tokens)))
    tmax = max(128, (_CHUNK_MB << 20) // (Vp * elt))
    n_chunks = -(-N // tmax)
    t = -(-N // n_chunks)
    return min(N, -(-t // 128) * 128)


def _padded_lowp_weight(w, dtype, vocab_pad=_VPAD):
    """rows [0, V) = w.to(dtype) (RNE, == autocast), rows [V, Vp) = 0."""
    V, D = w.shape
    Vp = (V + vocab_pad - 1) // vocab_pad * vocab_pad
    out = torch.empty((Vp, D), dtype=dtype, device=w.device)
    out[:V].copy_(w.detach())
    if Vp > V:
        out[V:].zero_()
    return out


class _LinearCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, labels, opts):
        ignore_index, chunk_tokens, cdtype, need_dh, need_dw, grad_scale = opts
        # forward of an autograd.Function runs under no_grad: the caller decides which gradients are needed
        need_grad = need_dh or need_dw
        D = h.shape[-1]
        V = w.shape[0]
        dev = h.device
        x = h.reshape(-1, D)
        N = x.shape[0]
        lab = labels.reshape(-1)
        if lab.dtype != torch.int64:
            lab = lab.long()
        lab = lab.contiguous()
        xb = (x if x.dtype == cdtype else x.to(cdtype)).contiguous()      # == autocast cast (RNE)
        wp = _padded_lowp_weight(w, cdtype)                               # == autocast cast of W (RNE) + zero rows
        Vp = wp.shape[0]
        n_valid = (lab != ignore_index).sum()
        inv_n = torch.reciprocal(n_valid.to(torch.float32)).reshape(1)
        loss_rows = torch.empty(N, dtype=torch.float32, device=dev)
        T = _pick_chunk(N, Vp, chunk_tokens, wp.element_size())
        buf = torch.empty((T, Vp), dtype=cdtype, device=dev)
        dx = torch.empty((N, D), dtype=torch.float32, device=dev) if need_dh else None
        # dW is computed for all Vp rows: an output with an odd row count (e.g. 50,277) makes cuBLAS fall back to an
        # unaligned kernel that is several times slower.
        dw = torch.empty((Vp, D), dtype=torch.float32, device=dev) if need_dw else None
        gs = 1.0 if grad_scale is None else float(grad_scale)
        first = True
        for s in range(0, N, T):
            e = min(N, s + T)
            lg = buf[:e - s]
            xc = xb[s:e]
            torch.mm(xc, wp.t(), out=lg)                                   # logits chunk (low-precision out, fp32 acc)
            _launch_rows(lg, lab[s:e], loss_rows[s:e], inv_n, V, ignore_index, need_grad, gscale=gs)
            if need_dh:                                                    # dx = G @ W   (padded K, zero rows)
                if _HAS_OUT_DTYPE:
                    torch.mm(lg, wp, out_dtype=torch.float32, out=dx[s:e])
                else:
                    dx[s:e].copy_(torch.mm(lg, wp))
            if need_dw:                                                    # dW (+)= Gᵀ @ x  (fp32 accumulate)
                gt = lg.t()
                if _HAS_OUT_DTYPE:
                    if first:
                        torch.mm(gt, xc, out_dtype=torch.float32, out=dw)
                    else:
                        torch.addmm(dw, gt, xc, out_dtype=torch.float32, out=dw)
                else:
                    part = torch.mm(gt, xc)
                    if first:
                        dw.copy_(part)
                    else:
                        dw.add_(part)
            first = False
        del buf
        loss = loss_rows.sum() / n_valid
        ctx.h_shape, ctx.h_dtype, ctx.w_dtype, ctx.grad_scale, ctx.V = h.shape, h.dtype, w.dtype, grad_scale, V
        ctx.save_for_backward(dx, dw)
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        dx, dw = ctx.saved_tensors
        if ctx.grad_scale is not None:
            # G was pre-scaled by grad_scale in forward: nothing to multiply. Device-side check of the promise
            # (no host sync); a mismatch raises a device assert.
            torch._assert_async(g.to(torch.float32) == float(ctx.grad_scale),
                                "linear_cross_entropy: backward seed differs from the promised grad_scale")
            dh = dx.view(ctx.h_shape) if dx is not None else None
            dW = dw[:ctx.V] if dw is not None else None                     # contiguous prefix view
        else:
            g = g.to(torch.float32)
            dh = torch.mul(dx, g).view(ctx.h_shape) if dx is not None else None
            dW = torch.mul(dw[:ctx.V], g) if dw is not None else None
        if dh is not None and dh.dtype != ctx.h_dtype:
            dh = dh.to(ctx.h_dtype)
        if dW is not None and dW.dtype != ctx.w_dtype:
            dW = dW.to(ctx.w_dtype)
        return dh, dW, None, None


def reference_linear_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                                   ignore_index: int = -100) -> torch.Tensor:
    """The eager formula: ``F.cross_entropy(F.linear(hidden, weight).float(), labels, ignore_index)``."""
    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits.view(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=ignore_index)


@torch.compiler.disable
def linear_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100,
                         *, chunk_tokens: int | None = None, grad_scale: float | None = None) -> torch.Tensor:
    """Mean cross-entropy of ``hidden @ weight.T`` against ``labels`` without materialising the full logits.

    Args:
        hidden: ``(..., D)`` hidden states (e.g. the fp32 final-norm output under autocast, or bf16/fp16).
        weight: ``(V, D)`` LM-head weight (tied embedding), fp32 master or bf16/fp16; receives ``dL/dW``.
        labels: ``(...)`` integer targets aligned with ``hidden`` rows (no shifting is done here). Rows equal to
            ``ignore_index`` get neither loss nor gradient (``F.cross_entropy`` semantics).
        chunk_tokens: rows per chunk (default: balanced chunks under a 1 GiB logits buffer).
        grad_scale: opt-in promise that backward will be seeded with exactly this value (e.g. ``1/grad_accum`` when
            the loss is scaled before ``backward``). G is then rounded once at the final scale and backward does no
            extra passes. A different seed trips a device-side assert.

    Returns:
        Scalar fp32 loss (mean over non-ignored rows; NaN if every row is ignored, like ``F.cross_entropy``).

    The compute dtype is the autocast dtype when autocast is on, else ``hidden.dtype``. Without CUDA or with an fp32
    compute dtype, the eager formula is used.
    """
    cdtype = _compute_dtype(hidden)
    if cdtype not in (torch.bfloat16, torch.float16) or not hidden.is_cuda:
        return reference_linear_cross_entropy(hidden, weight, labels, ignore_index)
    if weight.dim() != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError(f"weight must be (V, {hidden.shape[-1]}), got {tuple(weight.shape)}")
    if labels.numel() != hidden.numel() // hidden.shape[-1]:
        raise ValueError(f"labels have {labels.numel()} elements, expected one per hidden row "
                         f"({hidden.numel() // hidden.shape[-1]})")
    ge = torch.is_grad_enabled()
    opts = (ignore_index, chunk_tokens, cdtype, bool(ge and hidden.requires_grad), bool(ge and weight.requires_grad),
            grad_scale)
    return _LinearCrossEntropyFn.apply(hidden, weight, labels, opts)
