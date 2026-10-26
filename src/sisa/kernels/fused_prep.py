"""Fused SISA layer preparation + attention (Triton).

Replaces the torch glue between the projections and the attention (softplus and cumulative sums, the rotation of C
and B by Φ, dtype copies, ``.contiguous()``) with four Triton kernels (two forward, two backward) and wraps them,
together with the attention kernels of :mod:`sisa.kernels.attention`, in one autograd function.

Inputs
------
``proj = x_norm @ W_catᵀ + b_cat``: the seven projections W_Q, W_K, W_V, W_B, W_C, θ, α in one GEMM, bf16 or fp16,
shape ``(B, L, NPAD)``, column layout (see :func:`fused_layout`)::

    [ Q  H·64 | K  H·64 | V  H·64 | B  H·32 | C  H·32 | θ  H·16 | α  H | zero padding to a multiple of 16 ]

The padding matters: with a row stride that is only a multiple of 8, Triton cannot specialise the loads for
alignment and the forward pass runs about 2.5x slower. ``b_cat`` is zero except for the α columns (the decay bias).

``z = W_f x_norm + b_f``: the forget-gate pre-activation, fp32, shape ``(B, L, H)``, computed with autocast off.

Inside the kernels: ``g = cumsum(-softplus(α))``, ``F = cumsum(logsigmoid(z))``, ``Φ = cumsum(θ)`` and the
interleaved-pair rotation of C and B by Φ. q and k are not rotated (NoPE): the kernels keep a q/k rotation stage
that is fed internal identity tables (cos = 1, sin = 0), which is bit-identical to no rotation up to the sign of
zeros. The tables are created and cached per device here; callers never see them.

Numerics (relative to the unfused torch glue)
---------------------------------------------
 - q/k rotation stage: mimics torch's bf16 operation order exactly (table -> bf16 (RNE), q1·c -> bf16, q2·s -> bf16,
   difference -> bf16; a bf16 x bf16 product is exact in fp32, so the rounding points match torch). ``_prep_fwd`` is
   compiled with ``enable_fp_fusion=False``: with fusion NVPTX contracts bf16(q1·c) - bf16(q2·s) into
   mul.bf16 + fma.rn.bf16 and drops one rounding.
 - g, F, Φ: the element functions use torch's CUDA formulas (softplus threshold 20,
   logsigmoid = min(z, 0) - log1p(exp(-|z|)), libdevice exp/log1p/cos/sin). The cumulative sums are a tree scan within
   64-row blocks plus a sequential sum of the preceding block totals (a different order than torch's scan: fp32
   rounding-level differences). The block totals come from the same code as the last value of the in-block scan, so
   g stays monotone across block boundaries.
 - Rotation of C/B: fp32 (FMA contraction allowed), then one rounding to bf16; torch rounds the products and the
   difference separately (fp32 rounding-level difference).
 - Backward rounds less than torch (higher accuracy, bf16-level differences): the inverse q/k rotation is computed in
   fp32 and rounded once; dc/db arrive from the attention in fp32; the input gradient of the seven projections is one
   GEMM with fp32 accumulation.
 - dα and dθ are rounded once to the projection dtype and dz once to fp32, the same rounding points as torch. The
   reverse cumulative sums (d log α = Σ_{s>=t} dg_s, d log f, dθ = Σ_{s>=t} dΦ_s) are accumulated in fp64 and rounded
   to fp32 once.

Non-finite guard
----------------
The fused path skips the monotonicity check of the unfused attention (g is monotone by construction), so the
remaining risk is non-finite values. The decay clamps propagate NaN, and :func:`_nan_guard` checks the sum of lse and
of the block totals (see :func:`sisa.backends.set_check_mode`).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .. import backends
from . import attention as A

__all__ = ["sisa_fused", "fused_layout", "concat_fused_weights"]

LOG2E = tl.constexpr(1.4426950408889634)
BL = 64          # block rows = the decay-factor grouping T of the attention (rowf/colf are built here)
D_HEAD = 64
D_STATE = 32
assert BL == 64  # frozen: must equal the attention tile T for bf16/fp16


def fused_layout(n_head: int, d_head: int = D_HEAD, d_state: int = D_STATE) -> dict:
    """Column offsets of the fused projection ``proj`` for ``n_head`` heads.

    Returns a dict with the start column of each block, ``Q, K, V`` (width ``n_head·64`` each), ``B, C`` (width
    ``n_head·32`` each), ``TH`` (θ, width ``n_head·16``) and ``A`` (α, width ``n_head``), plus ``NTOT`` (= 273·H, used
    columns) and ``NPAD`` (row width, NTOT rounded up to a multiple of 16; for H = 8: 2184 -> 2192). Within each
    block the column of head h, component j is ``start + h·width_per_head + j``, i.e. the row order of the original
    ``nn.Linear`` weights. Columns ``[NTOT, NPAD)`` are zero.
    """
    if d_head != D_HEAD or d_state != D_STATE:
        raise ValueError(f"the fused kernels support d_head={D_HEAD} and d_state={D_STATE} only")
    H = int(n_head)
    S2 = d_state // 2
    o = {}
    o["Q"] = 0
    o["K"] = H * d_head
    o["V"] = 2 * H * d_head
    o["B"] = 3 * H * d_head
    o["C"] = 3 * H * d_head + H * d_state
    o["TH"] = 3 * H * d_head + 2 * H * d_state
    o["A"] = o["TH"] + H * S2
    o["NTOT"] = o["A"] + H
    # The row stride must be a multiple of 16 for Triton to specialise on alignment and vectorise the loads.
    o["NPAD"] = (o["NTOT"] + 15) // 16 * 16
    assert o["NPAD"] % 16 == 0
    return o


def concat_fused_weights(W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor, W_B: torch.Tensor,
                         W_C: torch.Tensor, W_theta: torch.Tensor, W_alpha: torch.Tensor,
                         b_alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(W_cat, b_cat)`` for ``proj = F.linear(x_norm, W_cat, b_cat)`` from the per-projection weights.

    All weights are ``(out, d_model)`` ``nn.Linear`` weights (W_Q/W_K/W_V: H·64 rows, W_B/W_C: H·32, W_theta: H·16,
    W_alpha: H) and ``b_alpha`` is the ``(H,)`` decay bias. Padding rows and every bias entry except the α columns
    are zero. The concatenation is differentiable, so gradients reach the original parameters.
    """
    H = W_alpha.shape[0]
    off = fused_layout(H)
    pad = off["NPAD"] - off["NTOT"]
    d = W_Q.shape[1]
    ws = [W_Q, W_K, W_V, W_B, W_C, W_theta, W_alpha]
    bs = [b_alpha]
    zb = torch.zeros(off["A"], device=b_alpha.device, dtype=b_alpha.dtype)
    if pad:
        ws.append(torch.zeros(pad, d, device=W_Q.device, dtype=W_Q.dtype))
        bs.append(torch.zeros(pad, device=b_alpha.device, dtype=b_alpha.dtype))
    W = torch.cat(ws, 0)
    b = torch.cat([zb] + bs, 0)
    if W.shape[0] != off["NPAD"]:
        raise ValueError(f"concatenated weight has {W.shape[0]} rows, expected {off['NPAD']} for n_head={H}")
    return W, b


# ============================================================================ q/k identity tables
_TABLE_ROWS = 2048
_TABLES: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_ROPE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _norm_dev(device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


def _identity_tables(L: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 contiguous (rows, 32) all-ones cos / all-zeros sin, rows >= L (a multiple of 2048), one pair per device."""
    dev = _norm_dev(device)
    key = str(dev)
    t = _TABLES.get(key)
    if t is not None and t[0].shape[0] >= L:
        return t
    if dev.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"the fused SISA layer needs q/k identity tables of at least {L} rows on {key}; allocate them before "
            "CUDA-graph capture by running one eager forward at the longest sequence length"
        )
    rows = max(_TABLE_ROWS, -(-int(L) // _TABLE_ROWS) * _TABLE_ROWS)
    cos = torch.ones((rows, D_STATE), dtype=torch.float32, device=dev)
    sin = torch.zeros((rows, D_STATE), dtype=torch.float32, device=dev)
    _TABLES[key] = (cos, sin)
    return cos, sin


def _rope_table_problem(cos, sin, L, D2, device):
    """None if the q/k tables match the kernels' read contract, else the reason.

    The kernels read the tables with one row stride (cos.stride(0)) and column stride 1 over [0, L) x [0, D2) without
    bounds checks, so a too-short table would silently read past its end.
    """
    for name, t in (("cos", cos), ("sin", sin)):
        if not isinstance(t, torch.Tensor) or t.dim() != 2:
            return f"{name}: must be a 2-D tensor (shape={getattr(t, 'shape', None)})"
        if t.shape[0] < L:
            return f"{name}: table has {t.shape[0]} rows < L={L}"
        if t.shape[1] != D2:
            return f"{name}: {t.shape[1]} columns != d_head/2={D2}"
        if t.device != device:
            return f"{name}: device {t.device} != {device}"
        if t.dtype not in _ROPE_DTYPES:
            return f"{name}: unsupported dtype {t.dtype} (fp32/bf16/fp16)"
        if t.stride(1) != 1:
            return f"{name}: column stride {t.stride(1)} != 1"
    if sin.stride(0) != cos.stride(0):
        return f"cos/sin row strides differ ({cos.stride(0)} vs {sin.stride(0)}); the kernels take a single one"
    return None


def _check_rope_tables(cos, sin, L, D2, device):
    why = _rope_table_problem(cos, sin, L, D2, device)
    if why is not None:
        raise RuntimeError(f"fused SISA layer: {why}")


# ============================================================================ element functions (torch CUDA formulas)
@triton.jit
def _neg_softplus(a):
    """-softplus(a), torch formula: a > 20 ? a : log1p(exp(a))."""
    return -tl.where(a > 20., a, libdevice.log1p(libdevice.exp(a)))


@triton.jit
def _softplus_grad(a):
    """d softplus / da, torch softplus_backward formula: a > 20 ? 1 : e/(e+1), e = exp(a)."""
    e = libdevice.exp(a)
    return tl.where(a > 20., 1., e / (e + 1.))


@triton.jit
def _logsigmoid(z):
    """torch log_sigmoid_forward formula: min(z, 0) - log1p(exp(-|z|)).
    The min propagates NaN, like every clamp on the decay/forget path (bit-identical for finite z)."""
    return tl.minimum(z, 0., propagate_nan=tl.PropagateNan.ALL) - libdevice.log1p(libdevice.exp(-tl.abs(z)))


@triton.jit
def _logsigmoid_grad(z):
    """torch log_sigmoid_backward formula: z < 0 ? 1 - e/(1+e) : e/(1+e), e = exp(-|z|)."""
    e = libdevice.exp(-tl.abs(z))
    r = e / (1. + e)
    return tl.where(z < 0., 1. - r, r)


@triton.jit
def _last(x, idx, N: tl.constexpr):
    return tl.sum(tl.where(idx == N - 1, x, 0.), 0)


@triton.jit
def _first(x, idx):
    return tl.sum(tl.where(idx == 0, x, 0.), 0)


@triton.jit
def _ld_pairs(rowptr, m, BLK: tl.constexpr, W: tl.constexpr):
    """Vector-load W consecutive elements per row, then split them in registers into (even, odd) [BLK, W/2] (fp32)."""
    x = tl.load(rowptr[:, None] + tl.arange(0, W)[None, :], mask=m[:, None], other=0.).to(tl.float32)
    return tl.split(tl.reshape(x, [BLK, W // 2, 2]))


@triton.jit
def _st_pairs(rowptr, m, e, o, BLK: tl.constexpr, W: tl.constexpr):
    y = tl.reshape(tl.join(e, o), [BLK, W])
    tl.store(rowptr[:, None] + tl.arange(0, W)[None, :], y.to(rowptr.dtype.element_ty), mask=m[:, None])


@triton.jit
def _rot_load(P, rows, m, s_pl, col0, S2: tl.constexpr):
    """(even, odd) pairs [BL, S2] of proj as fp32: contiguous vector load + register split."""
    BLK: tl.constexpr = rows.shape[0]
    return _ld_pairs(P + rows.to(tl.int64) * s_pl + col0, m, BLK, 2 * S2)


# ============================================================================ forward
@triton.jit
def _prep_sums(P, Z, TOT, s_pb, s_pl, s_zb, s_zl, L, NCH,
               H: tl.constexpr, S2: tl.constexpr, OFF_TH: tl.constexpr, OFF_A: tl.constexpr,
               BLK: tl.constexpr, HAS_F: tl.constexpr, ROPE: tl.constexpr):
    """Per-block (64-row) totals TOT[bh, c] = (Σ log α, Σ log f, Σ θ[16]), from the same code as the last value of the
    in-block scan in _prep_fwd."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    idx = tl.arange(0, BLK)
    rows = c * BLK + idx
    m = rows < L
    P += b.to(tl.int64) * s_pb
    TOT += (bh.to(tl.int64) * NCH + c) * (2 + S2)
    a = tl.load(P + rows.to(tl.int64) * s_pl + OFF_A + h, mask=m, other=0.).to(tl.float32)
    la = tl.where(m, _neg_softplus(a), 0.)
    tl.store(TOT, _last(tl.cumsum(la, 0), idx, BLK))
    if HAS_F:
        z = tl.load(Z + b.to(tl.int64) * s_zb + rows.to(tl.int64) * s_zl + h, mask=m, other=0.)
        lf = tl.where(m, _logsigmoid(z), 0.)
        tl.store(TOT + 1, _last(tl.cumsum(lf, 0), idx, BLK))
    if ROPE:
        j = tl.arange(0, S2)
        th = tl.load(P + rows[:, None].to(tl.int64) * s_pl + OFF_TH + h * S2 + j[None, :], mask=m[:, None],
                     other=0.).to(tl.float32)
        cs = tl.cumsum(th, 0)
        tl.store(TOT + 2 + j, tl.sum(tl.where(idx[:, None] == BLK - 1, cs, 0.), 0))


@triton.jit
def _prep_fwd(P, Z, COS, SIN, TOT, QR, KR, CR, BR, G, FO, ROWF, COLF,
              s_pb, s_pl, s_zb, s_zl, s_cl, L, NCH,
              H: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
              OFF_Q: tl.constexpr, OFF_K: tl.constexpr, OFF_B: tl.constexpr, OFF_C: tl.constexpr,
              OFF_TH: tl.constexpr, OFF_A: tl.constexpr,
              BLK: tl.constexpr, HAS_F: tl.constexpr, ROPE: tl.constexpr):
    c = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    S2: tl.constexpr = S // 2
    D2: tl.constexpr = D // 2
    idx = tl.arange(0, BLK)
    rows = c * BLK + idx
    m = rows < L
    P += b.to(tl.int64) * s_pb
    rb = bh.to(tl.int64) * L
    j = tl.arange(0, S2)
    # ---- totals of the preceding blocks (sequential, deterministic) ----
    ca = 0.
    cfv = 0.
    ct = tl.zeros([S2], tl.float32)
    for cc in range(0, c):
        tb = TOT + (bh.to(tl.int64) * NCH + cc) * (2 + S2)
        ca += tl.load(tb)
        if HAS_F:
            cfv += tl.load(tb + 1)
        if ROPE:
            ct += tl.load(tb + 2 + j)
    # ---- g, rowf, colf ----
    a = tl.load(P + rows.to(tl.int64) * s_pl + OFF_A + h, mask=m, other=0.).to(tl.float32)
    la = tl.where(m, _neg_softplus(a), 0.)
    g = ca + tl.cumsum(la, 0)
    tl.store(G + rb + rows, g, mask=m)
    gs = _first(g, idx)
    ge = _last(g, idx, BLK)                    # out-of-range rows have la = 0, so this equals the last valid row
    tl.store(ROWF + rb + rows, tl.exp2(tl.minimum((g - gs) * LOG2E, 0., propagate_nan=tl.PropagateNan.ALL)), mask=m)
    tl.store(COLF + rb + rows, tl.exp2(tl.minimum((ge - g) * LOG2E, 0., propagate_nan=tl.PropagateNan.ALL)), mask=m)
    # ---- F ----
    if HAS_F:
        z = tl.load(Z + b.to(tl.int64) * s_zb + rows.to(tl.int64) * s_zl + h, mask=m, other=0.)
        lf = tl.where(m, _logsigmoid(z), 0.)
        tl.store(FO + rb + rows, cfv + tl.cumsum(lf, 0), mask=m)
    # ---- Φ and the rotation of C/B (fp32 -> one rounding to bf16) ----
    ob = (b.to(tl.int64) * L + rows.to(tl.int64)) * (H * S) + h * S      # row start (S contiguous elements)
    if ROPE:
        th = tl.load(P + rows[:, None].to(tl.int64) * s_pl + OFF_TH + h * S2 + j[None, :], mask=m[:, None],
                     other=0.).to(tl.float32)
        phi = ct[None, :] + tl.cumsum(th, 0)
        cp = libdevice.cos(phi)
        sp = libdevice.sin(phi)
        ze, zo = _rot_load(P, rows, m, s_pl, OFF_C + h * S, S2)
        _st_pairs(CR + ob, m, ze * cp - zo * sp, ze * sp + zo * cp, BLK, S)
        ze, zo = _rot_load(P, rows, m, s_pl, OFF_B + h * S, S2)
        _st_pairs(BR + ob, m, ze * cp - zo * sp, ze * sp + zo * cp, BLK, S)
    else:
        ze, zo = _rot_load(P, rows, m, s_pl, OFF_C + h * S, S2)
        _st_pairs(CR + ob, m, ze, zo, BLK, S)
        ze, zo = _rot_load(P, rows, m, s_pl, OFF_B + h * S, S2)
        _st_pairs(BR + ob, m, ze, zo, BLK, S)
    # ---- q/k rotation stage: torch's bf16 operation order, bit for bit (identity tables -> unrotated q/k) ----
    jd = tl.arange(0, D2)
    cb = tl.load(COS + rows[:, None].to(tl.int64) * s_cl + jd[None, :], mask=m[:, None], other=0.)
    sb = tl.load(SIN + rows[:, None].to(tl.int64) * s_cl + jd[None, :], mask=m[:, None], other=0.)
    cb = cb.to(QR.dtype.element_ty).to(tl.float32)            # cos[:L].to(q.dtype)
    sb = sb.to(QR.dtype.element_ty).to(tl.float32)
    oq = (b.to(tl.int64) * L + rows.to(tl.int64)) * (H * D) + h * D
    qe, qo = _rot_load(P, rows, m, s_pl, OFF_Q + h * D, D2)
    ne = (qe * cb).to(QR.dtype.element_ty).to(tl.float32) - (qo * sb).to(QR.dtype.element_ty).to(tl.float32)
    no = (qe * sb).to(QR.dtype.element_ty).to(tl.float32) + (qo * cb).to(QR.dtype.element_ty).to(tl.float32)
    _st_pairs(QR + oq, m, ne, no, BLK, D)
    ke, ko = _rot_load(P, rows, m, s_pl, OFF_K + h * D, D2)
    ne = (ke * cb).to(KR.dtype.element_ty).to(tl.float32) - (ko * sb).to(KR.dtype.element_ty).to(tl.float32)
    no = (ke * sb).to(KR.dtype.element_ty).to(tl.float32) + (ko * cb).to(KR.dtype.element_ty).to(tl.float32)
    _st_pairs(KR + oq, m, ne, no, BLK, D)


# ============================================================================ backward
@triton.jit
def _phi_block(P, TOT, bh, c, rows, m, s_pl, NCH, h, OFF_TH: tl.constexpr, S2: tl.constexpr):
    """Recompute one Φ block with **the same code** as the forward pass (bit-identical)."""
    j = tl.arange(0, S2)
    ct = tl.zeros([S2], tl.float32)
    for cc in range(0, c):
        ct += tl.load(TOT + (bh.to(tl.int64) * NCH + cc) * (2 + S2) + 2 + j)
    th = tl.load(P + rows[:, None].to(tl.int64) * s_pl + OFF_TH + h * S2 + j[None, :], mask=m[:, None],
                 other=0.).to(tl.float32)
    return ct[None, :] + tl.cumsum(th, 0)


@triton.jit
def _dphi_rows(P, DZR, rows, m, b, h, s_pl, L, cp, sp, col0, H: tl.constexpr, S: tl.constexpr):
    """dΦ contribution and input gradient (unrotated basis) of one rotated input (C or B).
    dΦ = dout_o·out_e - dout_e·out_o."""
    S2: tl.constexpr = S // 2
    j = tl.arange(0, S2)  # noqa: F841 - kept identical to the validated kernel source
    ze, zo = _rot_load(P, rows, m, s_pl, col0 + h * S, S2)
    oe = ze * cp - zo * sp
    oo = ze * sp + zo * cp
    BLK: tl.constexpr = rows.shape[0]
    de, do_ = _ld_pairs(DZR + (b.to(tl.int64) * L + rows.to(tl.int64)) * (H * S) + h * S, m, BLK, S)
    dphi = do_ * oe - de * oo
    dze = de * cp + do_ * sp
    dzo = do_ * cp - de * sp
    return dphi, dze, dzo


@triton.jit
def _prep_bsums(P, TOT, DG, DF, DCR, DBR, BT, s_pb, s_pl, L, NCH,
                H: tl.constexpr, S: tl.constexpr, OFF_B: tl.constexpr, OFF_C: tl.constexpr,
                OFF_TH: tl.constexpr, BLK: tl.constexpr, HAS_F: tl.constexpr, ROPE: tl.constexpr):
    """Per-block totals BT[bh, c] = (Σ dg, Σ dF, Σ dΦ[16]) in **fp64** (the reverse cumulative sums d log α, d log f and
    dθ are accumulated in fp64 and rounded once)."""
    c = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    S2: tl.constexpr = S // 2
    idx = tl.arange(0, BLK)
    rows = c * BLK + idx
    m = rows < L
    P += b.to(tl.int64) * s_pb
    rb = bh.to(tl.int64) * L
    BT += (bh.to(tl.int64) * NCH + c) * (2 + S2)
    tl.store(BT, tl.sum(tl.load(DG + rb + rows, mask=m, other=0.).to(tl.float64), 0))
    if HAS_F:
        tl.store(BT + 1, tl.sum(tl.load(DF + rb + rows, mask=m, other=0.).to(tl.float64), 0))
    if ROPE:
        phi = _phi_block(P, TOT, bh, c, rows, m, s_pl, NCH, h, OFF_TH, S2)
        cp = libdevice.cos(phi)
        sp = libdevice.sin(phi)
        d1, _e1, _o1 = _dphi_rows(P, DCR, rows, m, b, h, s_pl, L, cp, sp, OFF_C, H, S)
        d2, _e2, _o2 = _dphi_rows(P, DBR, rows, m, b, h, s_pl, L, cp, sp, OFF_B, H, S)
        tl.store(BT + 2 + tl.arange(0, S2), tl.sum((d1 + d2).to(tl.float64), 0))


@triton.jit
def _prep_bwd(P, Z, COS, SIN, TOT, DG, DF, DCR, DBR, BT, DP, DZ,
              s_pb, s_pl, s_zb, s_zl, s_cl, s_db, s_dl, s_dzb, s_dzl, L, NCH,
              H: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
              OFF_Q: tl.constexpr, OFF_K: tl.constexpr, OFF_B: tl.constexpr, OFF_C: tl.constexpr,
              OFF_TH: tl.constexpr, OFF_A: tl.constexpr, NTOT: tl.constexpr, NPAD: tl.constexpr,
              BLK: tl.constexpr, HAS_F: tl.constexpr, ROPE: tl.constexpr):
    c = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    S2: tl.constexpr = S // 2
    D2: tl.constexpr = D // 2
    idx = tl.arange(0, BLK)
    rows = c * BLK + idx
    m = rows < L
    P += b.to(tl.int64) * s_pb
    DP += b.to(tl.int64) * s_db
    rb = bh.to(tl.int64) * L
    j = tl.arange(0, S2)
    # ---- totals of the following blocks (sequential, deterministic, fp64) ----
    sa = tl.full([], 0., tl.float64)
    sf = tl.full([], 0., tl.float64)
    st = tl.zeros([S2], tl.float64)
    for cc in range(c + 1, NCH):
        tb = BT + (bh.to(tl.int64) * NCH + cc) * (2 + S2)
        sa += tl.load(tb)
        if HAS_F:
            sf += tl.load(tb + 1)
        if ROPE:
            st += tl.load(tb + 2 + j)
    # ---- dα = -(Σ_{s>=t} dg_s)·softplus'(α) ----
    dg = tl.load(DG + rb + rows, mask=m, other=0.)
    dla = (sa + tl.cumsum(dg.to(tl.float64), 0, reverse=True)).to(tl.float32)
    a = tl.load(P + rows.to(tl.int64) * s_pl + OFF_A + h, mask=m, other=0.).to(tl.float32)
    tl.store(DP + rows.to(tl.int64) * s_dl + OFF_A + h, (-dla * _softplus_grad(a)).to(DP.dtype.element_ty), mask=m)
    # ---- dz = (Σ_{s>=t} dF_s)·logsigmoid'(z) ----
    if HAS_F:
        dfv = tl.load(DF + rb + rows, mask=m, other=0.)
        dlf = (sf + tl.cumsum(dfv.to(tl.float64), 0, reverse=True)).to(tl.float32)
        z = tl.load(Z + b.to(tl.int64) * s_zb + rows.to(tl.int64) * s_zl + h, mask=m, other=0.)
        tl.store(DZ + b.to(tl.int64) * s_dzb + rows.to(tl.int64) * s_dzl + h, dlf * _logsigmoid_grad(z), mask=m)
    # ---- inverse rotation of C/B, dθ = Σ_{s>=t} dΦ_s ----
    dpr = DP + rows.to(tl.int64) * s_dl                     # row start of d_proj
    if ROPE:
        phi = _phi_block(P, TOT, bh, c, rows, m, s_pl, NCH, h, OFF_TH, S2)
        cp = libdevice.cos(phi)
        sp = libdevice.sin(phi)
        d1, dze, dzo = _dphi_rows(P, DCR, rows, m, b, h, s_pl, L, cp, sp, OFF_C, H, S)
        _st_pairs(dpr + OFF_C + h * S, m, dze, dzo, BLK, S)
        d2, dze, dzo = _dphi_rows(P, DBR, rows, m, b, h, s_pl, L, cp, sp, OFF_B, H, S)
        _st_pairs(dpr + OFF_B + h * S, m, dze, dzo, BLK, S)
        dth = (st[None, :] + tl.cumsum((d1 + d2).to(tl.float64), 0, reverse=True)).to(tl.float32)
        tl.store(DP + rows[:, None].to(tl.int64) * s_dl + OFF_TH + h * S2 + j[None, :],
                 dth.to(DP.dtype.element_ty), mask=m[:, None])
    else:
        od = (b.to(tl.int64) * L + rows.to(tl.int64)) * (H * S) + h * S
        de, do_ = _ld_pairs(DCR + od, m, BLK, S)
        _st_pairs(dpr + OFF_C + h * S, m, de, do_, BLK, S)
        de, do_ = _ld_pairs(DBR + od, m, BLK, S)
        _st_pairs(dpr + OFF_B + h * S, m, de, do_, BLK, S)
        tl.store(DP + rows[:, None].to(tl.int64) * s_dl + OFF_TH + h * S2 + j[None, :],
                 tl.zeros([BLK, S2], tl.float32).to(DP.dtype.element_ty), mask=m[:, None])
    # ---- inverse q/k rotation, in place (the attention wrote dq/dk of the rotated basis into these columns);
    #      computed in fp32, rounded once ----
    jd = tl.arange(0, D2)
    cb = tl.load(COS + rows[:, None].to(tl.int64) * s_cl + jd[None, :], mask=m[:, None], other=0.)
    sb = tl.load(SIN + rows[:, None].to(tl.int64) * s_cl + jd[None, :], mask=m[:, None], other=0.)
    cb = cb.to(DP.dtype.element_ty).to(tl.float32)
    sb = sb.to(DP.dtype.element_ty).to(tl.float32)
    de, do_ = _ld_pairs(dpr + OFF_Q + h * D, m, BLK, D)
    _st_pairs(dpr + OFF_Q + h * D, m, de * cb + do_ * sb, do_ * cb - de * sb, BLK, D)
    de, do_ = _ld_pairs(dpr + OFF_K + h * D, m, BLK, D)
    _st_pairs(dpr + OFF_K + h * D, m, de * cb + do_ * sb, do_ * cb - de * sb, BLK, D)
    # ---- zero the padding columns (they multiply the zero rows of W_cat; keeps NaN out of dX) ----
    if NPAD > NTOT:
        if h == 0:
            jp = tl.arange(0, 16)
            tl.store(DP + rows[:, None].to(tl.int64) * s_dl + NTOT + jp[None, :],
                     tl.zeros([BLK, 16], tl.float32).to(DP.dtype.element_ty),
                     mask=m[:, None] & (NTOT + jp[None, :] < NPAD))


# ============================================================================ non-finite guard
_GUARD_MSG = ("fused SISA layer: non-finite input (NaN/inf in q/k/c/b/lambda/g/F, i.e. in a parameter such as "
              "alpha_proj, theta_proj, forget_proj, W_Q or lambda_raw)")


def _nan_guard(lse, TOT):
    """Non-finite check of the fused path.

    The fused path skips the monotonicity check of the unfused attention (``factors`` are passed) and g is monotone by
    construction, so the remaining risk is non-finite values. Two places are checked:
      · lse (B,H,L): every input of the score s_ij (q, k, c, b, λ, g, F) meets here. The decay clamps propagate NaN,
        so a NaN in g reaches s as well (the diagonal g_t - g_t is NaN), and one NaN in a row makes l = Σ pe and lse
        NaN.
      · TOT (per-block Σ log α, Σ log f, Σ θ): sees the α, z and θ projections directly.
    One non-finite element <=> a non-finite sum (inf + (-inf) = NaN included); a finite sum overflows only for values
    around 1e32. A NaN only in V (W_V) is not seen in lse, but it makes the output and the loss NaN, so it cannot go
    unnoticed. The cost is a handful of tiny kernels per layer and no host synchronisation (async mode).
    """
    mode = backends.get_check_mode()
    if mode == "off":
        return
    s = lse.sum() + TOT.sum()
    if mode == "sync":
        if not bool(torch.isfinite(s)):
            raise FloatingPointError(f"{_GUARD_MSG}: sum(lse) + sum(block totals) = {s.item()}")
    else:
        torch._assert_async(s.abs() < math.inf, _GUARD_MSG)


# ============================================================================ host launches
def _launch_prep_forward(proj, z, cos, sin, H):
    """Run _prep_sums and _prep_fwd. Returns a dict QR, KR, CR, BR ((B,L,H,·) proj dtype), G, F, ROWF, COLF ((B,H,L)
    fp32; F is None without z) and TOT ((B·H, NCH, 18) fp32 block totals)."""
    B, L, NP = proj.shape
    D, S = D_HEAD, D_STATE
    S2 = S // 2
    off = fused_layout(H)
    assert NP == off["NPAD"] and proj.stride(-1) == 1 and proj.stride(1) == NP, (NP, off, proj.stride())
    assert proj.dtype in (torch.bfloat16, torch.float16)
    _check_rope_tables(cos, sin, L, D // 2, proj.device)
    has_f = z is not None
    rope = True                     # the SSM term is always rotated by Φ
    dev = proj.device
    NCH = triton.cdiv(L, BL)
    bf = dict(device=dev, dtype=proj.dtype)
    f32 = dict(device=dev, dtype=torch.float32)
    QR = torch.empty((B, L, H, D), **bf)
    KR = torch.empty((B, L, H, D), **bf)
    CR = torch.empty((B, L, H, S), **bf)
    BR = torch.empty((B, L, H, S), **bf)
    G = torch.empty((B, H, L), **f32)
    FO = torch.empty((B, H, L), **f32) if has_f else G
    ROWF = torch.empty((B, H, L), **f32)
    COLF = torch.empty((B, H, L), **f32)
    # Unused columns (no forget gate -> Σ log f) are zero: the non-finite guard sums the whole TOT.
    TOT = (torch.empty if (has_f and rope) else torch.zeros)((B * H, NCH, 2 + S2), **f32)
    zz = z if has_f else G
    s_zb, s_zl = (z.stride(0), z.stride(1)) if has_f else (0, 0)
    if has_f:
        assert z.stride(2) == 1
    _prep_sums[(NCH, B * H)](proj, zz, TOT, proj.stride(0), proj.stride(1), s_zb, s_zl, L, NCH,
                             H, S2, off["TH"], off["A"], BL, has_f, rope, num_warps=4)
    # enable_fp_fusion=False is required (module docstring: the q/k stage must keep torch's rounding points)
    _prep_fwd[(NCH, B * H)](proj, zz, cos, sin, TOT, QR, KR, CR, BR, G, FO, ROWF, COLF,
                            proj.stride(0), proj.stride(1), s_zb, s_zl, cos.stride(0), L, NCH,
                            H, D, S, off["Q"], off["K"], off["B"], off["C"], off["TH"], off["A"],
                            BL, has_f, rope, num_warps=4, enable_fp_fusion=False)
    return dict(QR=QR, KR=KR, CR=CR, BR=BR, G=G, F=FO if has_f else None, ROWF=ROWF, COLF=COLF, TOT=TOT)


# ============================================================================ autograd
class _SISAFusedFn(torch.autograd.Function):
    """proj (B,L,NPAD) bf16/fp16, z (B,L,H) fp32 or None, lam (H,) fp32, cos/sin (>=L, 32) q/k tables
    -> y (B,L,H·64) in proj dtype, g (B,H,L) fp32 (not differentiable)."""

    @staticmethod
    def forward(ctx, proj, z, lam, cos, sin, H):
        B, L, NP = proj.shape
        D = D_HEAD
        off = fused_layout(H)
        has_f = z is not None
        pf = _launch_prep_forward(proj, z, cos, sin, H)
        QR, KR, CR, BR, G, ROWF, COLF, TOT = (pf[n] for n in ("QR", "KR", "CR", "BR", "G", "ROWF", "COLF", "TOT"))
        FO = pf["F"] if has_f else G
        q = QR.transpose(1, 2)
        k = KR.transpose(1, 2)
        c = CR.transpose(1, 2)
        b = BR.transpose(1, 2)
        v = proj[..., off["V"]:off["V"] + H * D].view(B, L, H, D).transpose(1, 2)
        need = any(ctx.needs_input_grad)
        o, lse, _, _, _, o32 = A.attn_forward(q, k, v, c, b, G, lam, FO if has_f else None, factors=(ROWF, COLF),
                                              want32=need)
        _nan_guard(lse, TOT)
        # exact mode also saves the self-normalised fp32 o'' for Δ'' (attention module docstring (D))
        ctx.save_for_backward(proj, z if has_f else G, lam, cos, sin, QR, KR, CR, BR, G, FO, ROWF, COLF,
                              o, lse, TOT, o32 if (need and o32 is not o) else None)
        ctx.meta = (H, has_f)
        ctx.mark_non_differentiable(G)
        return o.transpose(1, 2).reshape(B, L, H * D), G

    @staticmethod
    def backward(ctx, dy, _dG):
        proj, zz, lam, cos, sin, QR, KR, CR, BR, G, FO, ROWF, COLF, o, lse, TOT, o32 = ctx.saved_tensors
        H, has_f = ctx.meta
        rope = True
        exact = True
        B, L, NP = proj.shape
        D, S = D_HEAD, D_STATE
        S2 = S // 2
        off = fused_layout(H)
        dev = proj.device
        NCH = triton.cdiv(L, BL)
        f32 = dict(device=dev, dtype=torch.float32)
        do = dy.reshape(B, L, H, D).transpose(1, 2)
        DP = torch.empty_like(proj)
        vw = lambda col, w: DP[..., col:col + H * w].view(B, L, H, w).transpose(1, 2)     # noqa: E731
        DCR = torch.empty((B, L, H, S), **f32)
        DBR = torch.empty((B, L, H, S), **f32)
        q, k, c, b = QR.transpose(1, 2), KR.transpose(1, 2), CR.transpose(1, 2), BR.transpose(1, 2)
        v = proj[..., off["V"]:off["V"] + H * D].view(B, L, H, D).transpose(1, 2)
        _, _, _, _, _, dg, dlam, dF = A.attn_backward(
            q, k, v, c, b, G, FO if has_f else None, ROWF, COLF, lam, o, lse, do, exact,
            dq=vw(off["Q"], D), dk=vw(off["K"], D), dv=vw(off["V"], D),
            dc=DCR.transpose(1, 2), db=DBR.transpose(1, 2), o32=o32)
        BT = torch.empty((B * H, NCH, 2 + S2), device=dev, dtype=torch.float64)
        dFp = dF if has_f else dg
        _prep_bsums[(NCH, B * H)](proj, TOT, dg, dFp, DCR, DBR, BT, proj.stride(0), proj.stride(1), L, NCH,
                                  H, S, off["B"], off["C"], off["TH"], BL, has_f, rope, num_warps=4)
        DZ = torch.empty((B, L, H), **f32) if has_f else None
        dzz = DZ if has_f else dg
        s_zb, s_zl = (zz.stride(0), zz.stride(1)) if has_f else (0, 0)
        s_dzb, s_dzl = (DZ.stride(0), DZ.stride(1)) if has_f else (0, 0)
        _prep_bwd[(NCH, B * H)](proj, zz, cos, sin, TOT, dg, dFp, DCR, DBR, BT, DP, dzz,
                                proj.stride(0), proj.stride(1), s_zb, s_zl, cos.stride(0),
                                DP.stride(0), DP.stride(1), s_dzb, s_dzl, L, NCH,
                                H, D, S, off["Q"], off["K"], off["B"], off["C"], off["TH"], off["A"],
                                off["NTOT"], off["NPAD"], BL, has_f, rope, num_warps=4)
        return DP, DZ, dlam, None, None, None


def sisa_fused(proj: torch.Tensor, z: torch.Tensor | None, lam: torch.Tensor, n_head: int) -> torch.Tensor:
    """Fused SISA layer core: preparation kernels + exact Triton attention.

    Args:
        proj: ``(B, L, NPAD)`` bf16/fp16 output of ``F.linear(x_norm, W_cat, b_cat)`` with the column layout of
            :func:`fused_layout` (build ``W_cat, b_cat`` with :func:`concat_fused_weights`). Requires
            ``proj.stride(-1) == 1`` and ``proj.stride(1) == NPAD`` (a contiguous linear output).
        z: ``(B, L, H)`` fp32 forget-gate logits with ``z.stride(2) == 1`` (compute them with autocast off), or None
            for no forget gate.
        lam: ``(H,)`` fp32 side-term weights, ``softplus(lambda_raw)``.
        n_head: number of heads H (d_head = 64, d_state = 32).

    Returns:
        ``y`` of shape ``(B, L, H·64)`` in ``proj.dtype``: the attention output with heads concatenated in the same
        order as the W_Q rows, ready for ``W_O``. Differentiable with respect to ``proj``, ``z`` and ``lam``.

    Gradients use the exact-Δ attention backward; the gradient of the padding columns of ``proj`` is exactly zero.
    """
    H = int(n_head)
    if proj.dim() != 3:
        raise ValueError(f"proj must be (B, L, NPAD), got shape {tuple(proj.shape)}")
    off = fused_layout(H)
    B, L, NP = proj.shape
    if NP != off["NPAD"]:
        raise ValueError(f"proj has {NP} columns, expected NPAD={off['NPAD']} for n_head={H}")
    if proj.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"proj must be bf16 or fp16, got {proj.dtype}")
    if not proj.is_cuda:
        raise ValueError("proj must be a CUDA tensor")
    if L < 1:
        raise ValueError("sequence length must be >= 1")
    if proj.stride(-1) != 1 or proj.stride(1) != NP:
        proj = proj.contiguous()
    if z is not None:
        if z.shape != (B, L, H):
            raise ValueError(f"z must have shape {(B, L, H)}, got {tuple(z.shape)}")
        if z.dtype != torch.float32:
            raise TypeError(f"z must be fp32, got {z.dtype}")
        if z.stride(2) != 1:
            z = z.contiguous()
    if lam.shape != (H,):
        raise ValueError(f"lam must have shape {(H,)}, got {tuple(lam.shape)}")
    if lam.dtype != torch.float32:
        lam = lam.float()
    cos, sin = _identity_tables(L, proj.device)
    y, _g = _SISAFusedFn.apply(proj, z, lam, cos, sin, H)
    return y
