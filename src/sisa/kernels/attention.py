"""Exact SISA attention in Triton: forward, backward and the autograd wrapper.

    s_ij = q_i·k_j / sqrt(D) + lam_h · exp(g_i - g_j) · (c_i·b_j) + (F_i - F_j)        (j <= i)
    o_i  = sum_j softmax_j(s_ij) · v_j

Entry point: :func:`sisa_attention_triton`. The forward / dq / dkv kernel split follows FlashAttention-2; the
forget-gate term (F_i - F_j) follows the Forgetting Transformer (FoX).

Numerics notes
--------------
(A) No low-precision rounding of the decayed side term. The c·b GEMM runs on the original bf16/fp16 c and b, and the
    decay is applied in fp32 *after* the GEMM, factored per tile:

        exp(g_i - g_j) = rowf_i · mid(p, c) · colf_j      (query tile p after key tile c; all three factors <= 1)
        rowf_i = exp(g_i - g_{start of p}),  mid = exp(g_{start of p} - g_{end of c}),  colf_j = exp(g_{end of c} - g_j)

    The three exponents add up exactly to g_i - g_j and have the same sign, so the error is proportional to
    |g_i - g_j| (the same order as a pairwise evaluation). Diagonal tiles use the pairwise
    exp2(min((g_i - g_j)·log2e, 0)).
    In backward, the side-term GEMM operands dS·(colf·mid) or dS·(rowf·mid) are rounded to the input dtype (DT) once,
    the same rounding point as the FlashAttention-standard (dS ⊙ λE) -> DT; the row/column scales λ·rowf and λ·colf
    are applied in fp32 afterwards.
(B) Differences before log2e. Differences (F_i - F_j, g_i - g_j, tile-anchor differences) are taken in natural-log
    units first, and log2e is applied only inside the exponent (exp2(s·log2e - m), one FFMA). Errors therefore scale
    with |F_i - F_j| and |g_i - g_j| rather than |F| and |g| (shift invariance), as in the official FoX kernel.
(C) Any L >= 1, using masked loads and no padding copy. The monotonicity check of g needs no host synchronisation
    in the default check mode (``torch._assert_async``; see :func:`sisa.backends.set_check_mode`).
(D) Exact Δ. The FlashAttention-standard Δ̂_i = o_DT·do_i accumulates the low-precision rounding of o into the
    sums behind dλ, dg and dF (relative errors of tens to hundreds of percent in dλ on strongly forgetting heads; the
    FlashAttention scheme has the same issue). Instead the dq kernel also accumulates T_i = Σ_j P_ij dP_ij from the
    fp32 P (the fp32 value of do_i·o_i) and evaluates the row contraction Σ_{j<i} P_ij (dP_ij - T_i) u_ij as
    Σ dŜ_ij u_ij + (Δ̂_i - T_i)·Σ P_ij u_ij, where dŜ is the FA-standard dS and every sum is an fp32 FFMA chain. The
    dkv kernel uses **the same decomposition** dS = dŜ' - P'·(T̂ - Δ̂) for dk and the column parts of dg and dF: rows
    and columns must be evaluated by the same formula so that the internal pair cancellation in d log f holds in
    fp32 (see (E)).
    The shift used in exact mode is Δ'' = do·o'', with o'' = Σ DT(pe)·v / Σ DT(pe): an fp32 output stored separately
    by the forward pass and **self-normalised** by the DT weights the PV GEMM actually used. The FA-standard
    Δ̂ = o_DT·do leaves an error of 2^-9·|o||do| in every dŜ; on sharp heads dominated by one off-diagonal entry
    (P_max 0.8 .. 0.9999) that made dq, dc and db 14-150x the ideal error. The weights DT(pe)/Σ DT(pe) of o'' form an
    exactly normalised distribution, and the dominant entry pe = exp2(0) = 1 is exact in DT, so the perturbation is
    proportional to the non-dominant mass: dq, dc and db stay within 1-3x of the ideal error everywhere. (Dividing by
    the fp32 l = Σ pe instead leaves a common bias of ~2^-9, because the DT weights do not sum to l; that was
    rejected.) Cost: one extra row sum in forward and an fp32 copy of o (+4 bytes per element, only when gradients
    are needed). The off-diagonal dq/dc GEMMs use dŜ(Δ'') as is, so no GEMM is added.
    dB uses the same dŜ as dC: the score depends only on Φ_i - Φ_j, so dθ relies on the cancellation between the
    dC (row) and dB (column) contributions, and different Δ for the two would accumulate a mismatch in dθ.
    With EXACT=False the kernels use the FA-standard Δ̂ everywhere (diagonal included) but keep the diagonal exclusion
    of (E) for dg and dF. The public entry point always runs EXACT=True.
(E) Diagonal entries (i = j). On strongly forgetting heads (P_tt -> 1), dS_tt = P_tt(dP_tt - Δ_t) computes a true
    value of size ~ε as the difference of two O(|dP|) numbers, so the rounding of Δ (even an fp32 T, ulp(dP)) and the
    normalisation error of lse survive in it. Treatment:
     · dg and dF: the diagonal contribution cancels exactly (+dS_tt·x - dS_tt·x), so the row sums run over j < t and
       the column sums over i > t only.
     · The dq kernel accumulates per row, in fp32, two sums over the same P̃ values (diagonal excluded):
            D_t = Σ_{j<t} dŜ_tj = Σ_{j<t} P̃_tj (dP_tj - Δ̂_t),     Z_t = Σ_{j<t} P̃_tj
       dŜ is the FA-standard dS already built for the dq GEMM (no extra element-wise work), and every summand has
       the magnitude of dS. The diagonal value P̃_tt·dP_tt comes from the diagonal tile. Everything is then
       **self-normalised by the actual sum of P̃** (Σ_t = P̃_tt + Z_t; lse rounding gives Σ_j P̃_tj = 1 + ν with
       ν ~ u·|lse|):
            T̂_t   = Σ_j P̃_tj dP_tj / Σ_t = Δ̂_t + (P̃_tt(dP_tt - Δ̂_t) + D_t)/Σ_t         (exact Δ)
            dS_tt = P̃_tt(dP_tt - T̂_t) = P̃_tt·(Z_t(dP_tt - Δ̂_t) - D_t)/Σ_t             (Δ-free: Δ̂ cancels)
            row part of dF_t = Σ_{j<t} P̃_tj(dP_tj - T̂_t) = -dS_tt                   (zero row sum by construction)
       dS_tt = P̃_tt·Σ_{j<t} P̃_tj(dP_tt - dP_tj)/Σ consists of ε-sized terms and has fp32 relative precision; the
       low-precision error of Δ̂ and the normalisation error cancel or enter only as relative factors. The
       unnormalised T̃ = Σ P̃ dP would leave a bias -P̃·ν·T in every dS (proportional to |T|, not |dP - T|);
       shifting by dP_tt instead of Δ̂, or not shifting at all, each fails one family of degenerate inputs.
     · Evaluation consistency: the row part of dF is evaluated as D - Z·(T̂ - Δ̂), not as the closed form -dS_tt,
       and the column part (dkv kernel) element-wise as dŜ' - P'·(T̂ - Δ̂). Both use the same decomposition, so the
       two evaluations of each pair (i, j) consist of the same products and d log f (the reverse cumulative sum of
       dF, with internal pair cancellation) is accurate in fp32; the closed form on the row side was an order of
       magnitude worse. The closed form of dS_tt is used only for the vector gradients and the diagonal part of dλ.
       The diagonal parts of dq, dk, dc and db put this dS_tt on the diagonal of a GEMM operand, rounded once to DT
       (the rounding point FlashAttention applies to every dS): the dq kernel does small GEMMs
       dot(diag(dS_tt), k_t) and dot(diag(λ dS_tt), b_t) after its loop, the dkv kernel substitutes DSD in the
       diagonal tile. The diagonal operands of dc and db are the same DT rounding of the same fp32 λ·dS_tt, so the
       dθ cancellation (Φ_i - Φ_j) holds bit for bit. dλ adds dS_tt·cb_tt in fp32.
       Z must be accumulated in the same pass as D: taking it from the forward pass (next to l) sums *different*
       P̃ values, which misaligns rows and columns by ν and degrades d log f.

Structure
---------
 - T x T tiles (T = 64 for bf16/fp16 inputs, T = 32 for fp32 inputs). Three kernels: forward, dq (+Δ̂, T and the row
   contractions) and dkv (+ the final dg and dF sums). A small preparation kernel `_factors` builds rowf and colf
   ((B,H,L) fp32 each) and the maximum monotonicity violation in one pass. The fused layer path
   (:mod:`sisa.kernels.fused_prep`) builds the same factors with the same formula (bit-identical).
 - **The diagonal tile comes before the loop.** The off-diagonal loop may run zero times (first tile); starting the
   accumulator after it makes ptxas serialise every wgmma of the function (warning C7515).
 - The dq key slice BN and the dkv query slice BMQ accept values below T in code, but are frozen to T: smaller
   slices combined with software pipelining produced wrong results. Out-of-range rows read lse = +inf, so P = 0.
 - q/k/v/c/b/g/F/λ/do accept arbitrary strides (last dimension contiguous), e.g. the model's (B,L,H,·).transpose(1,2)
   views. o is allocated in (B,L,H,D) memory layout, so the model's y.transpose(1,2).reshape is a view.
 - Deterministic: no atomics. dq is owned by query tiles; dk/dv/db/dg/dF by key tiles; dλ goes through a per-row
   buffer and a sum. The dq kernel hands (B,H,L) fp32 buffers to the dkv kernel: Δ (exact T or Δ̂), T̂ - Δ̂, dS_tt
   (DSD), the row parts of dg (RG) and dF (RF), and the per-row dλ (RL).
 - Scheduling: causal query tiles carry unequal work, so within one (b, h) the heaviest tiles run first (LPT). (b, h)
   is the slow grid axis, so the CTAs running together read the K/V of only a few heads (L2 locality).
 - fp32 inputs pass tl.dot(input_precision="ieee") explicitly (independent of TRITON_F32_DEFAULT, whose default is
   TF32).

Masking rule
------------
Factors derived from g/F/λ at masked positions must never put inf/NaN into a GEMM operand: every side factor is
clamped to <= 1 (min(·, 0) in the exponent) and out-of-range rows and columns are zeroed with `where` before they are
multiplied. g is never read out of range (indices are clamped to L - 1), so the NaN-propagating clamp
(propagate_nan=ALL) cannot pick up memory beyond L; only a NaN inside the valid range of g reaches the output and
lse (where the fused path's non-finite guard sees it).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .. import backends

__all__ = ["sisa_attention_triton", "attn_forward", "attn_backward"]

NEG = tl.constexpr(float("-inf"))
LOG2E = tl.constexpr(1.4426950408889634)

# Maximum allowed increase of g between neighbouring positions (fp32 cumulative sums of non-positive increments can
# rise by a few ulps).
MONOTONE_TOL = 1e-3


# ============================================================================ factors
@triton.jit
def _factors(G, ROWF, COLF, MONO, sgb, sgh, sgl, L, NT,
             H: tl.constexpr, T: tl.constexpr):
    """rowf_i = exp(g_i - g_{tile start}), colf_j = exp(g_{tile end} - g_j) (both <= 1), max monotonicity violation."""
    p = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    G += b.to(tl.int64) * sgb + h.to(tl.int64) * sgh
    t0 = p * T
    offs = t0 + tl.arange(0, T)
    msk = offs < L
    last = tl.minimum(t0 + T - 1, L - 1)
    g = tl.load(G + tl.minimum(offs, L - 1).to(tl.int64) * sgl)       # out of range: the last value (finite)
    gs = tl.load(G + t0.to(tl.int64) * sgl)
    ge = tl.load(G + last.to(tl.int64) * sgl)
    rowf = tl.exp2(tl.minimum((g - gs) * LOG2E, 0., propagate_nan=tl.PropagateNan.ALL))
    colf = tl.exp2(tl.minimum((ge - g) * LOG2E, 0., propagate_nan=tl.PropagateNan.ALL))
    o = bh.to(tl.int64) * L + offs
    tl.store(ROWF + o, rowf, mask=msk)
    tl.store(COLF + o, colf, mask=msk)
    # monotonicity: max of g_i - g_{i-1} (tile boundaries included)
    prv = tl.load(G + tl.maximum(tl.minimum(offs, L - 1) - 1, 0).to(tl.int64) * sgl)
    inc = tl.where(msk & (offs > 0), g - prv, float("-inf"))
    tl.store(MONO + bh.to(tl.int64) * NT + p, tl.max(inc, 0))


# ============================================================================ paired bf16 conversion
# The forward o'' normalisation (STORE32) reads the bf16 weights used by the PV GEMM back as fp32 and sums them (lb_i).
# When the same bf16 tensor feeds both the GEMM and an upcast, Triton lowers fp32 -> bf16 element by element and then
# re-pairs and shifts the values. The two helpers below produce the same values with
#   cvt.rn.bf16x2.f32   (one instruction per pair, round-to-nearest-even, identical to tensor.to(bf16)) and
#   shl / and           (bf16 -> fp32 is a bit shift, hence exact),
# about 5% fewer instructions in the forward loop, with bit-identical results. Pair order: with pack=2, $1 = element i
# goes to the low 16 bits (Triton's 16-bit pairing convention).
@triton.jit
def _bf16x2(x):
    return tl.inline_asm_elementwise("cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,r,r", [x],
                                     dtype=tl.bfloat16, is_pure=True, pack=2)


@triton.jit
def _bf16x2_f32(x):
    return tl.inline_asm_elementwise("{ shl.b32 $0, $2, 16; and.b32 $1, $2, 0xffff0000; }", "=r,=r,r", [x],
                                     dtype=tl.float32, is_pure=True, pack=2)


# ============================================================================ forward
@triton.jit
def _fwd(Q, K, V, C, Bm, G, FF, ROWF, COLF, Lam, O, LSE, O32,
         sqb, sqh, sql, skb, skh, skl, svb, svh, svl,
         scb, sch, scl, sbb, sbh, sbl, sgb, sgh, sgl, sfb, sfh, sfl, s_lam,
         sob, soh, sol, L, NT,
         H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, T: tl.constexpr,
         SCALE: tl.constexpr, HAS_F: tl.constexpr, PREC: tl.constexpr, STORE32: tl.constexpr):
    p = NT - 1 - tl.program_id(0)            # heavy (late) query tiles first
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    b64 = b.to(tl.int64)
    h64 = h.to(tl.int64)
    Q += b64 * sqb + h64 * sqh
    K += b64 * skb + h64 * skh
    V += b64 * svb + h64 * svh
    C += b64 * scb + h64 * sch
    Bm += b64 * sbb + h64 * sbh
    G += b64 * sgb + h64 * sgh
    FF += b64 * sfb + h64 * sfh
    O += b64 * sob + h64 * soh
    ROWF += bh.to(tl.int64) * L
    COLF += bh.to(tl.int64) * L
    LSE += bh.to(tl.int64) * L
    lam = tl.load(Lam + h * s_lam).to(tl.float32)

    m0 = p * T
    offs_m = m0 + tl.arange(0, T)
    offs_d = tl.arange(0, D)
    offs_s = tl.arange(0, S)
    mrow = offs_m < L
    q = tl.load(Q + offs_m[:, None] * sql + offs_d[None, :], mask=mrow[:, None], other=0.)
    c = tl.load(C + offs_m[:, None] * scl + offs_s[None, :], mask=mrow[:, None], other=0.)
    rowf = tl.load(ROWF + offs_m, mask=mrow, other=0.)
    rs = lam * rowf                                            # row scale λ·rowf (fp32)
    ga = tl.load(G + m0 * sgl)                                 # query-tile anchor g_{m0} (m0 <= every row -> causal)
    if HAS_F:
        fi = tl.load(FF + offs_m * sfl, mask=mrow, other=0.)

    # ---- diagonal tile first (keys m0 .. min(m0+T, L)-1): pairwise decay + causal mask ----
    # The accumulator starts from a wgmma result. With the loop first, the zero initialisation on the zero-trip path
    # would sit inside the later wgmma region and ptxas (C7515) would serialise every wgmma of the function.
    offs_n = m0 + tl.arange(0, T)
    mcol = offs_n < L
    k = tl.load(K + offs_n[:, None] * skl + offs_d[None, :], mask=mcol[:, None], other=0.)
    bb = tl.load(Bm + offs_n[:, None] * sbl + offs_s[None, :], mask=mcol[:, None], other=0.)
    gi = tl.load(G + tl.minimum(offs_m, L - 1) * sgl)
    gj = tl.load(G + tl.minimum(offs_n, L - 1) * sgl)
    valid = (offs_m[:, None] >= offs_n[None, :]) & mcol[None, :]
    dec = tl.where(valid, tl.exp2(tl.minimum((gi[:, None] - gj[None, :]) * LOG2E, 0.,
                                             propagate_nan=tl.PropagateNan.ALL)), 0.)
    qk = tl.dot(q, tl.trans(k), input_precision=PREC)
    if HAS_F:
        fj = tl.load(FF + offs_n * sfl, mask=mcol, other=0.)
        s = qk * SCALE + (fi[:, None] - fj[None, :])
    else:
        s = qk * SCALE
    cb = tl.dot(c, tl.trans(bb), input_precision=PREC)
    s = s + lam * (dec * cb)
    s = tl.where(valid, s, NEG)
    m_new = tl.max(s, 1) * LOG2E
    m_i = tl.where(m_new == NEG, 0., m_new)                   # only out-of-range rows are all -inf
    pe = tl.exp2(s * LOG2E - m_i[:, None])
    l_i = tl.sum(pe, 1)
    v = tl.load(V + offs_n[:, None] * svl + offs_d[None, :], mask=mcol[:, None], other=0.)
    if STORE32 and v.dtype == tl.bfloat16:
        pb = _bf16x2(pe)                                       # == pe.to(bf16), bit for bit (paired conversion)
    else:
        pb = pe.to(v.dtype)
    acc = tl.dot(pb, v, input_precision=PREC)
    if STORE32:
        if v.dtype == tl.bfloat16:
            lb_i = tl.sum(_bf16x2_f32(pb), 1)                  # == pb.to(fp32), bit for bit
        else:
            lb_i = tl.sum(pb.to(tl.float32), 1)                # sum of the DT weights the PV GEMM used (o'' norm)

    # ---- earlier tiles (all keys j < m0): decay = rowf_i · mid · colf_j, applied in fp32 after the GEMM ----
    for n0 in range(0, m0, T):
        offs_n = n0 + tl.arange(0, T)
        k = tl.load(K + offs_n[:, None] * skl + offs_d[None, :])
        bb = tl.load(Bm + offs_n[:, None] * sbl + offs_s[None, :])
        ge = tl.load(G + (n0 + T - 1) * sgl)
        mid = tl.exp2(tl.minimum((ga - ge) * LOG2E, 0., propagate_nan=tl.PropagateNan.ALL))
        cf = tl.load(COLF + offs_n) * mid
        qk = tl.dot(q, tl.trans(k), input_precision=PREC)
        if HAS_F:
            fj = tl.load(FF + offs_n * sfl)
            s = qk * SCALE + (fi[:, None] - fj[None, :])
        else:
            s = qk * SCALE
        cb = tl.dot(c, tl.trans(bb), input_precision=PREC)
        s = s + rs[:, None] * (cb * cf[None, :])
        m_new = tl.maximum(m_i, tl.max(s, 1) * LOG2E)
        pe = tl.exp2(s * LOG2E - m_new[:, None])
        corr = tl.exp2(m_i - m_new)
        l_i = l_i * corr + tl.sum(pe, 1)
        v = tl.load(V + offs_n[:, None] * svl + offs_d[None, :])
        if STORE32 and v.dtype == tl.bfloat16:
            pb = _bf16x2(pe)
        else:
            pb = pe.to(v.dtype)
        acc = acc * corr[:, None] + tl.dot(pb, v, input_precision=PREC)
        if STORE32:
            if v.dtype == tl.bfloat16:
                lb_i = lb_i * corr + tl.sum(_bf16x2_f32(pb), 1)
            else:
                lb_i = lb_i * corr + tl.sum(pb.to(tl.float32), 1)
        m_i = m_new

    l_safe = tl.where(l_i == 0., 1., l_i)
    if STORE32:
        # For the backward shift Δ'' = do·o'' (module docstring (D)): o'' = Σ DT(pe)·v / Σ DT(pe). Numerator and
        # denominator use the same weights, so the normalisation is exact (Σ = 1). acc/l (fp32 pe in the denominator)
        # would leave the common bias "sum of DT-rounded weights != l" of ~2^-9 in Δ. Stored (B,H,L,D) contiguous fp32.
        lb_safe = tl.where(lb_i == 0., 1., lb_i)
        tl.store(O32 + bh.to(tl.int64) * L * D + offs_m[:, None] * D + offs_d[None, :], acc / lb_safe[:, None],
                 mask=mrow[:, None])
    acc = acc / l_safe[:, None]
    tl.store(O + offs_m[:, None] * sol + offs_d[None, :], acc.to(O.dtype.element_ty), mask=mrow[:, None])
    tl.store(LSE + offs_m, m_i + tl.log2(l_safe), mask=mrow)


# ============================================================================ backward: dq
@triton.jit
def _bwd_dq(Q, K, V, C, Bm, G, FF, ROWF, COLF, Lam, O, DO, LSE, O32,
            DQ, DC, DELTA, DDEL, RG, RL, RF, DSD,
            sqb, sqh, sql, skb, skh, skl, svb, svh, svl,
            scb, sch, scl, sbb, sbh, sbl, sgb, sgh, sgl, sfb, sfh, sfl, s_lam,
            sob, soh, sol, sdob, sdoh, sdol,
            sdqb, sdqh, sdql, sdcb, sdch, sdcl, L, NT,
            H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, T: tl.constexpr, BN: tl.constexpr,
            SCALE: tl.constexpr, HAS_F: tl.constexpr, EXACT: tl.constexpr, PREC: tl.constexpr,
            HAS32: tl.constexpr):
    """Owned by a query tile: dq, dc, Δ (-> dkv), row parts of dg and dF (diagonal excluded) and dλ, and in exact mode
    the diagonal dS_tt (-> dkv). Diagonal tile first (avoids wgmma serialisation). Diagonal handling: module
    docstring (E)."""
    p = NT - 1 - tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    b64 = b.to(tl.int64)
    h64 = h.to(tl.int64)
    Q += b64 * sqb + h64 * sqh
    K += b64 * skb + h64 * skh
    V += b64 * svb + h64 * svh
    C += b64 * scb + h64 * sch
    Bm += b64 * sbb + h64 * sbh
    G += b64 * sgb + h64 * sgh
    FF += b64 * sfb + h64 * sfh
    O += b64 * sob + h64 * soh
    DO += b64 * sdob + h64 * sdoh
    DQ += b64 * sdqb + h64 * sdqh
    DC += b64 * sdcb + h64 * sdch
    rb = bh.to(tl.int64) * L
    lam = tl.load(Lam + h * s_lam).to(tl.float32)

    m0 = p * T
    offs_m = m0 + tl.arange(0, T)
    offs_d = tl.arange(0, D)
    offs_s = tl.arange(0, S)
    mrow = offs_m < L
    q = tl.load(Q + offs_m[:, None] * sql + offs_d[None, :], mask=mrow[:, None], other=0.)
    do = tl.load(DO + offs_m[:, None] * sdol + offs_d[None, :], mask=mrow[:, None], other=0.)
    o = tl.load(O + offs_m[:, None] * sol + offs_d[None, :], mask=mrow[:, None], other=0.).to(tl.float32)
    if HAS32:
        # Exact-mode shift Δ'' = o''·do (module docstring (D)). Where o'' agrees with the DT o to within 2^-15
        # relative, the DT value is used: the numerator of o'' is a tensor-core accumulation, not an IEEE sum, and on
        # degenerate inputs where o = v is exact in bf16 its noise (tens of ulps) exceeds the DT rounding. 2^-15 is
        # 5x above that noise and 64x below the DT rounding being removed (half an ulp, 2^-9); on ordinary data
        # ~0.8% of the elements snap to DT, each by <= 3e-5 relative (the gain on sharp heads is unchanged).
        o32 = tl.load(O32 + bh.to(tl.int64) * L * D + offs_m[:, None] * D + offs_d[None, :], mask=mrow[:, None],
                      other=0.)
        o = tl.where(tl.abs(o32 - o) <= 3.0517578125e-05 * tl.abs(o), o, o32)
    dhat = tl.sum(o * do.to(tl.float32), 1)     # exact mode: Δ'' (above) / FA mode: Δ̂ = o_DT·do
    c = tl.load(C + offs_m[:, None] * scl + offs_s[None, :], mask=mrow[:, None], other=0.)
    lse = tl.load(LSE + rb + offs_m, mask=mrow, other=float("inf"))   # out-of-range rows: P = exp2(s - inf) = 0
    rowf = tl.load(ROWF + rb + offs_m, mask=mrow, other=0.)
    rs = lam * rowf
    ga = tl.load(G + m0 * sgl)
    if HAS_F:
        fi = tl.load(FF + offs_m * sfl, mask=mrow, other=0.)

    # ---- diagonal block (keys m0 .. m0+T-1, pairwise decay) first, statically unrolled (T/BN slices of width BN) ----
    # The accumulators start from wgmma results in straight-line code before the loop (with the loop first, the zero
    # initialisation on the zero-trip path lands inside the later wgmma pipeline and ptxas C7515 serialises every
    # wgmma of the function). Diagonal entries (i = j) cancel analytically in dg and dF, so they are excluded from
    # both the row and the column sums. In exact mode they are also excluded from the vector gradients and dλ in the
    # GEMMs and added after the loop with the Δ-free dS_tt = P̃_tt(Z(dP_tt - Δ̂) - D)/Σ (module docstring (E)).
    gi = tl.load(G + tl.minimum(offs_m, L - 1) * sgl)
    dq = tl.zeros([T, D], tl.float32)
    dc = tl.zeros([T, S], tl.float32)
    xa = tl.zeros([T], tl.float32)      # Σ_{j<t} dŜ·u
    ya = tl.zeros([T], tl.float32)      # Σ_{j<t} P·u                          (exact)
    wa = tl.zeros([T], tl.float32)      # D = Σ_{j<t} dŜ  (exact: input of T̂ and dS_tt; FA: row part of dF)
    za = tl.zeros([T], tl.float32)      # Z = Σ_{j<t} P                        (exact)
    ptt = tl.zeros([T], tl.float32)     # P_tt                                 (exact)
    dptt = tl.zeros([T], tl.float32)    # dP_tt                                (exact)
    xd = tl.zeros([T], tl.float32)      # diagonal dλ part: exact = cb_tt, FA = dŜ_tt·cb_tt
    for jj in tl.static_range(T // BN):
        offs_n = m0 + jj * BN + tl.arange(0, BN)
        mcol = offs_n < L
        k = tl.load(K + offs_n[:, None] * skl + offs_d[None, :], mask=mcol[:, None], other=0.)
        v = tl.load(V + offs_n[:, None] * svl + offs_d[None, :], mask=mcol[:, None], other=0.)
        bb = tl.load(Bm + offs_n[:, None] * sbl + offs_s[None, :], mask=mcol[:, None], other=0.)
        gj = tl.load(G + tl.minimum(offs_n, L - 1) * sgl)
        valid = (offs_m[:, None] >= offs_n[None, :]) & mcol[None, :] & mrow[:, None]
        eye = offs_m[:, None] == offs_n[None, :]            # diagonal entries (harmless on out-of-range rows: pr = 0)
        dec = tl.where(valid, tl.exp2(tl.minimum((gi[:, None] - gj[None, :]) * LOG2E, 0.,
                                                 propagate_nan=tl.PropagateNan.ALL)), 0.)
        qk = tl.dot(q, tl.trans(k), input_precision=PREC)
        if HAS_F:
            fj = tl.load(FF + offs_n * sfl, mask=mcol, other=0.)
            s = qk * SCALE + (fi[:, None] - fj[None, :])
        else:
            s = qk * SCALE
        cb = tl.dot(c, tl.trans(bb), input_precision=PREC)
        u = dec * cb
        s = s + lam * u
        pr = tl.where(valid, tl.exp2(s * LOG2E - lse[:, None]), 0.)
        dp = tl.dot(do, tl.trans(v), input_precision=PREC)
        ds = pr * (dp - dhat[:, None])
        dso = tl.where(eye, 0., ds)                         # off-diagonal (j < t) only; pr already carries the mask
        if EXACT:
            dq += tl.dot(dso.to(q.dtype), k, input_precision=PREC)
            dc += tl.dot((dso * (lam * dec)).to(q.dtype), bb, input_precision=PREC)
            ptt += tl.sum(tl.where(eye, pr, 0.), 1)
            dptt += tl.sum(tl.where(eye, dp, 0.), 1)
            xd += tl.sum(tl.where(eye, cb, 0.), 1)
            po = tl.where(eye, 0., pr)
            za += tl.sum(po, 1)
            ya += tl.sum(po * u, 1)
        else:
            dq += tl.dot(ds.to(q.dtype), k, input_precision=PREC)          # FA standard: diagonal included
            dc += tl.dot((ds * (lam * dec)).to(q.dtype), bb, input_precision=PREC)
            xd += tl.sum(tl.where(eye, ds * u, 0.), 1)
        if EXACT or HAS_F:
            wa += tl.sum(dso, 1)
        xa += tl.sum(dso * u, 1)

    # ---- earlier tiles (keys j < m0, all off-diagonal): side-term parts as u' = u / rowf, row scale at the end ----
    dco = tl.zeros([T, S], tl.float32)
    xo = tl.zeros([T], tl.float32)
    yo = tl.zeros([T], tl.float32)
    for n0 in range(0, m0, BN):
        offs_n = n0 + tl.arange(0, BN)
        k = tl.load(K + offs_n[:, None] * skl + offs_d[None, :])
        v = tl.load(V + offs_n[:, None] * svl + offs_d[None, :])
        bb = tl.load(Bm + offs_n[:, None] * sbl + offs_s[None, :])
        ge = tl.load(G + ((n0 // T) * T + T - 1) * sgl)          # end of the key's T-block (colf anchor)
        mid = tl.exp2(tl.minimum((ga - ge) * LOG2E, 0., propagate_nan=tl.PropagateNan.ALL))
        cf = tl.load(COLF + rb + offs_n) * mid
        qk = tl.dot(q, tl.trans(k), input_precision=PREC)
        if HAS_F:
            fj = tl.load(FF + offs_n * sfl)
            s = qk * SCALE + (fi[:, None] - fj[None, :])
        else:
            s = qk * SCALE
        cb = tl.dot(c, tl.trans(bb), input_precision=PREC)
        uc = cb * cf[None, :]
        s = s + rs[:, None] * uc
        pr = tl.exp2(s * LOG2E - lse[:, None])
        dp = tl.dot(do, tl.trans(v), input_precision=PREC)
        ds = pr * (dp - dhat[:, None])
        dq += tl.dot(ds.to(q.dtype), k, input_precision=PREC)
        dco += tl.dot((ds * cf[None, :]).to(q.dtype), bb, input_precision=PREC)
        xo += tl.sum(ds * uc, 1)
        if EXACT:
            yo += tl.sum(pr * uc, 1)
            za += tl.sum(pr, 1)
        if EXACT or HAS_F:
            wa += tl.sum(ds, 1)

    dc = dc + rs[:, None] * dco
    xa = xa + rowf * xo
    if EXACT:
        ya = ya + rowf * yo
        # Module docstring (E): self-normalise by the actual sum Σ = P_tt + Z of P̃; from the Δ̂-shifted sum
        # D = Σ_{j<t} P̃(dP - Δ̂):
        #   T̂ = Σ_j P̃ dP / Σ_j P̃ = Δ̂ + (P̃_tt(dP_tt - Δ̂) + D)/Σ          (exact Δ)
        #   dS_tt = P̃_tt(dP_tt - T̂) = P̃_tt·(Z·(dP_tt - Δ̂) - D)/Σ          (Δ-free: Δ̂ cancels on both sides)
        #   row part of dF = Σ_{j<t} P̃(dP - T̂) = -dS_tt                   (zero row sum by construction)
        ssum = ptt + za
        ssum = tl.where(ssum > 0., ssum, 1.)   # out-of-range rows (ptt = za = 0)
        x0 = dptt - dhat
        tt = dhat + (ptt * x0 + wa) / ssum     # T̂ (exact Δ)
        dstt = ptt * (za * x0 - wa) / ssum     # diagonal dS_tt, Δ-free
        dl_row = xa + (dhat - tt) * ya         # Σ_{j<t} P̃(dP - T̂)u
        # Diagonal parts (u_tt = cb_tt, decay 1): dq += dS_tt·k_t, dc += λ·dS_tt·b_t as diagonal-matrix GEMMs (one
        # DT rounding of the operand, the same DT value as the db diagonal in dkv). Faster than an fp32 element-wise
        # add (no accumulator layout conversion or spills). These wgmmas follow the loop, but the accumulators are
        # already defined by the diagonal-tile wgmma, so there is no C7515 serialisation.
        kd = tl.load(K + offs_m[:, None] * skl + offs_d[None, :], mask=mrow[:, None], other=0.)
        bd = tl.load(Bm + offs_m[:, None] * sbl + offs_s[None, :], mask=mrow[:, None], other=0.)
        eyem = offs_m[:, None] == offs_m[None, :]
        dmat = tl.where(eyem, dstt[:, None], 0.)
        dq += tl.dot(dmat.to(q.dtype), kd, input_precision=PREC)
        dc += tl.dot((dmat * lam).to(q.dtype), bd, input_precision=PREC)
        dd = tt - dhat                         # T̂ - Δ̂
        tl.store(DELTA + rb + offs_m, dhat, mask=mrow)    # dkv builds dŜ' = P'(dP' - Δ̂), then dS = dŜ' - P'(T̂ - Δ̂)
        tl.store(DDEL + rb + offs_m, dd, mask=mrow)
        tl.store(DSD + rb + offs_m, dstt, mask=mrow)
        tl.store(RL + rb + offs_m, dl_row + dstt * xd, mask=mrow)
        if HAS_F:
            # row part of dF = Σ_{j<t} (dŜ - P·(T̂ - Δ̂)) = D - Z·(T̂ - Δ̂)  (= -dS_tt; the **same decomposition**
            # as the column side, for evaluation consistency)
            tl.store(RF + rb + offs_m, wa - za * dd, mask=mrow)
    else:
        dl_row = xa
        tl.store(DELTA + rb + offs_m, dhat, mask=mrow)
        tl.store(RL + rb + offs_m, xa + xd, mask=mrow)        # FA-standard dλ: diagonal included
        if HAS_F:
            tl.store(RF + rb + offs_m, wa, mask=mrow)
    tl.store(DQ + offs_m[:, None] * sdql + offs_d[None, :], (dq * SCALE).to(DQ.dtype.element_ty), mask=mrow[:, None])
    tl.store(DC + offs_m[:, None] * sdcl + offs_s[None, :], dc.to(DC.dtype.element_ty), mask=mrow[:, None])
    tl.store(RG + rb + offs_m, lam * dl_row, mask=mrow)


# ============================================================================ backward: dk dv db dg dF
@triton.jit
def _bwd_dkv(Q, K, V, C, Bm, G, FF, ROWF, COLF, Lam, DO, LSE, DELTA, DDEL, RG, RF, DSD,
             DK, DV, DB, DG, DF,
             sqb, sqh, sql, skb, skh, skl, svb, svh, svl,
             scb, sch, scl, sbb, sbh, sbl, sgb, sgh, sgl, sfb, sfh, sfl, s_lam,
             sdob, sdoh, sdol, sdkb, sdkh, sdkl, sdvb, sdvh, sdvl, sdbb, sdbh, sdbl,
             L, NT,
             H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, T: tl.constexpr, BMQ: tl.constexpr,
             SCALE: tl.constexpr, HAS_F: tl.constexpr, EXACT: tl.constexpr, EVEN: tl.constexpr,
             PREC: tl.constexpr):
    """Owned by a key tile (transposed orientation [key, query]): dk, dv, db, and dg = row part (RG) - column part,
    dF = row part (RF) - column part (both excluding the diagonal, module docstring (E)). Diagonal tile first.
    Exact-mode Δ assignment (cancelling pairs share Δ): DELTA = Δ̂ builds the FA-standard dŜ' = P'(dP' - Δ̂), and the
    exact dS = dŜ' - P'·(T̂ - Δ̂) (DDEL = T̂ - Δ̂) feeds dk and the column parts of dg and dF (the row parts use the
    same decomposition; evaluation consistency). dB uses dŜ' as is, the same as dC (dq kernel, Δ̂): the score depends
    only on Φ_i - Φ_j, so Σ_s dΦ_s = 0 and dθ_t = Σ_{s>=t} dΦ_s relies on this cancellation; different Δ for dC and
    dB would accumulate a mismatch. The dq·dk pair has no such cancellation structure.
    The diagonal dS_tt is the same fp32 value the dq kernel produced (DSD), so the dc and db diagonals are the same
    bits."""
    cidx = tl.program_id(0)                 # early key tiles are the heaviest (most later queries)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    b64 = b.to(tl.int64)
    h64 = h.to(tl.int64)
    Q += b64 * sqb + h64 * sqh
    K += b64 * skb + h64 * skh
    V += b64 * svb + h64 * svh
    C += b64 * scb + h64 * sch
    Bm += b64 * sbb + h64 * sbh
    G += b64 * sgb + h64 * sgh
    FF += b64 * sfb + h64 * sfh
    DO += b64 * sdob + h64 * sdoh
    DK += b64 * sdkb + h64 * sdkh
    DV += b64 * sdvb + h64 * sdvh
    DB += b64 * sdbb + h64 * sdbh
    rb = bh.to(tl.int64) * L
    lam = tl.load(Lam + h * s_lam).to(tl.float32)

    n0 = cidx * T
    offs_n = n0 + tl.arange(0, T)
    offs_d = tl.arange(0, D)
    offs_s = tl.arange(0, S)
    mcol = offs_n < L
    k = tl.load(K + offs_n[:, None] * skl + offs_d[None, :], mask=mcol[:, None], other=0.)
    v = tl.load(V + offs_n[:, None] * svl + offs_d[None, :], mask=mcol[:, None], other=0.)
    bb = tl.load(Bm + offs_n[:, None] * sbl + offs_s[None, :], mask=mcol[:, None], other=0.)
    colf = tl.load(COLF + rb + offs_n, mask=mcol, other=0.)
    rk = lam * colf                                             # column scale λ·colf (fp32)
    ge = tl.load(G + tl.minimum(n0 + T - 1, L - 1) * sgl)       # key-tile anchor g_{end}
    if HAS_F:
        fj = tl.load(FF + offs_n * sfl, mask=mcol, other=0.)

    # ---- diagonal block (queries n0 .. n0+T-1, pairwise decay) first, statically unrolled (T/BMQ slices) ----
    gj = tl.load(G + tl.minimum(offs_n, L - 1) * sgl)
    dv = tl.zeros([T, D], tl.float32)
    dk = tl.zeros([T, D], tl.float32)
    db = tl.zeros([T, S], tl.float32)
    dgu = tl.zeros([T], tl.float32)
    dfs = tl.zeros([T], tl.float32)
    for jj in tl.static_range(T // BMQ):
        offs_m = n0 + jj * BMQ + tl.arange(0, BMQ)
        mrow = offs_m < L
        q = tl.load(Q + offs_m[:, None] * sql + offs_d[None, :], mask=mrow[:, None], other=0.)
        do = tl.load(DO + offs_m[:, None] * sdol + offs_d[None, :], mask=mrow[:, None], other=0.)
        c = tl.load(C + offs_m[:, None] * scl + offs_s[None, :], mask=mrow[:, None], other=0.)
        lse = tl.load(LSE + rb + offs_m, mask=mrow, other=float("inf"))
        dl = tl.load(DELTA + rb + offs_m, mask=mrow, other=0.)
        gi = tl.load(G + tl.minimum(offs_m, L - 1) * sgl)
        valid = (offs_m[None, :] >= offs_n[:, None]) & mrow[None, :]
        strict = (offs_m[None, :] > offs_n[:, None]) & mrow[None, :]
        decT = tl.where(valid, tl.exp2(tl.minimum((gi[None, :] - gj[:, None]) * LOG2E, 0.,
                                                  propagate_nan=tl.PropagateNan.ALL)), 0.)
        qkT = tl.dot(k, tl.trans(q), input_precision=PREC)
        if HAS_F:
            fi = tl.load(FF + offs_m * sfl, mask=mrow, other=0.)
            sT = qkT * SCALE + (fi[None, :] - fj[:, None])
        else:
            sT = qkT * SCALE
        cbT = tl.dot(bb, tl.trans(c), input_precision=PREC)
        uT = decT * cbT
        sT = sT + lam * uT
        pT = tl.where(valid, tl.exp2(sT * LOG2E - lse[None, :]), 0.)
        dpT = tl.dot(v, tl.trans(do), input_precision=PREC)
        dsT = pT * (dpT - dl[None, :])                    # exact: dŜ' (Δ̂) / FA: dŜ'
        if EXACT:
            dd = tl.load(DDEL + rb + offs_m, mask=mrow, other=0.)
            dsh = dsT                                      # dB uses the same Δ̂ as dC (docstring)
            dsT = dsT - pT * dd[None, :]                   # exact dS = dŜ' - P'·(T̂ - Δ̂)
        dsTo = tl.where(strict, dsT, 0.)
        dv += tl.dot(pT.to(q.dtype), do, input_precision=PREC)
        if EXACT:
            # the diagonal entries are replaced by the Δ-free dS_tt of the dq kernel (DSD), rounded once to DT as a
            # GEMM operand (the FA-standard rounding point)
            eyeT = (offs_m[None, :] == offs_n[:, None]) & mrow[None, :]
            dstt = tl.load(DSD + rb + offs_n, mask=mcol, other=0.)
            dsTd = tl.where(eyeT, dstt[:, None], dsTo)
            dk += tl.dot(dsTd.to(q.dtype), q, input_precision=PREC)
            dsb = tl.where(eyeT, dstt[:, None], tl.where(strict, dsh, 0.))
        else:
            dk += tl.dot(dsT.to(q.dtype), q, input_precision=PREC)       # FA standard: diagonal included
            dsb = dsT
        db += tl.dot((dsb * (lam * decT)).to(q.dtype), c, input_precision=PREC)
        dgu += tl.sum(dsTo * uT, 1)
        if HAS_F:
            dfs += tl.sum(dsTo, 1)

    # ---- later queries (every i after the key tile, BMQ slices): side terms as u / colf, column scale at the end ----
    dbo = tl.zeros([T, S], tl.float32)
    dgo = tl.zeros([T], tl.float32)
    for m0 in range(n0 + T, L, BMQ):
        offs_m = m0 + tl.arange(0, BMQ)
        if EVEN:
            q = tl.load(Q + offs_m[:, None] * sql + offs_d[None, :])
            do = tl.load(DO + offs_m[:, None] * sdol + offs_d[None, :])
            c = tl.load(C + offs_m[:, None] * scl + offs_s[None, :])
            lse = tl.load(LSE + rb + offs_m)
            dl = tl.load(DELTA + rb + offs_m)
            rowf = tl.load(ROWF + rb + offs_m)
            if HAS_F:
                fi = tl.load(FF + offs_m * sfl)
        else:
            mrow = offs_m < L
            q = tl.load(Q + offs_m[:, None] * sql + offs_d[None, :], mask=mrow[:, None], other=0.)
            do = tl.load(DO + offs_m[:, None] * sdol + offs_d[None, :], mask=mrow[:, None], other=0.)
            c = tl.load(C + offs_m[:, None] * scl + offs_s[None, :], mask=mrow[:, None], other=0.)
            lse = tl.load(LSE + rb + offs_m, mask=mrow, other=float("inf"))   # out-of-range queries: P = 0
            dl = tl.load(DELTA + rb + offs_m, mask=mrow, other=0.)
            rowf = tl.load(ROWF + rb + offs_m, mask=mrow, other=0.)
            if HAS_F:
                fi = tl.load(FF + offs_m * sfl, mask=mrow, other=0.)
        gm = tl.load(G + ((m0 // T) * T) * sgl)                  # start of the query's T-block (rowf anchor)
        mid = tl.exp2(tl.minimum((gm - ge) * LOG2E, 0., propagate_nan=tl.PropagateNan.ALL))
        rq = rowf * mid
        qkT = tl.dot(k, tl.trans(q), input_precision=PREC)
        if HAS_F:
            sT = qkT * SCALE + (fi[None, :] - fj[:, None])
        else:
            sT = qkT * SCALE
        cbT = tl.dot(bb, tl.trans(c), input_precision=PREC)
        ucT = cbT * rq[None, :]
        sT = sT + rk[:, None] * ucT
        pT = tl.exp2(sT * LOG2E - lse[None, :])
        dpT = tl.dot(v, tl.trans(do), input_precision=PREC)
        dsT = pT * (dpT - dl[None, :])                    # dŜ' (Δ̂ = DELTA)
        dsb = dsT                                          # dB uses the same Δ̂ as dC (docstring)
        if EXACT:
            if EVEN:
                dd = tl.load(DDEL + rb + offs_m)
            else:
                dd = tl.load(DDEL + rb + offs_m, mask=mrow, other=0.)
            dsT = dsT - pT * dd[None, :]                   # exact dS = dŜ' - P'·(T̂ - Δ̂)
        dv += tl.dot(pT.to(q.dtype), do, input_precision=PREC)
        dk += tl.dot(dsT.to(q.dtype), q, input_precision=PREC)
        dgo += tl.sum(dsT * ucT, 1)
        if HAS_F:
            dfs += tl.sum(dsT, 1)
        dbo += tl.dot((dsb * rq[None, :]).to(q.dtype), c, input_precision=PREC)

    db = db + rk[:, None] * dbo
    dgu = dgu + colf * dgo
    tl.store(DK + offs_n[:, None] * sdkl + offs_d[None, :], (dk * SCALE).to(DK.dtype.element_ty), mask=mcol[:, None])
    tl.store(DV + offs_n[:, None] * sdvl + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=mcol[:, None])
    tl.store(DB + offs_n[:, None] * sdbl + offs_s[None, :], db.to(DB.dtype.element_ty), mask=mcol[:, None])
    rg = tl.load(RG + rb + offs_n, mask=mcol, other=0.)
    tl.store(DG + rb + offs_n, rg - lam * dgu, mask=mcol)
    if HAS_F:
        rf = tl.load(RF + rb + offs_n, mask=mcol, other=0.)
        tl.store(DF + rb + offs_n, rf - dfs, mask=mcol)


# ============================================================================ host cores (no autograd)
def _st4(x):
    return x.stride(0), x.stride(1), x.stride(2)


def _like(x):
    return torch.empty_like(x) if x.stride(-1) == 1 else torch.empty(x.shape, device=x.device, dtype=x.dtype)


def _check_monotone(mono: torch.Tensor, mode: str) -> None:
    if mode == "off":
        return
    if mode == "sync":
        mx = float(mono.max())
        if not mx <= MONOTONE_TOL:
            raise ValueError(f"g must be non-increasing along the sequence (max increase {mx:.3e} > {MONOTONE_TOL})")
    else:
        torch._assert_async(mono.max() <= MONOTONE_TOL, "SISA attention: g must be non-increasing along the sequence")


def attn_forward(q, k, v, c, b, g, lam, fgate, factors=None, check=None, want32=False):
    """Forward core (no autograd).

    ``factors=(rowf, colf)`` (built by the fused preparation, 64-row blocks) skips the `_factors` kernel and the
    monotonicity check. ``check`` overrides the global check mode. ``want32=True`` also returns the self-normalised
    fp32 o'' = Σ DT(pe)v / Σ DT(pe) as a contiguous (B,H,L,D) tensor, used by the exact-mode backward (for fp32
    inputs, o itself).

    Returns ``(o, lse, rowf, colf, cfg, o32 or None)``; o is a (B,H,L,D) tensor in (B,L,H,D) memory layout.
    """
    B, H, L, D = q.shape
    S = c.shape[-1]
    dev = q.device
    cfg = backends.attention_config(dev, q.dtype)
    T = cfg["T"]
    NT = triton.cdiv(L, T)
    prec = "ieee" if q.dtype == torch.float32 else "tf32"
    has_f = fgate is not None
    Fp = fgate if has_f else g
    if factors is None:
        rowf = torch.empty((B, H, L), device=dev, dtype=torch.float32)
        colf = torch.empty((B, H, L), device=dev, dtype=torch.float32)
        mono = torch.empty((B * H * NT,), device=dev, dtype=torch.float32)
        _factors[(NT, B * H)](g, rowf, colf, mono, *_st4(g), L, NT, H, T, num_warps=2)
        if L > 1:
            _check_monotone(mono, backends.get_check_mode() if check is None else check)
    else:
        rowf, colf = factors
        assert T == 64 and rowf.is_contiguous() and colf.is_contiguous()
    # o in (B,L,H,D) memory layout, so the model's transpose(1,2).reshape is a view
    o = torch.empty_strided((B, H, L, D), (L * H * D, D, H * D, 1), device=dev, dtype=q.dtype)
    lse = torch.empty((B, H, L), device=dev, dtype=torch.float32)
    store32 = bool(want32) and q.dtype != torch.float32
    o32 = torch.empty((B, H, L, D), device=dev, dtype=torch.float32) if store32 else None
    _fwd[(NT, B * H)](
        q, k, v, c, b, g, Fp, rowf, colf, lam, o, lse, o32 if store32 else lse,
        *_st4(q), *_st4(k), *_st4(v), *_st4(c), *_st4(b), *_st4(g), *_st4(Fp), lam.stride(0),
        *_st4(o), L, NT, H, D, S, T, 1.0 / math.sqrt(D), has_f, prec, store32,
        num_warps=cfg["nw"], num_stages=cfg["ns_f"])
    if want32 and not store32:
        o32 = o                                       # fp32 inputs: o itself is fp32
    return o, lse, rowf, colf, cfg, o32


def attn_backward(q, k, v, c, b, g, fgate, rowf, colf, lam, o, lse, do, exact=True,
                  dq=None, dk=None, dv=None, dc=None, db=None, o32=None):
    """Backward core (no autograd).

    dq .. db may be caller buffers of any stride and dtype (e.g. views of the fused layer's d_proj, fp32 dc/db).
    ``o32`` is the self-normalised fp32 o'' of ``attn_forward(want32=True)`` (exact-mode Δ''; None means
    Δ̂ = o·do). Returns ``(dq, dk, dv, dc, db, dg, dlam (H,), dF or None)``.
    """
    B, H, L, D = q.shape
    S = c.shape[-1]
    dev = q.device
    cfg = backends.attention_config(dev, q.dtype)
    T = cfg["T"]
    NT = triton.cdiv(L, T)
    prec = "ieee" if q.dtype == torch.float32 else "tf32"
    has_f = fgate is not None
    Fp = fgate if has_f else g
    if do.stride(-1) != 1:
        do = do.contiguous()
    dq = _like(q) if dq is None else dq
    dk = _like(k) if dk is None else dk
    dv = _like(v) if dv is None else dv
    dc = _like(c) if dc is None else dc
    db = _like(b) if db is None else db
    f32 = dict(device=dev, dtype=torch.float32)
    delta = torch.empty((B, H, L), **f32)
    ddel = torch.empty((B, H, L), **f32) if exact else delta
    dsd = torch.empty((B, H, L), **f32) if exact else delta
    rg = torch.empty((B, H, L), **f32)
    rl = torch.empty((B, H, L), **f32)
    rf = torch.empty((B, H, L), **f32) if has_f else rg
    dg = torch.empty((B, H, L), **f32)
    dF = torch.empty((B, H, L), **f32) if has_f else dg
    sc = 1.0 / math.sqrt(D)
    has32 = bool(exact) and o32 is not None and o32 is not o
    if has32:
        assert o32.shape == (B, H, L, D) and o32.is_contiguous() and o32.dtype == torch.float32
    _bwd_dq[(NT, B * H)](
        q, k, v, c, b, g, Fp, rowf, colf, lam, o, do, lse, o32 if has32 else lse,
        dq, dc, delta, ddel, rg, rl, rf, dsd,
        *_st4(q), *_st4(k), *_st4(v), *_st4(c), *_st4(b), *_st4(g), *_st4(Fp), lam.stride(0),
        *_st4(o), *_st4(do), *_st4(dq), *_st4(dc), L, NT,
        H, D, S, T, cfg["bn_dq"], sc, has_f, bool(exact), prec, has32,
        num_warps=cfg["nw"], num_stages=cfg["ns_dq"])
    _bwd_dkv[(NT, B * H)](
        q, k, v, c, b, g, Fp, rowf, colf, lam, do, lse, delta, ddel, rg, rf, dsd,
        dk, dv, db, dg, dF,
        *_st4(q), *_st4(k), *_st4(v), *_st4(c), *_st4(b), *_st4(g), *_st4(Fp), lam.stride(0),
        *_st4(do), *_st4(dk), *_st4(dv), *_st4(db), L, NT,
        H, D, S, T, cfg["bm_dkv"], sc, has_f, bool(exact), (L % cfg["bm_dkv"] == 0), prec,
        num_warps=cfg["nw"], num_stages=cfg["ns_dkv"] if db.dtype == torch.float32 else cfg["ns_dkv_lp"])
    # dλ = Σ_(b,t) of the per-row parts. These have mixed signs and cancel by a factor of ~sqrt(B·L), so an fp32 sum
    # would have relative error ~log(n)·u32·sqrt(BL); sum in fp64 and round to fp32 once.
    dlam = rl.sum(dim=(0, 2), dtype=torch.float64).float()
    return dq, dk, dv, dc, db, dg, dlam, (dF if has_f else None)


# ============================================================================ autograd
class _SISAAttentionFn(torch.autograd.Function):
    """Autograd wrapper of the Triton cores. ``exact`` selects the exact-Δ backward (always True in the public API;
    False gives the FlashAttention-standard Δ̂ and exists for comparisons in the test suite)."""

    @staticmethod
    def forward(ctx, q, k, v, c, b, g, lam, fgate, exact):
        need = bool(exact) and any(ctx.needs_input_grad)
        o, lse, rowf, colf, _, o32 = attn_forward(q, k, v, c, b, g, lam, fgate, want32=need)
        has_f = fgate is not None
        # exact mode saves the fp32 o'' for Δ'', FA mode the DT o for the FA-standard Δ̂
        ctx.save_for_backward(q, k, v, c, b, g, fgate if has_f else g, rowf, colf, lam, o, lse,
                              o32 if (need and o32 is not o) else None)
        ctx.meta = (has_f, bool(exact))
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, c, b, g, Fp, rowf, colf, lam, o, lse, o32 = ctx.saved_tensors
        has_f, exact = ctx.meta
        dq, dk, dv, dc, db, dg, dlam, dF = attn_backward(q, k, v, c, b, g, Fp if has_f else None, rowf, colf, lam,
                                                         o, lse, do, exact, o32=o32)
        return dq, dk, dv, dc, db, dg, dlam, dF, None


def _prepare_inputs(q, k, v, c, b, g, lam, fgate):
    if q.dim() != 4 or q.shape[-1] != 64 or c.dim() != 4 or c.shape[-1] != 32:
        raise ValueError(f"the Triton kernel supports d_head=64 and d_state=32 only; got q {tuple(q.shape)}, "
                         f"c {tuple(c.shape)}")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"q must be bf16, fp16 or fp32, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError(f"q/k/v dtypes differ: {q.dtype}, {k.dtype}, {v.dtype}")
    if not (q.is_cuda and k.device == q.device and v.device == q.device and c.device == q.device
            and b.device == q.device and g.device == q.device and lam.device == q.device):
        raise ValueError("all inputs must be on the same CUDA device")
    B, H, L, _ = q.shape
    if L < 1:
        raise ValueError("sequence length must be >= 1")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"k/v shapes {tuple(k.shape)}/{tuple(v.shape)} != q shape {tuple(q.shape)}")
    if c.shape != (B, H, L, 32) or b.shape != (B, H, L, 32):
        raise ValueError(f"c/b must have shape {(B, H, L, 32)}, got {tuple(c.shape)}/{tuple(b.shape)}")
    if g.dtype != torch.float32:
        raise TypeError(f"g must be fp32, got {g.dtype}")
    if g.shape != (B, H, L):
        raise ValueError(f"g must have shape {(B, H, L)}, got {tuple(g.shape)}")
    if fgate is not None:
        if fgate.dtype != torch.float32:
            raise TypeError(f"fgate must be fp32, got {fgate.dtype}")
        if fgate.shape != g.shape:
            raise ValueError(f"fgate shape {tuple(fgate.shape)} != g shape {tuple(g.shape)}")
        if fgate.device != q.device:
            raise ValueError("all inputs must be on the same CUDA device")
    if lam.shape != (H,):
        raise ValueError(f"lam must have shape {(H,)}, got {tuple(lam.shape)}")
    if lam.dtype != torch.float32:
        lam = lam.float()
    if c.dtype != q.dtype:
        c = c.to(q.dtype)
    if b.dtype != q.dtype:
        b = b.to(q.dtype)
    q, k, v, c, b = [x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v, c, b)]
    return q, k, v, c, b, g, lam, fgate


def sisa_attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, c: torch.Tensor, b: torch.Tensor,
                          g: torch.Tensor, lam: torch.Tensor, fgate: torch.Tensor | None = None) -> torch.Tensor:
    """Exact SISA attention (Triton), differentiable with respect to every tensor argument.

    Computes ``o = softmax(s) v`` with, for ``j <= i``,
    ``s_ij = q_i·k_j / 8 + lam_h · exp(g_i - g_j) · (c_i·b_j) + (fgate_i - fgate_j)``.

    Args:
        q, k, v: ``(B, H, L, 64)`` bf16, fp16 or fp32. Any strides with a contiguous last dimension (e.g. the
            ``(B, L, H, 64).transpose(1, 2)`` view of a projection).
        c, b: ``(B, H, L, 32)`` rotated SSM read/write vectors (C̃, B̃). Cast to ``q.dtype`` if they differ (the
            gradient flows back in their own dtype).
        g: ``(B, H, L)`` fp32 cumulative log-decay (natural log), non-increasing along L. Checked according to
            :func:`sisa.backends.set_check_mode` (maximum allowed increase 1e-3).
        lam: ``(H,)`` fp32 side-term weights (``softplus(lambda_raw)``).
        fgate: ``(B, H, L)`` fp32 cumulative log-sigmoid forget gate, or ``None`` (no forget term).

    Returns:
        ``(B, H, L, 64)`` tensor in ``q.dtype``, stored in ``(B, L, H, 64)`` memory layout.

    The backward pass uses the exact-Δ scheme with an fp32 copy of the output (module docstring, (D) and (E)).
    Deterministic: two calls with the same inputs give bit-identical outputs and gradients.
    """
    q, k, v, c, b, g, lam, fgate = _prepare_inputs(q, k, v, c, b, g, lam, fgate)
    return _SISAAttentionFn.apply(q, k, v, c, b, g, lam, fgate, True)
