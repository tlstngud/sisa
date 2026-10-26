"""Correctness judge for the Triton SISA attention kernel (forward and all gradients).

Target: ``sisa.kernels.attention.sisa_attention_triton(q, k, v, c, b, g, lam, fgate=None) -> o`` with

    s_ij = q_i·k_j/sqrt(D) + lam_h·exp(g_i - g_j)·(c_i·b_j) + (F_i - F_j)   (j <= i),   o_i = Σ_j softmax_j(s_ij)·v_j

q, k, v (B,H,L,64) and c, b (B,H,L,32) in DT ∈ {bf16, fp16, fp32}; g, F (B,H,L) fp32 (g non-increasing); λ (H,) fp32.
Judged tensors: o, dq, dk, dv, dc, db, dg, dλ, dF and the two derived quantities training actually uses,
d log α = revcumsum(dg) (the model's g = cumsum(log α)) and d log f = revcumsum(dF) (F = cumsum(log f)); the derived
quantities are reverse-accumulated from the kernel outputs in fp64, so only the kernel's error is measured.

Reference (TRUTH)
-----------------
fp64 dense evaluation on **the same-dtype inputs**: the tensors the kernel actually received (bf16 q..b as they are,
fp32 g/F/λ) are lifted exactly to fp64 and the formula is differentiated with torch autograd (in row blocks). Input
quantisation therefore does not count as error. The fp64 formula is verified by gradcheck in the selftest.

Tolerance, relative to the error the input dtype forces
-------------------------------------------------------
Two more dense emulations on the same inputs (fp32, row blocks, hand-written backward; the selftest checks the fp64
version of the hand-written backward against autograd):
  FA    = the unavoidable rounding points of a standard bf16/fp16 FlashAttention-2 pipeline: logits, softmax and
          accumulations in fp32 (natural-log units, differences g_i - g_j and F_i - F_j taken first), except that
          (1) P̂ = exp(s - m) is rounded to DT for the PV MMA, (2) o is stored in DT, (3) Δ_i = o_DT·do,
          (4) dv = P_DTᵀ·do, (5) dS is rounded to DT for the dq/dk MMAs, (6) (dS ⊙ λE) is rounded to DT for the dc/db
          MMAs, (7) dq, dk, dv, dc, db are output in DT; dg, dλ, dF in fp32. With DT = fp32, FA is a pure fp32 pipeline.
  IDEAL = no intermediate rounding (Δ from the fp32 o); outputs in DT. The floor a perfect fp32-internal kernel could
          reach (informational).
Gates (per head, per tensor):
  L2  : ‖X_k - X_T‖ <= 2·‖X_FA - X_T‖ + 0.05·‖ulp_DT(X_T)‖ + 8·u32·‖M‖
  row : for every row r (token): ‖X_k[r] - X_T[r]‖ <= c_row·(‖X_FA[r] - X_T[r]‖ + rms_r‖X_FA - X_T‖) + 0.05·‖ulp[r]‖
        + 8·u32·‖M[r]‖, with c_row = 4 (vector rows: o, dq, dk, dv, dc, db) or 8 (scalar rows: dg, dF, d log α,
        d log f; single-element tails are heavy).
  scalar (dλ, one per head): |e_k| <= 4·max(|e_FA|, σ) + floor; cumulative sums (d log α, d log f):
        L2 <= 4·max(e_FA, ‖σ‖) + floor.
  σ = the **expected** size of the FA error caused by the rounding of Δ (δΔ_i = (o_DT - o_fp32)·do_i) when it is
  independent across rows. These three quantities accumulate the Δ errors coherently, so the realised error swings by
  10-100x (dλ) or 0.4-2.4x (d log f); using one realisation as the scale would fail correct kernels.
  M = fp32 forward-error scale (textbook sum of |terms|): logit scale s̄_ij = |q||k|/sqrt(D) + |λ|E(1+|g_i-g_j|)|c||b|
      + |F_i - F_j|; dS scale = P·(|do|·|v| + |o|·|do|) + |dS|·s̄ (the size *before* cancellation); o = P(1+s̄)|V|;
      dq = |dS| scale·|K| (vectors: sums of absolute values); dg, dF, dλ, d log α, d log f use the root of the sum of
      squares (RSS) of the term scales (an absolute-value sum would grow the floor by L² for the scalar dλ).
      The size of the *differences* (g_i - g_j, F_i - F_j) is used, so an error proportional to |F|·u32 (e.g.
      multiplying F by log2e first) is not hidden under the floor.
  Rationale: the factor 2 is FlashAttention's own test rule (max error <= 2x the PyTorch reference error) moved to L2:
  "more than twice as bad as a standard bf16 kernel is a bug or an unnecessary approximation". The row gate catches
  local bugs (chunk-boundary rows, one masked entry) that an L2 average would hide. The 0.05 ulp and 8·u32·M floor
  allows a one-rounding difference in degenerate cases where the truth is exact in DT (FA error 0).

Run ``pytest -m "cuda and not slow"`` for the fast subset and ``pytest -m cuda`` for the full grid.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as Fn


# CUDA tests are skipped by tests/conftest.py when the Triton kernels cannot run on this machine.
pytestmark = [pytest.mark.cuda]

DEV = "cuda"
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
U32 = 2.0 ** -24
FAC_L2, FAC_ROW_VEC, FAC_ROW_TOK, FAC_SCALAR, FAC_CUM = 2.0, 4.0, 8.0, 4.0, 4.0
CUM = ("dloga", "dlogf")
SLACK_ULP, KAPPA = 0.05, 8.0
IN_NAMES = ["q", "k", "v", "c", "b", "g", "lam", "F"]
GRAD_NAMES = ["dq", "dk", "dv", "dc", "db", "dg", "dlam", "dF"]
KIND = dict(o="vec", dq="vec", dk="vec", dv="vec", dc="vec", db="vec", dg="tok", dF="tok", dlam="head",
            dloga="tok", dlogf="tok")
MEM_BLOCK = 1.2e9           # approximate bytes used by one row block of the references


@pytest.fixture(scope="module", autouse=True)
def _precise_matmul():
    """The fp32 emulations must not use TF32."""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[0], old[1]
    torch.set_float32_matmul_precision(old[2])


def kernel_fn(q, k, v, c, b, g, lam, F=None):
    from sisa.kernels.attention import sisa_attention_triton
    return sisa_attention_triton(q, k, v, c, b, g, lam, F)


# =============================================================================== inputs
# Per-token increments.  g: log α = -softplus(a + s·n)   F: log f = logsigmoid(a + s·n)
G_KINDS = {"zero": None, "slow": (-7.0, 0.5), "model": (-2.5, 1.0), "med": (1.0, 1.0), "fast": (3.0, 1.0),
           "extreme": "extreme"}
F_KINDS = {"none": None, "fox": (4.0, 1.5), "model": (1.0, 1.5), "strong": (-3.0, 1.5), "veryneg": (-10.0, 1.0),
           "extreme": "extreme", "walk": "walk"}
TINY = 1e-7
# 12 heads = λ{0, tiny, 0.31, 3} x F{none, fox, strong} (g cycles through model/med/slow); head h = (λ, g, F)
MIX12 = [((0.0, TINY, 0.31, 3.0)[h % 4], ("model", "med", "slow")[(h + h // 4) % 3], ("none", "fox", "strong")[h // 4])
         for h in range(12)]
MIX4 = [(0.31, "model", "fox"), (3.0, "slow", "strong"), (0.0, "med", "none"), (TINY, "fast", "model")]


def heads_for(H):
    return MIX12 if H == 12 else (MIX4[:H] if H <= 4 else MIX12[:H])


def _incs(kind, table, B, L, gen, idx, is_g, grid):
    spec = table[kind]
    if spec is None:
        return torch.zeros(B, L, device=DEV)
    u = torch.randn(B, L, device=DEV, generator=gen)[:, idx]
    if spec == "extreme":          # end value ≈ -20000
        r = torch.rand(B, L, device=DEV, generator=gen)[:, idx]
        inc = -(20000.0 / L) * (0.5 + r)
    elif spec == "walk":           # non-monotone F (the kernel does not assume F is monotone)
        inc = 0.5 * u
    else:
        a, s = spec
        inc = -Fn.softplus(a + s * u) if is_g else Fn.logsigmoid(a + s * u)
    if grid:                       # 2^-8 grid: shifts are exact in fp32 (shift section)
        inc = torch.round(inc * 256.0) / 256.0
    return inc


def make_inputs(spec):
    B, H, L = spec["B"], spec["H"], spec["L"]
    dt = DTYPES[spec["dtype"]]
    gen = torch.Generator(device=DEV).manual_seed(int(spec.get("seed", 0)))
    pat = spec.get("pattern", "random")
    P = {"random": L, "repeat7": 7, "const": 1}[pat]
    P = max(1, min(P, L))
    idx = torch.arange(L, device=DEV) % P

    def seq(d, scale):
        return (torch.randn(B, H, P, d, device=DEV, generator=gen) * scale)[:, :, idx].contiguous()
    qs, cbs = spec.get("qk_scale", 1.0), spec.get("cb_scale", 0.6)
    q, k, v = seq(64, qs).to(dt), seq(64, qs).to(dt), seq(64, 1.0).to(dt)
    c, b = seq(32, cbs).to(dt), seq(32, cbs).to(dt)
    heads = spec.get("heads") or heads_for(H)
    assert len(heads) == H
    g = torch.zeros(B, H, L, device=DEV)
    F = torch.zeros(B, H, L, device=DEV)
    for h, (lm, gk, fk) in enumerate(heads):
        g[:, h] = torch.cumsum(_incs(gk, G_KINDS, B, L, gen, idx, True, spec.get("grid")), -1)
        F[:, h] = torch.cumsum(_incs(fk, F_KINDS, B, L, gen, idx, False, spec.get("grid")), -1)
    g = (g + float(spec.get("shift_g", 0.0))).contiguous()
    F = (F + float(spec.get("shift_F", 0.0))).contiguous()
    lam = torch.tensor([h[0] for h in heads], device=DEV, dtype=torch.float32)
    do = torch.randn(B, H, L, 64, device=DEV, generator=gen).to(dt)
    return dict(q=q, k=k, v=v, c=c, b=b, g=g, lam=lam, F=None if spec.get("fgate_none") else F, do=do)


# =============================================================================== reference (TRUTH, fp64 autograd)
def _scores(q, k, c, b, g, F, lam, r0, r1, sc):
    """Logits of rows r0:r1 x keys 0:r1 (masked to -inf). Returns s, E (0 where masked), cb, mask."""
    qb, kb = q[:, :, r0:r1], k[:, :, :r1]
    s = torch.matmul(qb, kb.transpose(-1, -2)) * sc
    i = torch.arange(r0, r1, device=q.device)[:, None]
    j = torch.arange(0, r1, device=q.device)[None, :]
    mask = i >= j
    dg = g[:, :, r0:r1, None] - g[:, :, None, :r1]
    E = torch.exp(torch.where(mask, dg, torch.full((), -math.inf, dtype=dg.dtype, device=dg.device)))
    cb = torch.matmul(c[:, :, r0:r1], b[:, :, :r1].transpose(-1, -2))
    s = s + lam.view(1, -1, 1, 1) * E * cb
    if F is not None:
        s = s + (F[:, :, r0:r1, None] - F[:, :, None, :r1])
    s = s.masked_fill(~mask, -math.inf)
    return s, E, cb, mask


def _smag(q, k, c, b, g, F, lam, r0, r1, sc, E, mask):
    """fp32 rounding scale of the logits (natural-log units): |q||k|/sqrt(D) + |λ|·E(1+|g_i-g_j|)·|c||b| + |F_i-F_j|.
    Exponents and differences are measured by the size of the *difference*: a correct kernel subtracts g_i - g_j and
    F_i - F_j first, so its error is proportional to |F_i - F_j|, not |F| (a kernel proportional to |F| is detected
    above this floor). Returns smag, aE = E(1+|Δg|), |c||b|."""
    qka = torch.matmul(q[:, :, r0:r1].abs(), k[:, :, :r1].abs().transpose(-1, -2)) * sc
    cba = torch.matmul(c[:, :, r0:r1].abs(), b[:, :, :r1].abs().transpose(-1, -2))
    dga = (g[:, :, r0:r1, None] - g[:, :, None, :r1]).abs()
    aE = E * (1 + torch.where(mask, dga, torch.zeros((), dtype=dga.dtype, device=dga.device)))
    smag = qka + lam.abs().view(1, -1, 1, 1) * aE * cba
    if F is not None:
        smag = smag + (F[:, :, r0:r1, None] - F[:, :, None, :r1]).abs()
    return torch.where(mask, smag, torch.zeros((), dtype=smag.dtype, device=smag.device)), aE, cba


def _rb(B, H, L, bytes_per, n_tmp):
    return int(max(8, min(1024, MEM_BLOCK // max(1, B * H * max(L, 1) * bytes_per * n_tmp))))


def truth64(ins, rb=None):
    X = {n: (ins[n].detach().double().requires_grad_(True) if ins[n] is not None else None) for n in IN_NAMES}
    q, k, v, c, b, g, lam, F = [X[n] for n in IN_NAMES]
    B, H, L, D = q.shape
    sc = 1.0 / math.sqrt(D)
    dO = ins["do"].double()
    rb = rb or _rb(B, H, L, 8, 24)
    o = torch.empty(B, H, L, D, dtype=torch.float64, device=q.device)
    for r0 in range(0, L, rb):
        r1 = min(L, r0 + rb)
        s, _, _, _ = _scores(q, k, c, b, g, F, lam, r0, r1, sc)
        ob = torch.matmul(torch.softmax(s, -1), v[:, :, :r1])
        ob.backward(dO[:, :, r0:r1])
        o[:, :, r0:r1] = ob.detach()
    out = {"o": o}
    for n, gn in zip(IN_NAMES, GRAD_NAMES):
        if X[n] is not None:
            out[gn] = X[n].grad if X[n].grad is not None else torch.zeros_like(X[n])
    return _derive(out)


def dense64(q, k, v, c, b, g, lam, F):
    """Unblocked fp64 formula (any D/S), for gradcheck."""
    L = q.shape[2]
    s, _, _, _ = _scores(q, k, c, b, g, F, lam, 0, L, 1.0 / math.sqrt(q.shape[-1]))
    return torch.matmul(torch.softmax(s, -1), v)


def _revcumsum(x):
    return x.double().flip(-1).cumsum(-1).flip(-1)


def _derive(d):
    if "dg" in d and d["dg"] is not None:
        d["dloga"] = _revcumsum(d["dg"])
    if "dF" in d and d["dF"] is not None:
        d["dlogf"] = _revcumsum(d["dF"])
    return d


# =============================================================================== emulations (FA / IDEAL / exact64)
def emulate(ins, mode, rb=None, want_mag=False):
    """mode: 'fa' (FlashAttention-2 standard rounding points), 'ideal' (no intermediate rounding), 'exact64' (fp64,
    no rounding: checks the hand-written backward)."""
    cdt = torch.float64 if mode == "exact64" else torch.float32
    dt = ins["q"].dtype
    low = (mode == "fa") and dt != torch.float32
    R = (lambda x: x.to(dt).to(cdt)) if low else (lambda x: x)

    def RO(x, odt):
        return x if mode == "exact64" else x.to(odt).to(cdt)
    q, k, v, c, b = [ins[n].to(cdt) for n in "qkvcb"]
    g, lam = ins["g"].to(cdt), ins["lam"].to(cdt)
    F = ins["F"].to(cdt) if ins["F"] is not None else None
    dO = ins["do"].to(cdt)
    B, H, L, D = q.shape
    S = c.shape[-1]
    sc = 1.0 / math.sqrt(D)
    rb = rb or _rb(B, H, L, 4 if cdt == torch.float32 else 8, 40 if want_mag else 24)
    lamv = lam.view(1, H, 1, 1)
    o = torch.empty(B, H, L, D, dtype=cdt, device=DEV)
    lse = torch.empty(B, H, L, dtype=cdt, device=DEV)
    mag = {}
    if want_mag:
        mag["o"] = torch.empty_like(o)
    for r0 in range(0, L, rb):
        r1 = min(L, r0 + rb)
        s, E, cb, mask = _scores(q, k, c, b, g, F, lam, r0, r1, sc)
        m = s.amax(-1, keepdim=True)
        Ph = torch.exp(s - m)
        lsum = Ph.sum(-1, keepdim=True)
        o[:, :, r0:r1] = torch.matmul(R(Ph), v[:, :, :r1]) / lsum
        lse[:, :, r0:r1] = (m + torch.log(lsum)).squeeze(-1)
        if want_mag:
            Pn = Ph / lsum
            smag, _, _ = _smag(q, k, c, b, g, F, lam, r0, r1, sc, E, mask)
            PS = Pn * smag
            mag["o"][:, :, r0:r1] = (torch.matmul(Pn + PS, v[:, :, :r1].abs())
                                     + PS.sum(-1, keepdim=True) * o[:, :, r0:r1].abs())
    o_out = RO(o, dt)
    Delta = ((o_out if mode == "fa" else o) * dO).sum(-1)
    # The FA error of dλ = Σ_ij dS_ij u_ij is almost entirely the rounding of Δ: δdλ = -Σ_i δΔ_i w_i
    # (w_i = Σ_j P_ij u_ij). One realisation of a scalar swings by 10-100x through sign cancellation, so the gate also
    # uses its statistical size σ = sqrt(Σ(δΔ_i w_i)²).
    dD = Delta - (o * dO).sum(-1) if mode == "fa" else None
    absD = (o.abs() * dO.abs()).sum(-1) if want_mag else None
    sig2 = torch.zeros(H, dtype=cdt, device=DEV)
    # d log f_t = Σ_{i>=t, j<t} dS_ij and d log α_t = λ Σ_{i>=t, j<t} dS_ij u_ij are coherent sums of the Δ errors as
    # well, so their realised L2 swings like a handful of random variables. The expected size
    # σ_t² = Σ_{i>=t} δΔ_i² (Σ_{j<t} P_ij [u_ij])² is computed alongside and max(realised, expected) is the scale.
    sig2_f = torch.zeros(B, H, L, dtype=cdt, device=DEV)
    sig2_a = torch.zeros(B, H, L, dtype=cdt, device=DEV)
    z = lambda *s_: torch.zeros(*s_, dtype=cdt, device=DEV)        # noqa: E731
    dq, dk, dv = z(B, H, L, D), z(B, H, L, D), z(B, H, L, D)
    dc, db = z(B, H, L, S), z(B, H, L, S)
    dg, dF, dlam = z(B, H, L), z(B, H, L), z(H)
    if want_mag:
        for n, t in (("dq", dq), ("dk", dk), ("dv", dv), ("dc", dc), ("db", db), ("dg", dg), ("dF", dF),
                     ("dlam", dlam)):
            mag[n] = torch.zeros_like(t)
    for r0 in range(0, L, rb):
        r1 = min(L, r0 + rb)
        s, E, cb, mask = _scores(q, k, c, b, g, F, lam, r0, r1, sc)
        P = torch.exp(s - lse[:, :, r0:r1, None])
        dP = torch.matmul(dO[:, :, r0:r1], v[:, :, :r1].transpose(-1, -2))
        dS = P * (dP - Delta[:, :, r0:r1, None])
        dv[:, :, :r1] += torch.matmul(R(P).transpose(-1, -2), dO[:, :, r0:r1])
        dq[:, :, r0:r1] = torch.matmul(R(dS), k[:, :, :r1]) * sc
        dk[:, :, :r1] += torch.matmul(R(dS).transpose(-1, -2), q[:, :, r0:r1]) * sc
        W = dS * lamv * E
        dc[:, :, r0:r1] = torch.matmul(R(W), b[:, :, :r1])
        db[:, :, :r1] += torch.matmul(R(W).transpose(-1, -2), c[:, :, r0:r1])
        T = dS * E * cb
        if dD is not None:
            PU = P * E * cb
            w = PU.sum(-1)
            sig2 += (dD[:, :, r0:r1] * w).pow(2).sum(dim=(0, 2))
            dd2 = dD[:, :, r0:r1, None].pow(2) * mask                     # [i >= t] (t = column index)
            sig2_f[:, :, :r1] += (dd2 * (P.cumsum(-1) - P).pow(2)).sum(-2)
            sig2_a[:, :, :r1] += lam.pow(2).view(1, H, 1) * (dd2 * (PU.cumsum(-1) - PU).pow(2)).sum(-2)
        dg[:, :, r0:r1] += lam.view(1, H, 1) * T.sum(-1)
        dg[:, :, :r1] -= lam.view(1, H, 1) * T.sum(-2)
        dlam += T.sum(dim=(0, 2, 3))
        dF[:, :, r0:r1] += dS.sum(-1)
        dF[:, :, :r1] -= dS.sum(-2)
        if want_mag:
            smag, aE, cba = _smag(q, k, c, b, g, F, lam, r0, r1, sc, E, mask)
            absdP = torch.matmul(dO[:, :, r0:r1].abs(), v[:, :, :r1].abs().transpose(-1, -2))
            aS = P * (absdP + absD[:, :, r0:r1, None]) + dS.abs() * smag   # fp32 error scale of dS (pre-cancellation)
            aW = aS * lam.abs().view(1, H, 1, 1) * aE
            aT = aS * aE * cba
            mag["dv"][:, :, :r1] += torch.matmul((P * (1 + smag)).transpose(-1, -2), dO[:, :, r0:r1].abs())
            mag["dq"][:, :, r0:r1] = torch.matmul(aS, k[:, :, :r1].abs()) * sc
            mag["dk"][:, :, :r1] += torch.matmul(aS.transpose(-1, -2), q[:, :, r0:r1].abs()) * sc
            mag["dc"][:, :, r0:r1] = torch.matmul(aW, b[:, :, :r1].abs())
            mag["db"][:, :, :r1] += torch.matmul(aW.transpose(-1, -2), c[:, :, r0:r1].abs())
            # Summed quantities (dg, dF, dλ) use the root of the sum of squares: the typical size for independent
            # fp32 roundings per term. A plain absolute sum makes the scalar dλ floor thousands of times the fp32
            # error, which disables the gate; RSS still exceeds any single term, which covers diagonal cancellation.
            la2 = lam.pow(2).view(1, H, 1)
            aT2, aS2 = aT.pow(2), aS.pow(2)
            mag["dg"][:, :, r0:r1] += la2 * aT2.sum(-1)
            mag["dg"][:, :, :r1] += la2 * aT2.sum(-2)
            mag["dlam"] += aT2.sum(dim=(0, 2, 3))
            mag["dF"][:, :, r0:r1] += aS2.sum(-1)
            mag["dF"][:, :, :r1] += aS2.sum(-2)
    out = dict(o=o_out, dq=RO(dq, dt), dk=RO(dk, dt), dv=RO(dv, dt), dc=RO(dc, ins["c"].dtype),
               db=RO(db, ins["b"].dtype), dg=RO(dg, torch.float32), dlam=RO(dlam, torch.float32))
    if F is not None:
        out["dF"] = RO(dF, torch.float32)
    out = _derive(out)
    if dD is not None:
        out["_sig_dlam"] = sig2.sqrt()
        out["_sig_dloga"] = sig2_a.sqrt()
        if F is not None:
            out["_sig_dlogf"] = sig2_f.sqrt()
    if want_mag:
        if F is None:
            mag.pop("dF")
        mag["dloga"] = _revcumsum(mag["dg"]).sqrt()
        if "dF" in mag:
            mag["dlogf"] = _revcumsum(mag["dF"]).sqrt()
        for n in ("dg", "dF", "dlam"):
            if n in mag:
                mag[n] = mag[n].sqrt()
        return out, mag
    return out


# =============================================================================== kernel call
def call_kernel(fn, args, do, fwd_only=False):
    """args: 8 tensors (F may be None), passed as they are (including non-contiguous views).
    Returns o, {grad name: tensor}."""
    o = fn(*args)
    if fwd_only:
        return o.detach(), {}
    wrt = [(gn, a) for gn, a in zip(GRAD_NAMES, args) if a is not None and a.requires_grad]
    gs = torch.autograd.grad(o, [a for _, a in wrt], do, allow_unused=True)
    return o.detach(), {gn: (x.detach() if x is not None else None) for (gn, _), x in zip(wrt, gs)}


def leaves(ins):
    return [(ins[n].detach().clone().requires_grad_(True) if ins[n] is not None else None) for n in IN_NAMES]


def run_kernel_checked(fn, ins):
    """Run the kernel and check that it did not modify its inputs. Returns o, grads, message."""
    args = leaves(ins)
    snap = [a.detach().clone() if a is not None else None for a in args]
    o, gr = call_kernel(fn, args, ins["do"])
    torch.cuda.synchronize()
    mutated = [n for n, a, s in zip(IN_NAMES, args, snap) if a is not None and not torch.equal(a.detach(), s)]
    return o, gr, ("inputs mutated: " + ",".join(mutated)) if mutated else ""


# =============================================================================== metrics / gates
def ulp(x, dt):
    mant = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23, torch.float64: 52}[dt]
    emin = {torch.bfloat16: -126, torch.float16: -14, torch.float32: -126, torch.float64: -1022}[dt]
    e = torch.floor(torch.log2(x.abs().double().clamp_min(2.0 ** emin)))
    return torch.pow(2.0, e - mant)


def _rows(x, kind):
    """(B,H,L,F) / (B,H,L) / (H,) -> per-head list of (rows, features)."""
    if kind == "head":
        return [x[h].reshape(1, 1) for h in range(x.shape[0])]
    if kind == "tok":
        return [x[:, h].reshape(-1, 1) for h in range(x.shape[1])]
    return [x[:, h].reshape(-1, x.shape[-1]) for h in range(x.shape[1])]


def gate_tensor(name, Xk, Xt, Xb, Xi, M, odt, sig=None):
    """Per-head gate. Returns dict(ratio=max over heads, ...). sig: statistical size of the FA error of a scalar
    (dλ) or of a cumulative quantity, per head."""
    kind = KIND[name]
    Xk, Xt, Xb, Xi, M = [z.double() for z in (Xk, Xt, Xb, Xi, M)]
    SG = _rows(sig.double(), kind) if (sig is not None and kind == "tok") else None
    res = dict(heads=[])
    for h, (xk, xt, xb, xi, m) in enumerate(zip(*[_rows(z, kind) for z in (Xk, Xt, Xb, Xi, M)])):
        u = ulp(xt, odt)
        ek, eb, ei = (xk - xt).norm().item(), (xb - xt).norm().item(), (xi - xt).norm().item()
        nt = xt.norm().item()
        floor = SLACK_ULP * u.norm().item() + KAPPA * U32 * m.norm().item()
        if kind == "head":
            eb_eff = max(eb, float(sig[h])) if sig is not None else eb
            tol = FAC_SCALAR * eb_eff + floor
        else:
            eb_eff = max(eb, SG[h].norm().item()) if SG is not None else eb
            tol = (FAC_CUM if name in CUM else FAC_L2) * eb_eff + floor
        r_l2 = ek / tol if tol > 0 else (0.0 if ek == 0 else math.inf)
        hr = dict(e_k=ek, e_fa=eb, e_fa_eff=eb_eff, floor=floor, e_ideal=ei, norm=nt, r_l2=r_l2,
                  rel_l2=ek / nt if nt > 0 else (0.0 if ek == 0 else math.inf),
                  r_ideal=ek / (ei + floor) if (ei + floor) > 0 else (0.0 if ek == 0 else math.inf))
        if kind != "head":
            rk = (xk - xt).norm(dim=-1)
            rbv = (xb - xt).norm(dim=-1)
            if SG is not None:
                rbv = torch.maximum(rbv, SG[h].norm(dim=-1))
            rms_b = rbv.pow(2).mean().sqrt()
            fl_r = SLACK_ULP * u.norm(dim=-1) + KAPPA * U32 * m.norm(dim=-1)
            c_row = FAC_ROW_VEC if kind == "vec" else FAC_ROW_TOK
            tol_r = c_row * (rbv + rms_b) + fl_r
            rr = torch.where(tol_r > 0, rk / tol_r.clamp_min(1e-300), torch.where(rk > 0, math.inf, 0.0))
            ir = int(torch.argmax(rr))
            hr.update(r_row=float(rr[ir]), row_at=ir)
        res["heads"].append(hr)
    res["ratio"] = max(max(h["r_l2"], h.get("r_row", 0.0)) for h in res["heads"])
    res["rel_l2"] = max(h["rel_l2"] for h in res["heads"])
    return res


def judge(kout, T, FA, ID, MAG, odts, names=None):
    """kout: kernel output dict. Returns (fails list, metrics dict)."""
    fails, met = [], {}
    names = names or [n for n in ("o", "dq", "dk", "dv", "dc", "db", "dg", "dlam", "dF", "dloga", "dlogf") if n in T]
    for n in names:
        if n not in T:
            continue
        if kout.get(n) is None:
            fails.append(f"{n}: missing")
            continue
        x = kout[n]
        if not bool(torch.isfinite(x).all()):
            fails.append(f"{n}: non-finite ({int((~torch.isfinite(x)).sum())} elems)")
            met[n] = dict(ratio=math.inf)
            continue
        r = gate_tensor(n, x, T[n], FA[n], ID[n], MAG[n], odts.get(n, torch.float32), sig=FA.get("_sig_" + n))
        met[n] = r
        if not r["ratio"] <= 1.0:
            hw = max(range(len(r["heads"])), key=lambda h: max(r["heads"][h]["r_l2"], r["heads"][h].get("r_row", 0)))
            hh = r["heads"][hw]
            fails.append(f"{n}: ratio {r['ratio']:.3g} (head {hw}: L2 {hh['r_l2']:.3g}, row {hh.get('r_row', 0):.3g}"
                         f"@{hh.get('row_at', '-')}, rel_l2 {hh['rel_l2']:.3g} "
                         f"vs FA {hh['e_fa'] / max(hh['norm'], 1e-300):.3g})")
    return fails, met


def kout_dict(o, gr):
    d = dict(o=o)
    d.update({k_: v_ for k_, v_ in gr.items()})
    return _derive(d)


def out_dtypes(ins):
    dt = ins["q"].dtype
    return dict(o=dt, dq=dt, dk=dt, dv=dt, dc=ins["c"].dtype, db=ins["b"].dtype)


def references(ins):
    T = truth64(ins)
    FA = emulate(ins, "fa")
    ID, MAG = emulate(ins, "ideal", want_mag=True)
    return T, FA, ID, MAG


# =============================================================================== case runners (return fails list)
def case_numeric(fn, spec):
    ins = make_inputs(spec)
    o, gr, msg = run_kernel_checked(fn, ins)
    T, FA, ID, MAG = references(ins)
    fails, _ = judge(kout_dict(o, gr), T, FA, ID, MAG, out_dtypes(ins))
    if msg:
        fails.append(msg)
    return fails


def case_shift(fn, spec):
    """Shift invariance: g and F on a 2^-8 grid plus a constant c -> exact in fp32, identical truth and FA. Only the
    *growth* is judged: for every tensor and head e_k(c) <= 2·max(e_k(0), e_FA) + floor (dλ and the cumulative
    quantities: 4·max(..., σ)). Error growing with c means the kernel subtracts rounded products of large numbers
    (e.g. F·log2e, g·log2e)."""
    fails = []
    base = dict(spec)
    base.update(shift_g=0.0, shift_F=0.0, grid=True)
    ins0 = make_inputs(base)
    T, FA, ID, MAG = references(ins0)
    odt = out_dtypes(ins0)
    e0 = None
    for c_ in spec["shifts"]:
        sp = dict(base)
        sp["shift_" + spec["which"]] = float(c_)
        ins = make_inputs(sp)
        # the grid makes the shift exact element-wise (all differences preserved -> truth and FA can be reused)
        for n in ("g", "F"):
            want = float(c_) if n == spec["which"] else 0.0
            assert bool(((ins[n].double() - ins0[n].double()) == want).all()), f"shift not exact for {n}"
        o, gr, msg = run_kernel_checked(fn, ins)
        if msg:
            fails.append(f"shift {c_}: {msg}")
        _, met = judge(kout_dict(o, gr), T, FA, ID, MAG, odt)
        ek = {n: [h["e_k"] for h in r["heads"]] for n, r in met.items() if "heads" in r}
        if e0 is None:
            e0 = ek
            continue
        for n, r in met.items():
            if "heads" not in r or n not in e0:
                continue
            fac = FAC_SCALAR if KIND[n] == "head" else (FAC_CUM if n in CUM else FAC_L2)
            for h, hr in enumerate(r["heads"]):
                tol = fac * max(e0[n][h], hr["e_fa_eff"]) + hr["floor"]
                if ek[n][h] > tol:
                    fails.append(f"shift {c_}: {n} head {h} error grew {ek[n][h] / max(e0[n][h], 1e-300):.3g}x "
                                 f"(rel_l2 {hr['rel_l2']:.3g}, ratio to tol {ek[n][h] / tol:.3g})")
    return fails


def case_noncontig(fn, spec):
    """Model-like (B,L,H,·).transpose(1,2) views, sliced views (F, λ, optionally g) and a non-contiguous do.
    Must be bitwise equal to the contiguous call and pass the truth gate."""
    ins = make_inputs(spec)
    B, H, L = spec["B"], spec["H"], spec["L"]
    fails = []

    def tview(x):            # same values, (B,L,H,·) storage
        base = x.detach().transpose(1, 2).contiguous().requires_grad_(True)
        return base.transpose(1, 2)
    views = {n: tview(ins[n]) for n in "qkvcb"}
    lam_base = torch.zeros(2 * H, device=DEV)
    lam_base[::2] = ins["lam"]
    lam_base.requires_grad_(True)
    views["lam"] = lam_base[::2]
    Fb = torch.zeros(B, H, 2 * L, device=DEV)
    Fb[..., ::2] = ins["F"]
    Fb.requires_grad_(True)
    views["F"] = Fb[..., ::2]
    if spec.get("g_view"):
        gb = torch.zeros(B, H, 2 * L, device=DEV)
        gb[..., ::2] = ins["g"]
        gb.requires_grad_(True)
        views["g"] = gb[..., ::2]
    else:
        views["g"] = ins["g"].detach().clone().requires_grad_(True)
    do_v = ins["do"].transpose(1, 2).contiguous().transpose(1, 2)
    args = [views[n] for n in IN_NAMES]
    assert not args[0].is_contiguous() and not args[6].is_contiguous() and not args[7].is_contiguous()
    o1, g1 = call_kernel(fn, args, do_v)
    torch.cuda.synchronize()
    o0, g0, msg = run_kernel_checked(fn, ins)
    if msg:
        fails.append("contiguous reference call: " + msg)
    same = dict(o=bool(torch.equal(o0, o1)))
    same.update({n: bool(torch.equal(g0[n], g1[n])) for n in g0 if g0[n] is not None})
    if not all(same.values()):
        fails.append("not bitwise equal to contiguous call: " + ",".join(n for n, v_ in same.items() if not v_))
    T, FA, ID, MAG = references(ins)
    f2, _ = judge(kout_dict(o1, g1), T, FA, ID, MAG, out_dtypes(ins))
    return fails + f2


def case_determinism(fn, spec):
    ins = make_inputs(spec)
    o1, g1, msg = run_kernel_checked(fn, ins)
    o2, g2, msg2 = run_kernel_checked(fn, ins)
    with torch.no_grad():
        o3 = fn(*[ins[n] for n in IN_NAMES])
    torch.cuda.synchronize()
    same = dict(o=bool(torch.equal(o1, o2)), o_nograd=bool(torch.equal(o1, o3)))
    same.update({n: bool(torch.equal(g1[n], g2[n])) for n in g1 if g1[n] is not None})
    fails = [f"run-to-run differs: {n}" for n, v_ in same.items() if not v_]
    return fails + [m_ for m_ in (msg, msg2) if m_]


def case_causality(fn, spec):
    """Perturb every input at positions >= t (new random q, k, v, c, b; g and F shifted by -0.5, which keeps g
    monotone). (1) rows < t of the output are bitwise unchanged and rows >= t change; (2) with do[>=t] = 0 the
    gradients at positions >= t are exactly 0 and the gradients at positions < t (and dλ) are bitwise unchanged."""
    ins = make_inputs(spec)
    fails = []
    gen = torch.Generator(device=DEV).manual_seed(999)
    for t in spec["ts"]:
        pert = dict(ins)
        for n in "qkvcb":
            x = ins[n].clone()
            noise = torch.randn(x[:, :, t:].shape, device=DEV, generator=gen) * float(x.float().std())
            x[:, :, t:] = noise.to(x.dtype)
            pert[n] = x
        for n in ("g", "F"):
            if ins[n] is not None:
                x = ins[n].clone()
                x[:, :, t:] -= 0.5
                pert[n] = x.contiguous()
        dom = ins["do"].clone()
        dom[:, :, t:] = 0
        with torch.no_grad():
            y0 = fn(*[ins[n] for n in IN_NAMES])
            y1 = fn(*[pert[n] for n in IN_NAMES])
        _, ga = call_kernel(fn, leaves(ins), dom)
        _, gb = call_kernel(fn, leaves(pert), dom)
        torch.cuda.synchronize()
        if not torch.equal(y0[:, :, :t], y1[:, :, :t]):
            fails.append(f"t={t}: forward rows < t changed")
        if torch.equal(y0[:, :, t:], y1[:, :, t:]):
            fails.append(f"t={t}: forward rows >= t unchanged")
        for n in ga:
            if ga[n] is None:
                continue
            if n == "dlam":
                if not torch.equal(ga[n], gb[n]):
                    fails.append(f"t={t}: dlam depends on the future")
                continue
            if not bool((ga[n][:, :, t:] == 0).all() and (gb[n][:, :, t:] == 0).all()):
                fails.append(f"t={t}: {n} at >= t nonzero")
            if not torch.equal(ga[n][:, :, :t], gb[n][:, :, :t]):
                fails.append(f"t={t}: {n} at < t depends on the future")
    return fails


def _vs_other(kd, od, T, MAG, names, odt, fac=FAC_L2):
    """Kernel error <= fac x (the other implementation's error) + floor, per head (L2)."""
    fails = []
    for n in names:
        if kd.get(n) is None or od.get(n) is None or KIND[n] == "head":   # scalars are judged by the FA gate (σ) only
            continue
        for h, (xk, xo, xt, m) in enumerate(zip(*[_rows(z.double(), KIND[n]) for z in (kd[n], od[n], T[n], MAG[n])])):
            ek, eo = (xk - xt).norm().item(), (xo - xt).norm().item()
            floor = SLACK_ULP * ulp(xt, odt.get(n, torch.float32)).norm().item() + KAPPA * U32 * m.norm().item()
            tol = fac * eo + floor
            if not ek <= tol:
                fails.append(f"{n} head {h}: kernel error {ek / max(eo, 1e-300):.3g}x the other implementation")
    return fails


def case_sdpa(fn, spec):
    """λ = 0, F = 0 -> standard causal attention. Calibration: PyTorch's flash SDPA must pass the same FA gate, and
    our kernel must not be more than 2x worse than it."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    ins = make_inputs(spec)
    T, FA, ID, MAG = references(ins)
    odt = out_dtypes(ins)
    q, k, v = [ins[n].detach().clone().requires_grad_(True) for n in "qkv"]
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            of = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
    except RuntimeError as e:
        pytest.skip(f"flash SDPA unavailable: {e}")
    gq, gk, gv = torch.autograd.grad(of, (q, k, v), ins["do"])
    fl = dict(o=of.detach(), dq=gq, dk=gk, dv=gv)
    ff, _ = judge(fl, T, FA, ID, MAG, odt, ["o", "dq", "dk", "dv"])
    fails = [f"calibration (flash SDPA vs FA gate): {x}" for x in ff]
    o, gr, msg = run_kernel_checked(fn, ins)
    kd = kout_dict(o, gr)
    f1, _ = judge(kd, T, FA, ID, MAG, odt, ["o", "dq", "dk", "dv"])
    f2 = _vs_other(kd, fl, T, MAG, ["o", "dq", "dk", "dv"], odt)
    return fails + [f"vs FA gate: {x}" for x in f1] + [f"vs flash SDPA: {x}" for x in f2] + ([msg] if msg else [])


def case_fox(fn, spec):
    """λ = 0: compare with the official forgetting_attention (optional dependency). F is built with the same
    torch.cumsum (fp32) the official kernel uses internally, so both kernels see the same inputs."""
    fox = pytest.importorskip("forgetting_transformer")
    ins = make_inputs(spec)
    gen = torch.Generator(device=DEV).manual_seed(int(spec.get("seed", 0)) + 7)
    B, H, L = spec["B"], spec["H"], spec["L"]
    logf = torch.zeros(B, H, L, device=DEV)
    for h, (_, _, fk) in enumerate(spec["heads"]):
        logf[:, h] = _incs(fk, F_KINDS, B, L, gen, torch.arange(L, device=DEV), False, False)
    ins["F"] = torch.cumsum(logf, dim=-1, dtype=torch.float32).contiguous()
    T, FA, ID, MAG = references(ins)
    odt = out_dtypes(ins)
    q, k, v = [ins[n].detach().clone().requires_grad_(True) for n in "qkv"]
    lf = logf.clone().requires_grad_(True)
    of = fox.forgetting_attention(q, k, v, lf, head_first=True, sm_scale=1.0 / math.sqrt(64))
    gq, gk, gv, glf = torch.autograd.grad(of, (q, k, v, lf), ins["do"])
    ref = dict(o=of.detach(), dq=gq, dk=gk, dv=gv, dlogf=glf.double())
    o, gr, msg = run_kernel_checked(fn, ins)
    kd = kout_dict(o, gr)
    kd_train = dict(kd)
    kd_train["dlogf"] = gr["dF"].float().flip(-1).cumsum(-1).flip(-1).double()
    f1, _ = judge(kd, T, FA, ID, MAG, odt, ["o", "dq", "dk", "dv", "dF", "dlogf"])
    f2 = _vs_other(kd_train, ref, T, MAG, ["o", "dq", "dk", "dv", "dlogf"], odt)
    return [f"vs FA gate: {x}" for x in f1] + [f"vs official FoX: {x}" for x in f2] + ([msg] if msg else [])


# =============================================================================== case table
LS = [1, 2, 17, 63, 64, 65, 127, 128, 129, 511, 1000, 1537, 2047, 2048, 4096]


def build_cases():
    C = []

    def add(section, name, runner, **spec):
        spec.setdefault("B", 1)
        spec.setdefault("dtype", "bf16")
        C.append(dict(name=f"{section}/{name}", runner=runner, spec=spec))
    # sweep: every L, bf16, H=12 (every λ x F combination)
    for L in LS:
        add("sweep", f"L{L}_H12_bf16", "numeric", H=12, L=L, seed=L)
    for L in LS:
        for s_ in (1, 2):
            add("sweep", f"L{L}_H12_bf16_s{s_}", "numeric", H=12, L=L, seed=10000 * s_ + L)
    add("sweep", "L8192_H4_bf16", "numeric", H=4, L=8192, seed=8192)
    # batch
    add("batch", "B2_H12_L65", "numeric", B=2, H=12, L=65, seed=11)
    add("batch", "B2_H12_L1000", "numeric", B=2, H=12, L=1000, seed=12)
    add("batch", "B3_H2_L2047", "numeric", B=3, H=2, L=2047, seed=13)
    add("batch", "B2_H8_L129", "numeric", B=2, H=8, L=129, seed=14)
    # dtype
    for L in (1, 2, 65, 129, 1000, 2048, 4096):
        add("dtype", f"fp32_L{L}_H4", "numeric", H=4, L=L, dtype="fp32", seed=100 + L)
    for L in (129, 2048):
        add("dtype", f"fp16_L{L}_H4", "numeric", H=4, L=L, dtype="fp16", seed=200 + L)
    # adversarial
    mixF = ["fox", "strong", "none", "model"]
    lamm = [0.31, 3.0, 0.0, TINY]
    for L in (129, 2048):
        add("adv", f"g_extreme_L{L}", "numeric", H=4, L=L, seed=300 + L,
            heads=[(lamm[h], "extreme", mixF[h]) for h in range(4)])
        add("adv", f"F_extreme_L{L}", "numeric", H=4, L=L, seed=310 + L,
            heads=[(lamm[h], ["model", "slow", "med", "zero"][h], "extreme") for h in range(4)])
    add("adv", "gF_extreme_L4096", "numeric", H=4, L=4096, seed=320,
        heads=[(lamm[h], "extreme", "extreme") for h in range(4)])
    for L in (1000, 2048):
        add("adv", f"forget_strong_L{L}", "numeric", H=4, L=L, seed=330 + L,
            heads=[(lamm[h], ["model", "slow", "med", "fast"][h], "veryneg") for h in range(4)])
    add("adv", "repeat7_L2048_H12", "numeric", H=12, L=2048, seed=340, pattern="repeat7")
    for L in (1537, 4096):
        add("adv", f"repeat7_L{L}_H12", "numeric", H=12, L=L, seed=1340 + L, pattern="repeat7")
    add("adv", "const_L1000_H12", "numeric", H=12, L=1000, seed=341, pattern="const")
    add("adv", "sharp_qk4_L2048", "numeric", H=4, L=2048, seed=342, qk_scale=4.0)
    add("adv", "lam_edge_L1000", "numeric", H=4, L=1000, seed=343,
        heads=[(0.0, "model", "fox"), (1e-40, "slow", "model"), (TINY, "med", "none"), (3.0, "zero", "strong")])
    add("adv", "slow_decay_lam3_L2048", "numeric", H=4, L=2048, seed=344,
        heads=[(3.0, "slow", "none"), (3.0, "slow", "fox"), (3.0, "model", "none"), (3.0, "zero", "none")])
    add("adv", "bigcb_lam3_L1000", "numeric", H=4, L=1000, seed=345, cb_scale=1.5,
        heads=[(3.0, "slow", "none"), (3.0, "model", "fox"), (3.0, "med", "none"), (3.0, "zero", "model")])
    add("adv", "F_walk_nonmonotone_L1000", "numeric", H=4, L=1000, seed=346,
        heads=[(lamm[h], "model", "walk") for h in range(4)])
    add("adv", "fp32_extreme_L2048", "numeric", H=4, L=2048, dtype="fp32", seed=347,
        heads=[(lamm[h], "extreme", ["extreme", "strong", "none", "extreme"][h]) for h in range(4)])
    add("adv", "fgate_none_L1000", "numeric", H=4, L=1000, seed=348, fgate_none=True)
    # shift invariance
    for dt in ("fp32", "bf16"):
        for which in ("F", "g"):
            add("shift", f"{which}_shift_{dt}_L2048", "shift", H=4, L=2048, dtype=dt, seed=400, which=which,
                shifts=[0, -1000, -5000, -20000],
                heads=[(0.31, "slow", "fox"), (3.0, "slow", "fox"), (TINY, "model", "none"), (0.0, "slow", "fox")])
    # non-contiguous inputs
    for L in (129, 1000):
        add("noncontig", f"views_L{L}", "noncontig", H=4, L=L, seed=500 + L)
    add("noncontig", "g_view_L1000", "noncontig", H=4, L=1000, seed=510, g_view=True)
    # determinism
    add("determinism", "L129_H12", "determinism", H=12, L=129, seed=600)
    add("determinism", "L2048_H12", "determinism", H=12, L=2048, seed=601)
    add("determinism", "L4096_H4", "determinism", H=4, L=4096, seed=602)
    add("determinism", "fp32_L1000", "determinism", H=4, L=1000, dtype="fp32", seed=603)
    # causality
    add("causality", "L1000", "causality", H=4, L=1000, seed=700, ts=[1, 127, 128, 129, 517, 999])
    add("causality", "L2048", "causality", H=4, L=2048, seed=701, ts=[1, 128, 1024, 1500, 2047])
    # calibration against PyTorch's flash SDPA (λ = 0, F = 0)
    sdh = [(0.0, "model", "none")] * 4
    for L in (129, 2048, 4096):
        add("sdpa", f"plain_L{L}_bf16", "sdpa", H=4, L=L, seed=1000 + L, heads=sdh)
    add("sdpa", "plain_L2048_fp16", "sdpa", H=4, L=2048, dtype="fp16", seed=1099, heads=sdh)
    # official FoX (λ = 0), optional dependency
    foxh = [(0.0, "model", f_) for f_ in ("fox", "model", "strong", "extreme")]
    for L in (63, 129, 1000, 2048, 4096):
        add("fox", f"lam0_L{L}_bf16", "fox", H=4, L=L, seed=800 + L, heads=foxh)
    return C


# Cases of the fast default subset (`-m "not slow"`); everything else is marked slow.
FAST = {
    "sweep/L1_H12_bf16", "sweep/L2_H12_bf16", "sweep/L63_H12_bf16", "sweep/L64_H12_bf16", "sweep/L65_H12_bf16",
    "sweep/L127_H12_bf16", "sweep/L128_H12_bf16", "sweep/L129_H12_bf16", "sweep/L1000_H12_bf16",
    "batch/B2_H12_L65", "batch/B2_H8_L129",
    "dtype/fp32_L65_H4", "dtype/fp16_L129_H4",
    "adv/g_extreme_L129", "adv/F_extreme_L129", "adv/forget_strong_L1000", "adv/lam_edge_L1000",
    "adv/F_walk_nonmonotone_L1000", "adv/fgate_none_L1000",
    "noncontig/views_L129", "noncontig/g_view_L1000",
    "determinism/L129_H12",
    "causality/L1000",
}
_RUNNERS = dict(numeric=case_numeric, shift=case_shift, noncontig=case_noncontig, determinism=case_determinism,
                causality=case_causality, sdpa=case_sdpa, fox=case_fox)
_CASES = build_cases()


@pytest.mark.parametrize("case", [pytest.param(c, id=c["name"], marks=() if c["name"] in FAST else (pytest.mark.slow,))
                                  for c in _CASES])
def test_attention_case(case):
    fails = _RUNNERS[case["runner"]](kernel_fn, case["spec"])
    torch.cuda.empty_cache()
    assert not fails, f"{case['name']}:\n  " + "\n  ".join(fails[:12])


# =============================================================================== self-test of the references
def test_reference_selftest():
    """(1) gradcheck of the fp64 formula, (2) blocked autograd == unblocked, (3) hand-written backward (exact64) ==
    autograd, (4) FA == IDEAL for fp32 inputs."""
    gen = torch.Generator(device=DEV).manual_seed(0)
    B, H, L, D, S = 1, 2, 9, 4, 3
    mk = lambda *s: torch.randn(*s, device=DEV, dtype=torch.float64, generator=gen)       # noqa: E731
    q, k, v, c, b = mk(B, H, L, D), mk(B, H, L, D), mk(B, H, L, D), mk(B, H, L, S), mk(B, H, L, S)
    g = torch.cumsum(-Fn.softplus(mk(B, H, L) + 1), -1)
    F = torch.cumsum(Fn.logsigmoid(mk(B, H, L)), -1)
    lam = torch.tensor([0.31, 3.0], device=DEV, dtype=torch.float64)
    xs = [t.clone().requires_grad_(True) for t in (q, k, v, c, b, g, lam, F)]
    assert torch.autograd.gradcheck(dense64, xs, eps=1e-6, atol=1e-8, rtol=1e-6)
    for sp in (dict(B=1, H=4, L=300, dtype="bf16", seed=3), dict(B=2, H=4, L=77, dtype="fp32", seed=4, heads=[
            (0.31, "fast", "veryneg"), (3.0, "zero", "walk"), (0.0, "extreme", "extreme"), (TINY, "slow", "none")])):
        ins = make_inputs(sp)
        T = truth64(ins)
        T2 = truth64(ins, rb=7)
        X = {n: ins[n].detach().double().requires_grad_(True) for n in IN_NAMES}
        o = dense64(*[X[n] for n in IN_NAMES])
        gs = torch.autograd.grad(o, [X[n] for n in IN_NAMES], ins["do"].double())
        D0 = _derive(dict(o=o.detach(), **dict(zip(GRAD_NAMES, gs))))
        E64 = emulate(ins, "exact64", rb=11)
        for n in D0:
            den = D0[n].norm().item() or 1.0
            worst = max((T[n] - D0[n]).norm().item(), (T2[n] - D0[n]).norm().item(),
                        (E64[n] - D0[n]).norm().item()) / den
            assert worst <= 1e-10, f"L{sp['L']} {sp['dtype']} {n}: reference disagreement {worst:.2e}"
        if sp["dtype"] == "fp32":
            fa, idl = emulate(ins, "fa"), emulate(ins, "ideal")
            assert all(torch.equal(fa[n], idl[n]) for n in idl), "FA != IDEAL for fp32 inputs"
