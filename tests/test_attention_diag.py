"""Diagonal-entry and strong-forgetting judge for the Triton SISA attention kernel.

Background: on strongly forgetting heads (forget increments of -14 to -25 nats per token or stronger) P_tt -> 1 and the
diagonal dS_tt = P_tt(dP_tt - Δ_t) computes a true value of size ~ε as the difference of two O(|dP|) numbers. Where
P_tt rounds to 1.0 in fp64 (forgetting below about -30 nats per token), even an fp64 autograd truth carries the
diagonal noise in the vector gradients and dλ.

Reference (REF): an independent fp64 analytic formula in this file, **with Σ_j P_tj = 1 substituted on the diagonal
(Δ-free)**:
    dS_tj = P_tj (dP_tj - T_t)                (j < t),     T_t = Σ_j P_tj dP_tj
    dS_tt = P_tt · Σ_{j<t} P_tj (dP_tt - dP_tj)          (directly from the row contraction)
    dg_t  = λ[Σ_{j<t} dS_tj u_tj - Σ_{i>t} dS_it u_it],  dF_t = Σ_{j<t} dS_tj - Σ_{i>t} dS_it   (diagonal cancels)
Every term has fp64 relative precision, so REF stays accurate when ε is below the fp64 ulp. It is verified against a
60-digit mpmath evaluation of the *direct* formula (dS = P(dP - T) including the diagonal; a different formula) and
against fp64 autograd at mild forgetting.

Gates (exact-mode kernel, per head and tensor, rel-L2 = ‖X - REF‖/‖REF‖):
  o, dq, dk, dv, dc, db (DT outputs): <= 3·ideal_DT + 1e-6, where ideal_DT is the error of REF rounded to DT (the
      reachable floor; fp32 inputs: 3·ideal + 5e-6), or <= 0.5·(error of the FlashAttention-standard Δ̂ mode of the
      same kernel) where ideal itself is noisy (L <= 2).
  dg, dF, d log α, d log f (fp32): <= 3e-5;  dλ (one scalar per head): <= max(3e-5, 1e-6·cond) with
      cond = Σ_(b,t)|row part| / |dλ| (when the row parts cancel, an fp32 dense evaluation of the same formula has
      error ~u32·cond as well).
  sharp section, scalar chain: <= max(the above, 3·error of an fp32 IDEAL dense emulation): the fp32 condition number
      u/(1 - P_max) of the dominant entry dP_{t,j*} - T is a floor for any fp32 evaluation.
  Comparison gate: exact mode <= 1.5·(FlashAttention-standard Δ̂ mode) + 1e-5 for dg, dF, d log α, d log f and dλ.
  Tensors whose REF is outside the fp32 normal range (‖REF‖ < 1e-30) are judged by absolute error <= 1e-30·sqrt(n).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as Fn


# CUDA tests are skipped by tests/conftest.py when the Triton kernels cannot run on this machine.
pytestmark = [pytest.mark.cuda]

DEV = "cuda"
TENS = ("o", "dq", "dk", "dv", "dc", "db", "dg", "dF", "dlam", "dloga", "dlogf")
VEC = ("o", "dq", "dk", "dv", "dc", "db")
SCAL = ("dg", "dF", "dloga", "dlogf", "dlam")


@pytest.fixture(scope="module", autouse=True)
def _precise_matmul():
    old = (torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    yield
    torch.backends.cuda.matmul.allow_tf32 = old[0]
    torch.set_float32_matmul_precision(old[1])


def kernels():
    """exact: the public entry point; fa: the same kernels with the FlashAttention-standard Δ̂ (comparison only)."""
    from sisa.kernels import attention as A

    def fa(q, k, v, c, b, g, lam, F):
        args = A._prepare_inputs(q, k, v, c, b, g, lam, F)
        return A._SISAAttentionFn.apply(*args, False)
    return {"exact": A.sisa_attention_triton, "fa": fa}


# ============================================================================================ reference
def ref_head(q, k, v, c, b, g, lam, F, do):
    """fp64 analytic gradients of one (b, h) with the Δ-free diagonal. 2-D fp64 inputs. Returns a dict."""
    L, D = q.shape
    sc = 1.0 / math.sqrt(D)
    i = torch.arange(L, device=q.device)
    mask = i[:, None] >= i[None, :]
    strict = i[:, None] > i[None, :]
    ninf = torch.full((), -math.inf, dtype=torch.float64, device=q.device)
    E = torch.exp(torch.where(mask, g[:, None] - g[None, :], ninf))
    CB = c @ b.T
    U = E * CB
    s = (q @ k.T) * sc + lam * U
    if F is not None:
        s = s + (F[:, None] - F[None, :])
    s = torch.where(mask, s, ninf)
    P = torch.softmax(s, -1)
    o = P @ v
    dP = do @ v.T
    T = (P * dP).sum(1)
    zero = torch.zeros((), dtype=torch.float64, device=q.device)
    dS = torch.where(strict, P * (dP - T[:, None]), zero)
    Ptt, dPtt = torch.diagonal(P), torch.diagonal(dP)
    dSd = Ptt * torch.where(strict, P * (dPtt[:, None] - dP), zero).sum(1)
    dSf = dS + torch.diag(dSd)
    dSE = dSf * E
    dSU = dS * U
    out = dict(o=o, dq=(dSf @ k) * sc, dk=(dSf.T @ q) * sc, dv=P.T @ do,
               dc=lam * (dSE @ b), db=lam * (dSE.T @ c),
               dg=lam * (dSU.sum(1) - dSU.sum(0)),
               dlam=(dSf * U).sum())
    out["dlam_abs"] = ((dS * U).sum(1) + dSd * torch.diagonal(CB)).abs().sum()   # Σ_t |row part| (condition number)
    out["dF"] = (dS.sum(1) - dS.sum(0)) if F is not None else None
    return out


def ref64(ins):
    q, k, v, c, b, g, lam, F, do = [ins[n] for n in ("q", "k", "v", "c", "b", "g", "lam", "F", "do")]
    B, H, L, D = q.shape
    R = {n: torch.zeros(ins[src].shape, dtype=torch.float64, device=DEV)
         for n, src in (("o", "q"), ("dq", "q"), ("dk", "k"), ("dv", "v"), ("dc", "c"), ("db", "b"), ("dg", "g"))}
    R["dF"] = torch.zeros(g.shape, dtype=torch.float64, device=DEV) if F is not None else None
    R["dlam"] = torch.zeros(H, dtype=torch.float64, device=DEV)
    R["dlam_abs"] = torch.zeros(H, dtype=torch.float64, device=DEV)
    for bi in range(B):
        for h in range(H):
            r = ref_head(*[x[bi, h].double() for x in (q, k, v, c, b, g)], lam[h].double(),
                         None if F is None else F[bi, h].double(), do[bi, h].double())
            for n in ("o", "dq", "dk", "dv", "dc", "db", "dg"):
                R[n][bi, h] = r[n]
            if F is not None:
                R["dF"][bi, h] = r["dF"]
            R["dlam"][h] += r["dlam"]
            R["dlam_abs"][h] += r["dlam_abs"]
    return derive(R)


def revcumsum(x):
    return x.double().flip(-1).cumsum(-1).flip(-1)


def derive(d):
    d["dloga"] = revcumsum(d["dg"])
    d["dlogf"] = revcumsum(d["dF"]) if d.get("dF") is not None else None
    return d


def autograd64(ins):
    """fp64 autograd of the dense formula (cross-check of REF)."""
    q, k, v, c, b, g, lam, F, do = [ins[n] for n in ("q", "k", "v", "c", "b", "g", "lam", "F", "do")]
    X = [x.detach().double().requires_grad_(True) for x in (q, k, v, c, b, g, lam)]
    Fx = F.detach().double().requires_grad_(True) if F is not None else None
    q_, k_, v_, c_, b_, g_, l_ = X
    L = q.shape[2]
    i = torch.arange(L, device=DEV)
    mask = i[:, None] >= i[None, :]
    ninf = torch.full((), -math.inf, dtype=torch.float64, device=DEV)
    s = (q_ @ k_.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    E = torch.exp(torch.where(mask, g_[..., :, None] - g_[..., None, :], ninf))
    s = s + l_.view(1, -1, 1, 1) * E * (c_ @ b_.transpose(-1, -2))
    if Fx is not None:
        s = s + (Fx[..., :, None] - Fx[..., None, :])
    s = torch.where(mask, s, ninf)
    o = torch.softmax(s, -1) @ v_
    o.backward(do.double())
    d = dict(o=o.detach(), dq=q_.grad, dk=k_.grad, dv=v_.grad, dc=c_.grad, db=b_.grad, dg=g_.grad, dlam=l_.grad,
             dF=Fx.grad if Fx is not None else None)
    return derive(d)


def ideal32(ins):
    """fp32 dense emulation without intermediate rounding (Δ from the fp32 o), outputs rounded to their dtypes.
    The floor of any fp32 evaluation of the same formula (used by the sharp section)."""
    cdt = torch.float32
    dt = ins["q"].dtype
    q, k, v, c, b = [ins[n].to(cdt) for n in "qkvcb"]
    g, lam = ins["g"].to(cdt), ins["lam"].to(cdt)
    F = ins["F"].to(cdt) if ins["F"] is not None else None
    dO = ins["do"].to(cdt)
    B, H, L, D = q.shape
    sc = 1.0 / math.sqrt(D)
    i = torch.arange(L, device=DEV)[:, None]
    j = torch.arange(L, device=DEV)[None, :]
    mask = i >= j
    E = torch.exp(torch.where(mask, g[..., :, None] - g[..., None, :], torch.full((), -math.inf, device=DEV)))
    cb = c @ b.transpose(-1, -2)
    s = (q @ k.transpose(-1, -2)) * sc + lam.view(1, H, 1, 1) * E * cb
    if F is not None:
        s = s + (F[..., :, None] - F[..., None, :])
    s = s.masked_fill(~mask, -math.inf)
    m = s.amax(-1, keepdim=True)
    Ph = torch.exp(s - m)
    lsum = Ph.sum(-1, keepdim=True)
    o = (Ph @ v) / lsum
    lse = (m + torch.log(lsum)).squeeze(-1)
    Delta = (o * dO).sum(-1)
    P = torch.exp(s - lse[..., None])
    dP = dO @ v.transpose(-1, -2)
    dS = P * (dP - Delta[..., None])
    W = dS * lam.view(1, H, 1, 1) * E
    T = dS * E * cb
    out = dict(o=o.to(dt), dq=((dS @ k) * sc).to(dt), dk=((dS.transpose(-1, -2) @ q) * sc).to(dt),
               dv=(P.transpose(-1, -2) @ dO).to(dt), dc=(W @ b).to(ins["c"].dtype),
               db=(W.transpose(-1, -2) @ c).to(ins["b"].dtype),
               dg=lam.view(1, H, 1) * (T.sum(-1) - T.sum(-2)), dlam=T.sum(dim=(0, 2, 3)))
    out["dF"] = (dS.sum(-1) - dS.sum(-2)) if F is not None else None
    return derive(out)


def mp_head(q, k, v, c, b, g, lam, F, do, dps=60):
    """60-digit mpmath evaluation of the **direct** formula (dS = P(dP - T) including the diagonal), a different
    formula than REF's Δ-free diagonal. Small L only."""
    M = pytest.importorskip("mpmath")
    M.mp.dps = dps
    L, D = q.shape
    S = c.shape[1]
    t = lambda x: [[M.mpf(float(x[i, j])) for j in range(x.shape[1])] for i in range(x.shape[0])]   # noqa: E731
    Q, K, V, C, Bm, DO = t(q), t(k), t(v), t(c), t(b), t(do)
    G = [M.mpf(float(x)) for x in g]
    FF = [M.mpf(float(x)) for x in F] if F is not None else [M.mpf(0)] * L
    lm = M.mpf(float(lam))
    sc = 1 / M.sqrt(D)
    dot = lambda a, bb: M.fsum(x * y for x, y in zip(a, bb))   # noqa: E731
    CB = [[dot(C[i], Bm[j]) for j in range(L)] for i in range(L)]
    E = [[M.exp(G[i] - G[j]) if j <= i else M.mpf(0) for j in range(L)] for i in range(L)]
    P = [[M.mpf(0)] * L for _ in range(L)]
    for i in range(L):
        s = [dot(Q[i], K[j]) * sc + lm * E[i][j] * CB[i][j] + (FF[i] - FF[j]) for j in range(i + 1)]
        m = max(s)
        e = [M.exp(x - m) for x in s]
        z = M.fsum(e)
        for j in range(i + 1):
            P[i][j] = e[j] / z
    dP = [[dot(DO[i], V[j]) for j in range(L)] for i in range(L)]
    dS = [[M.mpf(0)] * L for _ in range(L)]
    for i in range(L):
        T = M.fsum(P[i][j] * dP[i][j] for j in range(i + 1))
        for j in range(i + 1):
            dS[i][j] = P[i][j] * (dP[i][j] - T)
    o = [[M.fsum(P[i][j] * V[j][d] for j in range(L)) for d in range(D)] for i in range(L)]
    dq = [[M.fsum(dS[i][j] * K[j][d] for j in range(L)) * sc for d in range(D)] for i in range(L)]
    dk = [[M.fsum(dS[i][j] * Q[i][d] for i in range(L)) * sc for d in range(D)] for j in range(L)]
    dc = [[lm * M.fsum(dS[i][j] * E[i][j] * Bm[j][d] for j in range(L)) for d in range(S)] for i in range(L)]
    db = [[lm * M.fsum(dS[i][j] * E[i][j] * C[i][d] for i in range(L)) for d in range(S)] for j in range(L)]
    dlam = M.fsum(dS[i][j] * E[i][j] * CB[i][j] for i in range(L) for j in range(L))
    dg = [lm * (M.fsum(dS[t_][j] * E[t_][j] * CB[t_][j] for j in range(L))
                - M.fsum(dS[i][t_] * E[i][t_] * CB[i][t_] for i in range(L))) for t_ in range(L)]
    dF = [M.fsum(dS[t_][j] for j in range(L)) - M.fsum(dS[i][t_] for i in range(L)) for t_ in range(L)]
    f = lambda x: torch.tensor([[float(y) for y in r] for r in x], dtype=torch.float64)   # noqa: E731
    return dict(o=f(o), dq=f(dq), dk=f(dk), dc=f(dc), db=f(db), dlam=torch.tensor(float(dlam), dtype=torch.float64),
                dg=torch.tensor([float(x) for x in dg], dtype=torch.float64),
                dF=torch.tensor([float(x) for x in dF], dtype=torch.float64))


# ============================================================================================ inputs / runs
def make(B, H, L, seed, dt=torch.bfloat16, steps=None, lam=0.31, gstep="model", layout="bhld", qk=1.0, cbs=0.6,
         qk_diag=None, cb_diag=None):
    """steps: per-head mean forget increment c (None -> F = None); increment = min(c + 2·N(0,1), 0).
    qk_diag: k = q·qk_diag (diagonal dominated by q·k); cb_diag: b = c·cb_diag (diagonal dominated by cb_tt)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=DEV, generator=gen)   # noqa: E731
    q, k, v = qk * rn(B, H, L, 64), qk * rn(B, H, L, 64), rn(B, H, L, 64)
    c, b = cbs * rn(B, H, L, 32), cbs * rn(B, H, L, 32)
    if qk_diag is not None:
        k = q * qk_diag
    if cb_diag is not None:
        b = c * cb_diag
    do = rn(B, H, L, 64)
    if gstep == "model":
        ga = -Fn.softplus(-2.5 + rn(B, H, L))
    elif gstep == "fast":
        ga = -20.0 + 2.0 * rn(B, H, L)
        ga = ga.clamp(max=0.0)
    else:
        ga = torch.zeros(B, H, L, device=DEV)
    g = ga.cumsum(-1).float().contiguous()
    if steps is None:
        F = None
    else:
        st = torch.stack([(cc + 2.0 * rn(B, L)).clamp(max=0.0) for cc in steps], 1)
        F = st.cumsum(-1).float().contiguous()
    lamt = torch.tensor(lam if isinstance(lam, (list, tuple)) else [lam] * H, device=DEV, dtype=torch.float32)

    def lay(x):
        x = x.to(dt)
        if layout == "blhd":
            return x.transpose(1, 2).contiguous().transpose(1, 2)
        return x.contiguous()
    return dict(q=lay(q), k=lay(k), v=lay(v), c=lay(c), b=lay(b), g=g, lam=lamt, F=F, do=lay(do))


def run(fn, ins):
    X = [ins[n].detach().clone().requires_grad_(True) for n in ("q", "k", "v", "c", "b", "g", "lam")]
    Fx = ins["F"].detach().clone().requires_grad_(True) if ins["F"] is not None else None
    o = fn(*X, Fx)
    o.backward(ins["do"])
    torch.cuda.synchronize()
    d = dict(o=o.detach(), dq=X[0].grad, dk=X[1].grad, dv=X[2].grad, dc=X[3].grad, db=X[4].grad, dg=X[5].grad,
             dlam=X[6].grad, dF=Fx.grad if Fx is not None else None)
    return derive(d)


def rel(a, r):
    if a is None or r is None:
        return None
    a, r = a.double(), r.double()
    if not torch.isfinite(a).all():
        return float("inf")
    n = r.norm().item()
    e = (a - r).norm().item()
    if n < 1e-30:                                   # truth outside the fp32 normal range: absolute error
        return e / (1e-30 * math.sqrt(r.numel()))
    return e / n


def head_errs(out, ref, h):
    e = {}
    for t in TENS:
        if out.get(t) is None or ref.get(t) is None:
            continue
        if t == "dlam":
            a, r = out[t][h:h + 1], ref[t][h:h + 1]
        else:
            a, r = out[t][:, h], ref[t][:, h]
        e[t] = rel(a, r)
    return e


def ideal(ref, h, dt, t):
    r = ref[t][:, h]
    return rel(r.to(dt), r)


def gate(case, ref, outs, H, dt, cmp_ok=True, emul=None):
    """Exact-mode gate (module docstring). emul = fp32 IDEAL emulation (scalar-chain floor = 3x its error)."""
    fails = []
    for h in range(H):
        e = {kn: head_errs(outs[kn], ref, h) for kn in outs if outs[kn] is not None}
        bad = []
        r = e["exact"]
        for t in VEC:
            if t not in r:
                continue
            idl = ideal(ref, h, dt, t)
            tol = 3 * idl + (5e-6 if dt == torch.float32 else 1e-6)
            fa = e.get("fa", {}).get(t)
            if not (r[t] <= tol or (fa is not None and r[t] <= 0.5 * fa)):
                bad.append(f"{t} {r[t]:.2e} > {tol:.2e} (ideal {idl:.1e}) and > 0.5·FA")
        cond = (ref["dlam_abs"][h] / ref["dlam"][h].abs().clamp_min(1e-300)).item()
        for t in SCAL:
            if t not in r:
                continue
            tol = max(3e-5, 1e-6 * cond) if t == "dlam" else 3e-5
            if emul is not None:
                ee = head_errs(emul, ref, h).get(t)
                if ee is not None:
                    tol = max(tol, 3 * ee)       # the fp32 floor of this condition number
            if not (r[t] <= tol):
                bad.append(f"{t} {r[t]:.2e} > {tol:.1e}" + (f" (cond {cond:.0f})" if t == "dlam" else ""))
            if cmp_ok:
                others = [e[kn][t] for kn in ("fa",) if kn in e and t in e[kn] and e[kn][t] is not None]
                if others and not (r[t] <= 1.5 * min(others) + 1e-5):
                    bad.append(f"{t} {r[t]:.2e} > 1.5·FA={min(others):.2e}")
        if bad:
            fails.append(f"{case} h{h}: " + "; ".join(bad))
    return fails


def check_case(tag, ins, dt, cmp_ok=True, with_emul=False):
    H = ins["q"].shape[1]
    ref = ref64(ins)
    emul = ideal32(ins) if with_emul else None
    outs = {kn: run(fn, ins) for kn, fn in kernels().items()}
    fails = gate(tag, ref, outs, H, dt, cmp_ok, emul)
    torch.cuda.empty_cache()
    assert not fails, "\n".join(fails)


# ============================================================================================ refcheck
@pytest.mark.slow
@pytest.mark.parametrize("c_", [-3.0, -20.0, -40.0, -60.0])
def test_refcheck_mpmath(c_):
    """REF vs 60-digit mpmath of the direct formula, L = 10: <= 1e-12."""
    L = 10
    ins = make(1, 1, L, seed=7, dt=torch.bfloat16, steps=[c_], lam=0.9)
    R = ref64(ins)
    M = mp_head(*[ins[n][0, 0].double().cpu() for n in ("q", "k", "v", "c", "b", "g")], ins["lam"][0].item(),
                ins["F"][0, 0].double().cpu(), ins["do"][0, 0].double().cpu())
    worst = 0.0
    for t in ("o", "dq", "dk", "dc", "db", "dg", "dF", "dlam"):
        m = M[t].to(DEV)
        rr = R[t][0, 0] if t != "dlam" else R[t][0]
        worst = max(worst, rel(rr, m))
    assert worst <= 1e-12, f"REF vs mpmath {worst:.2e} at forgetting {c_}/token"


def test_refcheck_autograd():
    """REF vs fp64 autograd at mild forgetting, L = 512: <= 1e-10."""
    ins = make(1, 4, 512, seed=8, steps=[-0.05, -1.0, -3.0, -6.0], lam=[0.31, 3.0, 0.0, 0.8])
    R, A = ref64(ins), autograd64(ins)
    w = max(rel(A[t], R[t]) for t in TENS if R.get(t) is not None)
    assert w <= 1e-10, f"REF vs fp64 autograd {w:.2e}"


# ============================================================================================ sections
CS = [-1.0, -5.0, -10.0, -14.0, -17.0, -20.0, -25.0, -30.0, -40.0, -60.0]


@pytest.mark.parametrize("dtn", ["bf16", pytest.param("fp16", marks=pytest.mark.slow),
                                 pytest.param("fp32", marks=pytest.mark.slow)])
def test_sweep_forgetting(dtn):
    """Forget increments c ± 2·N(0,1) per token, one c per head, L = 1024, λ = 0.31."""
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtn]
    ins = make(1, len(CS), 1024, seed=5, dt=dt, steps=CS, lam=0.31)
    check_case(f"sweep_{dtn}_L1024", ins, dt)


@pytest.mark.slow
def test_sweep_forgetting_lams_model_layout():
    ins = make(1, 4, 2048, seed=9, steps=[-14.0, -20.0, -25.0, -20.0], lam=[0.31, 3.0, 0.0, 0.8], layout="blhd")
    check_case("sweep_bf16_L2048_lams_blhd", ins, torch.bfloat16)


def test_local_qk_diagonal():
    """F = None, diagonal dominated by q·k (k = 6q)."""
    ins = make(1, 2, 1024, seed=11, steps=None, lam=[0.31, 0.0], qk_diag=6.0)
    check_case("local_qkdiag_Fnone", ins, torch.bfloat16)


def test_local_cb_diagonal_fast_decay():
    """F = None, diagonal dominated by the side term (b = 8c, fast decay, λ = 3)."""
    ins = make(1, 2, 1024, seed=12, steps=None, lam=[3.0, 1.0], cb_diag=8.0, gstep="fast")
    check_case("local_cbdiag_fastg_Fnone", ins, torch.bfloat16)


def test_local_forget_large_lambda_fast_decay():
    ins = make(1, 3, 1024, seed=13, steps=[-20.0, -10.0, -30.0], lam=[3.0, 3.0, 3.0], gstep="fast")
    check_case("local_forget_lam3_fastg", ins, torch.bfloat16)


@pytest.mark.parametrize("L", [1, 2, 63, 64, 65, 129, pytest.param(4096, marks=pytest.mark.slow)])
def test_edge_lengths(L):
    """Forgetting -20 nats/token, B = 2, model layout ((B,L,H,D).transpose) inputs."""
    ins = make(2, 2, L, seed=20 + L, steps=[-20.0, -3.0], lam=[0.31, 3.0], layout="blhd")
    check_case(f"edge_L{L}", ins, torch.bfloat16, cmp_ok=(L > 2))


@pytest.mark.parametrize("sc", [1.0, 2.0, 3.0, pytest.param(1.5, marks=pytest.mark.slow),
                                pytest.param(2.5, marks=pytest.mark.slow)])
def test_sharp_previous_token(sc):
    """One dominant off-diagonal entry (previous-token head: k_j = s·q_{j+1}; median P_max 0.81 .. 0.999999).
    The precision of Δ decides the vector gradients here. s <= 3 only: beyond, P_{t,t-1} ≈ 1 - 1e-11 and the fp64
    truth itself carries the cancellation noise of that entry."""
    ins = make(1, 2, 1024, seed=41, steps=[-0.05, -0.05], lam=[0.31, 3.0])
    q = ins["q"].float()
    k = torch.zeros_like(q)
    k[:, :, :-1] = sc * q[:, :, 1:]
    k[:, :, -1] = q[:, :, -1]
    ins["k"] = k.to(torch.bfloat16).contiguous()
    check_case(f"sharp_prevtok_s{sc:g}", ins, torch.bfloat16, with_emul=True)
