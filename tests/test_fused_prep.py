"""Tests of the fused SISA layer core (:mod:`sisa.kernels.fused_prep`).

  identity  The q/k stage is fed identity tables (cos = 1, sin = 0). Bitwise criteria (no tolerance):
            P1 prep outputs QR/KR == the Q/K columns of proj (after +0.0, i.e. only the sign of zeros may differ),
               including injected ±0, bf16 subnormals, ±1e30 and ±max-bf16;
            P2 the tables feed only the q/k stage: CR, BR, G, F, ROWF, COLF, TOT equal those obtained with a
               10000-base rotary table;
            P3 fp32 and bf16 identity tables give identical prep outputs;
            P4 the fused function equals the attention kernels called on the *unrotated* strided q/k views of proj
               (same prep factors): y, d proj[Q|K|V] == dq|dk|dv, dλ;
            P5 fp32 and bf16 identity tables give identical y, d proj, dz, dλ.
  prep      g, F and the rotated C/B against fp64, compared with the unfused torch glue; g monotone across block
            boundaries; rowf/colf bitwise equal to the attention's own `_factors` kernel on the same g.
  grads     d proj (per column group), dz and dλ against an fp64 reference that passes the values the candidate
            actually saw straight through (so only kernel and backward-prep error is measured); the fused path must not
            be worse than torch glue + the Triton attention (5% + 1e-6; 1.5x + 1e-5 for the fp32 scalar chains dz, dλ,
            whose error swings by 0.3-3x between seeds for both paths). Padding columns of d proj are exactly zero.
  props     determinism (two runs bitwise equal, inputs unchanged) and causality (perturbing proj/z at positions
            >= t leaves the output and the gradients at < t and dλ bitwise unchanged; with dy[>=t] = 0 the
            gradients at >= t are exactly 0).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F


# CUDA tests are skipped by tests/conftest.py when the Triton kernels cannot run on this machine.
cuda = pytest.mark.cuda

DEV = "cuda"
D, S = 64, 32


@pytest.fixture(scope="module")
def FP():
    from sisa.kernels import fused_prep
    return fused_prep


@pytest.fixture(scope="module")
def AT():
    from sisa.kernels import attention
    return attention


@pytest.fixture(autouse=True)
def _no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a + 0.0, b + 0.0))


def rel(a, b):
    a, b = a.double(), b.double()
    n = b.norm().item()
    return (a - b).norm().item() / (n if n > 0 else 1.0)


def rotary_tables(rows, dim=64, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    ang = torch.outer(torch.arange(rows).float(), freqs)
    return ang.cos().to(DEV), ang.sin().to(DEV)


def identity_tables(L, dtype=torch.float32):
    rows = max(L, 2048)
    return (torch.ones((rows, 32), device=DEV, dtype=dtype), torch.zeros((rows, 32), device=DEV, dtype=dtype))


def rotate_pairs(z, phi):
    """Interleaved-pair rotation of the last dimension of z by phi (d_state/2 angles)."""
    ze, zo = z[..., 0::2], z[..., 1::2]
    cp, sp = torch.cos(phi), torch.sin(phi)
    return torch.stack([ze * cp - zo * sp, ze * sp + zo * cp], dim=-1).flatten(-2)


# ------------------------------------------------------------------------------------------------ inputs
def make_proj(FP, B, L, H, seed=0, dt=torch.bfloat16, alpha_mu=-3.0, alpha_sd=2.0, th_mu=0.02, th_sd=0.2):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    off = FP.fused_layout(H)
    P = torch.zeros(B, L, off["NPAD"], device=DEV)
    P[..., :off["TH"]] = torch.randn(B, L, off["TH"], device=DEV, generator=gen)
    P[..., off["B"]:off["TH"]] *= 0.6
    P[..., off["TH"]:off["A"]] = th_mu + th_sd * torch.randn(B, L, H * 16, device=DEV, generator=gen)
    P[..., off["A"]:off["NTOT"]] = alpha_mu + alpha_sd * torch.randn(B, L, H, device=DEV, generator=gen)
    z = (4.0 + 1.5 * torch.randn(B, L, H, device=DEV, generator=gen)).contiguous()
    return P.to(dt).contiguous(), z, off


def make_proj_special(FP, B, L, H, seed, special):
    """proj with ±0 and bf16 subnormals (and, if `special`, ±1e30 and ±max-bf16) injected into the Q/K columns."""
    off = FP.fused_layout(H)
    g = torch.Generator(device=DEV).manual_seed(seed)
    p = torch.randn((B, L, off["NPAD"]), device=DEV, generator=g) * 0.5
    p[..., off["TH"]:off["A"]] *= 0.05                                    # θ: small angles
    p[..., off["A"]:off["NTOT"]] = p[..., off["A"]:off["NTOT"]] * 2 - 5   # α around the init bias -5
    p[..., off["NTOT"]:] = 0
    p = p.to(torch.bfloat16)
    qk = p[..., :2 * H * D].clone()
    n = qk.numel()
    flat = qk.reshape(-1)
    idx = torch.randperm(n, device=DEV, generator=g)
    k = max(1, n // 50)
    vals = [0.0, -0.0, 1e-40, -1e-40, 5e-39, -5e-39]
    if special:
        vals += [1e30, -1e30, 3.3e38, -3.3e38]
    for j, v in enumerate(vals):
        flat[idx[j * k:(j + 1) * k]] = v
    p[..., :2 * H * D] = flat.view(qk.shape)
    return p.contiguous()


def torch_glue(FP, proj, z, H):
    """The unfused torch path after the projection (no q/k rotation; C/B rotated by Φ in fp32)."""
    B, L, _ = proj.shape
    off = FP.fused_layout(H)
    q = proj[..., off["Q"]:off["Q"] + H * D].view(B, L, H, D).transpose(1, 2)
    k = proj[..., off["K"]:off["K"] + H * D].view(B, L, H, D).transpose(1, 2)
    v = proj[..., off["V"]:off["V"] + H * D].view(B, L, H, D).transpose(1, 2)
    bs = proj[..., off["B"]:off["B"] + H * S].view(B, L, H, S).transpose(1, 2)
    cs = proj[..., off["C"]:off["C"] + H * S].view(B, L, H, S).transpose(1, 2)
    alpha = proj[..., off["A"]:off["A"] + H].transpose(1, 2)
    theta = proj[..., off["TH"]:off["TH"] + H * S // 2].view(B, L, H, S // 2).transpose(1, 2)
    g = torch.cumsum(-F.softplus(alpha.float()), dim=2)
    Phi = torch.cumsum(theta.float(), dim=2)
    cs = rotate_pairs(cs.float(), Phi)
    bs = rotate_pairs(bs.float(), Phi)
    Fg = None
    if z is not None:
        Fg = torch.cumsum(F.logsigmoid(z.float()).transpose(1, 2), dim=2)
    return dict(q=q, k=k, v=v, c=cs.contiguous(), b=bs.contiguous(), g=g.contiguous(),
                F=None if Fg is None else Fg.contiguous())


def glue64(FP, proj, z, H):
    """fp64 truth of g, F, Φ and the rotated C/B (no rounding)."""
    B, L, _ = proj.shape
    off = FP.fused_layout(H)
    P = proj.double()
    alpha = P[..., off["A"]:off["A"] + H].transpose(1, 2)
    theta = P[..., off["TH"]:off["TH"] + H * 16].view(B, L, H, 16).transpose(1, 2)
    g = torch.cumsum(-F.softplus(alpha), 2)
    Phi = torch.cumsum(theta, 2)
    cs = P[..., off["C"]:off["C"] + H * 32].view(B, L, H, 32).transpose(1, 2)
    bs = P[..., off["B"]:off["B"] + H * 32].view(B, L, H, 32).transpose(1, 2)
    cs, bs = rotate_pairs(cs, Phi), rotate_pairs(bs, Phi)
    Fg = None if z is None else torch.cumsum(F.logsigmoid(z.double()).transpose(1, 2), 2)
    return dict(g=g, F=Fg, c=cs, b=bs, Phi=Phi)


def dense_attn64(q, k, v, c, b, g, lam, Fg):
    L = q.shape[2]
    s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    cm = torch.ones(L, L, device=q.device, dtype=torch.bool).tril()
    dg = torch.where(cm, g.unsqueeze(-1) - g.unsqueeze(-2), torch.full((), -math.inf, dtype=g.dtype, device=g.device))
    s = s + lam.view(1, -1, 1, 1) * torch.exp(dg) * torch.matmul(c, b.transpose(-1, -2))
    if Fg is not None:
        s = s + Fg.unsqueeze(-1) - Fg.unsqueeze(-2)
    return torch.matmul(torch.softmax(s.masked_fill(~cm, -math.inf), -1), v)


def ref64_fused(FP, proj, z, lam, H, fwdvals, dy):
    """fp64 reference gradients of the layer function. The forward values q/k/c/b/g/F are the ones the candidate
    actually computed (`fwdvals`), passed straight through (value = candidate's, gradient = exact formula), so the
    comparison measures kernel and backward error only, not the rounding of the forward prep."""
    B, L, _ = proj.shape
    off = FP.fused_layout(H)
    P = proj.detach().double().requires_grad_(True)
    Z = None if z is None else z.detach().double().requires_grad_(True)
    LAM = lam.detach().double().requires_grad_(True)

    def cols(nm, w):
        return P[..., off[nm]:off[nm] + H * w].view(B, L, H, w).transpose(1, 2)

    def st(x, val):
        return x + (val.double() - x).detach()
    q = st(cols("Q", 64), fwdvals["q"])
    k = st(cols("K", 64), fwdvals["k"])
    v = cols("V", 64)
    g = st(torch.cumsum(-F.softplus(P[..., off["A"]:off["A"] + H].transpose(1, 2)), 2), fwdvals["g"])
    Phi = torch.cumsum(P[..., off["TH"]:off["TH"] + H * 16].view(B, L, H, 16).transpose(1, 2), 2)
    cs = st(rotate_pairs(cols("C", 32), Phi), fwdvals["c"])
    bs = st(rotate_pairs(cols("B", 32), Phi), fwdvals["b"])
    Fg = None
    if Z is not None:
        Fg = st(torch.cumsum(F.logsigmoid(Z).transpose(1, 2), 2), fwdvals["F"])
    o = dense_attn64(q, k, v, cs, bs, g, LAM, Fg)
    y = o.transpose(1, 2).reshape(B, L, H * 64)
    gr = torch.autograd.grad(y, [P, LAM] + ([Z] if Z is not None else []), dy.double())
    return y.detach(), gr[0], gr[1], (gr[2] if Z is not None else None)


def run_candidate(FP, name, proj, z, lam, H, dy):
    from sisa.kernels.attention import sisa_attention_triton
    P = proj.detach().clone().requires_grad_(True)
    Z = None if z is None else z.detach().clone().requires_grad_(True)
    LAM = lam.detach().clone().requires_grad_(True)
    B, L, _ = proj.shape
    if name == "fused":
        y = FP.sisa_fused(P, Z, LAM, H)
    else:
        T = torch_glue(FP, P, Z, H)
        o = sisa_attention_triton(T["q"], T["k"], T["v"], T["c"].to(torch.bfloat16), T["b"].to(torch.bfloat16),
                                  T["g"], LAM, T["F"])
        y = o.transpose(1, 2).reshape(B, L, H * 64)
    gr = torch.autograd.grad(y, [P, LAM] + ([Z] if Z is not None else []), dy)
    return y.detach(), gr[0], gr[1], (gr[2] if Z is not None else None)


# ------------------------------------------------------------------------------------------------ layout (CPU)
def test_layout_and_padding():
    pytest.importorskip("triton")
    from sisa.kernels.fused_prep import fused_layout
    for H in range(1, 33):
        off = fused_layout(H)
        assert off["NTOT"] == 273 * H
        assert off["NPAD"] % 16 == 0 and 0 <= off["NPAD"] - off["NTOT"] < 16
        assert [off[k] for k in ("Q", "K", "V", "B", "C", "TH", "A")] == [0, 64 * H, 128 * H, 192 * H, 224 * H,
                                                                          256 * H, 272 * H]
    assert fused_layout(8)["NPAD"] == 2192 and fused_layout(12)["NPAD"] == 3280 and fused_layout(16)["NPAD"] == 4368
    with pytest.raises(ValueError):
        fused_layout(8, d_head=128)


def test_concat_fused_weights_matches_separate_linears():
    pytest.importorskip("triton")
    from sisa.kernels.fused_prep import concat_fused_weights, fused_layout
    torch.manual_seed(0)
    H, d = 2, 128
    shapes = dict(W_Q=64 * H, W_K=64 * H, W_V=64 * H, W_B=32 * H, W_C=32 * H, W_theta=16 * H, W_alpha=H)
    ws = {n: torch.randn(r, d, dtype=torch.float64, requires_grad=True) for n, r in shapes.items()}
    b_alpha = torch.randn(H, dtype=torch.float64, requires_grad=True)
    W, bvec = concat_fused_weights(*ws.values(), b_alpha)
    off = fused_layout(H)
    x = torch.randn(3, 5, d, dtype=torch.float64)
    proj = F.linear(x, W, bvec)
    assert proj.shape == (3, 5, off["NPAD"])
    for (n, r), col in zip(shapes.items(), ("Q", "K", "V", "B", "C", "TH", "A")):
        want = F.linear(x, ws[n], b_alpha if n == "W_alpha" else None)
        assert torch.equal(proj[..., off[col]:off[col] + r], want), n
    assert torch.equal(proj[..., off["NTOT"]:], torch.zeros_like(proj[..., off["NTOT"]:]))
    proj.sum().backward()
    assert all(w.grad is not None for w in ws.values()) and b_alpha.grad is not None


# ------------------------------------------------------------------------------------------------ identity tables
_ID_GRID = [(8, 1, 1), (8, 2, 65), (12, 1, 512), (16, 2, 2100), (8, 1, 63), (8, 2, 64), (12, 2, 2048), (16, 1, 1)]
_ID_FAST = {(8, 1, 1), (8, 2, 65), (12, 1, 512), (16, 2, 2100)}
_ID_PARAMS = [pytest.param(*c, id=f"H{c[0]}_B{c[1]}_L{c[2]}", marks=() if c in _ID_FAST else (pytest.mark.slow,))
              for c in _ID_GRID]


@cuda
@pytest.mark.parametrize("H,B,L", _ID_PARAMS)
def test_identity_tables_prep(FP, H, B, L):
    off = FP.fused_layout(H)
    seed = 1000 + 7 * H + 3 * B + L
    proj = make_proj_special(FP, B, L, H, seed, special=True)
    z = torch.randn((B, L, H), device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed + 1)) + 3.0
    ci, si = identity_tables(L)
    cr, sr = rotary_tables(max(L, 2048))
    pi = FP._launch_prep_forward(proj, z, ci, si, H)
    pb = FP._launch_prep_forward(proj, z, ci.bfloat16(), si.bfloat16(), H)
    pr = FP._launch_prep_forward(proj, z, cr, sr, H)
    Q = proj[..., off["Q"]:off["Q"] + H * D].reshape(B, L, H, D)
    K = proj[..., off["K"]:off["K"] + H * D].reshape(B, L, H, D)
    assert eq(pi["QR"], Q) and eq(pi["KR"], K), "P1: identity tables must leave q/k unrotated"
    for n in ("CR", "BR", "G", "F", "ROWF", "COLF", "TOT"):
        assert eq(pi[n], pr[n]), f"P2: {n} depends on the q/k tables"
    for n in pi:
        assert eq(pi[n], pb[n]), f"P3: {n} differs between fp32 and bf16 identity tables"


@cuda
@pytest.mark.parametrize("H,B,L", _ID_PARAMS)
def test_identity_tables_fused_equals_unrotated_attention(FP, AT, H, B, L):
    off = FP.fused_layout(H)
    seed = 2000 + 7 * H + 3 * B + L
    proj0 = make_proj_special(FP, B, L, H, seed, special=False)
    gz = torch.Generator(device=DEV).manual_seed(seed + 1)
    z0 = torch.randn((B, L, H), device=DEV, generator=gz) + 3.0
    lam0 = torch.rand((H,), device=DEV, generator=gz) + 0.2
    dy = (torch.randn((B, L, H * D), device=DEV, generator=gz) * 0.1).to(torch.bfloat16)
    ci, si = identity_tables(L)

    def run(fn):
        proj = proj0.clone().requires_grad_(True)
        z = z0.clone().requires_grad_(True)
        lam = lam0.clone().requires_grad_(True)
        y = fn(proj, z, lam)
        y.backward(dy)
        return y.detach(), proj.grad, z.grad, lam.grad

    yI, dpI, dzI, dlI = run(lambda p, z, l_: FP._SISAFusedFn.apply(p, z, l_, ci, si, H)[0])
    yB, dpB, dzB, dlB = run(lambda p, z, l_: FP._SISAFusedFn.apply(p, z, l_, ci.bfloat16(), si.bfloat16(), H)[0])
    yP, dpP, dzP, dlP = run(lambda p, z, l_: FP.sisa_fused(p, z, l_, H))
    assert eq(yI, yB) and eq(dpI, dpB) and eq(dzI, dzB) and eq(dlI, dlB), "P5: fp32 vs bf16 identity tables differ"
    assert eq(yI, yP) and eq(dpI, dpP) and eq(dzI, dzP) and eq(dlI, dlP), "public sisa_fused differs from the core"
    # reference: no q/k rotation at all (strided views of proj), same prep for C/B/g/F/factors
    pf = FP._launch_prep_forward(proj0, z0, ci, si, H)
    q = proj0[..., off["Q"]:off["Q"] + H * D].view(B, L, H, D).transpose(1, 2)
    k = proj0[..., off["K"]:off["K"] + H * D].view(B, L, H, D).transpose(1, 2)
    v = proj0[..., off["V"]:off["V"] + H * D].view(B, L, H, D).transpose(1, 2)
    c = pf["CR"].transpose(1, 2)
    b = pf["BR"].transpose(1, 2)
    o, lse, _, _, _, o32 = AT.attn_forward(q, k, v, c, b, pf["G"], lam0, pf["F"], factors=(pf["ROWF"], pf["COLF"]),
                                           want32=True)
    y_ref = o.transpose(1, 2).reshape(B, L, H * D)
    do = dy.reshape(B, L, H, D).transpose(1, 2)
    DCR = torch.empty((B, L, H, S), device=DEV, dtype=torch.float32)
    DBR = torch.empty((B, L, H, S), device=DEV, dtype=torch.float32)
    dq, dk, dv, dc, db, dg, dlam, dF = AT.attn_backward(q, k, v, c, b, pf["G"], pf["F"], pf["ROWF"], pf["COLF"], lam0,
                                                        o, lse, do, True, dc=DCR.transpose(1, 2),
                                                        db=DBR.transpose(1, 2), o32=o32)
    dQ = dpI[..., off["Q"]:off["Q"] + H * D].reshape(B, L, H, D)
    dK = dpI[..., off["K"]:off["K"] + H * D].reshape(B, L, H, D)
    dV = dpI[..., off["V"]:off["V"] + H * D].reshape(B, L, H, D)
    assert eq(yI, y_ref), "P4: y"
    assert eq(dQ, dq.transpose(1, 2).contiguous()), "P4: dq"
    assert eq(dK, dk.transpose(1, 2).contiguous()), "P4: dk"
    assert eq(dV, dv.transpose(1, 2).contiguous()), "P4: dv"
    assert eq(dlI, dlam), "P4: dlam"
    assert bool(torch.isfinite(yI.float()).all() and torch.isfinite(dpI.float()).all())


@cuda
def test_identity_table_cache_length_independent(FP):
    """Tables longer than L (cache rows rounded up to multiples of 2048) give bit-identical results."""
    H, L = 8, 777
    proj, z, _ = make_proj(FP, 2, L, H, seed=L + 1)
    lam = torch.full((H,), 0.3, device=DEV)
    dy = torch.randn(2, L, H * 64, device=DEV, generator=torch.Generator(device=DEV).manual_seed(3)).to(torch.bfloat16)
    outs = []
    for rows in (L, 2048, 8192):
        ci, si = torch.ones((rows, 32), device=DEV), torch.zeros((rows, 32), device=DEV)
        p_ = proj.clone().requires_grad_(True)
        z_ = z.clone().requires_grad_(True)
        l_ = lam.clone().requires_grad_(True)
        y, _ = FP._SISAFusedFn.apply(p_, z_, l_, ci, si, H)
        (y.float() * dy.float()).sum().backward()
        outs.append((y.detach(), p_.grad, z_.grad, l_.grad))
    for other in outs[1:]:
        assert all(torch.equal(a, b) for a, b in zip(outs[0], other))
    cos, sin = FP._identity_tables(5000, proj.device)
    assert cos.shape == (6144, 32) and sin.shape == (6144, 32) and cos.dtype == torch.float32
    assert bool((cos == 1).all()) and bool((sin == 0).all())


# ------------------------------------------------------------------------------------------------ prep accuracy
_PREP = [(2, 2048, 8, 1, True), (1, 300, 12, 2, True), (1, 1, 8, 3, True), (1, 129, 8, 4, False),
         (2, 4096, 8, 5, True), (1, 2048, 8, 6, True)]
_PREP_FAST = {1, 2, 3, 4, 6}


@cuda
@pytest.mark.parametrize("B,L,H,seed,hasf", [pytest.param(*c, id=f"B{c[0]}_L{c[1]}_H{c[2]}_F{int(c[4])}",
                                                          marks=() if c[3] in _PREP_FAST else (pytest.mark.slow,))
                                             for c in _PREP])
def test_prep_accuracy(FP, AT, B, L, H, seed, hasf):
    kw = dict(alpha_mu=-8.0, alpha_sd=0.5) if seed == 6 else {}        # log α close to 0 (monotonicity at boundaries)
    proj, z, off = make_proj(FP, B, L, H, seed, **kw)
    z = z if hasf else None
    ci, si = identity_tables(L)
    R = FP._launch_prep_forward(proj, z, ci, si, H)
    T = torch_glue(FP, proj, z, H)
    X = glue64(FP, proj, z, H)
    bad = []
    if not (eq(R["QR"].transpose(1, 2), T["q"]) and eq(R["KR"].transpose(1, 2), T["k"])):
        bad.append("q/k differ from the unrotated projection")
    if L > 1:
        inc = float((R["G"][..., 1:] - R["G"][..., :-1]).max())
        if inc > 0:
            bad.append(f"g not monotone (+{inc:.3e})")
    eg, egt = rel(R["G"], X["g"]), rel(T["g"], X["g"])
    if eg > 2 * egt + 1e-7:
        bad.append(f"g error {eg:.2e} > 2x torch {egt:.2e}")
    if hasf:
        ef, eft = rel(R["F"], X["F"]), rel(T["F"], X["F"])
        if ef > 2 * eft + 1e-7:
            bad.append(f"F error {ef:.2e} > 2x torch {eft:.2e}")
    for n, key in (("c", "CR"), ("b", "BR")):
        e_k = rel(R[key].transpose(1, 2), X[n])
        e_t = rel(T[n].to(torch.bfloat16), X[n])
        if e_k > 1.02 * e_t + 1e-7:
            bad.append(f"{n} error {e_k:.2e} above the torch bf16 path {e_t:.2e}")
    # rowf/colf == the attention's _factors kernel on the same g
    rowf = torch.empty_like(R["G"])
    colf = torch.empty_like(R["G"])
    NT = (L + 63) // 64
    mono = torch.empty(B * H * NT, device=DEV)
    AT._factors[(NT, B * H)](R["G"], rowf, colf, mono, *AT._st4(R["G"]), L, NT, H, 64, num_warps=2)
    if not (torch.equal(rowf, R["ROWF"]) and torch.equal(colf, R["COLF"])):
        bad.append("rowf/colf != _factors(g)")
    assert not bad, "; ".join(bad)


# ------------------------------------------------------------------------------------------------ gradient accuracy
_GRAD = [(1, 300, 4, 11, True), (1, 1024, 8, 12, True), (1, 129, 4, 13, False), (1, 1, 4, 14, True),
         (1, 2048, 4, 15, True)]
_GRAD_FAST = {11, 13, 14}


@cuda
@pytest.mark.parametrize("B,L,H,seed,hasf", [pytest.param(*c, id=f"B{c[0]}_L{c[1]}_H{c[2]}_F{int(c[4])}",
                                                          marks=() if c[3] in _GRAD_FAST else (pytest.mark.slow,))
                                             for c in _GRAD])
def test_fused_gradients_vs_fp64(FP, B, L, H, seed, hasf):
    proj, z, off = make_proj(FP, B, L, H, seed)
    z = z if hasf else None
    lam = torch.tensor([0.31, 3.0, 1e-7, 0.8, 0.31, 0.3, 0.29, 0.31][:H], device=DEV)
    dy = torch.randn(B, L, H * 64, device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed + 100)
                     ).to(torch.bfloat16)
    ci, si = identity_tables(L)
    fv = FP._launch_prep_forward(proj, z, ci, si, H)
    Tg = torch_glue(FP, proj, z, H)
    vals = {"fused": dict(q=fv["QR"].transpose(1, 2), k=fv["KR"].transpose(1, 2), c=fv["CR"].transpose(1, 2),
                          b=fv["BR"].transpose(1, 2), g=fv["G"], F=fv["F"]),
            "glue": dict(q=Tg["q"], k=Tg["k"], c=Tg["c"].to(torch.bfloat16), b=Tg["b"].to(torch.bfloat16),
                         g=Tg["g"], F=Tg["F"])}
    res = {}
    for name in ("fused", "glue"):
        y64, dP64, dl64, dz64 = ref64_fused(FP, proj, z, lam, H, vals[name], dy)
        y, dP, dl, dz = run_candidate(FP, name, proj, z, lam, H, dy)
        r = dict(y=rel(y, y64), dlam=rel(dl, dl64))
        for grp, (a, w) in dict(Q=("Q", H * 64), K=("K", H * 64), V=("V", H * 64), B=("B", H * 32),
                                C=("C", H * 32), TH=("TH", H * 16), A=("A", H)).items():
            r["d" + grp] = rel(dP[..., off[a]:off[a] + w], dP64[..., off[a]:off[a] + w])
        if dz is not None:
            r["dz"] = rel(dz, dz64)
        if name == "fused":
            assert bool(torch.isfinite(dP).all()) and (dz is None or bool(torch.isfinite(dz).all()))
            if off["NPAD"] > off["NTOT"]:
                assert bool((dP[..., off["NTOT"]:] == 0).all()), "padding columns of d proj must be zero"
        res[name] = r
    bad = []
    for kk, vv in res["fused"].items():
        fac, fl = (1.5, 1e-5) if kk in ("dz", "dlam") else (1.05, 1e-6)
        if vv > fac * res["glue"][kk] + fl:
            bad.append(f"{kk}: fused {vv:.3e} > glue {res['glue'][kk]:.3e}")
    assert not bad, "; ".join(bad) + f"\nfused {res['fused']}\nglue  {res['glue']}"


# ------------------------------------------------------------------------------------------------ properties
@cuda
@pytest.mark.parametrize("L,t_cut", [(1000, 437), pytest.param(2048, 1024, marks=pytest.mark.slow), (2048, 1)])
def test_fused_determinism_and_causality(FP, L, t_cut):
    H = 8
    proj, z, off = make_proj(FP, 2, L, H, 21)
    lam = torch.tensor([0.31, 3.0, 1e-7, 0.8, 0.31, 0.3, 0.29, 0.31], device=DEV)
    dy = torch.randn(2, L, H * 64, device=DEV, generator=torch.Generator(device=DEV).manual_seed(5)).to(torch.bfloat16)

    def run(P_, Z_, dy_):
        Pp = P_.detach().clone().requires_grad_(True)
        Zp = Z_.detach().clone().requires_grad_(True)
        Lp = lam.detach().clone().requires_grad_(True)
        snap = (Pp.detach().clone(), Zp.detach().clone())
        y = FP.sisa_fused(Pp, Zp, Lp, H)
        gP, gZ, gL = torch.autograd.grad(y, (Pp, Zp, Lp), dy_)
        same_in = torch.equal(Pp.detach(), snap[0]) and torch.equal(Zp.detach(), snap[1])
        return y.detach(), gP, gZ, gL, same_in
    a = run(proj, z, dy)
    b = run(proj, z, dy)
    assert all(torch.equal(x, y_) for x, y_ in zip(a[:4], b[:4])), "not deterministic"
    assert a[4] and b[4], "inputs were modified"
    # causality: perturb proj and z at positions >= t (θ and α change as well -> suffixes of g, Φ and F change)
    P2 = proj.clone()
    z2 = z.clone()
    gen = torch.Generator(device=DEV).manual_seed(77)
    P2[:, t_cut:] = (torch.randn(P2[:, t_cut:].shape, device=DEV, generator=gen) * 0.5).to(P2.dtype)
    z2[:, t_cut:] = torch.randn(z2[:, t_cut:].shape, device=DEV, generator=gen)
    dyc = dy.clone()
    dyc[:, t_cut:] = 0
    c1 = run(proj, z, dyc)
    c2 = run(P2, z2, dyc)
    assert torch.equal(c1[0][:, :t_cut], c2[0][:, :t_cut]), "output prefix depends on the future"
    assert torch.equal(c1[1][:, :t_cut], c2[1][:, :t_cut]) and torch.equal(c1[2][:, :t_cut], c2[2][:, :t_cut]), \
        "gradient prefix depends on the future"
    assert torch.equal(c1[3], c2[3]), "dlam depends on the future"
    assert bool((c1[1][:, t_cut:] == 0).all() and (c1[2][:, t_cut:] == 0).all()
                and (c2[1][:, t_cut:] == 0).all() and (c2[2][:, t_cut:] == 0).all()), "gradients at >= t not zero"


@cuda
def test_fused_input_validation(FP):
    H, L = 4, 70
    proj, z, off = make_proj(FP, 1, L, H, seed=3)
    lam = torch.full((H,), 0.3, device=DEV)
    with pytest.raises(ValueError):
        FP.sisa_fused(proj[..., :-16], z, lam, H)                  # wrong NPAD
    with pytest.raises(TypeError):
        FP.sisa_fused(proj.float(), z, lam, H)                     # fp32 proj
    with pytest.raises(TypeError):
        FP.sisa_fused(proj, z.double(), lam, H)                    # fp64 z
    with pytest.raises(ValueError):
        FP.sisa_fused(proj, z, lam[:2], H)                         # wrong lam shape
    y = FP.sisa_fused(proj, None, lam, H)                          # no forget gate
    assert y.shape == (1, L, H * 64) and y.dtype == torch.bfloat16 and bool(torch.isfinite(y.float()).all())
    y16 = FP.sisa_fused(proj.to(torch.float16), z, lam, H)
    assert y16.dtype == torch.float16 and bool(torch.isfinite(y16.float()).all())
