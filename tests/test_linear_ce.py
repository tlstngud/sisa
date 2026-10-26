"""Tests of the chunked linear + cross-entropy (:mod:`sisa.kernels.linear_ce`).

For each case the loss, dL/dh and dL/dW (backward seeded with g, e.g. 1/grad_accum) are computed with
  ref64    fp64 math on the fp32 master inputs (no low precision anywhere)               -- "the math"
  ref64q   fp64 math on bf16-rounded inputs (h, W rounded like autocast does)   -- isolates the op's own rounding
  eager    the eager formula F.cross_entropy(F.linear(h, w).float(), y) under autocast
  lmce     linear_cross_entropy under autocast (default chunks and odd chunks)
and compared as rel-L2 = ||a - ref|| / ||ref||. Rule: err(lmce) <= 2 x err(eager) + 1e-7 against both references for
every tensor. Semantic checks: ignore_index, gradients for h only / W only, no_grad forward, bf16 tensors without
autocast, fp16 autocast, a non-power-of-two seed, 2-D vs 3-D input, NaN/inf propagation and the grad_scale promise.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F


# CUDA tests are skipped by tests/conftest.py when the Triton kernels cannot run on this machine.
cuda = pytest.mark.cuda
DEV = "cuda"


@pytest.fixture(autouse=True)
def _no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def _lce():
    from sisa.kernels.linear_ce import linear_cross_entropy
    return linear_cross_entropy


def _eager(h, w, labels, ignore_index=-100):
    logits = F.linear(h, w)
    return F.cross_entropy(logits.view(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=ignore_index)


def rel(a, b):
    a = a.double()
    b = b.double()
    nb = b.norm().item()
    return (a - b).norm().item() / (nb if nb > 0 else 1.0)


def ref64(h, w, labels, g, quant, ignore_index=-100, chunk=1024):
    """fp64 reference, chunked over tokens (never materialises (N, V) in fp64)."""
    D = h.shape[-1]
    x = h.reshape(-1, D)
    W = w
    if quant:
        x = x.to(torch.bfloat16)
        W = W.to(torch.bfloat16)
    x = x.double()
    W = W.double()
    y = labels.reshape(-1)
    valid = y != ignore_index
    n = valid.sum().double()
    loss = torch.zeros((), dtype=torch.float64, device=h.device)
    dx = torch.zeros_like(x)
    dW = torch.zeros_like(W)
    for s in range(0, x.shape[0], chunk):
        e = min(x.shape[0], s + chunk)
        lg = x[s:e] @ W.t()
        lse = torch.logsumexp(lg, 1)
        yc = y[s:e]
        vc = valid[s:e]
        ys = torch.where(vc, yc, torch.zeros_like(yc))
        tgt = lg.gather(1, ys[:, None])[:, 0]
        loss += torch.where(vc, lse - tgt, torch.zeros_like(lse)).sum()
        G = torch.exp(lg - lse[:, None])
        G[torch.arange(e - s, device=h.device), ys] -= 1.0
        G *= (vc.double() * (g / n))[:, None]
        dx[s:e] = G @ W
        dW += G.t() @ x[s:e]
        del lg, G
    return (loss / n), dx.view(h.shape), dW


def run_path(fn, h, w, labels, g, need=(True, True), autocast=torch.bfloat16, **kw):
    hh = h.detach().clone().requires_grad_(need[0])
    ww = w.detach().clone().requires_grad_(need[1])
    if autocast is not None:
        with torch.autocast("cuda", dtype=autocast):
            loss = fn(hh, ww, labels, **kw)
    else:
        loss = fn(hh, ww, labels, **kw)
    (loss * g).backward()
    torch.cuda.synchronize()
    return loss.detach().float(), (hh.grad if need[0] else None), (ww.grad if need[1] else None)


def compare(res, refs, tol=2.0, floor=1e-7):
    """res: {path: (loss, dh, dw)} including 'eager'; refs: {name: (loss, dh, dw)}. Returns failure strings."""
    fails = []
    names = ("loss", "dh", "dw")
    for p, t in res.items():
        if p == "eager":
            continue
        for i, nm in enumerate(names):
            if t[i] is None:
                continue
            for r, R in refs.items():
                e_new, e_ref = rel(t[i], R[i]), rel(res["eager"][i], R[i])
                if not (e_new <= tol * e_ref + floor):
                    fails.append(f"{p} {nm} vs {r}: {e_new:.3e} > {tol} x eager {e_ref:.3e}")
    return fails


def make_case(D, N, V, seed, sharp=3.0):
    """Synthetic final-norm-like hidden states and labels drawn from the model's own softmax (peaked targets)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    B = max(1, N // 1024)
    L = N // B
    h = torch.randn(B, L, D, device=DEV, generator=g)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    h = h * (1.0 + 0.8 * torch.rand(D, device=DEV, generator=g))
    w = torch.randn(V, D, device=DEV, generator=g) * (sharp / math.sqrt(D))
    lab = torch.empty(B * L, dtype=torch.int64, device=DEV)
    xf = h.reshape(-1, D)
    for s in range(0, B * L, 512):
        p = torch.softmax((xf[s:s + 512] @ w.t()).float(), -1)
        lab[s:s + 512] = torch.multinomial(p, 1, generator=g)[:, 0]
    return h, w, lab.view(B, L)


_SHAPES = [("D512_V50277", 512, 2048, 50277, 1 / 16), ("D768_Vodd", 768, 1024, 1003, 1 / 32),
           pytest.param(("D1024_V50277_N4096", 1024, 4096, 50277, 1 / 64), marks=pytest.mark.slow)]


@cuda
@pytest.mark.parametrize("case", _SHAPES, ids=lambda c: c[0])
def test_accuracy_vs_fp64(case):
    _, D, N, V, g = case
    lce = _lce()
    h, w, lab = make_case(D, N, V, seed=D + V)
    refs = {"ref64": ref64(h, w, lab, g, False), "ref64q": ref64(h, w, lab, g, True)}
    res = {"eager": run_path(_eager, h, w, lab, g), "lmce": run_path(lce, h, w, lab, g),
           "lmce_c300": run_path(lce, h, w, lab, g, chunk_tokens=300),
           "lmce_full": run_path(lce, h, w, lab, g, chunk_tokens=N),
           "lmce_gs": run_path(lce, h, w, lab, g, grad_scale=g)}
    fails = compare(res, refs)
    assert not fails, "\n".join(fails)
    # the loss differs from the eager one only by the bf16 rounding of individual logits (GEMM algorithm choice)
    assert abs(res["lmce"][0].item() - res["eager"][0].item()) < 1e-3


@cuda
def test_semantics():
    lce = _lce()
    D, N, V = 512, 1024, 4099
    h, w, lab = make_case(D, N, V, seed=7)
    # ignore_index: 5% of the labels -> -100; ignored rows get exactly zero gradient
    lab_i = lab.clone()
    gm = torch.Generator(device=DEV).manual_seed(3)
    lab_i[torch.rand(lab.shape, device=DEV, generator=gm) < 0.05] = -100
    refs = {"ref64": ref64(h, w, lab_i, 1 / 16, False), "ref64q": ref64(h, w, lab_i, 1 / 16, True)}
    res = {"eager": run_path(_eager, h, w, lab_i, 1 / 16), "lmce": run_path(lce, h, w, lab_i, 1 / 16),
           "lmce_c100": run_path(lce, h, w, lab_i, 1 / 16, chunk_tokens=100)}
    assert not compare(res, refs)
    assert res["lmce"][1].reshape(-1, D)[lab_i.reshape(-1) == -100].abs().max().item() == 0.0
    # a non-power-of-two seed (one extra rounding of G allowed)
    gg = 0.3
    refs = {"ref64": ref64(h, w, lab, gg, False), "ref64q": ref64(h, w, lab, gg, True)}
    res = {"eager": run_path(_eager, h, w, lab, gg), "lmce": run_path(lce, h, w, lab, gg),
           "lmce_gs": run_path(lce, h, w, lab, gg, grad_scale=gg)}
    assert not compare(res, refs)
    # gradients for h only / W only: bit-identical to the run with both
    both = run_path(lce, h, w, lab, 1 / 16)
    for need in ((True, False), (False, True)):
        r1 = run_path(lce, h, w, lab, 1 / 16, need=need)
        assert torch.equal(r1[0], both[0])
        for x, y, n in zip(r1[1:], both[1:], need):
            assert (x is None) == (not n) and (x is None or torch.equal(x, y))
    # no_grad forward == grad forward loss, and returns a tensor without grad
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        l0 = lce(h, w, lab)
    assert torch.equal(l0.float(), both[0]) and not l0.requires_grad
    # bf16 tensors without autocast (pure low-precision regime)
    hb, wb = h.to(torch.bfloat16), w.to(torch.bfloat16)
    refs = {"ref64": ref64(hb.float(), wb.float(), lab, 1 / 16, False)}
    res = {"eager": run_path(_eager, hb, wb, lab, 1 / 16, autocast=None),
           "lmce": run_path(lce, hb, wb, lab, 1 / 16, autocast=None)}
    assert not compare(res, refs)
    assert res["lmce"][1].dtype == torch.bfloat16 and res["lmce"][2].dtype == torch.bfloat16
    # fp16 autocast
    refs = {"ref64": ref64(h, w, lab, 1 / 16, False)}
    res = {"eager": run_path(_eager, h, w, lab, 1 / 16, autocast=torch.float16),
           "lmce": run_path(lce, h, w, lab, 1 / 16, autocast=torch.float16)}
    assert not compare(res, refs)
    # 2-D input == 3-D input (bit-identical)
    r3 = run_path(lce, h, w, lab, 1 / 16)
    r2 = run_path(lce, h.reshape(-1, D), w, lab.reshape(-1), 1 / 16)
    assert torch.equal(r3[0], r2[0]) and torch.equal(r3[1].reshape(-1, D), r2[1]) and torch.equal(r3[2], r2[2])
    # deterministic
    r3b = run_path(lce, h, w, lab, 1 / 16)
    assert all(torch.equal(a, b) for a, b in zip(r3, r3b))


@cuda
def test_nan_inf_propagation():
    lce = _lce()
    D, N, V = 256, 512, 20011       # V spans three 8192-column blocks
    h, w, lab = make_case(D, N, V, seed=9)
    hn = h.clone()
    hn[0, 5, 3] = float("nan")
    rl = run_path(lce, hn, w, lab, 1 / 16)
    assert bool(torch.isnan(rl[0])) and bool(torch.isnan(rl[2]).any()) and bool(torch.isnan(rl[1][0, 5]).any())
    # a NaN in one weight row (one logit column for every token), in the first and in a later column block
    for col in (5, 2 * 8192 + 17, V - 1):
        wn = w.clone()
        wn[col, 7] = float("nan")
        rp = run_path(_eager, h, wn, lab, 1 / 16)
        rl = run_path(lce, h, wn, lab, 1 / 16)
        assert bool(torch.isnan(rl[0])) and bool(torch.isnan(rp[0])), col
    hi = h.clone()
    hi[0, 9, :] = float("inf")
    rl = run_path(lce, hi, w, lab, 1 / 16)
    assert not bool(torch.isfinite(rl[0]))


@cuda
def test_grad_scale_promise_broken_trips_assert():
    """A backward seed different from grad_scale trips a device-side assert (subprocess: it stops the context)."""
    import sisa
    src = os.path.dirname(os.path.dirname(os.path.abspath(sisa.__file__)))
    frac = os.environ.get("SISA_TEST_CUDA_MEM_FRACTION", "0.10")
    code = "\n".join([
        "import torch",
        f"torch.cuda.set_per_process_memory_fraction({float(frac)})",
        "from sisa.kernels.linear_ce import linear_cross_entropy as f",
        "h = torch.randn(64, 512, device='cuda', requires_grad=True)",
        "w = torch.randn(1000, 512, device='cuda', requires_grad=True)",
        "y = torch.randint(0, 1000, (64,), device='cuda')",
        "with torch.autocast('cuda', dtype=torch.bfloat16):",
        "    l = f(h, w, y, grad_scale=0.25)",
        "(l * 0.5).backward()",
        "torch.cuda.synchronize()",
        "print('NO_ASSERT')"])
    env = dict(os.environ)
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    pr = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=env)
    assert pr.returncode != 0 and "NO_ASSERT" not in pr.stdout, pr.stdout + pr.stderr[-500:]
    assert "device-side assert" in pr.stderr or "assert" in pr.stderr.lower()


def test_cpu_and_fp32_fall_back_to_eager():
    pytest.importorskip("triton")
    from sisa.kernels.linear_ce import linear_cross_entropy, reference_linear_cross_entropy
    torch.manual_seed(0)
    h = torch.randn(2, 5, 16, requires_grad=True)
    w = torch.randn(37, 16, requires_grad=True)
    y = torch.randint(0, 37, (2, 5))
    y[0, 1] = -100
    a = linear_cross_entropy(h, w, y)
    b = reference_linear_cross_entropy(h, w, y)
    assert torch.equal(a, b)
    a.backward()
    assert h.grad is not None and w.grad is not None
