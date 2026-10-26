"""Runtime guards of the Triton paths and backend selection.

* Non-finite guard of the fused layer: NaN/±inf injected into the α, θ, Q, K, B, C, V columns of ``proj``, into ``z``
  or into λ, followed by one forward pass and a synchronisation. Every case where a non-finite value reaches the
  score must raise. Cases that must not raise: α = -inf (softplus = 0, no decay: finite), z = +inf (log σ = 0, no
  forgetting: finite), V (not guarded; the output must then be non-finite rather than silently wrong) and check mode
  "off" (the output must be non-finite: the decay clamps propagate NaN). In "sync" mode the guard raises
  ``FloatingPointError`` during the forward call.
* Monotonicity check of the unfused attention: an increasing g raises ``ValueError`` in "sync" mode and trips a
  device-side assert in "async" mode.
* q/k table contract of the fused kernels (internal): tables that are too short or violate the read contract raise.
* Backend selection: forced Triton implementations on ineligible inputs raise, "auto" falls back to "torch".

A failed device-side assert stops the CUDA context, so every case that is expected to trip one runs in a subprocess.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


# CUDA tests are skipped by tests/conftest.py when the Triton kernels cannot run on this machine.
cuda = pytest.mark.cuda


# ============================================================================================ subprocess side
def _layer_inputs(B, H, L, has_f, seed):
    from sisa.kernels.fused_prep import fused_layout
    dev = "cuda"
    off = fused_layout(H)
    g0 = torch.Generator(device=dev).manual_seed(seed)
    proj = torch.randn(B, L, off["NPAD"], device=dev, generator=g0) * .7
    z = (torch.randn(B, L, H, device=dev, generator=g0) + 2) if has_f else None
    proj = proj.to(torch.bfloat16)
    proj[..., off["NTOT"]:] = 0.
    lam = torch.rand(H, device=dev, generator=g0) + .2
    return proj, z, lam, off


def _child_fused(s):
    """One fused forward with an injected value, then a synchronisation."""
    from sisa import backends
    from sisa.kernels.fused_prep import sisa_fused
    backends.set_check_mode(s.get("check", "async"))
    H, L, B = 8, 512, 2
    proj, z, lam, off = _layer_inputs(B, H, L, s["has_f"], seed=11)
    val = dict(nan=float("nan"), pinf=float("inf"), ninf=float("-inf"))[s["val"]]
    rows = dict(head=slice(None), r100=slice(100, 101), last=slice(L - 1, L))[s["where"]]
    col = dict(A=off["A"], TH=off["TH"], Q=off["Q"], K=off["K"], B=off["B"], C=off["C"], V=off["V"])
    t = s["target"]
    with torch.no_grad():
        if t in col:
            proj[:, rows, col[t]] = val              # first element of head 0 (α: the column of head 0)
        elif t == "z":
            z[:, rows, 0] = val
        elif t == "lam":
            lam[0] = val
    proj.requires_grad_(True)
    out = dict(spec=s)
    stage = "forward"
    try:
        y = sisa_fused(proj, z, lam, H)
        stage = "sync"
        torch.cuda.synchronize()
        out.update(raised=False, y_finite=bool(torch.isfinite(y.float()).all()))
    except BaseException as ex:  # noqa: BLE001
        out.update(raised=True, stage=stage, exc=type(ex).__name__, msg=str(ex).splitlines()[0][:160])
    return out


def _child_monotone(s):
    """Unfused attention with an increasing g."""
    from sisa import backends
    from sisa.kernels.attention import sisa_attention_triton
    backends.set_check_mode(s.get("check", "async"))
    B, H, L = 1, 2, 300
    gen = torch.Generator(device="cuda").manual_seed(3)
    q, k, v = [torch.randn(B, H, L, 64, device="cuda", generator=gen).bfloat16() for _ in range(3)]
    c, b = [torch.randn(B, H, L, 32, device="cuda", generator=gen).bfloat16() for _ in range(2)]
    g = torch.cumsum(-torch.rand(B, H, L, device="cuda", generator=gen) * 0.1, -1)
    g[:, :, 150:] += 0.5                       # one increase of +0.5 between positions 149 and 150
    lam = torch.full((H,), 0.3, device="cuda")
    out = dict(spec=s)
    stage = "forward"
    try:
        sisa_attention_triton(q, k, v, c, b, g.contiguous(), lam, None)
        stage = "sync"
        torch.cuda.synchronize()
        out.update(raised=False)
    except BaseException as ex:  # noqa: BLE001
        out.update(raised=True, stage=stage, exc=type(ex).__name__, msg=str(ex).splitlines()[0][:160])
    return out


def _child_main(payload):
    frac = float(os.environ.get("SISA_TEST_CUDA_MEM_FRACTION", "0.10"))
    torch.cuda.set_per_process_memory_fraction(frac)
    for s in json.loads(payload):
        fn = _child_monotone if s.get("kind") == "monotone" else _child_fused
        print("[child] " + json.dumps(fn(s)), flush=True)


# ============================================================================================ parent side
def _run_child(specs, timeout=600):
    import sisa
    src = os.path.dirname(os.path.dirname(os.path.abspath(sisa.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", json.dumps(specs)],
                       capture_output=True, text=True, env=env, timeout=timeout, cwd=HERE)
    out = [json.loads(line[8:]) for line in r.stdout.splitlines() if line.startswith("[child] ")]
    if len(out) != len(specs):
        raise AssertionError(f"child process failed (rc={r.returncode}): {(r.stderr or '')[-800:]}")
    return out


def _expect_raise(s):
    if s.get("check") == "off" or s["target"] in ("none", "V"):
        return False
    if s["target"] == "A" and s["val"] == "ninf":
        return False                     # softplus(-inf) = 0 -> no decay (finite)
    if s["target"] == "z" and s["val"] == "pinf":
        return False                     # log σ(+inf) = 0 -> no forgetting (finite)
    return True


def _all_specs():
    specs = []
    for tgt in ("A", "TH", "Q", "K", "B", "C", "V", "z", "lam"):
        for val in ("nan", "pinf", "ninf"):
            for where in (("head", "r100", "last") if tgt != "lam" else ("head",)):
                if val != "nan" and where != "r100":
                    continue
                specs.append(dict(target=tgt, val=val, where=where, has_f=True))
    for tgt in ("A", "TH", "Q"):         # no forget gate
        specs.append(dict(target=tgt, val="nan", where="r100", has_f=False))
    for chk in ("sync", "off"):
        specs.append(dict(target="A", val="nan", where="head", has_f=True, check=chk))
    specs.append(dict(target="none", val="nan", where="head", has_f=True))
    return specs


_FAST_SPECS = [dict(target="A", val="nan", where="head", has_f=True),
               dict(target="TH", val="nan", where="r100", has_f=True),
               dict(target="Q", val="nan", where="last", has_f=True),
               dict(target="C", val="pinf", where="r100", has_f=True),
               dict(target="z", val="nan", where="r100", has_f=True),
               dict(target="lam", val="nan", where="head", has_f=True),
               dict(target="A", val="nan", where="head", has_f=True, check="sync")]


def _sid(s):
    return "-".join(str(s[k]) for k in ("target", "val", "where")) + ("" if s["has_f"] else "-noF") + \
        (f"-{s['check']}" if "check" in s else "")


def _check_row(s, r):
    want = _expect_raise(s)
    assert r.get("raised") == want, f"{_sid(s)}: raised={r.get('raised')} expected {want} ({r})"
    if s.get("check") == "sync":
        assert r.get("exc") == "FloatingPointError" and r.get("stage") == "forward", r
    if not want:
        # V and check 'off' must give a non-finite output (never a silently wrong finite one); the rest is finite
        want_fin = not (s["target"] == "V" or s.get("check") == "off")
        assert r.get("y_finite") == want_fin, f"{_sid(s)}: y_finite={r.get('y_finite')} expected {want_fin}"


_RAISING = [s for s in _all_specs() if _expect_raise(s)]
_QUIET = [s for s in _all_specs() if not _expect_raise(s)]


@cuda
@pytest.mark.parametrize("spec", [pytest.param(s, id=_sid(s), marks=() if s in _FAST_SPECS else (pytest.mark.slow,))
                                  for s in _RAISING])
def test_nan_guard_raises(spec):
    (r,) = _run_child([spec])
    _check_row(spec, r)


@cuda
def test_nan_guard_quiet_cases():
    """Cases that must not raise, all in one subprocess (none of them trips an assert)."""
    for s, r in zip(_QUIET, _run_child(_QUIET)):
        _check_row(s, r)


@cuda
def test_monotone_check_async_trips_device_assert():
    (r,) = _run_child([dict(kind="monotone", check="async")])
    # the assert is asynchronous: it surfaces at the next CUDA call, which may already be inside the forward call
    assert r["raised"] and r["exc"] == "RuntimeError" and "device-side assert" in r["msg"], r


@cuda
def test_monotone_check_sync_and_off():
    (r_sync, r_off) = _run_child([dict(kind="monotone", check="sync"), dict(kind="monotone", check="off")])
    assert r_sync["raised"] and r_sync["exc"] == "ValueError" and r_sync["stage"] == "forward", r_sync
    assert not r_off["raised"], r_off


# ============================================================================================ q/k table contract
@cuda
def test_qk_table_contract():
    """The kernels read the q/k tables without bounds checks, so the host check must reject every table outside the
    contract before any launch (these raise on the host, the CUDA context is unaffected)."""
    from sisa.kernels.fused_prep import _launch_prep_forward, fused_layout
    H, L = 8, 256
    off = fused_layout(H)
    proj = torch.zeros(1, L, off["NPAD"], device="cuda", dtype=torch.bfloat16)
    z = torch.zeros(1, L, H, device="cuda")
    ones, zeros = torch.ones(4096, 32, device="cuda"), torch.zeros(4096, 32, device="cuda")
    c64 = torch.ones(L, 64, device="cuda")
    bad = {
        "short": (ones[:L - 1], zeros[:L - 1]),
        "cols64": (c64, c64),
        "colstride2": (c64[:, ::2], c64[:, ::2]),
        "fp64": (ones[:L].double(), zeros[:L].double()),
        "rowstride_mismatch": (ones[:L], torch.cat([zeros[:L], zeros[:L]], 1)[:, :32]),
        "cpu": (ones[:L].cpu(), zeros[:L].cpu()),
        "dim3": (ones[:L][None], zeros[:L][None]),
    }
    for name, (c_, s_) in bad.items():
        with pytest.raises(RuntimeError):
            _launch_prep_forward(proj, z, c_, s_, H)
        torch.cuda.synchronize()
    out = _launch_prep_forward(proj, z, ones, zeros, H)       # a long table is fine
    torch.cuda.synchronize()
    assert out["QR"].shape == (1, L, H, 64)


# ============================================================================================ backends (CPU)
def test_check_mode_api():
    from sisa import backends
    assert backends.get_check_mode() in ("async", "sync", "off")
    old = backends.get_check_mode()
    try:
        for m in ("sync", "off", "async"):
            backends.set_check_mode(m)
            assert backends.get_check_mode() == m
        with pytest.raises(ValueError):
            backends.set_check_mode("1")
    finally:
        backends.set_check_mode(old)


def test_select_impl_cpu():
    from sisa import backends
    cfg = SimpleNamespace(d_model=128, n_head=2, d_state=32)
    x = torch.zeros(1, 4, 128)
    assert backends.select_impl("torch", x, cfg) == "torch"
    assert backends.select_impl("auto", x, cfg) == "torch"
    for impl in ("triton", "triton_fused"):
        with pytest.raises(ValueError):
            backends.select_impl(impl, x, cfg)
    with pytest.raises(ValueError):
        backends.select_impl("flash", x, cfg)
    assert not backends.fused_eligible(x)
    ok, why = backends.triton_status("cpu")
    assert not ok and why


@cuda
def test_select_impl_cuda():
    from sisa import backends
    cfg = SimpleNamespace(d_model=128, n_head=2, d_state=32)
    x = torch.zeros(1, 4, 128, device="cuda")
    assert backends.select_impl("auto", x, cfg) == "triton"                 # fp32 compute dtype
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert backends.fused_eligible(x)
        assert backends.select_impl("auto", x, cfg) == "triton_fused"
        assert backends.select_impl("triton", x, cfg) == "triton"
    with pytest.raises(ValueError):
        backends.select_impl("triton_fused", x, cfg)                         # fp32 without autocast
    assert backends.select_impl("triton_fused", x.bfloat16(), cfg) == "triton_fused"
    with pytest.raises(ValueError):
        backends.select_impl("triton", x.double(), cfg)
    wide = SimpleNamespace(d_model=256, n_head=2, d_state=32)                # d_head 128
    with pytest.raises(ValueError):
        backends.select_impl("triton", torch.zeros(1, 4, 256, device="cuda"), wide)
    backends._WARNED.clear()                                                  # the fallback warning is issued once
    with pytest.warns(UserWarning):
        assert backends.select_impl("auto", torch.zeros(1, 4, 256, device="cuda"), wide) == "torch"
    odd_state = SimpleNamespace(d_model=128, n_head=2, d_state=16)
    with pytest.raises(ValueError):
        backends.select_impl("triton_fused", x.bfloat16(), odd_state)
    text = backends.describe()
    assert "triton kernels: available" in text
    cfg32 = backends.attention_config(x.device, torch.float32)
    cfg16 = backends.attention_config(x.device, torch.bfloat16)
    assert cfg32["T"] == 32 and cfg16["T"] == 64
    assert cfg16["bn_dq"] == cfg16["bm_dkv"] == cfg16["T"]


def test_kernels_package_is_lazy():
    import sisa.kernels as K
    assert set(K.__all__) == {"sisa_attention_triton", "sisa_fused", "fused_layout", "concat_fused_weights",
                              "linear_cross_entropy"}
    with pytest.raises(AttributeError):
        K.does_not_exist  # noqa: B018


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--child":
    _child_main(sys.argv[2])
