"""Backend selection, device checks, kernel launch tables and the runtime check mode.

SISA attention has three implementations of the same function:

* ``"torch"``: an exact blocked reference in plain PyTorch. Works on any device and dtype.
* ``"triton"``: torch glue (projections, cumulative sums, rotation) plus the Triton attention kernel
  :func:`sisa.kernels.attention.sisa_attention_triton`.
* ``"triton_fused"``: one projection GEMM, then fused Triton preparation kernels plus the Triton attention kernel
  (:func:`sisa.kernels.fused_prep.sisa_fused`).

``"auto"`` picks ``triton_fused`` when eligible, else ``triton``, else ``torch``. Forcing a Triton implementation on an
input it cannot handle raises ``ValueError``; nothing ever falls back silently.

Triton requirements: an NVIDIA GPU with compute capability 8.0 or newer (the kernels use ``cvt.rn.bf16x2.f32`` inline
PTX), at least 99 KiB of opt-in shared memory per block, Triton 3.3 or newer, ``d_head == 64`` and ``d_state == 32``.
The fused path additionally needs a bf16/fp16 compute dtype (autocast on, or low-precision parameters).

Launch tables are chosen by the shared-memory budget of the device, never by its compute capability. No autotuning
is used anywhere: every launch configuration below was validated against fp64 references.
"""
from __future__ import annotations

import importlib
import threading
import warnings
from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:  # pragma: no cover
    from .config import SISAConfig

DeviceLike = Union[torch.device, str, int]

__all__ = [
    "IMPLS",
    "triton_status",
    "fused_eligible",
    "set_check_mode",
    "get_check_mode",
    "describe",
    "select_impl",
    "device_info",
    "attention_config",
]

IMPLS = ("auto", "torch", "triton", "triton_fused")
MIN_TRITON = (3, 3)

# Shared-memory classes (bytes of opt-in shared memory per block).
SMEM227 = 200 * 1024  # Hopper class (H100: 232,448 B)
SMEM99 = 99 * 1024  # 101,376 B class: sm86 (A6000), sm89 (RTX 4090), sm120 (RTX PRO Blackwell); A100 also lands here

# Attention launch tables. T = tile size, nw = num_warps, ns_* = num_stages of the forward, dq and dkv kernels.
# ns_dkv applies when the dkv kernel stores fp32 dc/db (the fused path); ns_dkv_lp when it stores bf16/fp16 dc/db
# (the unfused entry point). num_stages never changes the arithmetic (results are bit-identical across stage counts);
# num_warps changes fp32 reduction order only.
_ATTN_BASE = {
    "smem227": dict(T=64, nw=4, ns_f=2, ns_dq=3, ns_dkv=1, ns_dkv_lp=2),
    "smem99": dict(T=64, nw=4, ns_f=2, ns_dq=2, ns_dkv=2, ns_dkv_lp=2),
}
# fp32 attention inputs (tests and fp32 inference): T = 32 on every device.
_ATTN_FP32 = dict(T=32, nw=4, ns_f=2, ns_dq=1, ns_dkv=1, ns_dkv_lp=1)

_CHECK_MODES = ("async", "sync", "off")
_STATE = {"check": "async"}
_LOCK = threading.Lock()
_DEVICE_INFO: dict[int, dict] = {}
_TRITON_STATUS: dict[int, tuple[bool, str]] = {}
_ATTN_CFG: dict[tuple[int, torch.dtype], dict] = {}
_WARNED: set = set()


# ---------------------------------------------------------------------------------------------------- check mode
def set_check_mode(mode: str) -> None:
    """Set how the Triton paths check their inputs at run time.

    ``"async"`` (default): device-side asserts (``torch._assert_async``), no host synchronisation. A failed check
    stops the CUDA context: the next CUDA call raises ``RuntimeError`` (device-side assert) and the process must be
    restarted. ``"sync"``: the check synchronises once per call and raises a Python exception right away
    (``ValueError`` for a non-monotone decay ``g``, ``FloatingPointError`` for non-finite values in the fused layer).
    ``"off"``: no checks (benchmarks only).

    Checked: ``g`` is non-increasing (unfused Triton attention; the fused path builds ``g`` monotone by construction),
    and the fused layer sees only finite values in every input of the score (q, k, c, b, lambda, g, F).
    """
    if mode not in _CHECK_MODES:
        raise ValueError(f"check mode must be one of {_CHECK_MODES}, got {mode!r}")
    _STATE["check"] = mode


def get_check_mode() -> str:
    """The current check mode (``"async"``, ``"sync"`` or ``"off"``)."""
    return _STATE["check"]


# ---------------------------------------------------------------------------------------------------- device facts
def _cuda_index(device: DeviceLike | None = None) -> int:
    if device is None:
        return torch.cuda.current_device()
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"not a CUDA device: {dev}")
    return dev.index if dev.index is not None else torch.cuda.current_device()


def _smem_class(smem):
    if smem is None:
        return None
    if smem >= SMEM227:
        return "smem227"
    if smem >= SMEM99:
        return "smem99"
    return None


def _triton_version():
    try:
        import triton
    except Exception:  # noqa: BLE001 - any import failure means "no Triton"
        return None
    return triton.__version__


def _version_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split("+")[0].split(".")[:3]:
        num = ""
        for ch in part:
            if not ch.isdigit():
                break
            num += ch
        out.append(int(num) if num else 0)
    return tuple(out)


def device_info(device: DeviceLike | None = None) -> dict:
    """Facts about a CUDA device that the launch tables depend on.

    Shared memory is read from Triton's driver (``max_shared_mem``, the limit the Triton compiler checks). Older torch
    builds lack ``shared_memory_per_block_optin``, and guessing 99 KiB there would silently pick the slow table on
    Hopper; torch's value is used only when Triton cannot be queried.
    """
    idx = _cuda_index(device)
    info = _DEVICE_INFO.get(idx)
    if info is not None:
        return dict(info)
    p = torch.cuda.get_device_properties(idx)
    smem, src = None, None
    try:
        import triton

        smem = triton.runtime.driver.active.utils.get_device_properties(idx).get("max_shared_mem")
        src = "triton" if smem is not None else None
    except Exception:  # noqa: BLE001 - fall back to torch below
        smem = None
    if smem is None:
        smem = getattr(p, "shared_memory_per_block_optin", None)
        src = "torch" if smem is not None else None
    info = dict(
        index=idx,
        name=p.name,
        capability=(p.major, p.minor),
        sm_count=int(p.multi_processor_count),
        smem_optin=None if smem is None else int(smem),
        smem_source=src,
        smem_class=_smem_class(None if smem is None else int(smem)),
        total_memory=int(p.total_memory),
    )
    _DEVICE_INFO[idx] = info
    return dict(info)


def attention_config(device: DeviceLike | None, dtype: torch.dtype) -> dict:
    """Launch configuration of the attention kernels for ``device`` and input ``dtype`` (cached).

    Keys: T, nw, ns_f, ns_dq, ns_dkv, ns_dkv_lp, bn_dq, bm_dkv. ``bn_dq`` and ``bm_dkv`` (the key slice of the dq
    kernel and the query slice of the dkv kernel) are frozen to T: smaller slices combined with software pipelining
    produced wrong dkv results and illegal memory accesses.
    """
    idx = _cuda_index(device)
    key = (idx, dtype)
    cfg = _ATTN_CFG.get(key)
    if cfg is not None:
        return cfg
    if dtype == torch.float32:
        cfg = dict(_ATTN_FP32)
    else:
        cls = device_info(idx)["smem_class"]
        if cls is None:
            raise RuntimeError(
                f"{device_info(idx)['name']}: less than 99 KiB of opt-in shared memory per block; "
                "the SISA Triton kernels do not support this GPU (use impl='torch')"
            )
        cfg = dict(_ATTN_BASE[cls])
        assert cfg["T"] == 64  # frozen: the decay-factor grouping of the fused preparation uses 64-row blocks
    cfg["bn_dq"] = cfg["T"]
    cfg["bm_dkv"] = cfg["T"]
    assert cfg["bn_dq"] == cfg["T"] and cfg["bm_dkv"] == cfg["T"]  # frozen (see docstring)
    _ATTN_CFG[key] = cfg
    return cfg


# ---------------------------------------------------------------------------------------------------- eligibility
def triton_status(device: DeviceLike | None = None) -> tuple[bool, str]:
    """Whether the Triton kernels can run on ``device`` (default: the current CUDA device), with the reason.

    Returns ``(True, summary)`` or ``(False, why_not)``. The result is cached per device.
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    if getattr(torch.version, "hip", None):
        return False, "ROCm builds are not supported by the SISA Triton kernels"
    if device is not None and torch.device(device).type != "cuda":
        return False, f"device {torch.device(device)} is not a CUDA device"
    idx = _cuda_index(device)
    hit = _TRITON_STATUS.get(idx)
    if hit is not None:
        return hit
    with _LOCK:
        hit = _TRITON_STATUS.get(idx)
        if hit is not None:
            return hit
        res = _triton_status_uncached(idx)
        _TRITON_STATUS[idx] = res
        return res


def _triton_status_uncached(idx: int) -> tuple[bool, str]:
    v = _triton_version()
    if v is None:
        return False, "Triton is not installed"
    if _version_tuple(v)[:2] < MIN_TRITON:
        return False, f"Triton {v} is older than {MIN_TRITON[0]}.{MIN_TRITON[1]} (the minimum validated version)"
    info = device_info(idx)
    if info["capability"] < (8, 0):
        cc = "%d.%d" % info["capability"]
        return False, f"{info['name']}: compute capability {cc} < 8.0 (the kernels need sm_80 or newer)"
    if info["smem_class"] is None:
        return False, f"{info['name']}: opt-in shared memory {info['smem_optin']} B < 99 KiB per block"
    try:
        importlib.import_module("sisa.kernels.attention")
        importlib.import_module("sisa.kernels.fused_prep")
    except Exception as e:  # noqa: BLE001 - report any import/compile-time failure as a reason
        return False, f"importing the SISA Triton kernels failed: {type(e).__name__}: {e}"
    cc = "%d%d" % info["capability"]
    return True, (
        f"Triton {v} on {info['name']} (sm_{cc}, {info['sm_count']} SMs, {info['smem_optin']} B shared memory, "
        f"{info['smem_class']} launch table)"
    )


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    dt = x.device.type
    try:
        if torch.is_autocast_enabled(dt):
            return torch.get_autocast_dtype(dt)
    except (TypeError, RuntimeError):  # device types without autocast support
        pass
    return x.dtype


def fused_eligible(x: torch.Tensor) -> bool:
    """Whether the fused Triton layer path can run for layer input ``x`` (shape checks are done by select_impl)."""
    if not isinstance(x, torch.Tensor) or x.device.type != "cuda":
        return False
    if _compute_dtype(x) not in (torch.bfloat16, torch.float16):
        return False
    return triton_status(x.device)[0]


def _shape_problem(cfg) -> str | None:
    if cfg is None:
        return None
    d_model, n_head = int(cfg.d_model), int(cfg.n_head)
    d_head = getattr(cfg, "d_head", None)
    d_head = int(d_head) if d_head is not None else d_model // n_head
    d_state = int(cfg.d_state)
    if d_head != 64 or d_state != 32:
        return f"the Triton kernels need d_head == 64 and d_state == 32 (got d_head={d_head}, d_state={d_state})"
    return None


def select_impl(requested: str, x: torch.Tensor, cfg: SISAConfig | None) -> str:
    """Resolve ``requested`` (one of ``IMPLS``) to the implementation used for layer input ``x``.

    ``cfg`` is the model config (``d_model``, ``n_head``, ``d_state``; ``None`` skips the shape checks).
    ``"auto"`` returns ``"triton_fused"`` if eligible, else ``"triton"``, else ``"torch"``. A forced Triton
    implementation on an ineligible input raises ``ValueError`` with the reason.
    """
    if requested not in IMPLS:
        raise ValueError(f"attn_impl must be one of {IMPLS}, got {requested!r}")
    if requested == "torch":
        return "torch"
    on_cuda = isinstance(x, torch.Tensor) and x.device.type == "cuda"
    if on_cuda:
        ok, why = triton_status(x.device)
    else:
        ok, why = False, f"input is on {x.device.type if isinstance(x, torch.Tensor) else type(x).__name__}, not CUDA"
    shape = _shape_problem(cfg)
    cdt = _compute_dtype(x) if isinstance(x, torch.Tensor) else None
    tri_dtype_ok = cdt in (torch.bfloat16, torch.float16, torch.float32)
    fused_dtype_ok = cdt in (torch.bfloat16, torch.float16)

    if requested == "triton_fused":
        if not ok:
            raise ValueError(f"attn_impl='triton_fused' is not available: {why}")
        if shape:
            raise ValueError(f"attn_impl='triton_fused' is not available: {shape}")
        if not fused_dtype_ok:
            raise ValueError(
                f"attn_impl='triton_fused' needs a bf16/fp16 compute dtype (enable autocast or use low-precision "
                f"parameters); got {cdt}"
            )
        return "triton_fused"
    if requested == "triton":
        if not ok:
            raise ValueError(f"attn_impl='triton' is not available: {why}")
        if shape:
            raise ValueError(f"attn_impl='triton' is not available: {shape}")
        if not tri_dtype_ok:
            raise ValueError(f"attn_impl='triton' supports bf16, fp16 and fp32 compute dtypes; got {cdt}")
        return "triton"
    # auto
    if ok and not shape:
        if fused_dtype_ok:
            return "triton_fused"
        if tri_dtype_ok:
            return "triton"
    if on_cuda:
        reason = why if not ok else (shape or f"compute dtype {cdt}")
        key = (x.device.index, reason)
        if key not in _WARNED:
            _WARNED.add(key)
            warnings.warn(f"SISA: using the torch attention implementation on CUDA because {reason}", stacklevel=2)
    return "torch"


def describe() -> str:
    """Human-readable summary of the environment, the Triton status and the default backend choice."""
    lines = [f"torch {torch.__version__} (CUDA {torch.version.cuda})", f"triton {_triton_version() or 'not installed'}"]
    lines.append(f"check mode: {get_check_mode()}")
    if not torch.cuda.is_available():
        lines.append("CUDA: not available -> attn_impl='auto' uses 'torch'")
        return "\n".join(lines)
    for idx in range(torch.cuda.device_count()):
        info = device_info(idx)
        ok, why = triton_status(idx)
        cc = "%d.%d" % info["capability"]
        lines.append(
            f"cuda:{idx} {info['name']} (cc {cc}, {info['sm_count']} SMs, shared memory {info['smem_optin']} B "
            f"from {info['smem_source']}, class {info['smem_class']})"
        )
        lines.append(f"  triton kernels: {'available' if ok else 'unavailable'} - {why}")
        if ok:
            lines.append("  attn_impl='auto': 'triton_fused' under bf16/fp16 autocast, 'triton' for fp32 inputs")
        else:
            lines.append("  attn_impl='auto': 'torch'")
    return "\n".join(lines)
