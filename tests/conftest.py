"""Shared pytest configuration.

Markers:

* ``cuda``: needs an NVIDIA GPU (sm_80+) with working SISA Triton kernels; skipped otherwise.
* ``slow``: long-running cases, deselect with ``-m "not slow"``.

CUDA is never initialised unless a selected test carries the ``cuda`` marker. When it is, the process is limited
to a fraction of the device memory (``SISA_TEST_CUDA_MEM_FRACTION``, default 0.10) so the tests can share a GPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
try:
    import sisa  # noqa: F401
except ImportError:  # running from a source checkout without installation
    sys.path.insert(0, str(_SRC))

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:  # make the shared reference modules importable
    sys.path.insert(0, _HERE)

_CUDA_STATE: dict[str, object] = {}


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs an NVIDIA GPU with the SISA Triton kernels")
    config.addinivalue_line("markers", "slow: long-running test")


def _cuda_skip_reason() -> str | None:
    if "reason" not in _CUDA_STATE:
        reason = None
        if not torch.cuda.is_available():
            reason = "CUDA is not available"
        else:
            from sisa import backends

            ok, why = backends.triton_status()
            reason = None if ok else why
        _CUDA_STATE["reason"] = reason
    return _CUDA_STATE["reason"]  # type: ignore[return-value]


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # trylast: runs after -m/-k deselection, so CUDA is only probed when a cuda test is selected.
    cuda_items = [item for item in items if item.get_closest_marker("cuda") is not None]
    if not cuda_items:
        return
    reason = _cuda_skip_reason()
    if reason is not None:
        skip = pytest.mark.skip(reason=f"cuda test skipped: {reason}")
        for item in cuda_items:
            item.add_marker(skip)


def _limit_cuda_memory() -> None:
    if _CUDA_STATE.get("limited"):
        return
    # Opt-in cap for shared GPUs, e.g. SISA_TEST_CUDA_MEM_FRACTION=0.1. The suite needs about 2.5 GB.
    fraction = os.environ.get("SISA_TEST_CUDA_MEM_FRACTION")
    if fraction:
        torch.cuda.set_per_process_memory_fraction(float(fraction))
    _CUDA_STATE["limited"] = True


@pytest.fixture(autouse=True)
def _cuda_guard(request):
    if request.node.get_closest_marker("cuda") is not None and torch.cuda.is_available():
        _limit_cuda_memory()
    yield


@pytest.fixture
def cuda_device() -> torch.device:
    """The CUDA device used by ``cuda`` tests (memory already limited)."""
    _limit_cuda_memory()
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(autouse=True)
def _no_tf32():
    """Keep fp32 matmuls exact (TF32 off) for the duration of each test."""
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
