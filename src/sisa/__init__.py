"""SISA (SSM-Informed Softmax Attention): softmax attention with state-space signals in the score.

Each head scores query ``i`` against key ``j <= i`` with::

    s_ij = q_i . k_j / sqrt(d_head) + lam_h * exp(g_i - g_j) * (C~_i . B~_j) + (F_i - F_j)

i.e. content attention without positional encoding, plus a decaying,
rotating SSM read-out (``g`` cumulative log-decay, ``C~``/``B~`` rotated by a
data-dependent cumulative phase) and an additive forget gate ``F``.

Main entry points:

* :class:`SISAConfig`, :data:`PRESETS` - model sizes of the paper.
* :class:`SISAForCausalLM` - the language model (HF-style shifted labels).
* :func:`sisa_attention` - the attention function on prepared tensors.
* :func:`load_paper_checkpoint` / :func:`convert_state_dict` - weights of
  the original training code.
* :mod:`sisa.backends` - Triton availability and the runtime check mode.
"""

from __future__ import annotations

import importlib
from typing import Any

from .checkpoint import convert_state_dict, load_paper_checkpoint
from .config import PRESETS, SISAConfig
from .functional import sisa_attention
from .modeling import SISAAttention, SISABlock, SISAForCausalLM, SISAModel, SISAOutput

__version__ = "0.1.0"

__all__ = [
    "SISAConfig",
    "PRESETS",
    "SISAAttention",
    "SISABlock",
    "SISAModel",
    "SISAForCausalLM",
    "SISAOutput",
    "sisa_attention",
    "load_paper_checkpoint",
    "convert_state_dict",
    "set_check_mode",
    "__version__",
]


def set_check_mode(mode: str) -> None:
    """Set how the Triton kernels check their inputs: ``"async"`` (default), ``"sync"`` or ``"off"``.

    See :func:`sisa.backends.set_check_mode`.
    """
    from .backends import set_check_mode as _set_check_mode

    _set_check_mode(mode)


_LAZY_SUBMODULES = ("backends", "kernels", "generation")


def __getattr__(name: str) -> Any:
    # Submodules that may import Triton are loaded on first access only.
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
