"""Checkpoint conversion, paper-checkpoint loading and ``save_pretrained`` / ``from_pretrained``.

Paper checkpoints are ``torch.save`` dictionaries whose ``"model"`` entry is
a state dict with flat attention keys (``layers.{i}.W_Q.weight``) and, for
modules that were compiled during training, an ``._orig_mod.`` infix
(``layers.{i}.ffn._orig_mod.w_gate.weight``). :func:`convert_state_dict`
maps them to this package's names (``layers.{i}.attn.W_Q.weight``,
``layers.{i}.ffn.w_gate.weight``). Only the weights are read: optimizer and
scheduler state and the training configuration stored next to them are
discarded and never copied into the model.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch import Tensor

from .config import SISAConfig

if TYPE_CHECKING:  # pragma: no cover
    from .modeling import SISAForCausalLM

__all__ = [
    "convert_state_dict",
    "infer_config_from_state_dict",
    "load_paper_checkpoint",
    "save_pretrained",
    "from_pretrained",
]

CONFIG_NAME = "config.json"
SAFETENSORS_NAME = "model.safetensors"
TORCH_WEIGHTS_NAME = "pytorch_model.bin"

_ATTN_KEY = re.compile(r"^layers\.(\d+)\.(W_[QKVOBC]|alpha_proj|theta_proj|forget_proj|lambda_raw)(?=\.|$)")
_LAYER_KEY = re.compile(r"^layers\.(\d+)\.")
_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def _strip_wrappers(key: str) -> str:
    """Remove DDP (``module.``) and ``torch.compile`` (``_orig_mod.``) wrapper names."""
    key = key.replace("._orig_mod.", ".")
    stripped = True
    while stripped:
        stripped = False
        for prefix in _WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                stripped = True
    return key


def convert_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Map a paper state dict to this package's parameter names.

    Idempotent: keys already in the package layout are left unchanged.

    * ``module.`` / ``_orig_mod.`` prefixes and ``._orig_mod.`` infixes are
      removed (``layers.{i}.ffn._orig_mod.w_gate.weight`` becomes
      ``layers.{i}.ffn.w_gate.weight``; same for ``attn_norm``, ``ffn_norm``
      and ``final_norm``).
    * Attention parameters move under ``attn.``:
      ``layers.{i}.{W_Q,W_K,W_V,W_O,W_B,W_C,alpha_proj,theta_proj,forget_proj,lambda_raw}...``
      becomes ``layers.{i}.attn.{...}...``.
    * ``embed.weight`` (also the tied LM head) is unchanged.
    """
    out: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = _ATTN_KEY.sub(r"layers.\1.attn.\2", _strip_wrappers(key))
        if new_key in out:
            raise ValueError(f"two checkpoint keys map to {new_key!r}")
        out[new_key] = value
    return out


def infer_config_from_state_dict(state_dict: Mapping[str, Tensor], **overrides: Any) -> SISAConfig:
    """Read the model shape from tensor shapes (never from stored training configs).

    ``d_ff`` comes from ``ffn.w_gate.weight``: a stored training config may
    hold a different, nominal width.
    """
    sd = convert_state_dict(state_dict)
    try:
        vocab_size, d_model = sd["embed.weight"].shape
        n_head = sd["layers.0.attn.lambda_raw"].numel()
        b_rows = sd["layers.0.attn.W_B.weight"].shape[0]
        d_ff = sd["layers.0.ffn.w_gate.weight"].shape[0]
    except KeyError as err:
        raise ValueError(f"not a SISA state dict: missing {err.args[0]!r}") from None
    layer_ids = sorted({int(m.group(1)) for k in sd if (m := _LAYER_KEY.match(k))})
    if layer_ids != list(range(len(layer_ids))):
        raise ValueError(f"layer indices are not contiguous: {layer_ids}")
    if b_rows % n_head != 0:
        raise ValueError(f"W_B has {b_rows} rows, not a multiple of n_head={n_head}")
    shape = dict(vocab_size=int(vocab_size), d_model=int(d_model), n_layer=len(layer_ids), n_head=int(n_head),
                 d_ff=int(d_ff), d_state=b_rows // n_head)
    clash = sorted(set(shape) & set(overrides))
    if clash:
        raise ValueError(f"shape fields {clash} are read from the weights and cannot be overridden")
    return SISAConfig(**shape, **overrides)


def _empty_model(config: SISAConfig) -> "SISAForCausalLM":
    """Build a model without running the random init (weights are loaded next)."""
    from .modeling import SISAForCausalLM

    with torch.device("meta"):
        model = SISAForCausalLM(config)
    return model.to_empty(device="cpu")


def _load_strict(model: torch.nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    if missing or unexpected:
        raise ValueError(f"state dict does not match the model: {len(missing)} missing {missing[:8]}, "
                         f"{len(unexpected)} unexpected {unexpected[:8]}")
    bad = [(k, tuple(state_dict[k].shape), tuple(v.shape)) for k, v in expected.items()
           if tuple(state_dict[k].shape) != tuple(v.shape)]
    if bad:
        raise ValueError(f"shape mismatch (key, checkpoint, model): {bad[:8]}")
    model.load_state_dict(state_dict, strict=True)


def _extract_model_state(obj: Any) -> dict[str, Tensor]:
    if isinstance(obj, Mapping):
        for key in ("model", "state_dict"):
            inner = obj.get(key)
            if isinstance(inner, Mapping):
                return dict(inner)
        if obj and all(isinstance(v, Tensor) for v in obj.values()):
            return dict(obj)
    raise ValueError("checkpoint holds neither a 'model' state dict nor a plain state dict")


def _torch_load(path: Path, weights_only: bool) -> Any:
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=weights_only)
    except pickle.UnpicklingError as err:
        raise pickle.UnpicklingError(
            f"{err}\nThe checkpoint holds objects that torch.load(weights_only=True) refuses. If you trust "
            "the file, pass trust_pickle=True.") from None
    except RuntimeError as err:
        if "mmap" not in str(err):
            raise
        return torch.load(path, map_location="cpu", weights_only=weights_only)   # legacy (non-zip) format


def load_paper_checkpoint(path: str | Path, *, trust_pickle: bool = False, dtype: torch.dtype | None = None,
                          **config_overrides: Any) -> "SISAForCausalLM":
    """Load a training checkpoint of the paper code into a :class:`sisa.SISAForCausalLM`.

    The file is memory-mapped and read with ``weights_only=True`` unless
    ``trust_pickle=True``. Only ``checkpoint["model"]`` is used; the
    architecture is inferred from tensor shapes; the key set must match
    exactly and the load is strict.

    Args:
        path: ``.pt`` file written by the training code.
        trust_pickle: Allow arbitrary pickled objects (only for trusted files).
        dtype: Optional dtype to cast the model to after loading.
        **config_overrides: Non-shape :class:`sisa.SISAConfig` fields, e.g.
            ``attn_impl="torch"``.

    Returns:
        The model on CPU in eval mode (float32 unless ``dtype`` is given).
    """
    obj = _torch_load(Path(path), weights_only=not trust_pickle)
    state_dict = convert_state_dict(_extract_model_state(obj))
    del obj                                                       # drop optimizer state and stored configs
    config = infer_config_from_state_dict(state_dict, **config_overrides)
    model = _empty_model(config)
    _load_strict(model, state_dict)
    if dtype is not None:
        model.to(dtype)
    return model.eval()


def save_pretrained(model: torch.nn.Module, path: str | Path, *, safe_serialization: bool = True) -> None:
    """Write ``config.json`` and the weights to directory ``path``.

    Weights go to ``model.safetensors`` when ``safe_serialization`` and the
    ``safetensors`` package is installed, else to ``pytorch_model.bin``. A
    weights file of the other format left in ``path`` is removed so that
    :func:`from_pretrained` cannot pick up stale weights.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / CONFIG_NAME, "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, indent=2)
        f.write("\n")
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if safe_serialization:
        try:
            from safetensors.torch import save_file
        except ImportError:
            safe_serialization = False
    if safe_serialization:
        save_file(state_dict, str(path / SAFETENSORS_NAME), metadata={"format": "pt"})
        stale = path / TORCH_WEIGHTS_NAME
    else:
        torch.save(state_dict, path / TORCH_WEIGHTS_NAME)
        stale = path / SAFETENSORS_NAME
    if stale.exists():
        stale.unlink()


def from_pretrained(path: str | Path, *, dtype: torch.dtype | None = None,
                    **config_overrides: Any) -> "SISAForCausalLM":
    """Load a directory written by :func:`save_pretrained`.

    Args:
        path: Directory with ``config.json`` and ``model.safetensors`` or
            ``pytorch_model.bin``.
        dtype: Optional dtype to cast the model to after loading.
        **config_overrides: Fields of the stored config to replace, e.g.
            ``attn_impl="torch"``.

    Returns:
        The model on CPU in eval mode.
    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory written by save_pretrained "
                         "(use load_paper_checkpoint for training checkpoints)")
    with open(path / CONFIG_NAME, encoding="utf-8") as f:
        config = SISAConfig.from_dict(json.load(f), **config_overrides)
    if (path / SAFETENSORS_NAME).exists():
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(f"{path / SAFETENSORS_NAME} needs the 'safetensors' package "
                              "(pip install safetensors)") from None
        state_dict = load_file(str(path / SAFETENSORS_NAME))
    elif (path / TORCH_WEIGHTS_NAME).exists():
        state_dict = torch.load(path / TORCH_WEIGHTS_NAME, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"no {SAFETENSORS_NAME} or {TORCH_WEIGHTS_NAME} in {path}")
    model = _empty_model(config)
    _load_strict(model, convert_state_dict(state_dict))
    if dtype is not None:
        model.to(dtype)
    return model.eval()
