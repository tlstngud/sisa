"""Incremental decoding with a per-layer cache, in pure PyTorch.

Besides ``k`` and ``v``, SISA needs, per head and cached token, the rotated
SSM write vector ``b~_j`` (``d_state`` values), the cumulative decay ``g_j``
and the cumulative forget value ``F_j``: ``d_state + 2`` extra numbers per
head and token (about +27% over ``k, v`` at ``d_head`` 64, ``d_state`` 32). The
running phase ``Phi`` is kept only for the last position.

A new token ``t`` extends the recurrences ``g_t = g_{t-1} - softplus(a_t)``,
``F_t = F_{t-1} + logsigmoid(z_t)`` and ``Phi_t = Phi_{t-1} + theta_t`` in
float32 and attends to the cache with the exact score of
:mod:`sisa.functional`. The prompt (prefill) goes through the regular
unfused attention backend; decode steps always use float32 PyTorch.

Limitations: decoding is not part of the paper and is not benchmarked.
Batched decoding assumes equal-length prompts (there is no padding mask).
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from .modeling import SISAForCausalLM

__all__ = ["LayerState", "SISACache", "generate"]


@dataclass
class LayerState:
    """Cached tensors of one layer.

    Attributes:
        k, v: ``(B, H, T, d_head)`` in the attention compute dtype.
        b: ``(B, H, T, d_state)`` rotated SSM write vectors, same dtype as ``k``.
        g: ``(B, H, T)`` float32 cumulative log-decay.
        fgate: ``(B, H, T)`` float32 cumulative log forget gate.
        phi: ``(B, H, d_state // 2)`` float32 phase at the last cached position.
    """

    k: Tensor
    v: Tensor
    b: Tensor
    g: Tensor
    fgate: Tensor
    phi: Tensor

    @property
    def length(self) -> int:
        return self.k.shape[2]


class SISACache:
    """Decode cache holding one :class:`LayerState` per layer."""

    def __init__(self) -> None:
        self._layers: dict[int, LayerState] = {}

    def get(self, layer_idx: int) -> LayerState | None:
        return self._layers.get(layer_idx)

    def update(self, layer_idx: int, *, k: Tensor, v: Tensor, b: Tensor, g: Tensor, fgate: Tensor,
               phi: Tensor) -> LayerState:
        """Append new positions to a layer's cache and return the updated state."""
        old = self._layers.get(layer_idx)
        if old is None:
            state = LayerState(k=k, v=v, b=b, g=g, fgate=fgate, phi=phi)
        else:
            state = LayerState(
                k=torch.cat([old.k, k], dim=2),
                v=torch.cat([old.v, v], dim=2),
                b=torch.cat([old.b, b], dim=2),
                g=torch.cat([old.g, g], dim=2),
                fgate=torch.cat([old.fgate, fgate], dim=2),
                phi=phi,
            )
        self._layers[layer_idx] = state
        return state

    @property
    def seq_len(self) -> int:
        """Number of cached positions (0 when empty)."""
        state = self._layers.get(0)
        return 0 if state is None else state.length

    def reset(self) -> None:
        self._layers.clear()

    def __len__(self) -> int:
        return len(self._layers)


def _eos_ids(eos_token_id: int | Iterable[int] | Tensor | None) -> list[int]:
    """Normalise ``eos_token_id`` (an integer, a 0-d/1-d tensor or array, or an iterable) to a list."""
    if eos_token_id is None:
        return []
    bad = TypeError(f"eos_token_id must be an integer or an iterable of integers, "
                    f"got {eos_token_id!r} ({type(eos_token_id).__name__})")
    if isinstance(eos_token_id, (bool, str, bytes)):
        raise bad
    if isinstance(eos_token_id, numbers.Integral):
        return [int(eos_token_id)]
    dtype = getattr(eos_token_id, "dtype", None)
    if dtype is not None:                                          # torch tensor or numpy array
        is_float = dtype.is_floating_point if torch.is_tensor(eos_token_id) else dtype.kind not in "iu"
        if is_float or getattr(eos_token_id, "ndim", 1) > 1:
            raise bad
        if getattr(eos_token_id, "ndim", None) == 0:
            return [int(eos_token_id)]
        return [int(t) for t in eos_token_id.tolist()]
    try:
        ids = list(eos_token_id)
    except TypeError:
        raise bad from None
    if not all(isinstance(t, numbers.Integral) and not isinstance(t, bool) for t in ids):
        raise bad
    return [int(t) for t in ids]


def _next_token(logits: Tensor, *, do_sample: bool, temperature: float, top_k: int | None,
                generator: torch.Generator | None) -> Tensor:
    logits = logits.float()
    if not do_sample:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(model: "SISAForCausalLM", input_ids: Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, eos_token_id: int | Iterable[int] | Tensor | None = None,
             do_sample: bool = False, *, generator: torch.Generator | None = None) -> Tensor:
    """Generate continuations of ``input_ids``.

    Args:
        model: A :class:`sisa.SISAForCausalLM`.
        input_ids: ``(B, L)`` or ``(L,)`` prompt token ids. All prompts in a
            batch must have the same length.
        max_new_tokens: Number of tokens to append.
        temperature: Softmax temperature (sampling only).
        top_k: Keep only the ``top_k`` most likely tokens (sampling only).
        eos_token_id: Token id (or ids) that ends a sequence. Finished rows are
            padded with the first id; generation stops when all rows finished.
        do_sample: Sample instead of greedy decoding.
        generator: Optional random generator for sampling.

    Returns:
        ``(B, L + n)`` token ids with ``n <= max_new_tokens`` (same rank as
        ``input_ids``).
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if do_sample and temperature <= 0:
        raise ValueError("temperature must be positive when do_sample=True")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    squeeze = input_ids.dim() == 1
    if squeeze:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.dim() != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be a non-empty (B, L) tensor")
    if max_new_tokens == 0:
        return input_ids.squeeze(0) if squeeze else input_ids

    eos = _eos_ids(eos_token_id)
    eos_t = torch.tensor(eos, device=input_ids.device, dtype=input_ids.dtype) if eos else None
    cache = SISACache()
    logits = model(input_ids, cache=cache, logits_to_keep=1).logits[:, -1]
    pieces = [input_ids]
    done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    for step in range(max_new_tokens):
        nxt = _next_token(logits, do_sample=do_sample, temperature=temperature, top_k=top_k,
                          generator=generator).to(input_ids.dtype)
        if eos_t is not None:
            nxt = torch.where(done, eos_t[0], nxt)
            done = done | torch.isin(nxt, eos_t)
        pieces.append(nxt.unsqueeze(1))
        if step + 1 == max_new_tokens or (eos_t is not None and bool(done.all())):
            break
        logits = model(nxt.unsqueeze(1), cache=cache, logits_to_keep=1).logits[:, -1]
    out = torch.cat(pieces, dim=1)
    return out.squeeze(0) if squeeze else out
