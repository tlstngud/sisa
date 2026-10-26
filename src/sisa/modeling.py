"""SISA language model: pre-norm blocks of SISA attention and SwiGLU, tied embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .config import ATTN_IMPLS, SISAConfig
from .functional import (_AUTOCAST_DEVICE_TYPES, _attend_rows, _autocast_off, _work_dtype, rotate_pairs,
                         sisa_attention)

if TYPE_CHECKING:  # pragma: no cover
    from .generation import SISACache

__all__ = ["RMSNorm", "SwiGLU", "SISAAttention", "SISABlock", "SISAModel", "SISAForCausalLM", "SISAOutput"]

IGNORE_INDEX = -100


class RMSNorm(nn.Module):
    """RMSNorm computed in float32; the output has the input's dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return ((x.float() * norm) * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    """``w_down(silu(w_gate x) * w_up x)``, no biases."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def _compute_dtype(x: Tensor) -> torch.dtype:
    """The autocast dtype when autocast is on for ``x``'s device, else ``x.dtype``."""
    device_type = x.device.type
    if device_type in _AUTOCAST_DEVICE_TYPES and torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return x.dtype


def _resolve_impl(requested: str, x: Tensor, config: SISAConfig) -> str:
    """Map a requested attention implementation to a concrete one for input ``x``."""
    if requested not in ATTN_IMPLS:
        raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {requested!r}")
    if requested == "torch":
        return "torch"
    if x.device.type != "cuda":
        if requested == "auto":
            return "torch"
        raise ValueError(f"attn_impl={requested!r} needs a CUDA tensor, got one on {x.device.type}; "
                         "use attn_impl='torch' or 'auto'")
    from . import backends

    impl = backends.select_impl(requested, x, config)
    if impl not in ("torch", "triton", "triton_fused"):
        raise RuntimeError(f"backend selection returned an unknown implementation {impl!r}")
    return impl


class SISAAttention(nn.Module):
    """SISA attention sublayer: projections, recurrences, attention and output projection.

    Parameters (row layouts follow the paper checkpoints): ``W_Q, W_K, W_V,
    W_O`` (``d -> d``); ``W_B, W_C`` (``d -> H*d_state``, row ``h*d_state + s``);
    ``alpha_proj`` (``d -> H``, decay); ``theta_proj`` (``d -> H*d_state/2``,
    rotation rate, row ``h*(d_state/2) + p``); ``forget_proj`` (``d -> H``,
    forget gate); ``lambda_raw`` (``H``, SSM term weight
    ``lam = softplus(lambda_raw)``). ``lambda_raw`` stays a float32 parameter
    with its full-precision value when the module is cast to bf16/fp16
    (``.to``, ``.half()``, ``.bfloat16()``), because bf16 would freeze it during
    training.

    The constructor applies the SISA initialisation (see
    :meth:`reset_parameters`), so the sublayer can be used on its own.

    Computation, with ``x`` the normalised input::

        q, k, v = W_Q x, W_K x, W_V x                     no positional encoding
        g   = cumsum(-softplus(alpha_proj x))             float32
        Phi = cumsum(theta_proj x)                        float32
        c~, b~ = rotate_pairs(W_C x, Phi), rotate_pairs(W_B x, Phi)
        F   = cumsum(logsigmoid(forget_proj x))           float32, autocast off
        y   = W_O sisa_attention(q, k, v, c~, b~, g, lam, F)
    """

    def __init__(self, config: SISAConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        d, h, s = config.d_model, config.n_head, config.d_state
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)
        self.W_B = nn.Linear(d, h * s, bias=False)
        self.W_C = nn.Linear(d, h * s, bias=False)
        self.alpha_proj = nn.Linear(d, h, bias=True)
        self.theta_proj = nn.Linear(d, h * (s // 2), bias=False)
        self.forget_proj = nn.Linear(d, h, bias=True)
        self.lambda_raw = nn.Parameter(torch.full((h,), float(config.lambda_raw_init)))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Apply the SISA initialisation to this sublayer.

        Every weight ~ N(0, ``init_std``) except ``W_C.weight = 0`` if
        ``zero_init_c`` and ``theta_proj.weight ~ N(0, theta_init_std)``;
        ``alpha_proj.bias = decay_bias_init``; ``forget_proj.bias =
        forget_bias_init``; ``lambda_raw = lambda_raw_init``.
        """
        cfg = self.config
        for lin in (self.W_Q, self.W_K, self.W_V, self.W_O, self.W_B, self.alpha_proj, self.forget_proj):
            nn.init.normal_(lin.weight, std=cfg.init_std)
        if cfg.zero_init_c:
            nn.init.zeros_(self.W_C.weight)
        else:
            nn.init.normal_(self.W_C.weight, std=cfg.init_std)
        nn.init.normal_(self.theta_proj.weight, std=cfg.theta_init_std)
        nn.init.constant_(self.alpha_proj.bias, cfg.decay_bias_init)
        nn.init.constant_(self.forget_proj.bias, cfg.forget_bias_init)
        self.lambda_raw.fill_(cfg.lambda_raw_init)

    # ------------------------------------------------------------------ helpers
    def _apply(self, fn: Any, recurse: bool = True) -> "SISAAttention":
        # Keep lambda_raw in float32: when ``fn`` casts it to fp16/bf16, restore the
        # original full-precision value (and gradient) on the new device instead.
        low = (torch.float16, torch.bfloat16)
        old = self.lambda_raw
        # ``.data`` aliases keep the original storage even after _apply swaps the tensors' data in place.
        old_data = old.data
        old_grad = None if old.grad is None else old.grad.data
        super()._apply(fn, recurse)
        p = self.lambda_raw
        if p.dtype in low:
            src = old_data if old_data.dtype not in low else p.data
            p.data = src.to(device=p.device, dtype=torch.float32)
            if p.grad is not None:
                gsrc = old_grad if old_grad is not None and old_grad.dtype not in low else p.grad
                p.grad = gsrc.to(device=p.device, dtype=torch.float32)
        return self

    def lam(self) -> Tensor:
        """Per-head SSM weight ``softplus(lambda_raw)`` as float32, shape ``(H,)``."""
        return F.softplus(self.lambda_raw).float()

    def _forget_logits(self, x: Tensor) -> Tensor:
        """Forget-gate pre-activations ``(B, L, H)`` in float32 with autocast disabled."""
        with _autocast_off(x.device.type):
            return F.linear(x.float(), self.forget_proj.weight.float(), self.forget_proj.bias.float())

    def _prepare(self, x: Tensor, state: Any = None):
        """Projections and recurrences for the unfused path.

        With a decode ``state`` the recurrences continue from its last position.
        Returns ``q, k, v, c~, b~`` in the compute dtype and ``g, F, Phi`` in
        float32.
        """
        bsz, seq_len, _ = x.shape
        h, dh, s = self.config.n_head, self.config.d_head, self.config.d_state
        q = self.W_Q(x).view(bsz, seq_len, h, dh).transpose(1, 2)
        k = self.W_K(x).view(bsz, seq_len, h, dh).transpose(1, 2)
        v = self.W_V(x).view(bsz, seq_len, h, dh).transpose(1, 2)
        b = self.W_B(x).view(bsz, seq_len, h, s).transpose(1, 2)
        c = self.W_C(x).view(bsz, seq_len, h, s).transpose(1, 2)
        a = self.alpha_proj(x).transpose(1, 2)                                   # (B, H, L)
        theta = self.theta_proj(x).view(bsz, seq_len, h, s // 2).transpose(1, 2)
        g = torch.cumsum(-F.softplus(a.float()), dim=2)
        phi = torch.cumsum(theta.float(), dim=2)
        z = self._forget_logits(x)
        with _autocast_off(x.device.type):
            fgate = torch.cumsum(F.logsigmoid(z).transpose(1, 2), dim=2)       # (B, H, L)
        if state is not None:
            g = g + state.g[..., -1:]
            phi = phi + state.phi.unsqueeze(2)
            fgate = fgate + state.fgate[..., -1:]
        c = rotate_pairs(c.float(), phi)
        b = rotate_pairs(b.float(), phi)
        cd = _compute_dtype(x)
        return q.to(cd), k.to(cd), v.to(cd), c.to(cd), b.to(cd), g, fgate, phi

    def _attend(self, q, k, v, c, b, g, fgate, impl: str) -> Tensor:
        if impl == "triton":
            q, k, v, c, b = (t.contiguous() for t in (q, k, v, c, b))
            g, fgate = g.contiguous(), fgate.contiguous()
        return sisa_attention(q, k, v, c, b, g, self.lam(), fgate, impl=impl,
                              block_q=self.config.torch_block_q)

    def _output(self, y: Tensor) -> Tensor:
        bsz, h, seq_len, dh = y.shape
        return self.W_O(y.transpose(1, 2).reshape(bsz, seq_len, h * dh))

    # ------------------------------------------------------------------ paths
    def _forward_fused(self, x: Tensor) -> Tensor:
        from .kernels.fused_prep import concat_fused_weights, sisa_fused

        bsz, seq_len, d = x.shape
        weight, bias = concat_fused_weights(self.W_Q.weight, self.W_K.weight, self.W_V.weight, self.W_B.weight,
                                            self.W_C.weight, self.theta_proj.weight, self.alpha_proj.weight,
                                            self.alpha_proj.bias)
        proj = F.linear(x, weight, bias)                  # (B, L, NPAD), autocast dtype
        y = sisa_fused(proj, self._forget_logits(x), self.lam(), self.config.n_head)
        return self.W_O(y.reshape(bsz, seq_len, d))

    def _forward_cached(self, x: Tensor, cache: "SISACache", requested: str) -> Tensor:
        state = cache.get(self.layer_idx)
        q, k, v, c, b, g, fgate, phi = self._prepare(x, state)
        if state is None:
            # Prefill: regular unfused attention; the fused path does not expose k, v, b~, g, F.
            impl = _resolve_impl("triton" if requested == "triton_fused" else requested, x, self.config)
            y = self._attend(q, k, v, c, b, g, fgate, "triton" if impl == "triton_fused" else impl)
            cache.update(self.layer_idx, k=k, v=v, b=b, g=g, fgate=fgate, phi=phi[:, :, -1])
            return self._output(y)
        start = state.length
        state = cache.update(self.layer_idx, k=k, v=v, b=b, g=g, fgate=fgate, phi=phi[:, :, -1])
        wd = _work_dtype(q.dtype)
        y = _attend_rows(q.to(wd), state.k.to(wd), state.v.to(wd), c.to(wd), state.b.to(wd),
                         g.to(wd), state.g.to(wd), self.lam().to(wd), fgate.to(wd), state.fgate.to(wd), start)
        return self._output(y.to(q.dtype))

    def forward(self, x: Tensor, *, cache: "SISACache | None" = None, impl: str | None = None) -> Tensor:
        """Apply the sublayer to the normalised input ``x`` ``(B, L, d_model)``; returns ``(B, L, d_model)``.

        Args:
            cache: Decode cache (see :mod:`sisa.generation`). An empty cache is
                filled by this call (prefill); a filled one is extended.
            impl: Override of ``config.attn_impl`` for this call.
        """
        requested = impl or self.config.attn_impl
        if cache is not None:
            return self._forward_cached(x, cache, requested)
        impl = _resolve_impl(requested, x, self.config)
        if impl == "triton_fused":
            return self._forward_fused(x)
        q, k, v, c, b, g, fgate, _ = self._prepare(x)
        return self._output(self._attend(q, k, v, c, b, g, fgate, impl))


class SISABlock(nn.Module):
    """``x + attn(attn_norm(x))`` followed by ``x + ffn(ffn_norm(x))``.

    The constructor applies the SISA initialisation (see
    :meth:`reset_parameters`), so the block can be used on its own.
    """

    def __init__(self, config: SISAConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = SISAAttention(config, layer_idx)          # initialised by its constructor
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)
        self._reset_ffn_and_norms()

    @torch.no_grad()
    def _reset_ffn_and_norms(self) -> None:
        for lin in (self.ffn.w_gate, self.ffn.w_up, self.ffn.w_down):
            nn.init.normal_(lin.weight, std=self.config.init_std)
        nn.init.ones_(self.attn_norm.weight)
        nn.init.ones_(self.ffn_norm.weight)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Re-initialise the block: MLP weights ~ N(0, ``init_std``), norms 1, and
        :meth:`SISAAttention.reset_parameters`."""
        self._reset_ffn_and_norms()
        self.attn.reset_parameters()

    def forward(self, x: Tensor, cache: "SISACache | None" = None) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cache=cache)
        return x + self.ffn(self.ffn_norm(x))


@dataclass
class SISAOutput:
    """Output of :meth:`SISAForCausalLM.forward`."""

    loss: Tensor | None = None
    logits: Tensor | None = None


def _lm_head_loss(hidden: Tensor, weight: Tensor, labels: Tensor) -> Tensor:
    """Cross-entropy of the tied LM head; ``labels`` are already aligned with ``hidden``."""
    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1),
                           ignore_index=IGNORE_INDEX)


class SISAModel(nn.Module):
    """Embedding, ``n_layer`` :class:`SISABlock` s and the final RMSNorm.

    ``forward(input_ids)`` returns the final-normalised hidden states
    ``(B, L, d_model)``.
    """

    def __init__(self, config: SISAConfig):
        super().__init__()
        self.config = config.replace()          # private, validated copy shared by all submodules
        cfg = self.config
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([SISABlock(cfg, i) for i in range(cfg.n_layer)])   # initialised on construction
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self._reset_embedding_and_norm()

    @torch.no_grad()
    def _reset_embedding_and_norm(self) -> None:
        nn.init.normal_(self.embed.weight, std=self.config.init_std)
        nn.init.ones_(self.final_norm.weight)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Re-initialise every parameter as at construction.

        The embedding (the tied LM head) and every linear weight ~ N(0,
        ``init_std``), norms 1, and per attention sublayer
        ``alpha_proj.bias = decay_bias_init``, ``theta_proj.weight ~ N(0,
        theta_init_std)``, ``forget_proj.bias = forget_bias_init``,
        ``lambda_raw = lambda_raw_init`` and ``W_C.weight = 0`` if
        ``zero_init_c`` (see :meth:`SISABlock.reset_parameters`).
        """
        self._reset_embedding_and_norm()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, input_ids: Tensor, *, cache: "SISACache | None" = None) -> Tensor:
        x = self.embed(input_ids)
        use_ckpt = (self.config.gradient_checkpointing and self.training and cache is None
                    and torch.is_grad_enabled())
        for layer in self.layers:
            if use_ckpt:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x, cache=cache)
        return self.final_norm(x)

    # ------------------------------------------------------------------ utilities
    def set_attn_impl(self, impl: str) -> None:
        """Switch the attention implementation (``auto``/``torch``/``triton``/``triton_fused``)."""
        self.config.replace(attn_impl=impl)     # validates
        self.config.attn_impl = impl

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Number of parameters; the tied LM head is counted once."""
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if exclude_embeddings else n

    def param_groups(self, lr: float, weight_decay: float = 0.1, lambda_lr: float | None = None,
                     no_decay_timescale_biases: bool = True) -> list[dict[str, Any]]:
        """Optimizer parameter groups.

        1. ``"default"``: every other parameter (norms and the embedding
           included) with ``weight_decay``.
        2. ``"timescale_biases"``: ``alpha_proj.bias`` and ``forget_proj.bias``
           without weight decay (only when ``no_decay_timescale_biases``;
           otherwise they stay in group 1, as in the 50M paper runs).
        3. ``"lambda"``: ``lambda_raw`` without weight decay at ``lambda_lr``
           (defaults to ``lr``; the paper models were trained with equal rates).
        """
        default, timescale, lam = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("lambda_raw"):
                lam.append(p)
            elif no_decay_timescale_biases and name.endswith(("alpha_proj.bias", "forget_proj.bias")):
                timescale.append(p)
            else:
                default.append(p)
        groups = [{"params": default, "lr": lr, "weight_decay": weight_decay, "name": "default"}]
        if timescale:
            groups.append({"params": timescale, "lr": lr, "weight_decay": 0.0, "name": "timescale_biases"})
        if lam:
            groups.append({"params": lam, "lr": lr if lambda_lr is None else lambda_lr, "weight_decay": 0.0,
                           "name": "lambda"})
        return groups

    def compile_common(self, **compile_kwargs: Any) -> "SISAModel":
        """``torch.compile`` the norms, the SwiGLU blocks and the logits-free loss in place.

        The compiled loss (LM head plus cross-entropy) is used only by
        ``forward(..., labels=..., return_logits=False)``; the default call
        with ``return_logits=True`` computes its loss eagerly from the
        returned logits. Uses :meth:`torch.nn.Module.compile`, so
        ``state_dict`` keys are unchanged. The attention sublayers are left
        eager (the Triton autograd functions would break the graph);
        whole-model compilation is not supported.
        """
        for layer in self.layers:
            layer.attn_norm.compile(**compile_kwargs)
            layer.ffn_norm.compile(**compile_kwargs)
            layer.ffn.compile(**compile_kwargs)
        self.final_norm.compile(**compile_kwargs)
        if hasattr(self, "_loss_fn"):
            self._loss_fn = torch.compile(_lm_head_loss, **compile_kwargs)
        return self


class SISAForCausalLM(SISAModel):
    """:class:`SISAModel` with the tied LM head ``logits = final_norm(h) @ embed.weight.T``.

    Labels follow the Hugging Face convention: ``labels`` has the shape of
    ``input_ids`` and the model shifts internally,
    ``loss = CE(logits[:, :-1], labels[:, 1:])`` with ``ignore_index=-100``,
    averaged over non-ignored targets. To train on blocks of ``N + 1`` tokens
    as in the paper (``N`` = 2048 targets), pass the whole block as both
    ``input_ids`` and ``labels``.
    """

    def __init__(self, config: SISAConfig):
        super().__init__(config)
        self._loss_fn = _lm_head_loss

    def forward(self, input_ids: Tensor, labels: Tensor | None = None, *, cache: "SISACache | None" = None,
                fused_ce: bool = False, return_logits: bool = True, logits_to_keep: int = 0) -> SISAOutput:
        """
        Args:
            input_ids: ``(B, L)`` token ids.
            labels: ``(B, L)`` targets (shifted internally), ``-100`` ignored.
            cache: Decode cache, see :func:`sisa.generation.generate`.
            fused_ce: Compute the loss with the chunked Triton LM-head
                cross-entropy (CUDA); logits are then not returned. Without
                ``labels`` it has no effect.
            return_logits: Return the logits. With ``False`` the loss is
                computed without keeping them (this saves memory mainly
                after :meth:`compile_common`; ``fused_ce`` saves the most).
            logits_to_keep: If positive, return logits for the last
                ``logits_to_keep`` positions only (not with ``labels``).
        """
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError(f"labels shape {tuple(labels.shape)} must match input_ids {tuple(input_ids.shape)}")
        if labels is not None and logits_to_keep:
            raise ValueError("logits_to_keep cannot be combined with labels")
        hidden = super().forward(input_ids, cache=cache)
        weight = self.embed.weight
        loss = logits = None
        fused_ce = fused_ce and labels is not None              # only the loss is fused
        if fused_ce:
            if not hidden.is_cuda:
                raise ValueError("fused_ce=True needs CUDA tensors")
            from .kernels.linear_ce import linear_cross_entropy

            d = hidden.shape[-1]
            loss = linear_cross_entropy(hidden[:, :-1].reshape(-1, d), weight, labels[:, 1:].reshape(-1),
                                        ignore_index=IGNORE_INDEX)
        elif labels is not None and not return_logits:
            loss = self._loss_fn(hidden[:, :-1], weight, labels[:, 1:])
        if return_logits and not fused_ce:
            logits = F.linear(hidden[:, -logits_to_keep:] if logits_to_keep else hidden, weight)
            if labels is not None:
                vocab = logits.shape[-1]
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab).float(), labels[:, 1:].reshape(-1),
                                       ignore_index=IGNORE_INDEX)
        return SISAOutput(loss=loss, logits=logits)

    def generate(self, input_ids: Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, eos_token_id: int | Iterable[int] | Tensor | None = None,
                 do_sample: bool = False, **kwargs: Any) -> Tensor:
        """Greedy or sampled decoding with a cache; see :func:`sisa.generation.generate`."""
        from .generation import generate

        return generate(self, input_ids, max_new_tokens, temperature=temperature, top_k=top_k,
                        eos_token_id=eos_token_id, do_sample=do_sample, **kwargs)

    # ------------------------------------------------------------------ persistence
    def save_pretrained(self, path: str, *, safe_serialization: bool = True) -> None:
        """Write ``config.json`` and ``model.safetensors`` (or ``pytorch_model.bin``) to ``path``."""
        from .checkpoint import save_pretrained

        save_pretrained(self, path, safe_serialization=safe_serialization)

    @classmethod
    def from_pretrained(cls, path: str, *, dtype: torch.dtype | None = None, **config_overrides: Any
                        ) -> "SISAForCausalLM":
        """Load a directory written by :meth:`save_pretrained`."""
        from .checkpoint import from_pretrained

        return from_pretrained(path, dtype=dtype, **config_overrides)
