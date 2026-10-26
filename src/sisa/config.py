"""Model configuration and the paper presets."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["SISAConfig", "PRESETS", "ATTN_IMPLS"]

#: Accepted values of :attr:`SISAConfig.attn_impl`.
ATTN_IMPLS = ("auto", "torch", "triton", "triton_fused")

#: Head and state sizes the Triton kernels are compiled for.
TRITON_D_HEAD = 64
TRITON_D_STATE = 32


@dataclass
class SISAConfig:
    """Hyper-parameters of a SISA language model.

    The defaults describe the 50M paper model (``sisa-50m``). ``d_ff`` is the
    actual SwiGLU width; no parameter-matching rule is applied inside the
    package.

    Attributes:
        vocab_size: Embedding rows. The paper models use 50,277, which is
            ``len(tokenizer)`` of the GPT-NeoX-20B tokenizer (not its
            ``vocab_size`` attribute, 50,254).
        d_model: Residual width.
        n_layer: Number of blocks.
        n_head: Attention heads; ``d_head = d_model // n_head``. The Triton
            backends require ``d_head == 64``.
        d_ff: SwiGLU hidden width.
        d_state: Per-head SSM state size (even). The Triton backends require 32.
        norm_eps: RMSNorm epsilon.
        init_std: Std of the normal init of every linear weight and the
            embedding.
        theta_init_std: Std of the rotation-rate projection ``theta_proj``.
        decay_bias_init: Bias of ``alpha_proj``; ``softplus(-5) = 0.0067``
            nats per token, a decay half-life of about 100 tokens.
        forget_bias_init: Bias of ``forget_proj``; ``sigmoid(4) = 0.982``.
        lambda_raw_init: Initial ``lambda_raw``; ``softplus(-1) = 0.313``.
        zero_init_c: Initialise ``W_C`` to zero, so the SSM term of the score
            is exactly zero at init and the model starts as a forgetting
            transformer without positional encoding.
        max_seq_len: Training context. Informational only: there is no
            positional table and no hard length limit.
        attn_impl: ``"auto"``, ``"torch"``, ``"triton"`` or ``"triton_fused"``.
            See :mod:`sisa.backends`.
        torch_block_q: Query block size of the pure PyTorch attention.
        gradient_checkpointing: Recompute each block in the backward pass.
    """

    vocab_size: int = 50277
    d_model: int = 512
    n_layer: int = 6
    n_head: int = 8
    d_ff: int = 1829
    d_state: int = 32
    norm_eps: float = 1e-6
    init_std: float = 0.02
    theta_init_std: float = 0.01
    decay_bias_init: float = -5.0
    forget_bias_init: float = 4.0
    lambda_raw_init: float = -1.0
    zero_init_c: bool = True
    max_seq_len: int = 2048
    attn_impl: str = "auto"
    torch_block_q: int = 256
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_layer", "n_head", "d_ff", "d_state",
                     "max_seq_len", "torch_block_q"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})")
        if self.d_state % 2 != 0:
            raise ValueError(f"d_state must be even (channels are rotated in pairs), got {self.d_state}")
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")
        if self.attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {self.attn_impl!r}")
        if self.attn_impl in ("triton", "triton_fused"):
            if self.d_head != TRITON_D_HEAD or self.d_state != TRITON_D_STATE:
                raise ValueError(
                    f"attn_impl={self.attn_impl!r} requires d_head={TRITON_D_HEAD} and "
                    f"d_state={TRITON_D_STATE} (got d_head={self.d_head}, d_state={self.d_state}); "
                    "use attn_impl='torch' or 'auto' for other shapes")

    # ------------------------------------------------------------------ derived sizes
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_head

    def num_parameters(self) -> int:
        """Analytic parameter count; the tied LM head is counted once."""
        d, h, s, f = self.d_model, self.n_head, self.d_state, self.d_ff
        per_layer = (
            4 * d * d              # W_Q, W_K, W_V, W_O
            + 2 * h * s * d        # W_B, W_C
            + h * (s // 2) * d     # theta_proj
            + 2 * (h * d + h)      # alpha_proj, forget_proj (weight + bias)
            + h                    # lambda_raw
            + 2 * d                # attn_norm, ffn_norm
            + 3 * d * f            # SwiGLU
        )
        return self.n_layer * per_layer + self.vocab_size * d + d   # + embedding + final norm

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], **overrides: Any) -> "SISAConfig":
        """Build a config from a mapping. Unknown keys raise ``ValueError``."""
        merged = {**dict(data), **overrides}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(merged) - known)
        if unknown:
            raise ValueError(f"unknown SISAConfig fields: {unknown}")
        return cls(**merged)

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "SISAConfig":
        """Return the config of a paper model size, e.g. ``SISAConfig.preset("sisa-50m")``."""
        try:
            base = PRESETS[name]
        except KeyError:
            raise ValueError(f"unknown preset {name!r}; available: {sorted(PRESETS)}") from None
        return cls.from_dict(base, **overrides)

    def replace(self, **changes: Any) -> "SISAConfig":
        """Return a validated copy with some fields changed."""
        return dataclasses.replace(self, **changes)


#: Paper model sizes. All use d_head 64, d_state 32, vocabulary 50,277 and a
#: 2,048-token training context. Parameter counts (tied embedding counted
#: once): 50,911,376 / 151,878,576 / 369,278,080.
PRESETS: dict[str, dict[str, Any]] = {
    "sisa-50m": dict(d_model=512, n_layer=6, n_head=8, d_ff=1829),
    "sisa-152m": dict(d_model=768, n_layer=12, n_head=12, d_ff=2744),
    "sisa-369m": dict(d_model=1024, n_layer=24, n_head=16, d_ff=2507),
}
