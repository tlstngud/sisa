"""Small helpers shared by the tests."""
from __future__ import annotations

import torch

from sisa import SISAConfig, SISAForCausalLM


def tiny_config(**overrides) -> SISAConfig:
    """Two layers, two heads of 64 (kernel-compatible), d_state 32, vocabulary 257."""
    base = dict(vocab_size=257, d_model=128, n_layer=2, n_head=2, d_ff=256, d_state=32, attn_impl="torch")
    base.update(overrides)
    return SISAConfig(**base)


def tiny_model(seed: int = 0, randomize: bool = True, **overrides) -> SISAForCausalLM:
    torch.manual_seed(seed)
    model = SISAForCausalLM(tiny_config(**overrides))
    if randomize:
        randomize_(model, seed)
    return model


@torch.no_grad()
def randomize_(model: SISAForCausalLM, seed: int = 0) -> SISAForCausalLM:
    """Move the parameters away from the init so every term of the score matters.

    ``W_C`` becomes non-zero (SSM term on), rotation rates and lambda grow, decay and forget biases spread out.
    """
    gen = torch.Generator().manual_seed(seed + 1)

    def rnd(t, lo, hi):
        return torch.rand(t.shape, generator=gen) * (hi - lo) + lo

    for layer in model.layers:
        a = layer.attn
        a.W_C.weight.copy_(torch.randn(a.W_C.weight.shape, generator=gen) * 0.08)
        a.W_B.weight.copy_(torch.randn(a.W_B.weight.shape, generator=gen) * 0.08)
        a.theta_proj.weight.copy_(torch.randn(a.theta_proj.weight.shape, generator=gen) * 0.2)
        a.alpha_proj.bias.copy_(rnd(a.alpha_proj.bias, -4.0, -1.0))
        a.forget_proj.bias.copy_(rnd(a.forget_proj.bias, 0.0, 3.0))
        a.lambda_raw.copy_(rnd(a.lambda_raw, -0.5, 1.5))
        for norm in (layer.attn_norm, layer.ffn_norm):
            norm.weight.copy_(1.0 + 0.1 * torch.randn(norm.weight.shape, generator=gen))
    return model


def rel_err(x: torch.Tensor, ref: torch.Tensor) -> float:
    """``||x - ref|| / ||ref||`` in float64."""
    x, ref = x.detach().double(), ref.detach().double()
    return float((x - ref).norm() / ref.norm().clamp_min(1e-300))
