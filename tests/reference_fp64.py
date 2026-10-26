"""Independent dense float64 reference of the SISA language model.

Reads only the parameter values of a :class:`sisa.SISAForCausalLM` (by name) and recomputes logits, the loss and
all parameter gradients in float64 with plain torch ops: no model code, no kernels, no float32 casts (the model's
RMSNorm and forget gate hard-code ``.float()``, so ``model.double()`` would still run parts in float32).

Formula, per layer with ``n = RMSNorm(x)``::

    q, k, v = W_Q n, W_K n, W_V n;   c, b = W_C n, W_B n;   Phi = cumsum_t(theta_proj n)
    c~, b~  = interleaved-pair rotation of c, b by Phi
    g = cumsum_t(-softplus(alpha_proj n));   F = cumsum_t(logsigmoid(forget_proj n));   lam = softplus(lambda_raw)
    s_ij = q_i.k_j / sqrt(d_head) + lam * exp(g_i - g_j) * (c~_i . b~_j) + (F_i - F_j)   (j <= i)
    x += W_O softmax(s) v;   x += W_down(silu(W_gate m) * W_up m), m = RMSNorm(x)
    logits = RMSNorm_final(x) E^T;   loss = CE(logits[:, :-1], labels[:, 1:]), ignore_index=-100

Per-layer activation checkpointing keeps memory at about one layer of ``L x L`` float64 maps.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rot(z, phi):
    ze, zo = z[..., 0::2], z[..., 1::2]
    c, s = phi.cos(), phi.sin()
    return torch.stack([ze * c - zo * s, ze * s + zo * c], -1).flatten(-2)


def _layer(x, P, pre, H, dh, ds, eps, causal):
    B, L, _ = x.shape

    def lin(inp, name, bias=True):
        b = P.get(f"{pre}{name}.bias") if bias else None
        return F.linear(inp, P[f"{pre}{name}.weight"], b)

    n = _rms(x, P[f"{pre}attn_norm.weight"], eps)

    def heads(name, w):
        return lin(n, f"attn.{name}").reshape(B, L, H, w).transpose(1, 2)

    q, k, v = heads("W_Q", dh), heads("W_K", dh), heads("W_V", dh)
    c, b = heads("W_C", ds), heads("W_B", ds)
    phi = heads("theta_proj", ds // 2).cumsum(2)
    ct, bt = _rot(c, phi), _rot(b, phi)
    g = (-F.softplus(lin(n, "attn.alpha_proj"))).transpose(1, 2).cumsum(2)              # (B, H, L)
    delta = (g.unsqueeze(-1) - g.unsqueeze(-2)).masked_fill(~causal, 0.0)
    lam = F.softplus(P[f"{pre}attn.lambda_raw"]).reshape(1, H, 1, 1)
    s = q @ k.transpose(-1, -2) / math.sqrt(dh) + lam * delta.exp() * (ct @ bt.transpose(-1, -2))
    fg = F.logsigmoid(lin(n, "attn.forget_proj")).transpose(1, 2).cumsum(2)
    s = (s + (fg.unsqueeze(-1) - fg.unsqueeze(-2))).masked_fill(~causal, float("-inf"))
    y = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, L, H * dh)
    x = x + lin(y, "attn.W_O")
    m = _rms(x, P[f"{pre}ffn_norm.weight"], eps)
    h = F.linear(F.silu(F.linear(m, P[f"{pre}ffn.w_gate.weight"])) * F.linear(m, P[f"{pre}ffn.w_up.weight"]),
                 P[f"{pre}ffn.w_down.weight"])
    return x + h


def _params(model, dtype, requires_grad):
    return {n: p.detach().to(dtype).requires_grad_(requires_grad) for n, p in model.named_parameters()}


def _logits(model, P, ids):
    cfg = model.config
    L = ids.shape[1]
    causal = torch.ones(L, L, dtype=torch.bool, device=ids.device).tril()
    x = F.embedding(ids, P["embed.weight"])
    for i in range(cfg.n_layer):
        x = checkpoint(_layer, x, P, f"layers.{i}.", cfg.n_head, cfg.d_head, cfg.d_state, cfg.norm_eps, causal,
                       use_reentrant=False)
    xf = _rms(x, P["final_norm.weight"], cfg.norm_eps)
    return xf @ P["embed.weight"].t()


@torch.no_grad()
def logits_fp64(model, ids, dtype=torch.float64):
    """``(B, L, V)`` logits computed from the model's parameter values in ``dtype``."""
    return _logits(model, _params(model, dtype, False), ids)


def loss_and_grads(model, ids, labels, dtype=torch.float64):
    """-> ``(loss: float, {parameter name: gradient in dtype})`` with HF-style shifted labels."""
    names = [n for n, _ in model.named_parameters()]
    P = _params(model, dtype, True)
    logits = _logits(model, P, ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                           ignore_index=-100)
    grads = torch.autograd.grad(loss, [P[n] for n in names])
    return float(loss.detach()), {n: g.detach() for n, g in zip(names, grads)}
