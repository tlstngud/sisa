"""The float32 torch implementation against the independent float64 dense reference.

Gradient errors are judged relative to the error of the same dense formula evaluated in float32
(``reference_fp64`` with ``dtype=torch.float32``): cancellations in the softmax backward and the float32
cumulative sums put the float32 floor at 1e-5 to 2e-4 for some parameters, so a fixed 1e-5 bound would test the
float format, not the implementation. A single float32 run is only one sample of that rounding noise (summing the
projection gradients in another order moves it by several times), hence the additive slack.
"""
import pytest
import torch

import reference_fp64
from model_utils import rel_err, tiny_model


def _ids(bsz, seq_len, seed=0):
    return torch.randint(0, 257, (bsz, seq_len), generator=torch.Generator().manual_seed(seed))


def _check_grads(model, ids, loss_rtol=1e-6):
    loss64, grads64 = reference_fp64.loss_and_grads(model, ids, ids)
    _, grads32 = reference_fp64.loss_and_grads(model, ids, ids, dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    loss = model(ids, labels=ids).loss
    loss.backward()
    assert abs(loss.item() - loss64) / abs(loss64) < loss_rtol
    for name, p in model.named_parameters():
        assert grads64[name].norm() > 0, name                   # every term of the score is active
        floor = rel_err(grads32[name], grads64[name])
        err = rel_err(p.grad, grads64[name])
        assert err <= 2.0 * floor + 2e-5, (name, err, floor)
        assert err < 1e-3, (name, err)


@pytest.mark.parametrize("seq_len", [3, 7, 64, 129, 300])
def test_loss_and_grads_match_fp64(seq_len):
    _check_grads(tiny_model(seed=seq_len), _ids(2, seq_len, seed=seq_len))


@pytest.mark.parametrize("seq_len,block_q", [(1, 256), (129, 64), (300, 37)])
def test_logits_match_fp64(seq_len, block_q):
    model = tiny_model(seed=3, torch_block_q=block_q)
    ids = _ids(2, seq_len, seed=3)
    ref = reference_fp64.logits_fp64(model, ids)
    floor = rel_err(reference_fp64.logits_fp64(model, ids, dtype=torch.float32), ref)
    err = rel_err(model(ids).logits, ref)
    assert err < 1e-5 and err <= 2.0 * floor + 1e-6, (err, floor)


def test_blocked_backward_matches_fp64():
    """Several checkpointed query blocks in the backward pass."""
    _check_grads(tiny_model(seed=11, torch_block_q=48), _ids(1, 150, seed=11))


def test_strong_forgetting_and_fast_decay():
    model = tiny_model(seed=5)
    with torch.no_grad():
        for layer in model.layers:
            layer.attn.forget_proj.bias.fill_(-3.0)              # about -3 nats per token
            layer.attn.alpha_proj.bias.fill_(2.0)                # decay of about -2 nats per token
    _check_grads(model, _ids(1, 200, seed=5))


def test_initialised_model_matches_fp64():
    """At init (W_C = 0) the SSM term is exactly zero in both implementations."""
    model = tiny_model(seed=7, randomize=False)
    ids = _ids(2, 50, seed=7)
    loss64, _ = reference_fp64.loss_and_grads(model, ids, ids)
    assert abs(model(ids, labels=ids).loss.item() - loss64) / loss64 < 1e-6
