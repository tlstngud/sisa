import pytest
import torch

import reference_fp64
from model_utils import rel_err, tiny_model
from sisa.generation import SISACache


def _ids(bsz, seq_len, seed=0):
    return torch.randint(0, 257, (bsz, seq_len), generator=torch.Generator().manual_seed(seed))


def _cached_logits(model, ids, prompt_len, step=1):
    """Logits for every position, computed by prefill + incremental decoding in chunks of ``step``."""
    cache = SISACache()
    out = [model(ids[:, :prompt_len], cache=cache).logits]
    for start in range(prompt_len, ids.shape[1], step):
        out.append(model(ids[:, start:start + step], cache=cache).logits)
    assert cache.seq_len == ids.shape[1]
    return torch.cat(out, dim=1)


def _check_decode(model, ids, cached):
    """Decode and full forward are two float32 evaluations of the same function: compare both with float64.

    Per position they differ by up to about 3e-5 (the full forward's own float32 error reaches 3e-5 at some
    positions), so the bounds are relative to the full forward's error rather than a fixed 1e-5.
    """
    full = model(ids).logits
    ref = reference_fp64.logits_fp64(model, ids)
    assert rel_err(cached, full) < 2e-5
    assert rel_err(cached, ref) <= 2.0 * rel_err(full, ref) + 1e-6
    for t in range(ids.shape[1]):
        assert rel_err(cached[:, t], full[:, t]) < 1e-4, t


@pytest.mark.parametrize("prompt_len", [1, 5, 40])
def test_cached_decode_matches_full_forward(prompt_len):
    model = tiny_model(seed=1)
    ids = _ids(2, 64, seed=1)
    _check_decode(model, ids, _cached_logits(model, ids, prompt_len))


def test_chunked_decode_matches_full_forward():
    model = tiny_model(seed=2)
    ids = _ids(1, 50, seed=2)
    _check_decode(model, ids, _cached_logits(model, ids, prompt_len=7, step=4))


def test_greedy_matches_uncached_argmax():
    model = tiny_model(seed=3)
    prompt = _ids(2, 6, seed=3)
    out = model.generate(prompt, max_new_tokens=10)
    assert out.shape == (2, 16) and torch.equal(out[:, :6], prompt)
    seq = prompt
    for _ in range(10):
        nxt = model(seq).logits[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
    assert torch.equal(out, seq)
    assert torch.equal(model.generate(prompt, max_new_tokens=10), out)            # deterministic


def test_eos_stops_and_pads():
    model = tiny_model(seed=4)
    prompt = _ids(2, 4, seed=4)
    greedy = model.generate(prompt, max_new_tokens=8)
    eos = int(greedy[0, 6])                       # the token row 0 emits at its third step
    out = model.generate(prompt, max_new_tokens=8, eos_token_id=eos)
    first = (out[0, 4:] == eos).nonzero()[0].item()
    assert torch.all(out[0, 4 + first:] == eos)
    if (out[1, 4:] == eos).any():
        assert out.shape[1] < 12
    else:
        assert out.shape[1] == 12


def test_eos_token_id_types():
    np = pytest.importorskip("numpy")
    from sisa.generation import _eos_ids

    assert _eos_ids(None) == []
    assert _eos_ids(5) == [5]
    assert _eos_ids(np.int64(5)) == [5]
    assert _eos_ids(torch.tensor(5)) == [5]
    assert _eos_ids(torch.tensor([5, 7])) == [5, 7]
    assert _eos_ids(np.array([5, 7])) == [5, 7]
    assert _eos_ids((5, 7)) == [5, 7]
    for bad in (object(), "50", b"5", True, 5.5, [True], [1, 2.0], torch.tensor(5.7), np.array([1.5]),
                torch.tensor([[5]])):
        with pytest.raises(TypeError, match="eos_token_id"):
            _eos_ids(bad)
    model = tiny_model(seed=4)
    prompt = _ids(2, 4, seed=4)
    eos = int(model.generate(prompt, max_new_tokens=8)[0, 6])
    ref = model.generate(prompt, max_new_tokens=8, eos_token_id=eos)
    assert torch.equal(model.generate(prompt, max_new_tokens=8, eos_token_id=np.int64(eos)), ref)
    assert torch.equal(model.generate(prompt, max_new_tokens=8, eos_token_id=torch.tensor(eos)), ref)


def test_sampling_reproducible_and_top_k_one_is_greedy():
    model = tiny_model(seed=5)
    prompt = _ids(1, 5, seed=5)
    a = model.generate(prompt, 12, do_sample=True, temperature=0.8, generator=torch.Generator().manual_seed(0))
    b = model.generate(prompt, 12, do_sample=True, temperature=0.8, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b)
    k1 = model.generate(prompt, 12, do_sample=True, top_k=1, generator=torch.Generator().manual_seed(1))
    assert torch.equal(k1, model.generate(prompt, 12))


def test_generate_arguments():
    model = tiny_model(seed=6)
    prompt = _ids(1, 5, seed=6)[0]
    out = model.generate(prompt, 3)
    assert out.shape == (8,)
    assert torch.equal(model.generate(prompt, 0), prompt)
    with pytest.raises(ValueError):
        model.generate(prompt, -1)
    with pytest.raises(ValueError):
        model.generate(prompt, 2, do_sample=True, temperature=0.0)
    with pytest.raises(ValueError):
        model.generate(prompt, 2, do_sample=True, top_k=0)


def test_autocast_decode_close_to_full_forward():
    model = tiny_model(seed=7)
    ids = _ids(1, 30, seed=7)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        full = model(ids).logits.float()
        cached = _cached_logits(model, ids, prompt_len=10).float()
    assert rel_err(cached, full) < 2e-2


@pytest.mark.cuda
def test_cuda_bf16_decode_close_to_full_forward(cuda_device):
    """bf16: decode (float32 torch attention on the bf16 cache) vs the full Triton forward, both against float64."""
    model = tiny_model(seed=8, attn_impl="auto")
    with torch.no_grad():
        for layer in model.layers:
            layer.attn.theta_proj.weight.mul_(0.25)
    model = model.to(cuda_device)
    ids = _ids(2, 96, seed=8).to(cuda_device)
    ref = reference_fp64.logits_fp64(model, ids)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full = model(ids).logits.float()
        cached = _cached_logits(model, ids, prompt_len=33).float()
        out = model.generate(ids[:, :33], max_new_tokens=5)
    assert out.shape == (2, 38)
    assert rel_err(cached, ref) <= 2.0 * rel_err(full, ref) + 1e-3
    for t in range(ids.shape[1]):
        assert rel_err(cached[:, t], full[:, t]) < 5e-2, t
