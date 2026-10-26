"""Model-level CUDA tests: the three attention implementations through SISAForCausalLM against float64.

Under bf16 autocast the error against float64 is dominated by bf16 rounding outside the attention (projections,
logits; the rotation rate theta is a bf16 projection output, so its rounding accumulates in Phi), and it is the
same for all implementations. The Triton paths are therefore judged against the torch glue under the same
autocast (whose attention core runs in float32, while the kernels round P to bf16 for P @ V): loss deviation
``<= max(2 * d_torch, 2e-3)`` nats, logits ``e <= 2 * e_torch + 1e-3`` and, per parameter group,
``e <= 3 * e_torch + 2e-3``. These are integration bounds; the kernel tests hold the tight per-kernel gates.
"""
import pytest
import torch

import reference_fp64
from model_utils import rel_err, tiny_model

pytestmark = pytest.mark.cuda


def _model(seed, device, theta_scale=0.25, **overrides):
    model = tiny_model(seed=seed, **overrides)
    with torch.no_grad():
        for layer in model.layers:
            layer.attn.theta_proj.weight.mul_(theta_scale)
    return model.to(device)


_GROUPS = {
    "qkvo": ("W_Q", "W_K", "W_V", "W_O"), "bc": ("W_B", "W_C"), "alpha": ("alpha_proj",),
    "theta": ("theta_proj",), "lambda": ("lambda_raw",), "forget": ("forget_proj",), "norms": ("norm",),
    "ffn": ("ffn.",), "embed": ("embed",),
}


def _grouped(grads):
    """Concatenate per-parameter gradients into the parameter groups of the paper's gradient gate."""
    out = {}
    for group, keys in _GROUPS.items():
        parts = [g.flatten().double() for n, g in sorted(grads.items()) if any(k in n for k in keys)]
        out[group] = torch.cat(parts)
    return out


def _ids(bsz, seq_len, device, seed=0):
    return torch.randint(0, 257, (bsz, seq_len), generator=torch.Generator().manual_seed(seed)).to(device)


def _run(model, ids, impl, autocast=True):
    model.set_attn_impl(impl)
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = model(ids, labels=ids)
    out.loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
    return out.loss.item(), out.logits.detach().float(), grads


def test_auto_selection(cuda_device):
    from sisa import backends

    model = tiny_model(seed=0, attn_impl="auto").to(cuda_device)
    x = torch.randn(1, 8, 128, device=cuda_device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert backends.select_impl("auto", x, model.config) == "triton_fused"
    assert backends.select_impl("auto", x, model.config) == "triton"


@pytest.mark.parametrize("seq_len", [2, 63, 200])
def test_bf16_impls_vs_fp64(cuda_device, seq_len):
    model = _model(seq_len, cuda_device)
    ids = _ids(2, seq_len, cuda_device, seed=seq_len)
    loss64, grads64 = reference_fp64.loss_and_grads(model, ids, ids)
    logits64 = reference_fp64.logits_fp64(model, ids)
    results = {impl: _run(model, ids, impl) for impl in ("torch", "triton", "triton_fused")}
    t_loss, t_logits, t_grads = results["torch"]
    assert abs(t_loss - loss64) < 2e-2 and rel_err(t_logits, logits64) < 5e-2
    for impl in ("triton", "triton_fused"):
        loss, logits, grads = results[impl]
        assert abs(loss - loss64) <= max(2.0 * abs(t_loss - loss64), 2e-3), impl
        assert rel_err(logits, logits64) <= 2.0 * rel_err(t_logits, logits64) + 1e-3, impl
        for name, g in grads.items():
            assert g.dtype == torch.float32 and torch.isfinite(g).all(), (impl, name)
        ref_g, tor_g, imp_g = _grouped(grads64), _grouped(t_grads), _grouped(grads)
        for group in _GROUPS:
            e_t = rel_err(tor_g[group], ref_g[group])
            assert rel_err(imp_g[group], ref_g[group]) <= 3.0 * e_t + 2e-3, (impl, group)


def test_fp32_triton_close_to_fp64(cuda_device):
    model = _model(3, cuda_device, theta_scale=1.0)
    ids = _ids(2, 130, cuda_device, seed=3)
    loss64, grads64 = reference_fp64.loss_and_grads(model, ids, ids)
    loss, _, grads = _run(model, ids, "triton", autocast=False)
    assert abs(loss - loss64) / loss64 < 1e-5
    for name, g in grads.items():
        assert rel_err(g, grads64[name]) < 1e-3, name


def test_triton_determinism(cuda_device):
    model = _model(4, cuda_device)
    ids = _ids(2, 150, cuda_device, seed=4)
    for impl in ("triton", "triton_fused"):
        a, b = _run(model, ids, impl), _run(model, ids, impl)
        assert a[0] == b[0] and torch.equal(a[1], b[1]), impl
        assert all(torch.equal(a[2][n], b[2][n]) for n in a[2]), impl


def test_fused_ce_matches_logits_loss(cuda_device):
    model = _model(5, cuda_device, attn_impl="triton")
    ids = _ids(2, 70, cuda_device, seed=5)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = model(ids, labels=ids).loss
        out = model(ids, labels=ids, fused_ce=True)
    assert out.logits is None
    assert abs(out.loss.item() - ref.item()) < 1e-2


def test_auto_falls_back_for_non_fp32_recurrences(cuda_device):
    """impl='auto' on prepared inputs the kernel cannot take uses torch; forcing 'triton' raises ValueError."""
    from sisa import backends, sisa_attention
    from sisa.functional import sisa_attention_torch

    gen = torch.Generator(device=cuda_device).manual_seed(0)
    q, k, v = (torch.randn(1, 2, 80, 64, device=cuda_device, generator=gen).bfloat16() for _ in range(3))
    c, b = (torch.randn(1, 2, 80, 32, device=cuda_device, generator=gen).bfloat16() for _ in range(2))
    g = torch.cumsum(-torch.rand(1, 2, 80, device=cuda_device, generator=gen), 2)
    f = torch.cumsum(torch.nn.functional.logsigmoid(torch.randn(1, 2, 80, device=cuda_device, generator=gen)), 2)
    lam = torch.tensor([0.3, 1.0], device=cuda_device)
    backends._WARNED.clear()
    for g_in, f_in in ((g.bfloat16(), f), (g, f.bfloat16())):
        with pytest.warns(UserWarning, match="using the torch attention implementation"):
            out = sisa_attention(q, k, v, c, b, g_in, lam, f_in, impl="auto")
        assert torch.equal(out, sisa_attention_torch(q, k, v, c, b, g_in, lam, f_in))
        with pytest.raises(ValueError, match="float32"):
            sisa_attention(q, k, v, c, b, g_in, lam, f_in, impl="triton")


def test_torch_backend_is_fp32_under_cuda_autocast(cuda_device):
    from sisa.functional import sisa_attention_torch

    gen = torch.Generator(device=cuda_device).manual_seed(1)
    q, k, v = (torch.randn(1, 2, 200, 64, device=cuda_device, generator=gen) for _ in range(3))
    c, b = (torch.randn(1, 2, 200, 32, device=cuda_device, generator=gen) for _ in range(2))
    g = torch.cumsum(-torch.rand(1, 2, 200, device=cuda_device, generator=gen) * 0.5, 2)
    f = torch.cumsum(torch.nn.functional.logsigmoid(torch.randn(1, 2, 200, device=cuda_device, generator=gen)), 2)
    lam = torch.tensor([0.3, 1.0], device=cuda_device)
    plain = sisa_attention_torch(q, k, v, c, b, g, lam, f, block_q=64)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = sisa_attention_torch(q, k, v, c, b, g, lam, f, block_q=64)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, plain, rtol=1e-6, atol=1e-6)
