import math

import pytest
import torch
import torch.nn.functional as F

from model_utils import randomize_, rel_err, tiny_config, tiny_model
from sisa import SISAAttention, SISABlock, SISAConfig, SISAForCausalLM, sisa_attention
from sisa.functional import _attend_rows, rotate_pairs, sisa_attention_torch


def _ids(bsz, seq_len, vocab=257, seed=0):
    return torch.randint(0, vocab, (bsz, seq_len), generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------------------------------- shapes, loss
def test_shapes_and_shifted_loss():
    model = tiny_model()
    ids = _ids(2, 33)
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 33, 257) and out.logits.dtype == torch.float32
    assert out.loss.dim() == 0 and torch.isfinite(out.loss)
    ref = F.cross_entropy(out.logits[:, :-1].reshape(-1, 257), ids[:, 1:].reshape(-1))
    torch.testing.assert_close(out.loss, ref, rtol=0, atol=0)


def test_ignore_index():
    model = tiny_model()
    ids = _ids(2, 20)
    labels = ids.clone()
    labels[0, 5:12] = -100
    out = model(ids, labels=labels)
    lg, tg = out.logits[:, :-1].reshape(-1, 257), labels[:, 1:].reshape(-1)
    keep = tg != -100
    ref = F.cross_entropy(lg[keep], tg[keep])
    torch.testing.assert_close(out.loss, ref, rtol=1e-6, atol=1e-6)


def test_loss_without_logits_and_logits_to_keep():
    model = tiny_model()
    ids = _ids(2, 25)
    full = model(ids, labels=ids)
    lossonly = model(ids, labels=ids, return_logits=False)
    assert lossonly.logits is None
    torch.testing.assert_close(lossonly.loss, full.loss, rtol=1e-6, atol=1e-6)
    last = model(ids, logits_to_keep=3).logits
    assert last.shape == (2, 3, 257)
    torch.testing.assert_close(last, full.logits[:, -3:], rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        model(ids, labels=ids, logits_to_keep=1)
    with pytest.raises(ValueError):
        model(ids, labels=ids[:, :-1])


def test_fused_ce_needs_cuda():
    model = tiny_model()
    ids = _ids(1, 8)
    with pytest.raises(ValueError, match="CUDA"):
        model(ids, labels=ids, fused_ce=True)


def test_fused_ce_without_labels_returns_logits():
    model = tiny_model()
    ids = _ids(1, 8)
    out = model(ids, fused_ce=True)
    assert out.loss is None
    assert torch.equal(out.logits, model(ids).logits)


# ---------------------------------------------------------------------------------------------------- init
def test_init_values():
    torch.manual_seed(0)
    cfg = tiny_config(d_model=256, n_head=4, d_ff=512)
    model = SISAForCausalLM(cfg)
    for layer in model.layers:
        a = layer.attn
        assert torch.all(a.alpha_proj.bias == -5.0)
        assert torch.all(a.forget_proj.bias == 4.0)
        assert torch.all(a.lambda_raw == -1.0) and a.lambda_raw.dtype == torch.float32
        assert torch.all(a.W_C.weight == 0)
        assert abs(a.theta_proj.weight.std().item() - 0.01) < 0.001
        for lin in (a.W_Q, a.W_K, a.W_V, a.W_O, a.W_B, a.alpha_proj, a.forget_proj,
                    layer.ffn.w_gate, layer.ffn.w_up, layer.ffn.w_down):
            assert abs(lin.weight.std().item() - 0.02) < 0.002
        assert torch.all(layer.attn_norm.weight == 1) and torch.all(layer.ffn_norm.weight == 1)
    assert abs(model.embed.weight.std().item() - 0.02) < 0.001
    assert torch.all(model.final_norm.weight == 1)
    torch.testing.assert_close(model.layers[0].attn.lam(), torch.full((4,), math.log1p(math.exp(-1.0))))


def _check_attention_init(a: SISAAttention, cfg: SISAConfig):
    assert torch.all(a.alpha_proj.bias == cfg.decay_bias_init)
    assert torch.all(a.forget_proj.bias == cfg.forget_bias_init)
    assert torch.all(a.lambda_raw == cfg.lambda_raw_init) and a.lambda_raw.dtype == torch.float32
    assert torch.all(a.W_C.weight == 0)
    assert abs(a.theta_proj.weight.std().item() - cfg.theta_init_std) < 0.001
    for lin in (a.W_Q, a.W_K, a.W_V, a.W_O, a.W_B, a.alpha_proj, a.forget_proj):
        assert abs(lin.weight.std().item() - cfg.init_std) < 0.002


def test_standalone_layers_are_initialised():
    """SISAAttention and SISABlock built outside SISAModel get the same init values."""
    torch.manual_seed(0)
    cfg = tiny_config(d_model=256, n_head=4, d_ff=512)
    _check_attention_init(SISAAttention(cfg), cfg)
    block = SISABlock(cfg)
    _check_attention_init(block.attn, cfg)
    for lin in (block.ffn.w_gate, block.ffn.w_up, block.ffn.w_down):
        assert abs(lin.weight.std().item() - 0.02) < 0.002
    assert torch.all(block.attn_norm.weight == 1) and torch.all(block.ffn_norm.weight == 1)


def test_reset_parameters_restores_init():
    torch.manual_seed(0)
    cfg = tiny_config(d_model=256, n_head=4, d_ff=512)
    model = SISAForCausalLM(cfg)
    randomize_(model)
    model.reset_parameters()
    for layer in model.layers:
        _check_attention_init(layer.attn, cfg)
        assert torch.all(layer.attn_norm.weight == 1) and torch.all(layer.ffn_norm.weight == 1)
    assert abs(model.embed.weight.std().item() - 0.02) < 0.001


def test_zero_init_c_starts_as_forgetting_attention():
    """With W_C = 0 the SSM term vanishes: the output equals the same model with lambda := 0, bitwise."""
    model = tiny_model(randomize=False)
    ids = _ids(2, 40)
    ref = model(ids).logits
    with torch.no_grad():
        for layer in model.layers:
            layer.attn.lambda_raw.fill_(float("-inf"))           # softplus(-inf) == 0 exactly
    assert torch.equal(model.layers[0].attn.lam(), torch.zeros(2))
    assert torch.equal(model(ids).logits, ref)


def test_zero_init_c_can_be_disabled():
    torch.manual_seed(0)
    model = SISAForCausalLM(tiny_config(zero_init_c=False))
    assert model.layers[0].attn.W_C.weight.abs().sum() > 0


# ---------------------------------------------------------------------------------------------------- causality
@pytest.mark.parametrize("block_q", [16, 256])
def test_causality_is_exact(block_q):
    model = tiny_model(torch_block_q=block_q)
    ids = _ids(2, 50)
    t = 23
    ref = model(ids).logits
    ids2 = ids.clone()
    ids2[:, t] = (ids2[:, t] + 7) % 257
    out = model(ids2).logits
    assert torch.equal(out[:, :t], ref[:, :t])
    assert not torch.equal(out[:, t], ref[:, t])


def test_prefix_consistency_and_block_size():
    model = tiny_model()
    ids = _ids(2, 70)
    full = model(ids).logits
    assert rel_err(model(ids[:, :31]).logits, full[:, :31]) < 1e-5
    model.config.torch_block_q = 9
    assert rel_err(model(ids).logits, full) < 1e-5


# ---------------------------------------------------------------------------------------------------- precision
def test_autocast_keeps_fp32_grads_and_lambda():
    model = tiny_model()
    ids = _ids(2, 24)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(ids, labels=ids)
    assert out.logits.dtype == torch.bfloat16
    out.loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert p.grad.dtype == torch.float32, name
        assert torch.isfinite(p.grad).all(), name
    assert all(layer.attn.lambda_raw.dtype == torch.float32 for layer in model.layers)


def test_autocast_close_to_fp32():
    model = tiny_model()
    ids = _ids(2, 24)
    ref = model(ids, labels=ids).loss
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(ids, labels=ids).loss
    assert abs(loss.item() - ref.item()) < 2e-2


def test_lambda_raw_stays_fp32_when_cast():
    model = tiny_model().to(torch.bfloat16)
    for layer in model.layers:
        assert layer.attn.lambda_raw.dtype == torch.float32
        assert layer.attn.W_Q.weight.dtype == torch.bfloat16
    ids = _ids(1, 12)
    out = model(ids, labels=ids)
    assert out.logits.dtype == torch.bfloat16 and torch.isfinite(out.loss)
    model.double()
    assert model.layers[0].attn.lambda_raw.dtype == torch.float64        # only bf16/fp16 are upcast


def test_lambda_raw_value_survives_low_precision_cast():
    """A bf16/fp16 cast keeps lambda_raw (and its gradient) at full float32 precision."""
    model = tiny_model()
    attn = model.layers[0].attn
    with torch.no_grad():
        attn.lambda_raw.fill_(-1.0 / 3.0)
    attn.lambda_raw.grad = torch.full_like(attn.lambda_raw, 1.0 / 3.0)
    ref, ref_grad = attn.lambda_raw.detach().clone(), attn.lambda_raw.grad.clone()
    for dtype in (torch.bfloat16, torch.float16):
        model.to(dtype)
        assert attn.lambda_raw.dtype == torch.float32 and attn.W_Q.weight.dtype == dtype
        assert torch.equal(attn.lambda_raw.detach(), ref)
        assert attn.lambda_raw.grad.dtype == torch.float32 and torch.equal(attn.lambda_raw.grad, ref_grad)
    model.float()
    assert torch.equal(attn.lambda_raw.detach(), ref)


def test_float64_model():
    model = tiny_model().double()
    ids = _ids(1, 10)
    assert model(ids).logits.dtype == torch.float64


# ---------------------------------------------------------------------------------------------------- training utils
def test_param_groups():
    model = tiny_model()
    groups = model.param_groups(lr=6e-4, weight_decay=0.1)
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"default", "timescale_biases", "lambda"}
    assert by_name["default"]["weight_decay"] == 0.1 and by_name["default"]["lr"] == 6e-4
    assert by_name["timescale_biases"]["weight_decay"] == 0.0 and len(by_name["timescale_biases"]["params"]) == 4
    assert by_name["lambda"]["weight_decay"] == 0.0 and by_name["lambda"]["lr"] == 6e-4
    assert len(by_name["lambda"]["params"]) == 2
    assert all(p.dtype == torch.float32 for p in by_name["lambda"]["params"])
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))

    groups = model.param_groups(lr=1e-3, lambda_lr=5e-4, no_decay_timescale_biases=False)
    by_name = {g["name"]: g for g in groups}
    assert set(by_name) == {"default", "lambda"} and by_name["lambda"]["lr"] == 5e-4
    torch.optim.AdamW(groups, betas=(0.9, 0.95))


def test_gradient_checkpointing_matches():
    ids = _ids(2, 30)
    grads = []
    for ckpt in (False, True):
        model = tiny_model(gradient_checkpointing=ckpt).train()
        model(ids, labels=ids).loss.backward()
        grads.append({n: p.grad for n, p in model.named_parameters()})
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name], rtol=1e-6, atol=1e-8)


def test_num_parameters_and_counts():
    model = tiny_model()
    cfg = model.config
    assert model.num_parameters() == cfg.num_parameters()
    assert model.num_parameters(exclude_embeddings=True) == cfg.num_parameters() - cfg.vocab_size * cfg.d_model


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_compile_common_keeps_state_dict_keys():
    model = tiny_model()
    keys = list(model.state_dict())
    model.compile_common()
    assert list(model.state_dict()) == keys
    assert not any("_orig_mod" in k for k in keys)


# ---------------------------------------------------------------------------------------------------- backends
def test_impl_selection_on_cpu():
    model = tiny_model(attn_impl="auto")
    ids = _ids(1, 9)
    ref = tiny_model()(ids).logits
    assert torch.equal(model(ids).logits, ref)                     # auto -> torch on CPU
    for impl in ("triton", "triton_fused"):
        model.set_attn_impl(impl)
        with pytest.raises(ValueError, match="CUDA"):
            model(ids)
    with pytest.raises(ValueError):
        model.set_attn_impl("flash")
    small = SISAForCausalLM(tiny_config(d_model=64, n_head=2))     # d_head 32
    with pytest.raises(ValueError, match="d_head"):
        small.set_attn_impl("triton")


def test_attention_module_impl_override():
    model = tiny_model(attn_impl="auto")
    attn: SISAAttention = model.layers[0].attn
    x = torch.randn(2, 11, 128)
    torch.testing.assert_close(attn(x, impl="torch"), attn(x), rtol=0, atol=0)


# ---------------------------------------------------------------------------------------------------- functional
def _dense64(q, k, v, c, b, g, lam, fgate):
    q, k, v, c, b, g, lam = (t.double() for t in (q, k, v, c, b, g, lam))
    L = q.shape[2]
    causal = torch.ones(L, L, dtype=torch.bool).tril()
    delta = (g[..., :, None] - g[..., None, :]).masked_fill(~causal, 0)
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + lam.view(1, -1, 1, 1) * delta.exp() * (c @ b.transpose(-1, -2))
    if fgate is not None:
        fgate = fgate.double()
        s = s + fgate[..., :, None] - fgate[..., None, :]
    return s.masked_fill(~causal, float("-inf")).softmax(-1) @ v


@pytest.mark.parametrize("with_f", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 1, 64, 32), (2, 3, 37, 16, 6), (1, 2, 130, 64, 32)])
def test_attention_torch_vs_dense_fp64(shape, with_f):
    bsz, h, seq_len, d, s = shape
    gen = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(bsz, h, seq_len, d, generator=gen) for _ in range(3))
    c, b = (torch.randn(bsz, h, seq_len, s, generator=gen) for _ in range(2))
    g = torch.cumsum(-torch.rand(bsz, h, seq_len, generator=gen) * 0.5, dim=2)
    lam = torch.rand(h, generator=gen) + 0.1
    fgate = torch.cumsum(F.logsigmoid(torch.randn(bsz, h, seq_len, generator=gen) + 1), 2) if with_f else None
    ref = _dense64(q, k, v, c, b, g, lam, fgate)
    for block_q in (7, 256):
        out = sisa_attention_torch(q, k, v, c, b, g, lam, fgate, block_q=block_q)
        assert out.dtype == torch.float32 and out.shape == v.shape
        assert rel_err(out, ref) < 1e-5
    assert torch.equal(sisa_attention(q, k, v, c, b, g, lam, fgate, impl="auto"),
                       sisa_attention_torch(q, k, v, c, b, g, lam, fgate))


def test_attention_torch_bf16_inputs_and_grads():
    gen = torch.Generator().manual_seed(2)
    q, k, v = (torch.randn(1, 2, 70, 64, generator=gen).bfloat16().requires_grad_() for _ in range(3))
    c, b = (torch.randn(1, 2, 70, 32, generator=gen).bfloat16().requires_grad_() for _ in range(2))
    g = torch.cumsum(-torch.rand(1, 2, 70, generator=gen), 2).requires_grad_()
    lam = torch.tensor([0.3, 1.0], requires_grad=True)
    fgate = torch.cumsum(F.logsigmoid(torch.randn(1, 2, 70, generator=gen)), 2).requires_grad_()
    out = sisa_attention_torch(q, k, v, c, b, g, lam, fgate, block_q=32)
    assert out.dtype == torch.bfloat16
    out.float().square().sum().backward()
    for t in (q, k, v, c, b, g, lam, fgate):
        assert t.grad is not None and torch.isfinite(t.grad).all()


def _attention_inputs(seed, bsz=1, h=2, seq_len=300, d=64, s=32):
    gen = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(bsz, h, seq_len, d, generator=gen) for _ in range(3))
    c, b = (torch.randn(bsz, h, seq_len, s, generator=gen) for _ in range(2))
    g = torch.cumsum(-torch.rand(bsz, h, seq_len, generator=gen) * 0.5, dim=2)
    lam = torch.rand(h, generator=gen) + 0.1
    fgate = torch.cumsum(F.logsigmoid(torch.randn(bsz, h, seq_len, generator=gen) + 1), 2)
    return q, k, v, c, b, g, lam, fgate


@pytest.mark.parametrize("block_q", [64, 512])
def test_attention_torch_is_fp32_under_autocast(block_q):
    """The torch backend disables autocast inside, so bf16 autocast does not round its scores."""
    ins = _attention_inputs(3)
    ref = _dense64(*ins)
    plain = sisa_attention_torch(*ins, block_q=block_q)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = sisa_attention_torch(*ins, block_q=block_q)
    assert out.dtype == torch.float32
    assert rel_err(out, ref) < 1e-5
    torch.testing.assert_close(out, plain, rtol=1e-6, atol=1e-6)
    low = [t.bfloat16() for t in ins[:5]] + list(ins[5:])              # bf16 q, k, v, c, b
    plain = sisa_attention_torch(*low, block_q=block_q)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = sisa_attention_torch(*low, block_q=block_q)
    assert out.dtype == torch.bfloat16 and torch.equal(out, plain)


def test_attention_torch_compiles_fullgraph():
    """The torch backend (autocast guard included) traces into a single graph with torch.compile."""
    ins = _attention_inputs(5, seq_len=40)
    compiled = torch.compile(sisa_attention_torch, fullgraph=True, backend="eager")
    torch.testing.assert_close(compiled(*ins, block_q=16), sisa_attention_torch(*ins, block_q=16))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = compiled(*ins, block_q=16)
    assert out.dtype == torch.float32


def test_decode_rows_are_fp32_under_autocast():
    """The decode step (_attend_rows on the cache) also stays float32 under autocast."""
    q, k, v, c, b, g, lam, fgate = _attention_inputs(4, seq_len=40)
    rows = (q[:, :, -3:], k, v, c[:, :, -3:], b, g[:, :, -3:], g, lam, fgate[:, :, -3:], fgate, 37)
    plain = _attend_rows(*rows)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = _attend_rows(*rows)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, plain, rtol=1e-6, atol=1e-6)
    assert rel_err(out, _dense64(q, k, v, c, b, g, lam, fgate)[:, :, -3:]) < 1e-5


def test_attention_strong_decay_no_overflow():
    """exp(g_i - g_j) for j > i would overflow without masking; the output must stay finite."""
    q = torch.randn(1, 1, 300, 8)
    c = torch.randn(1, 1, 300, 4)
    g = torch.cumsum(torch.full((1, 1, 300), -50.0), 2)
    out = sisa_attention_torch(q, q, q, c, c, g, torch.ones(1), None)
    assert torch.isfinite(out).all()


def test_attention_errors():
    q = torch.randn(1, 2, 5, 64)
    c = torch.randn(1, 2, 5, 32)
    g = torch.zeros(1, 2, 5)
    lam = torch.ones(2)
    with pytest.raises(ValueError, match="triton"):
        sisa_attention(q, q, q, c, c, g, lam, impl="triton")
    with pytest.raises(ValueError, match="triton_fused"):
        sisa_attention(q, q, q, c, c, g, lam, impl="triton_fused")
    with pytest.raises(ValueError):
        sisa_attention(q, q, q, c, c, g, lam, impl="other")
    with pytest.raises(ValueError):
        sisa_attention_torch(q, q, q, c, c, g, torch.ones(3))


def test_rotate_pairs_is_complex_multiplication():
    z = torch.randn(3, 5, 8, dtype=torch.float64)
    phi = torch.randn(3, 5, 4, dtype=torch.float64)
    zc = torch.view_as_complex(z.reshape(3, 5, 4, 2).contiguous())
    ref = torch.view_as_real(zc * torch.polar(torch.ones_like(phi), phi)).reshape(3, 5, 8)
    torch.testing.assert_close(rotate_pairs(z, phi), ref)


def test_config_is_private_copy():
    cfg = tiny_config()
    model = SISAForCausalLM(cfg)
    model.set_attn_impl("auto")
    assert cfg.attn_impl == "torch"
    assert model.layers[1].attn.config is model.config
    assert isinstance(model.config, SISAConfig)
