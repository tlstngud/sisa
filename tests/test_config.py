import pytest
import torch

from sisa import PRESETS, SISAConfig, SISAForCausalLM

EXPECTED = {
    "sisa-50m": 50_911_376,
    "sisa-152m": 151_878_576,
    "sisa-369m": 369_278_080,
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_preset_parameter_counts(name):
    cfg = SISAConfig.preset(name)
    with torch.device("meta"):
        model = SISAForCausalLM(cfg)
    assert model.num_parameters() == EXPECTED[name]
    assert cfg.num_parameters() == EXPECTED[name]
    assert len(model.state_dict()) == 2 + 17 * cfg.n_layer


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_shapes(name):
    cfg = SISAConfig.preset(name)
    assert cfg.d_head == 64 and cfg.d_state == 32
    assert cfg.n_head * cfg.d_state == cfg.d_model // 2
    assert cfg.vocab_size == 50277 and cfg.max_seq_len == 2048


def test_defaults_are_the_50m_preset():
    assert SISAConfig() == SISAConfig.preset("sisa-50m")


def test_preset_overrides_and_unknown_name():
    cfg = SISAConfig.preset("sisa-152m", attn_impl="torch")
    assert cfg.attn_impl == "torch" and cfg.d_model == 768
    with pytest.raises(ValueError, match="unknown preset"):
        SISAConfig.preset("sisa-1b")


def test_dict_round_trip_and_unknown_fields():
    cfg = SISAConfig(vocab_size=101, d_model=96, n_head=3, n_layer=2, d_ff=40, d_state=8)
    assert SISAConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown SISAConfig fields"):
        SISAConfig.from_dict({**cfg.to_dict(), "not_a_field": 1})


def test_triton_impl_requires_kernel_shapes():
    with pytest.raises(ValueError, match="d_head=64"):
        SISAConfig(d_model=256, n_head=8, attn_impl="triton")              # d_head 32
    with pytest.raises(ValueError, match="d_state=32"):
        SISAConfig(d_state=16, attn_impl="triton_fused")
    SISAConfig(d_model=256, n_head=8, attn_impl="torch")                   # fine for torch
    SISAConfig(d_model=256, n_head=8, attn_impl="auto")                    # auto falls back at run time


def test_invalid_values():
    with pytest.raises(ValueError, match="even"):
        SISAConfig(d_state=31, attn_impl="torch")
    with pytest.raises(ValueError, match="divisible"):
        SISAConfig(d_model=100, n_head=8)
    with pytest.raises(ValueError, match="attn_impl"):
        SISAConfig(attn_impl="flash")
    with pytest.raises(ValueError, match="positive"):
        SISAConfig(n_layer=0)
    with pytest.raises(ValueError, match="positive"):
        SISAConfig(d_ff=True)


def test_replace_validates():
    cfg = SISAConfig()
    assert cfg.replace(attn_impl="torch").attn_impl == "torch"
    with pytest.raises(ValueError):
        cfg.replace(attn_impl="nope")
