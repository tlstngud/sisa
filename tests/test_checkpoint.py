import pickle
import re

import pytest
import torch

from model_utils import tiny_model
from sisa import SISAConfig, SISAForCausalLM, convert_state_dict, load_paper_checkpoint
from sisa.checkpoint import infer_config_from_state_dict

_ATTN = re.compile(r"^layers\.(\d+)\.attn\.")
_COMPILED = re.compile(r"^(layers\.\d+\.(?:attn_norm|ffn_norm|ffn)|final_norm)\.")


def paper_state_dict(model, prefix=""):
    """The model's weights in the original training code's layout (flat attention keys, compiled submodules)."""
    out = {}
    for key, value in model.state_dict().items():
        key = _ATTN.sub(r"layers.\1.", key)
        key = _COMPILED.sub(r"\1._orig_mod.", key)
        out[prefix + key] = value.clone()
    return out


class _TrainConfig:
    """Stands in for a pickled training-config object."""

    d_ff = 2048


def _ids(seq_len=20):
    return torch.randint(0, 257, (2, seq_len), generator=torch.Generator().manual_seed(0))


def test_paper_layout_keys():
    sd = paper_state_dict(tiny_model())
    assert "layers.0.W_Q.weight" in sd and "layers.1.lambda_raw" in sd
    assert "layers.0.ffn._orig_mod.w_gate.weight" in sd and "final_norm._orig_mod.weight" in sd
    assert "layers.0.attn_norm._orig_mod.weight" in sd and "embed.weight" in sd


@pytest.mark.parametrize("prefix", ["", "module.", "_orig_mod.", "module._orig_mod."])
def test_convert_round_trip_gives_identical_outputs(prefix):
    src = tiny_model(seed=1)
    converted = convert_state_dict(paper_state_dict(src, prefix))
    assert set(converted) == set(src.state_dict())
    assert convert_state_dict(converted).keys() == converted.keys()               # idempotent
    dst = SISAForCausalLM(infer_config_from_state_dict(converted, attn_impl="torch"))
    dst.load_state_dict(converted, strict=True)
    ids = _ids()
    assert torch.equal(dst(ids).logits, src(ids).logits)


def test_infer_config_reads_shapes_only():
    src = tiny_model(seed=2, d_ff=200, n_layer=3)
    cfg = infer_config_from_state_dict(paper_state_dict(src))
    for field in ("vocab_size", "d_model", "n_layer", "n_head", "d_ff", "d_state"):
        assert getattr(cfg, field) == getattr(src.config, field), field
    with pytest.raises(ValueError, match="cannot be overridden"):
        infer_config_from_state_dict(paper_state_dict(src), d_ff=100)
    with pytest.raises(ValueError, match="not a SISA state dict"):
        infer_config_from_state_dict({"embed.weight": torch.zeros(3, 4)})


def test_load_paper_checkpoint(tmp_path):
    src = tiny_model(seed=3)
    ckpt = {
        "model": paper_state_dict(src),
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {"last_epoch": 7},
        "step": 7,
        "config": {"d_ff": 2048, "data_dir": "/somewhere/tokens", "run_name": "example"},
    }
    path = tmp_path / "final.pt"
    torch.save(ckpt, path)
    model = load_paper_checkpoint(path, attn_impl="torch")
    assert not model.training
    assert model.config.d_ff == src.config.d_ff                    # from w_gate, not the stored config
    assert model.num_parameters() == src.num_parameters()
    ids = _ids()
    assert torch.equal(model(ids).logits, src(ids).logits)
    stored = [v for v in vars(model).values() if isinstance(v, dict) and "data_dir" in v]
    assert not stored and isinstance(model.config, SISAConfig)
    bf16 = load_paper_checkpoint(path, dtype=torch.bfloat16)
    assert bf16.embed.weight.dtype == torch.bfloat16 and bf16.layers[0].attn.lambda_raw.dtype == torch.float32


def test_load_plain_state_dict_file(tmp_path):
    src = tiny_model(seed=4)
    path = tmp_path / "weights.pt"
    torch.save(paper_state_dict(src), path)
    model = load_paper_checkpoint(path, attn_impl="torch")
    assert torch.equal(model(_ids()).logits, src(_ids()).logits)


def test_pickled_objects_need_trust(tmp_path):
    src = tiny_model(seed=5)
    path = tmp_path / "final.pt"
    torch.save({"model": paper_state_dict(src), "config": _TrainConfig()}, path)
    with pytest.raises(pickle.UnpicklingError, match="trust_pickle"):
        load_paper_checkpoint(path)
    model = load_paper_checkpoint(path, trust_pickle=True, attn_impl="torch")
    assert torch.equal(model(_ids()).logits, src(_ids()).logits)


def test_key_and_shape_mismatches_raise(tmp_path):
    src = tiny_model(seed=6)
    sd = paper_state_dict(src)
    del sd["layers.1.forget_proj.bias"]
    torch.save({"model": sd}, tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="missing"):
        load_paper_checkpoint(tmp_path / "missing.pt")
    sd = paper_state_dict(src)
    sd["layers.0.extra.weight"] = torch.zeros(1)
    torch.save({"model": sd}, tmp_path / "extra.pt")
    with pytest.raises(ValueError, match="unexpected"):
        load_paper_checkpoint(tmp_path / "extra.pt")
    sd = paper_state_dict(src)
    sd["layers.1.ffn._orig_mod.w_up.weight"] = torch.zeros(10, 128)
    torch.save({"model": sd}, tmp_path / "shape.pt")
    with pytest.raises(ValueError, match="shape mismatch"):
        load_paper_checkpoint(tmp_path / "shape.pt")
    torch.save({"something": 1}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="state dict"):
        load_paper_checkpoint(tmp_path / "bad.pt")


@pytest.mark.parametrize("safe", [True, False])
def test_save_and_from_pretrained(tmp_path, safe):
    if safe:
        pytest.importorskip("safetensors")
    src = tiny_model(seed=7)
    src.save_pretrained(tmp_path, safe_serialization=safe)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / ("model.safetensors" if safe else "pytorch_model.bin")).exists()
    loaded = SISAForCausalLM.from_pretrained(tmp_path)
    assert loaded.config == src.config and not loaded.training
    ids = _ids()
    assert torch.equal(loaded(ids).logits, src(ids).logits)
    other = SISAForCausalLM.from_pretrained(tmp_path, attn_impl="auto", dtype=torch.bfloat16)
    assert other.config.attn_impl == "auto" and other.embed.weight.dtype == torch.bfloat16


def test_save_removes_stale_weights(tmp_path):
    pytest.importorskip("safetensors")
    src = tiny_model(seed=8)
    src.save_pretrained(tmp_path, safe_serialization=False)
    src.save_pretrained(tmp_path, safe_serialization=True)
    assert not (tmp_path / "pytorch_model.bin").exists() and (tmp_path / "model.safetensors").exists()


def test_from_pretrained_errors(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        SISAForCausalLM.from_pretrained(tmp_path / "missing")
    tiny_model().save_pretrained(tmp_path)
    (tmp_path / "model.safetensors").unlink(missing_ok=True)
    (tmp_path / "pytorch_model.bin").unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError):
        SISAForCausalLM.from_pretrained(tmp_path)
