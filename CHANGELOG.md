# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

Initial release.

### Added

- `SISAConfig` with the `sisa-50m`, `sisa-152m` and `sisa-369m` presets
  (d_head 64, d_state 32, vocabulary 50,277, trained at context 2,048).
- `SISAAttention`, `SISABlock`, `SISAModel` and `SISAForCausalLM`: a pre-norm
  RMSNorm / SwiGLU decoder with a tied embedding, where every attention layer
  uses the SISA score (NoPE q.k, rotated state-space term, forget gate).
  Each module applies the SISA initialization when constructed and exposes
  `reset_parameters()`; `lambda_raw` stays fp32 through dtype casts.
- HuggingFace-style loss: `forward(input_ids, labels=...)` shifts the labels
  internally and ignores `-100`.
- `param_groups()` for AdamW: no weight decay on `lambda_raw` (learning rate
  equal to the main one by default) and, by default, none on the decay and
  forget-gate biases.
- Attention backends, selected with `SISAConfig.attn_impl` (the standalone
  `sisa_attention` function takes `auto`, `torch` and `triton`):
  - `torch`: exact blocked pure-PyTorch reference, any device;
  - `triton`: exact tiled Triton attention kernel (forward and backward) with
    PyTorch glue;
  - `triton_fused`: fused projection prep and attention in Triton;
  - `auto`: picks `triton_fused`, then `triton`, then `torch`, and warns once
    when it falls back to `torch` on a CUDA input;
  - forcing `triton` or `triton_fused` on unsupported input raises
    `ValueError` (no silent fallback).
- `sisa.backends`: `describe()`, `triton_status()`, and `set_check_mode()`
  (`"async"`, `"sync"` or `"off"`) for the monotone-decay and NaN guards.
- Optional `sisa.kernels.linear_ce.linear_cross_entropy`, a chunked tied
  LM-head cross-entropy that never builds the full logits tensor.
- `generate()` with a pure-PyTorch incremental decode cache (greedy, or
  sampling with temperature and top-k).
- `save_pretrained()` / `from_pretrained()` (`config.json` plus
  `model.safetensors`, or `pytorch_model.bin` without safetensors).
- `sisa.checkpoint`: `convert_state_dict()` and `load_paper_checkpoint()` for
  converting checkpoints from the original training code. Only the model
  weights are kept.
- Test suite: CPU model tests, an fp64 whole-model reference, and CUDA kernel
  tests against fp64 (`-m cuda`, with exhaustive grids under `-m slow`).
- Typed package (`py.typed`).
- Examples: `quickstart.py`, `train_minimal.py` and `generate.py`.
