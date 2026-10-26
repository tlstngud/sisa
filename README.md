# SISA

SISA (SSM-Informed Softmax Attention) is a causal softmax attention layer
for decoder-only language models. Each head computes one softmax over a
score with three terms: a dot product q·k with **no positional encoding**
(NoPE), a **state-space term** that carries relative position and recency,
and an additive **forget gate**. Everything else is a standard pre-norm
decoder block with RMSNorm, a SwiGLU MLP and an embedding shared with the
LM head. Only the attention sublayer changes.

This package provides the model, exact Triton kernels (forward and backward),
an exact pure-PyTorch reference that runs on any device, incremental
generation, and a converter for checkpoints from the original training code.

> **Name.** This project has nothing to do with *SISA training (Sharded,
> Isolated, Sliced, and Aggregated)*, the machine-unlearning method. The
> distribution is called `sisa-attention`, and the import name is `sisa`.

## Contents

- [The score](#the-score)
- [Install](#install)
- [Quickstart](#quickstart)
- [Training with your own data loader](#training-with-your-own-data-loader)
- [Backends](#backends)
- [Kernel notes](#kernel-notes)
- [Model sizes](#model-sizes)
- [Converting checkpoints](#converting-checkpoints)
- [Limitations](#limitations)
- [Tests](#tests)
- [Citation](#citation) · [License](#license) · [Acknowledgements](#acknowledgements)

## The score

Let x̄_t = RMSNorm(x_t). For each head, x̄_t is mapped linearly to
q_t, k_t, v_t ∈ R^{d_h}, to B_t, C_t ∈ R^{d_s} and θ_t ∈ R^{d_s/2}, and to two
scalars: a decay logit a_t and a forget logit z_t. For j ≤ i:

```
s_ij = q_i·k_j / √d_h  +  λ_h · exp(g_i − g_j) · C̃_iᵀ B̃_j  +  (F_i − F_j)

o_i  = Σ_{j ≤ i} softmax_j(s_ij) · v_j
```

where

```
Φ_t = Σ_{τ≤t} θ_τ                      cumulative rotation phase
C̃_t = R(Φ_t) C_t,   B̃_t = R(Φ_t) B_t   R: planar rotations of channel pairs (0,1), (2,3), ...
g_t = −Σ_{τ≤t} softplus(a_τ)            cumulative log-decay (non-increasing)
F_t =  Σ_{τ≤t} log σ(z_τ)               cumulative log forget gate (non-increasing)
λ_h = softplus(λ_raw,h)                 learned per-head scale
```

- **Position.** q·k carries no positional encoding. Relative position enters
  only through the rotation, since C̃_iᵀB̃_j = C_iᵀ R(Φ_i − Φ_j)ᵀ B_j depends
  on Φ_i − Φ_j.
- **Recency.** Both g and F are non-increasing, so exp(g_i − g_j) ≤ 1 and
  F_i − F_j ≤ 0. Nothing can overflow, and the score needs no clamping or
  centering. The state-space term has no extra 1/√d scale.
- **Initialization.** W_C starts at zero, so the model starts out as a NoPE
  forgetting Transformer (FoX) and learns the state-space term during
  training. Other init values: decay bias −5 (half-life ≈ 100 tokens), forget
  bias +4 (σ ≈ 0.98), λ_raw = −1 (λ ≈ 0.31), W_θ ~ N(0, 0.01²), every
  other weight (including the embedding) ~ N(0, 0.02²), other biases 0, and
  RMSNorm weights 1. `SISAAttention` and `SISABlock` apply this
  initialization in their constructors, so they can be dropped into other
  models; `reset_parameters()` applies it again.
- **Cost.** With H·d_s = d/2, the state-space term adds 5/4·d² + dH + 2H
  parameters per layer (≈ 1.27·d² at the preset shapes), and the forget gate
  adds dH + H. The presets shrink the MLP so that the total matches a
  Transformer of the same shape.

## Install

The package is not on PyPI yet. Install from a clone:

```bash
git clone <repository-url> sisa-attention
cd sisa-attention
pip install -e ".[io,tokenizer,test]"     # extras are optional
```

| Extra | Adds | Used for |
|---|---|---|
| `io` | `safetensors>=0.4` | `save_pretrained` / `from_pretrained` (without it, `pytorch_model.bin`) |
| `tokenizer` | `tokenizers>=0.19` | the GPT-NeoX tokenizer in `examples/generate.py` |
| `test` | `pytest`, `numpy`, `mpmath` | the test suite |

Requirements:

- Python ≥ 3.10 and PyTorch ≥ 2.7.
- The Triton backends also need Linux, an NVIDIA GPU with compute capability
  8.0 or newer, and Triton ≥ 3.3. Linux CUDA wheels of PyTorch ≥ 2.7 already
  bundle a suitable Triton, so do not install Triton separately. Triton ≥ 3.6
  is recommended for speed.
- Without a supported GPU, everything runs on the pure-PyTorch backend.

To see which backends are available on your machine:

```bash
python -c "from sisa import backends; print(backends.describe())"
```

## Quickstart

```python
import torch
from sisa import SISAConfig, SISAForCausalLM

cfg = SISAConfig.preset("sisa-50m")
model = SISAForCausalLM(cfg).cuda()              # fp32 weights

ids = torch.randint(0, cfg.vocab_size, (2, 2048), device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(ids, labels=ids)                 # labels are shifted inside the model
out.loss.backward()
print(out.loss.item(), out.logits.shape)         # roughly ln(50277) at init; (2, 2048, 50277)
```

On CPU, drop `.cuda()`, `device="cuda"` and the autocast (or use a smaller
config); everything then runs on the pure-PyTorch backend. The scripts in
`examples/` handle both cases:

```bash
python examples/quickstart.py                    # add --device cpu to force CPU
python examples/train_minimal.py --device cpu    # tiny model on synthetic data, under a minute
python examples/generate.py --model my-sisa --prompt "The ocean is"   # a save_pretrained directory
```

To generate, pass a prompt. The call returns the prompt followed by the new
tokens. Training joined documents with EOS (id 0), so start a text prompt
with that token to make it look like the beginning of a document
(`examples/generate.py` does this by default).

```python
model.eval()
prompt = ids[:1, :32]
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    seq = model.generate(prompt, max_new_tokens=64, do_sample=True,
                         temperature=0.8, top_k=50, eos_token_id=0)
```

To save and load a model:

```python
model.save_pretrained("my-sisa")                 # config.json + model.safetensors
model = SISAForCausalLM.from_pretrained("my-sisa").cuda()   # loads on CPU in eval mode
```

The attention function can also be called on its own:

```python
from sisa import sisa_attention

# q, k, v: (B, H, L, d_h)   c, b: (B, H, L, d_s), already rotated (C̃, B̃), same dtype as q
# g, fgate: (B, H, L) fp32 cumulative sums (non-increasing)   lam: (H,) fp32 = softplus(lambda_raw)
o = sisa_attention(q, k, v, c, b, g, lam, fgate, impl="auto")   # (B, H, L, d_h), q.dtype
```

`sisa.functional.rotate_pairs(z, phi)` applies the pair rotation R(Φ).

### Label convention

The loss follows the Hugging Face convention. `forward(input_ids, labels)`
shifts internally and computes

```
loss = cross_entropy(logits[:, :-1], labels[:, 1:], ignore_index=-100)
```

as a mean over the positions whose label is not `-100`. Pass the same tensor
as `input_ids` and `labels`, and set the labels you want masked (for example
prompt tokens) to `-100`. A block of L tokens gives L − 1 predictions. The
original models were trained on 2,048 predictions per sequence, so feed
2,049-token blocks to match that. The attention is causal, so the extra
token changes nothing at earlier positions.

The optional `sisa.kernels.linear_ce.linear_cross_entropy(hidden, weight,
labels)` is a low-level function and does **not** shift: `labels[n]` is the
target for `hidden[n]`.

## Training with your own data loader

`loader` below is any iterator that yields `int64` token blocks of shape
(micro_batch, L), for example `iter(dataloader)` or
`itertools.cycle(dataloader)` for a PyTorch `DataLoader` (which is not an
iterator by itself). `examples/train_minimal.py` is a complete script: it runs on CPU in
well under a minute with synthetic data, and it also reads your own token
files.

```python
import math
import torch
from sisa import SISAConfig, SISAForCausalLM

cfg = SISAConfig.preset("sisa-50m")
model = SISAForCausalLM(cfg).cuda()        # keep fp32 master weights

lr, warmup, total_steps, grad_accum = 6e-4, 500, 19_073, 16
# The 50M runs decayed the timescale biases; the 152M/369M runs used the default (True).
groups = model.param_groups(lr, weight_decay=0.1, no_decay_timescale_biases=False)
opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

def lr_factor(step):                       # linear warm-up from 0, then cosine to 0
    if step < warmup:
        return step / warmup
    t = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * t))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

for step in range(total_steps):
    for _ in range(grad_accum):
        blk = next(loader).cuda(non_blocking=True)      # (16, 2049) token ids
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(blk, labels=blk, fused_ce=True).loss   # no (16, 2049, V) logits
        (loss / grad_accum).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    sched.step()
```

`fused_ce=True` (CUDA with Triton) computes the loss without the
(16, 2049, V) logits. Without it, the logits, their fp32 copy and their
gradient at this micro-batch need more memory than a 24 GB card has; use a
smaller micro-batch with more accumulation steps there.

The original runs stopped after 19,074 optimizer steps, once 10B tokens had
been seen. With this schedule the extra step has learning rate 0 and leaves
the weights unchanged, so `range(total_steps)` gives the same weights.

`model.param_groups(lr, weight_decay=0.1, lambda_lr=None,
no_decay_timescale_biases=True)` returns up to three AdamW groups (the
`name` key of each group is given in the first column):

| Group | Parameters | Weight decay | Learning rate |
|---|---|---|---|
| `default` | everything else (including norms and the embedding) | `weight_decay` | `lr` |
| `timescale_biases` | `attn.alpha_proj.bias`, `attn.forget_proj.bias` (only with `no_decay_timescale_biases=True`, the default; otherwise they stay in `default`) | 0 | `lr` |
| `lambda` | `attn.lambda_raw` | 0 | `lambda_lr`, which defaults to `lr` |

Recipe used to train the original models at each preset size:

| | `sisa-50m` | `sisa-152m` | `sisa-369m` |
|---|---|---|---|
| Tokens per optimizer step | 524,288 (256 × 2,048) | 524,288 | 524,288 |
| Peak learning rate | 6e-4 | 6e-4 | 3e-4 |
| Warm-up steps | 500 | 500 | 2,000 |
| Schedule | linear warm-up from 0 (factor step/warmup), cosine to 0 | same | same |
| AdamW | β = (0.9, 0.95), weight decay 0.1, fused | same | same |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| λ learning rate / weight decay | peak LR / 0 | peak LR / 0 | peak LR / 0 |
| Weight decay on timescale biases | 0.1 (`no_decay_timescale_biases=False`) | 0 | 0 |
| Training tokens | 10B: 19,074 optimizer steps, cosine horizon 19,073 | same | same |
| Precision | fp32 master weights, bf16 autocast | same | same |
| Data | FineWeb-Edu, GPT-NeoX tokenizer, documents joined with EOS (id 0) | same | same |

Notes:

- **Precision.** Train with fp32 parameters and bf16 autocast. `lambda_raw`
  always stays an fp32 parameter: after `model.to(torch.bfloat16)` it keeps
  its full-precision value and keeps training (the bf16 spacing near
  λ_raw ≈ −1 is 0.004–0.008, far coarser than typical updates). All other
  weights are then bf16, and training in that setting is not validated.
- **Packing.** There is no attention mask and no per-document reset: g, F and
  Φ accumulate across the whole row. Documents are concatenated with an EOS
  token between them, as in the original training.
- **Determinism.** The Triton kernels use no atomics, so two identical runs
  give bitwise-identical results. The backends differ from each other at
  rounding level, so do not switch backends partway through a run if you need
  bitwise reproducibility.
- **Initialization order.** The initial values follow the original code, but
  the order of RNG draws does not. A given seed does not reproduce the original
  initial weights bitwise.

Options for memory and speed:

- `model(ids, labels=ids, fused_ce=True)` (CUDA with Triton) computes the loss
  with the chunked LM-head cross-entropy of `sisa.kernels.linear_ce` and never
  builds the (B, L, V) logits; `logits` is then `None`. This saves the most
  memory. Without `labels`, `fused_ce` has no effect.
- `model(ids, labels=ids, return_logits=False)` returns only the loss. On its
  own this saves little memory, because the eager loss still builds the
  logits and an fp32 copy; it pays off after `compile_common()`.
- `SISAConfig(gradient_checkpointing=True)` recomputes each block in the
  backward pass.
- `model.compile_common()` applies `torch.compile` in place to the norms, the
  SwiGLU MLPs and the logits-free loss used with `return_logits=False` (the
  default call computes its loss eagerly from the returned logits);
  state-dict keys do not change. The attention sublayers stay eager.
- `model.set_attn_impl("triton")` switches the backend of an existing model.

## Backends

All backends compute the same function. They differ only at rounding level,
not bitwise. Pick one for a model with `SISAConfig(attn_impl=...)` or
`model.set_attn_impl(...)`. The standalone `sisa_attention` takes
`impl="auto"`, `"torch"` or `"triton"`; `triton_fused` also fuses the input
projections, so it exists only at the layer level.

| `attn_impl` | What runs | Requirements |
|---|---|---|
| `torch` | Exact blocked PyTorch reference. Computes in fp32 (also under autocast), builds full causal score rows for one block of queries at a time (memory O(block_q · L) per head), and casts the output back to the input dtype. Slow; meant for CPU, testing and unsupported GPUs. | Any device. Any d_head, any even d_state. |
| `triton` | PyTorch projections and cumulative sums, plus the Triton attention kernel (forward, dq and dk/dv kernels). | CUDA, compute capability ≥ 8.0, Triton ≥ 3.3, ≥ 99 KiB opt-in shared memory, d_head = 64, d_state = 32. Inputs in bf16, fp16 or fp32. |
| `triton_fused` | One GEMM for the seven per-token projections, then fused Triton prep (softplus, log-sigmoid, cumulative sums, pair rotations) and attention. | Everything `triton` needs, plus a bf16 or fp16 projection, i.e. bf16 autocast or bf16 parameters. |
| `auto` (default) | Picks `triton_fused` if eligible, then `triton`, then `torch`. | — |

Forcing `triton` or `triton_fused` on input they cannot handle raises
`ValueError`. When `auto` has to fall back to `torch` on a CUDA input, it
warns once per device and reason, both at the layer level and in the
standalone `sisa_attention`. To check eligibility
up front, use `sisa.backends.triton_status()` and
`sisa.backends.fused_eligible(x)`.

### Runtime checks

The kernels validate their inputs on the device. `sisa.set_check_mode(mode)`
controls how:

| Mode | Behaviour |
|---|---|
| `"async"` (default) | `torch._assert_async` on the device, with no host synchronization. A failure raises a device-side assert, which **leaves the CUDA context unusable**, so the process must be restarted. |
| `"sync"` | Checks on the host and raises a Python exception at the offending call. Costs one synchronization per layer call. Use it for debugging. |
| `"off"` | No checks. |

The checks are: g is non-increasing (tolerance 1e-3, unfused `triton`
path), and the scores and the decay, rotation and forget-gate projections are
finite (`triton_fused` path, where g is non-increasing by construction).

## Kernel notes

- **Exact, tiled.** Queries and keys are processed in T × T tiles: T = 64 for
  bf16/fp16 and T = 32 for fp32. fp32 inputs use IEEE dot products, never TF32.
  The kernels make no approximations: every causal pair is scored, and any
  length L ≥ 1 is handled with masked loads and no padding copy.
- **Chunk-factored decay.** Diagonal tiles compute exp(min(g_i − g_j, 0))
  pairwise. Off-diagonal tiles factor the decay, in the style of the Mamba-2
  SSD chunked algorithm, as

  ```
  exp(g_i − g_j) = exp(g_i − g_p) · exp(g_p − g_c) · exp(g_c − g_j)
  ```

  where g_p is the first g of the query tile and g_c is the last g of the key
  tile. Each factor is at most 1. The GEMM runs on the original C̃, B̃ values,
  and the fp32 row and column factors are applied after it. The error
  therefore scales with |g_i − g_j|, as in a pairwise evaluation.
- **Shift invariance.** Differences of g and F are taken in natural-log units
  before the conversion to base 2. Errors therefore scale with |g_i − g_j| and
  |F_i − F_j|, not with |g| or |F|.
- **Exact Δ in the backward pass.** The forward pass also stores an fp32,
  self-normalized copy of the output. The backward pass uses it for the
  softmax correction Δ_i = do_i·o_i, instead of the low-precision output used
  by the standard FlashAttention backward. Diagonal entries are handled so
  that gradients of strongly forgetting heads stay accurate in fp32. This costs
  4 extra bytes per output element, and only when gradients are needed.
- **Precision.** g, F and Φ are computed and consumed in fp32. C̃ and B̃ are
  cast to the dtype of q. λ, dλ and the forget-gate logits are fp32. The forget
  projection runs with autocast disabled.
- **Deterministic.** There are no atomics. dq and dC̃ are owned by query
  tiles; dk, dv, dB̃, dg and dF are finished by key tiles; and dλ is reduced
  from a per-row buffer.
- **Launch settings.** The tile configuration depends on shared memory, which
  is read from Triton's driver: one table for ≥ 200 KiB (H100) and one for
  everything from 99 KiB up (for example A6000, RTX 4090 and Blackwell
  workstation cards). There is no autotuning. The fused projection is
  padded to a multiple of 16 columns (273·H → next multiple of 16).
- **Accuracy check.** Before packaging, every backend was compared with an
  fp64 reference on trained 50M and 152M models, next to the earlier kernel
  that trained the original checkpoints, on the same inputs:
  - Logits, bf16 autocast with fp32 or bf16 weights, text and random tokens,
    L from 63 to 2,048, at least 24 blocks per configuration: the mean
    relative error of every backend is 0.99–1.00× that of the earlier kernel.
  - fp32 without autocast: logits within about 1e-5 of fp64 (the earlier
    kernel used TF32 there, about 3e-3).
  - Held-out cross-entropy of the trained 50M model with the Triton
    backends, 2,048 sequences of 2,049 tokens: within 1e-5 nats of the
    earlier kernel.
  - Triton kernel gradients on identical activations of every layer of the
    trained 50M model, with the true upstream gradient: closer to fp64 than
    the earlier kernel's for every input tensor (dg, dλ and dF about 3e-7
    relative error, against about 3e-3).

  Per-input tolerances such as loss |Δ| ≤ max(2·Δ_ref, 1e-3) nats or a
  per-parameter-group gradient error ≤ 1.25·e_ref + 1e-3 are dominated by
  bf16 rounding. Single inputs, especially short blocks, exceed them by chance
  for every backend, the earlier kernel included, so judge accuracy over many
  blocks or at the kernel level on identical inputs.
- **Hardware validated:** H100 PCIe (sm90), RTX A6000 (sm86), RTX 4090
  (sm89), RTX PRO 5000 Blackwell (sm120). **Untested:** A100 (sm80), H100
  SXM / H200, sm100. **Not supported by the kernels:** GPUs older than sm80,
  ROCm, MPS. Those use the `torch` backend.
- **Software validated:** torch 2.7.1 + Triton 3.3.1, torch 2.9.0 + Triton
  3.5.0, torch 2.11 + Triton 3.6.0, torch 2.12.1 + Triton 3.7.1, and torch
  2.14.1 + Triton 3.8.0. Triton 3.1 failed one fp64 gate group, so the
  minimum is 3.3. Triton 3.6 was about 15% faster than 3.3.1 on an A6000.
  Run the CUDA tests on your own setup before relying on it.

**Relation to the original checkpoints.** The original models were trained
and evaluated with an earlier Triton kernel that computes the same function.
That kernel is not shipped. Every backend here agrees with it at bf16
rounding level, not bitwise, and training trajectories will slowly diverge
from it.

## Model sizes

| Preset | d_model | n_layer | n_head | d_head | d_state | d_ff (SwiGLU) | Parameters |
|---|---|---|---|---|---|---|---|
| `sisa-50m` | 512 | 6 | 8 | 64 | 32 | 1,829 | 50,911,376 |
| `sisa-152m` | 768 | 12 | 12 | 64 | 32 | 2,744 | 151,878,576 |
| `sisa-369m` | 1024 | 24 | 16 | 64 | 32 | 2,507 | 369,278,080 |

- All presets use vocabulary 50,277 (the GPT-NeoX tokenizer, `len(tokenizer)`)
  and were trained at context 2,048.
- Parameter counts include the tied embedding once.
- `d_ff` is the actual MLP width. It was chosen so that each total matches a
  Transformer of the same shape (MLP width 2,048, 3,072 and 2,944) to within
  0.01%.
- Per layer, the count is 4d² + 2·H·d_s·d + H·(d_s/2)·d + 2(H·d + H) + H + 2d
  + 3·d·d_ff. The model adds V·d + d once.

## Converting checkpoints

**No weights are distributed.** `sisa.checkpoint` converts checkpoints that
you trained with the original research code:

```python
from sisa import load_paper_checkpoint, SISAForCausalLM

model = load_paper_checkpoint("path/to/final.pt")    # keeps only the model weights
model.save_pretrained("sisa-50m-converted")          # config.json + model.safetensors
model = SISAForCausalLM.from_pretrained("sisa-50m-converted")
```

- The file is loaded with `weights_only=True` and memory mapping, and only
  `ckpt["model"]` is read. The optimizer state, scheduler and the stored
  training configuration are discarded, never copied.
- The configuration is inferred from tensor shapes. d_ff comes from the
  `w_gate` rows, not from the saved config, whose `d_ff` field is the matched
  Transformer width.
- The converter strips the `_orig_mod.` infixes that per-submodule
  `torch.compile` leaves behind, and it moves the attention parameters under
  `layers.{i}.attn.`. Loading is strict: the key set and every shape must match.
- For a state dict you already hold in memory, use
  `model.load_state_dict(convert_state_dict(state_dict))`.

Both loaders return the model on CPU in eval mode; move it with
`.to(device)`.

The original evaluation cast the whole model to bf16 (`model.to(torch.bfloat16)`)
and ran under bf16 autocast. That works here too, as do fp32 weights with
autocast. After the cast, `lambda_raw` keeps its fp32 value here, while the
original code rounded it to bf16, a difference at bf16 rounding level.

## Limitations

- The Triton kernels support only d_head = 64 and d_state = 32. The `torch`
  backend accepts any d_head and any even d_state.
- There is no attention mask, padding mask or variable-length reset: the
  cumulative sums run over the whole row. Left padding is not supported.
  Batched generation works only with prompts of equal length.
- Decoding is plain PyTorch. It has not been optimized or benchmarked. The
  cache stores B̃, g and F next to K and V, which is d_s + 2 = 34 extra values
  per head per token (+27% over K and V at d_h = 64).
- The original models were trained at 2,048 tokens. There is no positional
  table, so longer inputs run, but this package makes no claim about quality
  beyond 2,048. Φ and g are fp32 cumulative sums that grow with L, and
  numerics have been validated up to about 4,096 tokens.
- fp16 works in the kernels but has not been validated at the model level.
  Training used bf16 autocast.
- DDP should work, since the kernels are plain `autograd.Function`s, but it
  is untested. FSDP, tensor parallelism, CUDA graphs and whole-model
  `torch.compile` are not supported in 0.1.0. The Triton autograd functions
  cause graph breaks.
- Vocabulary 50,277 is not a multiple of 64, which slows the LM-head GEMM.
  The optional `linear_ce` kernel pads to 50,304 internally and never stores
  the padding.
- Original training cannot be reproduced bitwise: the attention kernel and
  the initialization order both differ.
- A failed check in `"async"` mode leaves the CUDA context unusable (see
  [Runtime checks](#runtime-checks)).

## Tests

```bash
pip install -e ".[test]"
pytest -m "not slow"          # CPU tests; CUDA tests run if a supported GPU is present
pytest -m cuda                # kernel tests against fp64 references (needs sm80+ and Triton)
pytest -m slow                # exhaustive kernel grids (long)
```

The CUDA tests skip automatically when no supported GPU or Triton is found.
They need about 2.5 GB of GPU memory. On a shared GPU you can cap the test
process with `SISA_TEST_CUDA_MEM_FRACTION=0.1` (a fraction of the device's
memory).
The device-side guard tests run in subprocesses, because a failed
`torch._assert_async` makes the CUDA context unusable.

## Citation

To be added after publication.

## License

To be decided. See [NOTICE](NOTICE) for credits to the methods this work
builds on.

## Acknowledgements

The forget gate follows the Forgetting Transformer (FoX), the chunk-factored
decay follows the Mamba-2 SSD algorithm, the data-dependent rotation of the
state-space term follows Mamba-3, the kernel structure follows
FlashAttention-2, and the optional chunked cross-entropy follows Liger
Kernel. See [NOTICE](NOTICE) for details.
