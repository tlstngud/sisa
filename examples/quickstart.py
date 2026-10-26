"""A short tour of the ``sisa`` package.

The script

1. prints which attention backends are available,
2. calls the attention function directly and compares the selected backend
   with the pure-PyTorch reference,
3. builds a model, runs a forward/backward pass with labels,
4. generates a few tokens greedily, and
5. round-trips the weights through ``save_pretrained`` / ``from_pretrained``.

Usage::

    python examples/quickstart.py                      # tiny model, CPU or GPU
    python examples/quickstart.py --device cpu
    python examples/quickstart.py --preset sisa-50m --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import math
import tempfile

import torch
import torch.nn.functional as F

import sisa
from sisa import SISAConfig, SISAForCausalLM, backends, sisa_attention
from sisa.functional import rotate_pairs


def tiny_config() -> SISAConfig:
    """A small config that runs quickly on CPU.

    ``d_model / n_head = 64`` and ``d_state = 32`` (the default) keep it
    compatible with the Triton kernels, so the same config also exercises
    the GPU backends.
    """
    return SISAConfig(vocab_size=1024, d_model=128, n_layer=2, n_head=2, d_ff=256)


def autocast(device: torch.device):
    """bf16 autocast on CUDA (the training precision); fp32 elsewhere."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def functional_demo(device: torch.device) -> None:
    """Call ``sisa_attention`` on random inputs that satisfy its contract."""
    B, H, L, D, S = 1, 2, 96, 64, 32
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    gen = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(device)

    q, k, v = (randn(B, H, L, D).to(dtype) for _ in range(3))
    # C and B are rotated by the cumulative phase Phi = cumsum(theta) before
    # the kernel sees them; the rotation itself runs in fp32.
    phi = torch.cumsum(0.1 * randn(B, H, L, S // 2), dim=2)
    c = rotate_pairs(randn(B, H, L, S), phi).to(dtype)
    b = rotate_pairs(randn(B, H, L, S), phi).to(dtype)
    # g: cumulative log-decay, F: cumulative log forget gate. Both are fp32
    # and non-increasing along the sequence.
    g = -torch.cumsum(F.softplus(randn(B, H, L) - 3.0), dim=-1)
    fgate = torch.cumsum(F.logsigmoid(randn(B, H, L) + 4.0), dim=-1)
    lam = F.softplus(torch.full((H,), -1.0, device=device))

    o = sisa_attention(q, k, v, c, b, g, lam, fgate)                 # impl="auto"
    o_ref = sisa_attention(q, k, v, c, b, g, lam, fgate, impl="torch")
    err = (o.float() - o_ref.float()).abs().max().item()
    print(f"[functional] o: {tuple(o.shape)} {o.dtype}; max |auto - torch| = {err:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SISA quickstart")
    parser.add_argument("--preset", default=None, choices=sorted(sisa.PRESETS),
                        help="model size (default: a tiny config)")
    parser.add_argument("--device", default=None,
                        help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--attn-impl", default="auto",
                        choices=["auto", "torch", "triton", "triton_fused"])
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)

    print(f"sisa {sisa.__version__}")
    print(backends.describe())

    functional_demo(device)

    # Model ------------------------------------------------------------------
    cfg = SISAConfig.preset(args.preset) if args.preset else tiny_config()
    cfg = dataclasses.replace(cfg, attn_impl=args.attn_impl)
    model = SISAForCausalLM(cfg).to(device)
    # model.parameters() yields the tied embedding once.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.preset or 'tiny'}: {n_params:,} parameters on {device}")

    ids = torch.randint(0, cfg.vocab_size, (2, args.seq_len), device=device)

    # Labels are shifted inside the model (Hugging Face convention), so the
    # same tensor is passed as input_ids and labels.
    model.train()
    with autocast(device):
        out = model(ids, labels=ids)
    out.loss.backward()
    grad_dtypes = {p.grad.dtype for p in model.parameters() if p.grad is not None}
    print(f"[train] loss {out.loss.item():.4f} (ln V = {math.log(cfg.vocab_size):.4f}); "
          f"logits {tuple(out.logits.shape)}; grad dtypes {sorted(map(str, grad_dtypes))}")
    model.zero_grad(set_to_none=True)

    # Generation ---------------------------------------------------------------
    model.eval()
    prompt = ids[:1, :16]
    with torch.inference_mode(), autocast(device):
        seq = model.generate(prompt, max_new_tokens=16)              # greedy
    print(f"[generate] prompt {prompt[0].tolist()}")
    print(f"[generate] continuation {seq[0, prompt.shape[1]:].tolist()}")

    # Save / load ----------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded = SISAForCausalLM.from_pretrained(tmp).to(device).eval()
        with torch.inference_mode(), autocast(device):
            a = model(ids).logits.float()
            b = reloaded(ids).logits.float()
    print(f"[save/load] max |logits difference| after reload = {(a - b).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
