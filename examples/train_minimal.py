"""Minimal training loop for SISA on one device.

This is the recipe used to train the preset models, scaled down: AdamW over
``model.param_groups`` (``lambda_raw`` is not decayed and uses the base
learning rate), betas (0.9, 0.95), weight decay 0.1, gradients clipped to
norm 1.0, linear warm-up from zero (factor ``step / warmup``, so the first step
has learning rate 0) then cosine decay to zero, fp32 master weights and bf16
autocast on CUDA. The sisa-50m runs also decayed the timescale biases
(``alpha_proj.bias``, ``forget_proj.bias``); the larger sizes did not. The
preset sets this, and ``--[no-]decay-timescale-biases`` overrides it.

On CUDA with the Triton kernels the loss uses the chunked LM-head
cross-entropy (``fused_ce=True``), which never builds the (batch, seq-len,
vocab) logits. With full logits (``--no-fused-ce``) the sisa-50m command
below needs more memory than a 24 GB card has; lower ``--batch-size`` and
raise ``--grad-accum`` in that case.

Data
----
Without ``--data`` the script trains on a synthetic token stream drawn from a
random sparse Markov chain. The best achievable loss (at most
ln(branching)) is printed at start-up, so you can watch the loss approach it.

With ``--data`` it reads a flat file of token ids, either ``.npy`` (any
integer dtype) or raw binary (``--data-dtype``, default uint16), through a
memory map. Documents should be concatenated with an EOS token between them.
SISA uses no attention mask, so packed sequences need no extra handling.
Reading files needs numpy.

Labels are shifted inside the model, so every batch is passed as both
``input_ids`` and ``labels``. A block of ``--seq-len`` tokens trains
``seq-len - 1`` predictions; use ``--seq-len 2049`` to match the original
2,048 predictions per sequence.

Examples::

    # tiny model on synthetic data, well under a minute on CPU
    python examples/train_minimal.py --device cpu

    # sisa-50m on your own GPT-NeoX token file (524,288 predicted tokens per step, 10B tokens)
    python examples/train_minimal.py --preset sisa-50m --data tokens.bin \\
        --seq-len 2049 --batch-size 16 --grad-accum 16 --steps 19073 --out sisa-50m

The original runs used the same 19,073-step schedule but stopped after
19,074 optimizer steps; the extra step has learning rate 0 and leaves the
weights unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import math
import time

import torch

from sisa import SISAConfig, SISAForCausalLM, backends

# Peak learning rate, warm-up steps and weight decay on the timescale biases used for the preset sizes.
RECIPES = {
    "sisa-50m": {"lr": 6e-4, "warmup": 500, "decay_timescale_biases": True},
    "sisa-152m": {"lr": 6e-4, "warmup": 500, "decay_timescale_biases": False},
    "sisa-369m": {"lr": 3e-4, "warmup": 2000, "decay_timescale_biases": False},
}
# A small model on synthetic data tolerates a much larger learning rate.
TINY_RECIPE = {"lr": 3e-3, "warmup": 20, "decay_timescale_biases": False}
GPT_NEOX_VOCAB = 50277


# ----------------------------------------------------------------------------- data
class TokenSource:
    """Random access to a 1-D token stream held as a torch tensor or numpy memmap."""

    def __init__(self, data):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def window(self, start: int, length: int) -> torch.Tensor:
        chunk = self.data[start:start + length]
        if isinstance(chunk, torch.Tensor):
            return chunk.long()
        return torch.from_numpy(chunk.astype("int64"))


def load_token_file(path: str, dtype: str) -> TokenSource:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit("reading --data needs numpy (pip install numpy)") from exc
    if path.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
    else:
        arr = np.memmap(path, dtype=np.dtype(dtype), mode="r")
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return TokenSource(arr)


def markov_stream(vocab_size: int, n_tokens: int, branching: int,
                  gen: torch.Generator) -> torch.Tensor:
    """Sample a random sparse Markov chain: each token has `branching` successors.

    Successors are drawn uniformly, so the entropy rate is at most
    ln(branching) nats per token (less when a successor is drawn twice).
    """
    successors = torch.randint(0, vocab_size, (vocab_size, branching), generator=gen)
    n_chains = 64
    length = math.ceil(n_tokens / n_chains)
    out = torch.empty(n_chains, length, dtype=torch.long)
    cur = torch.randint(0, vocab_size, (n_chains,), generator=gen)
    for t in range(length):
        out[:, t] = cur
        pick = torch.randint(0, branching, (n_chains,), generator=gen)
        cur = successors[cur, pick]
    return out.reshape(-1)[:n_tokens]


def sample_batch(src: TokenSource, lo: int, hi: int, batch_size: int, seq_len: int,
                 gen: torch.Generator) -> torch.Tensor:
    """Draw `batch_size` random windows of `seq_len` tokens from src[lo:hi]."""
    starts = torch.randint(lo, hi - seq_len + 1, (batch_size,), generator=gen)
    return torch.stack([src.window(int(s), seq_len) for s in starts])


# ----------------------------------------------------------------------------- training
def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warm-up from zero to the peak learning rate, then cosine decay to zero."""
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model: SISAForCausalLM, batches: list[torch.Tensor], device: torch.device,
             fused_ce: bool = False) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    for batch in batches:
        batch = batch.to(device)
        with autocast(device):
            total += model(batch, labels=batch, fused_ce=fused_ce).loss.float()
    model.train()
    return (total / len(batches)).item()


def mean_lambda(model: torch.nn.Module) -> float:
    """Mean of lambda = softplus(lambda_raw) over all heads and layers."""
    vals = [torch.nn.functional.softplus(p.detach().float())
            for name, p in model.named_parameters() if name.endswith("lambda_raw")]
    return torch.cat(vals).mean().item() if vals else float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default=None, choices=sorted(RECIPES),
                   help="model size (default: a tiny CPU-friendly config)")
    p.add_argument("--data", default=None,
                   help="token file (.npy or raw binary); synthetic data if omitted")
    p.add_argument("--data-dtype", default="uint16",
                   choices=["uint16", "uint32", "int32", "int64"],
                   help="element type of a raw binary --data file")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="vocabulary size (default: 512 for synthetic data, "
                        f"{GPT_NEOX_VOCAB} for --data, preset value for --preset)")
    p.add_argument("--branching", type=int, default=4,
                   help="successors per token in the synthetic Markov chain")
    p.add_argument("--seq-len", type=int, default=128, help="tokens per sequence")
    p.add_argument("--batch-size", type=int, default=8, help="sequences per micro-batch")
    p.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    p.add_argument("--steps", type=int, default=150, help="optimizer steps")
    p.add_argument("--lr", type=float, default=None, help="peak learning rate")
    p.add_argument("--warmup", type=int, default=None, help="warm-up steps")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--decay-timescale-biases", action=argparse.BooleanOptionalAction, default=None,
                   help="apply --weight-decay to alpha_proj.bias and forget_proj.bias "
                        "(default: the preset's recipe, on for sisa-50m only)")
    p.add_argument("--fused-ce", action=argparse.BooleanOptionalAction, default=None,
                   help="chunked LM-head cross-entropy without full logits "
                        "(default: on when the Triton kernels are available)")
    p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping")
    p.add_argument("--attn-impl", default="auto",
                   choices=["auto", "torch", "triton", "triton_fused"])
    p.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--out", default=None, help="save_pretrained directory for the final model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    print(backends.describe())

    # Model ----------------------------------------------------------------------
    if args.preset:
        cfg = SISAConfig.preset(args.preset)
        if args.vocab_size is not None:
            cfg = dataclasses.replace(cfg, vocab_size=args.vocab_size)
        recipe = RECIPES[args.preset]
    else:
        vocab = args.vocab_size or (GPT_NEOX_VOCAB if args.data else 512)
        cfg = SISAConfig(vocab_size=vocab, d_model=128, n_layer=2, n_head=2, d_ff=256)
        recipe = TINY_RECIPE
    cfg = dataclasses.replace(cfg, attn_impl=args.attn_impl)
    lr = args.lr if args.lr is not None else recipe["lr"]
    warmup = args.warmup if args.warmup is not None else recipe["warmup"]
    warmup = max(1, min(warmup, args.steps))
    decay_tsb = (args.decay_timescale_biases if args.decay_timescale_biases is not None
                 else recipe["decay_timescale_biases"])
    fused_ce = args.fused_ce
    if fused_ce is None:
        fused_ce = device.type == "cuda" and backends.triton_status(device)[0]
    if fused_ce and device.type != "cuda":
        raise SystemExit("--fused-ce needs a CUDA device")

    model = SISAForCausalLM(cfg).to(device)          # fp32 master weights
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.preset or 'tiny'}, {n_params:,} parameters, device {device}")

    # Data -------------------------------------------------------------------------
    if args.data:
        src = load_token_file(args.data, args.data_dtype)
        print(f"data: {args.data}, {len(src):,} tokens")
    else:
        n_tokens = max(262_144, 4 * args.batch_size * args.seq_len)
        src = TokenSource(markov_stream(cfg.vocab_size, n_tokens, args.branching, gen))
        print(f"data: synthetic Markov chain, {len(src):,} tokens, "
              f"best achievable loss <= ln({args.branching}) = {math.log(args.branching):.3f}")
    n_val = max(len(src) // 20, args.seq_len)
    n_train = len(src) - n_val
    if n_train < args.seq_len:
        raise SystemExit(f"need more than {args.seq_len + n_val} tokens, got {len(src)}")
    val_batches = [sample_batch(src, n_train, len(src), args.batch_size, args.seq_len, gen)
                   for _ in range(args.eval_batches)]

    # Optimizer and schedule -------------------------------------------------------
    opt = torch.optim.AdamW(
        model.param_groups(lr, weight_decay=args.weight_decay, no_decay_timescale_biases=not decay_tsb),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: lr_factor(step, warmup, args.steps))

    # Loop -------------------------------------------------------------------------
    tokens_per_step = args.batch_size * args.grad_accum * (args.seq_len - 1)   # predicted tokens
    print(f"training: {args.steps} steps x {tokens_per_step:,} predicted tokens, peak lr {lr:g}, warm-up {warmup}, "
          f"timescale-bias decay {'on' if decay_tsb else 'off'}, fused CE {'on' if fused_ce else 'off'}")
    model.train()
    loss_acc = torch.zeros((), device=device)
    t_last, steps_since = time.perf_counter(), 0
    for step in range(args.steps):
        for _ in range(args.grad_accum):
            batch = sample_batch(src, 0, n_train, args.batch_size, args.seq_len, gen)
            if int(batch.max()) >= cfg.vocab_size:
                raise SystemExit(f"token id {int(batch.max())} >= vocab_size {cfg.vocab_size}")
            batch = batch.to(device, non_blocking=True)
            with autocast(device):
                loss = model(batch, labels=batch, fused_ce=fused_ce).loss
            (loss / args.grad_accum).backward()
            loss_acc += loss.detach().float() / args.grad_accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        steps_since += 1

        last = step + 1 == args.steps
        if (step + 1) % args.log_every == 0 or step == 0 or last:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            tok_s = steps_since * tokens_per_step / (now - t_last)  # this interval only
            print(f"step {step + 1:6d} | loss {loss_acc.item() / steps_since:.4f} "
                  f"| lr {sched.get_last_lr()[0]:.2e} | grad norm {grad_norm.item():.3f} "
                  f"| lambda {mean_lambda(model):.4f} | {tok_s:,.0f} tok/s")
            loss_acc.zero_()
            t_last, steps_since = time.perf_counter(), 0
        if (step + 1) % args.eval_every == 0 or last:
            t_eval = time.perf_counter()
            print(f"step {step + 1:6d} | val loss {evaluate(model, val_batches, device, fused_ce):.4f}")
            t_last += time.perf_counter() - t_eval     # keep eval time out of tok/s

    if args.out:
        model.save_pretrained(args.out)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
