"""Generate text, or token ids, with a SISA model.

Usage::

    # weights saved with save_pretrained (for example a converted checkpoint)
    python examples/generate.py --model path/to/model_dir --prompt "The ocean is"

    # sampling several continuations of the same prompt
    python examples/generate.py --model path/to/model_dir --prompt "Once upon a time" \\
        --sample --temperature 0.8 --top-k 50 --num-samples 4

    # untrained tiny model, only to exercise the decode path
    python examples/generate.py --prompt-ids 0,1,2 --max-new-tokens 16 --device cpu

When the optional ``tokenizers`` package is installed, prompts are encoded
with the GPT-NeoX-20B tokenizer that the presets use. It is downloaded from the
Hugging Face Hub on first use, or read from a local ``tokenizer.json`` given
with ``--tokenizer``. Without a tokenizer, pass ``--prompt-ids`` and the
script prints token ids.

Training joined documents with an EOS token (``<|endoftext|>``, id 0), so
every document started right after an EOS. Text prompts are therefore
prefixed with that token, so that they look like the start of a training
document; ``--no-eos-prefix`` turns this off. ``--prompt-ids`` are used
exactly as given.

Decoding uses a pure-PyTorch incremental cache. It is not optimized, and
batched decoding needs prompts of equal length (the same prompt is repeated
for ``--num-samples``).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time

import torch

from sisa import SISAConfig, SISAForCausalLM

DEFAULT_TOKENIZER = "EleutherAI/gpt-neox-20b"
EOS_TOKEN = "<|endoftext|>"


def load_tokenizer(name_or_path: str):
    """Return a `tokenizers.Tokenizer`, or None if unavailable."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    try:
        if os.path.isfile(name_or_path):
            return Tokenizer.from_file(name_or_path)
        return Tokenizer.from_pretrained(name_or_path)
    except Exception as exc:  # offline, unknown name, ...
        print(f"warning: could not load tokenizer {name_or_path!r} ({exc}); using token ids")
        return None


def autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate with a SISA model",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--model", default=None,
                   help="directory written by save_pretrained (default: random weights)")
    p.add_argument("--preset", default=None,
                   help="preset for random weights when --model is not given (default: tiny)")
    p.add_argument("--prompt", default=None, help="prompt text (needs a tokenizer)")
    p.add_argument("--prompt-ids", default=None,
                   help="comma-separated prompt token ids, used without a tokenizer")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="tokenizer name on the Hugging Face Hub, or a tokenizer.json path")
    p.add_argument("--no-tokenizer", action="store_true", help="work with token ids only")
    p.add_argument("--no-eos-prefix", action="store_true",
                   help=f"do not prepend the document separator {EOS_TOKEN} to a text prompt")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--eos-id", type=int, default=None,
                   help=f"stop token (default: the tokenizer's {EOS_TOKEN} id)")
    p.add_argument("--ignore-eos", action="store_true", help="always generate --max-new-tokens")
    p.add_argument("--bf16", action="store_true",
                   help="cast the weights to bf16 (as the original evaluation did)")
    p.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    tok = None if args.no_tokenizer else load_tokenizer(args.tokenizer)

    # Model ----------------------------------------------------------------------
    if args.model:
        model = SISAForCausalLM.from_pretrained(args.model)
    else:
        if args.preset:
            cfg = SISAConfig.preset(args.preset)
        else:
            cfg = SISAConfig(vocab_size=50277, d_model=128, n_layer=2, n_head=2, d_ff=256)
        model = SISAForCausalLM(cfg)
        print("note: randomly initialized weights, so the output is noise")
    if args.bf16:
        model = model.to(torch.bfloat16)
    model = model.to(device).eval()
    vocab_size = model.config.vocab_size if hasattr(model, "config") else None

    # Prompt ---------------------------------------------------------------------
    if args.prompt_ids is not None:
        ids = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
    elif args.prompt is not None:
        if tok is None:
            raise SystemExit("--prompt needs the `tokenizers` package; use --prompt-ids instead")
        ids = tok.encode(args.prompt).ids
        eos = tok.token_to_id(EOS_TOKEN)
        if eos is not None and not args.no_eos_prefix:
            ids = [eos] + ids                   # documents followed an EOS in training
    else:
        eos = tok.token_to_id(EOS_TOKEN) if tok is not None else None
        ids = [eos if eos is not None else 0]   # start from the document separator
    if not ids:
        raise SystemExit("empty prompt")
    if vocab_size is not None and max(ids) >= vocab_size:
        raise SystemExit(f"prompt token id {max(ids)} >= model vocab_size {vocab_size}")

    eos_id = args.eos_id
    if eos_id is None and tok is not None:
        eos_id = tok.token_to_id(EOS_TOKEN)
    if args.ignore_eos:
        eos_id = None

    # Generate -------------------------------------------------------------------
    prompt = torch.tensor([ids] * args.num_samples, dtype=torch.long, device=device)
    t0 = time.perf_counter()
    with torch.inference_mode(), autocast(device):
        out = model.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=eos_id,
            do_sample=args.sample,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # generate() returns the prompt followed by the new tokens.
    print(f"prompt: {args.prompt if args.prompt is not None and tok is not None else ids}")
    n_new = 0
    for i, row in enumerate(out.tolist()):
        new = row[len(ids):]
        if eos_id is not None and eos_id in new:
            new = new[:new.index(eos_id)]
        n_new += len(new)
        text = tok.decode(new) if tok is not None else new
        prefix = f"[{i}] " if args.num_samples > 1 else ""
        print(f"{prefix}{text}")
    print(f"\n{n_new} new tokens in {elapsed:.2f} s (unoptimized PyTorch decode)")


if __name__ == "__main__":
    main()
