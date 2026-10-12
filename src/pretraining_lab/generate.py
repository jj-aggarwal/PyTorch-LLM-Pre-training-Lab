"""Autoregressive text generation with temperature and top-k sampling, plus a CLI."""

from __future__ import annotations

import argparse
import math

import torch

from .checkpoint import build_model_from_checkpoint, load_checkpoint
from .config import ConfigError
from .data import CharTokenizer
from .model import GPT
from .utils import resolve_device


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend ``idx`` (B, T) by ``max_new_tokens`` tokens.

    ``temperature == 0`` means greedy decoding. ``top_k`` keeps only the k highest-logit tokens
    before sampling. The context is cropped to the model's ``block_size`` at every step.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(f"temperature must be a finite number >= 0, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 when given, got {top_k}")
    if idx.dim() != 2 or idx.size(1) == 0:
        raise ValueError("idx must have shape (batch, seq_len) with seq_len >= 1")

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -model.cfg.block_size :])
            logits = logits[:, -1, :].float()
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
    finally:
        model.train(was_training)
    return idx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)

    try:
        device = resolve_device(args.device)
        ckpt = load_checkpoint(args.checkpoint)
        model = build_model_from_checkpoint(ckpt).to(device)
        tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
        prompt_ids = (
            tokenizer.encode(args.prompt) if args.prompt else torch.zeros(1, dtype=torch.long)
        )
        idx = prompt_ids.unsqueeze(0).to(device)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        out = generate(model, idx, args.max_new_tokens, args.temperature, args.top_k, gen)
    except (ConfigError, FileNotFoundError, ValueError) as err:
        print(f"error: {err}")
        return 1
    print(tokenizer.decode(out[0]))
    return 0
