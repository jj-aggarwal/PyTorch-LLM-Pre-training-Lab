"""Validation-loss estimation and a CLI that evaluates a saved checkpoint."""

from __future__ import annotations

import argparse
import math
from contextlib import AbstractContextManager, nullcontext

import torch

from .checkpoint import build_model_from_checkpoint, load_checkpoint
from .config import ConfigError, ExperimentConfig
from .data import CharTokenizer, iter_eval_batches, prepare_data
from .model import GPT
from .utils import autocast_context, resolve_device


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data: torch.Tensor,
    *,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    autocast: AbstractContextManager | None = None,
) -> float:
    """Mean per-window cross-entropy over deterministic, non-overlapping windows."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for x, y in iter_eval_batches(data, batch_size, model.cfg.block_size, max_batches):
            with autocast or nullcontext():
                _, loss = model(x.to(device), y.to(device))
            total += float(loss) * x.size(0)
            count += x.size(0)
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("no evaluation windows available; the split is too small")
    return total / count


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 50.0))  # clamp to avoid overflow on diverged runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on its train/val split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)

    try:
        device = resolve_device(args.device)
        ckpt = load_checkpoint(args.checkpoint)
        cfg = ExperimentConfig.from_dict(ckpt["config"])
        model = build_model_from_checkpoint(ckpt).to(device)
        tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
        _, train_data, val_data = prepare_data(cfg.data, cfg.model.block_size, tokenizer)
        data = val_data if args.split == "val" else train_data
        loss = estimate_loss(
            model,
            data,
            batch_size=args.batch_size,
            max_batches=args.batches,
            device=device,
            autocast=autocast_context(device, "fp32"),
        )
    except (ConfigError, FileNotFoundError, ValueError) as err:
        print(f"error: {err}")
        return 1
    print(
        f"checkpoint step {ckpt['step']} | {args.split} loss {loss:.4f} | "
        f"perplexity {perplexity(loss):.3f}"
    )
    return 0
