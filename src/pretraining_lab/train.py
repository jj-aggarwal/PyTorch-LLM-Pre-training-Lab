"""Training loop: AdamW, LR warmup + decay, gradient accumulation/clipping, checkpoint/resume."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from .config import ConfigError, ExperimentConfig, TrainConfig
from .data import CharTokenizer, get_batch, prepare_data
from .evaluate import estimate_loss
from .model import GPT
from .utils import (
    autocast_context,
    peak_memory,
    reset_peak_memory,
    resolve_device,
    resolve_precision,
    set_seed,
    synchronize,
)


class TrainingInstabilityError(RuntimeError):
    """Raised when the loss or gradient norm becomes NaN/Inf."""


@dataclass
class TrainResult:
    num_parameters: int
    device: str
    precision: str
    resumed_from_step: int
    steps_completed: int
    tokens_per_step: int
    tokens_seen_this_run: int
    train_time_sec: float
    tokens_per_sec: float | None  # median over steps, first (warm-up) step excluded if >1 step
    peak_memory_bytes: int | None
    peak_memory_kind: str | None
    final_train_loss: float | None
    final_val_loss: float | None
    best_val_loss: float | None
    checkpoint_path: str | None
    notes: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    eval_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def lr_at_step(step: int, tc: TrainConfig) -> float:
    """Learning rate for the 0-based optimizer ``step``: linear warmup, then decay to min_lr."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    if tc.scheduler == "constant":
        return tc.lr
    span = max(1, tc.max_steps - tc.warmup_steps)
    progress = min(1.0, (step - tc.warmup_steps) / span)
    if tc.scheduler == "cosine":
        return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1.0 + math.cos(math.pi * progress))
    return tc.lr + (tc.min_lr - tc.lr) * progress  # linear


def build_optimizer(model: GPT, tc: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only (not biases or LayerNorm parameters)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": tc.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2))


def train(
    cfg: ExperimentConfig,
    resume_from: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> TrainResult:
    cfg.validate()
    tc = cfg.train
    set_seed(tc.seed)
    device = resolve_device(tc.device)
    precision, note = resolve_precision(tc.precision, device)
    notes = [note] if note else []
    if note:
        log(f"warning: {note}")

    ckpt = load_checkpoint(resume_from) if resume_from else None
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"]) if ckpt else None
    tokenizer, train_data, val_data = prepare_data(cfg.data, cfg.model.block_size, tokenizer)

    if cfg.model.vocab_size not in (None, tokenizer.vocab_size):
        raise ConfigError(
            f"model.vocab_size={cfg.model.vocab_size} does not match the tokenizer "
            f"({tokenizer.vocab_size}); leave it null to infer it"
        )
    model_cfg = dataclasses.replace(cfg.model, vocab_size=tokenizer.vocab_size)
    run_cfg = ExperimentConfig(model=model_cfg, data=cfg.data, train=tc)
    if ckpt and ckpt["config"]["model"] != model_cfg.__dict__:
        raise ConfigError("model config differs from the checkpoint being resumed")

    model = GPT(model_cfg).to(device)
    optimizer = build_optimizer(model, tc)
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    data_gen = torch.Generator().manual_seed(tc.seed)
    start_step = 0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        data_gen.set_state(ckpt["data_rng_state"])
        restore_rng_state(ckpt["rng"])
        start_step = int(ckpt["step"])
        log(f"resumed from {resume_from} at step {start_step}")

    n_params = model.num_parameters()
    tokens_per_step = tc.batch_size * tc.grad_accum_steps * model_cfg.block_size
    log(
        f"device={device} precision={precision} params={n_params:,} "
        f"tokens/step={tokens_per_step:,} train_tokens={len(train_data):,} "
        f"val_tokens={len(val_data):,}"
    )

    out_dir = Path(tc.out_dir) if tc.out_dir else None
    ckpt_path = out_dir / "last.pt" if out_dir else None

    def save(step: int) -> None:
        if ckpt_path is None:
            return
        save_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            config=run_cfg,
            tokenizer_state=tokenizer.to_dict(),
            data_rng_state=data_gen.get_state(),
            metrics={"best_val_loss": best_val, "last_val_loss": last_val},
        )

    def validate() -> float:
        return estimate_loss(
            model,
            val_data,
            batch_size=tc.batch_size,
            max_batches=tc.eval_batches,
            device=device,
            autocast=autocast_context(device, precision),
        )

    history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    step_times: list[float] = []
    best_val: float | None = ckpt["metrics"].get("best_val_loss") if ckpt else None
    last_val: float | None = ckpt["metrics"].get("last_val_loss") if ckpt else None
    last_train_loss: float | None = None

    model.train()
    reset_peak_memory(device)
    train_start = time.perf_counter()
    for step in range(start_step, tc.max_steps):
        t0 = time.perf_counter()
        lr = lr_at_step(step, tc)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        loss_sum = 0.0
        for _ in range(tc.grad_accum_steps):
            x, y = get_batch(train_data, tc.batch_size, model_cfg.block_size, data_gen, device)
            with autocast_context(device, precision):
                _, loss = model(x, y)
            loss_value = loss.detach().item()
            if not math.isfinite(loss_value):
                raise TrainingInstabilityError(
                    f"non-finite loss ({loss_value}) at step {step + 1}; "
                    "try a lower learning rate, gradient clipping, or fp32"
                )
            scaled = loss / tc.grad_accum_steps
            (scaler.scale(scaled) if scaler else scaled).backward()
            loss_sum += loss_value

        if scaler:
            scaler.unscale_(optimizer)
        max_norm = tc.grad_clip if tc.grad_clip > 0 else float("inf")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
        if not math.isfinite(grad_norm) and scaler is None:
            raise TrainingInstabilityError(f"non-finite gradient norm at step {step + 1}")
        if scaler:
            scaler.step(optimizer)  # skips the update if gradients overflowed
            scaler.update()
        else:
            optimizer.step()
        synchronize(device)
        step_times.append(time.perf_counter() - t0)

        last_train_loss = loss_sum / tc.grad_accum_steps
        done = step + 1
        history.append(
            {
                "step": done,
                "train_loss": last_train_loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_seen": done * tokens_per_step,
            }
        )
        if done % tc.log_interval == 0 or done == tc.max_steps:
            log(
                f"step {done:>5}/{tc.max_steps} | loss {last_train_loss:.4f} | lr {lr:.2e} "
                f"| grad_norm {grad_norm:.3f}"
            )
        if done % tc.eval_interval == 0 or done == tc.max_steps:
            last_val = validate()
            best_val = last_val if best_val is None else min(best_val, last_val)
            eval_history.append({"step": done, "val_loss": last_val})
            log(f"step {done:>5}/{tc.max_steps} | val_loss {last_val:.4f}")
        if tc.save_interval and done % tc.save_interval == 0 and done != tc.max_steps:
            save(done)

    train_time = time.perf_counter() - train_start
    if start_step >= tc.max_steps:  # nothing left to train; still report validation loss
        last_val = validate()
        best_val = last_val if best_val is None else min(best_val, last_val)
        notes.append(f"checkpoint already at step {start_step} >= max_steps; no training done")
    save(max(start_step, tc.max_steps))
    peak_bytes, peak_kind = peak_memory(device)

    timed = step_times[1:] or step_times
    result = TrainResult(
        num_parameters=n_params,
        device=str(device),
        precision=precision,
        resumed_from_step=start_step,
        steps_completed=len(step_times),
        tokens_per_step=tokens_per_step,
        tokens_seen_this_run=len(step_times) * tokens_per_step,
        train_time_sec=train_time,
        tokens_per_sec=tokens_per_step / statistics.median(timed) if timed else None,
        peak_memory_bytes=peak_bytes,
        peak_memory_kind=peak_kind,
        final_train_loss=last_train_loss,
        final_val_loss=last_val,
        best_val_loss=best_val,
        checkpoint_path=str(ckpt_path) if ckpt_path else None,
        notes=notes,
        history=history,
        eval_history=eval_history,
    )
    if out_dir:
        (out_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a small GPT language model.")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="K=V",
        help="override a config value, e.g. --set train.max_steps=500",
    )
    args = parser.parse_args(argv)
    try:
        cfg = ExperimentConfig.from_yaml(args.config).with_overrides(args.overrides)
        result = train(cfg, resume_from=args.resume)
    except (ConfigError, TrainingInstabilityError, FileNotFoundError, ValueError) as err:
        print(f"error: {err}")
        return 1
    summary = f"final val loss: {result.final_val_loss:.4f}"
    if result.tokens_per_sec:
        summary += f" | tokens/sec (median step): {result.tokens_per_sec:,.0f}"
    print(summary)
    if result.checkpoint_path:
        print(f"checkpoint: {result.checkpoint_path}")
    return 0
