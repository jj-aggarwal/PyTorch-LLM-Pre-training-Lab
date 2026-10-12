"""Checkpoint save / load with everything needed to resume training exactly.

Checkpoints are loaded with ``weights_only=True`` so a malicious pickle cannot execute code;
everything stored here is tensors and plain Python containers.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .model import GPT

CHECKPOINT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state(), "python": random.getstate()}
    state["cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(
    path: str | Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    step: int,
    config: ExperimentConfig,
    tokenizer_state: dict,
    data_rng_state: torch.Tensor,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Write atomically: a crash mid-save leaves the previous checkpoint intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": CHECKPOINT_VERSION,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config.to_dict(),
        "tokenizer": tokenizer_state,
        "data_rng_state": data_rng_state,
        "rng": capture_rng_state(),
        "metrics": metrics or {},
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location, weights_only=True)
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {state.get('version')!r} "
            f"(expected {CHECKPOINT_VERSION})"
        )
    return state


def build_model_from_checkpoint(ckpt: dict[str, Any]) -> GPT:
    cfg = ExperimentConfig.from_dict(ckpt["config"])
    model = GPT(cfg.model)
    model.load_state_dict(ckpt["model"])
    return model
