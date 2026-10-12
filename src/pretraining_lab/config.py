"""Configuration dataclasses with validation, YAML loading and CLI-style overrides."""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ATTENTION_IMPLS = ("manual", "sdpa")
SCHEDULERS = ("cosine", "linear", "constant")
PRECISIONS = ("fp32", "bf16", "fp16")
DEVICES = ("auto", "cpu", "cuda")
DATA_SOURCES = ("synthetic", "file")


class ConfigError(ValueError):
    """Raised when a configuration is invalid."""


@dataclass
class ModelConfig:
    vocab_size: int | None = None  # filled in from the tokenizer when None
    block_size: int = 64  # maximum context length
    n_layer: int = 2
    n_head: int = 2
    d_model: int = 64
    dropout: float = 0.0
    bias: bool = True
    attention_impl: str = "manual"  # "manual" (reference) or "sdpa" (torch fused/efficient path)

    def validate(self, require_vocab: bool = False) -> None:
        if require_vocab and (self.vocab_size is None or self.vocab_size < 1):
            raise ConfigError("model.vocab_size must be a positive integer")
        if self.vocab_size is not None and self.vocab_size < 1:
            raise ConfigError("model.vocab_size must be positive")
        for name in ("block_size", "n_layer", "n_head", "d_model"):
            if getattr(self, name) < 1:
                raise ConfigError(f"model.{name} must be >= 1, got {getattr(self, name)}")
        if self.d_model % self.n_head != 0:
            raise ConfigError(
                f"model.d_model ({self.d_model}) must be divisible by model.n_head ({self.n_head})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"model.dropout must be in [0, 1), got {self.dropout}")
        if self.attention_impl not in ATTENTION_IMPLS:
            raise ConfigError(
                f"model.attention_impl must be one of {ATTENTION_IMPLS}, "
                f"got {self.attention_impl!r}"
            )


@dataclass
class DataConfig:
    source: str = "synthetic"  # "synthetic" or "file"
    path: str | None = None  # required when source == "file"
    train_fraction: float = 0.9
    synthetic_chars: int = 20_000
    synthetic_seed: int = 1234

    def validate(self) -> None:
        if self.source not in DATA_SOURCES:
            raise ConfigError(f"data.source must be one of {DATA_SOURCES}, got {self.source!r}")
        if self.source == "file" and not self.path:
            raise ConfigError("data.path is required when data.source == 'file'")
        if not 0.0 < self.train_fraction < 1.0:
            raise ConfigError(f"data.train_fraction must be in (0, 1), got {self.train_fraction}")
        if self.synthetic_chars < 1:
            raise ConfigError("data.synthetic_chars must be >= 1")


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "auto"
    precision: str = "fp32"
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 200
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 20
    scheduler: str = "cosine"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # 0 disables clipping (the gradient norm is still logged)
    eval_interval: int = 50
    eval_batches: int = 8
    log_interval: int = 10
    save_interval: int = 0  # 0 = only save at the end
    out_dir: str | None = "runs/default"  # None disables all file output

    def validate(self) -> None:
        if self.device not in DEVICES:
            raise ConfigError(f"train.device must be one of {DEVICES}, got {self.device!r}")
        if self.precision not in PRECISIONS:
            raise ConfigError(
                f"train.precision must be one of {PRECISIONS}, got {self.precision!r}"
            )
        if self.scheduler not in SCHEDULERS:
            raise ConfigError(
                f"train.scheduler must be one of {SCHEDULERS}, got {self.scheduler!r}"
            )
        for name in (
            "batch_size",
            "grad_accum_steps",
            "max_steps",
            "eval_interval",
            "eval_batches",
        ):
            if getattr(self, name) < 1:
                raise ConfigError(f"train.{name} must be >= 1, got {getattr(self, name)}")
        if self.log_interval < 1:
            raise ConfigError("train.log_interval must be >= 1")
        if self.save_interval < 0 or self.warmup_steps < 0:
            raise ConfigError("train.save_interval and train.warmup_steps must be >= 0")
        if self.warmup_steps >= self.max_steps:
            raise ConfigError(
                f"train.warmup_steps ({self.warmup_steps}) must be < train.max_steps "
                f"({self.max_steps})"
            )
        if self.lr <= 0 or self.min_lr < 0 or self.min_lr > self.lr:
            raise ConfigError("need 0 <= train.min_lr <= train.lr and train.lr > 0")
        if self.weight_decay < 0 or self.grad_clip < 0:
            raise ConfigError("train.weight_decay and train.grad_clip must be >= 0")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ConfigError("train.beta1 and train.beta2 must be in [0, 1)")


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        self.train.validate()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        unknown = set(raw) - {"model", "data", "train"}
        if unknown:
            raise ConfigError(f"unknown top-level config sections: {sorted(unknown)}")
        cfg = cls(
            model=_build(ModelConfig, raw.get("model") or {}, "model"),
            data=_build(DataConfig, raw.get("data") or {}, "data"),
            train=_build(TrainConfig, raw.get("train") or {}, "train"),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"config file must contain a mapping at the top level: {path}")
        return cls.from_dict(raw)

    def with_overrides(self, overrides: list[str]) -> ExperimentConfig:
        """Return a validated copy with ``section.key=value`` overrides applied.

        Values are parsed as YAML, so ``train.lr=0.001`` is a float and ``model.bias=false``
        is a bool.
        """
        raw = self.to_dict()
        for item in overrides:
            if "=" not in item:
                raise ConfigError(f"override must look like section.key=value, got {item!r}")
            key, value = item.split("=", 1)
            section, _, name = key.strip().partition(".")
            if section not in raw or not name or name not in raw[section]:
                raise ConfigError(f"unknown override key {key!r}")
            raw[section][name] = yaml.safe_load(value)
        return ExperimentConfig.from_dict(raw)


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    if not isinstance(values, dict):
        raise ConfigError(f"config section {section!r} must be a mapping")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"unknown keys in section {section!r}: {sorted(unknown)}")
    return cls(**{k: _coerce(v, hints[k], f"{section}.{k}") for k, v in values.items()})


def _coerce(value: Any, hint: Any, name: str) -> Any:
    """Coerce YAML scalars to the annotated type (YAML parses ``3e-4`` as a string)."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        non_none = [a for a in typing.get_args(hint) if a is not type(None)]
        return _coerce(value, non_none[0], name)
    try:
        if hint is bool:
            if not isinstance(value, bool):
                raise TypeError("expected true/false")
            return value
        if hint is int:
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise TypeError("expected an integer")
            return int(value)
        if hint is float:
            if isinstance(value, bool):
                raise TypeError("expected a number")
            return float(value)
        if hint is str:
            if not isinstance(value, str):
                raise TypeError("expected a string")
            return value
    except (TypeError, ValueError) as err:
        raise ConfigError(f"invalid value for {name}: {value!r} ({err})") from err
    return value
