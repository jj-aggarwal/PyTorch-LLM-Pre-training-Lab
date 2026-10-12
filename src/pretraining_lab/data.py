"""Text -> token ids, leak-free splitting, and batching.

The default tokenizer is character-level: it needs no dependencies and keeps the vocabulary
tiny, which suits CPU smoke tests. The data path (``prepare_data`` / ``get_batch``) only deals in
integer token tensors, so swapping in a subword tokenizer later does not change the training loop.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import torch

from .config import ConfigError, DataConfig


class CharTokenizer:
    """Maps each distinct character to an integer id (sorted, so ids are deterministic)."""

    def __init__(self, chars: list[str]) -> None:
        if not chars:
            raise ValueError("tokenizer needs at least one character")
        if len(set(chars)) != len(chars):
            raise ValueError("tokenizer characters must be unique")
        self.chars = list(chars)
        self._stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        if not text:
            raise ValueError("cannot build a tokenizer from empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> torch.Tensor:
        try:
            return torch.tensor([self._stoi[c] for c in text], dtype=torch.long)
        except KeyError as err:
            raise ValueError(
                f"character {err.args[0]!r} is not in the tokenizer vocabulary"
            ) from err

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        values = ids.tolist() if isinstance(ids, torch.Tensor) else ids
        return "".join(self.chars[i] for i in values)

    def to_dict(self) -> dict:
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, data: dict) -> CharTokenizer:
        if data.get("type") != "char":
            raise ValueError(f"unsupported tokenizer type: {data.get('type')!r}")
        return cls(list(data["chars"]))


_SUBJECTS = ["the cat", "a dog", "the bird", "my friend", "the robot", "a student", "the farmer"]
_VERBS = ["sees", "likes", "finds", "follows", "draws", "helps", "calls"]
_OBJECTS = ["the moon", "a red ball", "the old tree", "some bread", "a small boat", "the river"]


def make_synthetic_text(num_chars: int, seed: int) -> str:
    """Deterministic toy corpus: simple sentences and arithmetic facts. Has real structure for
    a model to learn, so loss visibly drops, and involves no external or licensed data."""
    if num_chars < 1:
        raise ValueError("num_chars must be >= 1")
    rng = random.Random(seed)
    lines: list[str] = []
    total = 0
    while total < num_chars:
        if rng.random() < 0.7:
            line = f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}."
        else:
            a, b = rng.randint(0, 9), rng.randint(0, 9)
            line = f"{a} + {b} = {a + b}."
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)[:num_chars]


def load_text(cfg: DataConfig) -> str:
    if cfg.source == "synthetic":
        return make_synthetic_text(cfg.synthetic_chars, cfg.synthetic_seed)
    path = Path(cfg.path)
    if not path.is_file():
        raise ConfigError(f"data.path does not exist or is not a file: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ConfigError(f"data file is empty: {path}")
    return text


def split_tokens(
    ids: torch.Tensor, train_fraction: float, min_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous split: the first ``train_fraction`` of the stream trains, the rest validates.

    Splitting the token stream *before* any windowing means no training window can overlap
    validation text, so validation loss is free of window-level leakage.
    """
    if ids.dim() != 1:
        raise ValueError("ids must be a 1-D tensor")
    cut = int(len(ids) * train_fraction)
    train, val = ids[:cut], ids[cut:]
    if len(train) < min_len or len(val) < min_len:
        raise ValueError(
            f"dataset too small: {len(ids)} tokens gives train={len(train)}, val={len(val)}, "
            f"but each split needs at least {min_len} (block_size + 2). "
            "Use more data or a smaller model.block_size."
        )
    return train, val


def prepare_data(
    cfg: DataConfig, block_size: int, tokenizer: CharTokenizer | None = None
) -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    """Load text, tokenize it (building a tokenizer unless one is given) and split it."""
    text = load_text(cfg)
    tokenizer = tokenizer or CharTokenizer.from_text(text)
    train, val = split_tokens(tokenizer.encode(text), cfg.train_fraction, block_size + 2)
    return tokenizer, train, val


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` random windows. Targets are the inputs shifted left by one."""
    if len(data) < block_size + 1:
        raise ValueError(
            f"data has {len(data)} tokens, need at least block_size + 1 = {block_size + 1}"
        )
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts.tolist()])
    y = torch.stack([data[s + 1 : s + 1 + block_size] for s in starts.tolist()])
    return x.to(device), y.to(device)


def iter_eval_batches(
    data: torch.Tensor, batch_size: int, block_size: int, max_batches: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic, non-overlapping windows from the start of ``data`` (no randomness)."""
    starts = list(range(0, len(data) - block_size, block_size))
    for i in range(0, len(starts), batch_size):
        if i // batch_size >= max_batches:
            return
        chunk = starts[i : i + batch_size]
        yield (
            torch.stack([data[s : s + block_size] for s in chunk]),
            torch.stack([data[s + 1 : s + 1 + block_size] for s in chunk]),
        )
