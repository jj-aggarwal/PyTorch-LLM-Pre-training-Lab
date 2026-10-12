"""A GPT-style decoder-only Transformer, written to be read.

Architecture decisions (see docs/architecture.md for the longer discussion):

* Pre-LayerNorm blocks (``x + f(LN(x))``): more stable to train than post-LN without
  needing careful warmup tuning.
* Learned absolute position embeddings: simplest option, and fine for a fixed context length.
* Weight tying between the token embedding and the output projection: saves
  ``vocab_size * d_model`` parameters and is standard for small language models.
* GPT-2-style initialisation: N(0, 0.02) everywhere, with residual output projections scaled by
  ``1/sqrt(2 * n_layer)`` so the residual stream variance does not grow with depth.
* Two interchangeable attention paths. ``manual`` is an explicit masked softmax that serves as
  the readable reference; ``sdpa`` calls ``torch.nn.functional.scaled_dot_product_attention``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position ``t`` can only attend to positions ``<= t``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.dropout_p = cfg.dropout
        self.impl = cfg.attention_impl
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        if self.impl == "manual":
            mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
            # Non-persistent: not stored in checkpoints, so checkpoints work with either impl.
            self.register_buffer(
                "causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (
            t.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v)
        )
        if self.impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            allowed = self.causal_mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(~allowed, float("-inf"))
            weights = self.attn_dropout(F.softmax(scores, dim=-1))
            y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward block with a 4x hidden expansion and GELU."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate(require_vocab=True)
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):  # attention and MLP output projections
                nn.init.normal_(param, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, exclude_position_embeddings: bool = False) -> int:
        """Trainable parameters. Tied weights are counted once."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_position_embeddings:
            total -= self.wpe.weight.numel()
        return total

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(logits, loss)``. ``logits`` has shape (B, T, vocab); ``loss`` is None
        unless ``targets`` (B, T) is given."""
        if idx.dim() != 2:
            raise ValueError(f"idx must have shape (batch, seq_len), got {tuple(idx.shape)}")
        if idx.dtype != torch.long:
            raise ValueError(f"idx must be torch.long token ids, got {idx.dtype}")
        batch, seq_len = idx.shape
        if seq_len == 0:
            raise ValueError("idx must contain at least one token")
        if seq_len > self.cfg.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds the context length {self.cfg.block_size}"
            )
        positions = torch.arange(seq_len, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} must match idx {tuple(idx.shape)}"
                )
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def estimate_num_parameters(cfg: ModelConfig) -> int:
    """Closed-form parameter count (tied embeddings counted once); used to cross-check the model."""
    d = cfg.d_model
    b = 1 if cfg.bias else 0
    layer_norm = d + b * d
    attention = d * 3 * d + b * 3 * d + d * d + b * d
    mlp = d * 4 * d + b * 4 * d + 4 * d * d + b * d
    per_block = 2 * layer_norm + attention + mlp
    return cfg.vocab_size * d + cfg.block_size * d + cfg.n_layer * per_block + layer_norm
