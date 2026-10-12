# Architecture and design decisions

This document explains *why* the code is shaped the way it is. For *how to run it*, see the
[README](../README.md).

## Model (`src/pretraining_lab/model.py`)

```
idx (B, T) ─► wte(idx) + wpe(0..T-1) ─► dropout
           ─► Block x n_layer
           ─► LayerNorm
           ─► lm_head (weights tied to wte) ─► logits (B, T, vocab)

Block(x) = x + Attn(LN1(x));  then  y + MLP(LN2(y))
```

| Decision | Choice | Why |
|---|---|---|
| Normalisation placement | Pre-LayerNorm | Keeps a clean residual path; trains stably without delicate warmup tuning. |
| Position information | Learned absolute embeddings | Simplest option for a fixed context length; RoPE/ALiBi are listed as future work. |
| Output head | Tied to token embedding | Saves `vocab * d_model` parameters; standard for small LMs. |
| MLP | 4x expansion, GELU | Conventional GPT ratio. |
| Initialisation | N(0, 0.02); residual projections scaled by `0.02 / sqrt(2 * n_layer)` | GPT-2-style: keeps residual-stream variance from growing with depth. Tested. |
| Bias | Configurable (default on) | Allows comparing with bias-free variants. |
| Logits | Always computed for all positions | Simple and correct; a generation-only last-position optimisation is possible but unnecessary here. |

### Causal attention

Position `t` may only attend to positions `<= t`. Two interchangeable implementations are kept
on purpose:

- **`manual`** — explicit `softmax(QK^T / sqrt(d_head) + mask) V` with a lower-triangular
  mask. It is the readable reference and the baseline in experiments.
- **`sdpa`** — `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`, which
  lets PyTorch choose an efficient kernel when one is available for the hardware and dtype.

Tests check causality for both (changing token `t` must leave logits at positions `< t`
unchanged) and check that the two produce the same outputs for the same weights. Whether `sdpa`
is faster or lighter on a given machine is an empirical question for the benchmark harness; this
repository makes no a-priori claim.

The manual path's mask is a non-persistent buffer, so checkpoints contain no mask and load into
either implementation.

### Parameter counting

`GPT.num_parameters()` sums unique parameters, so the tied embedding/head matrix is counted
once. `estimate_num_parameters(cfg)` is an independent closed-form formula; a test asserts the
two agree for every combination of `bias` and attention implementation. With bias, per block:
`2 LayerNorms (2d each) + qkv (3d² + 3d) + attn proj (d² + d) + fc (4d² + 4d) + mlp proj (4d² + d)`.

## Data (`src/pretraining_lab/data.py`)

- **Tokenizer.** Character-level, vocabulary sorted for determinism, serialised into the
  checkpoint so generation and evaluation always use the training-time mapping. Unknown
  characters raise an error rather than being silently dropped.
- **Synthetic corpus.** Seeded generator of simple sentences and arithmetic facts. It has real
  structure to learn (so loss visibly falls in a CPU smoke test) and involves no external data
  or licensing questions.
- **No leakage.** The *token stream* is split contiguously (first 90% train, last 10%
  validation) before any windowing. Training windows are sampled only from the train tensor and
  validation windows only from the validation tensor, so no window can straddle the boundary.
  A test samples many training windows and asserts none touches a validation position.
- **Batching.** Training windows are random with a seeded `torch.Generator` (its state is
  checkpointed). Validation uses deterministic, non-overlapping windows from the start of the
  validation split, so validation loss is directly comparable between runs.

## Training (`src/pretraining_lab/train.py`)

- **Optimizer.** AdamW; weight decay applies to matrices (`dim >= 2`) only, not biases or
  LayerNorm parameters.
- **LR schedule.** A pure function `lr_at_step(step, config)`: linear warmup, then
  cosine / linear / constant decay to `min_lr`. There is no scheduler object, so there is no
  scheduler state to lose on resume, and the schedule is trivially unit-testable.
- **Gradient accumulation.** Each micro-batch loss is divided by `grad_accum_steps` before
  `backward()`, so the accumulated gradient equals that of one large batch of the same
  sequences.
- **Clipping.** `clip_grad_norm_` is always called so the gradient norm is always logged; with
  `grad_clip: 0` the max-norm is infinite and nothing is clipped. The logged norm is the
  pre-clipping norm.
- **Mixed precision.** `torch.autocast` for bf16/fp16; fp16 on CUDA additionally uses
  `GradScaler`. Unsupported combinations fall back to fp32 and the fallback is recorded in the
  result notes rather than failing silently.
- **Numerical safety.** A non-finite loss, or a non-finite gradient norm without a scaler,
  raises `TrainingInstabilityError` with a hint, instead of continuing with corrupted weights.
  The benchmark harness catches this and records the run as `diverged`.
- **Validation.** Evaluated at `eval_interval` and at the final step, under the same autocast
  setting as training.

## Checkpoints (`src/pretraining_lab/checkpoint.py`)

A checkpoint stores: model weights, optimizer state, gradient-scaler state, step, the full
resolved config, the tokenizer, the data-sampler RNG state, global torch/Python (and CUDA, when
present) RNG states, and the best/last validation loss.

- **Atomic writes.** Saved to a temporary file, then `os.replace`d, so an interruption cannot
  leave a half-written checkpoint.
- **Safe loading.** `torch.load(..., weights_only=True)`; the contents are only tensors and
  plain containers, so loading cannot execute arbitrary code from a tampered file.
- **Exact resume.** The resume test crashes a run after step 7 (last checkpoint at step 6),
  resumes, and requires the resumed losses for steps 7–12 and the final validation loss to match
  an uninterrupted run. This is verified for CPU, fp32, dropout 0. GPU resume is not claimed.
- **Compatibility.** A checkpoint can be loaded with either attention implementation, but
  resuming training rejects any other change to the model config.

## Error handling

Configuration problems raise `ConfigError` with the offending key and value (unknown keys,
`d_model % n_head != 0`, `warmup_steps >= max_steps`, bad enums, ...). YAML 1.1 reads `3e-4` as a
string, so numeric fields are coerced explicitly. The command-line tools catch expected errors
and print `error: ...` with exit code 1 instead of a traceback.
