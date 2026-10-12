# PyTorch LLM Pre-training Lab

A small, readable implementation of **GPT-style decoder-only language-model pre-training** in
PyTorch, plus an **experiment harness** for measuring how training choices (batch size,
gradient accumulation, learning-rate schedule, mixed precision, attention implementation,
gradient clipping) affect throughput, memory and validation loss.

It is a learning and portfolio project: the model is tiny by design so everything runs on an
ordinary CPU, and every claim the project makes about performance is meant to come from a
measurement the harness recorded, not from assumption.

> **Status.** The model, training pipeline, checkpoint/resume, evaluation, generation, benchmark
> harness and test suite are implemented and pass on CPU. **No benchmark results are published
> in this repository yet** — see [docs/experiments.md](docs/experiments.md). CUDA and
> mixed-precision-on-GPU code paths are implemented but have **not** been exercised on a GPU.

## Contents

- [What is in here](#what-is-in-here)
- [Model architecture](#model-architecture)
- [Training pipeline](#training-pipeline)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [CPU quick start](#cpu-quick-start)
- [Train, evaluate, generate, resume](#train-evaluate-generate-resume)
- [GPU usage](#gpu-usage)
- [Reproducing the optimization experiments](#reproducing-the-optimization-experiments)
- [How benchmarks are measured](#how-benchmarks-are-measured)
- [Tests and linting](#tests-and-linting)
- [Engineering questions the experiments investigate](#engineering-questions-the-experiments-investigate)
- [Limitations and future work](#limitations-and-future-work)
- [License](#license)

## What is in here

- A from-scratch GPT (`src/pretraining_lab/model.py`) with causal multi-head attention, an MLP
  block, pre-LayerNorm residual blocks, tied embeddings, GPT-2-style initialisation and
  temperature / top-k sampling.
- A complete training loop: character-level tokenization, leak-free train/validation split,
  AdamW, warmup + cosine/linear/constant LR schedules, gradient accumulation, gradient clipping
  with grad-norm logging, optional bf16/fp16 autocast, NaN/Inf detection, atomic checkpoints and
  exact resume.
- A benchmark harness that runs named sweeps and records configuration, software versions,
  hardware, wall time, tokens/sec, peak memory (when measurable) and validation loss.
- 81 automated tests that run on CPU in a couple of seconds.

## Model architecture

```
token ids ──► token embedding ─┐
                               ├─► dropout ─► [ Block ] x n_layer ─► LayerNorm ─► output head ─► logits
positions ──► position embedding ┘                                            (weights tied to token embedding)

Block:   x ─► x + CausalSelfAttention(LayerNorm(x)) ─► x + MLP(LayerNorm(x))
MLP:     Linear(d, 4d) ─► GELU ─► Linear(4d, d)
```

Configurable: vocabulary size (inferred from the tokenizer), context length (`block_size`),
`d_model`, `n_head`, `n_layer`, dropout, bias, and the attention implementation (`manual` or
`sdpa`). Design rationale is in [docs/architecture.md](docs/architecture.md).

## Training pipeline

1. **Data** — load text (a built-in deterministic synthetic corpus, or your own file) and
   tokenize it at character level.
2. **Split** — the token stream is cut into a contiguous train / validation split *before*
   windowing, so no training window overlaps validation text.
3. **Batching** — random fixed-length windows for training (seeded generator); deterministic,
   non-overlapping windows for validation so the metric is comparable across runs.
4. **Optimization** — cross-entropy next-token loss, AdamW (no weight decay on biases/norms),
   linear warmup then cosine/linear/constant schedule, optional gradient accumulation,
   gradient clipping, optional autocast.
5. **Monitoring** — train loss, LR and gradient norm every step; validation loss at a
   configurable interval. A NaN/Inf loss or gradient norm stops the run with a clear error.
6. **Persistence** — atomic checkpoints containing model, optimizer, scaler, RNG states, data
   sampler state, config and tokenizer; `--resume` continues from them.

## Repository structure

```
pytorch-llm-pretraining-lab/
├── README.md  LICENSE  pyproject.toml  .gitignore
├── configs/
│   ├── tiny_cpu.yaml        # 27,680 parameters; trains in seconds on CPU
│   └── baseline.yaml        # 806,272 parameters; baseline for optimization experiments
├── src/pretraining_lab/
│   ├── config.py            # dataclass configs, validation, YAML loading, overrides
│   ├── model.py             # GPT, attention, MLP, parameter counting
│   ├── data.py              # tokenizer, synthetic corpus, split, batching
│   ├── train.py             # training loop, LR schedule, optimizer, CLI
│   ├── evaluate.py          # validation loss / perplexity, CLI
│   ├── generate.py          # temperature + top-k sampling, CLI
│   ├── checkpoint.py        # atomic save, safe load, exact-resume state
│   ├── benchmark.py         # experiment sweeps, result recording, CLI
│   └── utils.py             # seeding, device/precision resolution, memory, environment info
├── scripts/                 # thin wrappers: train.py evaluate.py generate.py benchmark.py
├── tests/                   # pytest suite (CPU only)
├── experiments/README.md    # how results are produced and stored
└── docs/                    # architecture.md, experiments.md
```

`utils.py` was added to the originally sketched layout to keep device, precision and
measurement helpers out of the training loop. The scripts in `scripts/` only call each module's
`main()`; there is no duplicated logic.

## Installation

Requires Python 3.10+. The CPU quick start needs only `torch`, `numpy` and `pyyaml`.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Install PyTorch first. For a small CPU-only build:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# (The default PyPI torch wheel on Linux also pulls in several GB of NVIDIA CUDA libraries.)

pip install -e ".[dev]"              # the project + pytest + ruff
```

## CPU quick start

```bash
# 1. Run the tests (about 2 seconds on a laptop-class CPU)
pytest

# 2. Train the tiny model on the built-in synthetic corpus
python scripts/train.py --config configs/tiny_cpu.yaml

# 3. Evaluate the checkpoint on the validation split
python scripts/evaluate.py --checkpoint runs/tiny_cpu/last.pt

# 4. Generate text
python scripts/generate.py --checkpoint runs/tiny_cpu/last.pt \
    --prompt "the cat " --max-new-tokens 120 --temperature 0.8 --top-k 5
```

The same entry points are installed as commands: `pretrain-train`, `pretrain-evaluate`,
`pretrain-generate`, `pretrain-benchmark`.

## Train, evaluate, generate, resume

```bash
# Train; override any config value from the command line with --set section.key=value
python scripts/train.py --config configs/tiny_cpu.yaml --set train.max_steps=500 --set train.lr=0.002

# Train on your own text file (use only text you are permitted to use)
python scripts/train.py --config configs/baseline.yaml \
    --set data.source=file --set data.path=path/to/corpus.txt

# Evaluate on the training or validation split
python scripts/evaluate.py --checkpoint runs/tiny_cpu/last.pt --split val --batches 50

# Generate: temperature 0 means greedy decoding; --top-k restricts sampling.
# (A tiny model trained for a few hundred steps produces only roughly sentence-shaped text;
# it does not learn the arithmetic lines in the synthetic corpus.)
python scripts/generate.py --checkpoint runs/tiny_cpu/last.pt --prompt "the " \
    --temperature 0 --max-new-tokens 40

# Resume from a checkpoint
python scripts/train.py --config configs/tiny_cpu.yaml --resume runs/tiny_cpu/last.pt
```

Notes on resuming:

- A run writes `last.pt` and `result.json` to `train.out_dir`. Set `train.save_interval` to
  also checkpoint during training (e.g. `--set train.save_interval=50`).
- Resuming with the **same config** continues exactly where the checkpoint left off. This is
  verified by a test that crashes a CPU/fp32 run after step 7 and compares the resumed losses
  for steps 7–12 against an uninterrupted run.
- The learning rate is a function of `step` and `train.max_steps`. If you resume with a
  *larger* `max_steps` to train longer, the schedule is recomputed from that step on, so the
  LR can jump back up. This is expected, not a bug.
- The model section of the config must match the checkpoint; a mismatch is rejected.
- Checkpoints are loaded with `weights_only=True`, so a tampered file cannot execute code.

## GPU usage

CUDA is optional and never required to install or test the project.

```bash
python scripts/train.py --config configs/baseline.yaml \
    --set train.device=cuda --set train.precision=bf16 --set model.attention_impl=sdpa
```

- `train.device=auto` (the default) uses CUDA when it is available and CPU otherwise;
  explicitly requesting `cuda` without a GPU is an error.
- `train.precision` is `fp32`, `bf16` or `fp16` (fp16 uses a gradient scaler). If a precision is
  unsupported on the selected device it falls back to fp32 and records a note in the result.
- GPU peak memory is recorded from `torch.cuda.max_memory_allocated`.

> **Not yet verified on a GPU.** These paths are implemented but the development environment
> had no GPU, so only their CPU-side fallbacks are covered by tests.

## Reproducing the optimization experiments

```bash
# Available sweeps: baseline, batch_accum, batch_size, lr_schedule, attention, precision, clipping
python scripts/benchmark.py --sweep attention --config configs/baseline.yaml --repeats 3

# Shorter runs for a quick look
python scripts/benchmark.py --sweep batch_accum --set train.max_steps=30 --repeats 3
```

Each invocation writes a JSON report (full configs and environment) and a CSV summary to
`experiments/results/` (git-ignored; see [experiments/README.md](experiments/README.md)).
The sweeps and their rationale are described in [docs/experiments.md](docs/experiments.md).

## How benchmarks are measured

- **Tokens/sec** is `tokens_per_optimizer_step / median(step_time)`, where a step includes all
  micro-batches, the backward pass, clipping and the optimizer update. The first step (warm-up)
  is excluded, validation time is excluded, and CUDA is synchronized before stopping each timer.
- **Peak memory** is `torch.cuda.max_memory_allocated` on CUDA. On Linux CPU it is the process
  peak RSS (`VmHWM`) since a reset at the start of training, which includes data and the
  interpreter, not just tensors. If memory cannot be measured the field is `null`, never guessed.
- **Validation loss** is the mean cross-entropy over a fixed set of non-overlapping validation
  windows, so it is comparable across runs.
- **Every record** stores the full config, seed, status (`ok`, `diverged`, `invalid_config`),
  library versions, CPU/GPU model and thread count.
- Use `--repeats` and compare across seeds. Single runs on a shared or single-core machine are
  noisy (see [docs/experiments.md](docs/experiments.md)).

## Tests and linting

```bash
pytest                     # 81 tests: model, attention causality, data/leakage, loss,
                           # checkpoint/resume, generation, end-to-end CPU training, CLI, benchmark
ruff check .               # lint
ruff format --check .      # formatting
```

## Engineering questions the experiments investigate

The harness is built to answer questions like these with measurements, not assumptions:

1. **Gradient accumulation vs. micro-batch size.** For a fixed effective batch, how do
   throughput and peak memory change as the batch is split into more micro-batches, and does
   validation loss stay the same?
2. **Batch size.** How do tokens/sec and loss after a fixed number of steps change with batch size?
3. **Learning rate and schedule.** How sensitive is final validation loss to peak LR and to
   cosine vs. constant schedules at a fixed step budget?
4. **Attention implementation.** Does `scaled_dot_product_attention` match the manual
   implementation numerically (tested) and does it change throughput or memory on this hardware?
5. **Mixed precision.** What does bf16/fp16 autocast change in speed, memory and loss, and on
   which hardware?
6. **Gradient clipping and stability.** At an aggressive learning rate, does clipping change
   whether training diverges?

### What I learned

_This section is intentionally left for the author to fill in after running the experiments on
their own hardware. It should describe observed results and the lessons drawn from them —
nothing here is written in advance._

## Limitations and future work

- Character-level tokenizer only; the data path accepts any integer token tensor, so a BPE
  tokenizer (e.g. `tiktoken`, listed as an optional extra) is a natural extension.
- Single process, single device. No distributed training (DDP/FSDP), no activation
  checkpointing, no `torch.compile` experiments, no fused optimizer or flash-attention-specific
  kernels beyond what `sdpa` selects automatically.
- Datasets are held in memory; very large corpora would need a memory-mapped/sharded loader.
- Learned absolute position embeddings; no RoPE/ALiBi, no KV cache for generation.
- CUDA and GPU mixed precision are untested in the development environment.
- Resume is verified for CPU/fp32 with dropout 0; bit-exact resume on GPU is not claimed.

## License

[jj-aggarwal](LICENSE). You may use, modify and redistribute the code, including commercially, provided
the copyright and license notice are kept. The software is provided without warranty. The
synthetic corpus is generated by this repository's own code, and no third-party code or datasets
are included.
