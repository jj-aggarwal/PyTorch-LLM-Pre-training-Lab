# Optimization experiments

This project treats performance claims as hypotheses to measure. The harness
(`src/pretraining_lab/benchmark.py`) automates the measurement; this document records the design
and the **honest state of the results**.

## Current results

**No benchmark results are recorded in this repository yet.** The test suite and a CPU smoke
run were executed during development, but no sweep has been run to completion and published.
Do not read any statement here as a finding. When you run sweeps, put the exact commands,
hardware and the generated JSON/CSV next to your conclusions (see
[experiments/README.md](../experiments/README.md)).

## Method

All sweeps start from a base config (`configs/baseline.yaml` by default) and vary one factor.
Runs execute one after another as separate `train()` calls inside one process, each with a
different seed per repeat and `out_dir=None` (no checkpoints written). Each call builds a fresh
model and optimizer, so runs do not share training state, but they do share the process
(allocator caches, CPU frequency/thermal state), which is another reason to use repeats.

| Sweep | Variants | Question |
|---|---|---|
| `baseline` | the base config | Reference run. |
| `batch_accum` | batch 16/8/4/2 with accumulation 1/2/4/8 | Same effective batch of 16 sequences: how do throughput and peak memory change with micro-batch size, and does loss stay equal? |
| `batch_size` | batch 4/8/16/32, no accumulation | Throughput and loss vs. batch size at a fixed step budget (tokens per step vary). |
| `lr_schedule` | peak LR 3e-4/1e-3/3e-3 x cosine/constant | Sensitivity of final validation loss to LR and schedule. |
| `attention` | `manual` vs `sdpa` | Throughput/memory difference between the implementations on this hardware. |
| `precision` | fp32 / bf16 / fp16 | Effect of autocast on speed, memory, loss. Unsupported precisions fall back to fp32 and are labelled in the `notes` column. |
| `clipping` | clip 0 / 1.0 / 0.1 at LR 0.03 | Whether clipping changes divergence at an aggressive LR. Diverged runs are recorded with status `diverged`. |

Run, for example:

```bash
python scripts/benchmark.py --sweep batch_accum --config configs/baseline.yaml --repeats 5
```

## What is recorded

Per run: label, seed, overrides, full resolved config, status, device, effective precision,
parameter count, tokens per optimizer step, tokens seen, training wall time, median tokens/sec,
peak memory and its *kind*, final train loss, final and best validation loss, the validation-loss
history, and any notes (e.g. precision fallbacks). Per report: Python, torch and numpy versions,
platform, CPU model, logical cores, torch thread count, CUDA availability and, on GPU, the device
name, memory and CUDA runtime version.

## Measurement caveats

- **Run-to-run noise is large on small models and shared machines.** During development, an
  `fp16` variant that fell back to fp32 (so it ran *exactly the same computation* as the `fp32`
  variant) reported roughly 16% lower tokens/sec in a single 10-step run on a single-core VM.
  That is a smoke observation, not a benchmark, but it shows why conclusions need `--repeats`,
  several seeds, and effect sizes well above the noise.
- **Tokens/sec** uses the median step time with the first step excluded, and validation time is
  not included. Wall time for a short run also includes warm-up effects.
- **CPU memory** is process peak RSS (`VmHWM`) since a reset at the start of training. It
  includes the interpreter and data, so only *differences* between variants are meaningful, and
  only when the reset succeeded (the `peak_memory_kind` field says which measurement you got).
  When memory cannot be measured it is `null`.
- **CPU bf16** autocast is functionally supported but is not guaranteed to be faster than fp32 on
  a given CPU; measure it.
- **Loss comparisons across batch sizes** are at equal *steps*, not equal tokens, unless the
  effective batch is held constant (as in `batch_accum`).
- **Equal seeds do not mean equal data order across different batch sizes**, because the
  sampler draws a different number of windows per step.
- **GPU paths are untested** in the development environment, so no GPU conclusions exist.

## Writing up a result

For each conclusion, record: the command, config file, repeats and seeds, the hardware and
software versions (copy from the report's `environment`), the raw CSV/JSON, the measured
numbers with their spread, and what you did *not* test. Separate "measured" from "expected".
