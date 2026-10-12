# Experiments

Sweeps are run with `scripts/benchmark.py` (see [../docs/experiments.md](../docs/experiments.md)).

```bash
python scripts/benchmark.py --sweep attention --config configs/baseline.yaml --repeats 5
```

Reports are written to `experiments/results/` as `<sweep>_<UTC timestamp>.json` (complete
configs, environment, per-run records) and a matching `.csv` summary.

`experiments/results/` is git-ignored because raw outputs are machine-specific and can be large.
To publish a result you want to stand behind, copy the JSON/CSV into `experiments/reports/`
(commit that directory) and describe it in `docs/experiments.md` with the command, hardware,
repeats and the measured spread.

**No published results exist yet.** Nothing in this repository should be taken as a benchmark
finding until a report is committed here alongside its exact command and environment.
