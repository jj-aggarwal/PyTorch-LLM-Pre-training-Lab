"""Experiment harness: run named sweeps and record *measured* results with full context.

Nothing here assumes one setting is faster or better than another. Every run records the
configuration, software versions, hardware, wall-clock time, tokens/sec, peak memory (when
measurable on the device) and validation loss, so conclusions come from the recorded data.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ConfigError, ExperimentConfig
from .train import TrainingInstabilityError, train
from .utils import environment_info

Variant = tuple[str, list[str]]  # (label, config overrides)


def _batch_accum() -> list[Variant]:
    # Same effective batch (16 sequences/step) split into different micro-batch sizes.
    return [
        (f"bs{b}_acc{16 // b}", [f"train.batch_size={b}", f"train.grad_accum_steps={16 // b}"])
        for b in (16, 8, 4, 2)
    ]


def _batch_size() -> list[Variant]:
    return [
        (f"bs{b}", [f"train.batch_size={b}", "train.grad_accum_steps=1"]) for b in (4, 8, 16, 32)
    ]


def _lr_schedule() -> list[Variant]:
    return [
        (
            f"lr{lr}_{sched}",
            [f"train.lr={lr}", f"train.min_lr={lr / 10}", f"train.scheduler={sched}"],
        )
        for lr in (0.0003, 0.001, 0.003)
        for sched in ("cosine", "constant")
    ]


def _attention() -> list[Variant]:
    return [(impl, [f"model.attention_impl={impl}"]) for impl in ("manual", "sdpa")]


def _precision() -> list[Variant]:
    return [(p, [f"train.precision={p}"]) for p in ("fp32", "bf16", "fp16")]


def _clipping() -> list[Variant]:
    # A deliberately aggressive LR so clipping has a chance to matter for stability.
    return [
        (f"clip{c}", ["train.lr=0.03", "train.min_lr=0.003", f"train.grad_clip={c}"])
        for c in (0, 1.0, 0.1)
    ]


SWEEPS: dict[str, Callable[[], list[Variant]]] = {
    "baseline": lambda: [("baseline", [])],
    "batch_accum": _batch_accum,
    "batch_size": _batch_size,
    "lr_schedule": _lr_schedule,
    "attention": _attention,
    "precision": _precision,
    "clipping": _clipping,
}


def run_variant(
    label: str, base: ExperimentConfig, overrides: list[str], seed: int
) -> dict[str, Any]:
    """Run one configuration and return a self-describing record (never raises on divergence)."""
    cfg = base.with_overrides([*overrides, f"train.seed={seed}", "train.out_dir=null"])
    record: dict[str, Any] = {
        "label": label,
        "seed": seed,
        "overrides": overrides,
        "config": cfg.to_dict(),
    }
    try:
        result = train(cfg, log=lambda _msg: None).to_dict()
        result.pop("history")  # keep the record compact; eval_history is retained
        record.update(status="ok", result=result)
    except TrainingInstabilityError as err:
        record.update(status="diverged", error=str(err))
    except ConfigError as err:
        record.update(status="invalid_config", error=str(err))
    return record


def run_sweep(
    sweep: str, base: ExperimentConfig, repeats: int = 1, base_seed: int | None = None
) -> dict[str, Any]:
    if sweep not in SWEEPS:
        raise ConfigError(f"unknown sweep {sweep!r}; choose from {sorted(SWEEPS)}")
    if repeats < 1:
        raise ConfigError("repeats must be >= 1")
    seed0 = base.train.seed if base_seed is None else base_seed
    records = [
        run_variant(label, base, overrides, seed0 + r)
        for label, overrides in SWEEPS[sweep]()
        for r in range(repeats)
    ]
    return {
        "sweep": sweep,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_info(),
        "repeats": repeats,
        "records": records,
    }


CSV_COLUMNS = [
    "label",
    "seed",
    "status",
    "device",
    "precision",
    "tokens_per_sec",
    "train_time_sec",
    "peak_memory_bytes",
    "peak_memory_kind",
    "final_val_loss",
    "final_train_loss",
    "notes",
]


def to_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for rec in report["records"]:
        res = rec.get("result", {})
        row = {"label": rec["label"], "seed": rec["seed"], "status": rec["status"]}
        row.update({k: res.get(k) for k in CSV_COLUMNS[3:-1]})
        row["notes"] = "; ".join(res.get("notes", [])) or rec.get("error", "")
        rows.append(row)
    return rows


def write_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["created_utc"].replace(":", "").replace("-", "")
    json_path = out_dir / f"{report['sweep']}_{stamp}.json"
    csv_path = json_path.with_suffix(".csv")
    json_path.write_text(json.dumps(report, indent=2))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(to_rows(report))
    return json_path, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a named optimization-experiment sweep.")
    parser.add_argument("--sweep", required=True, choices=sorted(SWEEPS))
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="K=V",
        help="override the base config, e.g. --set train.max_steps=30",
    )
    parser.add_argument("--repeats", type=int, default=1, help="runs per variant (different seeds)")
    parser.add_argument("--out-dir", default="experiments/results")
    args = parser.parse_args(argv)
    try:
        base = ExperimentConfig.from_yaml(args.config).with_overrides(args.overrides)
        report = run_sweep(args.sweep, base, repeats=args.repeats)
    except ConfigError as err:
        print(f"error: {err}")
        return 1
    json_path, csv_path = write_report(report, args.out_dir)
    for row in to_rows(report):
        print(
            {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in row.items()
                if k in ("label", "seed", "status", "tokens_per_sec", "final_val_loss")
            }
        )
    print(f"wrote {json_path} and {csv_path}")
    return 0
