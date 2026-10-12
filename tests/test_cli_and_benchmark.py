import json
from pathlib import Path

import pytest

from pretraining_lab import benchmark, evaluate, generate, train
from pretraining_lab.config import ConfigError, ExperimentConfig
from pretraining_lab.utils import environment_info

ROOT = Path(__file__).resolve().parents[1]
FAST = [
    "train.max_steps=6",
    "train.warmup_steps=1",
    "train.eval_interval=3",
    "train.eval_batches=2",
    "train.log_interval=100",
    "train.device=cpu",
    "data.synthetic_chars=3000",
    "model.block_size=16",
    "model.n_layer=1",
    "model.d_model=16",
    "model.n_head=2",
]


def _cli_args(tmp_path):
    args = ["--config", str(ROOT / "configs/tiny_cpu.yaml")]
    for item in [*FAST, f"train.out_dir={tmp_path / 'run'}"]:
        args += ["--set", item]
    return args


def test_cli_train_evaluate_generate_resume(tmp_path, capsys):
    assert train.main(_cli_args(tmp_path)) == 0
    ckpt = str(tmp_path / "run" / "last.pt")
    assert Path(ckpt).exists()
    assert evaluate.main(["--checkpoint", ckpt, "--batches", "2", "--device", "cpu"]) == 0
    assert (
        generate.main(
            [
                "--checkpoint",
                ckpt,
                "--prompt",
                "the ",
                "--max-new-tokens",
                "10",
                "--temperature",
                "0.8",
                "--top-k",
                "5",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "val loss" in out and "perplexity" in out
    assert train.main(_cli_args(tmp_path) + ["--resume", ckpt]) == 0  # already complete: no-op


def test_cli_reports_errors_cleanly(tmp_path, capsys):
    assert train.main(["--config", str(tmp_path / "missing.yaml")]) == 1
    assert "error:" in capsys.readouterr().out
    bad = _cli_args(tmp_path) + ["--set", "train.batch_size=0"]
    assert train.main(bad) == 1


def test_generate_and_evaluate_report_errors_cleanly(tmp_path, capsys):
    assert train.main(_cli_args(tmp_path)) == 0
    ckpt = str(tmp_path / "run" / "last.pt")
    code = generate.main(["--checkpoint", ckpt, "--prompt", "\u2603", "--device", "cpu"])
    assert code == 1
    assert "not in the tokenizer" in capsys.readouterr().out
    assert generate.main(["--checkpoint", str(tmp_path / "nope.pt")]) == 1
    assert evaluate.main(["--checkpoint", str(tmp_path / "nope.pt")]) == 1


def test_environment_info_has_required_fields():
    info = environment_info()
    for key in (
        "python",
        "torch",
        "platform",
        "cpu_model",
        "cpu_logical_cores",
        "torch_num_threads",
        "cuda_available",
    ):
        assert key in info


def test_every_sweep_builds_valid_configs():
    base = ExperimentConfig.from_yaml(ROOT / "configs/baseline.yaml")
    for name, factory in benchmark.SWEEPS.items():
        variants = factory()
        assert variants, name
        for _label, overrides in variants:
            base.with_overrides(overrides)  # raises ConfigError if a variant is invalid
        assert len({label for label, _ in variants}) == len(variants)


def test_batch_accum_sweep_keeps_effective_batch_constant():
    for _, overrides in benchmark.SWEEPS["batch_accum"]():
        values = dict(item.split("=") for item in overrides)
        assert int(values["train.batch_size"]) * int(values["train.grad_accum_steps"]) == 16


def test_run_sweep_records_everything(tmp_path):
    base = ExperimentConfig.from_yaml(ROOT / "configs/tiny_cpu.yaml").with_overrides(FAST)
    report = benchmark.run_sweep("attention", base, repeats=1)
    assert [r["label"] for r in report["records"]] == ["manual", "sdpa"]
    for rec in report["records"]:
        assert rec["status"] == "ok"
        res = rec["result"]
        for key in ("tokens_per_sec", "train_time_sec", "final_val_loss", "peak_memory_kind"):
            assert key in res
    assert "torch" in report["environment"]
    json_path, csv_path = benchmark.write_report(report, tmp_path)
    assert json.loads(json_path.read_text())["sweep"] == "attention"
    assert csv_path.read_text().splitlines()[0].startswith("label,seed,status")


def test_diverged_runs_are_recorded_not_raised():
    base = ExperimentConfig.from_yaml(ROOT / "configs/tiny_cpu.yaml").with_overrides(FAST)
    rec = benchmark.run_variant(
        "boom", base, ["train.lr=1e30", "train.min_lr=1e29", "train.grad_clip=0"], seed=1
    )
    assert rec["status"] in ("diverged", "ok")  # outcome is whatever actually happened
    assert "config" in rec


def test_unknown_sweep_and_repeats():
    base = ExperimentConfig()
    with pytest.raises(ConfigError, match="unknown sweep"):
        benchmark.run_sweep("nope", base)
    with pytest.raises(ConfigError, match="repeats"):
        benchmark.run_sweep("attention", base, repeats=0)
