import pytest

from pretraining_lab.config import ConfigError, ExperimentConfig, ModelConfig, TrainConfig


def test_defaults_are_valid():
    ExperimentConfig().validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 30, "n_head": 4},  # not divisible
        {"n_layer": 0},
        {"block_size": 0},
        {"dropout": 1.0},
        {"attention_impl": "flash9000"},
    ],
)
def test_invalid_model_config(kwargs):
    with pytest.raises(ConfigError):
        ModelConfig(vocab_size=10, **kwargs).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scheduler": "exponential"},
        {"precision": "fp8"},
        {"batch_size": 0},
        {"warmup_steps": 500, "max_steps": 100},
        {"lr": 0.0},
        {"min_lr": 1.0, "lr": 0.1},
        {"grad_clip": -1.0},
    ],
)
def test_invalid_train_config(kwargs):
    with pytest.raises(ConfigError):
        TrainConfig(**kwargs).validate()


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError, match="unknown keys"):
        ExperimentConfig.from_dict({"train": {"learning_rate": 0.1}})
    with pytest.raises(ConfigError, match="unknown top-level"):
        ExperimentConfig.from_dict({"optimizer": {}})


def test_yaml_scientific_notation_string_is_coerced(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  lr: 3e-3\n  min_lr: 3e-4\n")  # YAML 1.1 reads these as strings
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.train.lr == pytest.approx(3e-3)


def test_overrides_and_errors():
    cfg = ExperimentConfig().with_overrides(["train.max_steps=500", "model.bias=false"])
    assert cfg.train.max_steps == 500 and cfg.model.bias is False
    with pytest.raises(ConfigError):
        ExperimentConfig().with_overrides(["train.nope=1"])
    with pytest.raises(ConfigError):
        ExperimentConfig().with_overrides(["no_equals_sign"])
    with pytest.raises(ConfigError):
        ExperimentConfig().with_overrides(["train.batch_size=1.5"])


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        ExperimentConfig.from_yaml(tmp_path / "missing.yaml")


def test_shipped_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("tiny_cpu.yaml", "baseline.yaml"):
        ExperimentConfig.from_yaml(root / name)
