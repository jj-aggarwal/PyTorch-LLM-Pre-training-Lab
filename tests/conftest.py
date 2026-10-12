import pytest
import torch

from pretraining_lab.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig


@pytest.fixture
def tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=17,
        block_size=16,
        n_layer=2,
        n_head=2,
        d_model=16,
        dropout=0.0,
        bias=True,
        attention_impl="manual",
    )


@pytest.fixture
def tiny_experiment(tmp_path) -> ExperimentConfig:
    """A tiny end-to-end configuration that trains in a few seconds on CPU."""
    return ExperimentConfig(
        model=ModelConfig(vocab_size=None, block_size=16, n_layer=1, n_head=2, d_model=16),
        data=DataConfig(source="synthetic", train_fraction=0.9, synthetic_chars=4000),
        train=TrainConfig(
            device="cpu",
            batch_size=8,
            max_steps=12,
            warmup_steps=2,
            lr=0.01,
            min_lr=0.001,
            eval_interval=6,
            eval_batches=2,
            log_interval=100,
            out_dir=str(tmp_path / "run"),
        ),
    )


@pytest.fixture(autouse=True)
def _deterministic_torch():
    torch.manual_seed(0)
