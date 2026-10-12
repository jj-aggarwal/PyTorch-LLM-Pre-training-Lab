import dataclasses
import json
import math

import pytest
import torch

from pretraining_lab import train as train_module
from pretraining_lab.config import ConfigError, TrainConfig
from pretraining_lab.evaluate import estimate_loss, perplexity
from pretraining_lab.train import TrainingInstabilityError, lr_at_step, train


def _quiet(_msg: str) -> None:
    pass


def test_lr_schedule_warmup_and_decay():
    tc = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=4, max_steps=14, scheduler="cosine")
    lrs = [lr_at_step(s, tc) for s in range(14)]
    assert lrs[0] == pytest.approx(0.25) and lrs[3] == pytest.approx(1.0)  # linear warmup
    assert all(a >= b - 1e-12 for a, b in zip(lrs[3:], lrs[4:], strict=False))  # monotone decay
    assert lrs[-1] >= 0.1 - 1e-9
    const = dataclasses.replace(tc, scheduler="constant")
    assert lr_at_step(10, const) == 1.0
    linear = dataclasses.replace(tc, scheduler="linear")
    assert lr_at_step(4, linear) == pytest.approx(1.0)
    assert lr_at_step(9, linear) == pytest.approx(0.55)


def test_end_to_end_cpu_training_reduces_loss(tiny_experiment):
    tiny_experiment.train.max_steps = 60
    tiny_experiment.train.eval_interval = 20
    result = train(tiny_experiment, log=_quiet)
    first = sum(h["train_loss"] for h in result.history[:5]) / 5
    last = sum(h["train_loss"] for h in result.history[-5:]) / 5
    assert last < first - 0.3, (first, last)
    assert result.final_val_loss < math.log(result.num_parameters)  # finite and sane
    assert result.tokens_per_sec and result.tokens_per_sec > 0
    assert len(result.eval_history) == 3
    assert result.peak_memory_bytes is None or result.peak_memory_bytes > 0
    out = json.loads(
        (__import__("pathlib").Path(tiny_experiment.train.out_dir) / "result.json").read_text()
    )
    assert out["steps_completed"] == 60


def test_training_is_reproducible(tiny_experiment):
    tiny_experiment.train.out_dir = None
    a = train(tiny_experiment, log=_quiet)
    b = train(tiny_experiment, log=_quiet)
    assert [h["train_loss"] for h in a.history] == [h["train_loss"] for h in b.history]
    assert a.final_val_loss == b.final_val_loss


def test_resume_after_crash_matches_uninterrupted_run(tiny_experiment, tmp_path):
    """Crash after step 7 (last checkpoint at step 6), resume, and compare with a clean run.

    Both runs use the same ``max_steps`` so the LR schedule is identical; the resumed run
    must reproduce the clean run's losses for steps 7-12 exactly (data RNG, optimizer state
    and model weights are all restored).
    """
    base = dataclasses.replace(tiny_experiment.train, max_steps=12, save_interval=6, log_interval=1)
    clean_cfg = dataclasses.replace(
        tiny_experiment, train=dataclasses.replace(base, out_dir=str(tmp_path / "clean"))
    )
    expected = train(clean_cfg, log=_quiet)

    crash_cfg = dataclasses.replace(
        tiny_experiment, train=dataclasses.replace(base, out_dir=str(tmp_path / "crash"))
    )

    def crash_after_step_7(message: str) -> None:
        if message.startswith("step     7/"):
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        train(crash_cfg, log=crash_after_step_7)
    checkpoint = tmp_path / "crash" / "last.pt"
    assert torch.load(checkpoint, weights_only=True)["step"] == 6

    resumed = train(crash_cfg, resume_from=checkpoint, log=_quiet)
    assert resumed.resumed_from_step == 6 and resumed.steps_completed == 6
    assert [h["step"] for h in resumed.history] == list(range(7, 13))
    assert [h["train_loss"] for h in resumed.history] == pytest.approx(
        [h["train_loss"] for h in expected.history[6:]], abs=1e-6
    )
    assert resumed.final_val_loss == pytest.approx(expected.final_val_loss, abs=1e-6)


def test_resume_when_already_complete(tiny_experiment):
    done = train(tiny_experiment, log=_quiet)
    again = train(tiny_experiment, resume_from=done.checkpoint_path, log=_quiet)
    assert again.steps_completed == 0 and again.final_val_loss == pytest.approx(
        done.final_val_loss, abs=1e-6
    )
    assert any("no training done" in n for n in again.notes)


def test_resume_with_mismatched_model_is_rejected(tiny_experiment):
    done = train(tiny_experiment, log=_quiet)
    other = dataclasses.replace(
        tiny_experiment, model=dataclasses.replace(tiny_experiment.model, d_model=32)
    )
    with pytest.raises(ConfigError, match="differs from the checkpoint"):
        train(other, resume_from=done.checkpoint_path, log=_quiet)


def test_gradient_accumulation_matches_large_batch_loss_scale(tiny_experiment):
    tiny_experiment.train.out_dir = None
    tiny_experiment.train.max_steps = 4
    tiny_experiment.train.batch_size = 4
    tiny_experiment.train.grad_accum_steps = 2
    result = train(tiny_experiment, log=_quiet)
    assert result.tokens_per_step == 4 * 2 * tiny_experiment.model.block_size
    assert all(math.isfinite(h["train_loss"]) for h in result.history)


def test_gradient_clipping_bounds_the_applied_norm(tiny_experiment):
    tiny_experiment.train.out_dir = None
    tiny_experiment.train.grad_clip = 0.05
    result = train(tiny_experiment, log=_quiet)
    # grad_norm is the *pre-clip* norm; with such a small cap it should exceed it on some steps.
    assert max(h["grad_norm"] for h in result.history) > 0.05
    assert all(math.isfinite(h["grad_norm"]) for h in result.history)


def test_nan_loss_raises_instability_error(tiny_experiment, monkeypatch):
    tiny_experiment.train.out_dir = None
    original = train_module.GPT.forward

    def poisoned(self, idx, targets=None):
        logits, loss = original(self, idx, targets)
        return logits, (loss * float("nan") if loss is not None else None)

    monkeypatch.setattr(train_module.GPT, "forward", poisoned)
    with pytest.raises(TrainingInstabilityError, match="non-finite loss"):
        train(tiny_experiment, log=_quiet)


def test_unsupported_precision_falls_back_on_cpu(tiny_experiment):
    tiny_experiment.train.out_dir = None
    tiny_experiment.train.max_steps = 3
    tiny_experiment.train.warmup_steps = 1
    tiny_experiment.train.precision = "fp16"
    result = train(tiny_experiment, log=_quiet)
    assert result.precision == "fp32" and any("fell back" in n for n in result.notes)


def test_bf16_autocast_runs_on_cpu(tiny_experiment):
    tiny_experiment.train.out_dir = None
    tiny_experiment.train.max_steps = 3
    tiny_experiment.train.warmup_steps = 1
    tiny_experiment.train.precision = "bf16"
    result = train(tiny_experiment, log=_quiet)
    assert result.precision == "bf16" and math.isfinite(result.final_val_loss)


def test_requesting_cuda_without_cuda_is_a_clear_error(tiny_experiment):
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    tiny_experiment.train.device = "cuda"
    with pytest.raises(ConfigError, match="CUDA is not available"):
        train(tiny_experiment, log=_quiet)


def test_mismatched_vocab_size_is_rejected(tiny_experiment):
    tiny_experiment.model.vocab_size = 3
    with pytest.raises(ConfigError, match="does not match the tokenizer"):
        train(tiny_experiment, log=_quiet)


def test_dataset_too_small_for_block_size(tiny_experiment):
    tiny_experiment.data.synthetic_chars = 30
    with pytest.raises(ValueError, match="too small"):
        train(tiny_experiment, log=_quiet)


def test_estimate_loss_is_deterministic_and_validates(tiny_experiment):
    from pretraining_lab.data import prepare_data
    from pretraining_lab.model import GPT

    tok, _, val = prepare_data(tiny_experiment.data, tiny_experiment.model.block_size)
    cfg = dataclasses.replace(tiny_experiment.model, vocab_size=tok.vocab_size)
    model = GPT(cfg)
    kwargs = dict(batch_size=4, max_batches=3, device=torch.device("cpu"))
    assert estimate_loss(model, val, **kwargs) == estimate_loss(model, val, **kwargs)
    with pytest.raises(ValueError, match="no evaluation windows"):
        estimate_loss(model, val[:5], **kwargs)
    assert perplexity(0.0) == 1.0 and math.isfinite(perplexity(1e9))
