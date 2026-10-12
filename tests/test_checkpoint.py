import pytest
import torch

from pretraining_lab.checkpoint import (
    build_model_from_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from pretraining_lab.config import ExperimentConfig, ModelConfig
from pretraining_lab.model import GPT
from pretraining_lab.train import build_optimizer


def _make(tmp_path):
    cfg = ExperimentConfig(
        model=ModelConfig(vocab_size=11, block_size=8, n_layer=1, n_head=2, d_model=8)
    )
    model = GPT(cfg.model)
    opt = build_optimizer(model, cfg.train)
    x = torch.randint(0, 11, (4, 8))
    _, loss = model(x, x)
    loss.backward()
    opt.step()  # populate optimizer state
    path = tmp_path / "ck" / "m.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=opt,
        scaler=None,
        step=5,
        config=cfg,
        tokenizer_state={"type": "char", "chars": list("abcdefghijk")},
        data_rng_state=torch.Generator().manual_seed(1).get_state(),
        metrics={"best_val_loss": 1.5},
    )
    return cfg, model, opt, path


def test_roundtrip_restores_weights_and_state(tmp_path):
    _, model, opt, path = _make(tmp_path)
    ckpt = load_checkpoint(path)
    assert ckpt["step"] == 5 and ckpt["metrics"]["best_val_loss"] == 1.5
    restored = build_model_from_checkpoint(ckpt)
    for (n1, p1), (n2, p2) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        assert n1 == n2 and torch.equal(p1, p2)
    assert len(ckpt["optimizer"]["state"]) == len(opt.state_dict()["state"]) > 0


def test_save_is_atomic_and_leaves_no_temp_file(tmp_path):
    _, _, _, path = _make(tmp_path)
    assert path.exists() and not list(path.parent.glob("*.tmp"))


def test_missing_and_wrong_version(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")
    bad = tmp_path / "bad.pt"
    torch.save({"version": 999}, bad)
    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        load_checkpoint(bad)


def test_checkpoint_loads_into_either_attention_impl(tmp_path):
    cfg, _, _, path = _make(tmp_path)
    ckpt = load_checkpoint(path)
    ckpt["config"]["model"]["attention_impl"] = "sdpa"
    assert build_model_from_checkpoint(ckpt).cfg.attention_impl == "sdpa"
