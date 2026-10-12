import dataclasses

import pytest
import torch

from pretraining_lab.generate import generate
from pretraining_lab.model import GPT


@pytest.fixture
def model(tiny_model_cfg):
    return GPT(tiny_model_cfg).eval()


def test_length_and_vocab_range(model):
    out = generate(model, torch.zeros(2, 3, dtype=torch.long), 7, temperature=1.0)
    assert out.shape == (2, 10)
    assert out.min() >= 0 and out.max() < 17


def test_seeded_sampling_is_reproducible(model):
    idx = torch.zeros(1, 2, dtype=torch.long)
    a = generate(model, idx, 20, 1.0, generator=torch.Generator().manual_seed(3))
    b = generate(model, idx, 20, 1.0, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_top_k_one_and_zero_temperature_are_greedy(model):
    idx = torch.zeros(1, 2, dtype=torch.long)
    greedy = generate(model, idx, 15, temperature=0)
    top1 = generate(model, idx, 15, 1.0, top_k=1, generator=torch.Generator().manual_seed(0))
    assert torch.equal(greedy, top1)


def test_top_k_restricts_choices(model):
    idx = torch.zeros(1, 1, dtype=torch.long)
    with torch.no_grad():
        top3 = set(torch.topk(model(idx)[0][0, -1], 3).indices.tolist())
    gen = torch.Generator().manual_seed(0)
    for _ in range(30):
        nxt = generate(model, idx, 1, 1.0, top_k=3, generator=gen)[0, -1].item()
        assert nxt in top3


def test_generation_beyond_context_length(model, tiny_model_cfg):
    out = generate(model, torch.zeros(1, 4, dtype=torch.long), tiny_model_cfg.block_size + 10)
    assert out.shape[1] == 4 + tiny_model_cfg.block_size + 10


def test_training_mode_is_restored(tiny_model_cfg):
    model = GPT(dataclasses.replace(tiny_model_cfg, dropout=0.1)).train()
    generate(model, torch.zeros(1, 1, dtype=torch.long), 2)
    assert model.training


@pytest.mark.parametrize(
    "kwargs",
    [{"temperature": -1.0}, {"temperature": float("nan")}, {"top_k": 0}, {"max_new_tokens": -1}],
)
def test_invalid_arguments(model, kwargs):
    args = {"max_new_tokens": 3, "temperature": 1.0, **kwargs}
    with pytest.raises(ValueError):
        generate(model, torch.zeros(1, 1, dtype=torch.long), **args)


def test_empty_prompt_rejected(model):
    with pytest.raises(ValueError):
        generate(model, torch.zeros(1, 0, dtype=torch.long), 3)
