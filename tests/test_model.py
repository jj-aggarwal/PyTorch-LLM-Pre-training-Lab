import dataclasses
import math

import pytest
import torch

from pretraining_lab.config import ConfigError
from pretraining_lab.model import GPT, estimate_num_parameters


def test_forward_shapes_and_loss(tiny_model_cfg):
    model = GPT(tiny_model_cfg)
    idx = torch.randint(0, 17, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, 17) and loss is None
    logits, loss = model(idx, idx)
    assert loss is not None and loss.dim() == 0


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_parameter_count_matches_closed_form(tiny_model_cfg, bias, impl):
    cfg = dataclasses.replace(tiny_model_cfg, bias=bias, attention_impl=impl)
    model = GPT(cfg)
    assert model.num_parameters() == estimate_num_parameters(cfg)
    assert model.num_parameters(exclude_position_embeddings=True) == (
        estimate_num_parameters(cfg) - cfg.block_size * cfg.d_model
    )


def test_weight_tying_counts_embedding_once(tiny_model_cfg):
    model = GPT(tiny_model_cfg)
    assert model.lm_head.weight is model.wte.weight


@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_attention_is_causal(tiny_model_cfg, impl):
    """Changing a token must not change the logits at any earlier position."""
    model = GPT(dataclasses.replace(tiny_model_cfg, attention_impl=impl)).eval()
    a = torch.randint(0, 17, (1, 12))
    b = a.clone()
    b[0, 8] = (b[0, 8] + 1) % 17
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-6)  # past positions unaffected
    assert not torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-6)  # position 8 onward changes


def test_manual_and_sdpa_agree(tiny_model_cfg):
    manual = GPT(tiny_model_cfg).eval()
    sdpa = GPT(dataclasses.replace(tiny_model_cfg, attention_impl="sdpa")).eval()
    sdpa.load_state_dict(manual.state_dict())
    idx = torch.randint(0, 17, (2, 16))
    with torch.no_grad():
        assert torch.allclose(manual(idx)[0], sdpa(idx)[0], atol=1e-5)


def test_initial_loss_is_near_log_vocab(tiny_model_cfg):
    model = GPT(tiny_model_cfg)
    idx = torch.randint(0, 17, (32, 16))
    # Targets must be independent of inputs: with tied embeddings, target == input would be
    # (slightly) easier than chance even at initialisation.
    targets = torch.randint(0, 17, (32, 16))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(17)) < 0.1


def test_loss_matches_manual_cross_entropy(tiny_model_cfg):
    model = GPT(tiny_model_cfg).eval()
    idx = torch.randint(0, 17, (2, 8))
    tgt = torch.randint(0, 17, (2, 8))
    logits, loss = model(idx, tgt)
    logp = torch.log_softmax(logits, dim=-1)
    expected = -logp.gather(-1, tgt.unsqueeze(-1)).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_residual_projections_use_scaled_init(tiny_model_cfg):
    model = GPT(tiny_model_cfg)
    expected = 0.02 / math.sqrt(2 * tiny_model_cfg.n_layer)
    stds = [p.std().item() for n, p in model.named_parameters() if n.endswith("proj.weight")]
    assert all(abs(s - expected) / expected < 0.35 for s in stds)


def test_input_validation(tiny_model_cfg):
    model = GPT(tiny_model_cfg)
    with pytest.raises(ValueError, match="exceeds the context length"):
        model(torch.zeros(1, 17, dtype=torch.long))
    with pytest.raises(ValueError, match="at least one token"):
        model(torch.zeros(1, 0, dtype=torch.long))
    with pytest.raises(ValueError, match="torch.long"):
        model(torch.zeros(1, 4))
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match="must match"):
        model(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 5, dtype=torch.long))
    with pytest.raises(IndexError):
        model(torch.full((1, 4), 999, dtype=torch.long))


def test_invalid_model_config_is_rejected(tiny_model_cfg):
    with pytest.raises(ConfigError):
        GPT(dataclasses.replace(tiny_model_cfg, d_model=15))
    with pytest.raises(ConfigError, match="vocab_size"):
        GPT(dataclasses.replace(tiny_model_cfg, vocab_size=None))
