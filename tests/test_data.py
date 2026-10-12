import pytest
import torch

from pretraining_lab.config import ConfigError, DataConfig
from pretraining_lab.data import (
    CharTokenizer,
    get_batch,
    iter_eval_batches,
    load_text,
    make_synthetic_text,
    prepare_data,
    split_tokens,
)


def test_tokenizer_roundtrip_and_serialisation():
    tok = CharTokenizer.from_text("hello world")
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"
    assert CharTokenizer.from_dict(tok.to_dict()).chars == tok.chars


def test_tokenizer_rejects_unknown_characters():
    with pytest.raises(ValueError, match="not in the tokenizer"):
        CharTokenizer.from_text("abc").encode("abz")
    with pytest.raises(ValueError):
        CharTokenizer.from_text("")


def test_synthetic_text_is_deterministic_and_sized():
    a, b = make_synthetic_text(500, seed=1), make_synthetic_text(500, seed=1)
    assert a == b and len(a) == 500
    assert make_synthetic_text(500, seed=2) != a
    with pytest.raises(ValueError):
        make_synthetic_text(0, seed=1)


def test_split_has_no_leakage():
    ids = torch.arange(1000)
    train, val = split_tokens(ids, 0.9, min_len=10)
    assert len(train) + len(val) == 1000
    assert train.max() < val.min()  # contiguous, disjoint token positions
    # Every possible training window (inputs and targets) stays inside the training region.
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        x, y = get_batch(train, 8, 16, gen)
        assert y.max() < val.min() and x.max() < val.min()


def test_split_rejects_tiny_datasets():
    with pytest.raises(ValueError, match="too small"):
        split_tokens(torch.arange(30), 0.9, min_len=20)


def test_get_batch_shapes_and_shift():
    data = torch.arange(200)
    x, y = get_batch(data, 4, 8, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)  # on arange data the target is the next token


def test_get_batch_is_reproducible_and_bounded():
    data = torch.arange(50)
    a = get_batch(data, 4, 8, torch.Generator().manual_seed(7))
    b = get_batch(data, 4, 8, torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0])
    gen = torch.Generator().manual_seed(1)
    for _ in range(200):  # window ending exactly at the last token must be reachable & valid
        _, y = get_batch(data, 4, 8, gen)
        assert y.max() <= 49
    with pytest.raises(ValueError):
        get_batch(torch.arange(5), 2, 8, gen)


def test_eval_batches_are_deterministic_and_non_overlapping():
    data = torch.arange(100)
    first = list(iter_eval_batches(data, 4, 8, max_batches=10))
    second = list(iter_eval_batches(data, 4, 8, max_batches=10))
    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, second, strict=True))
    starts = [int(x[0]) for xb, _ in first for x in xb]
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert starts == sorted(set(starts)) and all(g == 8 for g in gaps)
    assert len(list(iter_eval_batches(data, 4, 8, max_batches=1))) == 1


def test_load_text_file_errors(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_text(DataConfig(source="file", path=str(tmp_path / "nope.txt")))
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n")
    with pytest.raises(ConfigError, match="empty"):
        load_text(DataConfig(source="file", path=str(empty)))
    good = tmp_path / "good.txt"
    good.write_text("abc " * 100)
    assert load_text(DataConfig(source="file", path=str(good))).startswith("abc")


def test_prepare_data_uses_given_tokenizer():
    cfg = DataConfig(source="synthetic", synthetic_chars=3000)
    tok, train, val = prepare_data(cfg, block_size=16)
    tok2, train2, _ = prepare_data(cfg, block_size=16, tokenizer=tok)
    assert tok2 is tok and torch.equal(train, train2) and len(val) > 18
