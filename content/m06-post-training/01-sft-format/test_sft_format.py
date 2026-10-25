import pytest
import torch
import torch.nn.functional as F

from m06.sft import END, PAD, SEP, CharTokenizer, encode_example, masked_cross_entropy


def test_tokenizer_has_the_special_characters():
    tok = CharTokenizer("cab")
    assert set(tok.itos) == {"a", "b", "c", SEP, END, PAD} and len(tok) == 6
    assert tok.decode(tok.encode("ab>c.")) == "ab>c."


def test_encode_example_layout():
    tok = CharTokenizer("abcdefghijklmnopqrstuvwxyz")
    x, y, mask = encode_example(tok, "abc", "xy", block_size=10)
    assert x.shape == y.shape == mask.shape == (10,)
    assert tok.decode(x) == "abc>xy.___"
    assert tok.decode(y) == "bc>xy.____"
    # y holds b c > x y . _ _ _ _ : train only on x, y and the END
    assert mask.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_exact_fit_and_too_long():
    tok = CharTokenizer("ab")
    x, y, mask = encode_example(tok, "a", "bb", block_size=4)
    assert tok.decode(x) == "a>bb" and tok.decode(y) == ">bb." and mask.tolist() == [0, 1, 1, 1]
    with pytest.raises(ValueError):
        encode_example(tok, "aa", "bb", block_size=4)


def test_masked_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    targets = torch.randint(0, 7, (2, 5))
    mask = torch.tensor([[0, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float32)
    full = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1), reduction="none").reshape(2, 5)
    expected = (full[0, 1] + full[0, 2] + full[1, 0]) / 3
    assert torch.isclose(masked_cross_entropy(logits, targets, mask), expected)
    assert torch.isclose(masked_cross_entropy(logits, targets, torch.ones(2, 5)), full.mean())
