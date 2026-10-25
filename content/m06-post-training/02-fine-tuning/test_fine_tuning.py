import numpy as np
import torch

from m03.torch_gpt import TorchGPT
from m06.sft import CharTokenizer, complete, exact_match, sft
from workbench.data import names


def tokenizer():
    return CharTokenizer("abcdefghijklmnopqrstuvwxyz")


def data():
    ns = [n for n in names() if len(n) <= 5]
    np.random.default_rng(0).shuffle(ns)
    return [(n, n[::-1]) for n in ns[:2000]], [(n, n[::-1]) for n in ns[2000:2100]]


def test_learns_to_reverse_unseen_names():
    TOK = tokenizer()
    train, held_out = data()
    torch.manual_seed(0)
    m = TorchGPT(len(TOK), 12, n_embd=32, n_head=4, n_layer=2)
    seen = []
    losses = sft(m, TOK, train, 300, lr=3e-3, log=lambda s, l: seen.append(s))
    assert len(losses) == 300 and seen == list(range(300))
    assert losses[-1] < 0.1 * losses[0]
    acc = exact_match(m, TOK, held_out)
    assert acc > 0.8, f"held-out exact match {acc:.0%}"
    assert complete(m, TOK, "ann") == "nna"


class Scripted(torch.nn.Module):
    """Always predicts the next character of a fixed script."""

    def __init__(self, script):
        super().__init__()
        self.block_size = 32
        self.vocab = len(tokenizer())
        self.script = tokenizer().encode(script)

    def forward(self, idx):
        T = idx.shape[1]
        logits = torch.full((idx.shape[0], T, self.vocab), -10.0)
        for t in range(T):
            if t + 1 < len(self.script):
                logits[:, t, self.script[t + 1]] = 10.0
        return logits


def test_complete_stops_at_end_and_starts_after_sep():
    TOK = tokenizer()
    assert complete(Scripted("ab>xyz.qq"), TOK, "ab") == "xyz"
    assert complete(Scripted("ab>xyzw"), TOK, "ab", max_new=3) == "xyz"
    assert exact_match(Scripted("ab>xyz."), TOK, [("ab", "xyz"), ("ab", "xy")]) == 0.5
