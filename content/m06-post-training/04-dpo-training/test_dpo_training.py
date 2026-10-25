import numpy as np
import torch

from m03.torch_gpt import TorchGPT
from m06.dpo import dpo_train, sample
from m06.sft import CharTokenizer, sft
from workbench.data import names

TOK = CharTokenizer("abcdefghijklmnopqrstuvwxyz")


def vowel(n):
    return bool(n) and n[0] in "aeiou"


def test_dpo_shifts_the_preference():
    pool = [n for n in names() if len(n) <= 8]
    torch.manual_seed(0)
    m = TorchGPT(len(TOK), 12, n_embd=32, n_head=4, n_layer=2)
    sft(m, TOK, [("", n) for n in pool], 500, lr=3e-3)
    before = sample(m, TOK, "", 300, seed=1)
    assert np.mean([vowel(n) for n in before]) < 0.4, "the base generator mostly starts with consonants"
    triples = [("", a, b) for a, b in zip([n for n in pool if vowel(n)], [n for n in pool if not vowel(n)])]
    frozen = [p.detach().clone() for p in m.parameters()]
    history = dpo_train(m, TOK, triples, 150, beta=0.1)
    assert len(history) == 150 and set(history[0]) == {"loss", "margin"}
    assert abs(history[0]["loss"] - np.log(2)) < 1e-4, "policy == reference on the first step"
    assert history[-1]["margin"] > 0.5 and history[-1]["loss"] < 0.4
    after = sample(m, TOK, "", 300, seed=1)
    assert np.mean([vowel(n) for n in after]) > 0.7
    assert np.mean([len(n) >= 3 for n in after]) > 0.8, "still name-shaped"
    assert any(not torch.equal(a, b) for a, b in zip(frozen, m.parameters())), "the policy itself is trained"


@torch.no_grad()
def test_sample_cuts_at_end():
    torch.manual_seed(0)
    m = TorchGPT(len(TOK), 12, n_embd=16, n_head=2, n_layer=1)
    out = sample(m, TOK, "ab", 5, max_new=6, seed=3)
    assert len(out) == 5 and all(len(o) <= 6 and "." not in o for o in out)
    assert out == sample(m, TOK, "ab", 5, max_new=6, seed=3), "same seed, same samples"
