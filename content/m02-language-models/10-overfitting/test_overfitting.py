import numpy as np

from m02.evaluate import evaluate, generalization_gap, split
from m02.losses import cross_entropy
from m02.mlp_lm import MLPLanguageModel, build_dataset
from m02.tokens import build_vocab

WORDS = [f"w{i}" for i in range(1000)]


def test_split_sizes_and_coverage():
    train, val, test = split(WORDS)
    assert (len(train), len(val), len(test)) == (800, 100, 100)
    assert sorted(train + val + test) == sorted(WORDS), "every word lands in exactly one split"
    assert not set(train) & set(val) and not set(val) & set(test)


def test_split_shuffles_deterministically():
    original = list(WORDS)
    a = split(WORDS, seed=1)
    assert WORDS == original, "don't shuffle the caller's list in place"
    assert a == split(WORDS, seed=1), "the same seed gives the same split"
    assert a != split(WORDS, seed=2)
    assert a[0] != WORDS[:800], "shuffle first: datasets are often sorted, and then the splits differ"


def test_split_other_fractions():
    parts = split(WORDS[:10], fractions=(0.5, 0.5))
    assert [len(p) for p in parts] == [5, 5]
    parts = split(WORDS[:7], fractions=(0.7, 0.3))
    assert sum(len(p) for p in parts) == 7


def test_evaluate_matches_full_batch_loss():
    words = ["emma", "olivia", "ava", "isabella", "sophia"]
    stoi, itos = build_vocab(words)
    X, Y = build_dataset(words, stoi, 3)
    m = MLPLanguageModel(len(itos), n_embd=4, n_hidden=8, seed=0)
    expected = float(cross_entropy(m(X), Y).data)
    assert np.isclose(evaluate(m, X, Y), expected)
    assert np.isclose(evaluate(m, X, Y, batch_size=7), expected), (
        "batches can have different sizes: weight each by its number of rows")


def test_gap():
    assert generalization_gap(train_loss=1.5, val_loss=2.0) == 0.5
