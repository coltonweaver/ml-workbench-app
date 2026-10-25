import numpy as np

from m02.bigram import count_matrix, probabilities, sample
from m02.tokens import build_vocab

WORDS = ["emma", "ava", "anna", "eva"]


def test_counts():
    stoi, itos = build_vocab(WORDS)
    c = count_matrix(WORDS, stoi)
    assert c.shape == (len(itos), len(itos))
    assert np.issubdtype(c.dtype, np.integer), "counts should be integers"
    assert c[stoi["."], stoi["a"]] == 2, "two words start with 'a'"
    assert c[stoi["a"], stoi["."]] == 4, "all four words end in 'a'"
    assert c[stoi["n"], stoi["n"]] == 1
    assert c.sum() == sum(len(w) + 1 for w in WORDS), "every bigram is counted once"


def test_probabilities_rows_sum_to_one():
    stoi, _ = build_vocab(WORDS)
    c = count_matrix(WORDS, stoi)
    P = probabilities(c, smoothing=0.0)
    assert np.allclose(P.sum(axis=1)[c.sum(axis=1) > 0], 1.0), "each row is a distribution"
    assert P[stoi["."], stoi["a"]] == 0.5, "2 of the 4 starts are 'a'"
    S = probabilities(c, smoothing=1.0)
    assert (S > 0).all(), "with smoothing nothing has probability zero"
    assert np.allclose(S.sum(axis=1), 1.0)
    assert np.isclose(S[stoi["."], stoi["a"]], 3 / (4 + len(stoi)))
    assert c.sum() == sum(len(w) + 1 for w in WORDS), "probabilities shouldn't change the counts"


def test_sample_stops_at_boundary():
    stoi, itos = build_vocab(["ab"])
    P = probabilities(count_matrix(["ab"], stoi), smoothing=0.0)
    rng = np.random.default_rng(0)
    assert sample(P, itos, rng) == "ab", "this model can only ever say 'ab'"


def test_sample_follows_the_distribution():
    stoi, itos = build_vocab(["a", "b"])
    P = probabilities(count_matrix(["a", "a", "a", "b"], stoi), smoothing=0.0)
    rng = np.random.default_rng(1)
    words = [sample(P, itos, rng) for _ in range(2000)]
    share = words.count("a") / len(words)
    assert abs(share - 0.75) < 0.04, f"'a' should come up 75% of the time, got {share:.0%}"


def test_sample_respects_max_len():
    stoi, itos = build_vocab(["aa"])
    P = np.array([[0.0, 1.0], [0.0, 1.0]])  # never ends
    assert len(sample(P, itos, np.random.default_rng(0), max_len=7)) == 7
