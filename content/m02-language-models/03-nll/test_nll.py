import math

import numpy as np

from m02.nll import nll, perplexity
from m02.tokens import build_vocab


def test_uniform_model():
    stoi, _ = build_vocab(["abc"])
    V = len(stoi)
    P = np.full((V, V), 1 / V)
    assert math.isclose(nll(P, ["abc", "a"], stoi), math.log(V)), (
        "a uniform model has NLL log(V) on any data")


def test_perfect_model_has_zero_nll():
    stoi, _ = build_vocab(["ab"])
    P = np.zeros((3, 3))
    P[stoi["."], stoi["a"]] = 1
    P[stoi["a"], stoi["b"]] = 1
    P[stoi["b"], stoi["."]] = 1
    assert math.isclose(nll(P, ["ab"], stoi), 0.0, abs_tol=1e-12)


def test_average_is_per_bigram_not_per_word():
    stoi, _ = build_vocab(["ab"])
    P = np.full((3, 3), 0.5)
    # "ab" has 3 bigrams and "a" has 2: the mean is over all 5.
    P[stoi["a"], stoi["b"]] = 0.25
    expected = (4 * -math.log(0.5) + -math.log(0.25)) / 5
    assert math.isclose(nll(P, ["ab", "a"], stoi), expected), (
        "average over every bigram in every word, boundaries included")


def test_perplexity():
    assert math.isclose(perplexity(math.log(27)), 27.0)
    assert math.isclose(perplexity(0.0), 1.0)
