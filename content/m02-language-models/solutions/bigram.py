"""A bigram language model from counts."""

import numpy as np

from m02.tokens import bigrams


def count_matrix(words, stoi):
    """counts[i, j] = how many times character j follows character i."""
    n = len(stoi)
    counts = np.zeros((n, n), dtype=np.int64)
    for word in words:
        for a, b in bigrams(word):
            counts[stoi[a], stoi[b]] += 1
    return counts


def probabilities(counts, smoothing=1.0):
    """Row-normalize counts into next-character distributions. ``smoothing``
    is added to every count first, so nothing has probability zero."""
    c = counts.astype(np.float64) + smoothing
    return c / c.sum(axis=1, keepdims=True)


def sample(P, itos, rng, max_len=50):
    """Generate one word: start at the boundary, draw the next character from
    the current row of P, stop at the boundary."""
    out = []
    i = 0
    while len(out) < max_len:
        i = rng.choice(len(itos), p=P[i])
        if i == 0:
            break
        out.append(itos[i])
    return "".join(out)
