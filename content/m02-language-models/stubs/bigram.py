"""A bigram language model from counts."""

import numpy as np

from m02.tokens import bigrams


def count_matrix(words, stoi):
    """counts[i, j] = how many times character j follows character i.
    An integer array of shape (len(stoi), len(stoi))."""
    raise NotImplementedError


def probabilities(counts, smoothing=1.0):
    """Row-normalize counts into next-character distributions. ``smoothing``
    is added to every count first, so nothing has probability zero."""
    raise NotImplementedError


def sample(P, itos, rng, max_len=50):
    """Generate one word: start at the boundary (id 0), draw the next
    character from the current row of P with ``rng.choice``, and stop when
    the boundary comes up again (or after ``max_len`` characters)."""
    raise NotImplementedError
