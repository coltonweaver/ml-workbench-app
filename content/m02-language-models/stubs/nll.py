"""How good is a language model? The likelihood of real data."""

import numpy as np

from m02.tokens import bigrams


def nll(P, words, stoi):
    """Average negative log-likelihood per bigram of ``words`` under the
    bigram probabilities P: the mean of -log P[a, b] over every bigram (a, b),
    boundaries included."""
    raise NotImplementedError


def perplexity(avg_nll):
    """exp of the average NLL."""
    raise NotImplementedError
