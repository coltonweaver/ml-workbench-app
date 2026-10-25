"""How good is a language model? The likelihood of real data."""

import numpy as np

from m02.tokens import bigrams


def nll(P, words, stoi):
    """Average negative log-likelihood per bigram of ``words`` under the
    bigram probabilities P."""
    total, n = 0.0, 0
    for word in words:
        for a, b in bigrams(word):
            total -= np.log(P[stoi[a], stoi[b]])
            n += 1
    return total / n


def perplexity(avg_nll):
    """exp of the average NLL: the model is as unsure as a uniform choice
    among this many options."""
    return float(np.exp(avg_nll))
