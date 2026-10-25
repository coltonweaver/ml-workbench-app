"""The bigram model again, learned by gradient descent."""

import numpy as np

from m02.losses import cross_entropy
from m02.tensor import Tensor
from m02.tokens import bigrams


def dataset(words, stoi):
    """Every bigram of ``words`` as two integer arrays: inputs and targets."""
    raise NotImplementedError


def train_bigram(xs, ys, vocab_size, steps=100, lr=10.0, reg=0.01, seed=0, log=None):
    """Learn W (vocab × vocab, start from rng.normal) so that softmax(W[x])
    predicts y.

    Each step: logits = W[xs], loss = cross-entropy + reg * mean(W²), zero the
    grad, backward, W.data -= lr * W.grad. Returns W and the list of losses
    (floats). Call ``log(step, loss)`` each step if it's given.
    """
    raise NotImplementedError
