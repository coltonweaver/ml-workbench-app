"""The bigram model again, learned by gradient descent."""

import numpy as np

from m02.losses import cross_entropy
from m02.tensor import Tensor
from m02.tokens import bigrams


def dataset(words, stoi):
    """Every bigram of ``words`` as two integer arrays: inputs and targets."""
    xs, ys = [], []
    for word in words:
        for a, b in bigrams(word):
            xs.append(stoi[a])
            ys.append(stoi[b])
    return np.array(xs), np.array(ys)


def train_bigram(xs, ys, vocab_size, steps=100, lr=10.0, reg=0.01, seed=0, log=None):
    """Learn W (vocab × vocab) so that softmax(W[x]) predicts y.

    Each step: logits = W[xs] (row lookup, the same as one-hot(xs) @ W), loss =
    cross-entropy + reg * mean(W²), backward, W -= lr * grad. Returns W and the
    list of losses. ``log(step, loss)`` is called each step if given.
    """
    rng = np.random.default_rng(seed)
    W = Tensor(rng.normal(size=(vocab_size, vocab_size)))
    losses = []
    for step in range(steps):
        logits = W[xs]
        loss = cross_entropy(logits, ys) + reg * (W ** 2).mean()
        W.grad = np.zeros_like(W.data)
        loss.backward()
        W.data -= lr * W.grad
        losses.append(float(loss.data))
        if log is not None:
            log(step, losses[-1])
    return W, losses
