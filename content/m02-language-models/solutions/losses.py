"""Softmax and cross-entropy on Tensors."""

import numpy as np

from m02.tensor import Tensor


def softmax(x, axis=-1):
    """exp(x) / sum(exp(x)) along ``axis``, stable for large inputs.
    Built from Tensor ops so gradients flow through it."""
    shift = x.data.max(axis=axis, keepdims=True)  # a constant: no gradient needed
    e = (x - shift).exp()
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    shift = x.data.max(axis=axis, keepdims=True)
    z = x - shift
    return z - z.exp().sum(axis=axis, keepdims=True).log()


def cross_entropy(logits, targets):
    """Mean negative log-probability of ``targets`` (integer class ids) under
    softmax(logits), for logits of shape (N, C).

    One node with a hand-written backward: d loss / d logits is
    (softmax(logits) - onehot(targets)) / N, which is cheaper and more stable
    than backpropagating through exp, sum and log separately.
    """
    targets = np.asarray(targets, dtype=int)
    n = logits.shape[0]
    z = logits.data - logits.data.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    out = Tensor(-logp[np.arange(n), targets].mean(), (logits,), "cross_entropy")

    def _backward():
        g = np.exp(logp)
        g[np.arange(n), targets] -= 1.0
        logits.grad += g / n * out.grad

    out._backward = _backward
    return out
