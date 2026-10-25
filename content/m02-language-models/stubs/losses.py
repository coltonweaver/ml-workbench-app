"""Softmax and cross-entropy on Tensors."""

import numpy as np

from m02.tensor import Tensor


def softmax(x, axis=-1):
    """exp(x) / sum(exp(x)) along ``axis``. Build it from Tensor ops so
    gradients flow through it, and make it work for x = [1000, 1001, 1002]."""
    raise NotImplementedError


def log_softmax(x, axis=-1):
    """log(softmax(x)), computed without taking the log of a tiny number."""
    raise NotImplementedError


def cross_entropy(logits, targets):
    """Mean negative log-probability of ``targets`` (integer class ids) under
    softmax(logits), for logits of shape (N, C). Returns a scalar Tensor.

    Write it as one node with its own _backward: the gradient with respect to
    the logits has a simple closed form. Work it out on paper first.
    """
    raise NotImplementedError
