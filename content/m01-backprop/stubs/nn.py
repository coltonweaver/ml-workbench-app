"""Lesson 1.8: a small neural network on top of your engine."""

import random

from m01.engine import Value


class Neuron:
    def __init__(self, nin):
        """nin weights, each a Value drawn uniformly from [-1, 1], and one
        bias Value starting at 0."""
        raise NotImplementedError("Lesson 1.8: write Neuron.__init__")

    def __call__(self, x):
        """x is a list of nin numbers or Values. Return
        tanh(w0*x0 + w1*x1 + ... + b) as a Value."""
        raise NotImplementedError("Lesson 1.8: write Neuron.__call__")

    def parameters(self):
        """Every Value the training loop should adjust: weights then bias."""
        raise NotImplementedError("Lesson 1.8: write Neuron.parameters")


class Layer:
    def __init__(self, nin, nout):
        raise NotImplementedError("Lesson 1.8: write Layer.__init__")

    def __call__(self, x):
        """A list of nout Values, or a single Value when nout == 1."""
        raise NotImplementedError("Lesson 1.8: write Layer.__call__")

    def parameters(self):
        raise NotImplementedError("Lesson 1.8: write Layer.parameters")


class MLP:
    def __init__(self, nin, nouts):
        """nouts lists the size of each layer, e.g. MLP(3, [4, 4, 1])."""
        raise NotImplementedError("Lesson 1.8: write MLP.__init__")

    def __call__(self, x):
        raise NotImplementedError("Lesson 1.8: write MLP.__call__")

    def parameters(self):
        raise NotImplementedError("Lesson 1.8: write MLP.parameters")


def train(model, xs, ys, lr=0.05, steps=100):
    """Fit model to the examples (xs[i] should map to ys[i]).

    Each step: compute the loss (sum of squared errors over all examples),
    reset every parameter's grad to 0, call backward, then nudge every
    parameter against its grad by lr.

    Return the list of losses (plain floats), one per step.
    """
    raise NotImplementedError("Lesson 1.8: write train()")
