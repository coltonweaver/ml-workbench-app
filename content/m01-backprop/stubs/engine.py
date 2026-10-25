"""Lessons 1.4 to 1.6: a scalar autograd engine.

A Value wraps one number and remembers how it was made, so that later we can
ask how the final result depends on every number that went into it.
"""

import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0  # d(final output) / d(this value); filled in by backward()
        self._backward = lambda: None  # pushes this node's grad to its children
        self._prev = set(_children)  # the Values this one was computed from
        self._op = _op  # the operation that made it, e.g. "+" or "*"

    def __repr__(self):
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    # ---- Lesson 1.4: build the graph ---------------------------------------

    def __add__(self, other):
        """Return a new Value for self + other. `other` may be a plain number:
        wrap it in a Value first. Record both inputs as children and "+" as
        the op."""
        raise NotImplementedError("Lesson 1.4: write __add__")

    def __mul__(self, other):
        """Same as __add__, for multiplication ("*")."""
        raise NotImplementedError("Lesson 1.4: write __mul__")

    def __radd__(self, other):  # 3 + Value(2) ends up here
        return self + other

    def __rmul__(self, other):  # 3 * Value(2) ends up here
        return self * other

    # ---- Lesson 1.5: backward ----------------------------------------------

    def backward(self):
        """Fill in .grad on every Value that self depends on.

        Order matters: a node may only push its grad to its children once
        every node that uses it has finished pushing into it.
        """
        raise NotImplementedError("Lesson 1.5: write backward")

    # ---- Lesson 1.6: more operations ---------------------------------------

    def __pow__(self, k):
        """self ** k for a plain number k (not a Value)."""
        raise NotImplementedError("Lesson 1.6: write __pow__")

    def __neg__(self):
        raise NotImplementedError("Lesson 1.6: write __neg__")

    def __sub__(self, other):
        raise NotImplementedError("Lesson 1.6: write __sub__")

    def __rsub__(self, other):  # 3 - Value(2)
        raise NotImplementedError("Lesson 1.6: write __rsub__")

    def __truediv__(self, other):
        raise NotImplementedError("Lesson 1.6: write __truediv__")

    def __rtruediv__(self, other):  # 3 / Value(2)
        raise NotImplementedError("Lesson 1.6: write __rtruediv__")

    def exp(self):
        raise NotImplementedError("Lesson 1.6: write exp")

    def tanh(self):
        raise NotImplementedError("Lesson 1.6: write tanh")

    def relu(self):
        raise NotImplementedError("Lesson 1.6: write relu")
