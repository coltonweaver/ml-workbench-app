"""A small tensor autograd engine on NumPy.

The same idea as module 1's Value, but each node holds a whole array, so one
operation (and one backward step) covers millions of numbers at once.

Lesson 2.4: unbroadcast, __add__, __mul__, __pow__ and sum.
Lesson 2.5: __matmul__, exp, log, tanh, relu, reshape, transpose, __getitem__.
Everything else is built from those and is already here.
"""

from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, undoing NumPy broadcasting.

    If x of shape (3, 1) was broadcast to (3, 4) in the forward pass, each x
    entry was used 4 times, so its gradient is the sum over those 4 uses.
    Broadcasting can also add leading axes: (4,) used as (2, 3, 4).
    """
    raise NotImplementedError("lesson 2.4: unbroadcast")


class Tensor:
    def __init__(self, data, _prev=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = tuple(_prev)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    # ---- lesson 2.4: elementwise ops, broadcasting, sum ----

    def __add__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            raise NotImplementedError("lesson 2.4: gradient of +")

        out._backward = _backward
        return out

    def __mul__(self, other):
        raise NotImplementedError("lesson 2.4: elementwise *")

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        raise NotImplementedError("lesson 2.4: ** a constant")

    def sum(self, axis=None, keepdims=False):
        """Sum over ``axis`` (None means everything), like ndarray.sum."""
        raise NotImplementedError("lesson 2.4: sum")

    # Built from the ops above.

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod(
            [self.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Tensor.wrap(other))

    def __rsub__(self, other):
        return Tensor.wrap(other) + (-self)

    def __truediv__(self, other):
        return self * Tensor.wrap(other) ** -1

    def __rtruediv__(self, other):
        return Tensor.wrap(other) * self ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        """The same algorithm as module 1, written without recursion so deep
        graphs (a GPT has thousands of nodes) don't hit Python's recursion
        limit. Already done: you wrote this once."""
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            v, expanded = stack.pop()
            if expanded:
                topo.append(v)
                continue
            if id(v) in seen:
                continue
            seen.add(id(v))
            stack.append((v, True))
            for child in v._prev:
                if id(child) not in seen:
                    stack.append((child, False))
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    # ---- lesson 2.5: matmul, nonlinearities, reshaping, indexing ----

    def __matmul__(self, other):
        """Matrix product, batched over leading axes like NumPy's @."""
        raise NotImplementedError("lesson 2.5: @")

    def exp(self):
        raise NotImplementedError("lesson 2.5: exp")

    def log(self):
        raise NotImplementedError("lesson 2.5: log")

    def tanh(self):
        raise NotImplementedError("lesson 2.5: tanh")

    def relu(self):
        raise NotImplementedError("lesson 2.5: relu")

    def reshape(self, *shape):
        raise NotImplementedError("lesson 2.5: reshape")

    def transpose(self, *axes):
        """Permute axes like ndarray.transpose; no axes reverses them."""
        raise NotImplementedError("lesson 2.5: transpose")

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, index):
        """Indexing and row lookup: ``W[ids]`` picks rows of W. ``index`` may
        be a Tensor of ids."""
        raise NotImplementedError("lesson 2.5: indexing")
