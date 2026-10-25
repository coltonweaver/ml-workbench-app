"""A small tensor autograd engine on NumPy.

The same idea as module 1's Value, but each node holds a whole array, so one
operation (and one backward step) covers millions of numbers at once.
"""

from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape``, undoing NumPy broadcasting.

    If x of shape (3, 1) was broadcast to (3, 4) in the forward pass, each x
    entry was used 4 times, so its gradient is the sum over those 4 uses.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


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

    # ---- lesson 2.4: elementwise ops, broadcasting, sum, backward ----

    def __add__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = Tensor.wrap(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Tensor(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

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

    __radd__ = __add__
    __rmul__ = __mul__

    def backward(self):
        """Same algorithm as module 1, written without recursion so deep graphs
        (a GPT has thousands of nodes) don't hit Python's recursion limit."""
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
        other = Tensor.wrap(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2), self.shape)
            other.grad += unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, other.shape)

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        axes = axes or tuple(reversed(range(self.ndim)))
        out = Tensor(self.data.transpose(axes), (self,), "transpose")

        def _backward():
            self.grad += out.grad.transpose(np.argsort(axes))

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, index):
        if isinstance(index, Tensor):
            index = index.data.astype(int)
        out = Tensor(self.data[index], (self,), "[]")

        def _backward():
            # Repeated indices each add their share, so not grad[index] += ...
            np.add.at(self.grad, index, out.grad)

        out._backward = _backward
        return out
