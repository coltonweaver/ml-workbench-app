import numpy as np

from m02.tensor import Tensor
from m03.layers import LayerNorm

rng = np.random.default_rng(2)


def numeric_grad(f, x, h=1e-6):
    g = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x.data[i]
        x.data[i] = old + h
        hi = float(f().data)
        x.data[i] = old - h
        lo = float(f().data)
        x.data[i] = old
        g[i] = (hi - lo) / (2 * h)
    return g


def test_normalizes_each_vector():
    ln = LayerNorm(6)
    x = rng.normal(loc=5.0, scale=3.0, size=(2, 4, 6))
    y = ln(Tensor(x)).data
    assert y.shape == x.shape
    assert np.allclose(y.mean(-1), 0, atol=1e-9), "each vector (last axis) should have mean 0"
    assert np.allclose(y.var(-1), 1, atol=1e-3), "and variance 1"
    expected = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    assert np.allclose(y, expected), "use the biased variance (divide by n) and eps inside the sqrt"


def test_gain_and_bias():
    ln = LayerNorm(3)
    ln.gamma.data = np.array([1.0, 2.0, 3.0])
    ln.beta.data = np.array([0.5, 0.0, -1.0])
    x = np.array([[1.0, 2.0, 3.0]])
    norm = (x - 2.0) / np.sqrt(2 / 3 + 1e-5)
    assert np.allclose(ln(Tensor(x)).data, norm * ln.gamma.data + ln.beta.data)


def test_constant_vector_is_safe():
    y = LayerNorm(4)(Tensor(np.full((1, 4), 7.0))).data
    assert np.all(np.isfinite(y)) and np.allclose(y, 0), "eps keeps a constant vector from dividing by 0"


def test_gradients():
    ln = LayerNorm(5)
    ln.gamma.data = rng.normal(size=5)
    ln.beta.data = rng.normal(size=5)
    x = Tensor(rng.normal(size=(3, 5)))
    w = rng.normal(size=(3, 5))
    f = lambda: (ln(x) * w).sum()  # noqa: E731
    for t in (x, ln.gamma, ln.beta):
        t.grad = np.zeros_like(t.data)
    f().backward()
    for name, t in (("x", x), ("gamma", ln.gamma), ("beta", ln.beta)):
        assert np.allclose(t.grad, numeric_grad(f, t), atol=1e-5), f"gradient for {name} is wrong"
