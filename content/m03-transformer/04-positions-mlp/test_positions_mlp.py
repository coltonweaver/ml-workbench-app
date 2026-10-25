import numpy as np

from m02.tensor import Tensor
from m03.layers import MLP, gelu, sinusoidal_positions

rng = np.random.default_rng(3)


def ref_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def test_gelu_values():
    x = np.linspace(-4, 4, 17)
    y = gelu(Tensor(x))
    assert isinstance(y, Tensor)
    assert np.allclose(y.data, ref_gelu(x), atol=1e-12)
    assert np.isclose(gelu(Tensor([0.0])).data[0], 0.0)


def test_gelu_gradient():
    x = Tensor(rng.normal(size=7) * 2)
    gelu(x).sum().backward()
    h = 1e-6
    numeric = (ref_gelu(x.data + h) - ref_gelu(x.data - h)) / (2 * h)
    assert np.allclose(x.grad, numeric, atol=1e-6)


def test_mlp():
    mlp = MLP(4, seed=1)
    for p in mlp.parameters():
        p.data = rng.normal(size=p.shape)
    x = rng.normal(size=(2, 3, 4))
    out = mlp(Tensor(x))
    assert out.shape == (2, 3, 4)
    expected = ref_gelu(x @ mlp.W1.data + mlp.b1.data) @ mlp.W2.data + mlp.b2.data
    assert np.allclose(out.data, expected), "gelu(x @ W1 + b1) @ W2 + b2"


def test_sinusoidal_positions():
    pe = sinusoidal_positions(10, 8)
    assert pe.shape == (10, 8)
    assert np.allclose(pe[0], [0, 1] * 4), "position 0: sin(0) = 0 in even dims, cos(0) = 1 in odd ones"
    i = np.arange(0, 8, 2)
    assert np.allclose(pe[3, 0::2], np.sin(3 / 10000 ** (i / 8)))
    assert np.allclose(pe[3, 1::2], np.cos(3 / 10000 ** (i / 8)))
    assert len({tuple(np.round(row, 6)) for row in pe}) == 10, "every position gets a different vector"


def test_relative_positions_are_linear():
    pe = sinusoidal_positions(20, 16)
    d = pe @ pe.T
    assert np.allclose(d[2, 5], d[10, 13]), "the dot product depends only on the offset between positions"
