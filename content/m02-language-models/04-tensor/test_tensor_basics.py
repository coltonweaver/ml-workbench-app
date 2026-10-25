import numpy as np
import pytest

from m02.tensor import Tensor, unbroadcast


def numeric_grad(f, x, h=1e-6):
    """d f() / d x.data by central differences, entry by entry."""
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


def check(f, *xs):
    for x in xs:
        x.grad = np.zeros_like(x.data)
    f().backward()
    for k, x in enumerate(xs):
        expected = numeric_grad(f, x)
        assert np.allclose(x.grad, expected, atol=1e-5), (
            f"gradient of input {k} (shape {x.shape}) is wrong:\nyours\n{x.grad}\nexpected\n{expected}")


def test_unbroadcast():
    g = np.ones((2, 3, 4))
    assert unbroadcast(g, (2, 3, 4)).shape == (2, 3, 4)
    assert np.all(unbroadcast(g, (4,)) == 6), "(4,) used as (2, 3, 4): each entry used 6 times"
    assert np.all(unbroadcast(g, (3, 1)) == 8), "(3, 1) used as (2, 3, 4): each entry used 8 times"
    assert unbroadcast(g, (3, 1)).shape == (3, 1)
    assert unbroadcast(g, ()).shape == () and unbroadcast(g, ()) == 24


def test_add_same_shape():
    a, b = Tensor([1.0, 2.0]), Tensor([3.0, 4.0])
    c = a + b
    assert np.allclose(c.data, [4, 6])
    c.sum().backward()
    assert np.allclose(a.grad, 1) and np.allclose(b.grad, 1)


def test_add_broadcasting_gradient():
    a = Tensor(np.zeros((3, 1)))
    b = Tensor(np.zeros((1, 4)))
    out = a + b
    assert out.shape == (3, 4)
    out.sum().backward()
    assert a.grad.shape == (3, 1), "a gradient has the shape of its tensor"
    assert np.allclose(a.grad, 4), "each a entry was used in 4 sums"
    assert np.allclose(b.grad, 3), "each b entry was used in 3 sums"


def test_add_plain_numbers():
    a = Tensor([1.0, 2.0])
    out = (1 + a) + 2
    assert np.allclose(out.data, [4, 5])


def test_mul_gradients():
    rng = np.random.default_rng(0)
    a, b = Tensor(rng.normal(size=(2, 3))), Tensor(rng.normal(size=(3,)))
    check(lambda: (a * b).sum(), a, b)
    check(lambda: (a * 3.0).sum(), a)


def test_pow():
    a = Tensor([1.0, 2.0, 3.0])
    assert np.allclose((a ** 2).data, [1, 4, 9])
    check(lambda: (a ** 3).sum(), a)
    check(lambda: (a ** -1).sum(), a)


@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, True), ((0, 2), False)])
def test_sum_gradients(axis, keepdims):
    rng = np.random.default_rng(1)
    a = Tensor(rng.normal(size=(2, 3, 4)))
    w = Tensor(rng.normal(size=np.sum(a.data, axis=axis, keepdims=keepdims).shape))
    s = a.sum(axis=axis, keepdims=keepdims)
    assert s.shape == w.shape
    check(lambda: (a.sum(axis=axis, keepdims=keepdims) * w).sum(), a)


def test_reused_node_accumulates():
    a = Tensor([3.0])
    (a * a + a).sum().backward()
    assert np.allclose(a.grad, 7.0), "d/da (a² + a) = 2a + 1 = 7: a is used three times"


def test_composite_expression():
    rng = np.random.default_rng(2)
    x, y = Tensor(rng.normal(size=(4, 1))), Tensor(rng.normal(size=(1, 5)))
    check(lambda: ((x * y + x) ** 2 / (y * y + 1)).mean(), x, y)
