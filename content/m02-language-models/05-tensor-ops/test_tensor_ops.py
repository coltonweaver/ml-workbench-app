import numpy as np

from m02.tensor import Tensor


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


def check(f, *xs):
    for x in xs:
        x.grad = np.zeros_like(x.data)
    f().backward()
    for k, x in enumerate(xs):
        expected = numeric_grad(f, x)
        assert np.allclose(x.grad, expected, atol=1e-5), (
            f"gradient of input {k} (shape {x.shape}) is wrong:\nyours\n{x.grad}\nexpected\n{expected}")


rng = np.random.default_rng(0)


def test_matmul_2d():
    a, b = Tensor(rng.normal(size=(3, 4))), Tensor(rng.normal(size=(4, 2)))
    assert np.allclose((a @ b).data, a.data @ b.data)
    w = rng.normal(size=(3, 2))
    check(lambda: ((a @ b) * w).sum(), a, b)


def test_matmul_batched_and_broadcast():
    a, b = Tensor(rng.normal(size=(2, 3, 4))), Tensor(rng.normal(size=(4, 5)))
    assert (a @ b).shape == (2, 3, 5)
    w = rng.normal(size=(2, 3, 5))
    check(lambda: ((a @ b) * w).sum(), a, b)
    q, k = Tensor(rng.normal(size=(2, 3, 4))), Tensor(rng.normal(size=(2, 4, 3)))
    w2 = rng.normal(size=(2, 3, 3))
    check(lambda: ((q @ k) * w2).sum(), q, k)


def test_nonlinearities():
    x = Tensor(rng.normal(size=(5,)))
    check(lambda: x.exp().sum(), x)
    check(lambda: x.tanh().sum(), x)
    w = rng.normal(size=5)
    check(lambda: (x.relu() * w).sum(), x)
    p = Tensor(rng.uniform(0.5, 2.0, size=(5,)))
    check(lambda: p.log().sum(), p)
    assert np.allclose(Tensor([-1.0, 2.0]).relu().data, [0, 2])


def test_reshape_and_transpose():
    a = Tensor(rng.normal(size=(2, 3, 4)))
    assert a.reshape(6, 4).shape == (6, 4)
    assert a.reshape(2, -1).shape == (2, 12)
    w = rng.normal(size=(6, 4))
    check(lambda: (a.reshape(6, 4) * w).sum(), a)
    assert a.transpose(2, 0, 1).shape == (4, 2, 3)
    assert a.T.shape == (4, 3, 2)
    w2 = rng.normal(size=(4, 2, 3))
    check(lambda: (a.transpose(2, 0, 1) * w2).sum(), a)


def test_row_lookup():
    W = Tensor(rng.normal(size=(5, 3)))
    ids = np.array([[0, 2], [2, 4]])
    out = W[ids]
    assert out.shape == (2, 2, 3)
    assert np.allclose(out.data[1, 0], W.data[2])
    out.sum().backward()
    assert np.allclose(W.grad[2], 2), "row 2 was looked up twice, so its gradient counts twice"
    assert np.allclose(W.grad[1], 0) and np.allclose(W.grad[3], 0), "rows never used get no gradient"


def test_lookup_with_tensor_ids_and_slices():
    W = Tensor(rng.normal(size=(4, 3)))
    assert np.allclose(W[Tensor([1, 1])].data, W.data[[1, 1]])
    w = rng.normal(size=(2, 3))
    check(lambda: (W[1:3] * w).sum(), W)
    w4 = rng.normal(size=4)
    check(lambda: (W[:, -1] * w4).sum(), W)


def test_lookup_equals_one_hot_matmul():
    W = Tensor(rng.normal(size=(6, 4)))
    ids = np.array([3, 0, 3, 5])
    W[ids].sum().backward()
    via_lookup = W.grad.copy()
    W.grad = np.zeros_like(W.data)
    (Tensor(np.eye(6)[ids]) @ W).sum().backward()
    assert np.allclose(via_lookup, W.grad), "W[ids] should behave exactly like one_hot(ids) @ W"


def test_mlp_layer_end_to_end():
    x = Tensor(rng.normal(size=(8, 3)))
    W, b = Tensor(rng.normal(size=(3, 4))), Tensor(rng.normal(size=(4,)))
    check(lambda: ((x @ W + b).tanh() ** 2).mean(), W, b)
