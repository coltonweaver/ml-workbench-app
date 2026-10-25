import numpy as np

from m02.losses import cross_entropy, log_softmax, softmax
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


rng = np.random.default_rng(0)


def reference_softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def test_softmax_values():
    x = rng.normal(size=(4, 5))
    s = softmax(Tensor(x))
    assert isinstance(s, Tensor), "return a Tensor so gradients can flow"
    assert np.allclose(s.data, reference_softmax(x))
    assert np.allclose(s.data.sum(axis=-1), 1.0)
    assert np.allclose(softmax(Tensor(x), axis=0).data.sum(axis=0), 1.0)


def test_softmax_is_stable():
    s = softmax(Tensor([[1000.0, 1001.0, 1002.0]]))
    assert np.all(np.isfinite(s.data)), "exp(1000) overflows: subtract the max first"
    assert np.allclose(s.data, reference_softmax(np.array([[1.0, 2.0, 3.0]])))


def test_softmax_gradient():
    x = Tensor(rng.normal(size=(3, 4)))
    w = rng.normal(size=(3, 4))
    f = lambda: (softmax(x) * w).sum()  # noqa: E731
    x.grad = np.zeros_like(x.data)
    f().backward()
    assert np.allclose(x.grad, numeric_grad(f, x), atol=1e-6)


def test_log_softmax():
    x = rng.normal(size=(3, 4)) * 50
    ls = log_softmax(Tensor(x))
    assert np.allclose(ls.data, np.log(reference_softmax(x)))
    assert np.all(np.isfinite(log_softmax(Tensor([[0.0, -2000.0]])).data)), (
        "log(softmax) of a very unlikely class should be a large negative number, not -inf")


def test_cross_entropy_value():
    logits = rng.normal(size=(5, 4))
    targets = np.array([0, 3, 1, 1, 2])
    loss = cross_entropy(Tensor(logits), targets)
    expected = -np.log(reference_softmax(logits)[np.arange(5), targets]).mean()
    assert loss.shape == () and np.isclose(float(loss.data), expected)
    big = cross_entropy(Tensor([[1000.0, 0.0]]), np.array([1]))
    assert np.isclose(float(big.data), 1000.0), "stable for large logits too"


def test_cross_entropy_gradient():
    logits = Tensor(rng.normal(size=(6, 5)))
    targets = np.array([4, 0, 0, 2, 3, 1])
    f = lambda: cross_entropy(logits, targets)  # noqa: E731
    logits.grad = np.zeros_like(logits.data)
    f().backward()
    assert np.allclose(logits.grad, numeric_grad(f, logits), atol=1e-6), (
        "d loss / d logits should be (softmax - one_hot) / N")


def test_cross_entropy_in_a_graph():
    x = Tensor(rng.normal(size=(4, 3)))
    W = Tensor(rng.normal(size=(3, 5)))
    targets = np.array([1, 2, 3, 4])
    f = lambda: cross_entropy(x @ W, targets) * 2.0  # noqa: E731
    W.grad = np.zeros_like(W.data)
    f().backward()
    assert np.allclose(W.grad, numeric_grad(f, W), atol=1e-6), (
        "use out.grad in your backward: the loss may be scaled or summed afterwards")
