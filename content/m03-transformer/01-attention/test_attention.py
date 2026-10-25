import numpy as np

from m02.tensor import Tensor
from m03.attention import attention, causal_mask

rng = np.random.default_rng(0)


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


def reference(q, k, v, causal):
    s = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if causal:
        T = q.shape[-2]
        s = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, s)
    w = np.exp(s - s.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return w @ v, w


def test_causal_mask():
    m = causal_mask(3)
    assert m.shape == (3, 3)
    assert (m[np.tril_indices(3)] == 0).all(), "the past and the present are allowed: 0"
    future = m[np.triu_indices(3, 1)]
    assert (future < -1e8).all(), "the future must be masked with -inf (or a huge negative number)"


def test_attention_matches_reference():
    q, k, v = (rng.normal(size=(2, 5, 4)) for _ in range(3))
    for causal in (True, False):
        out, w = attention(Tensor(q), Tensor(k), Tensor(v), causal=causal)
        ref_out, ref_w = reference(q, k, v, causal)
        assert w.shape == (2, 5, 5) and out.shape == (2, 5, 4)
        assert np.allclose(w.data, ref_w, atol=1e-12), (
            f"weights should be softmax(q kᵀ / √d){' with the causal mask' if causal else ''}")
        assert np.allclose(out.data, ref_out, atol=1e-12)


def test_causal_rows():
    q, k, v = (Tensor(rng.normal(size=(4, 3))) for _ in range(3))
    _, w = attention(q, k, v)
    assert np.allclose(w.data.sum(-1), 1.0), "each row is a distribution over keys"
    assert np.allclose(w.data[0], [1, 0, 0, 0]), "the first token can only attend to itself"
    assert np.allclose(np.triu(w.data, 1), 0), "no weight on future positions"


def test_scaling_keeps_weights_soft():
    d = 256
    q = Tensor(rng.normal(size=(6, d)))
    k = Tensor(rng.normal(size=(6, d)))
    _, w = attention(q, k, Tensor(rng.normal(size=(6, d))), causal=False)
    assert w.data.max() < 0.99, "without dividing by √d, large d makes softmax nearly one-hot"


def test_gradients():
    q, k, v = (Tensor(rng.normal(size=(2, 4, 3))) for _ in range(3))
    wsum = rng.normal(size=(2, 4, 3))
    f = lambda: (attention(q, k, v)[0] * wsum).sum()  # noqa: E731
    for t in (q, k, v):
        t.grad = np.zeros_like(t.data)
    f().backward()
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert np.allclose(t.grad, numeric_grad(f, t), atol=1e-6), f"gradient for {name} is wrong"
        assert np.all(np.isfinite(t.grad)), "masked positions shouldn't produce NaN gradients"
