import numpy as np

from m02.tensor import Tensor
from m03.attention import MultiHeadAttention

rng = np.random.default_rng(1)


def reference(mha, x):
    B, T, C = x.shape
    H = mha.n_head
    hs = C // H

    def heads(t):
        return t.reshape(B, T, H, hs).transpose(0, 2, 1, 3)

    q, k, v = heads(x @ mha.Wq.data), heads(x @ mha.Wk.data), heads(x @ mha.Wv.data)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hs)
    s = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, s)
    w = np.exp(s - s.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    y = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
    return y @ mha.Wo.data, w


def test_split_and_merge():
    mha = MultiHeadAttention(8, 2)
    x = Tensor(rng.normal(size=(3, 5, 8)))
    s = mha.split_heads(x)
    assert s.shape == (3, 2, 5, 4)
    assert np.allclose(s.data[1, 1, 2], x.data[1, 2, 4:8]), "head h gets channels h*hs to (h+1)*hs"
    m = mha.merge_heads(s)
    assert m.shape == (3, 5, 8) and np.allclose(m.data, x.data), "merge undoes split"


def test_forward_matches_reference():
    mha = MultiHeadAttention(12, 3, seed=2)
    for p in mha.parameters():
        p.data = rng.normal(size=p.shape) * 0.5
    x = rng.normal(size=(2, 6, 12))
    out = mha(Tensor(x))
    ref, w = reference(mha, x)
    assert out.shape == (2, 6, 12)
    assert np.allclose(out.data, ref, atol=1e-10)
    assert mha.last_weights is not None and np.allclose(mha.last_weights, w), (
        "store the attention weights' data, shape (B, n_head, T, T), in last_weights")


def test_no_peeking_at_the_future():
    mha = MultiHeadAttention(8, 2, seed=3)
    for p in mha.parameters():
        p.data = rng.normal(size=p.shape)
    x = rng.normal(size=(1, 5, 8))
    y = x.copy()
    y[0, 4] += 10.0
    a, b = mha(Tensor(x)).data, mha(Tensor(y)).data
    assert np.allclose(a[0, :4], b[0, :4]), "changing the last token must not change earlier outputs"


def test_gradient_reaches_every_weight():
    mha = MultiHeadAttention(4, 2, seed=4)
    for p in mha.parameters():
        p.data = rng.normal(size=p.shape)
    x = Tensor(rng.normal(size=(1, 3, 4)))
    w = rng.normal(size=(1, 3, 4))
    f = lambda: (mha(x) * w).sum()  # noqa: E731
    f().backward()
    for name, p in zip(["Wq", "Wk", "Wv", "Wo"], mha.parameters()):
        i = (1, 2)
        old = p.data[i]
        p.data[i] = old + 1e-6
        hi = float(f().data)
        p.data[i] = old - 1e-6
        lo = float(f().data)
        p.data[i] = old
        assert np.isclose(p.grad[i], (hi - lo) / 2e-6, atol=1e-6), f"gradient for {name} is wrong"
