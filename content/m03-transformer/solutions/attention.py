"""Self-attention, on your tensor engine from module 2."""

import numpy as np

from m02.losses import softmax
from m02.tensor import Tensor


def swap_last(x):
    """Swap the last two axes of a Tensor: (..., T, d) -> (..., d, T)."""
    axes = list(range(x.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return x.transpose(*axes)


def causal_mask(T):
    """A (T, T) array to add to attention scores: 0 where query i may look at
    key j (j <= i), -inf where j is in the future."""
    return np.triu(np.full((T, T), -np.inf), k=1)


def attention(q, k, v, causal=True):
    """Scaled dot-product attention over the last two axes.

    q, k, v: Tensors of shape (..., T, d). Returns (out, weights): out has
    shape (..., T, d); weights (..., T, T) is softmax(q kᵀ / √d), each row a
    distribution over the keys.
    """
    d = q.shape[-1]
    scores = (q @ swap_last(k)) * (1.0 / np.sqrt(d))
    if causal:
        scores = scores + causal_mask(q.shape[-2])
    weights = softmax(scores, axis=-1)
    return weights @ v, weights


class MultiHeadAttention:
    """n_head attention heads side by side, then an output projection."""

    def __init__(self, n_embd, n_head, seed=0):
        assert n_embd % n_head == 0, "n_embd must divide evenly into heads"
        rng = np.random.default_rng(seed)
        self.n_head = n_head
        self.Wq = Tensor(rng.normal(size=(n_embd, n_embd)) * 0.02)
        self.Wk = Tensor(rng.normal(size=(n_embd, n_embd)) * 0.02)
        self.Wv = Tensor(rng.normal(size=(n_embd, n_embd)) * 0.02)
        self.Wo = Tensor(rng.normal(size=(n_embd, n_embd)) * 0.02)
        self.last_weights = None

    def parameters(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo]

    def split_heads(self, x):
        """(B, T, C) -> (B, n_head, T, C / n_head)."""
        B, T, C = x.shape
        return x.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

    def merge_heads(self, x):
        """(B, n_head, T, hs) -> (B, T, n_head * hs)."""
        B, H, T, hs = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * hs)

    def __call__(self, x):
        q = self.split_heads(x @ self.Wq)
        k = self.split_heads(x @ self.Wk)
        v = self.split_heads(x @ self.Wv)
        out, weights = attention(q, k, v, causal=True)
        self.last_weights = weights.data  # (B, n_head, T, T), for the viewer
        return self.merge_heads(out) @ self.Wo
