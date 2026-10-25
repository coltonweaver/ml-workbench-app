"""Self-attention, on your tensor engine from module 2.

Lesson 3.1: causal_mask and attention. Lesson 3.2: MultiHeadAttention.
"""

import numpy as np

from m02.losses import softmax
from m02.tensor import Tensor


def swap_last(x):
    """Swap the last two axes of a Tensor: (..., T, d) -> (..., d, T)."""
    axes = list(range(x.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return x.transpose(*axes)


def causal_mask(T):
    """A (T, T) NumPy array to add to attention scores: 0 where query i may
    look at key j (j <= i), -inf where j is in the future."""
    raise NotImplementedError("lesson 3.1")


def attention(q, k, v, causal=True):
    """Scaled dot-product attention over the last two axes.

    q, k, v: Tensors of shape (..., T, d). Returns (out, weights): weights
    (..., T, T) is softmax(q kᵀ / √d) (masked if causal), each row a
    distribution over the keys; out = weights @ v, shape (..., T, d).
    """
    raise NotImplementedError("lesson 3.1")


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
        raise NotImplementedError("lesson 3.2")

    def merge_heads(self, x):
        """(B, n_head, T, hs) -> (B, T, n_head * hs)."""
        raise NotImplementedError("lesson 3.2")

    def __call__(self, x):
        """x (B, T, C): project to q, k, v with Wq, Wk, Wv, split into heads,
        causal attention, merge, project with Wo. Keep the weights' data in
        self.last_weights (B, n_head, T, T) for the attention viewer."""
        raise NotImplementedError("lesson 3.2")
