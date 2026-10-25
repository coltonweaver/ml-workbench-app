"""The other pieces of a transformer block."""

import numpy as np

from m02.tensor import Tensor


class LayerNorm:
    """Normalize each vector (the last axis) to mean 0 and variance 1, then
    scale by gamma and shift by beta."""

    def __init__(self, n_embd, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(np.ones(n_embd))
        self.beta = Tensor(np.zeros(n_embd))

    def parameters(self):
        return [self.gamma, self.beta]

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered ** 2).mean(axis=-1, keepdims=True)
        return centered * (var + self.eps) ** -0.5 * self.gamma + self.beta


def gelu(x):
    """GELU, tanh approximation (as in GPT-2):
    0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))."""
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * ((x + 0.044715 * x ** 3) * c).tanh() + 0.5 * x


class MLP:
    """Expand to 4 × n_embd, GELU, project back."""

    def __init__(self, n_embd, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = Tensor(rng.normal(size=(n_embd, 4 * n_embd)) * 0.02)
        self.b1 = Tensor(np.zeros(4 * n_embd))
        self.W2 = Tensor(rng.normal(size=(4 * n_embd, n_embd)) * 0.02)
        self.b2 = Tensor(np.zeros(n_embd))

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def __call__(self, x):
        return gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


def sinusoidal_positions(T, n_embd):
    """The fixed position encoding of "Attention Is All You Need":
    PE[pos, 2i] = sin(pos / 10000^(2i/n_embd)), PE[pos, 2i+1] = cos(same)."""
    pos = np.arange(T)[:, None]
    i = np.arange(0, n_embd, 2)[None, :]
    angle = pos / 10000 ** (i / n_embd)
    pe = np.zeros((T, n_embd))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle[:, : n_embd // 2])
    return pe
