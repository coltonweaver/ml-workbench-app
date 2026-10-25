"""The other pieces of a transformer block.

Lesson 3.3: LayerNorm. Lesson 3.4: gelu, MLP and sinusoidal_positions.
"""

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
        """(x - mean) / sqrt(var + eps) * gamma + beta, with the mean and the
        (biased) variance taken over the last axis. Tensor ops only."""
        raise NotImplementedError("lesson 3.3")


def gelu(x):
    """GELU, tanh approximation (as in GPT-2):
    0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))."""
    raise NotImplementedError("lesson 3.4")


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
        raise NotImplementedError("lesson 3.4")


def sinusoidal_positions(T, n_embd):
    """The fixed position encoding of "Attention Is All You Need", a (T,
    n_embd) array: PE[pos, 2i] = sin(pos / 10000^(2i/n_embd)) and
    PE[pos, 2i+1] = cos(pos / 10000^(2i/n_embd))."""
    raise NotImplementedError("lesson 3.4")
