"""An MLP language model with learned embeddings."""

import numpy as np

from m02.losses import cross_entropy
from m02.tensor import Tensor
from m02.tokens import BOUNDARY


def build_dataset(words, stoi, block_size):
    """For every position in every word, the previous ``block_size`` ids
    (padded with the boundary) and the id that comes next, ending with the
    boundary. Returns X of shape (N, block_size) and Y of shape (N,)."""
    raise NotImplementedError


class MLPLanguageModel:
    def __init__(self, vocab_size, block_size=3, n_embd=10, n_hidden=200, seed=0):
        rng = np.random.default_rng(seed)
        self.block_size = block_size
        # C: one n_embd-vector per token. W1, b1: the hidden layer, reading
        # all block_size embeddings side by side. W2, b2: hidden to logits.
        raise NotImplementedError

    def parameters(self):
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, X):
        """Logits (N, vocab) for contexts X (N, block_size): look up the
        embeddings, flatten each context, tanh hidden layer, output layer."""
        raise NotImplementedError


def train(model, X, Y, steps, batch_size=32, lr=0.1, seed=0, log=None):
    """Minibatch SGD on cross-entropy: each step draws ``batch_size`` random
    rows with ``rng.integers``. Returns the list of minibatch losses.
    Call ``log(step, loss)`` each step if it's given."""
    raise NotImplementedError
