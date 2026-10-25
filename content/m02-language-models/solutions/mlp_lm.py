"""An MLP language model with learned embeddings (Bengio et al., 2003)."""

import numpy as np

from m02.losses import cross_entropy
from m02.tensor import Tensor
from m02.tokens import BOUNDARY


def build_dataset(words, stoi, block_size):
    """For every position in every word, the previous ``block_size`` ids
    (padded with the boundary) and the id that comes next. Returns X of shape
    (N, block_size) and Y of shape (N,)."""
    X, Y = [], []
    for word in words:
        context = [stoi[BOUNDARY]] * block_size
        for ch in word + BOUNDARY:
            X.append(context)
            Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    return np.array(X), np.array(Y)


class MLPLanguageModel:
    def __init__(self, vocab_size, block_size=3, n_embd=10, n_hidden=200, seed=0):
        rng = np.random.default_rng(seed)
        self.block_size = block_size
        self.C = Tensor(rng.normal(size=(vocab_size, n_embd)))
        fan_in = block_size * n_embd
        self.W1 = Tensor(rng.normal(size=(fan_in, n_hidden)) * (5 / 3) / np.sqrt(fan_in))
        self.b1 = Tensor(np.zeros(n_hidden))
        self.W2 = Tensor(rng.normal(size=(n_hidden, vocab_size)) * 0.01)
        self.b2 = Tensor(np.zeros(vocab_size))

    def parameters(self):
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, X):
        """Logits (N, vocab) for contexts X (N, block_size)."""
        emb = self.C[X]                                   # (N, block, n_embd)
        h = (emb.reshape(emb.shape[0], -1) @ self.W1 + self.b1).tanh()
        return h @ self.W2 + self.b2


def train(model, X, Y, steps, batch_size=32, lr=0.1, seed=0, log=None):
    """Minibatch SGD on cross-entropy. Returns the list of minibatch losses.
    ``log(step, loss)`` is called each step if given."""
    rng = np.random.default_rng(seed)
    losses = []
    for step in range(steps):
        idx = rng.integers(0, len(X), size=batch_size)
        loss = cross_entropy(model(X[idx]), Y[idx])
        for p in model.parameters():
            p.grad = np.zeros_like(p.data)
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        losses.append(float(loss.data))
        if log is not None:
            log(step, losses[-1])
    return losses
