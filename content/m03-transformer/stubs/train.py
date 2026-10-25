"""Training the GPT: batches from a corpus, and the AdamW optimizer."""

import numpy as np


def build_char_vocab(text):
    """Every distinct character of ``text``, sorted: (stoi, itos)."""
    raise NotImplementedError


def get_batch(data, block_size, batch_size, rng):
    """``batch_size`` random windows of ``data`` (a 1D int array), starts
    drawn with rng.integers(0, len(data) - block_size). X holds block_size
    tokens; Y is the same window shifted one to the right."""
    raise NotImplementedError


class AdamW:
    """Adam with decoupled weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr, self.eps, self.weight_decay = lr, eps, weight_decay
        self.beta1, self.beta2 = betas
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        """One update of every parameter. For each: m and v are running
        averages of g and g²; correct their bias toward 0 with
        1 - beta^t; decay 2D+ parameters by lr * weight_decay * p; then
        p -= lr * m_hat / (sqrt(v_hat) + eps)."""
        raise NotImplementedError


def train_gpt(model, data, steps, batch_size=16, lr=3e-3, seed=0, log=None):
    """AdamW (weight_decay=0.01) on get_batch batches drawn from
    np.random.default_rng(seed). Return the list of losses; call
    ``log(step, loss)`` each step if given."""
    raise NotImplementedError
