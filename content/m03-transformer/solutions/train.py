"""Training the GPT: batches from a corpus, and the AdamW optimizer."""

import numpy as np


def build_char_vocab(text):
    """Every distinct character of ``text``, sorted: (stoi, itos)."""
    itos = sorted(set(text))
    return {ch: i for i, ch in enumerate(itos)}, itos


def get_batch(data, block_size, batch_size, rng):
    """``batch_size`` random windows of ``data`` (a 1D int array). X holds
    block_size tokens, Y the same window shifted one to the right: at every
    position the target is the next token."""
    starts = rng.integers(0, len(data) - block_size, size=batch_size)
    X = np.stack([data[s:s + block_size] for s in starts])
    Y = np.stack([data[s + 1:s + block_size + 1] for s in starts])
    return X, Y


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019)."""

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
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            if self.weight_decay and p.data.ndim >= 2:  # decay matrices, not gains and biases
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train_gpt(model, data, steps, batch_size=16, lr=3e-3, seed=0, log=None):
    """Train with AdamW on random batches. Returns the list of losses.
    ``log(step, loss)`` is called each step if given."""
    rng = np.random.default_rng(seed)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    for step in range(steps):
        X, Y = get_batch(data, model.block_size, batch_size, rng)
        loss = model.loss(X, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
        if log is not None:
            log(step, losses[-1])
    return losses
