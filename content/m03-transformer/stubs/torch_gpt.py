"""The same GPT in PyTorch, checked against your engine."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_embd, bias=False)
        self.v = nn.Linear(n_embd, n_embd, bias=False)
        self.o = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        """Same as your MultiHeadAttention: heads via view + transpose,
        scores / sqrt(hs), mask the future with masked_fill, softmax,
        weights @ v, merge heads, project with self.o."""
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TorchGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=2):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd, eps=1e-5)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """Logits (B, T, vocab); with targets, also the cross-entropy loss:
        return (logits, loss)."""
        raise NotImplementedError


def load_from_engine(model, engine):
    """Copy every parameter of your engine's GPT into ``model`` (inside
    torch.no_grad(), with param.copy_). Return the model. Careful:
    nn.Linear stores its weight as (out, in)."""
    raise NotImplementedError


def pick_device():
    """Apple GPU when there is one."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def train_torch(model, data, steps, batch_size=64, lr=3e-3, device="cpu", seed=0, log=None):
    """AdamW on random windows of ``data`` (a 1D int array), on ``device``.
    Returns the list of losses. ``log(step, loss)`` is called each step."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    data = torch.as_tensor(np.asarray(data), dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    T = model.block_size
    losses = []
    for step in range(steps):
        starts = torch.as_tensor(rng.integers(0, len(data) - T, size=batch_size))
        X = torch.stack([data[s:s + T] for s in starts]).to(device)
        Y = torch.stack([data[s + 1:s + T + 1] for s in starts]).to(device)
        _, loss = model(X, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log is not None:
            log(step, losses[-1])
    return losses
