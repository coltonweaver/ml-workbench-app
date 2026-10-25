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
        B, T, C = x.shape
        hs = C // self.n_head

        def heads(t):
            return t.view(B, T, self.n_head, hs).transpose(1, 2)

        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        att = att.masked_fill(future, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.o(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))


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
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss


def load_from_engine(model, engine):
    """Copy every parameter of your engine's GPT into ``model``.
    nn.Linear stores its weight as (out, in), the transpose of your x @ W."""

    def put(param, array):
        param.copy_(torch.as_tensor(np.asarray(array), dtype=param.dtype))

    with torch.no_grad():
        put(model.wte.weight, engine.wte.data)
        put(model.wpe.weight, engine.wpe.data)
        for tb, eb in zip(model.blocks, engine.blocks):
            put(tb.ln1.weight, eb.ln1.gamma.data)
            put(tb.ln1.bias, eb.ln1.beta.data)
            put(tb.attn.q.weight, eb.attn.Wq.data.T)
            put(tb.attn.k.weight, eb.attn.Wk.data.T)
            put(tb.attn.v.weight, eb.attn.Wv.data.T)
            put(tb.attn.o.weight, eb.attn.Wo.data.T)
            put(tb.ln2.weight, eb.ln2.gamma.data)
            put(tb.ln2.bias, eb.ln2.beta.data)
            put(tb.mlp.fc.weight, eb.mlp.W1.data.T)
            put(tb.mlp.fc.bias, eb.mlp.b1.data)
            put(tb.mlp.proj.weight, eb.mlp.W2.data.T)
            put(tb.mlp.proj.bias, eb.mlp.b2.data)
        put(model.ln_f.weight, engine.ln_f.gamma.data)
        put(model.ln_f.bias, engine.ln_f.beta.data)
        put(model.lm_head.weight, engine.lm_head.data.T)
    return model


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
