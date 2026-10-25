"""A GPT: embeddings, a stack of transformer blocks, and a language-model head."""

import numpy as np

from m02.losses import cross_entropy
from m02.sampling import sample_next
from m02.tensor import Tensor
from m03.attention import MultiHeadAttention
from m03.layers import MLP, LayerNorm


class Block:
    """Pre-norm transformer block: x + attn(ln1(x)), then + mlp(ln2(x))."""

    def __init__(self, n_embd, n_head, seed=0):
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, seed=seed)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, seed=seed + 1)

    def parameters(self):
        return (self.ln1.parameters() + self.attn.parameters()
                + self.ln2.parameters() + self.mlp.parameters())

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT:
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=2, seed=0):
        rng = np.random.default_rng(seed)
        self.block_size = block_size
        self.wte = Tensor(rng.normal(size=(vocab_size, n_embd)) * 0.02)
        self.wpe = Tensor(rng.normal(size=(block_size, n_embd)) * 0.02)
        self.blocks = [Block(n_embd, n_head, seed=seed + 10 * (i + 1)) for i in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = Tensor(rng.normal(size=(n_embd, vocab_size)) * 0.02)

    def parameters(self):
        params = [self.wte, self.wpe]
        for block in self.blocks:
            params += block.parameters()
        return params + self.ln_f.parameters() + [self.lm_head]

    def __call__(self, idx):
        """Logits (B, T, vocab) for token ids idx (B, T)."""
        idx = np.asarray(idx)
        B, T = idx.shape
        assert T <= self.block_size, f"sequence of {T} is longer than block_size {self.block_size}"
        x = self.wte[idx] + self.wpe[np.arange(T)]
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x) @ self.lm_head

    def loss(self, idx, targets):
        logits = self(idx)
        B, T, V = logits.shape
        return cross_entropy(logits.reshape(B * T, V), np.asarray(targets).reshape(B * T))

    def generate(self, idx, max_new, temperature, rng):
        """Extend a list of ids by ``max_new`` sampled tokens, reading at most
        the last block_size tokens each time. Returns the whole list."""
        idx = list(idx)
        for _ in range(max_new):
            context = np.array([idx[-self.block_size:]])
            logits = self(context).data[0, -1]
            idx.append(sample_next(logits, temperature, rng))
        return idx

    def attention_weights(self, idx):
        """Attention of every layer and head on one sequence: an array of
        shape (n_layer, n_head, T, T), for the attention viewer."""
        self(np.asarray(idx)[None, -self.block_size:])
        return np.stack([block.attn.last_weights[0] for block in self.blocks])
