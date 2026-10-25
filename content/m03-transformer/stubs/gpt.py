"""A GPT: embeddings, a stack of transformer blocks, and a language-model head.

Lesson 3.5: Block. Lesson 3.6: GPT.
"""

import numpy as np

from m02.losses import cross_entropy
from m02.sampling import sample_next
from m02.tensor import Tensor
from m03.attention import MultiHeadAttention
from m03.layers import MLP, LayerNorm


class Block:
    """Pre-norm transformer block."""

    def __init__(self, n_embd, n_head, seed=0):
        self.ln1 = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, seed=seed)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, seed=seed + 1)

    def parameters(self):
        return (self.ln1.parameters() + self.attn.parameters()
                + self.ln2.parameters() + self.mlp.parameters())

    def __call__(self, x):
        """x + attn(ln1(x)), then that + mlp(ln2(that))."""
        raise NotImplementedError("lesson 3.5")


class GPT:
    def __init__(self, vocab_size, block_size, n_embd=64, n_head=4, n_layer=2, seed=0):
        rng = np.random.default_rng(seed)
        self.block_size = block_size
        self.wte = Tensor(rng.normal(size=(vocab_size, n_embd)) * 0.02)  # token embeddings
        self.wpe = Tensor(rng.normal(size=(block_size, n_embd)) * 0.02)  # position embeddings
        self.blocks = [Block(n_embd, n_head, seed=seed + 10 * (i + 1)) for i in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = Tensor(rng.normal(size=(n_embd, vocab_size)) * 0.02)

    def parameters(self):
        params = [self.wte, self.wpe]
        for block in self.blocks:
            params += block.parameters()
        return params + self.ln_f.parameters() + [self.lm_head]

    def __call__(self, idx):
        """Logits (B, T, vocab) for token ids idx (B, T): token embedding plus
        position embedding, every block, final layer norm, lm_head."""
        raise NotImplementedError("lesson 3.6")

    def loss(self, idx, targets):
        """Cross-entropy of the logits against targets (B, T), all positions."""
        raise NotImplementedError("lesson 3.6")

    def generate(self, idx, max_new, temperature, rng):
        """Extend a list of ids by ``max_new`` tokens sampled with
        m02.sampling.sample_next from the last position's logits, reading at
        most the last block_size tokens each time. Return the whole list."""
        raise NotImplementedError("lesson 3.6")

    def attention_weights(self, idx):
        """Attention of every layer and head on one sequence: an array of
        shape (n_layer, n_head, T, T), for the attention viewer."""
        self(np.asarray(idx)[None, -self.block_size:])
        return np.stack([block.attn.last_weights[0] for block in self.blocks])
