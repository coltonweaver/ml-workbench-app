"""A KV cache: keep each layer's keys and values so each new token is one
small forward pass."""

import math

import torch
import torch.nn.functional as F


class KVCache:
    """Per-layer keys and values, each (B, H, T, hs), growing along T."""

    def __init__(self, n_layer):
        self.k = [None] * n_layer
        self.v = [None] * n_layer

    @property
    def length(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def append(self, layer, k, v):
        """Add new keys and values; return everything cached for the layer."""
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]


def cached_attention(attn, x, cache, layer):
    """CausalSelfAttention on the new tokens x (B, T_new, C), attending to
    every cached token plus the new ones (causally among the new ones)."""
    B, T, C = x.shape
    H = attn.n_head
    hs = C // H

    def heads(t):
        return t.view(B, T, H, hs).transpose(1, 2)

    past = 0 if cache.k[layer] is None else cache.k[layer].shape[2]
    q = heads(attn.q(x))
    k, v = cache.append(layer, heads(attn.k(x)), heads(attn.v(x)))
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(hs)          # (B, H, T, past + T)
    # new token i sits at position past + i and may see keys 0 … past + i
    allowed = torch.arange(past + T)[None, :] <= (past + torch.arange(T))[:, None]
    att = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    return attn.o((att @ v).transpose(1, 2).reshape(B, T, C))


@torch.no_grad()
def forward_with_cache(model, idx, cache):
    """Logits for the new tokens idx (B, T_new), continuing from ``cache``,
    which is updated in place."""
    B, T = idx.shape
    pos = torch.arange(cache.length, cache.length + T)
    if pos[-1] >= model.block_size:
        raise ValueError(f"position {int(pos[-1])} is past block_size {model.block_size}")
    x = model.wte(idx) + model.wpe(pos)
    for i, block in enumerate(model.blocks):
        x = x + cached_attention(block.attn, block.ln1(x), cache, i)
        x = x + block.mlp(block.ln2(x))
    return model.lm_head(model.ln_f(x))


@torch.no_grad()
def generate_cached(model, idx, n):
    """Greedy decoding with a KV cache: the prompt in one pass, then one
    token per pass."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    cache = KVCache(len(model.blocks))
    logits = forward_with_cache(model, idx, cache)
    for step in range(n):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        if step < n - 1:
            logits = forward_with_cache(model, nxt, cache)
    return idx


def kv_cache_bytes(n_layer, n_embd, tokens, batch=1, bytes_per_value=2):
    """Memory for the cache: a key and a value vector per layer per token."""
    return 2 * n_layer * n_embd * tokens * batch * bytes_per_value
