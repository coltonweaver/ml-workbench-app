"""A KV cache: keep each layer's keys and values so each new token is one
small forward pass.

Lesson 7.2."""
import math
import torch
import torch.nn.functional as F


class KVCache:
    """Per-layer keys and values, each (B, H, T, hs), growing along T."""

    def __init__(self, n_layer):
        raise NotImplementedError('lesson 7.2')

    @property
    def length(self):
        raise NotImplementedError('lesson 7.2')

    def append(self, layer, k, v):
        """Add new keys and values; return everything cached for the layer."""
        raise NotImplementedError('lesson 7.2')


def cached_attention(attn, x, cache, layer):
    """CausalSelfAttention on the new tokens x (B, T_new, C), attending to
    every cached token plus the new ones (causally among the new ones)."""
    raise NotImplementedError('lesson 7.2')


@torch.no_grad()
def forward_with_cache(model, idx, cache):
    """Logits for the new tokens idx (B, T_new), continuing from ``cache``,
    which is updated in place."""
    raise NotImplementedError('lesson 7.2')


@torch.no_grad()
def generate_cached(model, idx, n):
    """Greedy decoding with a KV cache: the prompt in one pass, then one
    token per pass."""
    raise NotImplementedError('lesson 7.2')


def kv_cache_bytes(n_layer, n_embd, tokens, batch=1, bytes_per_value=2):
    """Memory for the cache: a key and a value vector per layer per token."""
    raise NotImplementedError('lesson 7.2')
