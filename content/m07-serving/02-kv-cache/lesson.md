---
id: "7.2"
title: The KV cache
concepts: [kv-cache, incremental-decoding, memory]
file: m07/kv_cache.py
stub: stubs/kv_cache.py
tests: test_kv_cache.py
predict:
  kind: number
  prompt: >
    A model with 32 layers and width 4096 stores a key and a value vector
    per layer for every token, in 16 bits. How many GiB does the cache take
    for one 4096-token sequence?
  answer: 2
  tolerance: 0.05
mutants:
  - name: new tokens see each other's future
    file: m07/kv_cache.py
    replace: ["allowed = torch.arange(past + T)[None, :] <= (past + torch.arange(T))[:, None]", "allowed = torch.ones(T, past + T, dtype=torch.bool)"]
  - name: positions restart at 0
    file: m07/kv_cache.py
    replace: ["pos = torch.arange(cache.length, cache.length + T)", "pos = torch.arange(T)"]
  - name: forgets the values
    file: m07/kv_cache.py
    replace: ["return 2 * n_layer * n_embd * tokens * batch * bytes_per_value", "return n_layer * n_embd * tokens * batch * bytes_per_value"]
warmups:
  - kind: number
    prompt: "32 layers, n_embd 4096, K and V stored in 2-byte floats. How many MiB of KV cache per token?"
    answer: 0.5
    tolerance: 0.02
    explain: "2 (K and V) × 32 × 4096 × 2 bytes = 524,288 bytes = 0.5 MiB per token, so a 4,000-token context is 2 GiB."
---

## Hook

When the model processes token t, the keys and values of tokens 0 … t − 1
are the same as last step: attention is causal, so nothing later changes
them. Store them, and each step only has to process the one new token.

## Build

In `m07/kv_cache.py`, working directly with your `TorchGPT`'s layers:

1. `KVCache(n_layer)` with `append(layer, k, v)` (concatenate along time)
   and `length`.
2. `cached_attention(attn, x, cache, layer)`: queries for the new tokens
   only; keys and values for everything cached plus the new ones. New
   token i is at absolute position past + i and may attend to keys
   0 … past + i.
3. `forward_with_cache(model, idx, cache)`: position embeddings continue
   from `cache.length`.
4. `generate_cached(model, idx, n)`: the prompt in one pass ("prefill"),
   then one token per pass ("decode").
5. `kv_cache_bytes(...)`.

```python
import time
t = time.perf_counter(); a = generate_naive(m, prompt, 200); t1 = time.perf_counter() - t
t = time.perf_counter(); b = generate_cached(m, prompt, 200); t2 = time.perf_counter() - t
torch.equal(a, b), t1 / t2
```

## Explain

2 (key and value) × 32 layers × 4096 numbers × 4096 tokens × 2 bytes =
2³¹ bytes = **2 GiB** per sequence. For a 14 GB model on an 80 GB GPU that
leaves room for about 30 such sequences, which caps the batch size, which
(from 7.1) caps throughput. KV memory, not compute, decides how many users
a server can take.

Hence the many tricks to shrink it: grouped-query attention (several query
heads share one key/value head; Ainslie et al., 2023), multi-query
attention, quantizing the cache, sliding windows, and multi-head latent
attention (DeepSeek-V2), which stores a small latent per token instead.

Prefill and decode behave differently too: prefill processes many tokens
at once (compute-bound, like training), decode one at a time (memory-bound).
Servers schedule them differently.

## Stretch

Implement grouped-query attention: 4 query heads sharing 1 key/value head.
How much smaller is the cache, and how much does a model trained from
scratch with it lose in loss on Shakespeare?
