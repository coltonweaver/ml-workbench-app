---
id: "3.1"
title: Tokens that look at each other
concepts: [self-attention, causal-mask, scaled-dot-product]
file: m03/attention.py
stub: stubs/attention.py
tests: test_attention.py
predict:
  kind: number
  prompt: >
    In a sequence of 4 tokens, each token computes a weight for every token
    (16 weights in all). When predicting the next token, a token must not
    look at tokens that come after it. How many of the 16 weights are forced
    to 0?
  answer: 6
  tolerance: 0
mutants:
  - name: no 1/√d scaling
    file: m03/attention.py
    replace: ["scores = (q @ swap_last(k)) * (1.0 / np.sqrt(d))", "scores = q @ swap_last(k)"]
  - name: mask hides the token itself
    file: m03/attention.py
    replace: ["np.triu(np.full((T, T), -np.inf), k=1)", "np.triu(np.full((T, T), -np.inf), k=0)"]
  - name: softmax over the queries
    file: m03/attention.py
    replace: ["weights = softmax(scores, axis=-1)", "weights = softmax(scores, axis=-2)"]
---

## Hook

Module 2's MLP read a fixed window of 3 characters, glued side by side. To
read 100 characters it would need a weight matrix 33 times wider, and it
would treat "the cat" at position 5 and at position 50 as unrelated
patterns.

Attention takes a different route. Every token emits three vectors: a
**query** ("what am I looking for?"), a **key** ("what do I contain?") and a
**value** ("what I'll pass on"). Token i scores every token j by how well its
query matches j's key (a dot product), turns the scores into weights with a
softmax, and takes the weighted average of the values. Same weights for any
sequence length.

A language model predicts the next token, so token i must not peek at
tokens after it.

## Build

In `m03/attention.py` (it imports your module 2 engine):

1. `causal_mask(T)`: a T × T array, 0 on and below the diagonal, -inf above.
2. `attention(q, k, v, causal=True)`: scores `q @ kᵀ / √d` (use `swap_last`
   for the transpose), add the mask, softmax over the last axis, then
   `weights @ v`. Return `(out, weights)`.

```python
import numpy as np
from m02.tensor import Tensor
x = Tensor(np.random.randn(5, 8))
out, w = attention(x, x, x)
w.data.round(2)          # lower triangular, rows sum to 1
```

## Explain

This is **scaled dot-product attention** from Vaswani et al., "Attention Is
All You Need" (2017): softmax(QKᵀ/√d)V. The causal mask makes it
**causal** (or "masked") self-attention, the kind in every GPT. Six of the
16 weights are in the future (the strict upper triangle: 3 + 2 + 1), so 6
are forced to 0.

Why √d? A dot product of two random d-dimensional vectors has variance
proportional to d. Large scores push softmax toward one-hot, where its
gradient vanishes. Dividing by √d keeps the scores' scale steady as d grows.

Masking with -inf before the softmax rather than zeroing weights after it
keeps every row a proper distribution: exp(-inf) is exactly 0, and the other
weights renormalize among themselves.

Notice that attention has no idea of order: shuffle the keys and values
together and the output is the same weighted average. Positions come in
lesson 3.4.

## Stretch

Implement attention for a single query against T keys without the T × T
matrix, then count operations: what does computing all T queries cost, as a
function of T and d? This T² term is why long contexts are expensive, and
what FlashAttention and the KV cache (module 7) are about.
