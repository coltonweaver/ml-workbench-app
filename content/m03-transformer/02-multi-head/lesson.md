---
id: "3.2"
title: Several ways of looking
concepts: [multi-head-attention, projections]
file: m03/attention.py
stub: stubs/attention.py
tests: test_multi_head.py
predict:
  kind: choice
  prompt: >
    Multi-head attention with n_embd = 64 projects x to queries, keys and
    values with three 64 × 64 matrices, splits each into heads, and projects
    the merged result with a fourth 64 × 64 matrix. Going from 4 heads of
    size 16 to 8 heads of size 8, what happens to the number of parameters?
  options:
    - it doubles
    - it halves
    - it stays the same
    - it grows by 8 × 8 per head
  answer: 2
mutants:
  - name: merge without moving heads back
    file: m03/attention.py
    replace: ["return x.transpose(0, 2, 1, 3).reshape(B, T, H * hs)", "return x.reshape(B, T, H * hs)"]
  - name: forgets the output projection
    file: m03/attention.py
    replace: ["return self.merge_heads(out) @ self.Wo", "return self.merge_heads(out)"]
warmups:
  - kind: number
    prompt: "n_embd = 128 split across 4 heads. What is each head's size?"
    answer: 32
    tolerance: 0.01
    explain: "128 / 4. The heads share the width rather than each getting the full 128, so multi-head attention costs about the same as one big head."
---

## Hook

One attention pattern per token is limiting. When reading "the cat sat on
the", a model might want to look at the previous word (for grammar), at the
subject (for meaning) and at the start of the line, all at once. One softmax
can only make one weighted average.

The fix is cheap: split the 64-number vectors into 4 groups of 16, run
attention separately in each group, and glue the results back together.
Each group, a **head**, can learn its own pattern.

## Build

Still in `m03/attention.py`, `MultiHeadAttention`:

1. `split_heads(x)`: (B, T, C) → reshape to (B, T, H, C/H) → transpose to
   (B, H, T, C/H), so the heads act like an extra batch axis.
2. `merge_heads(x)`: the reverse.
3. `__call__(x)`: `q = split_heads(x @ Wq)` (and k, v), causal attention
   (it's already batched over B and H), merge, then `@ Wo`. Save
   `weights.data` in `self.last_weights`.

```python
mha = MultiHeadAttention(16, 4)
y = mha(Tensor(np.random.randn(2, 6, 16)))
y.shape, mha.last_weights.shape        # (2, 6, 16), (2, 4, 6, 6)
from workbench import show_attention
show_attention(mha.last_weights[0], list("abcdef"), title="random heads")
```

Random weights give near-uniform heads. Trained ones won't.

## Explain

This is **multi-head attention**. The heads cost nothing extra: the
projections are the same four 64 × 64 matrices however you slice their
output, so the parameter count stays at 4 × 64² = 16,384. What changes is
how many separate attention patterns you get and how wide each is.

The output projection `Wo` mixes the heads' results back together. Without
it, each head's output would be stuck in its own 16 channels.

Heads specialize. In trained models, people find heads that attend to the
previous token, to matching brackets, or (module 5) to "the token after the
last time this token appeared": induction heads, which explain much of
in-context learning (Olsson et al., 2022).

## Stretch

Fuse Wq, Wk and Wv into one (C, 3C) matrix and split its output. It
computes the same thing in one matmul. Check it's identical, then time both.
