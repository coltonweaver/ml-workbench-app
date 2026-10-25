---
id: "3.5"
title: The residual stream
concepts: [residual-connection, transformer-block, pre-norm]
file: m03/gpt.py
stub: stubs/gpt.py
tests: test_block.py
predict:
  kind: choice
  prompt: >
    A transformer block computes x + attention(norm(x)), then adds
    mlp(norm(...)) to that. Suppose both sublayers' final weight matrices
    are all zeros, so each sublayer outputs 0. What does the block output?
  options:
    - all zeros
    - its input, unchanged
    - the layer-normalized input
    - something random
  answer: 1
mutants:
  - name: post-norm instead of pre-norm
    file: m03/gpt.py
    replace: ["        x = x + self.attn(self.ln1(x))\n", "        x = self.ln1(x + self.attn(x))\n"]
  - name: no residual around the MLP
    file: m03/gpt.py
    replace: ["        return x + self.mlp(self.ln2(x))", "        return self.mlp(self.ln2(x))"]
warmups:
  - kind: choice
    prompt: "In a pre-norm transformer block, what gets added to the residual stream x by the attention sublayer?"
    options: ["ln(x + attn(x))", "attn(ln(x))", "ln(attn(x))"]
    answer: 1
    explain: "x ← x + attn(ln(x)). The stream itself is never normalized, so there's a clean additive path from embedding to output."
---

## Hook

You have all the parts: attention to move information between tokens, an
MLP to compute on each token, layer norm to keep scales sane. How should
they be wired together so that 12, or 96, of them can be stacked and still
train?

The key move: each sublayer doesn't replace the token's vector, it **adds**
to it.

## Build

In `m03/gpt.py`, `Block.__call__`:

```python
x = x + self.attn(self.ln1(x))
return x + self.mlp(self.ln2(x))
```

That's the whole lesson in code. The tests check the wiring, and the
prediction is about why it matters.

## Explain

The vector that flows from the embeddings through every block to the output
is the **residual stream**. Each block reads from it (through a layer norm)
and writes back by addition. With zeroed sublayers a block is the identity,
so a deep network starts close to a shallow one, and the gradient has a
direct path from the loss to the embeddings: the derivative of x + f(x)
always includes 1. That's the trick from ResNets (He et al., 2015) that made
very deep networks trainable.

Normalizing *before* each sublayer (**pre-norm**, as in GPT-2) keeps the
stream itself un-normalized, so information written by early layers passes
through untouched. The original transformer normalized after the addition
(post-norm), which needs careful learning-rate warmup to train.

For module 5, this is the most important picture in the course: the
residual stream as a shared bus that every head and MLP reads from and
writes to. Interpretability work is mostly about figuring out what's written
on it.

## Stretch

Build a 12-block model with and without residual connections. Compare the
gradient norm reaching the token embeddings after one backward pass.
