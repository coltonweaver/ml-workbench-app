---
id: "3.6"
title: A whole GPT
concepts: [gpt, language-model-head, autoregressive-generation]
file: m03/gpt.py
stub: stubs/gpt.py
tests: test_gpt.py
predict:
  kind: number
  prompt: >
    GPT(vocab_size=65, block_size=32, n_embd=64, n_head=4, n_layer=2):
    token and position embedding tables; per block, two layer norms (gain
    and bias each), four 64 × 64 attention matrices and an MLP 64→256→64
    with biases; a final layer norm; and a 64 × 65 output matrix. How many
    parameters?
  answer: 109952
  tolerance: 0
mutants:
  - name: no position embeddings
    file: m03/gpt.py
    replace: ["x = self.wte[idx] + self.wpe[np.arange(T)]", "x = self.wte[idx]"]
  - name: generate doesn't crop the context
    file: m03/gpt.py
    replace: ["context = np.array([idx[-self.block_size:]])", "context = np.array([idx])"]
  - name: skips the final layer norm
    file: m03/gpt.py
    replace: ["return self.ln_f(x) @ self.lm_head", "return x @ self.lm_head"]
---

## Hook

Put it together: look up an embedding for each token, add an embedding for
each position, run the stack of blocks, normalize, and multiply by an output
matrix to get logits over the vocabulary at every position. That's the
architecture of GPT-2, GPT-3 and, give or take some details, the models
behind every chat assistant.

Before building it, count what it will learn.

## Build

In `m03/gpt.py`, `GPT`:

1. `__call__(idx)`: `wte[idx] + wpe[arange(T)]`, each block, `ln_f`, then
   `@ lm_head`. Shape (B, T, vocab).
2. `loss(idx, targets)`: reshape logits to (B·T, vocab) and use your
   `cross_entropy`. Every position predicts its next token, so one sequence
   of T tokens is T training examples.
3. `generate(idx, max_new, temperature, rng)`: crop to the last
   `block_size` tokens, take the logits at the last position, `sample_next`,
   append, repeat.

```python
m = GPT(65, 32)
sum(p.data.size for p in m.parameters())
m.generate([0], 40, 1.0, np.random.default_rng(0))     # gibberish ids for now
```

## Explain

This is a decoder-only transformer, the **GPT** of Radford et al. (2018,
2019). 109,952 parameters, against GPT-2 small's 124 million and GPT-3's
175 billion; the architecture is the same.

The count is easy to check by part. Embeddings: 65·64 + 32·64. Each block:
2·128 for the layer norms, 4·64² = 16,384 for attention, and
64·256 + 256 + 256·64 + 64 = 33,088 for the MLP. Then 128 for the final
norm and 64·65 for the head. Notice that attention is a third of a block,
and that with a tiny vocabulary the embeddings barely matter. In GPT-2 the
token embedding alone is 38 million parameters.

The **language-model head** maps the final residual stream to logits. GPT-2
ties it to the token embedding (uses `wte` transposed), saving parameters;
you kept them separate. Generation is the same autoregressive loop as
module 2, now with a model that reads up to `block_size` tokens of context.

## Stretch

Tie the weights: delete `lm_head` and use `wte` transposed. How many
parameters does that save here, and in GPT-2 small (vocab 50,257,
n_embd 768)?
