---
id: "2.9"
title: Embeddings and more context
concepts: [embedding, mlp-language-model, minibatch]
file: m02/mlp_lm.py
stub: stubs/mlp_lm.py
tests: test_mlp_lm.py
predict:
  kind: number
  prompt: >
    The model reads 3 previous characters. Each of the 27 tokens has a
    learned 10-number embedding; the 3 embeddings side by side (30 numbers)
    feed a hidden layer of 200 tanh units (weights and biases), which feeds
    27 output logits (weights and biases). How many parameters in total?
  answer: 11897
  tolerance: 0
mutants:
  - name: the dataset forgets that words end
    file: m02/mlp_lm.py
    replace: ["for ch in word + BOUNDARY:", "for ch in word:"]
  - name: hidden layer has no nonlinearity
    file: m02/mlp_lm.py
    replace: ["@ self.W1 + self.b1).tanh()", "@ self.W1 + self.b1)"]
  - name: output layer starts large
    file: m02/mlp_lm.py
    replace: ["size=(n_hidden, vocab_size)) * 0.01", "size=(n_hidden, vocab_size))"]
---

## Hook

One character of context gets a loss of 2.45. Three characters would help a
lot ("emm" is almost always followed by "a"), but a counting table for three
characters has 27³ rows, most of them empty.

Bengio and colleagues' 2003 idea: give every token a small learned vector,
an **embedding**. Characters that behave alike can end up with similar
vectors, so what the model learns about one transfers to the others. Then a
small neural network reads the embeddings of the last few characters and
outputs logits for the next one.

## Build

In `m02/mlp_lm.py`:

1. `build_dataset(words, stoi, block_size)`: for each position, the last
   `block_size` ids (padded with the boundary at the start) and the next id.
   Each word ends with the boundary as a target.
2. `MLPLanguageModel`: `C` (vocab × n_embd), `W1` (block·n_embd × n_hidden),
   `b1`, `W2` (n_hidden × vocab), `b2`. Forward: `C[X]`, reshape to
   (N, block·n_embd), tanh hidden layer, logits. Initialize `W1` with
   `rng.normal * (5/3) / sqrt(fan_in)` and `W2` with `rng.normal * 0.01`.
3. `train(model, X, Y, steps, batch_size, lr, seed, log)`: random minibatch,
   cross-entropy, zero grads, backward, SGD step.

```python
from workbench import show_training, show_embeddings, grad_norms
X, Y = build_dataset(words, stoi, 3)
model = MLPLanguageModel(27)
dash = show_training("MLP language model")
names_ = ["C", "W1", "b1", "W2", "b2"]
losses = train(model, X, Y, steps=5000, batch_size=64, lr=0.1,
               log=lambda s, l: dash.log(s, loss=l, tokens=64 * 3,
                   grad_norms=grad_norms(dict(zip(names_, model.parameters())))))
dash.done()
show_embeddings(model.C, itos, title="character embeddings")
```

## Explain

This is the **neural probabilistic language model** of Bengio, Ducharme,
Vincent and Jauvin (2003), the direct ancestor of every modern LM: tokens to
**embeddings**, a network over the context, softmax over the vocabulary.
The GPT in module 3 keeps all three pieces and replaces the middle with
attention.

Look at the embedding explorer. With no instruction to do so, the model
tends to put the vowels near each other, and the boundary "." far from
everything. It learned that from prediction alone.

Training on random **minibatches** of 64 instead of all 228,000 examples
makes each step thousands of times cheaper. The gradient is noisier, but far
more steps more than make up for it. That trade is **stochastic gradient
descent**, and it's how every large model is trained.

The initialization matters. Scaling W2 by 0.01 makes the first logits nearly
equal, so the initial loss is ln 27 ≈ 3.30 instead of something huge. The
(5/3)/√fan_in scaling keeps tanh units out of their flat regions (Kaiming
He et al., 2015, for the general rule). Watch the gradient norms: if a
layer's norm collapses or explodes, training stalls.

## Stretch

Try `n_embd=2` so the embeddings are 2D and need no projection. Then try
block sizes 1 through 8. How low does the loss go before more context stops
helping, and why might that be?
