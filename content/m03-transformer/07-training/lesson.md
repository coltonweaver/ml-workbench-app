---
id: "3.7"
title: Training your GPT
concepts: [adam, training-loop, corpus]
file: m03/train.py
stub: stubs/train.py
tests: test_training.py
predict:
  kind: choice
  prompt: >
    On its very first step, the Adam optimizer (with its usual bias
    correction) updates two weights whose gradients are 1000 and 0.001.
    Which one moves further?
  options:
    - the one with gradient 1000, by a million times more
    - the one with gradient 1000, by about 1000 times more
    - they move by about the same amount
    - the one with gradient 0.001
  answer: 2
mutants:
  - name: no bias correction
    file: m03/train.py
    replace: ["m_hat = m / (1 - self.beta1 ** self.t)", "m_hat = m"]
  - name: targets aren't shifted
    file: m03/train.py
    replace: ["Y = np.stack([data[s + 1:s + block_size + 1] for s in starts])", "Y = np.stack([data[s:s + block_size] for s in starts])"]
  - name: decays biases and gains too
    file: m03/train.py
    replace: ["if self.weight_decay and p.data.ndim >= 2:", "if self.weight_decay:"]
---

## Hook

Plain gradient descent struggles with transformers: different parameters
need wildly different step sizes. Embedding rows for rare tokens get tiny,
occasional gradients; layer-norm gains get large steady ones. One learning
rate is too big for some and too small for others.

Adam gives every parameter its own step size, based on the running
averages of its gradient and squared gradient.

Then you'll train your GPT on a corpus. Tiny Shakespeare (1 MB) is bundled.
If you'd rather it learn something else, any text file works: your own
writing, a codebase, a book.

## Build

In `m03/train.py`:

1. `build_char_vocab(text)` and `get_batch(data, block_size, batch_size,
   rng)`: random windows, targets shifted one to the right.
2. `AdamW.step()`: m ← β₁m + (1−β₁)g, v ← β₂v + (1−β₂)g², bias-correct
   both with 1 − βᵗ, decay matrices by lr·wd·p, then
   p ← p − lr · m̂ / (√v̂ + ε).
3. `train_gpt(model, data, steps, ...)`.

```python
from workbench.data import shakespeare      # or corpus("~/my-text.txt")
from workbench import show_training, show_attention, grad_norms
text = shakespeare()
stoi, itos = build_char_vocab(text)
data = np.array([stoi[c] for c in text])
model = GPT(len(itos), block_size=32, n_embd=64, n_head=4, n_layer=2)
dash = show_training("GPT on Shakespeare")
named = {f"p{i} {p.shape}": p for i, p in enumerate(model.parameters()) if p.data.ndim == 2}
losses = train_gpt(model, data, steps=1500, lr=3e-3,
    log=lambda s, l: dash.log(s, loss=l, tokens=16 * 32,
                              grad_norms=grad_norms(named) if s % 10 == 0 else None))
dash.done()
print("".join(itos[i] for i in model.generate([stoi["\n"]], 300, 0.8, np.random.default_rng(0))))
show_attention(lambda s: (list(s), model.attention_weights([stoi[c] for c in s])),
               "ROMEO:\nWhat light", title="your GPT's attention")
```

Expect a few minutes and a loss around 2.0: words, line breaks and speaker
names start to appear. Type prompts into the attention viewer and click
heads to pin them.

## Explain

**Adam** (Kingma and Ba, 2015) divides each parameter's averaged gradient by
the root of its averaged squared gradient. On the first step, with bias
correction, that ratio is exactly g/|g|, so every parameter moves by lr
regardless of its gradient's size: the answer was "about the same". Over
time the step adapts to each parameter's typical gradient scale.

The **bias correction** matters early: m and v start at 0, so without
dividing by 1 − βᵗ the first steps would be far too small. **AdamW**
(Loshchilov and Hutter, 2019) applies weight decay directly to the weights
instead of adding it to the gradient, where Adam's rescaling would distort
it. Gains and biases are conventionally not decayed.

Your engine trains at a few thousand tokens per second. In the next lesson
the same model in PyTorch runs much faster on the Apple GPU. That's not a
smarter algorithm, just the same arithmetic in fused, compiled kernels.

## Stretch

Add a learning-rate schedule: linear warmup for 100 steps, then cosine
decay to 10%. Log `lr` to the dashboard and compare final losses.
