---
id: "3.4"
title: Order, and thinking per token
concepts: [positional-encoding, gelu, feed-forward]
file: m03/layers.py
stub: stubs/layers.py
tests: test_positions_mlp.py
predict:
  kind: choice
  prompt: >
    Take attention without the causal mask and without any position
    information. Shuffle the order of the input tokens. What happens to the
    outputs?
  options:
    - they don't change at all
    - they're shuffled the same way, otherwise identical
    - they change unpredictably
    - they all become equal
  answer: 1
mutants:
  - name: gelu forgets half of x
    file: m03/layers.py
    replace: ["((x + 0.044715 * x ** 3) * c).tanh() + 0.5 * x", "((x + 0.044715 * x ** 3) * c).tanh()"]
  - name: sin and cos swapped
    file: m03/layers.py
    replace: ["pe[:, 0::2] = np.sin(angle)", "pe[:, 0::2] = np.cos(angle)"]
  - name: MLP drops its output bias
    file: m03/layers.py
    replace: ["@ self.W2 + self.b2", "@ self.W2"]
---

## Hook

"dog bites man" and "man bites dog" contain the same tokens. Attention only
compares contents, so without help it can't tell them apart. It needs each
token to carry a sense of where it is.

The other missing piece: attention moves information between tokens, but
only by averaging. Each token also needs a place to compute something
nonlinear with what it gathered.

## Build

In `m03/layers.py`:

1. `gelu(x)`: the tanh approximation, from Tensor ops.
2. `MLP.__call__`: `gelu(x @ W1 + b1) @ W2 + b2`. The hidden layer is 4×
   wider than n_embd.
3. `sinusoidal_positions(T, n_embd)`: sin in even dimensions, cos in odd
   ones, at wavelengths from 2π up to 10000·2π.

```python
pe = sinusoidal_positions(64, 32)
from workbench import show_matrix
show_matrix(pe, title="sinusoidal positions (rows = positions)")
show_matrix(pe @ pe.T, title="similarity between positions")
```

## Explain

Attention without positions is **permutation-equivariant**: permuting the
inputs permutes the outputs the same way. A **positional encoding** breaks
the symmetry by adding a different vector at each position. The
**sinusoidal** one from Vaswani et al. has a neat property your test checks:
the dot product between two positions depends only on their distance, so
"3 tokens back" looks the same everywhere. GPT-2 instead *learns* a position
embedding table, which is what your GPT will use. Modern models mostly use
RoPE (Su et al., 2021), which rotates queries and keys by position angle.

The **MLP** (or feed-forward) sublayer acts on each token separately. It
holds about two-thirds of a transformer's parameters, and much of what
module 5 finds about where models store facts points at these layers.

**GELU** (Hendrycks and Gimpel, 2016) is a smooth ReLU: x times the
probability that a standard normal is below x. GPT-2 used this tanh
approximation, and your PyTorch parity check in lesson 3.8 depends on
matching it exactly.

## Stretch

Implement RoPE: rotate each pair of query/key dimensions by an angle
proportional to position. Show that q·k then depends only on the offset.
