---
id: "3.3"
title: Keeping signals in range
concepts: [layer-normalization]
file: m03/layers.py
stub: stubs/layers.py
tests: test_layer_norm.py
predict:
  kind: choice
  prompt: >
    Layer norm rescales each vector to mean 0 and variance 1 (then applies a
    learned scale, here 1, and shift, here 0). What does it output for the
    vector [2, 2, 2, 2]?
  options:
    - "[1, 1, 1, 1]"
    - "[0, 0, 0, 0]"
    - "[0.5, 0.5, 0.5, 0.5]"
    - NaN, from dividing by zero
  answer: 1
mutants:
  - name: normalizes across the batch
    file: m03/layers.py
    replace: ["mean = x.mean(axis=-1, keepdims=True)", "mean = x.mean(axis=0, keepdims=True)"]
  - name: eps outside the square root
    file: m03/layers.py
    replace: ["(var + self.eps) ** -0.5", "var ** -0.5 + self.eps"]
---

## Hook

A GPT stacks many blocks, each adding its output to a running vector. With
nothing to hold them in check, those vectors drift: their size grows block
after block, the softmaxes saturate, and gradients vanish or explode. Deep
networks used to be nearly untrainable for this reason.

One standard fix: before each sublayer, rescale every vector to a standard
size.

## Build

In `m03/layers.py`, `LayerNorm.__call__`: using Tensor ops, take the mean
and the variance (`((x - mean) ** 2).mean`) over the last axis with
`keepdims=True`, then `(x - mean) * (var + eps) ** -0.5 * gamma + beta`.

```python
ln = LayerNorm(4)
ln(Tensor([[2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0]])).data.round(3)
```

## Explain

**Layer normalization** (Ba, Kiros and Hinton, 2016) normalizes each
token's vector on its own, across its features. Its cousin batch norm
normalizes each feature across the batch, which works for images but couples
examples together and breaks for variable-length text.

A constant vector has mean 2 and variance 0, so it becomes 0 / √(0 + ε) = 0.
That's what the ε is for. Put it outside the square root and you divide by
zero.

Because the normalization is followed by a learned gain γ and shift β, layer
norm doesn't limit what the network can represent. It only controls the
scale the next layer sees. Many recent models use RMSNorm (Zhang and
Sennrich, 2019), which skips the mean subtraction and the shift.

## Stretch

Implement RMSNorm: x / √(mean(x²) + ε) · γ. Swap it into the GPT after
lesson 3.6 and compare training curves.
