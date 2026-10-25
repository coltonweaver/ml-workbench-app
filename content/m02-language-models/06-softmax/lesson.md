---
id: "2.6"
title: From scores to probabilities
concepts: [softmax, cross-entropy, numerical-stability]
file: m02/losses.py
stub: stubs/losses.py
tests: test_softmax.py
predict:
  kind: choice
  prompt: >
    softmax turns scores into probabilities: exp(each score), divided by the
    sum of the exps. softmax([1, 2, 3]) ≈ [0.09, 0.24, 0.67]. What is
    softmax([1001, 1002, 1003])?
  options:
    - the same, about [0.09, 0.24, 0.67]
    - about [0.33, 0.33, 0.33]
    - about [0, 0, 1]
    - it can't be computed
  answer: 0
mutants:
  - name: softmax without the max shift
    file: m02/losses.py
    replace: ["    e = (x - shift).exp()", "    e = x.exp()"]
  - name: cross-entropy gradient not averaged
    file: m02/losses.py
    replace: ["logits.grad += g / n * out.grad", "logits.grad += g * out.grad"]
  - name: cross-entropy gradient ignores the upstream gradient
    file: m02/losses.py
    replace: ["logits.grad += g / n * out.grad", "logits.grad += g / n"]
warmups:
  - kind: number
    prompt: "Logits for two classes are 1000 and 1000. The right class is the first. What is the cross-entropy loss? (to 3 decimals)"
    answer: 0.693
    tolerance: 0.02
    explain: "Softmax only sees differences, so this is softmax([0, 0]) = [0.5, 0.5], and the loss is −ln 0.5 = ln 2. Computing exp(1000) directly overflows; subtract the max first."
---

## Hook

A neural network outputs arbitrary real numbers, one score per possible next
token. The bigram table, by contrast, held probabilities. To compare a
network to the data with NLL, you need a way to turn any vector of scores
into a probability distribution.

Exponentiate (now everything is positive) and divide by the total (now it
sums to 1). Before you code it, think about what adding 1000 to every score
does, mathematically, and what it does to `np.exp` on a computer.

## Build

In `m02/losses.py`, using your `Tensor` ops:

1. `softmax(x, axis=-1)`. Subtract the max along `axis` first. The max is a
   constant (use `x.data.max(...)`), so it needs no gradient.
2. `log_softmax(x, axis=-1)`: `z - log(sum(exp(z)))` with the same shift,
   never `log(softmax(x))`.
3. `cross_entropy(logits, targets)`: the mean of `-log_softmax(logits)` at
   each row's target. Make it one node with its own `_backward`. Work out the
   gradient on paper first: it is remarkably simple.

```python
from m02.losses import softmax, cross_entropy
softmax(Tensor([1001.0, 1002.0, 1003.0])).data
logits = Tensor(np.zeros((4, 27)))
cross_entropy(logits, np.array([5, 13, 13, 1])).data    # ln 27
```

## Explain

**Softmax** turns scores (called **logits**) into a distribution. Adding a
constant to every logit doesn't change it, since the constant's eᶜ cancels
between the numerator and denominator. So the answer was "the same". But
`exp(1001)` overflows to infinity in floating point, and inf / inf is NaN.
Subtracting the max is free mathematically and makes the largest exponent
exactly e⁰ = 1. This is **numerical stability**, and it's in every real
softmax.

**Cross-entropy** loss is the NLL from lesson 2.3, computed from logits:
the average of -log softmax(logits)[target]. Its gradient with respect to
the logits is

    (softmax(logits) − one_hot(target)) / N

Push up the score of the right answer, push down the others in proportion to
how much probability they took. Fusing it into one node makes it faster and
avoids taking the log of a probability that underflowed to 0. PyTorch's
`F.cross_entropy` takes logits for the same reason.

## Stretch

Derive the gradient of cross-entropy yourself, starting from
`-z[t] + log Σⱼ exp(z[j])`. Then derive the Jacobian of softmax alone. Why is
it a full matrix, while the fused gradient is so cheap?
