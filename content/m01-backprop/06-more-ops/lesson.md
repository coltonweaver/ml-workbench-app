---
id: "1.6"
title: A bigger vocabulary
concepts: [local-derivatives, activation-functions]
file: m01/engine.py
tests: test_ops.py
predict:
  kind: number
  prompt: >
    t = tanh(x). At x = 0.5, t = 0.4621. If you nudge x by a tiny ε, t moves
    by about ε times what number?
  answer: 0.7864
  tolerance: 0.02
mutants:
  - name: tanh uses 1 - t instead of 1 - t²
    file: m01/engine.py
    replace: ["(1 - t ** 2)", "(1 - t)"]
  - name: relu passes gradient everywhere
    file: m01/engine.py
    replace: ["(1.0 if self.data > 0 else 0.0)", "1.0"]
warmups:
  - kind: number
    prompt: "What is the derivative of the sigmoid σ(x) = 1 / (1 + e^−x) at x = 0?"
    answer: 0.25
    tolerance: 0.02
    explain: "σ′(x) = σ(x)(1 − σ(x)) = 0.5 · 0.5. That 0.25 is the most a sigmoid ever passes back, which is one reason deep sigmoid nets learned slowly."
---

## Hook

With `+`, `*` and `backward` you can already differentiate any polynomial.
Neural networks also need a few other shapes: powers, division, and
"squashing" functions that bend a straight line.

Here's the pattern. Each new operation needs a forward rule and a *local*
slope: how its output moves when its input moves. The chain rule and your
`backward` handle the rest.

Try the local slope of `tanh` before you code it. In the kernel,
`slope(math.tanh, 0.5)` from lesson 1.1 will tell you the answer, but predict
first.

## Build

Add these to `Value`. Each one builds the output, records its inputs, and
attaches a `_backward` that adds *(local slope) × out.grad* into the input:

| operation | forward | local slope |
|---|---|---|
| `x ** k` (k a plain number) | xᵏ | k·xᵏ⁻¹ |
| `x.exp()` | eˣ | eˣ |
| `x.tanh()` | tanh x | 1 − tanh²x |
| `x.relu()` | max(0, x) | 1 if x > 0, else 0 |

Then define these **without writing any new `_backward`**, by combining
what you have:

- `-x` as `x * -1`
- `x - y` as `x + (-y)`, plus `__rsub__` for `3 - x`
- `x / y` as `x * y ** -1`, plus `__rtruediv__` for `3 / x`

## Explain

Every differentiable program is a composition of a small set of primitives.
Each primitive needs a local derivative, and nothing else. That's why real
frameworks have a few hundred hand-written backward rules and can still
differentiate arbitrary user code.

You met two **activation functions**. `tanh` squashes any input into (−1, 1).
Its slope, 1 − t², vanishes as |x| grows, so a saturated tanh unit barely
learns. `relu` (Nair & Hinton, 2010) passes positives unchanged and zeros
negatives. Its gradient is exactly 0 or 1, which made deep networks much
easier to train. It can also "die": a unit stuck negative gets no gradient
at all.

Defining `-`, `/` in terms of `*`, `+` and `**` is the same trick frameworks
use for **composite operations**. Correctness comes for free. The cost is a
bigger graph.

## Stretch

Add `log()` and `sigmoid()` (σ(x) = 1 / (1 + e⁻ˣ)). Write `sigmoid` once
directly with its own `_backward` (slope σ(1 − σ)) and once by composing
`exp`, `+` and `/`. Do they give the same gradients? Count the graph nodes
each version creates.
