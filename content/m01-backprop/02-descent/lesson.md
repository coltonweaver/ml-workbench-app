---
id: "1.2"
title: Walking downhill
concepts: [gradient-descent, learning-rate]
file: m01/descent.py
stub: stubs/descent.py
tests: test_descent.py
predict:
  kind: choice
  prompt: >
    You walk downhill on f(w) = (w − 7)², starting at w = 10, moving
    w ← w − lr · slope each step. With lr = 1.1, what has happened after 20 steps?
  options:
    - w has settled at 7
    - w has settled somewhere other than 7
    - w swings back and forth, further from 7 each time
    - w hasn't moved
  answer: 2
mutants:
  - name: steps uphill
    file: m01/descent.py
    replace: ["x - lr", "x + lr"]
---

## Hook

You now know which way is downhill at any point. That means you can *walk*:
measure the slope, take a step against it, and repeat.

How big a step? If the slope is 3, a step of `0.1 × 3` seems careful and a
step of `2 × 3` seems bold. Your knob is the multiplier, called `lr` for now.

Before you write anything, predict what a multiplier of 1.1 does on
`f(w) = (w − 7)²` starting from 10.

## Build

Write `descend(f, x0, lr, steps)` in `m01/descent.py`, using the `slope` you
wrote in lesson 1.1 (it's already imported).

Each step moves `x` to `x - lr * slope(f, x)`. Return the whole path,
starting with `x0`, so you can look at it.

Then try it in the kernel:

```python
f = lambda w: (w - 7) ** 2
descend(f, 10, 0.1, 20)[-1]   # near 7?
descend(f, 10, 0.9, 20)       # look at the whole path
descend(f, 10, 1.1, 20)       # and this one
```

To *see* the walk, draw it on the landscape (it opens in the Instruments tab;
drag on the plot to probe the slope anywhere):

```python
from workbench import show_landscape
show_landscape(f, (0, 14), path=descend(f, 10, 0.9, 20), title="lr = 0.9")
```

## Explain

This is **gradient descent**, and `lr` is the **learning rate**. Cauchy
described the method in 1847. It is still the core of how every neural
network is trained.

On `f(w) = (w − 7)²` the slope is `2(w − 7)`. One step therefore multiplies
the distance to the bottom by `(1 − 2·lr)`:

- lr = 0.1: the distance shrinks by 0.8 each step, a steady approach.
- lr = 0.9: the factor is −0.8, so w overshoots and oscillates, but still converges.
- lr = 1.1: the factor is −1.2, so each step lands 20% further away on the other side. It diverges.

So "too large a learning rate" has a precise meaning: on this function, any lr
above 1 diverges. On a real network the safe range depends on the curvature,
which you can't see. That's why learning rates get tuned, scheduled and warmed up.

## Stretch

Run `descend` on `f(w) = w⁴` from `w = 2` with lr = 0.01 for 1,000 steps. It
gets slower and slower near the bottom. Why? Look at what the slope does as
w → 0. Is the same problem present for `(w − 7)²`?
