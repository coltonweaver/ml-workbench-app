---
id: "1.3"
title: Many knobs at once
concepts: [gradient, partial-derivative, conditioning]
file: m01/gradient.py
stub: stubs/gradient.py
tests: test_gradient.py
predict:
  kind: choice
  prompt: >
    f(x, y) = x² + 10·y². You start at (5, 5) and walk downhill with
    lr = 0.04, moving both knobs each step. Which knob gets close to its
    best value (0) first?
  options:
    - x
    - y
    - both at the same pace
  answer: 1
mutants:
  - name: moves every knob by the first knob's slope
    file: m01/gradient.py
    replace: ["x - lr * gi", "x - lr * g[0]"]
---

## Hook

Real systems have many knobs. Say your service cost depends on cache nodes
`x` and replica count `y`:

```
f(x, y) = x² + 10·y²
```

(The optimum here is (0, 0). That's unrealistic, but it keeps the arithmetic easy.)

You can still ask "which way is downhill?" Ask it one knob at a time: hold
`y` still and nudge `x`, then hold `x` still and nudge `y`.

Before building, predict which knob settles first when both move with the
same learning rate.

## Build

In `m01/gradient.py`:

1. `gradient(f, xs, h)`: `f` takes a **list** of numbers. Return the list of
   one-knob slopes, nudging only `xs[i]` for entry `i`. Don't mutate `xs`.
2. `descend_many(f, xs0, lr, steps)`: like `descend`, but every knob moves
   against its own slope. Return the path.

Try it:

```python
f = lambda v: v[0] ** 2 + 10 * v[1] ** 2
gradient(f, [5, 5])           # [10, 100]?
path = descend_many(f, [5, 5], 0.04, 30)
[p for p in path[::5]]
```

Then look at it from above. The probe's arrow points downhill:

```python
from workbench import show_landscape
show_landscape(f, ((-6, 6), (-6, 6)), path=path, labels=["x", "y"])
```

Or watch it live, one step at a time:

```python
h = show_landscape(f, ((-6, 6), (-6, 6)))
for p in descend_many(f, [5, 5], 0.09, 25):
    h.add(p)
```

## Explain

Each one-knob slope is a **partial derivative**, written ∂f/∂x. The list of
all of them is the **gradient**, written ∇f. The gradient points uphill, in the
steepest direction, and its length says how steep that is. Gradient descent
in many dimensions is exactly what you built: `x ← x − lr · ∇f(x)`.

Why did `y` settle first? Its slope is `20y` against `2x` for x, so with one
learning rate, `y` shrinks by a factor of `(1 − 20·lr) = 0.2` per step while
`x` only shrinks by `(1 − 2·lr) = 0.92`. Raise `lr` to speed up `x` and `y`
diverges once `lr > 0.1`. When directions have very different curvature, the
function is **ill-conditioned**. This is a big reason training is slow, and
the motivation for optimizers like momentum and Adam, which you'll meet later.

Notice the cost of your `gradient`, too. It calls `f` twice per knob. A
network with 100 million weights would need 200 million forward passes for a
single step. The next lessons fix that.

## Stretch

Count the calls: wrap `f` so it increments a counter, then compute
`gradient` for a function of 1,000 inputs. Now estimate how long one gradient
step would take for a 100M-parameter model if one forward pass takes 50 ms.
