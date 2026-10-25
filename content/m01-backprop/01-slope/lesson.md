---
id: "1.1"
title: Which way is downhill?
concepts: [derivative, finite-differences]
file: m01/slope.py
stub: stubs/slope.py
tests: test_slope.py
predict:
  kind: number
  prompt: >
    At w = 10 the cost is 50.00. At w = 10.01 it is 50.03. Near w = 10, about
    how much does the cost change for each 1 unit you add to w?
  answer: 3
  tolerance: 0.1
mutants:
  - name: divides by h instead of 2h
    file: m01/slope.py
    replace: ["(2 * h)", "h"]
warmups:
  - kind: number
    prompt: "f(x) = x². With h = 0.1, what does the central difference (f(3 + h) − f(3 − h)) / 2h give at x = 3?"
    answer: 6
    tolerance: 0.02
    explain: "(9.61 − 8.41) / 0.2 = 6, exactly f′(3) = 2·3. For a quadratic the central difference has no error at all: the h² terms cancel."
---

## Hook

You run a service whose monthly cost depends on one knob, `w` (say, how many
cache nodes you keep warm). You can't see the formula. You can only try a
setting and read the bill:

| w     | cost  |
|-------|-------|
| 10    | 50.00 |
| 10.01 | 50.03 |

Two questions. Which way should you turn the knob to spend less? And if you
turned it a whole unit, roughly how much would the bill move?

The function is in your file as `cost(w)`. Try a few values in the kernel
before you answer.

## Build

Write `slope(f, x, h)` in `m01/slope.py`. It should work for **any** function
of one number, not just `cost`.

Measure on both sides of `x`: look at `f(x + h)` and `f(x - h)`, and divide
the change in `f` by the change in `x`.

Then press **Run** and try it:

```python
slope(cost, 10)   # what sign? what size?
slope(cost, 7)
slope(cost, 3)
```

When you think it's right, press **Check**.

## Explain

What you measured is the **derivative**: the rate at which the output changes
per unit of input, at one point. It's written f′(x) or df/dx. Its sign says
which way is uphill, and its size says how steep.

Your `slope` is a **finite-difference** estimate. You measured on both sides,
which makes it a *central* difference. For a smooth function its error shrinks
with h², while the one-sided version `(f(x+h) - f(x)) / h` only shrinks with h.

For `cost(w) = ½(w − 7)² + 45.5`, calculus gives f′(w) = w − 7 exactly. That's
3 at w = 10, 0 at the bottom (w = 7), and negative to the left of it. The sign
alone tells you which way to turn the knob.

This is the only calculus the whole course needs at its core. Everything else
is bookkeeping for doing it with millions of knobs at once.

## Stretch

Make `h` tiny: try `slope(cost, 10, h=1e-12)` and then `h=1e-15`. The answer
gets *worse*. Why? Floating-point numbers carry about 16 significant digits,
and `f(x + h) - f(x - h)` subtracts two nearly equal numbers. Find the `h`
that gives the most accurate answer for `cost` at 10.
