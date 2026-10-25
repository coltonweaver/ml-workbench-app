---
id: "1.7"
title: Trust, but verify
concepts: [gradient-check, finite-differences]
file: m01/gradcheck.py
stub: stubs/gradcheck.py
tests: test_gradcheck.py
predict:
  kind: choice
  prompt: >
    Suppose your tanh's backward used (1 − t) instead of (1 − t²). You compare
    it with finite differences at x = 0 only. Does the check catch the bug?
  options:
    - Yes
    - No
    - Only if h is small enough
  answer: 1
mutants:
  - name: forgets the 2 in the central difference
    file: m01/gradcheck.py
    replace: ["/ (2 * h)", "/ h"]
warmups:
  - kind: choice
    prompt: "In float64 with h = 1e-6, your analytic gradient and the finite difference have a relative error of 1e-2. What does that say?"
    options: ["That's normal floating-point noise", "There's probably a bug in a backward rule", "h is too small"]
    answer: 1
    explain: "With float64 and a central difference, a correct gradient agrees to about 1e-7 or better. 1e-2 means a backward rule is wrong somewhere."
---

## Hook

You have two ways to get a gradient. Your engine's `backward()` is fast and
exact in principle, but it's code you just wrote. Lesson 1.3's finite
differences are slow and approximate, but they're too simple to get wrong.

Use the simple one to test the fast one. This is the checkpoint for module 1:
the rest of the course trusts this engine.

## Build

In `m01/gradcheck.py`:

1. `gradcheck(f, xs, h)`: `f` takes a list of `Value`s and returns a `Value`.
   Return `(analytic, numeric)` pairs, one per input. For `analytic`, run
   `f` once and call `backward()`. For `numeric`, use a central difference on
   input `i`, calling `f` on fresh Values each time.
2. `passes(f, xs, tol)`: True if every pair agrees to within `tol`, measured
   as `|a − n| / max(1, |a|, |n|)`.

Then point it at your engine:

```python
f = lambda v: (v[0] * v[1] + v[2].exp()).tanh() / (v[0] ** 2 + 1)
gradcheck(f, [0.3, -1.2, 0.5])
passes(f, [0.3, -1.2, 0.5])
```

The hidden tests also run your engine through random expressions. This is
the module's checkpoint: **your engine has to match finite differences.**

## Explain

This is a **gradient check**. Every serious autograd implementation is
tested this way, and you'll reach for it again whenever you write a custom
backward rule.

Two lessons hide in it:

- **Test at generic points.** At x = 0, tanh's correct slope 1 − t² and the
  buggy 1 − t are both 1, so a check there can't tell them apart. Points like
  0, 1 and integers often make wrong formulas coincide. Use random,
  non-round inputs.
- **Use relative error.** A gradient of 1,000 off by 0.01 is fine; a gradient
  of 0.001 off by 0.01 is completely wrong. Comparing relative error
  (floored at 1 so tiny values don't blow it up) handles both.

Finite differences in double precision usually agree with a correct backward
to about 1e-7 relative error. Disagreement at 1e-2 means a bug, not
round-off.

## Stretch

Break your engine on purpose: change `__mul__`'s backward to use
`self.data` for both inputs. Which test expressions does `passes` still
accept? What's the smallest expression that exposes the bug?
