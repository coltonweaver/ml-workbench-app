---
id: "6.5"
title: Scaling laws
concepts: [scaling-laws, power-law, irreducible-loss]
file: m06/scaling.py
stub: stubs/scaling.py
tests: test_scaling_laws.py
predict:
  kind: number
  prompt: >
    Loss follows L = 10 · N^(−0.1) with model size N. Going from 10 million
    to 1 billion parameters (100×), by what factor does the loss shrink?
    (Give L(1B) / L(10M), to two decimals.)
  answer: 0.63
  tolerance: 0.01
mutants:
  - name: exponent sign flipped
    file: m06/scaling.py
    replace: ["return float(np.exp(intercept)), float(-slope)", "return float(np.exp(intercept)), float(slope)"]
  - name: fits on linear axes
    file: m06/scaling.py
    replace: ["lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))", "lx, ly = np.asarray(x, float), np.log(np.asarray(y, float))"]
  - name: floor stuck at zero
    file: m06/scaling.py
    replace: ["for c in np.linspace(0, y.min() * 0.999, grid):", "for c in [0.0]:"]
  - name: counts the embeddings
    file: m06/scaling.py
    replace: ["    if not embeddings:\n        n -=", "    if False:\n        n -="]
warmups:
  - kind: number
    prompt: "Loss follows L(N) = 10 · N^−0.5. What is the loss at N = 400?"
    answer: 0.5
    tolerance: 0.02
    explain: "10 / √400 = 0.5. Every 4× in N halves this loss: a power law is a straight line on log-log axes."
---

## Hook

Kaplan et al., "Scaling Laws for Neural Language Models" (2020), trained
hundreds of models and found that loss falls as a power law in model size,
data and compute, smoothly over seven orders of magnitude. That turned
model building into forecasting: train small, fit, extrapolate. You'll do
it on your own GPTs.

## Build

In `m06/scaling.py`:

1. `fit_power_law(x, y)`: y = a·x^(−α) is a straight line on log-log axes,
   so fit log y against log x with `np.polyfit`.
2. `fit_power_law_offset(x, y)`: y = a·x^(−α) + c, with c the loss no model
   can beat (the entropy of the text itself). Search c on a grid.
3. `count_params(model, embeddings=False)`.
4. `size_sweep(widths, train, val, vocab_size, steps, …)`: train a GPT per
   width with your `train_torch` from 3.8, return parameters and
   validation losses.

```python
from workbench import series, show_plot
widths = [16, 24, 32, 48, 64, 96, 128]
N, L = size_sweep(widths, data[:900_000], data[900_000:], len(itos), steps=1500, block_size=64)
a, alpha, c = fit_power_law_offset(N, L)
xs = np.logspace(np.log10(N.min()), np.log10(N.max() * 10), 50)
show_plot([series("runs", N, L, kind="scatter"), series(f"fit α={alpha:.2f}", xs, a * xs ** -alpha + c)],
          log_x=True, log_y=True, x_label="non-embedding parameters", y_label="val loss")
```

Use the Apple GPU (`device=pick_device()` in `train_torch`); the sweep takes
a few minutes.

## Explain

L(1B)/L(10M) = (10⁹/10⁷)^(−0.1) = 100^(−0.1) ≈ **0.63**. Every 100× in size
buys the same 37% reduction. That's what a power law means: constant
returns per multiplicative step, and so diminishing returns per parameter.

Real curves bend toward a floor: a model can't beat the text's own
unpredictability. Fitting a pure power law to a curve with a floor gives a
much smaller α than the real one (the test checks this), so fit the
offset or stay far from the floor.

Two caveats for your sweep. Each model got the same number of steps, so
the bigger ones are undertrained relative to their size, and the curve
mixes model-size and data limits. And small models on a small corpus
reach the floor quickly; Kaplan et al. counted non-embedding parameters
precisely because embeddings distort the small end.

## Stretch

Hold the model fixed and sweep training steps instead (data scaling). Fit
L(D). Then fit L(N) on your three smallest models only and extrapolate:
how far off is the prediction for your largest?
