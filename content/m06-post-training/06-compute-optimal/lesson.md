---
id: "6.6"
title: Spending a compute budget
concepts: [compute-optimal, chinchilla, isoflop]
file: m06/scaling.py
stub: stubs/scaling.py
tests: test_compute_optimal.py
predict:
  kind: choice
  prompt: >
    Under the Chinchilla fit, how should model size N and training tokens D
    grow as the compute budget grows 100×?
  options:
    - N grows 100×, D stays the same
    - both grow about 10×
    - D grows 100×, N stays the same
    - N grows 1000×, D shrinks
  answer: 1
mutants:
  - name: forgets the factor of 6
    file: m06/scaling.py
    replace: ["    N = G * (C / 6) ** (beta / (alpha + beta))\n    return N, (C / 6) / N", "    N = G * C ** (beta / (alpha + beta))\n    return N, C / N"]
  - name: swapped exponents
    file: m06/scaling.py
    replace: ["N = G * (C / 6) ** (beta / (alpha + beta))", "N = G * (C / 6) ** (alpha / (alpha + beta))"]
  - name: parabola minimum sign
    file: m06/scaling.py
    replace: ["return float(np.exp(-b / (2 * a)))", "return float(np.exp(b / (2 * a)))"]
---

## Hook

Training compute is about 6·N·D FLOPs: N parameters, D tokens. For a
fixed budget, a bigger model sees fewer tokens. Kaplan et al. (2020)
concluded that you should mostly grow the model; Hoffmann et al., "Training
Compute-Optimal Large Language Models" (2022), re-ran the experiment more
carefully and found that GPT-3-era models were far too big for their data.
Their 70B Chinchilla, trained on 4× more tokens than the 280B Gopher with
the same compute, beat it.

## Build

In `m06/scaling.py`:

1. `flops(N, D)`.
2. `chinchilla_loss(N, D, E, A, B, alpha, beta)`.
3. `optimal_allocation(C, A, B, alpha, beta)`: substitute D = C/(6N),
   set the derivative in N to zero, and solve. The docstring has the answer
   to check your algebra against.
4. `isoflop_minimum(n_params, losses)`: for runs at one fixed budget,
   fit a parabola in log N and return its bottom.

```python
P = dict(A=406.4, B=410.7, alpha=0.34, beta=0.28)
for C in [1e19, 1e21, 1e23, 1e25]:
    N, D = optimal_allocation(C, **P)
    print(f"{C:.0e} FLOPs: {N / 1e9:.2f}B params, {D / 1e9:.0f}B tokens, {D / N:.0f} tokens/param")
```

Then make your own isoflop curve: pick a budget (say 2·10¹³ FLOPs), train
models of several widths each for D = C / (6N) tokens, and find the
minimum.

## Explain

With α = 0.34 and β = 0.28, N grows as C^(β/(α+β)) = C^0.45 and D as
C^0.55: **both grow about 10×** for 100× compute (C^0.5 if α = β). The
famous rule of thumb, about 20 tokens per parameter, comes from where the
constants put the line.

The derivation is a two-line Lagrangian, but the conclusion changed the
field: Llama trained small models far past "optimal" (because inference
cost depends on N, not training compute), and data, not parameters,
became the scarce resource.

The isoflop method is the robust one: at a fixed budget, too small and too
large models both lose, and the loss against log N is close to a parabola
whose bottom is the optimal size. Doing it at several budgets and fitting
a line through the optima gives the scaling exponent without assuming the
functional form.

## Stretch

Inference matters too. If a model will serve 10¹² tokens after training,
and serving costs 2·N FLOPs per token, what N minimizes total (training +
serving) compute for a target loss? (Sardana and Frankle, 2023.)
