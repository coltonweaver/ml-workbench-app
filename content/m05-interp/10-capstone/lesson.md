---
id: "5.10"
title: "Capstone: critique a claim"
concepts: [ablation, baselines, interpretability-methodology]
file: m05/capstone.py
stub: stubs/capstone.py
tests: test_capstone.py
predict:
  kind: choice
  prompt: >
    A write-up says: "Head 1.0 is the induction head. Its induction score
    is 0.8, and zero-ablating it raises the loss on repeated text by 0.6
    nats." What is the most important missing check?
  options:
    - a higher induction score
    - the same ablation on every other head, and on non-repeated text, with mean ablation as well as zero
    - more training steps
    - a visualization of the attention pattern
  answer: 1
mutants:
  - name: ablates every head
    file: m05/capstone.py
    replace: ["        z[:, :, head] = 0 if kind == \"zero\" else mean", "        z[:] = 0 if kind == \"zero\" else mean"]
  - name: reports the loss, not the increase
    file: m05/capstone.py
    replace: ["out[layer, head] = loss(ablate_head(layer, head, kind, m)) - base", "out[layer, head] = loss(ablate_head(layer, head, kind, m))"]
  - name: ignores the margin
    file: m05/capstone.py
    replace: ["return claimed > 0 and claimed >= margin * max(best_other, 0.0), claimed, best_other", "return claimed > 0 and claimed >= max(best_other, 0.0), claimed, best_other"]
---

## Hook

Interpretability results are easy to overclaim. A pretty attention pattern,
a high score on a metric you designed, one striking example: each is
suggestive, none is a demonstration. This lesson hands you a claim about
the induction model from 5.4 and asks you to test it properly.

> **Claim.** "Head 1.0 implements induction in our model. Its induction
> score is 0.8 (figure 1, one example sequence). Zero-ablating it raises
> the loss on repeated text by 0.6 nats. Therefore the model's in-context
> copying is performed by head 1.0."

## Build

In `m05/capstone.py`:

1. `ablate_head(layer, head, kind, mean)`: a patch on `blocks.{layer}.attn.z`
   that zeroes one head's output or replaces it with its mean.
2. `head_means(model, idx)`.
3. `ablation_table(model, idx, kind, positions)`: the loss increase from
   ablating each head in turn, over a chosen slice of predictions.
4. `claim_check(table, layer, head, margin)`: is the claimed head's effect
   positive and at least `margin` times any other head's?

Then run the experiments on your 5.4 model and write your critique in a
`critique.md` next to your code:

```python
data = repeated_tokens(256, 12, 32, np.random.default_rng(7))
second = ablation_table(m, data, "mean", slice(12, None))     # repeated half
first = ablation_table(m, data, "mean", slice(0, 11))         # random half
zero = ablation_table(m, data, "zero", slice(12, None))
show_matrix(second, title="mean ablation, repeated half")
claim_check(second, 1, 0)
```

## Explain

The claim's evidence has four gaps, and **the missing comparisons** are
the most important:

- **No baseline.** 0.6 nats means nothing without the same number for every
  other head. In small models the previous-token head in layer 0 often
  hurts as much when ablated, because the induction head can't work
  without it. "Head 1.0 performs copying" should be "heads 0.x and 1.0
  together".
- **No control text.** If ablating the head also hurts the random first
  half, it's doing something general, not induction.
- **Zero ablation only.** Zero may be far from any value the head outputs,
  so the damage can come from the shock, not the missing function. Mean
  ablation (or resampling from another input) is the fairer test.
- **One example.** A score from one sequence, shown as a figure, is an
  anecdote. Average over hundreds.

And a logical gap: "necessary" (ablating it hurts) isn't "sufficient" (it
does the job). Patching the head alone from a clean run into a corrupted
one (5.3) tests sufficiency. Redundant heads can make a necessary-looking
head look unnecessary, too. Chan et al.'s causal scrubbing (2022) and the
"Interpretability Illusions" literature are good next reading.

## Stretch

Write the claim the evidence actually supports, with numbers. Then try to
break your own claim: train two more models with different seeds. Does the
same structure appear, and in the same heads?
