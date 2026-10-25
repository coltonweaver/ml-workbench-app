---
id: "2.2"
title: Counting what comes next
concepts: [bigram-model, probability-distribution, sampling]
file: m02/bigram.py
stub: stubs/bigram.py
tests: test_bigram.py
predict:
  kind: choice
  prompt: >
    Across all 32,033 names, which letter do the most names end with?
  options:
    - a
    - e
    - n
    - y
  answer: 2
mutants:
  - name: normalizes columns instead of rows
    file: m02/bigram.py
    replace: ["c / c.sum(axis=1, keepdims=True)", "c / c.sum(axis=0, keepdims=True)"]
  - name: forgets the smoothing
    file: m02/bigram.py
    replace: ["counts.astype(np.float64) + smoothing", "counts.astype(np.float64)"]
---

## Hook

The simplest possible language model: to guess the next character, look at
the current one and ask "in all the names I've seen, what came after this
character, and how often?"

That's a table of counts with one row per current character and one column
per next character. Before you build it, guess one entry of its last column.

## Build

In `m02/bigram.py`:

1. `count_matrix(words, stoi)`: `counts[i, j]` is how often character `j`
   follows character `i`, boundaries included.
2. `probabilities(counts, smoothing=1.0)`: add `smoothing` to every count,
   then divide each row by its sum. Row `i` is now the distribution of what
   follows character `i`.
3. `sample(P, itos, rng)`: generate a name by starting at the boundary and
   repeatedly drawing the next character from the current row, until the
   boundary comes up again.

```python
import numpy as np
from workbench import show_matrix
counts = count_matrix(words, stoi)
show_matrix(counts, itos, itos, title="bigram counts", log_scale=True)
P = probabilities(counts)
rng = np.random.default_rng(0)
[sample(P, itos, rng) for _ in range(10)]
```

The last column of the heatmap is how names end. Was your guess right?

## Explain

This is a **bigram language model**: the probability of the next token
depends only on the current one, P(next | current). Each row of `P` is a
**probability distribution** (non-negative, sums to 1), and drawing from it
is **sampling**. Generating text means sampling one token at a time and
feeding it back in. Every model in this course, up to the GPT, generates this
way. What changes is how much context it reads and how it computes the
distribution.

The names it makes are bad (`"anaeli"`, `"s"`) because one character of
context is very little. Ending is easy, though: 6,763 names end in "n" and
6,640 in "a".

Adding 1 to every count is **Laplace smoothing**. Without it, any bigram
that never appeared in the data gets probability 0, and the model would call
a perfectly reasonable new name impossible.

## Stretch

Make a trigram model: condition on the previous *two* characters with a
27 × 27 × 27 table. Do the samples look better? How many of the table's
entries are zero before smoothing?
