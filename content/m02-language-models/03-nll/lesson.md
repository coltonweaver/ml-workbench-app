---
id: "2.3"
title: Scoring a model
concepts: [likelihood, negative-log-likelihood, perplexity]
file: m02/nll.py
stub: stubs/nll.py
tests: test_nll.py
predict:
  kind: number
  prompt: >
    A model that knows nothing gives each of the 27 tokens probability 1/27,
    every time. Scored by the average of -ln(probability it gave the
    character that actually came next), what score does it get on any
    data? (Natural log; two decimals is enough.)
  answer: 3.2958
  tolerance: 0.01
mutants:
  - name: averages per word instead of per bigram
    file: m02/nll.py
    replace: ["return total / n", "return total / len(words)"]
  - name: forgets the minus sign
    file: m02/nll.py
    replace: ["total -= np.log", "total += np.log"]
---

## Hook

You have a model, and its samples look sort of name-like. Is it any good?
You need one number that says how well a model predicts real text, so you
can compare models and, later, train one by making that number smaller.

A natural score: for each character in real data, look at the probability
the model gave it. A good model gave the true next character high
probability. Multiplying all those probabilities gives the probability of
the whole dataset, but with 200,000 characters that underflows to 0. Take
logs instead, and the product becomes a sum.

## Build

In `m02/nll.py`:

1. `nll(P, words, stoi)`: the average of `-log P[a, b]` over every bigram
   `(a, b)` in every word, boundaries included. Average per bigram, not per
   word.
2. `perplexity(avg_nll)`: `exp(avg_nll)`.

```python
nll(P, words, stoi)                        # about 2.45
nll(P, ["colton"], stoi), nll(P, ["xqzv"], stoi)
perplexity(nll(P, words, stoi))
```

## Explain

The probability a model assigns to the data is its **likelihood**. Taking
logs and negating gives the **negative log-likelihood** (NLL), and its
average per token is the number every language model reports as its
**loss**. Lower is better. 0 means it was certain and right every time.

The uniform model scores ln 27 ≈ 3.30 on anything, because it gives every
character 1/27. The bigram model scores about 2.45 on names. That gap is
what it learned.

**Perplexity** is exp(NLL): the size of a uniform choice that would be just
as unsure. The bigram model's perplexity of about 11.6 means it's as
uncertain as if it were choosing among 11.6 equally likely characters, down
from 27. The same quantity, measured in bits instead of nats, is the
cross-entropy from Shannon's information theory (1948). You'll compute it
as "cross-entropy" in two lessons.

Smoothing matters here: with zero counts, one unseen bigram would give
-log 0 = infinity.

## Stretch

Plot NLL on the dataset as you vary the smoothing from 0.01 to 1,000. Where
is it best? What happens to the samples at 1,000?
