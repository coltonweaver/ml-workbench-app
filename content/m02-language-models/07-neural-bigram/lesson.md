---
id: "2.7"
title: Learning the table
concepts: [gradient-descent, cross-entropy, regularization]
file: m02/neural_bigram.py
stub: stubs/neural_bigram.py
tests: test_neural_bigram.py
predict:
  kind: choice
  prompt: >
    Instead of counting, start W (27 × 27) at random and train it by gradient
    descent so softmax(W[current char]) predicts the next char, minimizing
    cross-entropy on all the names. After plenty of training, how does its
    loss compare to the counting model's 2.45?
  options:
    - clearly lower
    - about the same
    - clearly higher
  answer: 1
mutants:
  - name: steps uphill
    file: m02/neural_bigram.py
    replace: ["W.data -= lr * W.grad", "W.data += lr * W.grad"]
  - name: never resets the gradient
    file: m02/neural_bigram.py
    replace: ["        W.grad = np.zeros_like(W.data)\n", ""]
---

## Hook

Counting worked because a bigram model is simple enough to count. A model
that reads ten characters of context can't be counted: there are 27¹⁰
contexts, and almost none appear in the data.

So rebuild the bigram model the way every bigger model will be built:
parameters that start random, a loss that says how wrong they are, and
gradient descent. The model is one 27 × 27 matrix `W`. To predict what
follows character `x`, take row `x` of `W` as logits and softmax it.

Same model, different way of finding the numbers. Guess how the two compare.

## Build

In `m02/neural_bigram.py`:

1. `dataset(words, stoi)`: every bigram as two integer arrays, `xs` (current)
   and `ys` (next).
2. `train_bigram(xs, ys, vocab_size, steps, lr, reg, seed, log)`: start
   `W` from `rng.normal`, then each step: logits `W[xs]`, loss
   `cross_entropy(logits, ys) + reg * (W ** 2).mean()`, zero the grad,
   backward, step. Return `W` and the losses.

Watch it train in the dashboard. On all 228,000 bigrams each step takes
about half a second:

```python
from workbench import show_training
xs, ys = dataset(words, stoi)
dash = show_training("neural bigram")
W, losses = train_bigram(xs, ys, 27, steps=60, lr=50.0,
                         log=lambda s, l: dash.log(s, loss=l, tokens=len(xs)))
dash.done()
```

Then sample from it: `probabilities` is now `softmax(W)` row by row.

## Explain

The two models agree. Cross-entropy is minimized exactly when each row of
softmax(W) equals the observed next-character frequencies, which is what
counting computes directly. So training converges to the counts (to the
extent the regularization allows), at about 2.45 to 2.49.

The point is that gradient descent didn't need to know that. It only needed
a loss and gradients. That's why it scales to models nobody could count.

The `reg * mean(W²)` term is **L2 regularization** (weight decay). It pulls
W toward 0, which makes every row of softmax(W) closer to uniform: the same
effect as adding to every count, so it plays the role of smoothing. In
large models it's one of the main tools against overfitting.

The dashboard's tokens/sec is your engine's throughput. Remember the number:
you'll compare it with PyTorch in module 3.

## Stretch

Replace `W[xs]` with an explicit `Tensor(np.eye(27)[xs]) @ W` and time both.
Then sweep `reg` from 0 to 1 and plot the final loss against it.
