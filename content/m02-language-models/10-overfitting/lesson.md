---
id: "2.10"
title: Memorizing isn't learning
concepts: [train-val-split, overfitting, generalization]
file: m02/evaluate.py
stub: stubs/evaluate.py
tests: test_overfitting.py
predict:
  kind: choice
  prompt: >
    Train the MLP model with a big hidden layer (1,000 units) on just 200
    names for a long time. Loss on those 200 names keeps falling. What
    happens to the loss on 3,000 names it never saw?
  options:
    - falls too, just more slowly
    - falls, then rises
    - stays at ln 27
    - matches the training loss
  answer: 1
mutants:
  - name: no shuffle before splitting
    file: m02/evaluate.py
    replace: ["order = np.random.default_rng(seed).permutation(len(words))", "order = np.arange(len(words))"]
  - name: averages batch losses without weights
    file: m02/evaluate.py
    replace: ["total += float(loss.data) * len(X[i:i + batch_size])", "total += float(loss.data) * batch_size"]
---

## Hook

So far you've measured loss on the same names you trained on. A model with
enough parameters could simply memorize them, and its loss would look
wonderful while its samples were copies of the training set.

What you care about is how well the model predicts text it hasn't seen. The
only way to measure that is to hide some data from training.

## Build

In `m02/evaluate.py`:

1. `split(words, fractions=(0.8, 0.1, 0.1), seed)`: shuffle a copy, then cut
   it into pieces of the given fractions (train, validation, test).
2. `evaluate(model, X, Y, batch_size)`: mean cross-entropy over all of X,
   computed in batches. Weight each batch by its size.
3. `generalization_gap(train_loss, val_loss)`.

This is the module 2 checkpoint. Run the experiment from the prediction:

```python
train_w, val_w, test_w = split(words)
Xtr, Ytr = build_dataset(train_w[:200], stoi, 3)
Xva, Yva = build_dataset(val_w, stoi, 3)
model = MLPLanguageModel(27, n_hidden=1000)
dash = show_training("overfitting")
def log(step, loss):
    val = evaluate(model, Xva, Yva) if step % 200 == 0 else None
    dash.log(step, loss=loss, val_loss=val)
train(model, Xtr, Ytr, steps=6000, batch_size=64, lr=0.1, log=log)
dash.done()
```

## Explain

The validation loss falls while the model learns patterns that hold for all
names, then rises once it starts memorizing quirks of its 200. That is
**overfitting**, and the gap between validation and training loss measures
it. A model that does as well on new data as on its training data
**generalizes**.

The three splits have separate jobs. **Train** is for gradient descent.
**Validation** is for decisions: when to stop, how big a model, how much
regularization. Because you make choices based on it, you slowly overfit to
it too. **Test** is looked at once, at the end, for the number you report.

Shuffling first matters because datasets are often sorted (these names are
sorted by popularity), and an unshuffled split would put the rare names
only in validation.

The cures are what you'd expect: more data, a smaller model, weight decay
(lesson 2.7), or stopping when validation loss is lowest. Large language
models are in an unusual regime: they see most of their data only once, so
classic overfitting is rare, and module 6's scaling laws describe how loss
falls with model size and data instead.

## Stretch

Train the full-size MLP on the whole training split and report your test
loss once. Then try to beat it on validation with any change you like
(embedding size, hidden size, context length, learning rate decay) and only
then check test again.
