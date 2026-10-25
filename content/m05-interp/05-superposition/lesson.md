---
id: "5.5"
title: Toy models of superposition
concepts: [superposition, polysemanticity, sparsity]
file: m05/toy.py
stub: stubs/toy.py
tests: test_superposition.py
predict:
  kind: choice
  prompt: >
    A model must squeeze 5 features into 2 dimensions and reconstruct them
    through a ReLU. Each feature is zero 90% of the time. How many of the 5
    does it learn to represent?
  options:
    - "2: one per dimension"
    - "3"
    - "all 5"
    - "none: it's too lossy"
  answer: 2
mutants:
  - name: no ReLU
    file: m05/toy.py
    replace: ["return torch.relu(h @ self.W + self.b)", "return h @ self.W + self.b"]
  - name: sparsity inverted
    file: m05/toy.py
    replace: ["keep = rng.random((n, n_features)) >= sparsity", "keep = rng.random((n, n_features)) < sparsity"]
  - name: interference counts itself
    file: m05/toy.py
    replace: ["    overlap.fill_diagonal_(0)\n", ""]
---

## Hook

In module 4 some channels fired on unrelated things. If each neuron meant
one thing, interpretability would be a matter of labeling neurons. It
isn't, and Elhage et al., "Toy Models of Superposition" (2022), showed why
with a model small enough to understand completely.

## Build

In `m05/toy.py`:

1. `ToyModel(n_features, n_hidden)`: h = W x, x̂ = ReLU(Wᵀ h + b).
2. `sample_features(n, n_features, sparsity, rng)`: each feature 0 with
   probability `sparsity`, else uniform in [0, 1].
3. `train_toy(model, sparsity, importance, …)`: Adam on
   Σ_i importance_i (x_i − x̂_i)².
4. `feature_norms`, `represented`, and `interference(model)`: how much the
   other features overlap each feature's direction.

```python
from workbench import show_embeddings, show_matrix
importance = 0.8 ** np.arange(5)
for s in [0.0, 0.7, 0.9]:
    m = ToyModel(5, 2); train_toy(m, s, importance)
    show_embeddings(m.W.detach().T.numpy(), labels=[f"f{i}" for i in range(5)], title=f"sparsity {s}")
    show_matrix((m.W.T @ m.W).detach().numpy(), title=f"WᵀW, sparsity {s}")
```

## Explain

With dense features, a 2-dimensional model keeps the 2 most important
features at right angles and drops the rest, like PCA. With sparse
features it represents **all 5**, arranged as a pentagon. Five directions
in two dimensions can't be orthogonal, so features interfere: when f0 is
active, f1's readout sees some of it. But if features are rarely active at
the same time, the interference is rarely there, and the ReLU (with a
negative bias) filters out the small leaks. The model trades a little
noise for many more features.

That's **superposition**, and it explains polysemantic neurons: a model
with more features than neurons stores them as directions that aren't
aligned with any single neuron. To find the features, you need to find the
directions, which is what the sparse autoencoder in the next lesson does.

Without the ReLU (the mutant), the model is linear and can do no better
than PCA at any sparsity.

## Stretch

Sweep sparsity from 0 to 0.99 and plot the number of represented features
and the "dimensions per feature" (n_hidden / ‖W‖²_F). Elhage et al. found
it moves in sudden jumps between geometric arrangements. Do you see
plateaus?
