---
id: "2.5"
title: Matrices and lookups
concepts: [matrix-multiplication, embedding-lookup, autograd]
file: m02/tensor.py
stub: stubs/tensor.py
tests: test_tensor_ops.py
predict:
  kind: number
  prompt: >
    W is 27 × 27. A batch of 32 characters, as one-hot rows, forms X
    (32 × 27), and logits = X @ W. The batch happens to contain only 5
    distinct characters. After backward, how many rows of W have a nonzero
    gradient?
  answer: 5
  tolerance: 0
mutants:
  - name: lookup loses repeated rows
    file: m02/tensor.py
    replace: ["np.add.at(self.grad, index, out.grad)", "self.grad[index] += out.grad"]
  - name: matmul gradient without the transpose
    file: m02/tensor.py
    replace: ["out.grad @ np.swapaxes(other.data, -1, -2)", "out.grad @ other.data"]
  - name: tanh gradient uses the input
    file: m02/tensor.py
    replace: ["(1 - out.data ** 2) * out.grad", "(1 - self.data ** 2) * out.grad"]
---

## Hook

A neural network layer is a matrix multiply. Next lesson's model multiplies
one-hot rows by a weight matrix:

```python
X = np.eye(27)[[5, 13, 13, 1]]     # "emma" as one-hot rows, (4, 27)
W = np.random.randn(27, 27)
np.allclose(X @ W, W[[5, 13, 13, 1]])   # True
```

Multiplying by a one-hot row just picks out a row of W. That's worth
remembering: it's what an embedding layer is. Before building the gradient,
think about which rows of W the multiply ever touched.

## Build

Still in `m02/tensor.py`:

1. `__matmul__`: for `out = A @ B`, `dA = dout @ Bᵀ` and `dB = Aᵀ @ dout`
   (transpose the last two axes with `np.swapaxes`). Batched inputs
   broadcast, so `unbroadcast` both.
2. `exp`, `log`, `tanh`, `relu`. Use the output where it's handy: the
   derivative of eˣ is eˣ, and of tanh is 1 − tanh².
3. `reshape` and `transpose`: the gradient is reshaped (or transposed with
   the inverse permutation, `np.argsort(axes)`) back.
4. `__getitem__`: `W[ids]` picks rows. Its gradient adds the output gradient
   into the picked rows. The same row can be picked twice, so use
   `np.add.at`, which adds repeated indices instead of overwriting.

```python
W = Tensor(np.random.randn(27, 27))
ids = np.array([5, 13, 13, 1])
W[ids].sum().backward()
W.grad.sum(axis=1)       # nonzero only for rows 1, 5 and 13; row 13 counts double
```

## Explain

The rule for **matrix multiplication** comes from writing one entry out:
`out[i, k] = Σⱼ A[i, j] B[j, k]`, so `dA[i, j] = Σₖ dout[i, k] B[j, k]`,
which is `dout @ Bᵀ`. Shapes are a good check: `dA` must have A's shape, and
only one arrangement of `dout`, `B` and a transpose gives it.

**Row lookup** is how every language model turns token ids into vectors: an
**embedding** table `W[ids]`. It computes exactly `one_hot(ids) @ W` but
without building the one-hot matrix, and its gradient only touches the rows
that were used: 5 here. A model with a 50,000-token vocabulary updates only
the embeddings of the tokens in the batch.

`np.add.at` matters because `grad[ids] += g` silently drops repeats in
NumPy: each repeated index is written once, not accumulated.

## Stretch

Implement `concatenate([a, b], axis)` as a function. Its backward splits the
gradient back into pieces. You'll want it for multi-head attention, though
there's also a way around it with `reshape` and `transpose`.
