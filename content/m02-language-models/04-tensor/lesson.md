---
id: "2.4"
title: Numbers in bulk
concepts: [tensor, broadcasting, autograd]
file: m02/tensor.py
stub: stubs/tensor.py
tests: test_tensor_basics.py
predict:
  kind: number
  prompt: >
    a has shape (3, 1), b has shape (1, 4), and out = a + b has shape (3, 4)
    because NumPy stretches both. If loss = out.sum(), what is
    d loss / d a[0, 0]?
  answer: 4
  tolerance: 0
mutants:
  - name: add doesn't undo broadcasting
    file: m02/tensor.py
    replace: ["self.grad += unbroadcast(out.grad, self.shape)", "self.grad += out.grad"]
  - name: sum's gradient misses the dropped axis
    file: m02/tensor.py
    replace: ["            if axis is not None and not keepdims:\n                g = np.expand_dims(g, axis)\n", ""]
  - name: unbroadcast forgets size-1 axes
    file: m02/tensor.py
    replace: ["        if size == 1 and grad.shape[axis] != 1:", "        if False:"]
---

## Hook

Your module 1 `Value` holds one number. The neural bigram model coming up
has a 27 × 27 weight matrix and 228,000 training examples. As Values, that's
millions of Python objects per step, and it would take minutes.

The fix is the one NumPy already uses: make each node of the graph hold a
whole array. One `+` node adds two arrays; its backward pass handles every
entry at once.

There's one new wrinkle. NumPy lets you add arrays of different shapes by
**stretching** the smaller one:

```python
import numpy as np
a = np.array([[1], [2], [3]])      # (3, 1)
b = np.array([[10, 20, 30, 40]])   # (1, 4)
a + b                              # (3, 4)
```

Each entry of `a` ended up in four entries of the result. What does that
mean for its gradient?

## Build

In `m02/tensor.py`, the `Tensor` class wraps a NumPy array. `backward()` and
the ops built from others (`-`, `/`, `mean`) are already there.

1. `unbroadcast(grad, shape)`: sum `grad` down to `shape`. Sum away extra
   leading axes, then sum (keeping the axis) wherever `shape` has a 1 and
   `grad` doesn't.
2. `__add__`'s `_backward`, and `__mul__`: like `Value`, but pass each
   gradient through `unbroadcast` so it has its tensor's shape.
3. `__pow__` for a constant exponent.
4. `sum(axis, keepdims)`: the gradient of a sum copies the output gradient
   back to every entry that was summed. If the axis was dropped, put it back
   (`np.expand_dims`) before broadcasting.

```python
from m02.tensor import Tensor
a = Tensor(np.ones((3, 1))); b = Tensor(np.ones((1, 4)))
(a + b).sum().backward()
a.grad, b.grad
```

## Explain

This is a **tensor** autograd engine: the same reverse-mode differentiation
as module 1, one array per node instead of one number. PyTorch, JAX and MLX
work this way, with the array operations running in fast native code.

**Broadcasting** is NumPy's rule for combining shapes: line them up from the
right; a dimension of 1 stretches to match; missing leading dimensions count
as 1. In the forward pass a broadcast entry is *reused*, and module 1 taught
you that a reused node's gradients add up. So the backward pass of
broadcasting is a sum over the stretched axes. That's all `unbroadcast` does.
a[0, 0] was used in 4 outputs, so its gradient is 4.

The same thing explains `sum`: the forward pass reduces, the backward pass
broadcasts. Reduction and broadcasting are each other's gradients.

## Stretch

Add `max(axis)`. Its gradient goes only to the entry that was largest. What
should happen when two entries tie?
