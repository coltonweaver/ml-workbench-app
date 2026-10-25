---
id: "4.1"
title: Sliding a small window
concepts: [convolution, stride, padding, weight-sharing]
file: m04/conv.py
stub: stubs/conv.py
tests: test_convolution.py
predict:
  kind: number
  prompt: >
    A 5 × 5 kernel slides over a 32 × 32 image, one pixel at a time, with no
    padding. Only positions where the whole kernel fits on the image count.
    How wide is the output?
  answer: 28
  tolerance: 0
mutants:
  - name: flips the kernel
    file: m04/conv.py
    replace: ["w[:, :, u, v])", "w[:, :, kh - 1 - u, kw - 1 - v])"]
  - name: ignores the stride when reading
    file: m04/conv.py
    replace: ["x[:, :, u:u + stride * Ho:stride, v:v + stride * Wo:stride]", "x[:, :, u:u + Ho, v:v + Wo]"]
  - name: off by one in the size
    file: m04/conv.py
    replace: ["(size + 2 * padding - kernel) // stride + 1", "(size + 2 * padding - kernel) // stride"]
  - name: forgets the padding
    file: m04/conv.py
    replace: ["    if padding:\n", "    if False:\n"]
warmups:
  - kind: number
    prompt: "A 28×28 image, a 5×5 kernel, stride 1, no padding. What is the output's width?"
    answer: 24
    tolerance: 0.01
    explain: "28 − 5 + 1 = 24. In general (W − K + 2P) / S + 1."
---

## Hook

A 32 × 32 image is 1024 numbers. Feed it to an MLP and the first layer has a
separate weight for every pixel, so a circle in the top left and the same
circle in the bottom right are unrelated inputs. It has to learn "circle"
once for every place a circle can be.

A convolution learns one small pattern detector (say 5 × 5 weights) and
slides it over every position. Same weights everywhere: a detector that
finds an edge in one corner finds it in every corner.

## Build

In `m04/conv.py`, with NumPy only:

1. `output_size(size, kernel, stride=1, padding=0)`: how many positions the
   kernel fits along one side.
2. `conv2d(x, w, b=None, stride=1, padding=0)` for `x` of shape
   (N, C_in, H, W) and `w` of shape (C_out, C_in, kh, kw). Each output
   channel sums over every input channel and every kernel offset.

A loop over output pixels is correct but slow. Faster: loop over the kh × kw
kernel offsets instead. For each offset (u, v), take the strided view of `x`
that offset touches, and combine it with `w[:, :, u, v]` in one `np.einsum`.

```python
import numpy as np
x = np.arange(16.0).reshape(1, 1, 4, 4)
w = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
conv2d(x, w)    # -5 everywhere: x[i, j] - x[i+1, j+1]
```

The hidden tests compare against `torch.nn.functional.conv2d` with strides,
padding and biases.

## Explain

Along one side, the kernel's left edge can sit at positions 0 to
32 − 5 = 27, which is **28** positions. In general:
`(size + 2·padding − kernel) // stride + 1`. Padding by 2 on each side keeps
a 5 × 5 output the same size as its input ("same" padding); stride 2 halves
it.

What every framework calls convolution is mathematically
**cross-correlation**: the kernel isn't flipped. A true convolution flips
it, which matters for signal processing proofs and not at all for learning,
since the network would just learn the flipped weights.

Two properties make convolutions the right tool for images. **Weight
sharing**: a 5 × 5 conv from 1 to 8 channels has 8 × 25 + 8 = 208
parameters, whatever the image size. **Translation equivariance**: shift
the input and the output shifts with it. This is the inductive bias of
LeCun et al., "Gradient-Based Learning Applied to Document Recognition"
(1998), and of every CNN since.

## Stretch

Time your `conv2d` against a version that loops over output pixels, on a
batch of 64 images of 32 × 32 with 16 channels. Then write `conv2d` a third
way: build all the patches into one matrix (`im2col`) and do a single
matrix multiply. Which is fastest, and why do GPUs like the last one?
