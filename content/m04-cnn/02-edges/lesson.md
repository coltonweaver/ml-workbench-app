---
id: "4.2"
title: Edges and pooling
concepts: [edge-detection, max-pooling, receptive-field]
file: m04/conv.py
stub: stubs/conv.py
tests: test_edges.py
predict:
  kind: choice
  prompt: >
    The Sobel x kernel [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] subtracts the
    column on the left from the column on the right. Slide it over an image
    of horizontal black and white stripes. What does the output look like?
  options:
    - strong along every stripe
    - strong only at the stripes' left and right ends
    - zero everywhere
    - alternating positive and negative rows
  answer: 2
mutants:
  - name: sobel_y is sobel_x
    file: m04/conv.py
    replace: ["return sobel_x().T", "return sobel_x()"]
  - name: average pooling
    file: m04/conv.py
    replace: [".max(axis=(3, 5))", ".mean(axis=(3, 5))"]
  - name: magnitude without padding
    file: m04/conv.py
    replace: ["g = conv2d(x, w, padding=1)[0]", "g = conv2d(x, w)[0]"]
warmups:
  - kind: number
    prompt: "Two 3×3 convolutions with stride 1, one after the other. How wide is the receptive field of one output pixel?"
    answer: 5
    tolerance: 0.01
    explain: "Each 3×3 layer adds 2: 1 + 2 + 2 = 5. Stacking small kernels grows the receptive field with fewer weights than one big kernel."
---

## Hook

Before anyone trained CNNs, vision researchers designed kernels by hand.
The most famous one finds edges. Your `conv2d` can run it; you'll see the
same kind of detector reappear, learned rather than designed, two lessons
from now.

## Build

Still in `m04/conv.py`:

1. `sobel_x()`: the 3 × 3 kernel above. It responds where brightness
   changes from left to right, so to **vertical** edges.
2. `sobel_y()`: its transpose, for horizontal edges.
3. `edge_magnitude(image)`: run both on a 2D image (padding 1, so the output
   has the image's size) and return √(gx² + gy²).
4. `max_pool2d(x, k=2)`: the max over each non-overlapping k × k window of
   (N, C, H, W). Drop leftover rows and columns. A reshape to
   (N, C, H/k, k, W/k, k) does it without loops.

```python
from workbench import show_images
from workbench.data import shapes
X, Y = shapes(6)
show_images([X[i, 0] for i in range(6)] + [edge_magnitude(X[i, 0]) for i in range(6)])
```

## Explain

Every row of a horizontal stripe is constant from left to right, so the
right column minus the left column is **zero everywhere**. Sobel x is blind
to horizontal structure; Sobel y sees nothing but it. Edge magnitude
combines them so any orientation shows up. The weights 1, 2, 1 along the
edge also smooth a little, making the detector less sensitive to noise.
This is the Sobel–Feldman operator (1968).

**Max pooling** keeps the strongest response in each window and throws
away exactly where in the window it was. That buys a little tolerance to
small shifts and halves the resolution, so the next layer's 5 × 5 kernel
covers a 10 × 10 region of the original image. The region of input a unit
can see is its **receptive field**; stacking convs and pools grows it until
late layers see the whole image.

## Stretch

Compose two 3 × 3 kernels: convolve an image with one, then the result with
the other. Show that this equals a single 5 × 5 kernel, and find it. What
does that say about why deep CNNs use stacks of small kernels instead of
one big one?
