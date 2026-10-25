---
id: "4.3"
title: A CNN that knows its shapes
concepts: [cnn, image-classification, adam]
file: m04/cnn.py
stub: stubs/cnn.py
tests: test_cnn.py
predict:
  kind: number
  prompt: >
    ShapeCNN has conv1 (1 → 8 channels, 5 × 5, with biases), conv2
    (8 → 16 channels, 5 × 5, with biases) and a linear layer from the
    16 × 5 × 5 pooled features to 6 classes (with biases). How many
    parameters in total?
  answer: 5830
  tolerance: 0
mutants:
  - name: trains on shuffled labels
    file: m04/cnn.py
    replace: ["F.cross_entropy(model(X[idx]), Y[idx])", "F.cross_entropy(model(X[idx]), Y[idx.flip(0)])"]
  - name: 3 × 3 first kernel
    file: m04/cnn.py
    replace: ["nn.Conv2d(1, 8, 5)", "nn.Conv2d(1, 8, 3)"]
  - name: accuracy counts mistakes
    file: m04/cnn.py
    replace: ["(pred == np.asarray(Y)).mean()", "(pred != np.asarray(Y)).mean()"]
---

## Hook

Your NumPy convolution works, and module 3 showed that writing the backward
pass yourself teaches you everything it's going to. So from here we use
PyTorch's `nn.Conv2d` and spend the effort on what the network learns.

The data: `workbench.data.shapes()` draws 32 × 32 grayscale images of six
classes (circles, squares, triangles, crosses, horizontal and vertical
stripes) at random sizes and positions with noise.

## Build

In `m04/cnn.py`:

1. `ShapeCNN(n_classes)`: conv 5 × 5 (1 → 8) → ReLU → max pool 2 → conv 5 × 5
   (8 → 16) → ReLU → max pool 2 → flatten → linear. Name the layers
   `conv1`, `conv2`, `fc`; later lessons look inside them by name.
2. `train_cnn(model, X, Y, steps, batch_size, lr, seed, log)`: Adam on random
   minibatches with cross-entropy. Call `log(step, loss)` each step if given.
3. `accuracy(model, X, Y)`.

```python
from workbench import show_training, show_images
from workbench.data import shapes, SHAPE_CLASSES
X, Y = shapes(3000); Xt, Yt = shapes(500, seed=1)
show_images(X[:12], labels=[SHAPE_CLASSES[y] for y in Y[:12]])
model = ShapeCNN(6)
dash = show_training(title="ShapeCNN")
train_cnn(model, X, Y, steps=400, log=lambda s, l: dash.log(s, l))
accuracy(model, Xt, Yt)
```

Above 95% held-out accuracy in a few seconds is normal.

## Explain

conv1: 8 × (1 × 5 × 5) + 8 = 208. conv2: 16 × (8 × 5 × 5) + 16 = 3216. The
spatial size goes 32 → 28 (conv) → 14 (pool) → 10 (conv) → 5 (pool), so fc
reads 16 × 5 × 5 = 400 features: 400 × 6 + 6 = 2406. Total **5830**.

An MLP with one 64-unit hidden layer on the same 1024 pixels would need
over 65,000. Two-thirds of the parameters here are in the final linear
layer; the convolutions themselves are tiny because of weight sharing.

This is the LeNet shape. The same pattern, scaled up with more layers,
ReLUs, dropout and GPUs, is AlexNet (Krizhevsky, Sutskever and Hinton,
2012), which started the deep learning era on ImageNet.

## Stretch

Train an MLP on the same data with a similar parameter count and compare
held-out accuracy. Then test both on images shifted 4 pixels right, by
`np.roll`. Which degrades more?
