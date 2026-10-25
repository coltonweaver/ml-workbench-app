---
id: "4.4"
title: Asking a channel what it wants
concepts: [feature-visualization, forward-hooks, gradient-ascent]
file: m04/featvis.py
stub: stubs/featvis.py
tests: test_feature_vis.py
predict:
  kind: choice
  prompt: >
    You run gradient ascent on the input image to maximize a conv1 channel
    whose weights are the Sobel x kernel (a vertical-edge detector). What
    will the optimized image look like?
  options:
    - a single bright dot
    - vertical stripes
    - horizontal stripes
    - random noise, slightly brighter
  answer: 1
mutants:
  - name: gradient descent instead of ascent
    file: m04/featvis.py
    replace: ["(-objective).backward()", "objective.backward()"]
  - name: leaves the hook attached
    file: m04/featvis.py
    replace: ["        handle.remove()\n    return captured", "        pass\n    return captured"]
---

## Hook

Your CNN is 97% accurate and you have no idea how. Its 8 conv1 channels and
16 conv2 channels each compute something. To find out what, flip training
around: keep the weights fixed and optimize the **input** until a channel
fires as hard as it can. The image it converges to is a picture of what
that channel detects.

## Build

In `m04/featvis.py`:

1. `get_activations(model, layer, x)`: register a forward hook on `layer`
   (`layer.register_forward_hook(fn)`; `fn(module, inputs, output)`), run
   the model, remove the hook (even on an exception: `try`/`finally`), and
   return the captured output.
2. `visualize_channel(model, layer, channel, steps, lr, size, seed, l2,
   jitter)`: start from faint noise and use Adam on the image. Each step,
   roll the image by a random shift of up to `jitter` pixels, take the mean
   of `relu(activation)` over the channel's positions, subtract
   `l2 · ‖image‖²`, and step uphill.
   Seed a `torch.Generator` so the same seed gives the same image.

```python
from workbench import show_images
imgs = [visualize_channel(model, model.conv1, c) for c in range(8)]
show_images(imgs, labels=[f"conv1 #{c}" for c in range(8)], normalize="each")
imgs2 = [visualize_channel(model, model.conv2, c) for c in range(16)]
show_images(imgs2, labels=[f"conv2 #{c}" for c in range(16)], normalize="each")
```

## Explain

A Sobel x channel's activation is large wherever brightness rises from left
to right. Averaged over positions (after the ReLU), the image that maximizes it rises from
left to right everywhere it can, which means **vertical stripes**: dark,
bright, dark, bright, repeating at the kernel's scale. Gradient ascent
produces that pattern on its own, from noise.

Why the ReLU? A conv1 channel is linear in the image, and the plain mean of
a Sobel response over all positions telescopes: right column minus left
column, summed across the image, leaves only the borders. Averaging the
positive part rewards firing in many places, which is what "detects
vertical edges" means.

The two regularizers matter. Without the **L2 penalty**, the image grows
without bound (activations are linear in it). Without **jitter**, the
optimizer finds high-frequency patterns that exploit exact pixel alignment,
images that look like static to us. Olah, Mordvintsev and Schubert,
"Feature Visualization" (Distill, 2017) catalogs these tricks; it is the
standard reference.

conv1 channels usually come out as oriented edges and blobs, much like
hand-designed filters. conv2 channels combine them into corners, curves and
stripe textures. That shift from simple to compound is what "a hierarchy of
features" means concretely.

## Stretch

Add a transformation robustness term: also randomly scale the image by
0.9–1.1 each step. Does it change the conv2 visualizations? Then visualize
a **class**: maximize the `fc` logit for "triangle". What do you get?
