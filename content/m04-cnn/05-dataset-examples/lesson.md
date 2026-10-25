---
id: "4.5"
title: What makes a channel fire, for real
concepts: [dataset-examples, feature-visualization]
file: m04/featvis.py
stub: stubs/featvis.py
tests: test_dataset_examples.py
predict:
  kind: choice
  prompt: >
    You rank images by how strongly a channel fires. Image A has a weak
    response spread over the whole image; image B has one strong response
    in a small spot and nothing elsewhere. If the score is the channel's
    maximum over positions, which ranks higher?
  options:
    - A
    - B
    - it depends on the image size
  answer: 1
mutants:
  - name: average over positions
    file: m04/featvis.py
    replace: [".max(dim=1).values", ".mean(dim=1)"]
  - name: smallest first
    file: m04/featvis.py
    replace: ["np.argsort(-score, kind=\"stable\")", "np.argsort(score, kind=\"stable\")"]
---

## Hook

An optimized image tells you what a channel *could* respond to. It can
mislead: the optimizer is free to build patterns no real image contains.
The complement is cheap and grounded: run the whole dataset through the
model and look at the images where the channel fires hardest.

## Build

In `m04/featvis.py`, `top_activating(model, layer, channel, X, k=9)`: get
the channel's activations on all of X, score each image by its **maximum**
over positions, and return the indices and scores of the top k, largest
first. Use `np.argsort(..., kind="stable")` so ties keep their order.

```python
from workbench import show_images
from workbench.data import SHAPE_CLASSES
for c in range(4):
    idx, acts = top_activating(model, model.conv2, c, Xt, k=8)
    show_images(Xt[idx], labels=[f"{SHAPE_CLASSES[Yt[i]]} {a:.1f}" for i, a in zip(idx, acts)],
                title=f"conv2 #{c}")
```

Put each channel's dataset examples next to its visualization from 4.4. Do
they agree?

## Explain

A conv channel is a detector that runs at every position, so the natural
question is "did it find its pattern anywhere?". The maximum answers that;
the mean mixes in all the places the pattern isn't, and ranks an image full
of weak texture above a clean single instance. So **B** ranks higher.

Dataset examples and feature visualizations fail in opposite directions.
Examples show only what the data contains and can be dominated by a
correlated feature (a "triangle" channel that really fires on sharp
corners). Visualizations can show patterns that never occur. Olah et al.
recommend always looking at both. When they disagree you've learned
something.

Many channels are **polysemantic**: their top examples are a mix of
unrelated things (corners and stripes, say). Module 5 is about why that
happens and how to pull those mixtures apart.

## Stretch

Show the activation map for each of the top examples, not just the image,
using `show_matrix`. Where in each image does the channel fire? Then find a
channel whose top 8 examples come from at least 3 different classes.
