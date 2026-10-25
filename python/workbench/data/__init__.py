"""Datasets bundled with the course.

- ``names()``: 32,033 first names, one per line, from the US Social Security
  Administration's public baby-name data (as collected for karpathy/makemore).
- ``shakespeare()``: about 1 MB of Shakespeare's plays ("tiny Shakespeare",
  from karpathy/char-rnn). Public domain.
- ``corpus(path)``: any text file of your own, for training your GPT on
  something else.
"""

from __future__ import annotations

import os

_DIR = os.path.dirname(os.path.abspath(__file__))


def names() -> list[str]:
    with open(os.path.join(_DIR, "names.txt")) as f:
        return f.read().split()


def shakespeare() -> str:
    with open(os.path.join(_DIR, "shakespeare.txt")) as f:
        return f.read()


def corpus(path: str) -> str:
    with open(os.path.expanduser(path)) as f:
        return f.read()


SHAPE_CLASSES = ["circle", "square", "triangle", "cross", "horizontal stripes", "vertical stripes"]


def shapes(n: int = 3000, size: int = 32, seed: int = 0, noise: float = 0.1):
    """A small image dataset, generated on the spot: white shapes on black,
    one of six classes (``SHAPE_CLASSES``), at random positions, sizes and
    stroke widths, plus Gaussian noise.

    Returns ``(images, labels)``: float32 images of shape (n, 1, size, size)
    in [0, 1] and int64 labels. Small enough to train a CNN on a laptop CPU
    in seconds, and structured enough (edges, corners, curves, orientations)
    to find recognizable features in it.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    images = np.zeros((n, 1, size, size), dtype=np.float32)
    labels = rng.integers(0, len(SHAPE_CLASSES), size=n)
    for i, label in enumerate(labels):
        r = rng.uniform(0.18, 0.34) * size
        cx, cy = rng.uniform(r + 1, size - r - 1, size=2)
        w = rng.uniform(1.0, 2.2)  # stroke width
        if label == 0:  # circle outline
            d = np.abs(np.hypot(xx - cx, yy - cy) - r)
            img = d < w
        elif label == 1:  # square outline, axis-aligned
            d = np.maximum(np.abs(xx - cx), np.abs(yy - cy))
            img = np.abs(d - r) < w
        elif label == 2:  # triangle outline, pointing up
            img = np.zeros_like(xx, dtype=bool)
            pts = [(cx, cy - r), (cx - r, cy + 0.8 * r), (cx + r, cy + 0.8 * r)]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
                t = np.clip(((xx - x0) * (x1 - x0) + (yy - y0) * (y1 - y0))
                            / ((x1 - x0) ** 2 + (y1 - y0) ** 2), 0, 1)
                img |= np.hypot(xx - (x0 + t * (x1 - x0)), yy - (y0 + t * (y1 - y0))) < w
        elif label == 3:  # plus sign
            img = ((np.abs(xx - cx) < w) & (np.abs(yy - cy) < r)) | (
                (np.abs(yy - cy) < w) & (np.abs(xx - cx) < r))
        else:  # stripes, filling a box
            period = rng.uniform(4.0, 7.0)
            coord = yy if label == 4 else xx
            box = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
            img = box & ((coord % period) < period / 2)
        images[i, 0] = img
    images += rng.normal(0, noise, size=images.shape).astype(np.float32)
    return np.clip(images, 0, 1), labels.astype(np.int64)
