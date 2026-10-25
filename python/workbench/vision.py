"""Image instruments: a grid of images, for datasets, convolution outputs and
feature visualizations.

    show_images(X[:16], labels=[classes[y] for y in Y[:16]])
    show_images(conv1.weight, title="first-layer filters")
    show_images(visualizations, labels=[f"channel {c}" for c in range(8)])
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import Handle, _provenance, tensor_summary
from .lm import as_array

IMAGES_MIME = "application/vnd.workbench.images+json"
MAX_IMAGES = 256
MAX_SIDE = 256


def _as_images(a: np.ndarray) -> np.ndarray:
    """Any common layout to (N, H, W, C) with C = 1 or 3."""
    if a.ndim == 2:
        return a[None, :, :, None]
    if a.ndim == 3:
        if a.shape[-1] == 3 and a.shape[0] != 3:
            return a[None]                      # (H, W, 3)
        return a[:, :, :, None]                 # (N, H, W)
    if a.ndim == 4:
        if a.shape[1] in (1, 3) and a.shape[-1] not in (1, 3):
            return a.transpose(0, 2, 3, 1)      # (N, C, H, W)
        if a.shape[-1] in (1, 3):
            return a                            # (N, H, W, C)
        # (C_out, C_in, kh, kw) filters with C_in not 1 or 3: show channel 0..
        n, c, h, w = a.shape
        return a.reshape(n * c, h, w)[:, :, :, None]
    raise ValueError(f"can't read shape {list(a.shape)} as images")


def images_payload(images: Any, labels: Sequence[str] | None = None, title: str | None = None,
                   normalize: str = "each") -> dict[str, Any]:
    """``normalize``: "each" stretches every image to the full gray range
    (good for filters and visualizations), "global" uses one range for all
    (good for comparing activations), "none" expects values in [0, 1]."""
    raw = as_array(images).astype(np.float64)
    a = _as_images(raw)
    if len(a) > MAX_IMAGES:
        raise ValueError(f"{len(a)} images is more than the grid shows ({MAX_IMAGES}); pass a slice")
    if max(a.shape[1:3]) > MAX_SIDE:
        raise ValueError(f"images are at most {MAX_SIDE} pixels on a side")
    if labels is not None and len(labels) != len(a):
        raise ValueError(f"{len(labels)} labels for {len(a)} images")
    a = np.nan_to_num(a)
    if normalize == "global":
        lo, hi = a.min(), a.max()
        scaled = (a - lo) / (hi - lo or 1.0)
    elif normalize == "each":
        lo = a.min(axis=(1, 2, 3), keepdims=True)
        hi = a.max(axis=(1, 2, 3), keepdims=True)
        scaled = (a - lo) / np.where(hi - lo == 0, 1.0, hi - lo)
    elif normalize == "none":
        scaled = np.clip(a, 0, 1)
    else:
        raise ValueError("normalize must be 'each', 'global' or 'none'")
    pixels = (scaled * 255).round().astype(np.uint8)
    out = []
    for i, img in enumerate(pixels):
        out.append({
            "label": str(labels[i]) if labels is not None else None,
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "channels": int(img.shape[2]),
            "pixels": base64.b64encode(img.tobytes()).decode("ascii"),
            "range": [float(a[i].min()), float(a[i].max())],
        })
    return {
        "kind": "images",
        "title": title,
        "images": out,
        "explain": [tensor_summary(raw, "images", _provenance(), layout="as passed to show_images")],
    }


def show_images(images: Any, labels: Sequence[str] | None = None, title: str | None = None,
                normalize: str = "each") -> Handle:
    """Show a grid of images: arrays or tensors shaped (H, W), (N, H, W),
    (N, 1 or 3, H, W) or (N, H, W, 3), or conv filters (C_out, C_in, kh, kw)."""
    return Handle(IMAGES_MIME, images_payload(images, labels, title, normalize))
