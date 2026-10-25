"""Looking inside: activations, feature visualization, dataset examples.

Lesson 4.4: get_activations and visualize_channel. Lesson 4.5:
top_activating.
"""

import numpy as np
import torch


def get_activations(model, layer, x):
    """Run ``model`` on ``x`` and return ``layer``'s output (a torch tensor),
    captured with ``layer.register_forward_hook``. Remove the hook
    afterwards, even if the model raises."""
    raise NotImplementedError("lesson 4.4")


def visualize_channel(model, layer, channel, steps=200, lr=0.05, size=32, seed=0,
                      l2=1e-3, jitter=2):
    """Gradient ascent on the input: find an image that makes ``channel`` of
    ``layer`` fire as strongly as possible, on average over positions.

    Start from faint noise (torch.randn * 0.1, from a torch.Generator seeded
    with ``seed``) and optimize it with Adam. Each step, roll the image by up
    to ``jitter`` pixels, take the mean of the channel's activation after a
    ReLU, subtract ``l2`` times the image's squared norm, and step to
    increase that. Return the
    image as a (size, size) NumPy array.
    """
    raise NotImplementedError("lesson 4.4")


@torch.no_grad()
def top_activating(model, layer, channel, X, k=9):
    """The k inputs in X where ``channel``'s activation, maximized over
    positions, is largest. Returns (indices, activations) as NumPy arrays,
    largest first."""
    raise NotImplementedError("lesson 4.5")
