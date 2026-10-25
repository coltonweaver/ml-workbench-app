"""Looking inside: activations, feature visualization, dataset examples."""

import numpy as np
import torch


def get_activations(model, layer, x):
    """Run ``model`` on ``x`` and return ``layer``'s output, captured with a
    forward hook (and the hook removed afterwards)."""
    captured = {}

    def hook(module, inputs, output):
        captured["out"] = output

    handle = layer.register_forward_hook(hook)
    try:
        model(torch.as_tensor(x, dtype=torch.float32))
    finally:
        handle.remove()
    return captured["out"]


def visualize_channel(model, layer, channel, steps=200, lr=0.05, size=32, seed=0,
                      l2=1e-3, jitter=2):
    """Gradient ascent on the input: find an image that makes ``channel`` of
    ``layer`` fire as strongly as possible, on average over positions.

    Start from faint noise. Each step, shift the image by up to ``jitter``
    pixels (so the result doesn't depend on exact pixel alignment), measure
    the mean of the channel's ReLU'd activation (where it fires; a plain mean
    of a linear filter mostly cancels out), subtract ``l2`` times the image's
    squared norm (so it can't just grow without bound), and step uphill.
    Returns the image as a (size, size) NumPy array.
    """
    gen = torch.Generator().manual_seed(seed)
    img = (torch.randn(1, 1, size, size, generator=gen) * 0.1).requires_grad_()
    opt = torch.optim.Adam([img], lr=lr)
    model.eval()
    for _ in range(steps):
        dx, dy = (int(v) for v in torch.randint(-jitter, jitter + 1, (2,), generator=gen))
        shifted = torch.roll(img, shifts=(dy, dx), dims=(2, 3))
        act = torch.relu(get_activations(model, layer, shifted)[0, channel]).mean()
        objective = act - l2 * (img ** 2).sum()
        opt.zero_grad()
        (-objective).backward()
        opt.step()
    return img.detach()[0, 0].numpy()


@torch.no_grad()
def top_activating(model, layer, channel, X, k=9):
    """The k inputs in X where ``channel``'s activation, maximized over
    positions, is largest. Returns (indices, activations), largest first."""
    acts = get_activations(model, layer, X)[:, channel]
    score = acts.flatten(1).max(dim=1).values.numpy()
    order = np.argsort(-score, kind="stable")[:k]
    return order, score[order]
