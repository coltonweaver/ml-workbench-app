"""Reading a circuit: which earlier features does a later feature use?"""

import contextlib

import numpy as np
import torch

from m04.featvis import get_activations


def circuit_strengths(conv_weight, out_channel):
    """How strongly each input channel feeds ``out_channel``: the L2 norm of
    its kh × kw weight slice, for a conv weight (C_out, C_in, kh, kw).
    Returns a (C_in,) array."""
    w = torch.as_tensor(conv_weight).detach()
    return w[out_channel].flatten(1).norm(dim=1).numpy()


def excitation(conv_weight, out_channel):
    """The sum of each input channel's weights into ``out_channel``: positive
    means it mostly excites, negative that it mostly inhibits. (C_in,)."""
    w = torch.as_tensor(conv_weight).detach()
    return w[out_channel].flatten(1).sum(dim=1).numpy()


@contextlib.contextmanager
def ablate(layer, channel):
    """Within the ``with`` block, ``layer``'s output has ``channel`` set to 0."""

    def hook(module, inputs, output):
        output = output.clone()
        output[:, channel] = 0
        return output

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def ablation_effect(model, ablate_layer, ablate_channel, read_layer, read_channel, X):
    """Mean activation of ``read_channel`` (of ``read_layer``) on X, normally
    and with ``ablate_channel`` of ``ablate_layer`` zeroed. Returns
    (normal, ablated)."""
    normal = get_activations(model, read_layer, X)[:, read_channel].mean().item()
    with ablate(ablate_layer, ablate_channel):
        ablated = get_activations(model, read_layer, X)[:, read_channel].mean().item()
    return normal, ablated


@torch.no_grad()
def class_accuracy_drop(model, layer, channel, X, Y, n_classes):
    """Per-class accuracy, normally minus with ``channel`` ablated: which
    classes depend on this feature."""
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = np.asarray(Y)

    def per_class():
        pred = model(X).argmax(1).numpy()
        return np.array([(pred[Y == c] == c).mean() if (Y == c).any() else np.nan
                         for c in range(n_classes)])

    model.eval()
    normal = per_class()
    with ablate(layer, channel):
        ablated = per_class()
    return normal - ablated
