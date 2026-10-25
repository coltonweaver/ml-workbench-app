"""Reading a circuit: which earlier features does a later feature use?"""

import contextlib

import numpy as np
import torch

from m04.featvis import get_activations


def circuit_strengths(conv_weight, out_channel):
    """How strongly each input channel feeds ``out_channel``: the L2 norm of
    its kh × kw weight slice, for a conv weight (C_out, C_in, kh, kw).
    Returns a (C_in,) NumPy array."""
    raise NotImplementedError


def excitation(conv_weight, out_channel):
    """The sum of each input channel's weights into ``out_channel``: positive
    means it mostly excites, negative that it mostly inhibits. (C_in,)."""
    raise NotImplementedError


@contextlib.contextmanager
def ablate(layer, channel):
    """Within the ``with`` block, ``layer``'s output has ``channel`` set to 0.
    A forward hook that returns a value replaces the layer's output."""
    raise NotImplementedError
    yield


@torch.no_grad()
def ablation_effect(model, ablate_layer, ablate_channel, read_layer, read_channel, X):
    """Mean activation of ``read_channel`` (of ``read_layer``) on X, normally
    and with ``ablate_channel`` of ``ablate_layer`` zeroed. Returns
    (normal, ablated) as floats."""
    raise NotImplementedError


@torch.no_grad()
def class_accuracy_drop(model, layer, channel, X, Y, n_classes):
    """Per-class accuracy, normally minus with ``channel`` ablated: which
    classes depend on this feature. A (n_classes,) array."""
    raise NotImplementedError
