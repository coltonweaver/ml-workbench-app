"""A small convolutional network, in PyTorch."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeCNN(nn.Module):
    """conv 5×5 (1→8) → ReLU → pool 2 → conv 5×5 (8→16) → ReLU → pool 2 →
    linear to the classes. For 32 × 32 inputs. Name the layers conv1, conv2
    and fc: later lessons look inside them."""

    def __init__(self, n_classes=6):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def train_cnn(model, X, Y, steps=300, batch_size=64, lr=1e-2, seed=0, log=None):
    """Adam on random minibatches of (X, Y) (NumPy arrays; draw indices with
    np.random.default_rng(seed)). Returns the list of losses (floats).
    ``log(step, loss)`` is called each step if given."""
    raise NotImplementedError


@torch.no_grad()
def accuracy(model, X, Y):
    """Fraction of X the model classifies correctly, as a float."""
    raise NotImplementedError
