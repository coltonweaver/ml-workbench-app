"""Train, validation and test splits."""

import numpy as np

from m02.losses import cross_entropy


def split(words, fractions=(0.8, 0.1, 0.1), seed=42):
    """Shuffle ``words`` (without changing the input list) and cut it into
    consecutive pieces of the given fractions. Returns a list of lists that
    together hold every word exactly once."""
    raise NotImplementedError


def evaluate(model, X, Y, batch_size=4096):
    """Mean cross-entropy of the model on all of X, Y, in batches."""
    raise NotImplementedError


def generalization_gap(train_loss, val_loss):
    """How much worse the model does on data it hasn't seen."""
    raise NotImplementedError
