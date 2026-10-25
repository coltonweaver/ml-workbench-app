"""Train, validation and test splits."""

import numpy as np

from m02.losses import cross_entropy


def split(words, fractions=(0.8, 0.1, 0.1), seed=42):
    """Shuffle ``words`` (without changing the input) and cut it into
    consecutive pieces of the given fractions. Returns a list of lists."""
    assert abs(sum(fractions) - 1.0) < 1e-9
    order = np.random.default_rng(seed).permutation(len(words))
    shuffled = [words[i] for i in order]
    out, start = [], 0
    for i, f in enumerate(fractions):
        end = len(words) if i == len(fractions) - 1 else start + int(round(f * len(words)))
        out.append(shuffled[start:end])
        start = end
    return out


def evaluate(model, X, Y, batch_size=4096):
    """Mean cross-entropy of the model on all of X, Y (no training)."""
    total = 0.0
    for i in range(0, len(X), batch_size):
        loss = cross_entropy(model(X[i:i + batch_size]), Y[i:i + batch_size])
        total += float(loss.data) * len(X[i:i + batch_size])
    return total / len(X)


def generalization_gap(train_loss, val_loss):
    """How much worse the model does on data it hasn't seen. Positive means
    it's memorizing the training set."""
    return val_loss - train_loss
