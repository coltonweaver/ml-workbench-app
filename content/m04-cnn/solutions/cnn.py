"""A small convolutional network, in PyTorch."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeCNN(nn.Module):
    """conv 5×5 (1→8) → ReLU → pool 2 → conv 5×5 (8→16) → ReLU → pool 2 →
    linear to the classes. For 32 × 32 inputs."""

    def __init__(self, n_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc = nn.Linear(16 * 5 * 5, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # (N, 8, 14, 14)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # (N, 16, 5, 5)
        return self.fc(x.flatten(1))


def train_cnn(model, X, Y, steps=300, batch_size=64, lr=1e-2, seed=0, log=None):
    """Adam on random minibatches of (X, Y) (NumPy arrays). Returns the
    losses. ``log(step, loss)`` is called each step if given."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(X), size=batch_size))
        loss = F.cross_entropy(model(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log is not None:
            log(step, losses[-1])
    return losses


@torch.no_grad()
def accuracy(model, X, Y):
    """Fraction of X the model classifies correctly."""
    model.eval()
    pred = model(torch.as_tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()
    return float((pred == np.asarray(Y)).mean())
