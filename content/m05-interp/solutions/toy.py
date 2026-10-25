"""Toy models of superposition: more features than dimensions."""

import numpy as np
import torch


class ToyModel(torch.nn.Module):
    """Squeeze n features into m < n dimensions and back:
    h = W x, x̂ = ReLU(Wᵀ h + b), with W of shape (m, n)."""

    def __init__(self, n_features, n_hidden, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(torch.randn(n_hidden, n_features, generator=gen) * 0.3)
        self.b = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        h = x @ self.W.T
        return torch.relu(h @ self.W + self.b)


def sample_features(n, n_features, sparsity, rng):
    """(n, n_features): each feature is 0 with probability ``sparsity``,
    otherwise uniform in [0, 1]."""
    x = rng.random((n, n_features))
    keep = rng.random((n, n_features)) >= sparsity
    return (x * keep).astype(np.float32)


def train_toy(model, sparsity, importance, steps=2000, batch_size=1024, lr=1e-2, seed=0):
    """Adam on importance-weighted squared error, Σ_i importance_i (x_i - x̂_i)²,
    averaged over the batch. Returns the losses."""
    rng = np.random.default_rng(seed)
    importance = torch.as_tensor(importance, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = model.W.shape[1]
    losses = []
    for _ in range(steps):
        x = torch.as_tensor(sample_features(batch_size, n, sparsity, rng))
        loss = (importance * (x - model(x)) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def feature_norms(model):
    """‖W_i‖ for each feature's column: near 1 if represented, near 0 if
    dropped. (n_features,)"""
    return model.W.detach().norm(dim=0).numpy()


def represented(model, threshold=0.5):
    """How many features the model represents."""
    return int((feature_norms(model) > threshold).sum())


def interference(model):
    """For each feature i, Σ_{j≠i} (Ŵ_i · W_j)²: how much the other features
    overlap its direction (Ŵ_i is W_i normalized). (n_features,)"""
    W = model.W.detach()
    unit = W / W.norm(dim=0, keepdim=True).clamp_min(1e-8)
    overlap = (unit.T @ W) ** 2
    overlap.fill_diagonal_(0)
    return overlap.sum(dim=1).numpy()
