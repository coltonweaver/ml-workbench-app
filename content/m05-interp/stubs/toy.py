"""Toy models of superposition: more features than dimensions.

Lesson 5.5."""
import numpy as np
import torch


class ToyModel(torch.nn.Module):
    """Squeeze n features into m < n dimensions and back:
    h = W x, x̂ = ReLU(Wᵀ h + b), with W of shape (m, n)."""

    def __init__(self, n_features, n_hidden, seed=0):
        raise NotImplementedError('lesson 5.5')

    def forward(self, x):
        raise NotImplementedError('lesson 5.5')


def sample_features(n, n_features, sparsity, rng):
    """(n, n_features): each feature is 0 with probability ``sparsity``,
    otherwise uniform in [0, 1]."""
    raise NotImplementedError('lesson 5.5')


def train_toy(model, sparsity, importance, steps=2000, batch_size=1024, lr=0.01, seed=0):
    """Adam on importance-weighted squared error, Σ_i importance_i (x_i - x̂_i)²,
    averaged over the batch. Returns the losses."""
    raise NotImplementedError('lesson 5.5')


def feature_norms(model):
    """‖W_i‖ for each feature's column: near 1 if represented, near 0 if
    dropped. (n_features,)"""
    raise NotImplementedError('lesson 5.5')


def represented(model, threshold=0.5):
    """How many features the model represents."""
    raise NotImplementedError('lesson 5.5')


def interference(model):
    """For each feature i, Σ_{j≠i} (Ŵ_i · W_j)²: how much the other features
    overlap its direction (Ŵ_i is W_i normalized). (n_features,)"""
    raise NotImplementedError('lesson 5.5')
