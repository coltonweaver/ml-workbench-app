"""Scaling laws: measure how loss falls with model size, fit a power law,
and choose model size for a compute budget.

Lessons 6.5 and 6.6."""
import numpy as np
import torch
from m03.torch_gpt import TorchGPT, train_torch


def fit_power_law(x, y):
    """Least-squares fit of y = a · x^(−α) on log-log axes. Returns (a, α)."""
    raise NotImplementedError('lesson 6.5')


def fit_power_law_offset(x, y, grid=400):
    """Fit y = a · x^(−α) + c, where c is the loss no model size can beat.
    For each c on a grid from 0 to just below min(y), fit a power law to
    y − c and keep the c with the smallest squared error in log space.
    Returns (a, α, c)."""
    raise NotImplementedError('lesson 6.5')


def count_params(model, embeddings=False):
    """Parameters, without the token and position embeddings unless asked
    (Kaplan et al. found non-embedding parameters give the cleaner law)."""
    raise NotImplementedError('lesson 6.5')


@torch.no_grad()
def eval_loss(model, data, block_size, n_batches=10, batch_size=32, seed=1):
    raise NotImplementedError('lesson 6.5')


def size_sweep(widths, train_data, val_data, vocab_size, steps, block_size=32, n_layer=2, seed=0, log=None):
    """Train one GPT per width (n_embd), all for the same number of steps,
    and return (non-embedding params, validation losses) as arrays."""
    raise NotImplementedError('lesson 6.5')


def flops(n_params, n_tokens):
    """Training compute: about 6 FLOPs per parameter per token (2 forward,
    4 backward)."""
    raise NotImplementedError('lesson 6.6')


def chinchilla_loss(N, D, E, A, B, alpha, beta):
    """L(N, D) = E + A / N^α + B / D^β (Hoffmann et al., 2022)."""
    raise NotImplementedError('lesson 6.6')


def optimal_allocation(C, A, B, alpha, beta):
    """The N and D that minimize chinchilla_loss subject to 6·N·D = C:
    N* = G · (C/6)^(β/(α+β)), D* = (C/6) / N*, with
    G = (α·A / (β·B))^(1/(α+β))."""
    raise NotImplementedError('lesson 6.6')


def isoflop_minimum(n_params, losses):
    """Chinchilla's second method: for runs that all used the same compute,
    fit a parabola to loss against log N and return the N at its bottom."""
    raise NotImplementedError('lesson 6.6')
