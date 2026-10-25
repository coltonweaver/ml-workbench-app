"""Scaling laws: measure how loss falls with model size, fit a power law,
and choose model size for a compute budget."""

import numpy as np
import torch

from m03.torch_gpt import TorchGPT, train_torch


def fit_power_law(x, y):
    """Least-squares fit of y = a · x^(−α) on log-log axes. Returns (a, α)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(np.exp(intercept)), float(-slope)


def fit_power_law_offset(x, y, grid=400):
    """Fit y = a · x^(−α) + c, where c is the loss no model size can beat.
    For each c on a grid from 0 to just below min(y), fit a power law to
    y − c and keep the c with the smallest squared error in log space.
    Returns (a, α, c)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    best = None
    for c in np.linspace(0, y.min() * 0.999, grid):
        a, alpha = fit_power_law(x, y - c)
        err = np.sum((np.log(y - c) - np.log(a * x ** -alpha)) ** 2)
        if best is None or err < best[0]:
            best = (err, a, alpha, float(c))
    return best[1], best[2], best[3]


def count_params(model, embeddings=False):
    """Parameters, without the token and position embeddings unless asked
    (Kaplan et al. found non-embedding parameters give the cleaner law)."""
    n = sum(p.numel() for p in model.parameters())
    if not embeddings:
        n -= model.wte.weight.numel() + model.wpe.weight.numel()
    return n


@torch.no_grad()
def eval_loss(model, data, block_size, n_batches=10, batch_size=32, seed=1):
    rng = np.random.default_rng(seed)
    data = torch.as_tensor(np.asarray(data), dtype=torch.long)
    model.eval()
    losses = []
    for _ in range(n_batches):
        s = rng.integers(0, len(data) - block_size - 1, size=batch_size)
        X = torch.stack([data[i:i + block_size] for i in s])
        Y = torch.stack([data[i + 1:i + block_size + 1] for i in s])
        losses.append(model(X, Y)[1].item())
    return float(np.mean(losses))


def size_sweep(widths, train_data, val_data, vocab_size, steps, block_size=32, n_layer=2, seed=0, log=None):
    """Train one GPT per width (n_embd), all for the same number of steps,
    and return (non-embedding params, validation losses) as arrays."""
    params, losses = [], []
    for w in widths:
        torch.manual_seed(seed)
        model = TorchGPT(vocab_size, block_size, n_embd=w, n_head=max(1, w // 16), n_layer=n_layer)
        train_torch(model, train_data, steps, seed=seed)
        params.append(count_params(model))
        losses.append(eval_loss(model, val_data, block_size))
        if log is not None:
            log(w, params[-1], losses[-1])
    return np.array(params), np.array(losses)


def flops(n_params, n_tokens):
    """Training compute: about 6 FLOPs per parameter per token (2 forward,
    4 backward)."""
    return 6.0 * n_params * n_tokens


def chinchilla_loss(N, D, E, A, B, alpha, beta):
    """L(N, D) = E + A / N^α + B / D^β (Hoffmann et al., 2022)."""
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def optimal_allocation(C, A, B, alpha, beta):
    """The N and D that minimize chinchilla_loss subject to 6·N·D = C:
    N* = G · (C/6)^(β/(α+β)), D* = (C/6) / N*, with
    G = (α·A / (β·B))^(1/(α+β))."""
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    N = G * (C / 6) ** (beta / (alpha + beta))
    return N, (C / 6) / N


def isoflop_minimum(n_params, losses):
    """Chinchilla's second method: for runs that all used the same compute,
    fit a parabola to loss against log N and return the N at its bottom."""
    a, b, _ = np.polyfit(np.log(n_params), losses, 2)
    return float(np.exp(-b / (2 * a)))
