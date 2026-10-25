"""Steering: add a direction to the residual stream and watch behavior move."""

import numpy as np
import torch

from m05.hooks import run, run_with_cache


@torch.no_grad()
def mean_resid(model, idx, layer, pos=-1):
    """The residual stream entering block ``layer`` at ``pos``, averaged over
    the batch: (C,)."""
    _, cache = run_with_cache(model, torch.as_tensor(idx, dtype=torch.long))
    return cache[f"blocks.{layer}.resid_pre"][:, pos].mean(0)


def steering_vector(model, pos_idx, neg_idx, layer, pos=-1):
    """Mean residual on the positive prompts minus the mean on the negative
    ones, at ``layer`` and ``pos``."""
    return mean_resid(model, pos_idx, layer, pos) - mean_resid(model, neg_idx, layer, pos)


def add_vector(vec, scale):
    """A patch function adding ``scale * vec`` at every position."""
    return lambda t: t + scale * vec


@torch.no_grad()
def steered_logits(model, idx, layer, vec, scale):
    idx = torch.as_tensor(idx, dtype=torch.long)
    return run(model, idx, patch={f"blocks.{layer}.resid_pre": add_vector(vec, scale)})


@torch.no_grad()
def generate(model, idx, max_new, layer=None, vec=None, scale=0.0, temperature=0.0, seed=0):
    """Extend idx (B, T) by ``max_new`` tokens, greedily when temperature is
    0, else by sampling. With ``vec``, steer at ``layer`` on every step.
    Crops the context to the model's block_size."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.as_tensor(idx, dtype=torch.long)
    patch = {} if vec is None else {f"blocks.{layer}.resid_pre": add_vector(vec, scale)}
    for _ in range(max_new):
        logits = run(model, idx[:, -model.block_size:], patch=patch)[:, -1]
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial((logits / temperature).softmax(-1), 1, generator=gen)
        idx = torch.cat([idx, nxt], dim=1)
    return idx
