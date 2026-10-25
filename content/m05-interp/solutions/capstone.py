"""Capstone: testing a claim about a head with experiments."""

import numpy as np
import torch

from m05.hooks import loss_from_logits, run


def ablate_head(layer, head, kind="zero", mean=None):
    """A patch for ``blocks.{layer}.attn.z`` that replaces one head's output
    with zeros, or with ``mean`` (its average output, shape (hs,))."""
    def fn(z):
        z = z.clone()
        z[:, :, head] = 0 if kind == "zero" else mean
        return z
    return {f"blocks.{layer}.attn.z": fn}


@torch.no_grad()
def head_means(model, idx):
    """Each head's mean output over the batch and positions:
    {layer: (H, hs)}."""
    cache = {}
    run(model, torch.as_tensor(idx, dtype=torch.long), cache=cache)
    return {i: cache[f"blocks.{i}.attn.z"].mean(dim=(0, 1)) for i in range(len(model.blocks))}


@torch.no_grad()
def ablation_table(model, idx, kind="mean", positions=None):
    """Increase in next-token loss when each head is ablated, as an
    (n_layer, n_head) array. Loss is over predictions at ``positions``
    (a slice of the T - 1 predictions; all of them if None)."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    x, y = idx[:, :-1], idx[:, 1:]
    sl = positions if positions is not None else slice(None)

    def loss(patch=None):
        return loss_from_logits(run(model, x, patch=patch)[:, sl], y[:, sl]).item()

    base = loss()
    means = head_means(model, x) if kind == "mean" else None
    L, H = len(model.blocks), model.blocks[0].attn.n_head
    out = np.zeros((L, H))
    for layer in range(L):
        for head in range(H):
            m = means[layer][head] if means is not None else None
            out[layer, head] = loss(ablate_head(layer, head, kind, m)) - base
    return out


def claim_check(table, layer, head, margin=2.0):
    """Is the claimed head's effect positive and at least ``margin`` times
    the next largest? Returns (holds, its effect, the largest other)."""
    claimed = float(table[layer, head])
    others = np.delete(table.flatten(), layer * table.shape[1] + head)
    best_other = float(others.max())
    return claimed > 0 and claimed >= margin * max(best_other, 0.0), claimed, best_other
