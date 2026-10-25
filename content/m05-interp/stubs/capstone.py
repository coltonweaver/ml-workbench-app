"""Capstone: testing a claim about a head with experiments.

Lesson 5.10."""
import numpy as np
import torch
from m05.hooks import loss_from_logits, run


def ablate_head(layer, head, kind='zero', mean=None):
    """A patch for ``blocks.{layer}.attn.z`` that replaces one head's output
    with zeros, or with ``mean`` (its average output, shape (hs,))."""
    raise NotImplementedError('lesson 5.10')


@torch.no_grad()
def head_means(model, idx):
    """Each head's mean output over the batch and positions:
    {layer: (H, hs)}."""
    raise NotImplementedError('lesson 5.10')


@torch.no_grad()
def ablation_table(model, idx, kind='mean', positions=None):
    """Increase in next-token loss when each head is ablated, as an
    (n_layer, n_head) array. Loss is over predictions at ``positions``
    (a slice of the T - 1 predictions; all of them if None)."""
    raise NotImplementedError('lesson 5.10')


def claim_check(table, layer, head, margin=2.0):
    """Is the claimed head's effect positive and at least ``margin`` times
    the next largest? Returns (holds, its effect, the largest other)."""
    raise NotImplementedError('lesson 5.10')
