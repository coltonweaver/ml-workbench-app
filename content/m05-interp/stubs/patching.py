"""Activation patching: which activations carry the difference between two
inputs?

Lesson 5.3."""
import numpy as np
import torch
from m05.hooks import run, run_with_cache


def logit_diff(logits, good, bad, pos=-1):
    """logit[good] - logit[bad] at ``pos``, averaged over the batch."""
    raise NotImplementedError('lesson 5.3')


def patch_position(name, pos, source):
    """A patch function that replaces ``name``'s value at ``pos`` with the
    value it had in ``source`` (a cache from another run)."""
    raise NotImplementedError('lesson 5.3')


@torch.no_grad()
def patching_sweep(model, clean, corrupt, metric, names):
    """For each hook point in ``names`` and each position, run the corrupted
    input with that one activation taken from the clean run, and measure
    how much of the clean behavior comes back:

        (metric(patched) - metric(corrupt)) / (metric(clean) - metric(corrupt))

    1 means that single activation restores everything, 0 means nothing.
    ``metric`` maps logits to a scalar. Returns a (len(names), T) array.
    """
    raise NotImplementedError('lesson 5.3')
