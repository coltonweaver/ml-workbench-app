"""Activation patching: which activations carry the difference between two
inputs?"""

import numpy as np
import torch

from m05.hooks import run, run_with_cache


def logit_diff(logits, good, bad, pos=-1):
    """logit[good] - logit[bad] at ``pos``, averaged over the batch."""
    last = logits[:, pos]
    return (last[:, good] - last[:, bad]).mean()


def patch_position(name, pos, source):
    """A patch function that replaces ``name``'s value at ``pos`` with the
    value it had in ``source`` (a cache from another run)."""

    def fn(t):
        t = t.clone()
        t[:, pos] = source[name][:, pos]
        return t

    return fn


@torch.no_grad()
def patching_sweep(model, clean, corrupt, metric, names):
    """For each hook point in ``names`` and each position, run the corrupted
    input with that one activation taken from the clean run, and measure
    how much of the clean behavior comes back:

        (metric(patched) - metric(corrupt)) / (metric(clean) - metric(corrupt))

    1 means that single activation restores everything, 0 means nothing.
    ``metric`` maps logits to a scalar. Returns a (len(names), T) array.
    """
    clean_logits, clean_cache = run_with_cache(model, clean)
    m_clean = float(metric(clean_logits))
    m_corrupt = float(metric(run(model, corrupt)))
    T = clean.shape[1]
    out = np.zeros((len(names), T))
    for r, name in enumerate(names):
        for pos in range(T):
            logits = run(model, corrupt, patch={name: patch_position(name, pos, clean_cache)})
            out[r, pos] = (float(metric(logits)) - m_corrupt) / (m_clean - m_corrupt)
    return out
