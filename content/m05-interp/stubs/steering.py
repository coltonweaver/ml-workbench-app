"""Steering: add a direction to the residual stream and watch behavior move.

Lesson 5.8."""
import numpy as np
import torch
from m05.hooks import run, run_with_cache


@torch.no_grad()
def mean_resid(model, idx, layer, pos=-1):
    """The residual stream entering block ``layer`` at ``pos``, averaged over
    the batch: (C,)."""
    raise NotImplementedError('lesson 5.8')


def steering_vector(model, pos_idx, neg_idx, layer, pos=-1):
    """Mean residual on the positive prompts minus the mean on the negative
    ones, at ``layer`` and ``pos``."""
    raise NotImplementedError('lesson 5.8')


def add_vector(vec, scale):
    """A patch function adding ``scale * vec`` at every position."""
    raise NotImplementedError('lesson 5.8')


@torch.no_grad()
def steered_logits(model, idx, layer, vec, scale):
    raise NotImplementedError('lesson 5.8')


@torch.no_grad()
def generate(model, idx, max_new, layer=None, vec=None, scale=0.0, temperature=0.0, seed=0):
    """Extend idx (B, T) by ``max_new`` tokens, greedily when temperature is
    0, else by sampling. With ``vec``, steer at ``layer`` on every step.
    Crops the context to the model's block_size."""
    raise NotImplementedError('lesson 5.8')
