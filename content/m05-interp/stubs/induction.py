"""Induction heads: how a transformer copies from earlier in its context.

Lesson 5.4."""
import numpy as np
import torch
from m05.hooks import run_with_cache


def repeated_tokens(batch, half, vocab, rng):
    """Random token sequences repeated once: shape (batch, 2 * half). The
    first half is unpredictable; the second half is predictable only by
    looking back."""
    raise NotImplementedError('lesson 5.4')


def train_on_repeats(model, steps, vocab, max_half, batch_size=32, lr=0.003, seed=0, log=None):
    """Train on repeated random sequences whose length varies from batch to
    batch (4 to ``max_half`` tokens before the repeat), so that no fixed
    position tells the model where to look. Returns the losses."""
    raise NotImplementedError('lesson 5.4')


def per_position_loss(model, idx):
    """Cross-entropy for predicting idx[:, t + 1] from idx[:, :t + 1],
    averaged over the batch: shape (T - 1,)."""
    raise NotImplementedError('lesson 5.4')


def previous_token_score(pattern):
    """Mean attention from each position t ≥ 1 to t - 1. ``pattern`` is
    (B, H, T, T); returns (H,)."""
    raise NotImplementedError('lesson 5.4')


def induction_score(pattern, half):
    """On a sequence repeated after ``half`` tokens: mean attention from each
    position t in the second half to t - half + 1, the token that followed
    the previous copy of the current token. Returns (H,)."""
    raise NotImplementedError('lesson 5.4')


@torch.no_grad()
def head_scores(model, idx, half):
    """(previous-token scores, induction scores), each (n_layer, n_head)."""
    raise NotImplementedError('lesson 5.4')
