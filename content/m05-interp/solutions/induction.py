"""Induction heads: how a transformer copies from earlier in its context."""

import numpy as np
import torch

from m05.hooks import run_with_cache


def repeated_tokens(batch, half, vocab, rng):
    """Random token sequences repeated once: shape (batch, 2 * half). The
    first half is unpredictable; the second half is predictable only by
    looking back."""
    first = rng.integers(0, vocab, size=(batch, half))
    return np.concatenate([first, first], axis=1)


def train_on_repeats(model, steps, vocab, max_half, batch_size=32, lr=3e-3, seed=0, log=None):
    """Train on repeated random sequences whose length varies from batch to
    batch (4 to ``max_half`` tokens before the repeat), so that no fixed
    position tells the model where to look. Returns the losses."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        half = int(rng.integers(4, max_half + 1))
        idx = torch.as_tensor(repeated_tokens(batch_size, half, vocab, rng), dtype=torch.long)
        _, loss = model(idx[:, :-1], idx[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log is not None:
            log(step, losses[-1])
    return losses


def per_position_loss(model, idx):
    """Cross-entropy for predicting idx[:, t + 1] from idx[:, :t + 1],
    averaged over the batch: shape (T - 1,)."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    with torch.no_grad():
        logits = model(idx[:, :-1])
    logp = logits.log_softmax(-1)
    nll = -logp.gather(-1, idx[:, 1:, None])[..., 0]
    return nll.mean(0).numpy()


def previous_token_score(pattern):
    """Mean attention from each position t ≥ 1 to t - 1. ``pattern`` is
    (B, H, T, T); returns (H,)."""
    return pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean(dim=(0, -1))


def induction_score(pattern, half):
    """On a sequence repeated after ``half`` tokens: mean attention from each
    position t in the second half to t - half + 1, the token that followed
    the previous copy of the current token. Returns (H,)."""
    return pattern.diagonal(offset=-(half - 1), dim1=-2, dim2=-1)[..., 1:].mean(dim=(0, -1))


@torch.no_grad()
def head_scores(model, idx, half):
    """(previous-token scores, induction scores), each (n_layer, n_head)."""
    _, cache = run_with_cache(model, torch.as_tensor(idx, dtype=torch.long))
    n = len(model.blocks)
    prev = torch.stack([previous_token_score(cache[f"blocks.{i}.attn.pattern"]) for i in range(n)])
    ind = torch.stack([induction_score(cache[f"blocks.{i}.attn.pattern"], half) for i in range(n)])
    return prev.numpy(), ind.numpy()
