"""Speculative decoding: a small draft model guesses, the big model checks
several guesses in one pass (Leviathan et al., 2023; Chen et al., 2023).

Lesson 7.6."""
import numpy as np
import torch


def residual(p, q):
    """norm(max(0, p − q)): where to resample after a rejection."""
    raise NotImplementedError('lesson 7.6')


def verify(p_rows, q_rows, draft, rng):
    """Check ``draft`` (k tokens sampled from the draft distributions
    q_rows[0..k-1]) against the target's distributions p_rows[0..k] (one
    more row: the position after the last draft token).

    Accept draft token i with probability min(1, p_i(x)/q_i(x)). At the
    first rejection, sample a replacement from residual(p_i, q_i) and stop.
    If all k are accepted, sample a bonus token from p_rows[k]. Returns the
    list of tokens produced (1 to k + 1 of them)."""
    raise NotImplementedError('lesson 7.6')


def first_token_distribution(p, q):
    """The exact distribution of the first token speculative decoding emits
    with one draft token: accepted draft (q(x)·min(1, p(x)/q(x))) plus the
    rejection mass times the residual. It equals p."""
    raise NotImplementedError('lesson 7.6')


def expected_tokens(alpha, k):
    """Tokens per target pass when each draft token is accepted with
    probability alpha: (1 − α^(k+1)) / (1 − α)."""
    raise NotImplementedError('lesson 7.6')


@torch.no_grad()
def speculative_greedy(target, draft, idx, n, k=4):
    """Greedy speculative decoding: the draft proposes k tokens greedily,
    the target scores them all in one pass, and we keep the longest prefix
    the target agrees with plus the target's own next token. Produces
    exactly the target's greedy output. Returns (tokens, target passes)."""
    raise NotImplementedError('lesson 7.6')
