"""Speculative decoding: a small draft model guesses, the big model checks
several guesses in one pass (Leviathan et al., 2023; Chen et al., 2023)."""

import numpy as np
import torch


def residual(p, q):
    """norm(max(0, p − q)): where to resample after a rejection."""
    r = np.maximum(np.asarray(p, float) - np.asarray(q, float), 0)
    s = r.sum()
    return r / s if s > 0 else np.asarray(p, float)


def verify(p_rows, q_rows, draft, rng):
    """Check ``draft`` (k tokens sampled from the draft distributions
    q_rows[0..k-1]) against the target's distributions p_rows[0..k] (one
    more row: the position after the last draft token).

    Accept draft token i with probability min(1, p_i(x)/q_i(x)). At the
    first rejection, sample a replacement from residual(p_i, q_i) and stop.
    If all k are accepted, sample a bonus token from p_rows[k]. Returns the
    list of tokens produced (1 to k + 1 of them)."""
    out = []
    for i, x in enumerate(draft):
        p, q = p_rows[i], q_rows[i]
        if rng.random() < min(1.0, p[x] / q[x]):
            out.append(int(x))
        else:
            out.append(int(rng.choice(len(p), p=residual(p, q))))
            return out
    out.append(int(rng.choice(len(p_rows[-1]), p=p_rows[-1])))
    return out


def first_token_distribution(p, q):
    """The exact distribution of the first token speculative decoding emits
    with one draft token: accepted draft (q(x)·min(1, p(x)/q(x))) plus the
    rejection mass times the residual. It equals p."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    accepted = np.minimum(q, p)
    return accepted + (1 - accepted.sum()) * residual(p, q)


def expected_tokens(alpha, k):
    """Tokens per target pass when each draft token is accepted with
    probability alpha: (1 − α^(k+1)) / (1 − α)."""
    return k + 1 if alpha == 1 else (1 - alpha ** (k + 1)) / (1 - alpha)


@torch.no_grad()
def speculative_greedy(target, draft, idx, n, k=4):
    """Greedy speculative decoding: the draft proposes k tokens greedily,
    the target scores them all in one pass, and we keep the longest prefix
    the target agrees with plus the target's own next token. Produces
    exactly the target's greedy output. Returns (tokens, target passes)."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    goal = idx.shape[1] + n
    passes = 0
    while idx.shape[1] < goal:
        proposal = idx
        for _ in range(k):
            nxt = draft(proposal[:, -draft.block_size:])[:, -1].argmax(-1, keepdim=True)
            proposal = torch.cat([proposal, nxt], 1)
        guesses = proposal[0, idx.shape[1]:]
        logits = target(proposal[:, -target.block_size:])[0]
        passes += 1
        # the target's choice after each prefix: the last k + 1 positions
        choices = logits[-(k + 1):].argmax(-1)
        n_ok = 0
        while n_ok < k and guesses[n_ok] == choices[n_ok]:
            n_ok += 1
        new = torch.cat([guesses[:n_ok], choices[n_ok:n_ok + 1]])
        idx = torch.cat([idx, new[None]], 1)
    return idx[:, :goal], passes
