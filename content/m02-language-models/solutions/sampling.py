"""Turning a model's logits into text."""

import numpy as np


def sample_next(logits, temperature, rng):
    """Draw one token id from softmax(logits / temperature).

    temperature == 0 means greedy: always the most likely token.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature == 0:
        return int(np.argmax(logits))
    z = logits / temperature
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate(next_logits, context, temperature, rng, max_new=50, stop=0):
    """Grow ``context`` (a list of ids) one token at a time.

    ``next_logits(context)`` returns the logits for the next token. Stops after
    ``max_new`` tokens or when ``stop`` is drawn (it isn't appended). Returns
    the new tokens only.
    """
    context = list(context)
    out = []
    for _ in range(max_new):
        i = sample_next(next_logits(context), temperature, rng)
        if i == stop:
            break
        out.append(i)
        context.append(i)
    return out
