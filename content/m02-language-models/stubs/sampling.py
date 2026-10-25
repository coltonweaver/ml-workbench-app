"""Turning a model's logits into text."""

import numpy as np


def sample_next(logits, temperature, rng):
    """Draw one token id from softmax(logits / temperature) with
    ``rng.choice``. temperature == 0 means greedy: the most likely token."""
    raise NotImplementedError


def generate(next_logits, context, temperature, rng, max_new=50, stop=0):
    """Grow ``context`` (a list of ids) one token at a time.

    ``next_logits(context)`` returns the logits for the next token. Stop after
    ``max_new`` tokens or when ``stop`` is drawn (don't append it). Return
    only the new tokens.
    """
    raise NotImplementedError
