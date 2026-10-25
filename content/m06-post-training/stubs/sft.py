"""Supervised fine-tuning: teach a pretrained model a prompt/response format.

Lessons 6.1 and 6.2."""
import numpy as np
import torch
import torch.nn.functional as F
SEP = '>'
END = '.'
PAD = '_'


class CharTokenizer:

    def __init__(self, chars):
        raise NotImplementedError('lesson 6.1')

    def __len__(self):
        raise NotImplementedError('lesson 6.1')

    def encode(self, s):
        raise NotImplementedError('lesson 6.1')

    def decode(self, ids):
        raise NotImplementedError('lesson 6.1')


def encode_example(tok, prompt, response, block_size):
    """One training row for ``prompt + SEP + response + END``, padded with
    PAD to ``block_size + 1`` tokens. Returns (x, y, mask), each of length
    block_size: x the inputs, y the next tokens, mask 1 where y is part of
    the response (including END) and 0 for the prompt, SEP and padding."""
    raise NotImplementedError('lesson 6.1')


def masked_cross_entropy(logits, targets, mask):
    """Mean cross-entropy over the positions where mask is 1."""
    raise NotImplementedError('lesson 6.1')


def sft(model, tok, pairs, steps, batch_size=32, lr=0.001, seed=0, log=None):
    """AdamW on random batches of (prompt, response) ``pairs`` with the
    masked loss. Returns the losses."""
    raise NotImplementedError('lesson 6.2')


@torch.no_grad()
def complete(model, tok, prompt, max_new=20):
    """Greedy response to ``prompt``: everything generated after SEP, up to
    (not including) END."""
    raise NotImplementedError('lesson 6.2')


def exact_match(model, tok, pairs):
    """Fraction of (prompt, response) pairs answered exactly."""
    raise NotImplementedError('lesson 6.2')
