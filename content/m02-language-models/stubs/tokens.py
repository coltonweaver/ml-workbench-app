"""Characters as tokens."""

BOUNDARY = "."


def build_vocab(words):
    """Map every character in ``words`` to an integer.

    The boundary token "." gets 0; the characters follow in sorted order.
    Returns ``(stoi, itos)``: a dict from character to id and a list from id
    to character.
    """
    raise NotImplementedError


def encode(text, stoi):
    """Characters to ids."""
    raise NotImplementedError


def decode(ids, itos):
    """Ids back to a string."""
    raise NotImplementedError


def bigrams(word):
    """Consecutive (previous, next) character pairs of ``word``, with the
    boundary marking where it starts and ends: "ab" gives
    [(".", "a"), ("a", "b"), ("b", ".")]."""
    raise NotImplementedError
