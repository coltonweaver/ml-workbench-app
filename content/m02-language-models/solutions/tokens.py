"""Characters as tokens."""

BOUNDARY = "."


def build_vocab(words):
    """Map every character in ``words`` to an integer.

    The boundary token "." gets 0; the characters follow in sorted order.
    Returns ``(stoi, itos)``: a dict from character to id and a list from id
    to character.
    """
    chars = sorted(set("".join(words)) - {BOUNDARY})
    itos = [BOUNDARY] + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def encode(text, stoi):
    return [stoi[ch] for ch in text]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def bigrams(word):
    """Consecutive (previous, next) character pairs of ``word``, with the
    boundary marking where it starts and ends: "ab" gives
    [(".", "a"), ("a", "b"), ("b", ".")]."""
    chars = [BOUNDARY] + list(word) + [BOUNDARY]
    return list(zip(chars, chars[1:]))
