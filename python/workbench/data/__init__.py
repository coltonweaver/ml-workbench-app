"""Datasets bundled with the course.

- ``names()``: 32,033 first names, one per line, from the US Social Security
  Administration's public baby-name data (as collected for karpathy/makemore).
- ``shakespeare()``: about 1 MB of Shakespeare's plays ("tiny Shakespeare",
  from karpathy/char-rnn). Public domain.
- ``corpus(path)``: any text file of your own, for training your GPT on
  something else.
"""

from __future__ import annotations

import os

_DIR = os.path.dirname(os.path.abspath(__file__))


def names() -> list[str]:
    with open(os.path.join(_DIR, "names.txt")) as f:
        return f.read().split()


def shakespeare() -> str:
    with open(os.path.join(_DIR, "shakespeare.txt")) as f:
        return f.read()


def corpus(path: str) -> str:
    with open(os.path.expanduser(path)) as f:
        return f.read()
