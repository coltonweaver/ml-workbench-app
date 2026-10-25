---
id: "2.1"
title: Text as numbers
concepts: [tokenization, vocabulary]
file: m02/tokens.py
stub: stubs/tokens.py
tests: test_tokens.py
predict:
  kind: number
  prompt: >
    A model will learn which character tends to follow which. To learn when
    a name starts and ends, we wrap each name in a boundary marker ".", so
    "emma" is read as ".emma.". How many (previous, next) pairs of
    characters does "emma" give it to learn from?
  answer: 5
  tolerance: 0
mutants:
  - name: forgets the closing boundary
    file: m02/tokens.py
    replace: ["[BOUNDARY] + list(word) + [BOUNDARY]", "[BOUNDARY] + list(word)"]
  - name: boundary isn't token 0
    file: m02/tokens.py
    replace: ["itos = [BOUNDARY] + chars", "itos = chars + [BOUNDARY]"]
---

## Hook

Module 1's engine works on numbers. A language model works on text. Before
anything can be learned, text has to become numbers, and the model's output
has to turn back into text.

The first model in this module is tiny: it reads one character and guesses
the next. You'll train it on 32,000 real first names:

```python
from workbench.data import names
words = names()
len(words), words[:5]      # (32033, ['emma', 'olivia', 'ava', 'isabella', 'sophia'])
```

A name has a beginning and an end, and "which letters start names" is
something worth learning. So each name gets a boundary marker on both sides.

## Build

In `m02/tokens.py`:

1. `build_vocab(words)`: every distinct character gets an id. The boundary
   `"."` is always 0, then the characters in sorted order. Return `stoi`
   (char to id) and `itos` (id to char).
2. `encode(text, stoi)` and `decode(ids, itos)`.
3. `bigrams(word)`: the (previous, next) pairs, with the boundary on both
   ends.

```python
stoi, itos = build_vocab(words)
len(itos)                      # 27: '.' and a-z
encode("emma", stoi)           # [5, 13, 13, 1]
bigrams("emma")
```

## Explain

You've built a **tokenizer** and its **vocabulary**. Each id is a **token**.
Here a token is one character; production models use subword tokens learned
with byte-pair encoding (Sennrich et al., 2016, from a 1994 compression
algorithm by Gage), so common words are one token and rare words are split
into pieces. GPT-2's vocabulary has 50,257 tokens. Everything you build in
this module works the same with either.

The pairs are **bigrams**: two consecutive tokens. "emma" gives five: `.e`,
`em`, `mm`, `ma`, `a.`. The boundary tokens let one table hold "how names
start" (the row for `.`) and "how names end" (the column for `.`).

## Stretch

Count the bigrams in the whole dataset. What fraction of all 27 × 27 possible
pairs never occurs even once? What will a model that only counts do when it
meets one of them?
