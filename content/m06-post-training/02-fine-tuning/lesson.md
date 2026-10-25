---
id: "6.2"
title: Fine-tuning to follow instructions
concepts: [supervised-fine-tuning, exact-match, greedy-decoding]
file: m06/sft.py
stub: stubs/sft.py
tests: test_fine_tuning.py
predict:
  kind: choice
  prompt: >
    You fine-tune a small GPT to reverse names ("emma" → "amme"), training
    only on names in the training set. On names it has never seen, how will
    it do?
  options:
    - near 0%: it can only memorize
    - well above chance, often near perfect: it learns the procedure
    - exactly as well as on the training names
    - it will refuse to stop generating
  answer: 1
mutants:
  - name: never stops at END
    file: m06/sft.py
    replace: ["        if tok.itos[nxt] == END:\n            break\n", ""]
  - name: forgets the separator
    file: m06/sft.py
    replace: ["idx = torch.tensor([tok.encode(prompt + SEP)])", "idx = torch.tensor([tok.encode(prompt)])"]
  - name: no optimizer step
    file: m06/sft.py
    replace: ["        loss.backward()\n        opt.step()\n        losses.append(loss.item())\n        if log is not None:\n            log(step, losses[-1])\n    return losses", "        loss.backward()\n        losses.append(loss.item())\n        if log is not None:\n            log(step, losses[-1])\n    return losses"]
---

## Hook

With the format from 6.1, fine-tuning is ordinary training on a new
dataset. The question is what the model learns: the answers, or the task?
A held-out set tells you.

## Build

In `m06/sft.py`:

1. `sft(model, tok, pairs, steps, batch_size, lr, seed, log)`: encode every
   pair, then AdamW on random batches with `masked_cross_entropy`.
2. `complete(model, tok, prompt, max_new)`: greedy decoding after
   `prompt + SEP`, stopping at END.
3. `exact_match(model, tok, pairs)`.

```python
from m03.torch_gpt import TorchGPT
from workbench import show_training
from workbench.data import names
ns = [n for n in names() if len(n) <= 6]
np.random.default_rng(0).shuffle(ns)
train = [(n, n[::-1]) for n in ns[:3000]]
held_out = [(n, n[::-1]) for n in ns[3000:3200]]
tok = CharTokenizer("abcdefghijklmnopqrstuvwxyz")
m = TorchGPT(len(tok), 16, n_embd=64, n_head=4, n_layer=2)
dash = show_training(title="SFT: reverse")
sft(m, tok, train, 800, lr=2e-3, log=lambda s, l: dash.log(s, l))
exact_match(m, tok, held_out), complete(m, tok, "colton")
```

## Explain

A 2-layer GPT reaches **near-perfect accuracy on unseen names**: it learns
the procedure (attend to the prompt position that mirrors the current
output position) rather than a lookup table. Exact match on held-out data
is what tells you so; training loss can't distinguish memorizing from
learning.

Real SFT starts from a pretrained model and uses far less data than
pretraining, because the model already knows the language; fine-tuning
mostly teaches the format and which of its existing behaviors to use. The
LIMA paper (Zhou et al., 2023) got a usable assistant from 1,000 examples.
The flip side is that SFT copies the demonstrations, flaws included: the
model can only be as good as the responses it imitates. Preference tuning,
next, can go beyond that.

## Stretch

Train on names of length ≤ 5 only, then test on length 6. Does the model
generalize to longer inputs? Where does it break, and what does its
attention look like when it does?
