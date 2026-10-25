---
id: "5.4"
title: Induction heads
concepts: [induction-heads, in-context-learning, circuits]
file: m05/induction.py
stub: stubs/induction.py
tests: test_induction.py
predict:
  kind: number
  prompt: >
    A model sees a sequence of 12 random tokens (vocabulary 32), then the
    same 12 again. For a model that has learned everything learnable, what
    is its average loss (in nats) on predicting the tokens of the first
    half?
  answer: 3.47
  tolerance: 0.1
mutants:
  - name: induction score looks one too far
    file: m05/induction.py
    replace: ["offset=-(half - 1)", "offset=-half"]
  - name: previous-token score on the diagonal
    file: m05/induction.py
    replace: ["pattern.diagonal(offset=-1, dim1=-2, dim2=-1)", "pattern.diagonal(offset=0, dim1=-2, dim2=-1)"]
  - name: trains on a fixed length
    file: m05/induction.py
    replace: ["half = int(rng.integers(4, max_half + 1))", "half = max_half"]
warmups:
  - kind: choice
    prompt: "An induction head sees “… A B … A”. What does it push the model to predict next?"
    options: ["A", "B", "The token before the first A"]
    answer: 1
    explain: "It finds the earlier A, looks at what followed it, and copies B. That's the mechanism behind a lot of in-context learning."
---

## Hook

Language models get better at predicting a document the further into it
they are. Much of that comes from one simple trick: if the current token
appeared earlier, predict whatever came after it last time. "Mr Dursley …
Mr" → "Dursley". Olsson et al., "In-context Learning and Induction Heads"
(2022), found the circuit that does this, in two layers, in every
transformer they studied, and saw it form in a sudden phase change during
training.

You'll train a tiny model on a task that only this trick can solve, then
find the heads.

## Build

In `m05/induction.py`:

1. `repeated_tokens(batch, half, vocab, rng)`: random tokens, repeated once.
2. `train_on_repeats(model, steps, vocab, max_half, …)`: AdamW, with the
   length of the repeated part drawn fresh each batch from 4 to `max_half`.
3. `per_position_loss(model, idx)`.
4. `previous_token_score(pattern)` and `induction_score(pattern, half)`
   from an attention pattern (B, H, T, T): the average weight on the
   previous token, and on the token after the current one's earlier copy.
   `torch.diagonal` with an offset does each in one line.
5. `head_scores(model, idx, half)`: both, for every layer and head.

```python
from m03.torch_gpt import TorchGPT
from workbench import show_training, show_attention, show_matrix
m = TorchGPT(32, 24, n_embd=32, n_head=2, n_layer=2)
dash = show_training(title="repeats")
train_on_repeats(m, 1000, vocab=32, max_half=12, log=lambda s, l: dash.log(s, l))
data = repeated_tokens(64, 12, 32, np.random.default_rng(1))
show_matrix(per_position_loss(m, data)[None], title="loss by position")
prev, ind = head_scores(m, data, 12)
```

## Explain

The first half is uniformly random, so the best possible loss there is
ln 32 ≈ **3.47** nats. In the second half every token is determined by the
first, and a working model gets close to 0. The loss curve over positions
falls off a cliff at the repeat.

The circuit takes two heads in different layers. A **previous-token head**
in layer 0 copies each token's identity into the next position, so the
stream at position t knows "the token before me was A". An **induction
head** in layer 1 then queries for positions whose previous token matches
the current token, finds the position after A's earlier occurrence, and
copies what's there. Neither head can do it alone, which is why it takes
two layers (Elhage et al., 2021, show a one-layer model can't).

Why vary the length? With a fixed repeat offset, position alone tells the
model where to look ("attend 11 back"), and it learns that shortcut
instead. Your mutant tests check that the varied version finds real
induction heads.

## Stretch

Plot the induction score every 50 steps during training, alongside the
loss. Do you see the phase change: a plateau, then a sudden drop in loss
exactly when the score jumps? Then ablate the layer 0 previous-token head
(lesson 5.10 has the tool) and watch the induction head stop working.
