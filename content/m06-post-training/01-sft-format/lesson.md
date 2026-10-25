---
id: "6.1"
title: A format to follow
concepts: [supervised-fine-tuning, chat-template, loss-masking]
file: m06/sft.py
stub: stubs/sft.py
tests: test_sft_format.py
predict:
  kind: choice
  prompt: >
    You fine-tune on prompt/response pairs. If you compute the loss on every
    token, prompt included, what does the model spend part of its capacity
    learning?
  options:
    - nothing different; the prompt tokens have zero gradient
    - to predict the prompts, which it will never have to write
    - to stop at the end of the response
    - to ignore the separator
  answer: 1
mutants:
  - name: trains on the prompt too
    file: m06/sft.py
    replace: ["mask = [0] * (n_prompt - 1) + [1] * (len(ids) - n_prompt)", "mask = [1] * (len(ids) - 1)"]
  - name: forgets the END token
    file: m06/sft.py
    replace: ["ids = tok.encode(prompt + SEP + response + END)", "ids = tok.encode(prompt + SEP + response)"]
  - name: averages over all positions
    file: m06/sft.py
    replace: ["return (nll * mask).sum() / mask.sum().clamp_min(1)", "return (nll * mask).mean()"]
---

## Hook

A pretrained model continues text. Ask it a question and it may continue
with more questions. Post-training turns that into a model that answers in
a format: a prompt, a separator, a response, a stop. The first step is
**supervised fine-tuning** (SFT): train on examples of the format, the same
next-token loss as pretraining, but only on the parts the model should
learn to write.

Real chat models use special tokens (`<|user|>`, `<|assistant|>`); here,
with characters, `>` ends the prompt, `.` ends the response and `_` pads.

## Build

In `m06/sft.py`:

1. `CharTokenizer(chars)`: adds `SEP`, `END` and `PAD` to the characters.
2. `encode_example(tok, prompt, response, block_size)`: the text
   `prompt > response .`, padded to block_size + 1 tokens, split into
   inputs x and targets y, and a mask that is 1 exactly where y is a
   response token or END.
3. `masked_cross_entropy(logits, targets, mask)`: the mean over masked
   positions only.

```python
tok = CharTokenizer("abcdefghijklmnopqrstuvwxyz")
x, y, mask = encode_example(tok, "emma", "amme", block_size=16)
list(zip(tok.decode(x), tok.decode(y), mask.tolist()))
```

## Explain

Positions in the prompt still have gradients; the loss decides whether you
use them. Train on every token and the model also learns **to predict the
prompts**, which costs capacity and, worse, skews it: with short responses
most of the loss comes from the prompt. Masking keeps every gradient on
what the model will actually produce.

END matters as much as any response token. A model that never saw END as a
target doesn't know when to stop, and keeps generating past its answer.

Note the denominator: average over the masked positions, not all of them.
Otherwise a batch with short responses gets a smaller loss and smaller
gradients for no reason.

## Stretch

How do real chat templates handle multi-turn conversations? Write
`encode_chat(tok, turns)` for alternating user/assistant turns, masking
every user turn and training on every assistant turn.
