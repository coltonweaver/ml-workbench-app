---
id: "6.4"
title: DPO in practice
concepts: [dpo, kl-regularization, reward-hacking]
file: m06/dpo.py
stub: stubs/dpo.py
tests: test_dpo_training.py
predict:
  kind: choice
  prompt: >
    A name generator was fine-tuned on baby names; about 20% start with a
    vowel. You run DPO on pairs where the chosen name starts with a vowel
    and the rejected one doesn't. With a small β (weak pull toward the
    reference), what happens to the generated names?
  options:
    - nothing; DPO needs a reward model
    - most start with a vowel, and they still look like names
    - exactly 50% start with a vowel
    - they become a single vowel repeated
  answer: 1
mutants:
  - name: reference that keeps training
    file: m06/dpo.py
    replace: ["    reference = copy.deepcopy(policy).eval()\n", "    reference = policy\n"]
  - name: rejected responses used twice
    file: m06/dpo.py
    replace: ["xc, yc, mc = batch_rows(tok, prompts, [triples[i][1] for i in b], T)", "xc, yc, mc = batch_rows(tok, prompts, [triples[i][2] for i in b], T)"]
---

## Hook

Time to train with it. The task is a stand-in for "be more helpful": a
preference the base model doesn't follow on its own, given only as pairs.

## Build

In `m06/dpo.py`:

1. `dpo_train(policy, tok, triples, steps, beta, …)`: freeze a copy of
   the starting policy as the reference (no gradients), then for each
   batch of (prompt, chosen, rejected) compute the four log-probabilities
   and step on `dpo_loss`. Log the loss and the reward margin.
2. `sample(model, tok, prompt, n, …)`: n sampled responses, cut at END.

```python
from m06.sft import CharTokenizer, sft
from workbench import show_training
tok = CharTokenizer("abcdefghijklmnopqrstuvwxyz")
base = TorchGPT(len(tok), 16, n_embd=64, n_head=4, n_layer=2)
sft(base, tok, [("", n) for n in names() if len(n) <= 8], 1500)   # a name generator
vowel = lambda n: n[0] in "aeiou"
pool = [n for n in names() if len(n) <= 8]
triples = [("", a, b) for a, b in zip([n for n in pool if vowel(n)], [n for n in pool if not vowel(n)])]
before = sample(base, tok, "", 200)
dash = show_training(title="DPO")
dpo_train(base, tok, triples, 300, beta=0.1, log=lambda s, h: dash.log(s, h["loss"], margin=h["margin"]))
after = sample(base, tok, "", 200)
np.mean([vowel(n) for n in before if n]), np.mean([vowel(n) for n in after if n])
```

## Explain

The share of vowel names rises from about 20% to **most of them, while the
outputs still look like names**: DPO moved probability mass toward a
region the base model already produced, rather than inventing new strings.
That's the KL term at work: the implicit reward is measured relative to
the reference, so straying far costs as much as it gains.

The reference must stay frozen. If it is the policy itself (the first
mutant), every reward reads 0, so the margin you monitor says nothing, and
the anchor is gone: each step still pushes chosen up and rejected down,
with nothing measuring how far the policy has moved.

Watch the reward margin: it climbs as the policy separates chosen from
rejected. If it keeps climbing while sample quality drops, the policy is
over-optimizing the preference data, the DPO version of reward hacking.
Larger β, fewer steps, or more diverse pairs help.

## Stretch

Run with β = 0.01 and β = 1.0. Plot the vowel rate and the average log
probability the *reference* assigns to the policy's samples (a proxy for
"still looks like the training distribution"). Where's the trade-off?
