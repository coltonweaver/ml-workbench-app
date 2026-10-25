---
id: "7.6"
title: Speculative decoding
concepts: [speculative-decoding, rejection-sampling, draft-model]
file: m07/speculative.py
stub: stubs/speculative.py
tests: test_speculative.py
predict:
  kind: number
  prompt: >
    A draft model proposes 4 tokens at a time; each is accepted with
    probability 0.8 (and everything after a rejection is discarded, but the
    target always contributes one token of its own). How many tokens does
    each pass of the big model produce on average? Two decimals.
  answer: 3.36
  tolerance: 0.01
mutants:
  - name: accepts every draft
    file: m07/speculative.py
    replace: ["if rng.random() < min(1.0, p[x] / q[x]):", "if True:"]
  - name: resamples from the target, not the residual
    file: m07/speculative.py
    replace: ["out.append(int(rng.choice(len(p), p=residual(p, q))))", "out.append(int(rng.choice(len(p), p=p)))"]
  - name: off-by-one target positions
    file: m07/speculative.py
    replace: ["choices = logits[-(k + 1):].argmax(-1)", "choices = logits[-k:].argmax(-1)"]
warmups:
  - kind: number
    prompt: "Speculative decoding with 1 draft token and an acceptance rate of 0.5. Expected tokens per verification step?"
    answer: 1.5
    tolerance: 0.02
    explain: "(1 − α^(γ+1)) / (1 − α) = 0.75 / 0.5 = 1.5: the target model's own token always comes out, plus the draft half the time."
---

## Hook

A decoding step reads all the weights to produce one token, and the GPU
could have checked many tokens in the same pass (7.1). Speculative
decoding uses that slack: a small, fast draft model guesses k tokens, and
the big model scores all k positions in one pass. Guesses it agrees with
are kept, so one expensive pass yields several tokens, and the output
distribution is exactly the big model's.

## Build

In `m07/speculative.py`:

1. `residual(p, q)`: normalize max(0, p − q).
2. `verify(p_rows, q_rows, draft, rng)` (Leviathan et al., 2023): accept
   draft token x with probability min(1, p(x)/q(x)); on the first
   rejection sample from the residual and stop; if all are accepted, add a
   bonus token from the last target row.
3. `first_token_distribution(p, q)`: the exact distribution of the first
   emitted token. It must equal p.
4. `expected_tokens(alpha, k)`.
5. `speculative_greedy(target, draft, idx, n, k)`: the greedy version with
   real models, counting target passes.

```python
small = TorchGPT(len(itos), 64, n_embd=32, n_head=2, n_layer=1)
train_torch(small, data, 1500)                           # a draft for your Shakespeare model
for k in [1, 2, 4, 8]:
    out, passes = speculative_greedy(m, small, prompt, 200, k)
    print(k, 200 / passes)
```

## Explain

The expected count is 1 + α + α² + … + α^k = (1 − α^(k+1)) / (1 − α); for
α = 0.8, k = 4 that's **3.36** tokens per pass of the big model.

Why is the output exactly p? The chance of emitting x through acceptance
is q(x)·min(1, p(x)/q(x)) = min(p(x), q(x)). The rejection probability is
1 − Σ min(p, q), and the residual gives x the rest, p(x) − min(p(x),
q(x)). Add them: p(x). The test checks this exactly and by sampling.
Speculation changes the speed, never the samples.

Speed depends on α (how often the draft agrees) and the draft's cost.
Variants drop the separate draft: Medusa adds extra heads to the big model
to guess several tokens ahead; EAGLE drafts from its hidden states; and
n-gram lookup drafts by copying from the prompt, which works remarkably
well for code edits.

## Stretch

Measure α for your draft on Shakespeare. Then plot tokens per target pass
against k and compare with expected_tokens(α, k). Why might the real curve
fall below the formula?
