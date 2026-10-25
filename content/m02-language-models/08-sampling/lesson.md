---
id: "2.8"
title: Turning the dial
concepts: [sampling, temperature, greedy-decoding]
file: m02/sampling.py
stub: stubs/sampling.py
tests: test_sampling.py
predict:
  kind: choice
  prompt: >
    To sample, divide the logits by a "temperature" T before the softmax.
    As T gets close to 0, what does sampling turn into?
  options:
    - uniform random choice
    - always picking the most likely token
    - the same as T = 1
    - it fails with a division error
  answer: 1
mutants:
  - name: multiplies by the temperature
    file: m02/sampling.py
    replace: ["z = logits / temperature", "z = logits * temperature"]
  - name: generate doesn't feed tokens back
    file: m02/sampling.py
    replace: ["        context.append(i)\n", ""]
---

## Hook

A model gives a distribution over the next token. How you pick from it
changes what comes out, and it's a dial every LLM API exposes. Dividing the
logits by T before the softmax makes differences between scores bigger
(T < 1) or smaller (T > 1).

Imagine the logits [2, 1, 0]. At T = 1 the top token gets 67%. Now picture T
shrinking toward 0.

## Build

In `m02/sampling.py`:

1. `sample_next(logits, temperature, rng)`: softmax of `logits / T` (shift
   by the max), then `rng.choice`. Temperature 0 is greedy: `argmax`.
2. `generate(next_logits, context, temperature, rng, max_new, stop)`: call
   `next_logits(context)`, sample, append to the context, repeat. Stop at
   `stop` or after `max_new` tokens. Return only the new tokens.

Use it with the neural bigram's `W`:

```python
rng = np.random.default_rng(0)
nxt = lambda ctx: W.data[ctx[-1]]
for T in [0, 0.5, 1.0, 2.0]:
    print(T, [decode(generate(nxt, [0], T, rng), itos) for _ in range(6)])
```

## Explain

**Temperature** comes from statistical physics (the Boltzmann distribution
has exactly this form). As T → 0 the largest logit dominates and sampling
becomes **greedy decoding**: always the most likely token. That's
deterministic, and for a bigram model it gets stuck repeating the same few
names. As T → ∞ every token becomes equally likely.

T = 1 samples the model's actual distribution, which is the only setting
where samples reflect what the model learned. Lower temperatures trade
diversity for typicality. Other samplers cut off the unlikely tail instead:
**top-k** keeps the k best tokens, **top-p** (nucleus sampling, Holtzman et
al., 2019) keeps the smallest set with probability mass p.

`generate` is the autoregressive loop. With a GPT behind `next_logits` it is
how every chat model produces text, one token at a time, each one fed back
in. Its cost is the subject of module 7.

## Stretch

Add `top_k` and `top_p` options to `sample_next`. Generate 1,000 names at
each setting and measure what fraction already exist in the dataset.
