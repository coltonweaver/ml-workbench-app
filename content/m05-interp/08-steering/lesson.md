---
id: "5.8"
title: Steering vectors
concepts: [steering, activation-addition, representation-engineering]
file: m05/steering.py
stub: stubs/steering.py
tests: test_steering.py
predict:
  kind: choice
  prompt: >
    Two one-token prompts, "a" and "b". You compute v = (stream at layer 0
    on "a") − (stream at layer 0 on "b"), then run "b" with v added to the
    layer 0 stream. What does the model output?
  options:
    - exactly what it outputs for "a"
    - something halfway between "a" and "b"
    - what it outputs for "b"
    - something unrelated to either
  answer: 0
mutants:
  - name: vector points the wrong way
    file: m05/steering.py
    replace: ["return mean_resid(model, pos_idx, layer, pos) - mean_resid(model, neg_idx, layer, pos)", "return mean_resid(model, neg_idx, layer, pos) - mean_resid(model, pos_idx, layer, pos)"]
  - name: ignores the scale
    file: m05/steering.py
    replace: ["return lambda t: t + scale * vec", "return lambda t: t + vec"]
  - name: never crops the context
    file: m05/steering.py
    replace: ["logits = run(model, idx[:, -model.block_size:], patch=patch)[:, -1]", "logits = run(model, idx, patch=patch)[:, -1]"]
---

## Hook

If a concept is a direction in the residual stream, adding that direction
should make the model act as if the concept were present. Turner et al.,
"Activation Addition" (2023), steered GPT-2 toward topics and moods by
adding the difference between the activations of two contrasting prompts;
Templeton et al. (2024) made Claude 3 Sonnet obsessed with the Golden Gate
Bridge by clamping one SAE feature.

## Build

In `m05/steering.py`:

1. `mean_resid(model, idx, layer, pos=-1)`: the stream entering `layer` at
   `pos`, averaged over the batch.
2. `steering_vector(model, pos_idx, neg_idx, layer, pos=-1)`: positive mean
   minus negative mean.
3. `add_vector(vec, scale)`: a patch function adding `scale · vec`
   everywhere.
4. `steered_logits(...)` and `generate(model, idx, max_new, layer, vec,
   scale, temperature, seed)`: greedy or sampled generation with the patch
   applied on every step; crop the context to `block_size`.

```python
pos = torch.stack([encode("sweet love, my joy"), encode("my love, my heart!")])
neg = torch.stack([encode("a curse upon thee!"), encode("thou villain, die!")])
v = steering_vector(m, pos, neg, layer=1)
for scale in [0, 2, 4, 8]:
    out = generate(m, encode("KING:\n")[None], 200, layer=1, vec=v, scale=scale, temperature=0.8)
    print(scale, decode(out[0]))
v2 = sae.W_dec[:, 37]                    # or steer with one SAE feature's direction
```

## Explain

With one-token prompts, the layer 0 stream is just the token's embedding
(plus the same position embedding for both), so b + (a − b) = a and the
output is **exactly what it outputs for "a"**. The test checks this. With
longer prompts and later layers it's no longer exact, which is the point:
a vector extracted from a few prompts generalizes to other contexts.

Scale matters. Too small, nothing happens; too large, the stream is pushed
far off-distribution and the text degrades into repetition of the
concept's strongest token. The window in between is usually narrow, and
it's widest at middle layers, where representations are most abstract.

Steering is also evidence. If adding a feature's direction produces the
behavior its label predicts, the label is better supported than top
examples alone could show.

## Stretch

Steer with the decoder direction of the SAE feature you labeled most
confidently in 5.7. Does the generated text shift in the way your label
predicts? Find the smallest scale that changes it.
