---
id: "5.2"
title: The logit lens and direct attribution
concepts: [logit-lens, direct-logit-attribution, layer-norm]
file: m05/lens.py
stub: stubs/lens.py
tests: test_logit_lens.py
predict:
  kind: choice
  prompt: >
    The final layer norm divides the residual stream by its standard
    deviation, which is not a linear operation. How can the logit still be
    split exactly into one share per component?
  options:
    - it can't; the split is only approximate
    - freeze the divisor at its value on this input; then the rest is linear
    - drop the layer norm entirely
    - split the logits after the softmax instead
  answer: 1
mutants:
  - name: lens skips the last block
    file: m05/lens.py
    replace: ["[cache[f\"blocks.{i}.resid_post\"] for i in range(n)]", "[cache[f\"blocks.{i}.resid_pre\"] for i in range(n)]"]
  - name: forgets ln's centering
    file: m05/lens.py
    replace: ["((c - c.mean()) / sigma * ln.weight) @ w_u", "(c / sigma * ln.weight) @ w_u"]
  - name: leaves out the MLPs
    file: m05/lens.py
    replace: ["        out.append((f\"mlp {i}\", cache[f\"blocks.{i}.mlp_out\"]))\n", ""]
warmups:
  - kind: choice
    prompt: "What does the logit lens do?"
    options: ["Applies the final layer norm and unembedding to an intermediate residual stream", "Trains a linear probe on each layer", "Removes layers one at a time"]
    answer: 0
    explain: "It reads off what the model would predict if it stopped at that layer. No training, which is its appeal and its weakness."
---

## Hook

The unembedding turns the final residual stream into logits. Nothing stops
you from applying it earlier: after block 0, what would the model predict if
it stopped there? Doing that at every layer is the **logit lens**
(nostalgebraist, 2020). It shows a prediction forming layer by layer.

Asking the reverse question, how much did each component push toward the
token the model chose, is **direct logit attribution** (DLA).

## Build

In `m05/lens.py`:

1. `logit_lens(model, cache)`: `lm_head(ln_f(stream))` for the stream after
   the embedding and after each block: (n_layer + 1, B, T, V).
2. `components(model, cache)`: `[("embed", …), ("attn 0", …), ("mlp 0", …), …]`,
   which sum to the final stream.
3. `direct_logit_attribution(model, cache, pos, token, batch=0)`: each
   component's share of the logit, using the divisor `cache["ln_final.scale"]`.
   Return the names, the shares, and the constant from ln_f's bias.

```python
from workbench import show_attribution, show_matrix
logits, cache = run_with_cache(m, idx)
probs = logit_lens(m, cache)[:, 0].softmax(-1)        # (layers, T, V)
nxt = idx[0, 1:].expand(probs.shape[0], -1)[..., None]
p_next = probs[:, :-1].gather(-1, nxt)[..., 0]        # (layers, T - 1)
show_matrix(p_next.detach().numpy(), title="P(actual next token) by layer")
# one attribution per position, for the token that actually came next
vals = [direct_logit_attribution(m, cache, t, int(idx[0, t + 1]))[1] for t in range(idx.shape[1] - 1)]
show_attribution([decode([i]) for i in idx[0, :-1].tolist()],
                 [n for n, _ in components(m, cache)], torch.stack(vals, 1).detach().numpy())
```

## Explain

ln_f computes (x − mean(x)) / σ · γ + β. Centering and scaling by γ are
linear. Only σ depends on x as a whole, and it's one number per position.
**Freeze σ at the value it had on this input** and the whole map from stream
to logit is linear, so the logit is exactly Σ_c ((c − mean(c)) / σ · γ) ·
W_U[token] plus β · W_U[token]. The tests check that it adds up to the real
logit.

The split is exact but "direct" is doing a lot of work: attn 0's share
counts only what it wrote that went straight to the output, not what it
contributed by changing what later layers did. An early component can look
unimportant by DLA and be essential. The attribution graph in 5.9 counts
the indirect paths too.

The logit lens works well on GPT-2 sized models and poorly on some others,
whose middle layers use a basis the unembedding doesn't read. The tuned
lens (Belrose et al., 2023) trains a small linear map per layer to fix that.

## Stretch

On your Shakespeare model, find a position where the final prediction is
confident and correct. At which layer does the logit lens first rank that
token top 1? Is it the same for spaces and for letters in the middle of a
word?
