---
id: "5.7"
title: Browsing features
concepts: [feature-browser, max-activating-examples, logit-effects]
file: m05/features.py
stub: stubs/features.py
tests: test_feature_browser.py
predict:
  kind: choice
  prompt: >
    A latent fires most strongly on the space after "my" and "thy". Which
    is stronger evidence that it means "a possessive was just used"?
  options:
    - its top 10 examples all look like that
    - its activations across the whole range (not just the top) fit the pattern, and its decoder direction promotes nouns
    - it has a high maximum activation
    - its density is low
  answer: 1
mutants:
  - name: least active first
    file: m05/features.py
    replace: ["torch.argsort(flat, descending=True, stable=True)[:k]", "torch.argsort(flat, descending=False, stable=True)[:k]"]
  - name: window off by one
    file: m05/features.py
    replace: ["start = max(0, pos - window + 1)", "start = max(0, pos - window)"]
  - name: effects ignore ln_f's scale
    file: m05/features.py
    replace: ["effect = model.lm_head.weight @ (d * model.ln_f.weight)", "effect = model.lm_head.weight @ d"]
---

## Hook

An SAE with 512 latents is 512 hypotheses. You need a fast way to look at
each one: where it fires, how strongly, and what it does to the output.
That's a **feature browser**, like Neuronpedia or the dashboards in
"Towards Monosemanticity".

## Build

In `m05/features.py`:

1. `latent_activations(sae, model, idx, layer)`: (B, T, n_latents).
2. `top_contexts(acts, latent, k, window)`: the k most active (row, pos)
   pairs, largest first, each with its activations over the `window`
   tokens ending there. Stop at activations of 0.
3. `logit_effects(sae, model, latent, k)`: push the decoder direction
   through ln_f's γ and the unembedding: the tokens it promotes and
   suppresses most (ignoring later layers).
4. `density(acts, latent)`.

```python
from workbench import feature_card, show_features
acts = latent_activations(sae, m, idx, layer=0)
dec = lambda i: itos[i]
cards = []
for f in range(64):
    ctx = top_contexts(acts, f, k=12, window=24)
    if not ctx:
        continue
    up, upv, down, downv = logit_effects(sae, m, f)
    cards.append(feature_card(f, ctx, idx, dec, up=(up, upv), down=(down, downv),
                              density=density(acts, f)))
show_features(cards, title="layer 0 SAE")
```

Do the same with the MLP neurons (`cache["blocks.0.mlp_out"]` is post-
projection; hook the GELU output with a forward hook on `mlp.fc` instead)
and compare how many you can label.

## Explain

Top examples are cherry-picked by construction: they show what the latent
does at its strongest, not what it does most of the time. A latent can fire
cleanly on possessives at 10.0 and on a dozen unrelated things at 1.0.
Stronger evidence looks **across the activation range** (sample examples
from each band, not only the top) **and at what the latent does**: if its
decoder direction promotes nouns, that is consistent with "a possessive was
just used" from the output side. The browser shows both.

Logit effects here read only the direct path, so they're most meaningful
for late layers; a layer 0 feature mostly acts through later layers.

Expect a mix: some latents are crisp (a specific letter after a space, the
end of a line), some are clusters you can name, some are still
polysemantic, and some are dead. Labeling a feature is a hypothesis, and
the next lessons test hypotheses by intervening.

## Stretch

Add an "activation histogram" to your analysis: for one latent, plot the
distribution of nonzero activations, and show 3 random examples from each
quartile. Does your label survive the bottom quartile?
