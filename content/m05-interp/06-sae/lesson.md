---
id: "5.6"
title: A sparse autoencoder on your GPT
concepts: [sparse-autoencoder, dictionary-learning, superposition]
file: m05/sae.py
stub: stubs/sae.py
tests: test_sae.py
predict:
  kind: choice
  prompt: >
    You train a sparse autoencoder and raise the L1 penalty coefficient.
    What happens to the average number of active latents per input (L0) and
    to the reconstruction quality?
  options:
    - L0 falls and reconstruction gets worse
    - L0 falls and reconstruction gets better
    - L0 rises and reconstruction gets worse
    - neither changes; L1 only affects the decoder
  answer: 0
mutants:
  - name: never renormalizes the decoder
    file: m05/sae.py
    replace: ["        sae.normalize_decoder()\n        losses", "        losses"]
  - name: encoder ignores b_dec
    file: m05/sae.py
    replace: ["return torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)", "return torch.relu(x @ self.W_enc.T + self.b_enc)"]
  - name: forgets the sparsity penalty
    file: m05/sae.py
    replace: ["return mse + l1 * f.abs().sum(-1).mean(), x_hat, f", "return mse, x_hat, f"]
---

## Hook

Superposition says features are directions, more of them than dimensions,
each active rarely. So write each activation vector as a **sparse**
combination of directions from a large dictionary, and learn the
dictionary. That is a sparse autoencoder (SAE): Bricken et al., "Towards
Monosemanticity" (2023), and Cunningham et al. (2023) showed its latents are
far more interpretable than neurons, and Templeton et al. (2024) scaled it
to Claude 3 Sonnet.

## Build

In `m05/sae.py`:

1. `SAE(d, n_latents)`: f = ReLU(W_enc (x − b_dec) + b_enc),
   x̂ = W_dec f + b_dec. Initialize W_dec with unit-norm columns and W_enc
   as its transpose. `normalize_decoder()` rescales the columns to norm 1.
2. `sae_loss(sae, x, l1)`: squared error plus `l1` × mean ‖f‖₁.
3. `train_sae(sae, X, …)`: initialize b_dec to the data mean, Adam on
   minibatches, renormalize the decoder after each step.
4. `sae_metrics(sae, X)`: L0, fraction of variance explained, fraction of
   dead latents.
5. `residual_activations(model, idx, layer)`.

First check it on data where you know the answer: 32 known directions in 16
dimensions, each input a sparse mix (the tests do this). Then your GPT:

```python
from workbench import show_training
data = encode(text)                                          # from lesson 5.1
idx = torch.stack([data[i:i + 64] for i in range(0, 200_000, 64)])
X = residual_activations(m, idx, layer=0)                    # (tokens, n_embd)
sae = SAE(X.shape[1], 8 * X.shape[1])
dash = show_training(title="SAE")
train_sae(sae, X, steps=4000, l1=0.1, log=lambda s, l: dash.log(s, l))
sae_metrics(sae, X)
```

Aim for L0 around 5–20 and variance explained above 0.8; adjust `l1`.

## Explain

The L1 term charges for every unit of activation, so raising it buys
sparsity at the cost of fidelity: **L0 falls and reconstruction gets
worse**. Every SAE is a point on that trade-off curve, and comparing SAEs
means comparing curves.

Why renormalize the decoder? The L1 penalty is on f. Without a constraint,
the model can halve f and double W_dec, reconstructing the same x̂ for half
the penalty, and repeat forever. Fixing ‖W_dec columns‖ = 1 closes that
loophole and makes a latent's activation mean "how much of this direction".

Dead latents (never fire) waste capacity; real SAE training resamples them
or uses auxiliary losses. Variants like TopK SAEs (Gao et al., 2024) fix L0
directly instead of penalizing it.

## Stretch

Train SAEs with l1 in {0.03, 0.1, 0.3, 1} and plot variance explained
against L0. Then plug the reconstruction x̂ back into the model in place of
the residual stream (a `patch` on `blocks.1.resid_pre`): how much does the
language-model loss go up? That "spliced-in loss" is the metric that matters.
