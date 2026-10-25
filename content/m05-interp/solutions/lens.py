"""Reading the residual stream in the vocabulary: the logit lens and
direct logit attribution."""

import torch


def logit_lens(model, cache):
    """Apply the final layer norm and unembedding to the residual stream
    after the embedding and after each block. Returns a tensor of shape
    (n_layer + 1, B, T, V): what the model would predict if it stopped
    there."""
    n = len(model.blocks)
    streams = [cache["blocks.0.resid_pre"]] + [cache[f"blocks.{i}.resid_post"] for i in range(n)]
    return torch.stack([model.lm_head(model.ln_f(s)) for s in streams])


def components(model, cache):
    """The residual stream's final value as a sum of named pieces:
    [("embed", t), ("attn 0", t), ("mlp 0", t), ...], each (B, T, C)."""
    out = [("embed", cache["embed"])]
    for i in range(len(model.blocks)):
        out.append((f"attn {i}", cache[f"blocks.{i}.attn_out"]))
        out.append((f"mlp {i}", cache[f"blocks.{i}.mlp_out"]))
    return out


def direct_logit_attribution(model, cache, pos, token, batch=0):
    """How much each component adds directly to the logit of ``token`` at
    position ``pos``.

    ln_f isn't linear, but with its scale frozen at the value it had on this
    input it is: ln(x) = (x - mean(x)) / σ · γ + β, and σ is shared by every
    piece of x (the cache has it, as "ln_final.scale"). So each component c
    contributes
    ((c - mean(c)) / σ · γ) · W_U[token], and β · W_U[token] is a constant.

    Returns (names, contributions as a 1D tensor, the constant); the
    contributions plus the constant sum to the actual logit.
    """
    ln = model.ln_f
    sigma = cache["ln_final.scale"][batch, pos]
    w_u = model.lm_head.weight[token]
    names, contribs = [], []
    for name, c in components(model, cache):
        c = c[batch, pos]
        names.append(name)
        contribs.append(((c - c.mean()) / sigma * ln.weight) @ w_u)
    return names, torch.stack(contribs), ln.bias @ w_u
