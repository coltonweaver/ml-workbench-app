"""A forward pass of your PyTorch GPT with named places to read and edit.

Hook point names, for each block i:

    embed                     token + position embeddings   (B, T, C)
    blocks.{i}.resid_pre      the residual stream entering block i
    blocks.{i}.ln1.scale      ln1's divisor √(var + eps)    (B, T, 1)
    blocks.{i}.attn.pattern   attention weights             (B, H, T, T)
    blocks.{i}.attn.z         each head's output, before o  (B, T, H, hs)
    blocks.{i}.attn_out       what attention adds           (B, T, C)
    blocks.{i}.resid_mid      after attention
    blocks.{i}.ln2.scale      ln2's divisor
    blocks.{i}.mlp_out        what the MLP adds             (B, T, C)
    blocks.{i}.resid_post     leaving block i
    ln_final.scale            ln_f's divisor
    final                     after ln_f, before lm_head    (B, T, C)

Patching a ``.scale`` to its detached value freezes that layer norm, which
makes it a linear map of the residual stream for attribution.
"""

import math

import torch
import torch.nn.functional as F


def hook_names(model):
    """Every hook point name for ``model``, in the order the forward pass
    reaches them."""
    names = ["embed"]
    for i in range(len(model.blocks)):
        names += [f"blocks.{i}.{n}" for n in
                  ("resid_pre", "ln1.scale", "attn.pattern", "attn.z", "attn_out",
                   "resid_mid", "ln2.scale", "mlp_out", "resid_post")]
    return names + ["ln_final.scale", "final"]


def layer_norm(ln, x, scale_hook):
    """``ln(x)`` computed by hand, so its divisor passes through
    ``scale_hook`` (the ``point`` function in ``run``)."""
    centered = x - x.mean(-1, keepdim=True)
    scale = scale_hook(torch.sqrt(centered.pow(2).mean(-1, keepdim=True) + ln.eps))
    return centered / scale * ln.weight + ln.bias


def attention_parts(attn, x):
    """Run a CausalSelfAttention on its (already layer-normed) input x and
    return (pattern (B, H, T, T), z (B, T, H, hs)) separately, so each can be
    read or replaced."""
    B, T, C = x.shape
    H = attn.n_head
    hs = C // H

    def heads(t):
        return t.view(B, T, H, hs).transpose(1, 2)

    q, k = heads(attn.q(x)), heads(attn.k(x))
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    pattern = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
    return pattern, heads(attn.v(x))


def run(model, idx, patch=None, cache=None):
    """The same computation as ``model(idx)``, with a hook point at each name.

    At each point, if ``patch`` has that name, the tensor is replaced by
    ``patch[name](tensor)``; then, if ``cache`` is a dict, the (possibly
    patched) tensor is stored in it under that name. Returns the logits.
    """
    patch = patch or {}

    def point(name, t):
        if name in patch:
            t = patch[name](t)
        if cache is not None:
            cache[name] = t
        return t

    B, T = idx.shape
    x = point("embed", model.wte(idx) + model.wpe(torch.arange(T, device=idx.device)))
    for i, block in enumerate(model.blocks):
        p = f"blocks.{i}."
        x = point(p + "resid_pre", x)
        h = layer_norm(block.ln1, x, lambda t: point(p + "ln1.scale", t))
        pattern, v = attention_parts(block.attn, h)
        pattern = point(p + "attn.pattern", pattern)
        z = point(p + "attn.z", (pattern @ v).transpose(1, 2))       # (B, T, H, hs)
        attn_out = point(p + "attn_out", block.attn.o(z.reshape(B, T, -1)))
        x = point(p + "resid_mid", x + attn_out)
        h = layer_norm(block.ln2, x, lambda t: point(p + "ln2.scale", t))
        mlp_out = point(p + "mlp_out", block.mlp(h))
        x = point(p + "resid_post", x + mlp_out)
    x = point("final", layer_norm(model.ln_f, x, lambda t: point("ln_final.scale", t)))
    return model.lm_head(x)


def run_with_cache(model, idx):
    """(logits, cache) with every hook point recorded."""
    cache = {}
    logits = run(model, idx, cache=cache)
    return logits, cache


def loss_from_logits(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
