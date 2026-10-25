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

Lesson 5.1."""
import math
import torch
import torch.nn.functional as F


def hook_names(model):
    """Every hook point name for ``model``, in the order the forward pass
    reaches them."""
    raise NotImplementedError('lesson 5.1')


def layer_norm(ln, x, scale_hook):
    """``ln(x)`` computed by hand, so its divisor passes through
    ``scale_hook`` (the ``point`` function in ``run``)."""
    raise NotImplementedError('lesson 5.1')


def attention_parts(attn, x):
    """Run a CausalSelfAttention on its (already layer-normed) input x and
    return (pattern (B, H, T, T), z (B, T, H, hs)) separately, so each can be
    read or replaced."""
    raise NotImplementedError('lesson 5.1')


def run(model, idx, patch=None, cache=None):
    """The same computation as ``model(idx)``, with a hook point at each name.

    At each point, if ``patch`` has that name, the tensor is replaced by
    ``patch[name](tensor)``; then, if ``cache`` is a dict, the (possibly
    patched) tensor is stored in it under that name. Returns the logits.
    """
    raise NotImplementedError('lesson 5.1')


def run_with_cache(model, idx):
    """(logits, cache) with every hook point recorded."""
    raise NotImplementedError('lesson 5.1')


def loss_from_logits(logits, targets):
    raise NotImplementedError('lesson 5.1')
