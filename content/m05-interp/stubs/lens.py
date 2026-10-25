"""Reading the residual stream in the vocabulary: the logit lens and
direct logit attribution.

Lesson 5.2."""
import torch


def logit_lens(model, cache):
    """Apply the final layer norm and unembedding to the residual stream
    after the embedding and after each block. Returns a tensor of shape
    (n_layer + 1, B, T, V): what the model would predict if it stopped
    there."""
    raise NotImplementedError('lesson 5.2')


def components(model, cache):
    """The residual stream's final value as a sum of named pieces:
    [("embed", t), ("attn 0", t), ("mlp 0", t), ...], each (B, T, C)."""
    raise NotImplementedError('lesson 5.2')


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
    raise NotImplementedError('lesson 5.2')
