"""Browsing features: where they fire, and what they push the model toward.

Lesson 5.7."""
import numpy as np
import torch
from m05.sae import residual_activations


@torch.no_grad()
def latent_activations(sae, model, idx, layer):
    """SAE latent activations on idx at ``layer``: (B, T, n_latents)."""
    raise NotImplementedError('lesson 5.7')


def top_contexts(acts, latent, k=10, window=12):
    """The k (row, position) pairs where ``latent`` is most active, largest
    first, each with the activations over a window of ``window`` tokens
    ending there. ``acts`` is (B, T, n_latents). Returns a list of dicts
    {"row", "pos", "start", "act", "acts"}; "start" is where the window
    begins and "acts" the latent's values over it."""
    raise NotImplementedError('lesson 5.7')


@torch.no_grad()
def logit_effects(sae, model, latent, k=5):
    """The tokens a latent's decoder direction pushes up and down most,
    read through ln_f's scale and the unembedding (ignoring the later
    layers). Returns (top token ids, their values, bottom ids, their values)."""
    raise NotImplementedError('lesson 5.7')


def density(acts, latent):
    """Fraction of tokens where the latent fires."""
    raise NotImplementedError('lesson 5.7')
