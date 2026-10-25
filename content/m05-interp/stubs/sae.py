"""A sparse autoencoder: pull the residual stream apart into features.

Lesson 5.6."""
import numpy as np
import torch
from m05.hooks import run_with_cache


class SAE(torch.nn.Module):
    """f = ReLU(W_enc (x - b_dec) + b_enc), x̂ = W_dec f + b_dec, with
    W_enc (n_latents, d) and W_dec (d, n_latents). Decoder columns are kept
    at unit norm, so a latent's size lives in f, not in W_dec."""

    def __init__(self, d, n_latents, seed=0):
        raise NotImplementedError('lesson 5.6')

    def encode(self, x):
        raise NotImplementedError('lesson 5.6')

    def decode(self, f):
        raise NotImplementedError('lesson 5.6')

    def forward(self, x):
        raise NotImplementedError('lesson 5.6')

    @torch.no_grad()
    def normalize_decoder(self):
        raise NotImplementedError('lesson 5.6')


def sae_loss(sae, x, l1):
    """Mean squared reconstruction error (summed over dimensions) plus
    ``l1`` times the mean L1 norm of f. Returns (loss, x̂, f)."""
    raise NotImplementedError('lesson 5.6')


def train_sae(sae, X, steps=2000, batch_size=256, lr=0.001, l1=0.1, seed=0, log=None):
    """Adam on minibatches of the rows of X (N, d), renormalizing the
    decoder after every step. ``log(step, loss)``. Returns the losses."""
    raise NotImplementedError('lesson 5.6')


@torch.no_grad()
def sae_metrics(sae, X):
    """{"l0": mean active latents per input, "fve": fraction of variance
    explained, "dead": fraction of latents that never fire on X}."""
    raise NotImplementedError('lesson 5.6')


@torch.no_grad()
def residual_activations(model, idx, layer):
    """The residual stream leaving block ``layer`` on idx, flattened to
    (B·T, C)."""
    raise NotImplementedError('lesson 5.6')
