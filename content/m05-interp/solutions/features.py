"""Browsing features: where they fire, and what they push the model toward."""

import numpy as np
import torch

from m05.sae import residual_activations


@torch.no_grad()
def latent_activations(sae, model, idx, layer):
    """SAE latent activations on idx at ``layer``: (B, T, n_latents)."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    B, T = idx.shape
    return sae.encode(residual_activations(model, idx, layer)).reshape(B, T, -1)


def top_contexts(acts, latent, k=10, window=12):
    """The k (row, position) pairs where ``latent`` is most active, largest
    first, each with the activations over a window of ``window`` tokens
    ending there. ``acts`` is (B, T, n_latents). Returns a list of dicts
    {"row", "pos", "start", "act", "acts"}; "start" is where the window
    begins and "acts" the latent's values over it."""
    a = torch.as_tensor(acts)[..., latent]
    flat = a.flatten()
    order = torch.argsort(flat, descending=True, stable=True)[:k]
    T = a.shape[1]
    out = []
    for i in order.tolist():
        if flat[i] <= 0:
            break
        row, pos = divmod(i, T)
        start = max(0, pos - window + 1)
        out.append({"row": row, "pos": pos, "start": start, "act": float(flat[i]),
                    "acts": a[row, start:pos + 1].tolist()})
    return out


@torch.no_grad()
def logit_effects(sae, model, latent, k=5):
    """The tokens a latent's decoder direction pushes up and down most,
    read through ln_f's scale and the unembedding (ignoring the later
    layers). Returns (top token ids, their values, bottom ids, their values)."""
    d = sae.W_dec[:, latent]
    effect = model.lm_head.weight @ (d * model.ln_f.weight)
    up = torch.argsort(effect, descending=True)[:k]
    down = torch.argsort(effect)[:k]
    return up.numpy(), effect[up].numpy(), down.numpy(), effect[down].numpy()


def density(acts, latent):
    """Fraction of tokens where the latent fires."""
    return float((torch.as_tensor(acts)[..., latent] > 0).float().mean())
