"""A sparse autoencoder: pull the residual stream apart into features."""

import numpy as np
import torch

from m05.hooks import run_with_cache


class SAE(torch.nn.Module):
    """f = ReLU(W_enc (x - b_dec) + b_enc), x̂ = W_dec f + b_dec, with
    W_enc (n_latents, d) and W_dec (d, n_latents). Decoder columns are kept
    at unit norm, so a latent's size lives in f, not in W_dec."""

    def __init__(self, d, n_latents, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        w = torch.randn(d, n_latents, generator=gen)
        w = w / w.norm(dim=0, keepdim=True)
        self.W_dec = torch.nn.Parameter(w.clone())
        self.W_enc = torch.nn.Parameter(w.T.clone())
        self.b_enc = torch.nn.Parameter(torch.zeros(n_latents))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))

    def encode(self, x):
        return torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec /= self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)


def sae_loss(sae, x, l1):
    """Mean squared reconstruction error (summed over dimensions) plus
    ``l1`` times the mean L1 norm of f. Returns (loss, x̂, f)."""
    x_hat, f = sae(x)
    mse = ((x - x_hat) ** 2).sum(-1).mean()
    return mse + l1 * f.abs().sum(-1).mean(), x_hat, f


def train_sae(sae, X, steps=2000, batch_size=256, lr=1e-3, l1=0.1, seed=0, log=None):
    """Adam on minibatches of the rows of X (N, d), renormalizing the
    decoder after every step. ``log(step, loss)``. Returns the losses."""
    rng = np.random.default_rng(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        sae.b_dec.copy_(X.mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        batch = X[torch.as_tensor(rng.integers(0, len(X), size=batch_size))]
        loss, _, _ = sae_loss(sae, batch, l1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.normalize_decoder()
        losses.append(loss.item())
        if log is not None:
            log(step, losses[-1])
    return losses


@torch.no_grad()
def sae_metrics(sae, X):
    """{"l0": mean active latents per input, "fve": fraction of variance
    explained, "dead": fraction of latents that never fire on X}."""
    X = torch.as_tensor(X, dtype=torch.float32)
    x_hat, f = sae(X)
    resid = ((X - x_hat) ** 2).sum()
    total = ((X - X.mean(0)) ** 2).sum()
    return {
        "l0": (f > 0).float().sum(-1).mean().item(),
        "fve": 1 - (resid / total).item(),
        "dead": (f.max(0).values <= 0).float().mean().item(),
    }


@torch.no_grad()
def residual_activations(model, idx, layer):
    """The residual stream leaving block ``layer`` on idx, flattened to
    (B·T, C)."""
    _, cache = run_with_cache(model, torch.as_tensor(idx, dtype=torch.long))
    r = cache[f"blocks.{layer}.resid_post"]
    return r.reshape(-1, r.shape[-1])
