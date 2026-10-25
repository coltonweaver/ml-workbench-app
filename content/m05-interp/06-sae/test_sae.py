import numpy as np
import torch

from m03.torch_gpt import TorchGPT
from m05.sae import SAE, residual_activations, sae_loss, sae_metrics, train_sae


def test_encode_decode_formulas():
    sae = SAE(4, 6, seed=0)
    assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(6)), "decoder columns start at unit norm"
    with torch.no_grad():
        sae.b_dec.copy_(torch.tensor([1.0, -1.0, 0.5, 0.0]))
        sae.b_enc.copy_(torch.linspace(-0.5, 0.5, 6))
    x = torch.randn(5, 4)
    f = sae.encode(x)
    assert torch.allclose(f, torch.relu((x - sae.b_dec) @ sae.W_enc.T + sae.b_enc))
    x_hat, f2 = sae(x)
    assert torch.allclose(f, f2) and torch.allclose(x_hat, f @ sae.W_dec.T + sae.b_dec)


def test_normalize_decoder():
    sae = SAE(3, 4)
    with torch.no_grad():
        sae.W_dec.mul_(torch.tensor([2.0, 0.5, 3.0, 1.0]))
    sae.normalize_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(4))


def test_loss():
    sae = SAE(2, 2)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(2))
        sae.W_dec.copy_(torch.eye(2))
    x = torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
    loss, x_hat, f = sae_loss(sae, x, l1=0.5)
    # f = [[1, 2], [0, 3]], x̂ = f: errors 0 and 1, L1 norms 3 and 3
    assert torch.allclose(f, torch.tensor([[1.0, 2.0], [0.0, 3.0]]))
    assert torch.isclose(loss, torch.tensor(0.5 + 0.5 * 3.0))


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(32, 16))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    coef = rng.random((8000, 32)) * (rng.random((8000, 32)) < 0.05)
    return (coef @ D).astype(np.float32), D


def test_recovers_the_true_features():
    X, D = synthetic()
    sae = SAE(16, 64)
    losses = train_sae(sae, X, steps=2000, lr=3e-3, l1=0.3)
    assert len(losses) == 2000
    assert torch.allclose(sae.W_dec.norm(dim=0), torch.ones(64), atol=1e-4), "renormalize after each step"
    best = (torch.tensor(D, dtype=torch.float32) @ sae.W_dec.detach()).max(dim=1).values
    assert (best > 0.9).float().mean() > 0.9, "almost every true direction should have a latent pointing at it"
    m = sae_metrics(sae, X)
    assert m["l0"] < 4, f"L0 {m['l0']:.1f}: each input uses only a few latents"
    assert m["fve"] > 0.8


def test_metrics_by_hand():
    sae = SAE(2, 3)
    with torch.no_grad():
        sae.W_enc.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
        sae.W_dec.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    X = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    m = sae_metrics(sae, X)
    assert np.isclose(m["l0"], 1.0) and np.isclose(m["fve"], 1.0) and np.isclose(m["dead"], 1 / 3)


@torch.no_grad()
def test_residual_activations():
    torch.manual_seed(0)
    gpt = TorchGPT(10, 8, n_embd=16, n_head=2, n_layer=2)
    idx = torch.randint(0, 10, (3, 5))
    r = residual_activations(gpt, idx, 0)
    assert r.shape == (15, 16)
    x = gpt.wte(idx) + gpt.wpe(torch.arange(5))
    assert torch.allclose(r.reshape(3, 5, 16), gpt.blocks[0](x), atol=1e-5)
