import numpy as np
import torch

from m03.torch_gpt import TorchGPT
from m05.features import density, latent_activations, logit_effects, top_contexts
from m05.sae import SAE, residual_activations


def test_top_contexts():
    acts = torch.zeros(2, 6, 3)
    acts[0, 4, 1] = 2.0
    acts[1, 1, 1] = 3.0
    acts[1, 0, 1] = 0.5
    acts[0, 5, 1] = 1.0
    out = top_contexts(acts, 1, k=10, window=3)
    assert [(c["row"], c["pos"]) for c in out] == [(1, 1), (0, 4), (0, 5), (1, 0)], (
        "largest first, and stop at activations that are 0")
    assert out[0]["act"] == 3.0
    assert out[0]["start"] == 0 and out[0]["acts"] == [0.5, 3.0], "a window can't start before 0"
    assert out[1]["start"] == 2 and out[1]["acts"] == [0.0, 0.0, 2.0], "window tokens end at the peak"
    assert len(top_contexts(acts, 1, k=2)) == 2


def test_density():
    acts = torch.zeros(2, 5, 2)
    acts[0, :2, 0] = 1.0
    assert abs(density(acts, 0) - 0.2) < 1e-6 and density(acts, 1) == 0.0


@torch.no_grad()
def test_latent_activations():
    torch.manual_seed(0)
    gpt = TorchGPT(10, 8, n_embd=16, n_head=2, n_layer=2)
    sae = SAE(16, 24)
    idx = torch.randint(0, 10, (3, 5))
    a = latent_activations(sae, gpt, idx, 1)
    assert a.shape == (3, 5, 24)
    assert torch.allclose(a.reshape(15, 24), sae.encode(residual_activations(gpt, idx, 1)))


@torch.no_grad()
def test_logit_effects():
    torch.manual_seed(0)
    gpt = TorchGPT(10, 8, n_embd=16, n_head=2, n_layer=1)
    gpt.ln_f.weight.uniform_(0.2, 2.0)
    sae = SAE(16, 4)
    up, up_v, down, down_v = logit_effects(sae, gpt, 2, k=3)
    expected = gpt.lm_head.weight @ (sae.W_dec[:, 2] * gpt.ln_f.weight)
    assert list(up) == expected.argsort(descending=True)[:3].tolist()
    assert list(down) == expected.argsort()[:3].tolist()
    assert np.allclose(up_v, expected[up].numpy()) and up_v[0] >= up_v[-1]
