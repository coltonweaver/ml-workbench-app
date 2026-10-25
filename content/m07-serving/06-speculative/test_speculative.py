import numpy as np
import torch

from m03.torch_gpt import TorchGPT
from m07.speculative import expected_tokens, first_token_distribution, residual, speculative_greedy, verify


def test_residual():
    assert np.allclose(residual([0.5, 0.3, 0.2], [0.2, 0.2, 0.6]), [0.75, 0.25, 0.0])


def test_output_distribution_is_the_target():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p, q = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
        assert np.allclose(first_token_distribution(p, q), p)


def test_verify_empirically():
    p = np.array([0.6, 0.3, 0.1])
    q = np.array([0.2, 0.2, 0.6])
    rng = np.random.default_rng(1)
    counts = np.zeros(3)
    for _ in range(20000):
        x = rng.choice(3, p=q)
        out = verify([p, p], [q], [x], rng)
        counts[out[0]] += 1
        assert 1 <= len(out) <= 2
    assert np.allclose(counts / counts.sum(), p, atol=0.015), "speculation must not change what is sampled"


def test_verify_accepts_matching_drafts():
    p = np.array([0.0, 1.0, 0.0])
    rng = np.random.default_rng(0)
    assert verify([p, p, p], [p, p], [1, 1], rng) == [1, 1, 1], "all accepted, plus a bonus token"
    q = np.array([1.0, 0.0, 0.0])
    assert verify([p, p, p], [q, q], [0, 0], rng) == [1], "a token the target never picks is rejected"


def test_expected_tokens():
    assert expected_tokens(0.0, 4) == 1
    assert expected_tokens(1.0, 4) == 5
    assert np.isclose(expected_tokens(0.8, 4), 1 + 0.8 + 0.64 + 0.512 + 0.4096)


@torch.no_grad()
def test_speculative_greedy_is_exact():
    torch.manual_seed(0)
    target = TorchGPT(15, 32, n_embd=32, n_head=4, n_layer=2)
    draft = TorchGPT(15, 32, n_embd=16, n_head=2, n_layer=1)
    idx = torch.randint(0, 15, (1, 4))
    ref = idx
    for _ in range(12):
        ref = torch.cat([ref, target(ref)[:, -1].argmax(-1, keepdim=True)], 1)
    out, passes = speculative_greedy(target, draft, idx, 12, k=3)
    assert torch.equal(out, ref), "same tokens as the target alone"
    assert passes <= 12
    same, passes_same = speculative_greedy(target, target, idx, 12, k=3)
    assert torch.equal(same, ref) and passes_same == 3, "a perfect draft gives k + 1 tokens per pass"
