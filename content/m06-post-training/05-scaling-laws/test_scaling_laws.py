import numpy as np
import torch

from m03.torch_gpt import TorchGPT
from m06.scaling import count_params, fit_power_law, fit_power_law_offset, size_sweep


def test_fit_power_law_exact():
    x = np.array([1e3, 1e4, 1e5, 1e6])
    a, alpha = fit_power_law(x, 50 * x ** -0.3)
    assert np.isclose(a, 50) and np.isclose(alpha, 0.3), "α is positive for a falling curve"


def test_fit_with_noise():
    rng = np.random.default_rng(0)
    x = np.logspace(3, 7, 12)
    y = 20 * x ** -0.25 * np.exp(rng.normal(0, 0.01, size=12))
    a, alpha = fit_power_law(x, y)
    assert abs(alpha - 0.25) < 0.01


def test_fit_with_irreducible_loss():
    x = np.logspace(4, 8, 10)
    y = 400 * x ** -0.35 + 1.7
    a, alpha, c = fit_power_law_offset(x, y)
    assert abs(c - 1.7) < 0.03 and abs(alpha - 0.35) < 0.03
    assert np.allclose(a * x ** -alpha + c, y, rtol=0.01)
    _, alpha_plain = fit_power_law(x, y)
    assert alpha_plain < 0.3, "ignoring the floor makes the curve look flatter than it is"


def test_count_params():
    m = TorchGPT(20, 16, n_embd=32, n_head=2, n_layer=2)
    total = sum(p.numel() for p in m.parameters())
    assert count_params(m, embeddings=True) == total
    assert count_params(m) == total - 20 * 32 - 16 * 32


def test_size_sweep_runs():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 12, size=4000)
    seen = []
    params, losses = size_sweep([16, 32], data[:3000], data[3000:], 12, steps=20, block_size=16,
                                log=lambda w, n, l: seen.append(w))
    assert seen == [16, 32]
    assert params.shape == losses.shape == (2,) and params[1] > params[0]
    assert np.all(np.isfinite(losses)) and np.all(losses < 3.0), "random tokens: loss near ln 12 ≈ 2.48"
