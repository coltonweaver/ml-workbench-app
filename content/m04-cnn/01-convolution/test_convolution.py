import numpy as np
import pytest
import torch
import torch.nn.functional as F

from m04.conv import conv2d, output_size

rng = np.random.default_rng(0)


def test_output_size():
    assert output_size(32, 5) == 28
    assert output_size(32, 3, padding=1) == 32
    assert output_size(9, 3, stride=2) == 4
    assert output_size(9, 3, stride=2, padding=1) == 5


def test_tiny_example():
    x = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    w = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
    out = conv2d(x, w)
    assert out.shape == (1, 1, 3, 3)
    assert np.allclose(out, -5.0), "x[i, j] - x[i+1, j+1] is -5 everywhere in this grid"


def test_not_flipped():
    x = np.zeros((1, 1, 3, 3))
    x[0, 0, 0, 0] = 1.0
    w = np.arange(9, dtype=float).reshape(1, 1, 3, 3)
    assert conv2d(x, w)[0, 0, 0, 0] == 0.0, (
        "deep learning 'convolution' doesn't flip the kernel: out[0, 0] = x[0, 0] * w[0, 0]")


@pytest.mark.parametrize("stride,padding,bias", [(1, 0, True), (1, 2, False), (2, 1, True), (3, 0, True)])
def test_matches_pytorch(stride, padding, bias):
    x = rng.normal(size=(2, 3, 11, 9))
    w = rng.normal(size=(4, 3, 3, 5))
    b = rng.normal(size=4) if bias else None
    ours = conv2d(x, w, b, stride=stride, padding=padding)
    ref = F.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b) if bias else None,
                   stride=stride, padding=padding).numpy()
    assert ours.shape == ref.shape, f"shape {ours.shape}, expected {ref.shape}"
    assert np.allclose(ours, ref), "values differ from torch.nn.functional.conv2d"
