import numpy as np
import torch
import torch.nn as nn

from m04.featvis import top_activating


class Spot(nn.Module):
    """One channel that just copies the image, so activation = pixel value."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)

    def forward(self, x):
        return self.conv(x)


def test_top_activating_order_and_k():
    X = np.zeros((6, 1, 4, 4), dtype=np.float32)
    for i, v in enumerate([0.1, 0.9, 0.5, 0.3, 0.7, 0.2]):
        X[i, 0, 1, 1] = v
    m = Spot()
    idx, acts = top_activating(m, m.conv, 0, X, k=3)
    assert list(idx) == [1, 4, 2], "largest first"
    assert np.allclose(acts, [0.9, 0.7, 0.5])


def test_peak_not_average():
    X = np.zeros((2, 1, 8, 8), dtype=np.float32)
    X[0] = 0.2              # weak everywhere
    X[1, 0, 3, 3] = 1.0     # strong in one place
    m = Spot()
    idx, _ = top_activating(m, m.conv, 0, X, k=1)
    assert idx[0] == 1, "rank by the strongest response anywhere in the image (max over positions)"
