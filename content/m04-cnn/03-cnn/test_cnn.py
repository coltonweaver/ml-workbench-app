import numpy as np
import torch

from m04.cnn import ShapeCNN, accuracy, train_cnn
from workbench.data import SHAPE_CLASSES, shapes


def test_architecture():
    m = ShapeCNN(6)
    assert sum(p.numel() for p in m.parameters()) == 5830
    assert m.conv1.kernel_size == (5, 5) and m.conv2.out_channels == 16
    out = m(torch.zeros(3, 1, 32, 32))
    assert out.shape == (3, 6)


def test_accuracy_function():
    class Fixed(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[1.0, 0.0]] * len(x))
    assert accuracy(Fixed(), np.zeros((4, 1, 2, 2)), np.array([0, 0, 1, 0])) == 0.75


def test_learns_the_shapes():
    X, Y = shapes(1500, seed=0)
    Xt, Yt = shapes(400, seed=1)
    torch.manual_seed(0)
    m = ShapeCNN(len(SHAPE_CLASSES))
    seen = []
    losses = train_cnn(m, X, Y, steps=250, batch_size=64, lr=1e-2, seed=0, log=lambda s, l: seen.append(s))
    assert len(losses) == 250 and seen == list(range(250))
    acc = accuracy(m, Xt, Yt)
    assert acc > 0.85, f"held-out accuracy {acc:.0%}: a working CNN gets well above 85% here"
