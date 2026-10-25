import numpy as np
import torch
import torch.nn as nn

from m04.featvis import get_activations, visualize_channel

SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


class EdgeNet(nn.Module):
    """A conv layer with two hand-set channels: vertical edges, horizontal edges."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, bias=False)
        with torch.no_grad():
            self.conv.weight[0, 0] = SOBEL_X
            self.conv.weight[1, 0] = SOBEL_X.T
        self.head = nn.Linear(2, 2)

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)).mean(dim=(2, 3)))


def test_get_activations_uses_a_hook_and_removes_it():
    net = EdgeNet()
    x = np.random.default_rng(0).normal(size=(3, 1, 8, 8)).astype(np.float32)
    a = get_activations(net, net.conv, x)
    assert a.shape == (3, 2, 6, 6)
    assert torch.allclose(a, net.conv(torch.as_tensor(x)))
    assert len(net.conv._forward_hooks) == 0, "remove the hook when you're done"


def energy(img):
    gx = np.abs(np.diff(img, axis=1)).mean()  # change left to right
    gy = np.abs(np.diff(img, axis=0)).mean()  # change top to bottom
    return gx, gy


def test_visualization_finds_the_orientation():
    net = EdgeNet()
    v0 = visualize_channel(net, net.conv, 0, steps=150, size=24)
    v1 = visualize_channel(net, net.conv, 1, steps=150, size=24)
    assert isinstance(v0, np.ndarray) and v0.shape == (24, 24)
    gx0, gy0 = energy(v0)
    gx1, gy1 = energy(v1)
    assert gx0 > 2 * gy0, "the vertical-edge channel should want vertical structure"
    assert gy1 > 2 * gx1, "the horizontal-edge channel should want horizontal structure"


def test_visualization_beats_noise_and_is_bounded():
    net = EdgeNet()
    v = visualize_channel(net, net.conv, 0, steps=150, size=24, seed=3)
    act = lambda img: torch.relu(get_activations(net, net.conv, img[None, None])[0, 0]).mean().item()  # noqa: E731
    noise = np.random.default_rng(0).normal(scale=np.abs(v).std(), size=v.shape).astype(np.float32)
    assert act(v.astype(np.float32)) > 5 * abs(act(noise)), "the optimized image should excite the channel"
    assert np.isfinite(v).all()
    again = visualize_channel(net, net.conv, 0, steps=150, size=24, seed=3)
    assert np.allclose(v, again), "same seed, same image"
