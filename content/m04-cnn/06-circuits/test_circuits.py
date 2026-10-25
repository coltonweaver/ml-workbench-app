import numpy as np
import torch
import torch.nn as nn

from m04.circuits import ablate, ablation_effect, circuit_strengths, class_accuracy_drop, excitation


class TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(3, 2, 1, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, 0, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
            self.conv2.weight[0, :, 0, 0] = torch.tensor([1.0, 0.0, -1.0])
            self.conv2.weight[1, :, 0, 0] = torch.tensor([0.0, 1.0, 0.0])
        self.fc = nn.Linear(2, 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(2))
            self.fc.bias.copy_(torch.tensor([0.0, -3.0]))

    def forward(self, x):
        h = torch.relu(self.conv1(x))
        return self.fc(self.conv2(h).mean(dim=(2, 3)))


def test_strengths_and_excitation():
    w = torch.zeros(2, 3, 2, 2)
    w[1, 0] = 3.0        # norm 6, sum 12
    w[1, 2] = -1.0       # norm 2, sum -4
    assert np.allclose(circuit_strengths(w, 1), [6.0, 0.0, 2.0])
    assert np.allclose(excitation(w, 1), [12.0, 0.0, -4.0])
    assert np.allclose(circuit_strengths(w, 0), 0)


def test_ablate_zeroes_and_restores():
    m = TwoLayer()
    x = torch.ones(1, 1, 2, 2)
    with ablate(m.conv1, 2):
        out = m.conv1(x)
        assert torch.all(out[:, 2] == 0) and torch.all(out[:, 1] == 2)
    assert torch.all(m.conv1(x)[:, 2] == 3), "the ablation ends with the with block"
    assert len(m.conv1._forward_hooks) == 0


def test_ablation_effect():
    m = TwoLayer()
    X = np.ones((4, 1, 3, 3), dtype=np.float32)
    normal, ablated = ablation_effect(m, m.conv1, 2, m.conv2, 0, X)
    assert np.isclose(normal, 1.0 - 3.0) and np.isclose(ablated, 1.0), (
        "removing an inhibitory input (weight -1) raises the activation")
    n2, a2 = ablation_effect(m, m.conv1, 0, m.conv2, 1, X)
    assert np.isclose(n2, a2), "channel 1 of conv2 doesn't read conv1 channel 0 at all"


def test_class_accuracy_drop():
    m = TwoLayer()
    X = np.ones((4, 1, 2, 2), dtype=np.float32)
    Y = np.array([1, 1, 1, 1])  # logits [-2, -1]: class 1, correct everywhere
    drop = class_accuracy_drop(m, m.conv1, 1, X, Y, 2)
    assert drop.shape == (2,)
    assert np.isnan(drop[0]), "no examples of class 0: NaN, not 0"
    assert drop[1] == 1.0, "without conv1 channel 1 the logits are [-2, -3]: every answer flips"
    assert np.all(class_accuracy_drop(m, m.conv1, 0, X, Y, 2)[1:] == 0), (
        "channel 0 only feeds conv2 channel 0, which is already losing")
