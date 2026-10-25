import numpy as np
import torch

from m03.torch_gpt import TorchGPT
from m05.capstone import ablate_head, ablation_table, claim_check, head_means
from m05.hooks import loss_from_logits, run


def setup():
    torch.manual_seed(6)
    m = TorchGPT(10, 10, n_embd=16, n_head=4, n_layer=2)
    return m, torch.randint(0, 10, (4, 9))


def test_ablate_one_head():
    z = torch.ones(2, 3, 4, 5)
    out = ablate_head(1, 2)["blocks.1.attn.z"](z)
    assert torch.all(out[:, :, 2] == 0) and torch.all(out[:, :, [0, 1, 3]] == 1)
    assert torch.all(z == 1), "don't edit in place"
    out = ablate_head(0, 1, "mean", torch.arange(5.0))["blocks.0.attn.z"](z)
    assert torch.equal(out[0, 2, 1], torch.arange(5.0)) and torch.all(out[:, :, 0] == 1)


@torch.no_grad()
def test_head_means():
    m, idx = setup()
    means = head_means(m, idx)
    cache = {}
    run(m, idx, cache=cache)
    assert means[1].shape == (4, 4)
    assert torch.allclose(means[1], cache["blocks.1.attn.z"].mean(dim=(0, 1)))


@torch.no_grad()
def test_ablation_table():
    m, idx = setup()
    table = ablation_table(m, idx, kind="zero")
    assert table.shape == (2, 4)
    x, y = idx[:, :-1], idx[:, 1:]
    base = loss_from_logits(run(m, x), y).item()
    hurt = loss_from_logits(run(m, x, patch=ablate_head(1, 3)), y).item()
    assert np.isclose(table[1, 3], hurt - base, atol=1e-5), "the increase in loss"
    part = ablation_table(m, idx, kind="zero", positions=slice(4, None))
    hurt = loss_from_logits(run(m, x, patch=ablate_head(0, 1))[:, 4:], y[:, 4:]).item()
    base = loss_from_logits(run(m, x)[:, 4:], y[:, 4:]).item()
    assert np.isclose(part[0, 1], hurt - base, atol=1e-5), "only the chosen predictions"
    assert not np.allclose(ablation_table(m, idx, kind="mean"), table), "mean ablation is a different experiment"


def test_claim_check():
    t = np.array([[0.1, 0.05], [0.9, 0.2]])
    assert claim_check(t, 1, 0) == (True, 0.9, 0.2)
    assert not claim_check(t, 1, 0, margin=5.0)[0], "0.9 is not 5 × 0.2"
    assert not claim_check(t, 0, 0)[0]
    assert not claim_check(np.array([[-0.1, -0.3]]), 0, 0)[0], "a head whose removal helps isn't necessary"
