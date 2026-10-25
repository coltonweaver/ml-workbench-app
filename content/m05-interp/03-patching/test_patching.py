import numpy as np
import torch

from m03.torch_gpt import TorchGPT
from m05.hooks import run_with_cache
from m05.patching import logit_diff, patch_position, patching_sweep


def setup():
    torch.manual_seed(2)
    m = TorchGPT(10, 8, n_embd=16, n_head=2, n_layer=2)
    clean = torch.tensor([[1, 4, 2, 7, 3, 3]])
    corrupt = clean.clone()
    corrupt[0, 2] = 9
    return m, clean, corrupt


def test_logit_diff():
    logits = torch.zeros(2, 3, 4)
    logits[0, -1, 1], logits[0, -1, 2] = 3.0, 1.0
    logits[1, -1, 1], logits[1, -1, 2] = 0.0, 2.0
    assert float(logit_diff(logits, 1, 2)) == 0.0, "mean of (3 - 1) and (0 - 2)"
    assert float(logit_diff(logits, 2, 1, pos=0)) == 0.0
    logits[:, 0, 2] = 5.0
    assert float(logit_diff(logits, 2, 1, pos=0)) == 5.0


@torch.no_grad()
def test_patch_position_changes_one_position():
    m, clean, corrupt = setup()
    _, src = run_with_cache(m, clean)
    t = torch.zeros_like(src["blocks.0.mlp_out"])
    out = patch_position("blocks.0.mlp_out", 3, src)(t)
    assert torch.equal(out[:, 3], src["blocks.0.mlp_out"][:, 3])
    assert torch.all(out[:, :3] == 0) and torch.all(out[:, 4:] == 0)
    assert torch.all(t == 0), "don't edit the tensor in place"


def test_sweep_finds_where_the_difference_lives():
    m, clean, corrupt = setup()
    metric = lambda logits: logit_diff(logits, 5, 6)  # noqa: E731
    names = ["blocks.0.resid_pre", "blocks.1.resid_pre", "blocks.1.resid_post"]
    sweep = patching_sweep(m, clean, corrupt, metric, names)
    assert sweep.shape == (3, 6)
    assert np.isclose(sweep[0, 2], 1.0), "the embedding at the changed token carries all of it"
    assert np.allclose(np.delete(sweep[0], 2), 0), "identical activations elsewhere change nothing"
    assert np.isclose(sweep[2, 5], 1.0), "and so does the final stream at the last position"
    assert np.allclose(sweep[2, :5], 0), "the last layer's stream at other positions doesn't reach position 5's output"
    assert np.allclose(sweep[1, :2], 0), "positions before the change never saw it"
