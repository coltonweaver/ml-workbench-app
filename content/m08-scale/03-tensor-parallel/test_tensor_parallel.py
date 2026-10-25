import pytest
import torch

from m08.tensor_parallel import split_mlp, tp_bytes_per_layer, tp_worker
from workbench.dist import launch


def test_split_shapes():
    fc_w, fc_b, proj_w, proj_b = torch.randn(32, 8), torch.randn(32), torch.randn(8, 32), torch.randn(8)
    a = split_mlp(fc_w, fc_b, proj_w, proj_b, 1, 4)
    assert a[0].shape == (8, 8) and a[1].shape == (8,) and a[2].shape == (8, 8) and a[3] is None
    assert torch.equal(a[0], fc_w[8:16]) and torch.equal(a[2], proj_w[:, 8:16])
    assert torch.equal(split_mlp(fc_w, fc_b, proj_w, proj_b, 0, 4)[3], proj_b), "bias added once, on rank 0"
    with pytest.raises(ValueError):
        split_mlp(fc_w, fc_b, proj_w, proj_b, 0, 5)


def test_matches_the_full_mlp():
    for out, full in launch(tp_worker, 4):
        assert torch.allclose(out, full, atol=1e-4)


def test_bytes():
    assert tp_bytes_per_layer(2048, 4096) == 2048 * 4096 * 2
