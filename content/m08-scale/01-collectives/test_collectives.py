import torch

from m08.collectives import all_reduce_worker, bytes_sent_per_rank
from workbench.dist import launch


def expected(n, world):
    return torch.arange(n, dtype=torch.float64) * sum(range(1, world + 1))


def test_ring_all_reduce_three_ranks():
    for t in launch(all_reduce_worker, 3, (10, "ring")):
        assert torch.equal(t, expected(10, 3)), "every rank ends with the full sum"


def test_ring_all_reduce_uneven_chunks():
    for t in launch(all_reduce_worker, 4, (7, "ring")):
        assert torch.equal(t, expected(7, 4)), "7 values over 4 ranks: chunks of different sizes"


def test_naive_all_reduce():
    for t in launch(all_reduce_worker, 3, (5, "naive")):
        assert torch.equal(t, expected(5, 3))


def test_bytes():
    assert bytes_sent_per_rank(1000, 4, "naive") == 3000
    assert bytes_sent_per_rank(1000, 4) == 1500
    assert bytes_sent_per_rank(1000, 1000) < 2000, "the ring's cost per rank barely grows with the world"
