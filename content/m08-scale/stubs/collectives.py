"""Collective communication from point-to-point sends and receives.

Lesson 8.1."""

import torch
import torch.distributed as dist


def naive_all_reduce(t):
    """Everyone sends to rank 0, which sums and sends the total back.
    In place; returns t."""
    raise NotImplementedError("lesson 8.1")


def ring_all_reduce(t):
    """Sum ``t`` across ranks, in place, with the ring algorithm.

    Split t into ``world`` chunks. Reduce-scatter: for world − 1 steps, each
    rank sends one chunk to the next rank and adds the chunk it receives
    from the previous one, so that rank r ends up with the full sum of chunk
    (r + 1) % world. All-gather: for world − 1 more steps, pass the finished
    chunks around the ring, overwriting. Every rank sends 2·(world − 1)/world
    of the data, however many ranks there are.
    """
    raise NotImplementedError("lesson 8.1")


def bytes_sent_per_rank(n_bytes, world, algorithm="ring"):
    """Bytes each rank sends to all-reduce ``n_bytes``. Naive: rank 0 sends
    (world − 1)·n (the bottleneck); ring: 2·(world − 1)/world·n."""
    raise NotImplementedError("lesson 8.1")


def all_reduce_worker(rank, world, n, algorithm):
    """Test helper: each rank holds arange(n) * (rank + 1); returns the
    all-reduced tensor."""
    t = torch.arange(n, dtype=torch.float64) * (rank + 1)
    (ring_all_reduce if algorithm == "ring" else naive_all_reduce)(t)
    return t
