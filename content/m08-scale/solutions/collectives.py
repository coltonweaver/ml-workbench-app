"""Collective communication from point-to-point sends and receives."""

import torch
import torch.distributed as dist


def naive_all_reduce(t):
    """Everyone sends to rank 0, which sums and sends the total back.
    In place; returns t."""
    rank, world = dist.get_rank(), dist.get_world_size()
    if rank == 0:
        buf = torch.empty_like(t)
        for src in range(1, world):
            dist.recv(buf, src=src)
            t += buf
        for dst in range(1, world):
            dist.send(t, dst=dst)
    else:
        dist.send(t, dst=0)
        dist.recv(t, src=0)
    return t


def ring_all_reduce(t):
    """Sum ``t`` across ranks, in place, with the ring algorithm.

    Split t into ``world`` chunks. Reduce-scatter: for world − 1 steps, each
    rank sends one chunk to the next rank and adds the chunk it receives
    from the previous one, so that rank r ends up with the full sum of chunk
    (r + 1) % world. All-gather: for world − 1 more steps, pass the finished
    chunks around the ring, overwriting. Every rank sends 2·(world − 1)/world
    of the data, however many ranks there are.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    if world == 1:
        return t
    flat = t.view(-1)
    chunks = list(torch.tensor_split(flat, world))
    right, left = (rank + 1) % world, (rank - 1) % world
    for step in range(world - 1):
        send_i = (rank - step) % world
        recv_i = (rank - step - 1) % world
        buf = torch.empty_like(chunks[recv_i])
        req = dist.isend(chunks[send_i].clone(), dst=right)
        dist.recv(buf, src=left)
        req.wait()
        chunks[recv_i] += buf
    for step in range(world - 1):
        send_i = (rank + 1 - step) % world
        recv_i = (rank - step) % world
        buf = torch.empty_like(chunks[recv_i])
        req = dist.isend(chunks[send_i].clone(), dst=right)
        dist.recv(buf, src=left)
        req.wait()
        chunks[recv_i].copy_(buf)
    return t


def bytes_sent_per_rank(n_bytes, world, algorithm="ring"):
    """Bytes each rank sends to all-reduce ``n_bytes``. Naive: rank 0 sends
    (world − 1)·n (the bottleneck); ring: 2·(world − 1)/world·n."""
    if algorithm == "naive":
        return (world - 1) * n_bytes
    return 2 * (world - 1) / world * n_bytes


def all_reduce_worker(rank, world, n, algorithm):
    """Test helper: each rank holds arange(n) * (rank + 1); returns the
    all-reduced tensor."""
    t = torch.arange(n, dtype=torch.float64) * (rank + 1)
    (ring_all_reduce if algorithm == "ring" else naive_all_reduce)(t)
    return t
