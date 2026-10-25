"""Data parallelism: every rank holds the whole model and a slice of each
batch, and gradients are averaged before every step."""

import numpy as np
import torch
import torch.distributed as dist

from m08.collectives import ring_all_reduce


def make_model(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)).double()


def make_data(seed, n=512):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = np.sin(X @ rng.normal(size=8))[:, None]
    return torch.tensor(X), torch.tensor(y)


def shard(batch_indices, rank, world):
    """This rank's part of a global batch: every world-th index starting at
    ``rank``."""
    return batch_indices[rank::world]


def average_gradients(model):
    """All-reduce every gradient and divide by the world size."""
    world = dist.get_world_size()
    for p in model.parameters():
        ring_all_reduce(p.grad)
        p.grad /= world


def global_batches(n, batch_size, steps, seed):
    rng = np.random.default_rng(seed)
    return [rng.choice(n, size=batch_size, replace=False) for _ in range(steps)]


def ddp_worker(rank, world, steps=20, batch_size=64, lr=0.05, seed=0):
    """Train with data parallelism; return the final parameters (as a list
    of tensors) and this rank's number of examples seen."""
    model = make_model(seed)
    X, y = make_data(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    seen = 0
    for b in global_batches(len(X), batch_size, steps, seed):
        mine = shard(b, rank, world)
        seen += len(mine)
        opt.zero_grad()
        # sum over my examples / global batch size: the average after the all-reduce sum
        loss = ((model(X[mine]) - y[mine]) ** 2).sum() / (len(b) / world)
        loss.backward()
        average_gradients(model)
        opt.step()
    return [p.detach().clone() for p in model.parameters()], seen


def single_process(steps=20, batch_size=64, lr=0.05, seed=0):
    """The same training on one process with the whole batch: the reference."""
    model = make_model(seed)
    X, y = make_data(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for b in global_batches(len(X), batch_size, steps, seed):
        opt.zero_grad()
        ((model(X[b]) - y[b]) ** 2).mean().backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]
