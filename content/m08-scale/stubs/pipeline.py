"""Pipeline parallelism: layers split into stages on different devices, and
micro-batches to keep the stages busy.

Lesson 8.4."""

import torch
import torch.distributed as dist


def gpipe_schedule(stages, microbatches, t_forward=1.0, t_backward=2.0):
    """GPipe (Huang et al., 2019): all forwards, then all backwards. Stage s
    runs forward of micro-batch m when it has finished m − 1 and stage s − 1
    has finished m; backwards run in reverse, from the last stage down.
    Returns spans (stage, start, end, label, kind)."""
    raise NotImplementedError("lesson 8.4")


def makespan(spans):
    raise NotImplementedError("lesson 8.4")


def bubble_fraction(spans, stages):
    """Idle stage-time over total stage-time."""
    raise NotImplementedError("lesson 8.4")


def ideal_bubble(stages, microbatches):
    """(p − 1) / (m + p − 1)."""
    raise NotImplementedError("lesson 8.4")


def pipeline_worker(rank, world, microbatches=4, seed=0, n_embd=8, batch=8):
    """Test helper: a stack of ``world`` linear+tanh layers, stage r on rank
    r. Micro-batches flow forward with send/recv; the last rank returns the
    output, which must equal running all layers on one process."""
    torch.manual_seed(seed)
    layers = [torch.nn.Linear(n_embd, n_embd).double() for _ in range(world)]
    x = torch.randn(batch, n_embd, dtype=torch.float64)
    full = x
    for layer in layers:
        full = torch.tanh(layer(full))
    mine = layers[rank]
    outs = []
    with torch.no_grad():
        for mb in torch.chunk(x, microbatches):
            if rank == 0:
                h = mb
            else:
                h = torch.empty_like(mb)
                dist.recv(h, src=rank - 1)
            h = torch.tanh(mine(h))
            if rank < world - 1:
                dist.send(h, dst=rank + 1)
            else:
                outs.append(h)
    return (torch.cat(outs), full) if rank == world - 1 else None
