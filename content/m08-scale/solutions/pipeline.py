"""Pipeline parallelism: layers split into stages on different devices, and
micro-batches to keep the stages busy."""

import torch
import torch.distributed as dist


def gpipe_schedule(stages, microbatches, t_forward=1.0, t_backward=2.0):
    """GPipe (Huang et al., 2019): all forwards, then all backwards. Stage s
    runs forward of micro-batch m when it has finished m − 1 and stage s − 1
    has finished m; backwards run in reverse, from the last stage down.
    Returns spans (stage, start, end, label, kind)."""
    spans = []
    f_end = {}
    for m in range(microbatches):
        for s in range(stages):
            start = max(f_end.get((s, m - 1), 0.0), f_end.get((s - 1, m), 0.0))
            f_end[(s, m)] = start + t_forward
            spans.append((s, start, start + t_forward, f"F{m}", "forward"))
    all_forward = max(f_end.values())
    b_end = {}
    for m in reversed(range(microbatches)):
        for s in reversed(range(stages)):
            start = max(b_end.get((s, m + 1), all_forward if s == stages - 1 else 0.0),
                        b_end.get((s + 1, m), 0.0), f_end[(s, m)])
            b_end[(s, m)] = start + t_backward
            spans.append((s, start, start + t_backward, f"B{m}", "backward"))
    return spans


def makespan(spans):
    return max(e for _, _, e, _, _ in spans) - min(s for _, s, _, _, _ in spans)


def bubble_fraction(spans, stages):
    """Idle stage-time over total stage-time."""
    busy = sum(e - s for _, s, e, _, _ in spans)
    return 1 - busy / (makespan(spans) * stages)


def ideal_bubble(stages, microbatches):
    """(p − 1) / (m + p − 1)."""
    return (stages - 1) / (microbatches + stages - 1)


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
