"""Run a function on several local processes that talk through
torch.distributed (the gloo backend, on CPU), to simulate a multi-GPU job
on one machine.

    from workbench.dist import launch
    results = launch(my_worker, world_size=4, args=(config,))

``my_worker(rank, world_size, *args)`` must be defined in a module (not in
the notebook or a cell), because each process imports it fresh. Whatever
it returns comes back in a list ordered by rank.
"""

from __future__ import annotations

import os
import pickle
import socket
import traceback
from collections.abc import Callable, Sequence
from typing import Any


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(rank: int, world_size: int, port: int, fn: Callable[..., Any], args: Sequence[Any],
           queue: Any, threads: int) -> None:
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(threads)
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        result = fn(rank, world_size, *args)
        # plain pickle copies tensor data; torch's queue would share memory
        # that disappears when this process exits
        queue.put((rank, "ok", pickle.dumps(result)))
    except BaseException:  # noqa: BLE001 - report every failure to the parent
        queue.put((rank, "error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch(fn: Callable[..., Any], world_size: int, args: Sequence[Any] = (), timeout: float = 120,
           threads: int = 1) -> list[Any]:
    """Run ``fn(rank, world_size, *args)`` on ``world_size`` processes and
    return their results by rank. Raises RuntimeError with the first
    failing rank's traceback."""
    import torch.multiprocessing as mp

    # gloo tries IPv6 first and logs a warning per rank when it's missing
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_entry, args=(r, world_size, port, fn, tuple(args), queue, threads))
             for r in range(world_size)]
    for p in procs:
        p.start()
    results: dict[int, Any] = {}
    try:
        for _ in range(world_size):
            rank, status, value = queue.get(timeout=timeout)
            if status == "error":
                raise RuntimeError(f"rank {rank} failed:\n{value}")
            results[rank] = pickle.loads(value)
    finally:
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    return [results[r] for r in range(world_size)]
