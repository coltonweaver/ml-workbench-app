---
id: "8.1"
title: All-reduce around a ring
concepts: [collectives, all-reduce, ring-algorithm, bandwidth]
file: m08/collectives.py
stub: stubs/collectives.py
tests: test_collectives.py
predict:
  kind: number
  prompt: >
    Eight GPUs each hold 1 GB of gradients and need the sum. With the ring
    algorithm, how many GB does each GPU send in total?
  answer: 1.75
  tolerance: 0.01
mutants:
  - name: reduce-scatter overwrites instead of adding
    file: m08/collectives.py
    replace: ["chunks[recv_i] += buf", "chunks[recv_i].copy_(buf)"]
  - name: skips the all-gather
    file: m08/collectives.py
    replace: ["    for step in range(world - 1):\n        send_i = (rank + 1 - step) % world", "    for step in range(0):\n        send_i = (rank + 1 - step) % world"]
  - name: naive cost ignores the bottleneck
    file: m08/collectives.py
    replace: ["return (world - 1) * n_bytes", "return n_bytes"]
---

## Hook

Every form of parallel training is held together by a few collective
operations, and the most important is all-reduce: every rank contributes a
tensor, every rank gets the sum. Done naively, one rank becomes a
bottleneck. The ring algorithm (Patarasuk and Yuan, 2009; popularized for
deep learning by Baidu in 2017 and built into NCCL) makes the cost per
rank almost independent of the number of ranks.

This module runs real multi-process jobs on your Mac. `workbench.dist.launch(fn,
world_size, args)` starts the processes, connects them with
`torch.distributed` (gloo backend), runs `fn(rank, world_size, *args)` in
each, and returns the results. Your functions use `dist.send`,
`dist.recv` and `dist.isend`, exactly as on a cluster.

## Build

In `m08/collectives.py`:

1. `naive_all_reduce(t)`: everyone sends to rank 0; rank 0 sums and sends
   the result back.
2. `ring_all_reduce(t)`: split t into `world` chunks. **Reduce-scatter**:
   world − 1 steps, each sending one chunk to the right neighbor and adding
   the one received from the left. **All-gather**: world − 1 more steps,
   passing the finished chunks around. Use `dist.isend` for the send so the
   ring doesn't deadlock with everyone blocked in `send`.
3. `bytes_sent_per_rank(n_bytes, world, algorithm)`.

```python
from workbench.dist import launch
from m08.collectives import all_reduce_worker
launch(all_reduce_worker, 4, (10, "ring"))
```

## Explain

Each rank sends world − 1 chunks of size n/world in each phase:
2·(world − 1)/world·n = 2·7/8 GB = **1.75 GB**. With 1000 ranks it's still
under 2n. The naive version makes rank 0 send (world − 1)·n, 7 GB here,
while the others wait. With the ring every link is busy all the time,
which is why it's bandwidth-optimal.

Reduce-scatter and all-gather are useful on their own: ZeRO and FSDP
(8.2's stretch) shard optimizer state and weights so that each rank only
ever holds the reduce-scattered part, and all-gather the weights layer by
layer when they're needed. Real clusters add hierarchy (fast links inside
a node, slower between nodes) and tree algorithms for small messages,
where the ring's 2·(world − 1) latency steps dominate.

## Stretch

Time `ring_all_reduce` and `naive_all_reduce` for 10⁶ floats on 2, 4 and 8
processes. Does the naive one get slower with more ranks while the ring
stays flat? (On one machine the "network" is shared memory, so the
effect is weaker than on a cluster.)
