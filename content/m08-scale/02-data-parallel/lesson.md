---
id: "8.2"
title: Data parallelism
concepts: [data-parallelism, gradient-averaging, ddp]
file: m08/data_parallel.py
stub: stubs/data_parallel.py
tests: test_data_parallel.py
predict:
  kind: choice
  prompt: >
    Four ranks each compute gradients on a quarter of a 64-example batch,
    average them with all-reduce, and take the same SGD step. Compared with
    one process training on the whole 64-example batch, the result is:
  options:
    - noisier, because each rank sees fewer examples
    - identical, up to floating-point rounding
    - better, because there are four models
    - different, because the ranks drift apart
  answer: 1
mutants:
  - name: sums instead of averaging
    file: m08/data_parallel.py
    replace: ["        p.grad /= world", "        pass"]
  - name: every rank takes the whole batch
    file: m08/data_parallel.py
    replace: ["return batch_indices[rank::world]", "return batch_indices"]
  - name: no communication
    file: m08/data_parallel.py
    replace: ["        ring_all_reduce(p.grad)\n", ""]
warmups:
  - kind: choice
    prompt: "Data parallelism where each rank averages its gradient with the others: it matches a single process trained on…"
    options: ["Each rank's own shard", "The whole global batch (all the shards together)", "The global batch with a smaller learning rate"]
    answer: 1
    explain: "The mean of the per-shard mean gradients is the gradient of the mean loss over the global batch (with equal shard sizes)."
---

## Hook

The simplest way to train faster: copy the model to every device, give
each a different slice of the batch, and average the gradients before
stepping. Every rank then takes the same step, and the replicas stay
identical without ever sending weights.

## Build

In `m08/data_parallel.py` (the model, data and batches are given):

1. `shard(batch_indices, rank, world)`.
2. `average_gradients(model)`: your `ring_all_reduce` on every gradient,
   then divide by the world size.
3. `ddp_worker(rank, world, …)`: per step, compute the loss on your shard
   scaled so that after averaging it's the global mean, backward, average,
   step. Return the final parameters and how many examples this rank saw.

```python
from workbench.dist import launch
results = launch(ddp_worker, 4, (20,))
ref = single_process(20)
max((a - b).abs().max() for a, b in zip(results[0][0], ref))
```

## Explain

The gradient of a mean over 64 examples is the mean of the four
16-example gradients, so the averaged step is **identical, up to rounding**
(the test checks to 10⁻¹⁰). Data parallelism changes nothing about the
optimization, only where the arithmetic happens. That's its appeal, and
its limit: the global batch grows with the number of ranks, and past the
critical batch size (McCandlish et al., 2018) bigger batches stop buying
faster progress.

PyTorch's DistributedDataParallel overlaps the all-reduce with the
backward pass: gradients are grouped into buckets and each bucket is sent
as soon as it's ready, so communication hides behind computation.

Every rank holds the full model, gradients and optimizer state (for Adam,
16 bytes per parameter in mixed precision). For a 7B model that's 112 GB
per GPU before activations, which no GPU has. Hence the next two lessons,
and ZeRO.

## Stretch

Implement ZeRO stage 1: each rank keeps Adam's moments for only 1/world of
the parameters. Reduce-scatter the gradients, update your shard, and
all-gather the parameters. Check it still matches the single process.
