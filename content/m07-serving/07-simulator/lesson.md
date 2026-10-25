---
id: "7.7"
title: A serving simulator
concepts: [serving, preemption, capacity-planning]
file: m07/simulator.py
stub: stubs/simulator.py
tests: test_simulator.py
predict:
  kind: choice
  prompt: >
    You halve the KV memory of a continuous-batching server that's running
    near capacity, keeping everything else the same. What happens?
  options:
    - latency and throughput are unchanged; memory only matters for model weights
    - sequences get preempted and recomputed, so latency rises and work is wasted
    - the server rejects half the requests
    - throughput doubles because batches are smaller
  answer: 1
mutants:
  - name: admits without checking memory
    file: m07/simulator.py
    replace: [" and alloc.can_allocate(queue[0].prompt + 1)):", "):"]
  - name: decode steps are free
    file: m07/simulator.py
    replace: ["dt = cost.step(sum(r.prompt for r in admitted), len(decoding))", "dt = cost.step(sum(r.prompt for r in admitted), 0)"]
---

## Hook

Put it together: continuous batching (7.4) on paged KV memory (7.5). When
memory runs out mid-generation, something has to give. This is the core
loop of vLLM and similar servers, and with a simulator you can answer
capacity questions without a GPU: how many users can this deployment take
before latency blows up?

## Build

In `m07/simulator.py`:

1. `simulate(requests, max_batch, n_blocks, block_size, cost)`: each
   iteration, first make room for every running sequence's next token,
   preempting the most recently admitted sequence (release its blocks,
   send it back to the front of the queue) when no block is free; then
   admit while both batch slots and blocks allow; then charge the
   iteration and retire finished requests. Record spans (queued, prefill,
   decode) per request and memory and batch size per iteration.
2. `timeline_parts(result)`: the arguments for `show_timeline`.

```python
from workbench import show_timeline
rng = np.random.default_rng(0)
reqs = [Request(i, float(i * 0.02), int(rng.integers(20, 300)), int(rng.integers(10, 400))) for i in range(60)]
for blocks in [400, 150, 80]:
    r = simulate(reqs, 32, blocks, 16)
    show_timeline(title=f"{blocks} KV blocks", **timeline_parts(r))
```

## Explain

With less memory, running sequences outgrow it and get **preempted: their
KV cache is thrown away and recomputed later, so latency rises and work is
wasted**, while the batch shrinks and throughput falls. The timeline makes
it visible: decode spans cut short, queued again, prefilled again.

vLLM recomputes; the alternative is swapping the blocks to CPU memory and
back, which wins for long sequences where recomputation is expensive.
Admission policy matters as much: admitting less aggressively avoids
preemption storms, and knowing (or predicting) output lengths lets a
scheduler pack sequences that will fit.

This simulator ignores many things a real one models: prefix sharing,
chunked prefill, multiple GPUs, network time. But it already explains the
shape of the latency-versus-load curves serving teams plan with.

## Stretch

Sweep the arrival rate and plot p50 and p99 latency against throughput for
two memory sizes. Where's the knee? Then add the swap policy (preempted
blocks move to a CPU pool and come back at a cost per block) and compare.
