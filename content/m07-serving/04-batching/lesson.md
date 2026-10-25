---
id: "7.4"
title: Continuous batching
concepts: [batching, scheduling, ttft, throughput]
file: m07/scheduler.py
stub: stubs/scheduler.py
tests: test_batching.py
predict:
  kind: choice
  prompt: >
    Requests arrive with very different output lengths. A static batch
    waits until its longest request finishes before starting the next
    batch. Continuous batching retires each request as soon as it's done
    and fills its slot immediately. What improves?
  options:
    - only throughput
    - only time to first token
    - both, because slots aren't left idle and new requests don't wait for stragglers
    - neither; the GPU does the same work
  answer: 2
mutants:
  - name: TTFT from time zero
    file: m07/scheduler.py
    replace: ["ttft = sum(r.first_token - r.arrival for r in rs) / len(rs)", "ttft = sum(r.first_token for r in rs) / len(rs)"]
  - name: decoding is free
    file: m07/scheduler.py
    replace: ["return self.base + self.per_token * (prefill_tokens + decode_seqs)", "return self.base + self.per_token * prefill_tokens"]
  - name: two tokens per iteration
    file: m07/scheduler.py
    replace: ["        for r in decoding:\n            r.generated += 1\n        running = []", "        for r in decoding:\n            r.generated += 2\n        running = []"]
---

## Hook

From 7.1: up to the critical batch size, extra sequences are nearly free.
So batch. But requests arrive at random times and want different numbers
of tokens. How you batch them decides both how many tokens per second the
server produces and how long each user waits.

## Build

In `m07/scheduler.py`, a discrete-event simulation (no model, just time):

1. `Request` and `Cost.step(prefill_tokens, decode_seqs)`: an iteration
   costs a fixed overhead plus a per-token amount.
2. `static_batching(requests, batch_size, cost)`: take up to batch_size
   waiting requests, prefill together, decode until all are done.
3. `continuous_batching(requests, max_batch, cost)` (Yu et al., "Orca",
   2022): every iteration, admit arrived requests while there is room,
   decode one token for the rest, retire finished ones immediately.
4. `metrics(trace)`: mean time to first token (TTFT), time per output
   token (TPOT), latency and throughput.

```python
from workbench import series, show_plot
rng = np.random.default_rng(0)
reqs = [Request(i, float(rng.exponential(0.05) * i), int(rng.integers(10, 200)), int(rng.integers(5, 300)))
        for i in range(200)]
for B in [4, 16, 64]:
    print(B, metrics(static_batching(reqs, B, Cost())), metrics(continuous_batching(reqs, B, Cost())))
```

## Explain

**Both improve.** In a static batch, a request that finishes early leaves
its slot empty until the longest one is done, and arriving requests queue
behind the whole batch. Continuous batching refills slots every iteration:
utilization stays high (throughput) and new requests start within one
iteration (TTFT). Orca reported up to 37× throughput over static batching
on real length distributions.

The cost is that prefill and decode now share iterations: a long prompt
admitted into a running batch makes that iteration slow for everyone
already decoding (a TPOT spike). Chunked prefill (Sarathi-Serve) splits
long prompts across iterations; some systems run prefill and decode on
separate GPUs entirely (DistServe, "disaggregated serving").

## Stretch

Add a token budget per iteration (max prefill + decode tokens) and chunk
prompts that exceed it. Plot the TPOT distribution with and without
chunking for a workload with a few very long prompts.
