---
id: "7.1"
title: What a token costs
concepts: [decoding, flops, memory-bandwidth, roofline]
file: m07/decoding.py
stub: stubs/decoding.py
tests: test_decoding.py
predict:
  kind: number
  prompt: >
    A 7-billion-parameter model in 16-bit weights runs on a GPU with 1 TB/s
    of memory bandwidth and 1000 TFLOP/s of compute. Generating for one
    user, each token must read every weight once. About how many tokens per
    second can it produce?
  answer: 71
  tolerance: 4
mutants:
  - name: forgets attention
    file: m07/decoding.py
    replace: ["return 2 * n_params + 4 * n_layer * context * n_embd", "return 2 * n_params"]
  - name: adds compute and memory time
    file: m07/decoding.py
    replace: ["return max(compute, memory)", "return compute + memory"]
  - name: never crops the context
    file: m07/decoding.py
    replace: ["logits = model(idx[:, -model.block_size:])", "logits = model(idx)"]
---

## Hook

Training processes a whole batch of sequences in parallel. Generation
can't: token t + 1 depends on token t. So serving a model is a different
engineering problem from training it, and its bottleneck is surprising:
not arithmetic, but reading the weights out of memory.

## Build

In `m07/decoding.py`:

1. `generate_naive(model, idx, n)`: greedy decoding that reruns the full
   (cropped) context every step. You'll replace it next lesson.
2. `forward_flops(n_params, n_layer, n_embd, context)`: 2 FLOPs per
   parameter plus attention's 4·n_layer·context·n_embd.
3. `naive_generation_flops` and `cached_generation_flops`: the total work
   with and without reprocessing old tokens.
4. `arithmetic_intensity(batch, bytes_per_param)`, `step_time(...)` (the
   roofline: the slower of compute and memory), and `critical_batch(...)`.

```python
import time
from workbench import series, show_plot
for n in [16, 32, 64, 128]:
    t = time.perf_counter(); generate_naive(m, prompt, n); print(n, time.perf_counter() - t)
B = np.arange(1, 2049)
show_plot([series("step time", B, [step_time(7e9, b, 2, 1e15, 1e12) for b in B])],
          log_x=True, x_label="batch", y_label="seconds per step")
```

## Explain

The weights are 14 GB. At 1 TB/s, reading them takes 14 ms, while the
arithmetic for one token (14 GFLOP) takes 14 µs. So one user gets about
1 / 0.014 ≈ **71 tokens/s**, and the GPU's compute sits 99.9% idle.

That's arithmetic intensity: decoding does 2 FLOPs per weight per sequence,
so at batch 1 in 16-bit it's 1 FLOP per byte, while the hardware can do
1000 per byte. Until the batch reaches the **critical batch size** (here
1000), more sequences cost almost nothing extra: they ride along on the
same weight reads. Nearly everything in serving follows from this: batch
as many requests as possible (7.4), keep their KV caches small enough to
fit (7.2, 7.5), make the weights smaller (7.3), or get more tokens out of
each weight read (7.6).

Naive generation also redoes all past work every step: total cost grows
with the square of the length. The KV cache removes that.

## Stretch

Time `generate_naive` for growing n and fit the curve: is it quadratic?
Then measure step time for batch 1, 8, 64 on MPS. Where does your Mac's
critical batch seem to be?
