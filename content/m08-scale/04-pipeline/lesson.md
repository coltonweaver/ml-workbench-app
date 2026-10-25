---
id: "8.4"
title: Pipeline parallelism
concepts: [pipeline-parallelism, micro-batches, pipeline-bubble]
file: m08/pipeline.py
stub: stubs/pipeline.py
tests: test_pipeline.py
predict:
  kind: number
  prompt: >
    A model is split into 4 pipeline stages and each batch into 4
    micro-batches, GPipe style. What fraction of the stages' time is idle
    (the bubble)? Two decimals.
  answer: 0.43
  tolerance: 0.01
mutants:
  - name: backward doesn't wait for the next stage's gradient
    file: m08/pipeline.py
    replace: ["b_end.get((s + 1, m), 0.0), f_end[(s, m)])", "f_end[(s, m)])"]
  - name: bubble formula without the fill
    file: m08/pipeline.py
    replace: ["return (stages - 1) / (microbatches + stages - 1)", "return (stages - 1) / microbatches"]
warmups:
  - kind: number
    prompt: "GPipe with 2 stages and 3 micro-batches. What fraction of stage-time is bubble?"
    answer: 0.25
    tolerance: 0.02
    explain: "(p − 1) / (m + p − 1) = 1 / 4. More micro-batches shrink it."
---

## Hook

Tensor parallelism communicates every layer. Pipeline parallelism cuts
the model into consecutive groups of layers (stages) on different devices
and only sends activations at the cuts. The catch: stage 2 can't start
until stage 1 is done. Split the batch into micro-batches and the stages
can overlap, like an assembly line.

## Build

In `m08/pipeline.py`:

1. `gpipe_schedule(stages, microbatches, t_forward, t_backward)` (Huang et
   al., "GPipe", 2019): compute when each stage runs each micro-batch's
   forward, then the backwards in reverse. Return spans for
   `show_timeline`.
2. `makespan`, `bubble_fraction(spans, stages)` and `ideal_bubble(p, m)`.
3. `pipeline_worker` (given) runs a real pipeline forward across
   processes with send/recv; read it and run it.

```python
from workbench import show_timeline, span
spans = gpipe_schedule(4, 8)
show_timeline([f"stage {s}" for s in range(4)], [span(*x) for x in spans],
              title="GPipe, 4 stages, 8 micro-batches", unit="step",
              metrics={"bubble": bubble_fraction(spans, 4)})
```

## Explain

With p stages and m micro-batches, the pipeline takes p − 1 slots to fill
and as many to drain, so the bubble is (p − 1)/(m + p − 1) = 3/7 ≈
**0.43** for p = m = 4. Make m much larger than p and it shrinks: 64
micro-batches on 4 stages leave under 5% idle.

But GPipe keeps every micro-batch's activations until its backward, so
memory grows with m. The 1F1B schedule (PipeDream, Narayanan et al., 2019)
starts each micro-batch's backward as soon as possible and alternates one
forward with one backward, capping in-flight activations at p while
keeping the same bubble. Interleaved schedules (several smaller stages
per device) shrink the bubble further at the cost of more communication.

## Stretch

Write `one_f_one_b_schedule(stages, microbatches)` and a function counting
the peak number of micro-batches whose activations each stage holds at
once. Compare with GPipe on the timeline.
