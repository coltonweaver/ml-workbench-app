---
id: "8.6"
title: Failures, checkpoints and stragglers
concepts: [checkpointing, fault-tolerance, stragglers, goodput]
file: m08/fault.py
stub: stubs/fault.py
tests: test_failures.py
predict:
  kind: number
  prompt: >
    A job fails on average every 10 hours. A checkpoint takes 1 minute to
    write and restarting takes 10 minutes. On each failure you also redo, on
    average, half an interval of work. About how many minutes between
    checkpoints maximize useful work?
  answer: 34
  tolerance: 4
mutants:
  - name: forgets the dropout RNG
    file: m08/fault.py
    replace: ["    torch.set_rng_state(state[\"torch_rng\"])\n", ""]
  - name: backup workers off by one
    file: m08/fault.py
    replace: ["return t[len(t) - 1 - backups]", "return t[len(t) - backups]"]
  - name: stragglers by the mean
    file: m08/fault.py
    replace: ["per_rank = np.median(t, axis=1)", "per_rank = np.mean(t, axis=1)"]
warmups:
  - kind: number
    prompt: "Synchronous training on 64 workers. One worker suddenly runs at half speed. What fraction of the healthy throughput is left?"
    answer: 0.5
    tolerance: 0.02
    explain: "Every step waits for the slowest worker, so the whole job runs at its speed. Backup workers or dropping the slowest gradient fix this."
---

## Hook

At scale, failure is the normal case. Meta's Llama 3 report (2024) counted
419 unexpected interruptions in 54 days on 16,000 GPUs, roughly one every
three hours. And in synchronous training, one slow GPU slows everyone.
You need a run that can stop at any moment and resume exactly, and a way
to keep stragglers from setting the pace.

## Build

In `m08/fault.py`:

1. `save_checkpoint(path, model, opt, step, rng)`: model, optimizer
   (Adam's moments!), step, NumPy and torch RNG states, written to a
   temporary file and then `os.replace`d, so a crash mid-write can't
   corrupt the last good checkpoint.
2. `load_checkpoint(...)`.
3. `train(path, steps, every, crash_at)`: resumes if a checkpoint exists.
   The test crashes it at step 17, resumes, and demands bit-identical
   weights to an uninterrupted run.
4. `sync_step_times`, `with_backup_workers(times, backups)` and
   `stragglers(times, factor)`.
5. `expected_goodput(mtbf_hours, checkpoint_minutes, interval_minutes,
   restart_minutes)`.

```python
from workbench import series, show_plot
intervals = np.arange(2, 240)
show_plot([series("goodput", intervals, [expected_goodput(10, 1, i, 10) for i in intervals])],
          x_label="minutes between checkpoints", y_label="useful fraction")
```

## Explain

Checkpointing often wastes time writing; rarely wastes time redoing work.
The optimum is close to Young's formula, √(2 · checkpoint cost · MTBF) =
√(2 · 1 · 600) ≈ **35 minutes**; this model's exact optimum is 33.
Faster checkpoints (asynchronous writes to CPU memory first, then disk)
move it lower and raise goodput.

Bit-exact resume needs everything that affects the future: optimizer
state, the data order (the NumPy RNG here), and dropout's RNG. Miss the
last one and the run continues fine but differently, a bug you can only
catch by comparing against an uninterrupted run, as the test does.

For stragglers, medians beat means: one slow step (a GC pause) shouldn't
flag a rank, but a consistently slow one (a thermally throttled GPU, a
bad link) should. Backup workers trade a little data per step for never
waiting on the slowest rank.

## Stretch

Simulate a 1000-rank job where each rank independently has a 1% chance of
a 3× slow step. What fraction of synchronous steps are slow? How many
backup workers make it rare?
