"""A serving simulator: continuous batching on top of paged KV memory, with
preemption when memory runs out."""

from dataclasses import dataclass, field

from m07.paged import BlockAllocator
from m07.scheduler import Cost, Request, _copy, metrics


@dataclass
class Result:
    requests: list
    spans: list = field(default_factory=list)       # (request id, start, end, kind)
    memory: list = field(default_factory=list)      # (time, blocks in use)
    batch: list = field(default_factory=list)       # (time, running sequences)
    preemptions: int = 0


def simulate(requests, max_batch, n_blocks, block_size, cost=None):
    """Run ``requests`` (scheduler.Request) through a continuous-batching
    engine with ``n_blocks`` KV blocks of ``block_size`` tokens.

    Each iteration:
      1. every running sequence needs a slot for its next token
         (allocator.append). If none is free, preempt the most recently
         admitted running sequence: release its blocks and put it back at
         the front of the queue to start over (vLLM's recompute policy).
         If it's the only one running, it can never fit: MemoryError.
      2. admit waiting, arrived requests in order while the batch has room
         and their prompt + 1 tokens fit in free blocks.
      3. charge cost.step(admitted prompt tokens, decoding sequences); the
         admitted get their first token, the others one more token.
      4. retire finished sequences and release their blocks.

    Records spans per request: "queued" (arrival or preemption to
    admission), "prefill" (its admission iteration), "decode" (to finish or
    preemption); and memory and batch size after every iteration.
    """
    cost = cost or Cost()
    alloc = BlockAllocator(n_blocks, block_size)
    queue, running, done = _copy(requests), [], []
    waiting_since = {r.id: r.arrival for r in queue}
    decode_start = {}
    res = Result(requests=[])
    t = 0.0
    while queue or running:
        if not running:
            t = max(t, queue[0].arrival)
        # 1. room for every running sequence's next token
        i = 0
        while i < len(running):
            r = running[i]
            try:
                alloc.append(r.id)
                i += 1
            except MemoryError:
                if len(running) == 1:
                    raise MemoryError(f"request {r.id} alone outgrows {n_blocks} blocks") from None
                victim = running.pop()               # most recently admitted
                alloc.release(victim.id)
                res.spans.append((victim.id, decode_start.pop(victim.id), t, "decode"))
                victim.generated, victim.first_token = 0, None
                queue.insert(0, victim)
                waiting_since[victim.id] = t
                res.preemptions += 1
                if victim is r:
                    break
        # 2. admission
        admitted = []
        while (queue and queue[0].arrival <= t and len(running) + len(admitted) < max_batch
               and alloc.can_allocate(queue[0].prompt + 1)):
            r = queue.pop(0)
            alloc.allocate(r.id, r.prompt + 1)
            res.spans.append((r.id, waiting_since.pop(r.id), t, "queued"))
            admitted.append(r)
        if not admitted and not running:
            if queue and queue[0].arrival <= t:
                raise MemoryError(f"request {queue[0].id} can never fit in {n_blocks} blocks")
            continue
        # 3. one iteration
        decoding = list(running)
        dt = cost.step(sum(r.prompt for r in admitted), len(decoding))
        for r in admitted:
            res.spans.append((r.id, t, t + dt, "prefill"))
            decode_start[r.id] = t + dt
        t += dt
        for r in admitted:
            r.first_token, r.generated = t, 1
        for r in decoding:
            r.generated += 1
        # 4. retire
        running = []
        for r in decoding + admitted:
            if r.generated >= r.output:
                r.finish = t
                alloc.release(r.id)
                res.spans.append((r.id, decode_start.pop(r.id), t, "decode"))
                done.append(r)
            else:
                running.append(r)
        res.memory.append((t, n_blocks - alloc.n_free))
        res.batch.append((t, len(running)))
    res.requests = sorted(done, key=lambda r: r.id)
    return res


def timeline_parts(result):
    """Arguments for workbench.show_timeline: one lane per request, its
    spans colored by kind, headline metrics, and memory and batch size
    plotted below."""
    from workbench import series, span
    ids = [r.id for r in result.requests]
    lane = {rid: i for i, rid in enumerate(ids)}
    m = metrics(result)
    m["preemptions"] = result.preemptions
    times = [t for t, _ in result.memory]
    return {
        "lanes": [f"req {rid}" for rid in ids],
        "spans": [span(lane[rid], s, e, None, kind) for rid, s, e, kind in result.spans if e > s],
        "metrics": m,
        "below": [series("KV blocks in use", times, [b for _, b in result.memory]),
                  series("running sequences", times, [b for _, b in result.batch])],
        "unit": "s",
    }
