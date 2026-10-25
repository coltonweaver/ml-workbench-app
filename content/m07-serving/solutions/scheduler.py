"""Batching requests: static batches versus continuous batching."""

from dataclasses import dataclass, field


@dataclass(eq=False)
class Request:
    id: int
    arrival: float
    prompt: int      # prompt tokens
    output: int      # tokens to generate (≥ 1)
    first_token: float | None = None
    finish: float | None = None
    generated: int = 0


@dataclass
class Cost:
    """Time for one engine iteration: ``base`` plus ``per_token`` for every
    token processed (all prefill tokens, one per decoding sequence)."""
    base: float = 0.01
    per_token: float = 0.0005

    def step(self, prefill_tokens, decode_seqs):
        return self.base + self.per_token * (prefill_tokens + decode_seqs)


@dataclass
class Trace:
    """What happened: an entry (time, request ids, prefill tokens, decoding
    count, duration) per iteration, and the finished requests."""
    steps: list = field(default_factory=list)
    requests: list = field(default_factory=list)


def _copy(requests):
    return sorted((Request(r.id, r.arrival, r.prompt, r.output) for r in requests), key=lambda r: r.arrival)


def static_batching(requests, batch_size, cost):
    """Wait for up to ``batch_size`` requests (whatever has arrived when the
    engine is free, or the next arrival), prefill them together, decode
    until the longest finishes, and only then start the next batch."""
    queue, done, t, trace = _copy(requests), [], 0.0, Trace()
    while queue:
        t = max(t, queue[0].arrival)
        batch = [r for r in queue if r.arrival <= t][:batch_size]
        queue = [r for r in queue if r not in batch]
        dt = cost.step(sum(r.prompt for r in batch), 0)
        trace.steps.append((t, [r.id for r in batch], sum(r.prompt for r in batch), 0, dt))
        t += dt
        for r in batch:
            r.first_token, r.generated = t, 1
            if r.output == 1:
                r.finish = t
        while any(r.generated < r.output for r in batch):
            # every sequence in the batch occupies a slot until the batch ends
            dt = cost.step(0, len(batch))
            trace.steps.append((t, [r.id for r in batch], 0, len(batch), dt))
            t += dt
            for r in batch:
                if r.generated < r.output:
                    r.generated += 1
                    if r.generated == r.output:
                        r.finish = t
        done += batch
    trace.requests = sorted(done, key=lambda r: r.id)
    return trace


def continuous_batching(requests, max_batch, cost):
    """Every iteration: admit arrived requests while there's room (their
    prefill runs in this iteration and yields their first token), decode one
    token for every other running request, and retire finished requests
    immediately so their slots free up for the next iteration."""
    queue, running, done, t, trace = _copy(requests), [], [], 0.0, Trace()
    while queue or running:
        if not running:
            t = max(t, queue[0].arrival)
        admitted = []
        while queue and queue[0].arrival <= t and len(running) + len(admitted) < max_batch:
            admitted.append(queue.pop(0))
        decoding = list(running)
        dt = cost.step(sum(r.prompt for r in admitted), len(decoding))
        trace.steps.append((t, [r.id for r in admitted + decoding], sum(r.prompt for r in admitted),
                            len(decoding), dt))
        t += dt
        for r in admitted:
            r.first_token, r.generated = t, 1
        for r in decoding:
            r.generated += 1
        running = []
        for r in decoding + admitted:
            if r.generated >= r.output:
                r.finish = t
                done.append(r)
            else:
                running.append(r)
    trace.requests = sorted(done, key=lambda r: r.id)
    return trace


def metrics(trace):
    """{"ttft": mean time to first token (from arrival), "tpot": mean time
    per output token after the first, "latency": mean arrival-to-finish,
    "throughput": output tokens per second over the whole run}."""
    rs = trace.requests
    ttft = sum(r.first_token - r.arrival for r in rs) / len(rs)
    multi = [r for r in rs if r.output > 1]
    tpot = sum((r.finish - r.first_token) / (r.output - 1) for r in multi) / max(len(multi), 1)
    latency = sum(r.finish - r.arrival for r in rs) / len(rs)
    span = max(r.finish for r in rs) - min(r.arrival for r in rs)
    return {"ttft": ttft, "tpot": tpot, "latency": latency,
            "throughput": sum(r.output for r in rs) / span}
