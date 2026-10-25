"""Batching requests: static batches versus continuous batching.

Lesson 7.4."""
from dataclasses import dataclass, field


@dataclass(eq=False)
class Request:
    id: int
    arrival: float
    prompt: int
    output: int
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
        raise NotImplementedError('lesson 7.4')


@dataclass
class Trace:
    """What happened: an entry (time, request ids, prefill tokens, decoding
    count, duration) per iteration, and the finished requests."""
    steps: list = field(default_factory=list)
    requests: list = field(default_factory=list)


def _copy(requests):
    """Fresh copies sorted by arrival, so a simulation never edits its input."""
    return sorted((Request(r.id, r.arrival, r.prompt, r.output) for r in requests), key=lambda r: r.arrival)


def static_batching(requests, batch_size, cost):
    """Wait for up to ``batch_size`` requests (whatever has arrived when the
    engine is free, or the next arrival), prefill them together, decode
    until the longest finishes, and only then start the next batch."""
    raise NotImplementedError('lesson 7.4')


def continuous_batching(requests, max_batch, cost):
    """Every iteration: admit arrived requests while there's room (their
    prefill runs in this iteration and yields their first token), decode one
    token for every other running request, and retire finished requests
    immediately so their slots free up for the next iteration."""
    raise NotImplementedError('lesson 7.4')


def metrics(trace):
    """{"ttft": mean time to first token (from arrival), "tpot": mean time
    per output token after the first, "latency": mean arrival-to-finish,
    "throughput": output tokens per second over the whole run}."""
    raise NotImplementedError('lesson 7.4')
