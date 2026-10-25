"""A serving simulator: continuous batching on top of paged KV memory, with
preemption when memory runs out.

Lesson 7.7."""
from dataclasses import dataclass, field
from m07.paged import BlockAllocator
from m07.scheduler import Cost, Request, _copy, metrics


@dataclass
class Result:
    requests: list
    spans: list = field(default_factory=list)
    memory: list = field(default_factory=list)
    batch: list = field(default_factory=list)
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
    raise NotImplementedError('lesson 7.7')


def timeline_parts(result):
    """Arguments for workbench.show_timeline: one lane per request, its
    spans colored by kind, headline metrics, and memory and batch size
    plotted below."""
    raise NotImplementedError('lesson 7.7')
