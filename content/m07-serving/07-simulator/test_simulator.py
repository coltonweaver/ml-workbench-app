import numpy as np
import pytest

from m07.scheduler import Cost, Request, continuous_batching, metrics
from m07.simulator import simulate, timeline_parts


def workload(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return [Request(i, float(rng.uniform(0, 1)), int(rng.integers(5, 50)), int(rng.integers(1, 60)))
            for i in range(n)]


def test_plenty_of_memory_is_continuous_batching():
    reqs = workload()
    sim = simulate(reqs, 8, n_blocks=1000, block_size=16)
    ref = continuous_batching(reqs, 8, Cost())
    assert sim.preemptions == 0
    for a, b in zip(sim.requests, ref.requests):
        assert a.id == b.id and a.first_token == pytest.approx(b.first_token) and a.finish == pytest.approx(b.finish)


def test_memory_limits_and_preemption():
    reqs = workload()
    roomy = metrics(simulate(reqs, 8, 1000, 16))
    tight = simulate(reqs, 8, 12, 16)
    assert tight.preemptions > 0, "12 blocks of 16 tokens can't hold 8 long sequences"
    assert all(b <= 12 for _, b in tight.memory), "never use more blocks than exist"
    assert len(tight.requests) == 30 and all(r.generated == r.output for r in tight.requests)
    assert metrics(tight)["latency"] > roomy["latency"]


def test_admission_waits_for_memory():
    cost = Cost(base=1.0, per_token=0.0)
    reqs = [Request(0, 0.0, 12, 3), Request(1, 0.0, 12, 1)]   # 15 and 13 tokens: one at a time
    sim = simulate(reqs, 4, n_blocks=1, block_size=16, cost=cost)
    r = {x.id: x for x in sim.requests}
    assert r[0].finish == 3.0 and r[1].first_token == 4.0, "request 1 waits until 0 frees the block"
    with pytest.raises(MemoryError):
        simulate([Request(0, 0.0, 40, 2)], 4, n_blocks=2, block_size=16)
    with pytest.raises(MemoryError):
        simulate([Request(0, 0.0, 15, 3)], 4, n_blocks=1, block_size=16)   # grows past its block


def test_spans_and_timeline():
    sim = simulate(workload(10), 4, 8, 16)
    kinds = {k for _, _, _, k in sim.spans}
    assert kinds <= {"queued", "prefill", "decode"} and "prefill" in kinds and "decode" in kinds
    for rid, s, e, _ in sim.spans:
        assert e >= s
    parts = timeline_parts(sim)
    assert len(parts["lanes"]) == 10 and parts["metrics"]["preemptions"] == sim.preemptions
    assert {s["kind"] for s in parts["spans"]} <= kinds
    assert [b["name"] for b in parts["below"]] == ["KV blocks in use", "running sequences"]
