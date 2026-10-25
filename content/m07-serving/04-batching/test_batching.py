import pytest

from m07.scheduler import Cost, Request, continuous_batching, metrics, static_batching

COST = Cost(base=1.0, per_token=0.0)   # every iteration takes 1 s


def reqs():
    # (id, arrival, prompt tokens, output tokens)
    return [Request(0, 0.0, 4, 5), Request(1, 0.0, 4, 1), Request(2, 0.0, 4, 2), Request(3, 0.5, 4, 3)]


def by_id(trace):
    return {r.id: r for r in trace.requests}


def test_static_batching():
    t = static_batching(reqs(), 3, COST)
    r = by_id(t)
    # batch {0, 1, 2}: prefill ends at 1, then decoding until request 0 has 5 tokens (t = 5)
    assert (r[0].first_token, r[0].finish) == (1.0, 5.0)
    assert (r[1].first_token, r[1].finish) == (1.0, 1.0)
    assert r[2].finish == 2.0
    # request 3 waits for the whole batch: prefill 5 → 6, decode 6 → 8
    assert (r[3].first_token, r[3].finish) == (6.0, 8.0)
    assert len(t.steps) == 5 + 3


def test_continuous_batching():
    t = continuous_batching(reqs(), 3, COST)
    r = by_id(t)
    assert (r[0].first_token, r[0].finish) == (1.0, 5.0)
    assert r[1].finish == 1.0 and r[2].finish == 2.0
    # request 1 frees its slot after the first iteration; request 3 joins at t = 1
    assert (r[3].first_token, r[3].finish) == (2.0, 4.0)


def test_cost_model_counts_tokens():
    c = Cost(base=0.5, per_token=0.1)
    assert c.step(10, 3) == pytest.approx(0.5 + 1.3)
    t = continuous_batching([Request(0, 0.0, 10, 2)], 4, c)
    assert t.requests[0].first_token == pytest.approx(1.5) and t.requests[0].finish == pytest.approx(2.1)


def test_metrics():
    m = metrics(continuous_batching(reqs(), 3, COST))
    assert m["ttft"] == pytest.approx((1 + 1 + 1 + 1.5) / 4)
    assert m["tpot"] == pytest.approx((4 / 4 + 1 / 1 + 2 / 2) / 3)
    assert m["throughput"] == pytest.approx(11 / 5)
    assert metrics(static_batching(reqs(), 3, COST))["latency"] > m["latency"]


def test_inputs_untouched():
    rs = reqs()
    continuous_batching(rs, 2, COST)
    assert all(r.finish is None and r.generated == 0 for r in rs), "simulate on copies"
