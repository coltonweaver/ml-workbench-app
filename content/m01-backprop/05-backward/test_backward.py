from pytest import approx

from m01.engine import Value


def test_output_grad_is_one():
    a, b = Value(3.0), Value(4.0)
    out = a * b
    out.backward()
    assert out.grad == 1.0, "backward() should start by setting the output's own grad to 1"


def test_mul_and_add():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    e = a * b + c
    e.backward()
    assert (a.grad, b.grad, c.grad) == approx((-3.0, 2.0, 1.0)), (
        f"for e = a*b + c expected grads (a, b, c) = (-3, 2, 1), got "
        f"({a.grad}, {b.grad}, {c.grad})"
    )


def test_reused_node_accumulates():
    a = Value(3.0)
    out = a + a
    out.backward()
    assert a.grad == 2.0, (
        f"for a + a, dout/da is 2 but got {a.grad}. When a node is used twice, "
        "its grad contributions must add up (+=, not =)"
    )


def test_two_paths():
    a, b = Value(3.0), Value(4.0)
    L = a * b + a
    L.backward()
    assert a.grad == approx(5.0), f"dL/da should be b + 1 = 5, got {a.grad}"
    assert b.grad == approx(3.0)


def test_order_matters():
    # d feeds e twice through f = e * e; d's grad is only complete once both
    # uses have pushed into it.
    a, b = Value(2.0), Value(3.0)
    d = a * b          # 6
    e = d + a          # 8
    f = e * e          # 64
    f.backward()
    # df/de = 2e = 16; df/dd = 16; df/da = 16*b + 16 = 64; df/db = 16*a = 32
    assert e.grad == approx(16.0), f"df/de should be 16, got {e.grad}"
    assert a.grad == approx(64.0), (
        f"df/da should be 64, got {a.grad}. If it's too small, a node passed its "
        "grad on before it had collected everything: check the topological order"
    )
    assert b.grad == approx(32.0)


def test_long_and_short_paths():
    # d reaches out directly and through a three-step chain. A node visited
    # "as soon as it's reachable" would pass on d's grad before the chain's
    # contribution arrives.
    a, b = Value(2.0), Value(3.0)
    d = a * b
    chain = ((d * 2) + 1) * 3
    out = chain + d
    out.backward()
    assert d.grad == approx(7.0), (
        f"d.grad should be 6 (through the chain) + 1 (direct) = 7, got {d.grad}. "
        "Process nodes in reverse topological order"
    )
    assert a.grad == approx(21.0) and b.grad == approx(14.0)


def test_plain_numbers_get_no_say():
    a = Value(2.0)
    out = 3 * a + 1
    out.backward()
    assert a.grad == approx(3.0)
