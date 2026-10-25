from m01.engine import Value


def test_add_and_mul_values():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == -1.0, "Value(2) + Value(-3) should have data -1"
    assert (a * b).data == -6.0, "Value(2) * Value(-3) should have data -6"


def test_results_are_new_values():
    a, b = Value(2.0), Value(-3.0)
    s = a + b
    assert isinstance(s, Value), "a + b should return a Value"
    assert s is not a and s is not b, "a + b should be a new Value, not one of the inputs"
    assert a.data == 2.0 and b.data == -3.0, "the inputs must not change"


def test_records_children_and_op():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    d = a * b
    e = d + c
    assert d._prev == {a, b}, "a * b should remember a and b in _prev"
    assert d._op == "*"
    assert e._prev == {d, c}, "d + c should remember d and c in _prev"
    assert e._op == "+"
    assert e.data == 4.0


def test_plain_numbers_on_either_side():
    a = Value(2.0)
    assert (a + 1).data == 3.0, "a + 1 should work: wrap 1 in a Value"
    assert (a * 3).data == 6.0
    assert (1 + a).data == 3.0, "1 + a should work too"
    assert (3 * a).data == 6.0
    child = next(iter((a * 3)._prev - {a}))
    assert isinstance(child, Value) and child.data == 3.0, "the wrapped number belongs in _prev too"


def test_graph_node_count():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    e = a * b + c
    seen = set()

    def walk(v):
        if v not in seen:
            seen.add(v)
            for p in v._prev:
                walk(p)

    walk(e)
    assert len(seen) == 5, f"the graph of a*b + c should have 5 nodes, found {len(seen)}"
