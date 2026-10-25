import math
import random

from pytest import approx

from m01.engine import Value
from m01.gradcheck import gradcheck, passes


def test_pairs_for_a_known_function():
    pairs = gradcheck(lambda v: v[0] * v[1] + v[0] ** 2, [3.0, 4.0])
    assert len(pairs) == 2, "one (analytic, numeric) pair per input"
    (a0, n0), (a1, n1) = pairs
    assert a0 == approx(10.0) and a1 == approx(3.0), "analytic grads should come from backward()"
    assert n0 == approx(10.0, rel=1e-5) and n1 == approx(3.0, rel=1e-5), (
        f"numeric grads should be ~(10, 3), got ({n0}, {n1}). Use a central difference"
    )


def test_numeric_side_is_independent_of_backward():
    # A function whose backward is wrong: the numeric column must still be right.
    def f(v):
        out = v[0] * 2
        out._backward = lambda: None  # sabotage: no gradient flows
        return out

    (analytic, numeric), = gradcheck(f, [1.5])
    assert analytic == 0.0
    assert numeric == approx(2.0, rel=1e-5), (
        "the numeric estimate should only use forward values (.data), never .grad"
    )


def test_passes_detects_a_bad_gradient():
    def wrong_tanh(v):
        x = v[0]
        t = math.tanh(x.data)
        out = Value(t, (x,), "tanh")

        def _backward():
            x.grad += (1 - t) * out.grad  # the bug from the prediction

        out._backward = _backward
        return out

    assert passes(lambda v: v[0].tanh(), [0.7]), "a correct tanh should pass"
    assert not passes(wrong_tanh, [0.7]), "1 - t instead of 1 - t**2 should fail at x = 0.7"


def test_relative_tolerance():
    big = lambda v: v[0] * 1e6
    assert passes(big, [1.0]), "large gradients need relative, not absolute, comparison"


def test_checkpoint_engine_matches_finite_differences():
    rng = random.Random(1234)
    exprs = [
        lambda v: (v[0] * v[1] + v[2]).tanh(),
        lambda v: (v[0] - v[1]) / (v[2] ** 2 + 1),
        lambda v: (v[0].exp() * v[1] + 3 - v[2]).relu() + v[0] * v[0],
        lambda v: ((v[0] + v[1]) * (v[1] + v[2]) * (v[2] + v[0])).tanh(),
        lambda v: 1 / (1 + (-v[0] * v[1] + v[2]).exp()),
    ]
    for i, f in enumerate(exprs):
        for _ in range(5):
            xs = [rng.uniform(-1.5, 1.5) for _ in range(3)]
            assert passes(f, xs), (
                f"your engine's gradients disagree with finite differences for expression "
                f"{i} at {[round(x, 3) for x in xs]}: {gradcheck(f, xs)}"
            )
