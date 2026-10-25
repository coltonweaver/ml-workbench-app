import math

from pytest import approx

from m01.slope import cost, slope


def test_cost_slope_at_10():
    got = slope(cost, 10)
    assert got == approx(3.0), (
        f"slope(cost, 10) returned {got}; the bill rises about 0.03 per 0.01 of w, "
        "so expect about 3"
    )


def test_sign_flips_across_the_bottom():
    assert slope(cost, 7) == approx(0.0, abs=1e-6), "at w = 7 the cost is flat"
    assert slope(cost, 3) < 0, "left of the bottom, increasing w lowers the cost"


def test_works_for_any_function():
    assert slope(lambda x: x ** 3, 2) == approx(12.0), (
        "slope should work for any f, not just cost: the slope of x**3 at 2 is 12"
    )
    assert slope(math.sin, 0) == approx(1.0)


def test_uses_both_sides():
    got = slope(lambda x: x ** 2, 1.0, h=0.1)
    assert got == approx(2.0, rel=1e-9), (
        f"with h=0.1, slope(x**2, 1) gave {got}. Measuring on one side only gives "
        "2.1 (or 1.9); measure at x - h and x + h"
    )


def test_respects_h():
    calls = []

    def f(x):
        calls.append(x)
        return x

    slope(f, 5.0, h=0.5)
    assert sorted(calls) == [4.5, 5.5], f"expected f called at x - h and x + h, got {calls}"
