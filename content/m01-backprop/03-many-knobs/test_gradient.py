from pytest import approx

from m01.gradient import descend_many, gradient


def f(v):
    return v[0] ** 2 + 10 * v[1] ** 2


def test_gradient_values():
    g = gradient(f, [5.0, 5.0])
    assert g == approx([10.0, 100.0], rel=1e-5), (
        f"expected [10, 100] (2x and 20y), got {g}. Nudge one knob at a time"
    )


def test_gradient_does_not_mutate_input():
    xs = [5.0, 5.0]
    gradient(f, xs)
    assert xs == [5.0, 5.0], f"gradient() changed its input to {xs}; copy the list before nudging"


def test_gradient_any_length():
    g = gradient(lambda v: sum(i * x for i, x in enumerate(v)), [1.0] * 6)
    assert g == approx([0, 1, 2, 3, 4, 5], abs=1e-5)


def test_descend_many_path():
    path = descend_many(f, [5.0, 5.0], 0.04, 30)
    assert len(path) == 31, "30 steps give 31 points"
    assert path[0] == [5.0, 5.0]
    assert path[1] == approx([4.6, 1.0], abs=1e-5), (
        f"one step from (5, 5) with lr 0.04 should reach (4.6, 1.0), got {path[1]}. "
        "Each knob moves by its own slope"
    )
    x, y = path[-1]
    assert abs(y) < 1e-6 and 0.2 < x < 0.5, f"after 30 steps y should be ~0 and x ~0.4; got {path[-1]}"


def test_descend_many_does_not_alias_steps():
    path = descend_many(f, [5.0, 5.0], 0.04, 3)
    assert path[0] != path[-1], (
        "every entry of the path changed together; append a copy of the list each step"
    )
