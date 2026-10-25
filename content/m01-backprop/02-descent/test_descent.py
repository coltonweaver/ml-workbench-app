from pytest import approx

from m01.descent import descend


def bowl(w):
    return (w - 7) ** 2


def test_path_shape():
    path = descend(bowl, 10, 0.1, 5)
    assert len(path) == 6, f"5 steps should give 6 points (start included), got {len(path)}"
    assert path[0] == 10, "the path should start at x0"


def test_reaches_the_bottom():
    end = descend(bowl, 10, 0.1, 100)[-1]
    assert end == approx(7.0, abs=1e-4), f"after 100 small steps expected ~7, got {end}"


def test_first_step_size():
    path = descend(bowl, 10, 0.1, 1)
    assert path[1] == approx(9.4, abs=1e-6), (
        f"slope at 10 is 6, so one step of lr=0.1 should land at 9.4; got {path[1]}. "
        "Move against the slope: x - lr * slope"
    )


def test_large_learning_rate_diverges():
    path = descend(bowl, 10, 1.1, 20)
    distances = [abs(x - 7) for x in path]
    assert distances[-1] > distances[0], "with lr=1.1 the walk should end further away than it began"
    assert path[1] < 7 < path[2], "it should overshoot the bottom and land on alternating sides"


def test_from_the_left():
    end = descend(bowl, 2, 0.1, 100)[-1]
    assert end == approx(7.0, abs=1e-4), "starting left of the bottom should walk right"
