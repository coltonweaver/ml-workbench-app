"""Lesson 1.7: check your engine against finite differences."""

from m01.engine import Value


def gradcheck(f, xs, h=1e-6):
    inputs = [Value(x) for x in xs]
    out = f(inputs)
    out.backward()
    pairs = []
    for i, v in enumerate(inputs):
        up = [Value(x + h if j == i else x) for j, x in enumerate(xs)]
        down = [Value(x - h if j == i else x) for j, x in enumerate(xs)]
        numeric = (f(up).data - f(down).data) / (2 * h)
        pairs.append((v.grad, numeric))
    return pairs


def passes(f, xs, tol=1e-4):
    return all(
        abs(a - n) / max(1.0, abs(a), abs(n)) <= tol
        for a, n in gradcheck(f, xs)
    )
