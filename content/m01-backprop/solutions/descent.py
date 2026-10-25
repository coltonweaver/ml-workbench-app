"""Lesson 1.2: walk downhill one knob at a time."""

from m01.slope import slope


def descend(f, x0, lr, steps):
    path = [x0]
    x = x0
    for _ in range(steps):
        x = x - lr * slope(f, x)
        path.append(x)
    return path
