"""Lesson 1.2: walk downhill one knob at a time."""

from m01.slope import slope


def descend(f, x0, lr, steps):
    """Start at x0 and take `steps` steps downhill on f.

    Each step: measure the slope at the current x, then move x against the
    slope, scaled by the learning rate `lr`.

    Return the whole path as a list: [x0, x1, ..., x_steps] (steps + 1 values).
    """
    raise NotImplementedError("Lesson 1.2: write descend()")
