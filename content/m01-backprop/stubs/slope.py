"""Lesson 1.1: measure the slope of a function at a point."""


def cost(w):
    """The service cost from the lesson. Don't change this one."""
    return 0.5 * (w - 7) ** 2 + 45.5


def slope(f, x, h=1e-5):
    """Return an estimate of how fast f changes at x: the change in f per
    unit change in x, measured over a tiny step h.

    Nudge x a little in both directions (x - h and x + h), see how much f
    moves, and divide by how far x moved.
    """
    raise NotImplementedError("Lesson 1.1: write slope()")
