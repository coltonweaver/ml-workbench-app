"""Lesson 1.1: measure the slope of a function at a point."""


def cost(w):
    """The service cost from the lesson. Don't change this one."""
    return 0.5 * (w - 7) ** 2 + 45.5


def slope(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
