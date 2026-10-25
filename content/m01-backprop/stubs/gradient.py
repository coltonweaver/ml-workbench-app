"""Lesson 1.3: many knobs at once."""


def gradient(f, xs, h=1e-5):
    """f takes a list of numbers and returns one number.

    Return a list the same length as xs whose i-th entry is the slope of f
    along knob i alone: nudge only xs[i], keep every other knob fixed.
    Don't modify xs itself.
    """
    raise NotImplementedError("Lesson 1.3: write gradient()")


def descend_many(f, xs0, lr, steps):
    """Like descend() from lesson 1.2, but every knob moves at once, each
    against its own slope. Return the path: a list of steps + 1 lists.
    """
    raise NotImplementedError("Lesson 1.3: write descend_many()")
