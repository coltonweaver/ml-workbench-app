"""Lesson 1.7: check your engine against finite differences."""

from m01.engine import Value


def gradcheck(f, xs, h=1e-6):
    """f takes a list of Values and returns one Value.

    Return a list of (analytic, numeric) pairs, one per input:
    - analytic: the gradient your engine computes with backward()
    - numeric: the slope measured by nudging that input by +/- h, as in
      lesson 1.3 (call f on fresh Values each time)
    """
    raise NotImplementedError("Lesson 1.7: write gradcheck()")


def passes(f, xs, tol=1e-4):
    """True when every pair from gradcheck(f, xs) agrees to within tol,
    measured relative to the size of the numbers: |a - n| / max(1, |a|, |n|).
    """
    raise NotImplementedError("Lesson 1.7: write passes()")
