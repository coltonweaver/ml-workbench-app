"""Convolution by hand, in NumPy.

Lesson 4.1: output_size and conv2d. Lesson 4.2: max_pool2d, the Sobel
kernels and edge_magnitude.
"""

import numpy as np


def output_size(size, kernel, stride=1, padding=0):
    """How many positions a kernel fits in ``size`` pixels."""
    raise NotImplementedError("lesson 4.1")


def conv2d(x, w, b=None, stride=1, padding=0):
    """2D convolution (strictly, cross-correlation, as in every DL framework).

    x: (N, C_in, H, W); w: (C_out, C_in, kh, kw); b: (C_out,) or None.
    Returns (N, C_out, H_out, W_out) where
    out[n, o, i, j] = b[o] + Σ_c Σ_u Σ_v x[n, c, i·s + u, j·s + v] · w[o, c, u, v].
    Zero-pad x by ``padding`` on each side first.
    """
    raise NotImplementedError("lesson 4.1")


def max_pool2d(x, k=2):
    """Max over non-overlapping k × k windows of (N, C, H, W); leftover rows
    and columns that don't fill a window are dropped."""
    raise NotImplementedError("lesson 4.2")


def sobel_x():
    """The 3 × 3 Sobel kernel that responds to changes from left to right
    (vertical edges): [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]."""
    raise NotImplementedError("lesson 4.2")


def sobel_y():
    """Responds to changes from top to bottom: horizontal edges."""
    raise NotImplementedError("lesson 4.2")


def edge_magnitude(image):
    """√(gx² + gy²) for a 2D image, where gx and gy are your conv2d with the
    two Sobel kernels, padded by 1 so the output has the input's size."""
    raise NotImplementedError("lesson 4.2")
