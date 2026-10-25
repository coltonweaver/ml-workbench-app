"""Convolution by hand, in NumPy."""

import numpy as np


def output_size(size, kernel, stride=1, padding=0):
    return (size + 2 * padding - kernel) // stride + 1


def conv2d(x, w, b=None, stride=1, padding=0):
    """2D convolution (strictly, cross-correlation, as in every DL framework).

    x: (N, C_in, H, W); w: (C_out, C_in, kh, kw); b: (C_out,) or None.
    Returns (N, C_out, H_out, W_out) where
    out[n, o, i, j] = b[o] + Σ_c Σ_u Σ_v x[n, c, i·s + u, j·s + v] · w[o, c, u, v].
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    N, C, H, W = x.shape
    O, C2, kh, kw = w.shape
    assert C == C2, f"x has {C} channels but w expects {C2}"
    Ho, Wo = output_size(H, kh, stride), output_size(W, kw, stride)
    out = np.zeros((N, O, Ho, Wo))
    # Loop over the kernel's offsets, not the output pixels: each offset is
    # one big multiply of a shifted view of x.
    for u in range(kh):
        for v in range(kw):
            patch = x[:, :, u:u + stride * Ho:stride, v:v + stride * Wo:stride]
            out += np.einsum("nchw,oc->nohw", patch, w[:, :, u, v])
    if b is not None:
        out += np.asarray(b)[None, :, None, None]
    return out


def max_pool2d(x, k=2):
    """Max over non-overlapping k × k windows of (N, C, H, W); leftover rows
    and columns that don't fill a window are dropped."""
    N, C, H, W = x.shape
    Ho, Wo = H // k, W // k
    x = x[:, :, :Ho * k, :Wo * k]
    return x.reshape(N, C, Ho, k, Wo, k).max(axis=(3, 5))


def sobel_x():
    """Responds to changes from left to right: vertical edges."""
    return np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def sobel_y():
    """Responds to changes from top to bottom: horizontal edges."""
    return sobel_x().T


def edge_magnitude(image):
    """√(gx² + gy²) for a 2D image, same size as the input (pad 1)."""
    x = np.asarray(image, dtype=np.float64)[None, None]
    w = np.stack([sobel_x(), sobel_y()])[:, None]  # (2, 1, 3, 3)
    g = conv2d(x, w, padding=1)[0]
    return np.sqrt(g[0] ** 2 + g[1] ** 2)
