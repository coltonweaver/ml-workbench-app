"""Quantization: store weights in 8 or 4 bits instead of 16 or 32.

Lesson 7.3."""
import copy
import torch


def quantize_int8(w):
    """Symmetric per-row (per output channel) int8: scale = max|row| / 127,
    q = round(w / scale) clamped to [-127, 127]. Returns (q as int8,
    scales of shape (rows, 1))."""
    raise NotImplementedError('lesson 7.3')


def dequantize(q, scale):
    raise NotImplementedError('lesson 7.3')


def quantize_int4(w, group=32):
    """Symmetric int4 with one scale per ``group`` consecutive weights in a
    row (levels -7 … 7). Returns (q (rows, cols) int8 holding the 4-bit
    values, scales (rows, cols / group))."""
    raise NotImplementedError('lesson 7.3')


def dequantize_int4(q, scale, group=32):
    raise NotImplementedError('lesson 7.3')


def quantize_model(model, bits=8, group=32):
    """A copy of ``model`` with every nn.Linear weight replaced by its
    quantized-then-dequantized value ("fake quantization": same compute,
    the accuracy of the low-bit weights)."""
    raise NotImplementedError('lesson 7.3')


def linear_weight_bytes(model, bits=16, group=32):
    """Bytes for all nn.Linear weights at ``bits`` per weight, plus a 16-bit
    scale per row (8-bit) or per group (4-bit)."""
    raise NotImplementedError('lesson 7.3')
