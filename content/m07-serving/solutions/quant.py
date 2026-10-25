"""Quantization: store weights in 8 or 4 bits instead of 16 or 32."""

import copy

import torch


def quantize_int8(w):
    """Symmetric per-row (per output channel) int8: scale = max|row| / 127,
    q = round(w / scale) clamped to [-127, 127]. Returns (q as int8,
    scales of shape (rows, 1))."""
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale


def dequantize(q, scale):
    return q.float() * scale


def quantize_int4(w, group=32):
    """Symmetric int4 with one scale per ``group`` consecutive weights in a
    row (levels -7 … 7). Returns (q (rows, cols) int8 holding the 4-bit
    values, scales (rows, cols / group))."""
    rows, cols = w.shape
    if cols % group:
        raise ValueError(f"{cols} columns don't split into groups of {group}")
    g = w.reshape(rows, cols // group, group)
    scale = g.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / 7
    q = torch.clamp(torch.round(g / scale), -7, 7).to(torch.int8)
    return q.reshape(rows, cols), scale[..., 0]


def dequantize_int4(q, scale, group=32):
    rows, cols = q.shape
    return (q.reshape(rows, cols // group, group).float() * scale[..., None]).reshape(rows, cols)


def quantize_model(model, bits=8, group=32):
    """A copy of ``model`` with every nn.Linear weight replaced by its
    quantized-then-dequantized value ("fake quantization": same compute,
    the accuracy of the low-bit weights)."""
    model = copy.deepcopy(model)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                if bits == 8:
                    m.weight.copy_(dequantize(*quantize_int8(m.weight)))
                elif bits == 4:
                    m.weight.copy_(dequantize_int4(*quantize_int4(m.weight, group), group))
                else:
                    raise ValueError("bits must be 8 or 4")
    return model


def linear_weight_bytes(model, bits=16, group=32):
    """Bytes for all nn.Linear weights at ``bits`` per weight, plus a 16-bit
    scale per row (8-bit) or per group (4-bit)."""
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            rows, cols = m.weight.shape
            total += rows * cols * bits / 8
            if bits == 8:
                total += rows * 2
            elif bits == 4:
                total += rows * (cols // group) * 2
    return total
