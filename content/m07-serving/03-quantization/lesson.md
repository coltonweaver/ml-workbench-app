---
id: "7.3"
title: Fewer bits per weight
concepts: [quantization, int8, int4, outliers]
file: m07/quant.py
stub: stubs/quant.py
tests: test_quantization.py
predict:
  kind: choice
  prompt: >
    One row of a weight matrix has a single weight of 50 among values
    around ±1. With one int8 scale for the whole row, what happens to the
    small weights?
  options:
    - nothing; int8 has 255 levels, plenty
    - they're rounded to steps of about 0.4, so most of them become 0 or ±0.4
    - the outlier is clipped to 1
    - they're stored exactly and the outlier is approximate
  answer: 1
mutants:
  - name: one scale for the whole tensor
    file: m07/quant.py
    replace: ["scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127", "scale = (w.abs().amax().clamp_min(1e-12) / 127).expand(w.shape[0], 1)"]
  - name: truncates instead of rounding
    file: m07/quant.py
    replace: ["torch.round(w / scale)", "torch.floor(w / scale)"]
  - name: quantizes the original
    file: m07/quant.py
    replace: ["    model = copy.deepcopy(model)\n", ""]
warmups:
  - kind: number
    prompt: "Symmetric int8 quantization of a row whose largest |value| is 2.54. What is the scale?"
    answer: 0.02
    tolerance: 0.02
    explain: "scale = max|w| / 127 = 0.02, and each weight is stored as round(w / 0.02). One outlier in the row makes that scale coarse for everyone."
---

## Hook

Decoding speed is weight bytes divided by bandwidth (7.1). Halve the bytes
and you double the speed, and fit a bigger model or more KV cache in the
same memory. Most served models today run with 8-bit or 4-bit weights.

## Build

In `m07/quant.py`:

1. `quantize_int8(w)`: symmetric, one scale per row (output channel):
   scale = max|row| / 127, q = round(w / scale). `dequantize(q, scale)`.
2. `quantize_int4(w, group)`: levels −7 … 7, one scale per `group`
   consecutive weights in a row. `dequantize_int4`.
3. `quantize_model(model, bits)`: a deep copy with every `nn.Linear`
   weight replaced by its dequantized version.
4. `linear_weight_bytes(model, bits)`.

```python
for bits in [8, 4]:
    q = quantize_model(m, bits)
    print(bits, linear_weight_bytes(m, bits) / linear_weight_bytes(m, 16),
          eval_loss(q, val_data, 64), eval_loss(m, val_data, 64))   # eval_loss from lesson 6.5
```

## Explain

The scale is set by the largest value: 50 / 127 ≈ 0.39 per step. Every
weight near ±1 rounds to one of −0.39, 0, 0.39, 0.79, …: **steps of about
0.4, most of them collapse to 0 or ±0.4**. One outlier wrecks the
precision of its whole group.

That's why granularity matters: per-tensor scales are cheap but fragile,
per-channel better, small groups (32 to 128 weights, as in GPTQ and AWQ
4-bit formats) better still, at the cost of storing more scales. And
large transformers do have outliers, systematically, in particular hidden
dimensions; Dettmers et al., "LLM.int8()" (2022), keep those few dimensions
in 16-bit. SmoothQuant and AWQ instead rescale weights and activations so
the outliers move where they're harmless.

What you implemented is "fake quantization": weights are stored back as
floats, so it measures accuracy, not speed. Real speedups need kernels
that read the int8 or int4 values directly.

## Stretch

Implement round-to-nearest with the scale chosen to minimize squared error
(search a few clipping ratios below max|w|) instead of max|w|. Does
clipping the outlier help the 4-bit model's loss?
