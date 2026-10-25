---
id: "8.5"
title: Mixed precision
concepts: [mixed-precision, bf16, fp16, loss-scaling, master-weights]
file: m08/precision.py
stub: stubs/precision.py
tests: test_mixed_precision.py
predict:
  kind: choice
  prompt: >
    A weight equal to 1.0 is stored in bfloat16 (8 bits of mantissa).
    The optimizer adds an update of 0.001. What is the weight afterwards?
  options:
    - "1.001"
    - "1.0: the update is smaller than half the gap to the next bf16 number"
    - "1.0078125, rounded up"
    - "NaN"
  answer: 1
mutants:
  - name: steps despite overflow
    file: m08/precision.py
    replace: ["    if found_inf:\n        return False\n    opt.step()", "    opt.step()"]
  - name: grows the scale on overflow
    file: m08/precision.py
    replace: ["self.scale /= 2", "self.scale *= 2"]
  - name: forgets to unscale
    file: m08/precision.py
    replace: ["return (true_grad * scale).half().float() / scale", "return (true_grad * scale).half().float()"]
---

## Hook

16-bit arithmetic is twice as fast and half the memory. But fp16 has a
tiny range (gradients below 6·10⁻⁸ become 0) and bf16 has coarse
precision (8 bits of mantissa). Micikevicius et al., "Mixed Precision
Training" (2018), made 16-bit training work with two tricks: keep a
32-bit master copy of the weights, and scale the loss.

## Build

In `m08/precision.py`:

1. `spacing(x, dtype)` and `update_survives(weight, update, dtype)`.
2. `fp16_gradient(true_grad, scale)`: what's left of a gradient after a
   scaled fp16 backward pass.
3. `LossScaler`: `scale_loss`, `unscale(grads)` (returns whether anything
   overflowed), and `update(found_inf)`: halve on overflow, double after
   `growth_interval` clean steps.
4. `train_step(params, grads_fp16, master, opt, scaler)`: copy grads to the
   fp32 master weights, unscale, skip the step on overflow, otherwise step
   and copy the master weights back to the fp16 model.

```python
spacing(1.0, torch.bfloat16), spacing(1.0, torch.float16), spacing(1.0, torch.float32)
fp16_gradient(torch.tensor([1e-8, 1e-4]), 1.0), fp16_gradient(torch.tensor([1e-8, 1e-4]), 2.0 ** 16)
```

## Explain

Near 1.0 bf16 numbers are 2⁻⁷ ≈ 0.0078 apart, so 1.0 + 0.001 rounds back
to **1.0: the update is lost**. Over thousands of steps the model just
stops learning. That's why the optimizer updates fp32 master weights, and
only the forward and backward passes run in 16 bits.

fp16 has more precision (10 bits) but its smallest normal number is
6·10⁻⁵ and its smallest subnormal 6·10⁻⁸; many gradients are smaller.
Multiplying the loss by 2¹⁶ shifts every gradient into range; dividing
afterwards in fp32 undoes it. If the scale is too large, something
overflows to inf, the step is skipped, and the scale halves. bf16 has
fp32's exponent range, so it needs no loss scaling, which is why it took
over for training. FP8 training (with per-tensor scaling again) is the
current frontier.

## Stretch

Train your Shakespeare GPT with `torch.autocast("cpu", dtype=torch.bfloat16)`
(or on MPS) and compare the loss curve with fp32. Then try keeping the
weights themselves in bf16 with no master copy. How much worse is it?
