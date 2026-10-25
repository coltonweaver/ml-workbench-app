---
id: "3.8"
title: Same model, real framework
concepts: [pytorch, parity-check, gpu]
file: m03/torch_gpt.py
stub: stubs/torch_gpt.py
tests: test_pytorch.py
predict:
  kind: choice
  prompt: >
    You copy your engine GPT's weights into an identical PyTorch model and
    run both, in 64-bit floats, on the same input. How closely will the
    logits agree?
  options:
    - bit-for-bit identical
    - to about 15 significant digits, but not bit-identical
    - to about 3 decimal places
    - not closely: the frameworks compute differently
  answer: 1
mutants:
  - name: copies Linear weights untransposed
    file: m03/torch_gpt.py
    replace: ["put(tb.attn.q.weight, eb.attn.Wq.data.T)", "put(tb.attn.q.weight, eb.attn.Wq.data)"]
  - name: exact GELU instead of the tanh approximation
    file: m03/torch_gpt.py
    replace: ["F.gelu(self.fc(x), approximate=\"tanh\")", "F.gelu(self.fc(x))"]
  - name: forgets the causal mask
    file: m03/torch_gpt.py
    replace: ["att = att.masked_fill(future, float(\"-inf\")).softmax(dim=-1)", "att = att.softmax(dim=-1)"]
---

## Hook

You've earned a framework. PyTorch does what your engine does: tensors, a
graph recorded as you compute, `backward()`. It also runs on the Apple GPU
and has years of optimization behind it.

But "I rewrote it in PyTorch and it trains" isn't evidence that it's the
same model. Copy your engine's weights into the PyTorch version and compare
outputs and gradients. If they match, both are right (or wrong in exactly
the same way, which is much less likely).

## Build

In `m03/torch_gpt.py`:

1. `CausalSelfAttention.forward`, `MLP.forward` and `TorchGPT.forward`, the
   same computation as your engine. `F.gelu(..., approximate="tanh")`.
2. `load_from_engine(model, engine)`: copy every parameter. `nn.Linear`
   stores its weight as (out, in), the transpose of your `x @ W`.

Then train it on the GPU and compare throughput with your engine:

```python
from m03.torch_gpt import TorchGPT, train_torch, pick_device
device = pick_device()                           # "mps" on Apple silicon
tm = TorchGPT(len(itos), block_size=128, n_embd=256, n_head=8, n_layer=6)
dash = show_training(f"PyTorch GPT on {device}")
losses = train_torch(tm, data, steps=3000, batch_size=64, device=device,
                     log=lambda s, l: dash.log(s, loss=l, tokens=64 * 128))
dash.done()
```

That's a 4.8M-parameter model with 4× the context: about 1.5 for loss, and
recognizable verse.

## Explain

In float64 the two agree to about 1e-15: not bit-identical, because
floating-point addition isn't associative and the two compute sums in
different orders (and with different matmul kernels), but equal to the last
couple of bits. A **parity check** against a reference implementation is
how ML engineers catch silent bugs: a missing transpose or the wrong GELU
still trains, just worse, and you'd never know.

The GPU speedup comes from running each operation as one kernel over the
whole tensor on thousands of cores, without Python in between. Your engine
spends most of its time in Python, building graph nodes. Same algorithm,
different machinery.

This is the module 3 checkpoint, and the model you'll take apart in
module 5. Save it: `torch.save(tm.state_dict(), "m03/gpt.pt")`.

## Stretch

Use `torch.compile(tm)` and measure the speedup. Then run the parity check
on MPS in float32. How close are the logits now, and why?
