---
id: "5.1"
title: Handles on the residual stream
concepts: [residual-stream, hook-points, activation-caching]
file: m05/hooks.py
stub: stubs/hooks.py
tests: test_hooks.py
predict:
  kind: number
  prompt: >
    In a 2-layer GPT, every attention layer and every MLP adds its output to
    the residual stream, which starts as the embedding (token plus position,
    counted as one term). The stream that reaches the final layer norm is a
    sum of how many terms?
  answer: 5
  tolerance: 0
mutants:
  - name: caches before patching
    file: m05/hooks.py
    replace: ["        if name in patch:\n            t = patch[name](t)\n        if cache is not None:\n            cache[name] = t", "        if cache is not None:\n            cache[name] = t\n        if name in patch:\n            t = patch[name](t)"]
  - name: forgets the residual connection
    file: m05/hooks.py
    replace: ["x = point(p + \"resid_mid\", x + attn_out)", "x = point(p + \"resid_mid\", attn_out)"]
  - name: no 1/√d
    file: m05/hooks.py
    replace: ["scores = (q @ k.transpose(-2, -1)) / math.sqrt(hs)", "scores = q @ k.transpose(-2, -1)"]
---

## Hook

Module 5 is about reading a trained network's internals and testing what
you think they do. Every experiment needs the same two abilities: **read**
any intermediate tensor, and **replace** it during a forward pass.

PyTorch's forward hooks (which you used in module 4) attach to modules, but
the interesting tensors here aren't module outputs: the residual stream
between blocks, the attention pattern inside a head, a layer norm's divisor.
So write the forward pass yourself, with a named checkpoint everywhere one
is useful. This is what the TransformerLens library does.

## Build

In `m05/hooks.py`, for the `TorchGPT` from lesson 3.8:

1. `hook_names(model)`: every name listed at the top of the file, in order.
2. `layer_norm(ln, x, scale_hook)`: `ln(x)` by hand, passing the divisor
   √(var + eps) through `scale_hook` before dividing.
3. `attention_parts(attn, x)`: the pattern (B, H, T, T) and the values,
   split into heads.
4. `run(model, idx, patch=None, cache=None)`: the forward pass. At every
   point, apply `patch[name]` if there is one, then record the result in
   `cache`. It must give exactly `model(idx)` when nothing is patched.
5. `run_with_cache(model, idx)`.

The rest of the module works on the Shakespeare GPT you trained in lesson
3.8 (save it there with `torch.save(m.state_dict(), "gpt.pt")`):

```python
import torch
from m03.torch_gpt import TorchGPT
from m03.train import build_char_vocab
from workbench.data import shakespeare
text = shakespeare()
stoi, itos = build_char_vocab(text)
encode = lambda s: torch.tensor([stoi[c] for c in s])
decode = lambda ids: "".join(itos[int(i)] for i in ids)
m = TorchGPT(len(itos), 64)               # the sizes you trained with
m.load_state_dict(torch.load("gpt.pt")); m.eval()
idx = encode("ROMEO:\nBut soft, what light")[None]
logits, cache = run_with_cache(m, idx)
{k: tuple(v.shape) for k, v in cache.items()}
run(m, idx, patch={"blocks.0.attn_out": torch.zeros_like})   # knock out layer 0's attention
```

## Explain

The stream reaching ln_f is embed + attn 0 + mlp 0 + attn 1 + mlp 1:
**5** terms. Each layer reads the whole stream (through its layer norm) and
writes by adding. Nothing is ever overwritten, which is why Elhage et al.,
"A Mathematical Framework for Transformer Circuits" (2021), treat the
residual stream as a shared communication channel: every component's output
stays in it, and later components can read any earlier one.

That additive structure is what the next lessons exploit. The final logits
are a (nearly) linear function of a sum, so they can be split into each
component's share (5.2), and replacing one term with its value on another
input isolates that term's role (5.3).

## Stretch

Load the Shakespeare GPT you trained in lesson 3.8, and use `show_attention`
on `cache["blocks.1.attn.pattern"][0]` (one head at a time). Can you spot a
head that always looks at the previous character?
