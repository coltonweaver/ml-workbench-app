---
id: "4.6"
title: Reading a circuit
concepts: [circuits, ablation, causal-intervention]
file: m04/circuits.py
stub: stubs/circuits.py
tests: test_circuits.py
predict:
  kind: choice
  prompt: >
    A conv2 channel reads a conv1 channel through weights that are all
    negative (inhibitory). conv1's outputs pass through ReLU, so they are
    never negative. You zero that conv1 channel. What happens to the conv2
    channel's activation (before its own ReLU)?
  options:
    - it goes down or stays the same
    - it goes up or stays the same
    - it can go either way
  answer: 1
mutants:
  - name: reads the wrong axis of the weights
    file: m04/circuits.py
    replace: ["return w[out_channel].flatten(1).norm(dim=1).numpy()", "return w[:, out_channel].flatten(1).norm(dim=1).numpy()"]
  - name: excitation ignores the sign
    file: m04/circuits.py
    replace: ["return w[out_channel].flatten(1).sum(dim=1).numpy()", "return w[out_channel].flatten(1).abs().sum(dim=1).numpy()"]
  - name: hook doesn't return its change
    file: m04/circuits.py
    replace: ["        output[:, channel] = 0\n        return output", "        output[:, channel] = 0\n        return None"]
  - name: drop with the wrong sign
    file: m04/circuits.py
    replace: ["return normal - ablated", "return ablated - normal"]
---

## Hook

You know what some conv1 channels detect and what some conv2 channels
detect. A conv2 channel is a weighted sum of conv1 channels, so its weights
are a wiring diagram: this curve detector is built from those two edge
detectors, and suppressed by that one. Reading the diagram is a hypothesis.
To test it, cut a wire and see what breaks.

## Build

In `m04/circuits.py`:

1. `circuit_strengths(conv_weight, out_channel)`: for each input channel,
   the L2 norm of its kh × kw slice into `out_channel`.
2. `excitation(conv_weight, out_channel)`: the sum of that slice. Positive
   mostly excites, negative mostly inhibits.
3. `ablate(layer, channel)`: a context manager (`@contextlib.contextmanager`)
   that, inside the `with` block, sets `channel` of `layer`'s output to 0.
   A forward hook that **returns** a value replaces the output; clone it
   rather than editing in place.
4. `ablation_effect(...)`: a channel's mean activation, normally and with
   another channel ablated.
5. `class_accuracy_drop(model, layer, channel, X, Y, n_classes)`: per-class
   accuracy, normal minus ablated, NaN for classes absent from Y.

```python
from workbench import show_matrix
c = 3
show_matrix(model.conv2.weight[c].flatten(1).detach().numpy(), title=f"inputs to conv2 #{c}")
strong = circuit_strengths(model.conv2.weight, c).argsort()[::-1][:3]
for i in strong:
    print(i, excitation(model.conv2.weight, c)[i],
          ablation_effect(model, model.conv1, int(i), model.conv2, c, Xt))
class_accuracy_drop(model, model.conv2, c, Xt, Yt, 6)
```

## Explain

The conv1 channel contributes (negative weight) × (non-negative input),
which is never positive. Zeroing it removes a term that was ≤ 0, so the
activation **goes up or stays the same**. If your measurement says
otherwise, either the weights aren't all negative or something else
changed; both are worth knowing.

This is the method of Olah et al., "Zoom In: An Introduction to Circuits"
(Distill, 2020): understand features, read the weights between them as
circuits, and verify. Ablation turns a story into an experiment, and
`class_accuracy_drop` connects one channel to the model's behavior. A
channel whose ablation costs "circle" accuracy and nothing else is
evidence it detects something circle-specific.

Ablation has a known flaw: zero may be a value the channel never takes, so
you test the network off its normal inputs. Mean ablation (replace with the
channel's average) or resampling (use its value on a different input) are
gentler. Module 5 builds on this as **activation patching**, the core
experiment of mechanistic interpretability in transformers.

## Stretch

Find the conv2 channel whose ablation hurts one class most. Trace it back:
which conv1 channels feed it most strongly, what are their
visualizations, and does ablating each of them hurt the same class? Write
the circuit down as a sentence and say how confident you are.
