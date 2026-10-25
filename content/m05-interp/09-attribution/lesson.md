---
id: "5.9"
title: Attribution graphs
concepts: [attribution-graph, gradient-times-activation, circuits]
file: m05/attribution.py
stub: stubs/attribution.py
tests: test_attribution.py
predict:
  kind: choice
  prompt: >
    Without freezing the final layer norm, you compute gradient × activation
    for every input to it (each a vector, dotted with its gradient) and add
    them up. Roughly what total do you get?
  options:
    - the logit
    - zero
    - the logit minus the bias term
    - the loss
  answer: 1
mutants:
  - name: layer norm not frozen
    file: m05/attribution.py
    replace: ["    patch[\"ln_final.scale\"] = torch.Tensor.detach\n", ""]
  - name: activation without its gradient
    file: m05/attribution.py
    replace: ["effects = torch.stack([(acts[n][0] * grads[n]).sum(-1).detach() for n, _ in names])", "effects = torch.stack([acts[n][0].sum(-1).detach() for n, _ in names])"]
  - name: weakest edges kept
    file: m05/attribution.py
    replace: ["edges.sort(key=lambda e: -abs(e[4]))", "edges.sort(key=lambda e: abs(e[4]))"]
---

## Hook

Direct logit attribution only sees the last hop. Patching sees any path but
costs one forward pass per question. An **attribution graph** traces every
path at once, as a graph: which components at which positions fed which,
on the way to one prediction. Ameisen et al., "Circuit Tracing" (2025),
built these from cross-layer transcoder features to trace multi-step
reasoning in Claude 3.5 Haiku. You'll build the same idea at component
level.

## Build

In `m05/attribution.py`:

1. `node_names(model)`: the hook points that write to the stream.
2. `attribution_graph(model, idx, pos, token, freeze_attention=True)`:
   - Run with patches that `retain_grad()` on each node and detach every
     layer norm's `.scale` (and the attention patterns, if frozen).
   - Backward from the target logit. A node's effect is a · ∂logit/∂a.
   - For each node B and position t, take B's effect with its gradient held
     fixed, and differentiate it with respect to every earlier node with
     `torch.autograd.grad`. That gives the edges.
3. `prune(graph, keep_nodes, keep_edges)`.

```python
from workbench import show_attribution
prompt = encode("ROMEO:\nWhat light through yonder")
g = attribution_graph(m, prompt[None], pos=len(prompt) - 1, token=stoi[" "])
nodes, edges = prune(g, 14, 30)
show_attribution(list(decode(prompt)), g["labels"], g["effects"].numpy(),
                 target=(len(prompt) - 1, " "), nodes=nodes, edges=edges)
```

## Explain

A layer norm divides by its input's standard deviation, so scaling the
input by any factor leaves the output unchanged. A function that ignores
scale has x · ∇f(x) = 0 (Euler's theorem for degree-0 homogeneous
functions), so gradient × activation over all of a norm's inputs sums to
**zero**: every attribution would cancel. Freezing the scale makes the norm
linear and the attributions meaningful, which is why Circuit Tracing
freezes norms and attention patterns.

Freezing attention means the graph shows what was **moved** (values
through the OV circuit), not why heads **looked** where they did (the QK
circuit). Both matter; the paper treats attention patterns as given and
studies them separately.

Gradient × activation is a first-order estimate. Through the MLPs'
nonlinearities it can be off. Always check the important edges with a
patch: if the graph says a0 at "y" feeds m1 at the last position, ablating
that node should change the logit by about the edge's weight.

## Stretch

Pick the strongest edge in your graph and verify it by patching (zero the
source node at its position and measure the change in the target node's
effect). How close is the linear estimate? Try one through an MLP and one
that isn't.
