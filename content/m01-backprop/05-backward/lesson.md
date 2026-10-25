---
id: "1.5"
title: Running it backwards
concepts: [chain-rule, backpropagation, topological-sort]
file: m01/engine.py
tests: test_backward.py
predict:
  kind: number
  prompt: >
    L = a·b + a, with a = 3 and b = 4. If you nudge a up by a tiny amount ε,
    L goes up by about ε times what number? (That number is dL/da.)
  answer: 5
  tolerance: 0
mutants:
  - name: overwrites grads instead of adding
    file: m01/engine.py
    replace: [".grad += ", ".grad = "]
  - name: walks the graph leaves-first
    file: m01/engine.py
    replace: ["for v in reversed(order):", "for v in order:"]
---

## Hook

Take the graph for `L = a·b + a` with `a = 3, b = 4` (so L = 15). Ask:

- If `L` itself moves by 1, it moves by 1. Call that L's gradient: 1.
- `L = d + a` where `d = a·b`. Nudging `d` by ε moves `L` by ε. So d's gradient is 1 × (L's gradient).
- `d = a · b`. Nudging `b` by ε moves `d` by `a·ε = 3ε`. So b's gradient is 3 × (d's gradient) = 3.
- And `a`? It feeds into `L` along **two** paths.

Each step only needs *local* knowledge (how one operation's output responds
to its inputs) and the gradient already computed for the output.

## Build

Two parts, both in `m01/engine.py`.

**1. Local rules.** In `__add__` and `__mul__`, before returning `out`, attach
a function `out._backward` that pushes `out.grad` into the inputs' `.grad`:

- for `+`: each input's grad increases by `out.grad`
- for `*`: `self`'s grad increases by `other.data * out.grad`, and vice versa

Use `+=`, not `=`. Think about `a + a` to see why.

**2. `backward()`.** Set `self.grad = 1.0`, then call `_backward` on every
node in the graph, each one only *after* every node that uses it. Get that
order by sorting the graph topologically: depth-first, adding a node to the
list after all of its children. Then walk the list in reverse.

```python
a, b = Value(3.0), Value(4.0)
L = a * b + a
L.backward()
a.grad, b.grad
```

Then step through it. Press play on the graph to watch the values flow
forward and the gradients flow back:

```python
from workbench import show_graph
show_graph(L, title="L = a*b + a")
```

## Explain

The rule you applied at each node is the **chain rule**: if L depends on d,
and d depends on a, then `dL/da = dL/dd · dd/da`. When `a` reaches L along
several paths, the contributions **add**. That's why `+=` matters, and why
dL/da = b + 1 = 5.

Doing this for every node in one reverse sweep is **reverse-mode automatic
differentiation**, known in neural networks as **backpropagation**
(Rumelhart, Hinton & Williams, *Learning representations by back-propagating
errors*, Nature 1986). It costs about as much as one forward pass, whatever
the number of inputs. That beats lesson 1.3's 2 × (number of knobs) forward
passes.

The reverse topological order guarantees a node's gradient is complete before
it's passed on. Get it wrong and some contributions arrive too late.

## Stretch

Build a long chain: `x = Value(0.5)`, then 2,000 times `x = x * 1.0001`. Call
`backward()`. If it crashes, why? Rewrite the topological sort without
recursion.
