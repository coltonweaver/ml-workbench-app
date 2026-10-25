---
id: "1.4"
title: Numbers that remember
concepts: [computation-graph]
file: m01/engine.py
stub: stubs/engine.py
tests: test_value.py
predict:
  kind: number
  prompt: >
    e = a·b + c with a = 2, b = −3, c = 10. Counting a, b and c themselves and
    every intermediate result, how many numbers does it take to compute e?
  answer: 5
  tolerance: 0
mutants:
  - name: forgets to record children for +
    file: m01/engine.py
    replace: ["Value(self.data + other.data, (self, other), \"+\")", "Value(self.data + other.data, (), \"+\")"]
---

## Hook

Lesson 1.3 ended on a problem. Finite differences cost two full evaluations
of `f` per knob, so a model with 100M weights needs 200M forward passes for
one step.

There's a better way. When your code computes a result, it already does every
multiplication and addition that makes it up. If each number *remembered how
it was made*, we could walk back through that record and work out how the
result depends on every input, in one pass.

Start with the record. In

```python
a, b, c = 2, -3, 10
e = a * b + c
```

how many numbers exist along the way?

## Build

`m01/engine.py` has a `Value` class that wraps one number. It stores the
number (`data`), the Values it was computed from (`_prev`) and the operation
(`_op`).

Write `__add__` and `__mul__` so that

```python
a = Value(2.0); b = Value(-3.0); c = Value(10.0)
d = a * b
e = d + c
e.data       # 4.0
e._op        # '+'
e._prev      # {d, c}
```

A plain number on either side should work too: `a + 1`, `2 * a` and `1 + a`.
Wrap the plain number in a `Value` first. (`__radd__` and `__rmul__` are done
for you; they're what Python calls for `1 + a`.)

Leave `backward` and the rest alone for now.

## Explain

You've built a **computation graph**: a directed acyclic graph whose leaves are
inputs and whose internal nodes are operations. Each node records its parents
(here `_prev`) and how it was made (`_op`).

This is the data structure at the heart of every deep learning framework.
PyTorch builds one on the fly as your code runs ("define-by-run"). Older
frameworks like TensorFlow 1 made you declare the graph first. Yours is the
same idea at the scale of single numbers. A tensor version comes in module 2.

Five numbers, by the way: a, b, c, a·b and e. Every one of them will get its
own gradient in the next lesson.

## Stretch

Write `draw(v)` that prints the graph below `v` as an indented tree, one line
per node, with its `_op` and `data`. Use it on `(a * b + c) * a`. Notice that
`a` appears twice in the tree, but only once in the graph. Why does that
matter for the next lesson?
