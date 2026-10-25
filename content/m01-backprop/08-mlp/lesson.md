---
id: "1.8"
title: A tiny neural network
concepts: [neuron, mlp, loss, training-loop]
file: m01/nn.py
stub: stubs/nn.py
tests: test_mlp.py
predict:
  kind: number
  prompt: >
    MLP(3, [4, 4, 1]) has 3 inputs, two hidden layers of 4 tanh neurons, and 1
    output neuron. Each neuron has one weight per input plus a bias. How many
    numbers will training adjust?
  answer: 41
  tolerance: 0
mutants:
  - name: never resets gradients between steps
    file: m01/nn.py
    replace: ["p.grad = 0.0", "pass"]
---

## Hook

Here are four examples, each three numbers in and a target out:

```python
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
```

You don't know the rule that maps one to the other. You want a function with
a lot of knobs that can *bend* into the right shape, and a way to turn every
knob downhill at once.

You have both pieces already. `tanh` bends, and your engine gives the
gradient of anything you can compute.

## Build

In `m01/nn.py`:

1. `Neuron(nin)`: `nin` weights (random in [−1, 1]) and a bias (0). Calling
   it on inputs `x` returns `tanh(Σ wᵢxᵢ + b)`.
2. `Layer(nin, nout)`: `nout` neurons all reading the same inputs. Calling it
   returns their outputs as a list, or just the one Value if `nout == 1`.
3. `MLP(nin, nouts)`: layers chained so each one's outputs are the next one's
   inputs.
4. `parameters()` on each, returning every Value training should adjust.
5. `train(model, xs, ys, lr, steps)`: each step, compute the **loss** (the sum
   of `(prediction − target)²`), zero every parameter's grad, call
   `backward()`, and move every parameter by `−lr × grad`. Return the losses.

```python
import random; random.seed(0)
model = MLP(3, [4, 4, 1])
losses = train(model, xs, ys, lr=0.05, steps=100)
losses[0], losses[-1]
[model(x).data for x in xs]
```

## Explain

You built a **multilayer perceptron**. Neurons in layers, with a nonlinearity
between them, can approximate any continuous function given enough units
(the universal approximation theorem; Cybenko 1989, Hornik 1991). You also
built the **training loop** that every model in this course will use, all the
way up to the GPT in module 3: forward, **loss**, zero grads, backward,
update.

The loss is **mean squared error** (summed here). It turns "how wrong is the
model on all examples" into one number, so one backward pass computes how
every one of the 41 parameters should move.

Forgetting to zero the grads is the classic bug. `backward()` *adds* into
`.grad` (it has to, for reused nodes), so without the reset, each step also
carries every previous step's gradient. PyTorch has the same behaviour,
which is why `optimizer.zero_grad()` appears in every training loop.

This is the module 1 checkpoint project: a working autograd engine that
trains a network. Module 2 swaps scalars for tensors so it can scale beyond a
few hundred parameters.

## Stretch

Train with `lr = 0.5` and with `lr = 0.001`. Plot or print the losses. Then
change every `tanh` to `relu` and train again. What happens to the outputs,
which no longer stay in (−1, 1)? Can you still fit targets of ±1?
