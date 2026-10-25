"""Lesson 1.3: many knobs at once."""


def gradient(f, xs, h=1e-5):
    grads = []
    for i in range(len(xs)):
        up = list(xs)
        down = list(xs)
        up[i] += h
        down[i] -= h
        grads.append((f(up) - f(down)) / (2 * h))
    return grads


def descend_many(f, xs0, lr, steps):
    xs = list(xs0)
    path = [list(xs)]
    for _ in range(steps):
        g = gradient(f, xs)
        xs = [x - lr * gi for x, gi in zip(xs, g)]
        path.append(list(xs))
    return path
