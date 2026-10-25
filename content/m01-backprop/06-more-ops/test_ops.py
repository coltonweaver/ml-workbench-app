import math

from pytest import approx

from m01.engine import Value


def grad_of(fn, x):
    v = Value(x)
    out = fn(v)
    out.backward()
    return out.data, v.grad


def test_pow():
    data, g = grad_of(lambda v: v ** 3, 2.0)
    assert data == approx(8.0)
    assert g == approx(12.0), f"d/dx x**3 at 2 is 12, got {g}"
    data, g = grad_of(lambda v: v ** -1, 4.0)
    assert (data, g) == approx((0.25, -1 / 16))


def test_exp():
    data, g = grad_of(lambda v: v.exp(), 1.5)
    assert (data, g) == approx((math.exp(1.5), math.exp(1.5)))


def test_tanh():
    data, g = grad_of(lambda v: v.tanh(), 0.5)
    t = math.tanh(0.5)
    assert data == approx(t)
    assert g == approx(1 - t ** 2), (
        f"tanh's local slope at 0.5 is 1 - tanh(0.5)**2 = {1 - t ** 2:.4f}, got {g}"
    )


def test_relu():
    assert grad_of(lambda v: v.relu(), 2.0) == approx((2.0, 1.0))
    data, g = grad_of(lambda v: v.relu(), -2.0)
    assert data == 0.0
    assert g == 0.0, f"relu passes no gradient for negative inputs, got {g}"


def test_neg_sub_div():
    a, b = Value(5.0), Value(2.0)
    out = (a - b) / b - (-a)
    out.backward()
    # out = (a - b)/b + a  ->  d/da = 1/b + 1 = 1.5 ; d/db = -a/b**2 = -1.25
    assert out.data == approx(6.5)
    assert (a.grad, b.grad) == approx((1.5, -1.25)), f"got a.grad={a.grad}, b.grad={b.grad}"


def test_plain_numbers_on_the_left():
    a = Value(4.0)
    out = 10 - a
    assert out.data == approx(6.0), "10 - a needs __rsub__"
    out = 2 / a
    assert out.data == approx(0.5), "2 / a needs __rtruediv__"
    out.backward()
    assert a.grad == approx(-2 / 16)


def test_composite_ops_reuse_backward():
    # -, / and neg should be built from +, * and **: their results' _op says so.
    a, b = Value(1.0), Value(2.0)
    assert (a - b)._op == "+", "define x - y as x + (-y) rather than a new primitive"
    assert (a / b)._op == "*", "define x / y as x * y**-1 rather than a new primitive"


def test_a_neuron_shaped_expression():
    x1, x2, w1, w2, b = Value(2.0), Value(0.0), Value(-3.0), Value(1.0), Value(6.8814)
    o = (x1 * w1 + x2 * w2 + b).tanh()
    o.backward()
    assert o.data == approx(0.7071, abs=1e-4)
    assert (w1.grad, x1.grad) == approx((1.0, -1.5), abs=1e-3)
