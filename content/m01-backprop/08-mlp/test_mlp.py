import random

from pytest import approx

from m01.engine import Value
from m01.nn import MLP, Layer, Neuron, train

XS = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
YS = [1.0, -1.0, -1.0, 1.0]


def test_neuron_parameters_and_output():
    random.seed(0)
    n = Neuron(3)
    params = n.parameters()
    assert len(params) == 4, f"a 3-input neuron has 3 weights + 1 bias, got {len(params)}"
    assert all(isinstance(p, Value) for p in params), "parameters should be Values"
    assert all(-1 <= p.data <= 1 for p in params[:3]), "weights start uniform in [-1, 1]"
    assert params[-1].data == 0.0, "the bias starts at 0, and comes last"
    out = n([1.0, -2.0, 0.5])
    assert isinstance(out, Value) and -1 < out.data < 1, "output should be a tanh Value"


def test_neuron_math():
    n = Neuron(2)
    w0, w1, b = n.parameters()
    w0.data, w1.data, b.data = 0.5, -1.0, 0.25
    import math

    assert n([2.0, 1.0]).data == approx(math.tanh(0.5 * 2 - 1.0 + 0.25)), (
        "a neuron computes tanh(w . x + b)"
    )


def test_layer_shapes():
    layer = Layer(3, 4)
    outs = layer([1.0, 2.0, 3.0])
    assert isinstance(outs, list) and len(outs) == 4, "Layer(3, 4) should return 4 Values"
    assert len(layer.parameters()) == 16
    single = Layer(3, 1)([1.0, 2.0, 3.0])
    assert isinstance(single, Value), "a 1-neuron layer returns the Value itself, not a list"


def test_mlp_parameter_count():
    model = MLP(3, [4, 4, 1])
    assert len(model.parameters()) == 41, (
        f"MLP(3, [4, 4, 1]) has 4*(3+1) + 4*(4+1) + 1*(4+1) = 41 parameters, got "
        f"{len(model.parameters())}"
    )
    assert isinstance(model([1.0, 2.0, 3.0]), Value)


def test_training_fits_the_examples():
    random.seed(7)
    model = MLP(3, [4, 4, 1])
    losses = train(model, XS, YS, lr=0.05, steps=150)
    assert len(losses) == 150, "train should return one loss per step"
    assert all(isinstance(l, float) for l in losses), "losses should be plain floats (loss.data)"
    assert losses[-1] < 0.05, (
        f"loss went from {losses[0]:.3f} to {losses[-1]:.3f}; it should fall below 0.05 in "
        "150 steps. Check that you zero grads every step and move against the gradient"
    )
    assert losses[-1] < losses[0] / 10
    preds = [model(x).data for x in XS]
    assert all((p > 0) == (y > 0) for p, y in zip(preds, YS)), f"predictions {preds} vs targets {YS}"


def test_training_zeroes_grads_each_step():
    random.seed(3)
    model = MLP(3, [4, 1])
    train(model, XS, YS, lr=0.0, steps=1)
    first = [p.grad for p in model.parameters()]
    train(model, XS, YS, lr=0.0, steps=1)
    second = [p.grad for p in model.parameters()]
    assert second == approx(first), (
        "with lr=0 the parameters don't change, so each step's grads should be identical. "
        "If they doubled, the grads weren't reset before backward()"
    )
