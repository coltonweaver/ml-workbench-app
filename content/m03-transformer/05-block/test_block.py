import numpy as np

from m02.tensor import Tensor
from m03.gpt import Block

rng = np.random.default_rng(4)


def randomize(block):
    for p in block.parameters():
        p.data = rng.normal(size=p.shape) * 0.3


def test_block_composes_its_parts():
    block = Block(8, 2, seed=1)
    randomize(block)
    x = Tensor(rng.normal(size=(2, 5, 8)))
    h = x.data + block.attn(block.ln1(x)).data
    expected = h + block.mlp(block.ln2(Tensor(h))).data
    out = block(x)
    assert out.shape == (2, 5, 8)
    assert np.allclose(out.data, expected), (
        "pre-norm: x = x + attn(ln1(x)); x = x + mlp(ln2(x))")


def test_block_starts_as_identity_when_outputs_are_zero():
    block = Block(8, 2, seed=2)
    randomize(block)
    block.attn.Wo.data[:] = 0
    block.mlp.W2.data[:] = 0
    block.mlp.b2.data[:] = 0
    x = rng.normal(size=(1, 4, 8))
    assert np.allclose(block(Tensor(x)).data, x), (
        "with the residual stream, a block whose sublayers output 0 passes x through unchanged")


def test_gradient_has_an_identity_path():
    block = Block(4, 2, seed=3)
    block.attn.Wo.data[:] = 0
    block.mlp.W2.data[:] = 0
    x = Tensor(rng.normal(size=(1, 3, 4)))
    block(x).sum().backward()
    assert np.allclose(x.grad, 1.0), "the residual connection carries the gradient straight through"
