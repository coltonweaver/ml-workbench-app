import numpy as np

from m02.tensor import Tensor
from m03.gpt import GPT

rng = np.random.default_rng(5)


def test_shapes_and_parameter_count():
    m = GPT(65, 32, n_embd=64, n_head=4, n_layer=2)
    assert sum(p.data.size for p in m.parameters()) == 109952
    logits = m(rng.integers(0, 65, size=(3, 10)))
    assert isinstance(logits, Tensor) and logits.shape == (3, 10, 65)


def test_initial_loss_is_uniform():
    m = GPT(30, 8, n_embd=16, n_head=2, n_layer=2)
    idx = rng.integers(0, 30, size=(4, 8))
    loss = m.loss(idx, rng.integers(0, 30, size=(4, 8)))
    assert loss.shape == ()
    assert abs(float(loss.data) - np.log(30)) < 0.1, "small init should give a loss near ln(vocab)"


def test_positions_matter():
    m = GPT(10, 6, n_embd=8, n_head=2, n_layer=1)
    m.wpe.data = rng.normal(size=m.wpe.shape)
    logits = m(np.array([[3, 3, 3, 3]])).data[0]
    assert not np.allclose(logits[0], logits[3]), (
        "the same token at different positions should get different logits: add position embeddings")


def test_causal():
    m = GPT(10, 6, n_embd=8, n_head=2, n_layer=2, seed=1)
    for p in m.parameters():
        p.data = p.data + rng.normal(size=p.shape) * 0.3
    a = m(np.array([[1, 2, 3, 4, 5]])).data
    b = m(np.array([[1, 2, 3, 9, 9]])).data
    assert np.allclose(a[0, :3], b[0, :3]), "logits at a position can't depend on later tokens"


def test_loss_matches_cross_entropy_over_all_positions():
    m = GPT(7, 5, n_embd=8, n_head=2, n_layer=1)
    for p in m.parameters():
        p.data = rng.normal(size=p.shape) * 0.5
    idx, tgt = rng.integers(0, 7, size=(2, 5)), rng.integers(0, 7, size=(2, 5))
    z = m(idx).data
    z = z - z.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    expected = -np.take_along_axis(logp, tgt[..., None], -1).mean()
    assert np.isclose(float(m.loss(idx, tgt).data), expected)


def test_gradient_reaches_embeddings():
    m = GPT(5, 4, n_embd=4, n_head=2, n_layer=1)
    for p in m.parameters():
        p.data = rng.normal(size=p.shape) * 0.5
    idx, tgt = np.array([[0, 1, 2]]), np.array([[1, 2, 3]])
    m.loss(idx, tgt).backward()
    i = (1, 2)
    g = m.wte.grad[i]
    m.wte.data[i] += 1e-6
    hi = float(m.loss(idx, tgt).data)
    m.wte.data[i] -= 2e-6
    lo = float(m.loss(idx, tgt).data)
    assert np.isclose(g, (hi - lo) / 2e-6, atol=1e-6)


def test_generate():
    m = GPT(10, 4, n_embd=8, n_head=2, n_layer=1)
    out = m.generate([1, 2, 3, 4, 5, 6], 5, 0, np.random.default_rng(0))
    assert out[:6] == [1, 2, 3, 4, 5, 6] and len(out) == 11, "returns the prompt plus max_new tokens"
    assert all(isinstance(t, int) and 0 <= t < 10 for t in out)
    again = m.generate([1, 2, 3, 4, 5, 6], 5, 0, np.random.default_rng(1))
    assert out == again, "temperature 0 is greedy, so the seed doesn't matter"
    # Greedy from the prompt should match the model's own argmax.
    last = m(np.array([[3, 4, 5, 6]])).data[0, -1]
    assert out[6] == int(np.argmax(last)), "the next token comes from the last position's logits"


def test_attention_weights_for_the_viewer():
    m = GPT(10, 8, n_embd=8, n_head=2, n_layer=3)
    w = m.attention_weights([1, 2, 3])
    assert w.shape == (3, 2, 3, 3)


def test_forward_composes_the_parts():
    m = GPT(9, 6, n_embd=8, n_head=2, n_layer=2, seed=2)
    for p in m.parameters():
        p.data = rng.normal(size=p.shape) * 0.5
    idx = rng.integers(0, 9, size=(2, 5))
    x = Tensor(m.wte.data[idx] + m.wpe.data[:5])
    for block in m.blocks:
        x = block(x)
    expected = m.ln_f(x).data @ m.lm_head.data
    assert np.allclose(m(idx).data, expected), (
        "embeddings + positions, every block in order, ln_f, then lm_head")
