import numpy as np

from m02.tensor import Tensor
from m03.gpt import GPT
from m03.train import AdamW, build_char_vocab, get_batch, train_gpt


def test_char_vocab():
    stoi, itos = build_char_vocab("hello, world")
    assert itos == sorted(set("hello, world"))
    assert all(stoi[c] == i for i, c in enumerate(itos))


def test_get_batch():
    data = np.arange(100)
    X, Y = get_batch(data, 8, 5, np.random.default_rng(0))
    assert X.shape == (5, 8) and Y.shape == (5, 8)
    assert np.all(Y == X + 1), "the target at every position is the next token"
    assert np.all(np.diff(X, axis=1) == 1), "each row is a contiguous window"
    assert X.max() < 100 and Y.max() < 100
    starts = {int(get_batch(data, 8, 1, np.random.default_rng(s))[0][0, 0]) for s in range(200)}
    assert max(starts) >= 85, "windows should start anywhere they fit"


def test_adamw_matches_the_formula():
    p = Tensor(np.array([[1.0, -2.0]]))
    opt = AdamW([p], lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
    m = v = 0
    expected = p.data.copy()
    for t, g in enumerate([np.array([[0.5, -1.0]]), np.array([[0.1, 0.3]])], start=1):
        p.grad = g.copy()
        opt.step()
        m = 0.9 * m + 0.1 * g
        v = 0.99 * v + 0.01 * g * g
        expected = expected - 0.1 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.99 ** t)) + 1e-8)
        assert np.allclose(p.data, expected), f"step {t} doesn't match Adam's update"


def test_first_adam_step_is_lr_times_sign():
    p = Tensor(np.array([[3.0, 3.0]]))
    opt = AdamW([p], lr=0.01)
    p.grad = np.array([[1000.0, -0.001]])
    opt.step()
    assert np.allclose(p.data, [[2.99, 3.01]], atol=1e-6), (
        "with bias correction the first step is lr in the gradient's direction, whatever its size")


def test_weight_decay_only_on_matrices():
    W, b = Tensor(np.ones((2, 2))), Tensor(np.ones(2))
    opt = AdamW([W, b], lr=0.1, weight_decay=0.5)
    W.grad = np.zeros((2, 2))
    b.grad = np.zeros(2)
    opt.step()
    assert np.allclose(W.data, 0.95), "decoupled decay: p -= lr * weight_decay * p"
    assert np.allclose(b.data, 1.0), "gains and biases (1D) aren't decayed"


def test_train_gpt_learns_a_pattern():
    text = "abcd" * 50
    stoi, itos = build_char_vocab(text)
    data = np.array([stoi[c] for c in text])
    m = GPT(len(itos), 8, n_embd=16, n_head=2, n_layer=1, seed=0)
    seen = []
    losses = train_gpt(m, data, steps=80, batch_size=8, lr=1e-2, seed=0, log=lambda s, l: seen.append(s))
    assert len(losses) == 80 and seen == list(range(80))
    assert losses[-1] < 0.3, f"'abcd' repeated is easy to predict; final loss {losses[-1]:.2f}"
