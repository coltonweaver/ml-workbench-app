import numpy as np

from m02.losses import cross_entropy
from m02.mlp_lm import MLPLanguageModel, build_dataset, train
from m02.tensor import Tensor
from m02.tokens import build_vocab


def test_dataset_contexts():
    stoi, _ = build_vocab(["ab"])
    X, Y = build_dataset(["ab"], stoi, block_size=3)
    assert X.tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 2]]
    assert Y.tolist() == [1, 2, 0], "the last target is the boundary: the word ends"
    X2, _ = build_dataset(["ab", "b"], stoi, block_size=2)
    assert X2.shape == (5, 2), "one row per character plus the end, for every word"
    assert X2[3].tolist() == [0, 0], "every word starts from a fresh boundary context"


def test_parameter_count():
    m = MLPLanguageModel(27, block_size=3, n_embd=10, n_hidden=200)
    assert all(isinstance(p, Tensor) for p in m.parameters())
    shapes = [p.shape for p in m.parameters()]
    assert shapes == [(27, 10), (30, 200), (200,), (200, 27), (27,)], shapes
    assert sum(p.data.size for p in m.parameters()) == 11897


def test_forward_shape_and_initial_loss():
    m = MLPLanguageModel(27, block_size=3, n_embd=4, n_hidden=16, seed=0)
    X = np.random.default_rng(0).integers(0, 27, size=(50, 3))
    logits = m(X)
    assert logits.shape == (50, 27)
    loss = float(cross_entropy(logits, np.zeros(50, dtype=int)).data)
    assert abs(loss - np.log(27)) < 0.3, (
        f"at init the model should be close to uniform (loss ≈ ln 27 = 3.30), got {loss:.2f}: "
        "start the output layer small")


def test_gradients_flow_to_every_parameter():
    m = MLPLanguageModel(7, block_size=2, n_embd=3, n_hidden=5, seed=1)
    X = np.array([[0, 1], [2, 3], [4, 4]])
    Y = np.array([1, 2, 3])
    loss = cross_entropy(m(X), Y)
    loss.backward()
    for name, p in zip(["C", "W1", "b1", "W2", "b2"], m.parameters()):
        assert np.abs(p.grad).sum() > 0, f"{name} gets no gradient"
    # numeric check on one embedding entry
    i, j, h = 4, 1, 1e-6
    g = m.C.grad[i, j]
    m.C.data[i, j] += h
    up = float(cross_entropy(m(X), Y).data)
    m.C.data[i, j] -= 2 * h
    down = float(cross_entropy(m(X), Y).data)
    m.C.data[i, j] += h
    assert np.isclose(g, (up - down) / (2 * h), atol=1e-6), "the embedding gradient is wrong"


def test_training_lowers_the_loss():
    words = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia", "harper"]
    stoi, itos = build_vocab(words)
    X, Y = build_dataset(words, stoi, 3)
    m = MLPLanguageModel(len(itos), n_embd=8, n_hidden=64, seed=0)
    seen = []
    losses = train(m, X, Y, steps=400, batch_size=32, lr=0.3, seed=0, log=lambda s, l: seen.append(s))
    assert len(losses) == 400 and seen == list(range(400))
    assert losses[0] > np.log(len(itos)) - 0.3, "the first loss should be near ln(vocab): uniform at init"
    assert np.mean(losses[-50:]) < losses[0] - 1.5, (
        f"the loss should fall on 8 names: first {losses[0]:.2f}, last {np.mean(losses[-50:]):.2f}")
    full = float(cross_entropy(m(X), Y).data)
    assert full < 1.0, f"after 400 steps the model should fit these names well (loss {full:.2f})"


def test_forward_matches_the_formula():
    m = MLPLanguageModel(9, block_size=2, n_embd=3, n_hidden=5, seed=2)
    for p in m.parameters():  # make every term matter
        p.data = np.random.default_rng(p.data.size).normal(size=p.shape)
    X = np.array([[0, 1], [8, 8], [3, 0]])
    emb = m.C.data[X].reshape(3, -1)
    expected = np.tanh(emb @ m.W1.data + m.b1.data) @ m.W2.data + m.b2.data
    assert np.allclose(m(X).data, expected), (
        "logits = tanh(concat(C[x]) @ W1 + b1) @ W2 + b2, embeddings in context order")
