import numpy as np

from m02.bigram import count_matrix, probabilities
from m02.neural_bigram import dataset, train_bigram
from m02.nll import nll
from m02.tensor import Tensor
from m02.tokens import build_vocab

WORDS = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia", "harper",
         "evelyn", "abigail", "emily", "ella", "elizabeth", "camila", "luna"]


def test_dataset():
    stoi, _ = build_vocab(["ab"])
    xs, ys = dataset(["ab", "b"], stoi)
    assert xs.tolist() == [0, 1, 2, 0, 2]
    assert ys.tolist() == [1, 2, 0, 2, 0]
    assert np.issubdtype(xs.dtype, np.integer), "ids should be integers so they can index W"


def test_training_reaches_the_counting_model():
    stoi, itos = build_vocab(WORDS)
    xs, ys = dataset(WORDS, stoi)
    W, losses = train_bigram(xs, ys, len(itos), steps=300, lr=20.0, reg=0.0, seed=0)
    assert isinstance(W, Tensor) and W.shape == (len(itos), len(itos))
    assert len(losses) == 300 and all(isinstance(l, float) for l in losses)
    assert losses[-1] < losses[0] - 1.0, "the loss should fall a long way"
    best = nll(probabilities(count_matrix(WORDS, stoi), smoothing=0.0), WORDS, stoi)
    assert losses[-1] < best + 0.1, (
        f"after training, loss {losses[-1]:.3f} should approach the counting model's {best:.3f}: "
        "both are the same model, one counted and one learned")
    assert losses[-1] > best - 1e-6, "nothing can beat the counts on their own data"


def test_regularization_pulls_towards_uniform():
    stoi, itos = build_vocab(WORDS)
    xs, ys = dataset(WORDS, stoi)
    W0, _ = train_bigram(xs, ys, len(itos), steps=200, lr=20.0, reg=0.0, seed=0)
    W1, _ = train_bigram(xs, ys, len(itos), steps=200, lr=20.0, reg=1.0, seed=0)
    assert (W1.data ** 2).mean() < (W0.data ** 2).mean(), "reg * mean(W²) should shrink W"


def test_log_callback_and_seed():
    stoi, itos = build_vocab(WORDS)
    xs, ys = dataset(WORDS, stoi)
    seen = []
    _, losses = train_bigram(xs, ys, len(itos), steps=5, lr=1.0, seed=3,
                             log=lambda step, loss: seen.append((step, loss)))
    assert seen == list(enumerate(losses))
    _, again = train_bigram(xs, ys, len(itos), steps=5, lr=1.0, seed=3)
    assert again == losses, "the same seed should give the same run"
