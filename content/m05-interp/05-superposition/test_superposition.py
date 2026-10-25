import numpy as np
import torch

from m05.toy import ToyModel, feature_norms, interference, represented, sample_features, train_toy

IMPORTANCE = 0.8 ** np.arange(5)


def test_forward_is_the_formula():
    m = ToyModel(4, 2, seed=1)
    with torch.no_grad():
        m.b.copy_(torch.tensor([0.1, -0.2, 0.0, 0.3]))
    x = torch.rand(6, 4)
    expected = torch.relu(x @ m.W.T @ m.W + m.b)
    assert torch.allclose(m(x), expected)
    assert m.W.shape == (2, 4)


def test_sample_features():
    x = sample_features(20000, 4, 0.75, np.random.default_rng(0))
    assert x.shape == (20000, 4) and x.dtype == np.float32
    assert abs((x == 0).mean() - 0.75) < 0.01, "sparsity is the fraction of zeros"
    assert x.max() <= 1 and x.min() >= 0
    assert abs(x[x > 0].mean() - 0.5) < 0.01


def test_norms_and_interference():
    m = ToyModel(3, 2)
    with torch.no_grad():
        m.W.copy_(torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]]))
    assert np.allclose(feature_norms(m), [1.0, 2.0, 1.0])
    assert represented(m) == 3 and represented(m, threshold=1.5) == 1
    assert np.allclose(interference(m), [1.0, 0.0, 1.0]), "features 0 and 2 share a direction"


def test_dense_features_get_one_dimension_each():
    m = ToyModel(5, 2)
    train_toy(m, sparsity=0.0, importance=IMPORTANCE, steps=1500)
    assert represented(m) == 2, "with dense features, 2 dimensions hold 2 features"
    assert (feature_norms(m)[:2] > 0.5).all(), "the two most important ones"


def test_sparse_features_go_into_superposition():
    m = ToyModel(5, 2)
    losses = train_toy(m, sparsity=0.9, importance=IMPORTANCE, steps=1500)
    assert len(losses) == 1500 and losses[-1] < losses[0]
    assert represented(m) >= 4, "sparse features can share dimensions: more features than dimensions"
