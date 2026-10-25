import numpy as np
import torch
import torch.nn.functional as F

from m03.torch_gpt import TorchGPT
from m05.induction import (head_scores, induction_score, per_position_loss, previous_token_score,
                           repeated_tokens, train_on_repeats)


def test_repeated_tokens():
    x = repeated_tokens(5, 7, 20, np.random.default_rng(0))
    assert x.shape == (5, 14)
    assert np.array_equal(x[:, :7], x[:, 7:])
    assert x.min() >= 0 and x.max() < 20 and len(np.unique(x)) > 7


def test_scores_on_handmade_patterns():
    T, half = 8, 4
    p = torch.zeros(1, 2, T, T)
    p[0, 0, 0, 0] = 1
    for t in range(1, T):
        p[0, 0, t, t - 1] = 1          # head 0: previous token
    for t in range(T):
        src = t - half + 1 if t >= half else 0
        p[0, 1, t, src] = 1            # head 1: induction
    prev = previous_token_score(p)
    ind = induction_score(p, half)
    assert prev.shape == (2,) and prev[0] == 1.0 and prev[1] < 0.2
    assert ind[1] == 1.0 and ind[0] == 0.0, "position t looks at t - half + 1, the token after its earlier copy"


@torch.no_grad()
def test_per_position_loss():
    torch.manual_seed(0)
    m = TorchGPT(9, 8, n_embd=16, n_head=2, n_layer=1)
    idx = np.random.default_rng(0).integers(0, 9, size=(3, 7))
    loss = per_position_loss(m, idx)
    assert loss.shape == (6,)
    t = torch.as_tensor(idx)
    ref = F.cross_entropy(m(t[:, :-1]).reshape(-1, 9), t[:, 1:].reshape(-1), reduction="none").reshape(3, 6).mean(0)
    assert np.allclose(loss, ref.numpy(), atol=1e-5)


def test_induction_heads_form():
    torch.manual_seed(0)
    m = TorchGPT(32, 24, n_embd=32, n_head=2, n_layer=2)
    losses = train_on_repeats(m, 1000, vocab=32, max_half=12)
    assert len(losses) == 1000
    data = repeated_tokens(64, 12, 32, np.random.default_rng(1))
    loss = per_position_loss(m, data)
    assert loss[:11].mean() > 2.5, "the first half is random: nothing to predict"
    assert loss[12:].mean() < 1.0, "the second half is a copy: an induction circuit predicts it"
    prev, ind = head_scores(m, data[:16], 12)
    assert prev.shape == ind.shape == (2, 2)
    assert ind[1].max() > 0.5, "a layer 1 head should attend to the token after the earlier copy"
    assert prev[0].max() > 0.25, "helped by a layer 0 head that looks at the previous token"
