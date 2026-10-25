import math

import torch
import torch.nn.functional as F

from m03.torch_gpt import TorchGPT
from m06.dpo import dpo_loss, sequence_logprob
from m06.sft import CharTokenizer, encode_example


def test_dpo_loss_by_hand():
    pc, pr = torch.tensor([-3.0, -5.0]), torch.tensor([-4.0, -2.0])
    rc, rr = torch.tensor([-3.5, -5.0]), torch.tensor([-3.5, -3.0])
    loss, chosen, rejected = dpo_loss(pc, pr, rc, rr, beta=0.5)
    # row 0: 0.5 * ((0.5) - (-0.5)) = 0.5; row 1: 0.5 * (0 - 1) = -0.5
    expected = (-math.log(1 / (1 + math.exp(-0.5))) - math.log(1 / (1 + math.exp(0.5)))) / 2
    assert math.isclose(loss.item(), expected, rel_tol=1e-6)
    assert torch.allclose(chosen, torch.tensor([0.25, 0.0])) and torch.allclose(rejected, torch.tensor([-0.25, 0.5]))


def test_policy_equal_to_reference_gives_log_2():
    lp = torch.randn(8)
    loss, _, _ = dpo_loss(lp, lp - 1, lp, lp - 1, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6), "no preference learned yet: σ(0) = ½"


def test_gradient_pushes_chosen_up_and_rejected_down():
    pc = torch.tensor([-2.0], requires_grad=True)
    pr = torch.tensor([-2.0], requires_grad=True)
    loss, _, _ = dpo_loss(pc, pr, torch.tensor([-2.0]), torch.tensor([-2.0]), beta=1.0)
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()


@torch.no_grad()
def test_sequence_logprob():
    torch.manual_seed(0)
    tok = CharTokenizer("abc")
    m = TorchGPT(len(tok), 8, n_embd=16, n_head=2, n_layer=1)
    x, y, mask = (t[None] for t in encode_example(tok, "ab", "ca", 8))
    lp = sequence_logprob(m, x, y, mask)
    logp = F.log_softmax(m(x), -1)[0]
    # response tokens are c, a, END at target positions 2, 3, 4
    expected = sum(logp[t, y[0, t]] for t in (2, 3, 4))
    assert lp.shape == (1,) and torch.isclose(lp[0], expected, atol=1e-5)
