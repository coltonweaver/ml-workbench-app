"""Direct preference optimization: learn from pairs of better and worse
responses, without a reward model.

Lessons 6.3 and 6.4."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from m06.sft import encode_example


def sequence_logprob(model, x, y, mask):
    """Sum of log p(y_t | x_≤t) over masked positions, one value per row:
    the log-probability of each response given its prompt."""
    raise NotImplementedError('lesson 6.3')


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """−log σ(β · ((π_c − ref_c) − (π_r − ref_r))), averaged, from summed
    log-probabilities. Returns (loss, chosen rewards, rejected rewards); a
    response's implicit reward is β · (π − ref)."""
    raise NotImplementedError('lesson 6.3')


def batch_rows(tok, prompts, responses, block_size):
    raise NotImplementedError('lesson 6.4')


def dpo_train(policy, tok, triples, steps, beta=0.1, batch_size=32, lr=0.0005, seed=0, log=None):
    """DPO on (prompt, chosen, rejected) ``triples``. The reference is a
    frozen copy of the policy as it starts. Returns a list of dicts with the
    loss and the reward margin each step."""
    raise NotImplementedError('lesson 6.4')


@torch.no_grad()
def sample(model, tok, prompt, n, max_new=12, temperature=1.0, seed=0):
    """``n`` sampled responses to ``prompt`` (cut at END)."""
    raise NotImplementedError('lesson 6.4')
