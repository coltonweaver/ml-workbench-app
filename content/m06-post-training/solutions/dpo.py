"""Direct preference optimization: learn from pairs of better and worse
responses, without a reward model."""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from m06.sft import encode_example


def sequence_logprob(model, x, y, mask):
    """Sum of log p(y_t | x_≤t) over masked positions, one value per row:
    the log-probability of each response given its prompt."""
    logp = model(x).log_softmax(-1).gather(-1, y[..., None])[..., 0]
    return (logp * mask).sum(-1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """−log σ(β · ((π_c − ref_c) − (π_r − ref_r))), averaged, from summed
    log-probabilities. Returns (loss, chosen rewards, rejected rewards); a
    response's implicit reward is β · (π − ref)."""
    chosen = beta * (policy_chosen - ref_chosen)
    rejected = beta * (policy_rejected - ref_rejected)
    return -F.logsigmoid(chosen - rejected).mean(), chosen.detach(), rejected.detach()


def batch_rows(tok, prompts, responses, block_size):
    rows = [encode_example(tok, p, r, block_size) for p, r in zip(prompts, responses)]
    return tuple(torch.stack([r[i] for r in rows]) for i in range(3))


def dpo_train(policy, tok, triples, steps, beta=0.1, batch_size=32, lr=5e-4, seed=0, log=None):
    """DPO on (prompt, chosen, rejected) ``triples``. The reference is a
    frozen copy of the policy as it starts. Returns a list of dicts with the
    loss and the reward margin each step."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    T = policy.block_size
    history = []
    for step in range(steps):
        b = rng.integers(0, len(triples), size=batch_size)
        prompts = [triples[i][0] for i in b]
        xc, yc, mc = batch_rows(tok, prompts, [triples[i][1] for i in b], T)
        xr, yr, mr = batch_rows(tok, prompts, [triples[i][2] for i in b], T)
        with torch.no_grad():
            ref_c = sequence_logprob(reference, xc, yc, mc)
            ref_r = sequence_logprob(reference, xr, yr, mr)
        loss, rc, rr = dpo_loss(sequence_logprob(policy, xc, yc, mc), sequence_logprob(policy, xr, yr, mr),
                                ref_c, ref_r, beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append({"loss": loss.item(), "margin": (rc - rr).mean().item()})
        if log is not None:
            log(step, history[-1])
    return history


@torch.no_grad()
def sample(model, tok, prompt, n, max_new=12, temperature=1.0, seed=0):
    """``n`` sampled responses to ``prompt`` (cut at END)."""
    from m06.sft import END, SEP
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    idx = torch.tensor([tok.encode(prompt + SEP)] * n)
    done = torch.zeros(n, dtype=torch.bool)
    outs = [[] for _ in range(n)]
    end = tok.stoi[END]
    for _ in range(max_new):
        probs = (model(idx[:, -model.block_size:])[:, -1] / temperature).softmax(-1)
        nxt = torch.multinomial(probs, 1, generator=gen)
        for i in range(n):
            if not done[i]:
                if int(nxt[i]) == end:
                    done[i] = True
                else:
                    outs[i].append(int(nxt[i]))
        idx = torch.cat([idx, nxt], 1)
        if done.all():
            break
    return [tok.decode(o) for o in outs]
