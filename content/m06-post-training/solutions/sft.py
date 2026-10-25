"""Supervised fine-tuning: teach a pretrained model a prompt/response format."""

import numpy as np
import torch
import torch.nn.functional as F

SEP = ">"   # ends the prompt
END = "."   # ends the response
PAD = "_"   # fills a batch row; never trained on


class CharTokenizer:
    def __init__(self, chars):
        self.itos = sorted(set(chars) | {SEP, END, PAD})
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def encode_example(tok, prompt, response, block_size):
    """One training row for ``prompt + SEP + response + END``, padded with
    PAD to ``block_size + 1`` tokens. Returns (x, y, mask), each of length
    block_size: x the inputs, y the next tokens, mask 1 where y is part of
    the response (including END) and 0 for the prompt, SEP and padding."""
    ids = tok.encode(prompt + SEP + response + END)
    if len(ids) > block_size + 1:
        raise ValueError(f"example is {len(ids)} tokens; block_size {block_size} fits {block_size + 1}")
    n_prompt = len(prompt) + 1
    mask = [0] * (n_prompt - 1) + [1] * (len(ids) - n_prompt)
    ids = ids + [tok.stoi[PAD]] * (block_size + 1 - len(ids))
    mask = mask + [0] * (block_size - len(mask))
    return (torch.tensor(ids[:-1]), torch.tensor(ids[1:]), torch.tensor(mask, dtype=torch.float32))


def masked_cross_entropy(logits, targets, mask):
    """Mean cross-entropy over the positions where mask is 1."""
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    mask = mask.reshape(-1)
    return (nll * mask).sum() / mask.sum().clamp_min(1)


def sft(model, tok, pairs, steps, batch_size=32, lr=1e-3, seed=0, log=None):
    """AdamW on random batches of (prompt, response) ``pairs`` with the
    masked loss. Returns the losses."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = [encode_example(tok, p, r, model.block_size) for p, r in pairs]
    X = torch.stack([r[0] for r in rows])
    Y = torch.stack([r[1] for r in rows])
    M = torch.stack([r[2] for r in rows])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for step in range(steps):
        b = torch.as_tensor(rng.integers(0, len(rows), size=batch_size))
        loss = masked_cross_entropy(model(X[b]), Y[b], M[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log is not None:
            log(step, losses[-1])
    return losses


@torch.no_grad()
def complete(model, tok, prompt, max_new=20):
    """Greedy response to ``prompt``: everything generated after SEP, up to
    (not including) END."""
    model.eval()
    idx = torch.tensor([tok.encode(prompt + SEP)])
    out = []
    for _ in range(max_new):
        nxt = int(model(idx[:, -model.block_size:])[0, -1].argmax())
        if tok.itos[nxt] == END:
            break
        out.append(nxt)
        idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
    return tok.decode(out)


def exact_match(model, tok, pairs):
    """Fraction of (prompt, response) pairs answered exactly."""
    return float(np.mean([complete(model, tok, p) == r for p, r in pairs]))
