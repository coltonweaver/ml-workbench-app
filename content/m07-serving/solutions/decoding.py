"""The cost of generating text, and why decoding is limited by memory."""

import torch


@torch.no_grad()
def generate_naive(model, idx, n):
    """Greedy decoding that reruns the whole context for every new token."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    for _ in range(n):
        logits = model(idx[:, -model.block_size:])
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return idx


def forward_flops(n_params, n_layer, n_embd, context):
    """FLOPs to process one token at position ``context``: 2 per parameter
    (a multiply and an add) plus attention, which compares the token's query
    with ``context`` keys and mixes as many values: 2 · 2 · n_layer ·
    context · n_embd."""
    return 2 * n_params + 4 * n_layer * context * n_embd


def naive_generation_flops(n_params, n_layer, n_embd, prompt, new):
    """Total FLOPs of generate_naive: every step reprocesses every token so
    far (each at its own position)."""
    total = 0
    for length in range(prompt, prompt + new):
        total += sum(forward_flops(n_params, n_layer, n_embd, t + 1) for t in range(length))
    return total


def cached_generation_flops(n_params, n_layer, n_embd, prompt, new):
    """Total FLOPs with a KV cache: the prompt once, then one token per step."""
    return sum(forward_flops(n_params, n_layer, n_embd, t + 1) for t in range(prompt + new - 1))


def arithmetic_intensity(batch, bytes_per_param):
    """FLOPs per byte of weights read in one decoding step: each weight is
    read once and used for 2 FLOPs per sequence in the batch."""
    return 2 * batch / bytes_per_param


def step_time(n_params, batch, bytes_per_param, flops_per_s, bytes_per_s):
    """Roofline estimate of one decoding step: the larger of compute time and
    the time to stream the weights from memory."""
    compute = 2 * n_params * batch / flops_per_s
    memory = n_params * bytes_per_param / bytes_per_s
    return max(compute, memory)


def critical_batch(bytes_per_param, flops_per_s, bytes_per_s):
    """The batch size where compute time equals memory time; below it,
    decoding is memory-bound and extra sequences are almost free."""
    return flops_per_s * bytes_per_param / (2 * bytes_per_s)
