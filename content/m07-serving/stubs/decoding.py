"""The cost of generating text, and why decoding is limited by memory.

Lesson 7.1."""
import torch


@torch.no_grad()
def generate_naive(model, idx, n):
    """Greedy decoding that reruns the whole context for every new token."""
    raise NotImplementedError('lesson 7.1')


def forward_flops(n_params, n_layer, n_embd, context):
    """FLOPs to process one token at position ``context``: 2 per parameter
    (a multiply and an add) plus attention, which compares the token's query
    with ``context`` keys and mixes as many values: 2 · 2 · n_layer ·
    context · n_embd."""
    raise NotImplementedError('lesson 7.1')


def naive_generation_flops(n_params, n_layer, n_embd, prompt, new):
    """Total FLOPs of generate_naive: every step reprocesses every token so
    far (each at its own position)."""
    raise NotImplementedError('lesson 7.1')


def cached_generation_flops(n_params, n_layer, n_embd, prompt, new):
    """Total FLOPs with a KV cache: the prompt once, then one token per step."""
    raise NotImplementedError('lesson 7.1')


def arithmetic_intensity(batch, bytes_per_param):
    """FLOPs per byte of weights read in one decoding step: each weight is
    read once and used for 2 FLOPs per sequence in the batch."""
    raise NotImplementedError('lesson 7.1')


def step_time(n_params, batch, bytes_per_param, flops_per_s, bytes_per_s):
    """Roofline estimate of one decoding step: the larger of compute time and
    the time to stream the weights from memory."""
    raise NotImplementedError('lesson 7.1')


def critical_batch(bytes_per_param, flops_per_s, bytes_per_s):
    """The batch size where compute time equals memory time; below it,
    decoding is memory-bound and extra sequences are almost free."""
    raise NotImplementedError('lesson 7.1')
