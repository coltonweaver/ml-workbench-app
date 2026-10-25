"""Tensor parallelism: split each weight matrix across ranks (Megatron-LM,
Shoeybi et al., 2019).

Lesson 8.3."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from m08.collectives import ring_all_reduce


def split_mlp(fc_weight, fc_bias, proj_weight, proj_bias, rank, world):
    """This rank's share of a GPT MLP, proj(gelu(fc(x))). fc (4C × C) is
    split by rows (column-parallel: each rank computes a slice of the hidden
    units), proj (C × 4C) by columns to match (row-parallel: each rank
    turns its slice into a partial output). Only rank 0 keeps proj's bias,
    so it's added once. Returns (fc_w, fc_b, proj_w, proj_b or None)."""
    raise NotImplementedError("lesson 8.3")


def tp_mlp_forward(x, fc_w, fc_b, proj_w, proj_b):
    """The MLP on this rank's shard, then one all-reduce to sum the partial
    outputs. GELU is elementwise, so it runs on each slice with no
    communication."""
    raise NotImplementedError("lesson 8.3")


def tp_bytes_per_layer(tokens, n_embd, bytes_per_value=2):
    """Data each MLP all-reduces per forward pass: the (tokens × C) output."""
    raise NotImplementedError("lesson 8.3")


def tp_worker(rank, world, seed=0, tokens=6, n_embd=16):
    """Test helper: a random MLP and input (same on every rank, from the
    seed); returns (tensor-parallel output, full output)."""
    torch.manual_seed(seed)
    fc_w, fc_b = torch.randn(4 * n_embd, n_embd), torch.randn(4 * n_embd)
    proj_w, proj_b = torch.randn(n_embd, 4 * n_embd), torch.randn(n_embd)
    x = torch.randn(tokens, n_embd)
    full = F.gelu(x @ fc_w.T + fc_b, approximate="tanh") @ proj_w.T + proj_b
    return tp_mlp_forward(x, *split_mlp(fc_w, fc_b, proj_w, proj_b, rank, world)), full
