---
id: "8.3"
title: Tensor parallelism
concepts: [tensor-parallelism, megatron, column-parallel, row-parallel]
file: m08/tensor_parallel.py
stub: stubs/tensor_parallel.py
tests: test_tensor_parallel.py
predict:
  kind: number
  prompt: >
    Megatron-style tensor parallelism splits the attention and the MLP of
    each transformer block across GPUs so that each needs exactly one
    all-reduce in the forward pass. How many all-reduces per block per
    forward pass?
  answer: 2
  tolerance: 0
mutants:
  - name: bias on every rank
    file: m08/tensor_parallel.py
    replace: ["proj_bias if rank == 0 else None", "proj_bias"]
  - name: proj split by rows
    file: m08/tensor_parallel.py
    replace: ["return fc_weight[s], fc_bias[s], proj_weight[:, s]", "return fc_weight[s], fc_bias[s], proj_weight[s]"]
  - name: no all-reduce
    file: m08/tensor_parallel.py
    replace: ["    return ring_all_reduce(out)", "    return out"]
---

## Hook

When the model itself doesn't fit on one device, split the weight
matrices. Shoeybi et al., "Megatron-LM" (2019), found a way to split a
transformer block so that communication happens only once per sublayer,
by pairing the two matrices of each sublayer cleverly.

## Build

In `m08/tensor_parallel.py`:

1. `split_mlp(fc_w, fc_b, proj_w, proj_b, rank, world)`: fc split by rows
   (each rank computes its own slice of the 4C hidden units), proj split
   by columns (each rank turns its slice into a partial C-dim output).
   Only rank 0 keeps proj's bias.
2. `tp_mlp_forward(x, …)`: local fc, GELU, local proj, then one
   all-reduce to sum the partial outputs.
3. `tp_bytes_per_layer(tokens, n_embd)`.

```python
from workbench.dist import launch
[(out - full).abs().max() for out, full in launch(tp_worker, 4)]
```

## Explain

GELU is elementwise, so if each rank has a whole slice of the hidden
units, it can apply GELU with no communication. Splitting the second
matrix along the matching dimension makes each rank's output a partial
sum, and one all-reduce finishes the job. Attention splits the same way:
heads are independent, so each rank takes some heads (column-parallel
Q, K, V) and the output projection is row-parallel. **Two all-reduces**
per block forward (and two in backward).

Each all-reduce moves the full activation (tokens × C) every layer, so
tensor parallelism needs very fast links and is normally kept inside one
node (8 GPUs on NVLink). Across nodes, pipeline and data parallelism
take over. Sequence parallelism (Korthikanti et al., 2022) additionally
splits the layer norms and dropout along the sequence to save activation
memory.

## Stretch

Write the backward pass by hand: what communication does the
column-parallel fc need for the input gradient? (Megatron's "f" and "g"
operators: identity forward / all-reduce backward, and the reverse.)
