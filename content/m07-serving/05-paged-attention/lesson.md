---
id: "7.5"
title: Paged KV memory
concepts: [paged-attention, memory-fragmentation, block-tables]
file: m07/paged.py
stub: stubs/paged.py
tests: test_paged.py
predict:
  kind: choice
  prompt: >
    A server doesn't know how long each response will be, so a simple
    design reserves room for the maximum length (say 2048 tokens) for every
    sequence. With paged memory in 16-token blocks, at most how many slots
    are wasted per sequence?
  options:
    - none
    - at most 15 (part of the last block)
    - about half of 2048 on average, same as before
    - 16 per block used
  answer: 1
mutants:
  - name: a new block for every token
    file: m07/paged.py
    replace: ["if self.lengths[seq] == len(self.tables[seq]) * self.block_size:", "if True:"]
  - name: no check before taking blocks
    file: m07/paged.py
    replace: ["        if need > self.n_free:\n            raise MemoryError(f\"need {need} blocks, {self.n_free} free\")\n", ""]
  - name: release forgets the blocks
    file: m07/paged.py
    replace: ["self.free.extend(self.tables.pop(seq))", "self.tables.pop(seq)"]
warmups:
  - kind: number
    prompt: "KV blocks hold 16 tokens. A sequence has 50 tokens. How many token slots are allocated but empty?"
    answer: 14
    tolerance: 0.01
    explain: "50 tokens need 4 blocks = 64 slots, so 14 are unused. Paging bounds the waste to under one block per sequence."
---

## Hook

KV memory caps the batch (7.2), and the batch caps throughput (7.1). Kwon
et al., "Efficient Memory Management for Large Language Model Serving with
PagedAttention" (2023), found that existing servers wasted 60–80% of KV
memory: each sequence reserved a contiguous region for its maximum
length. Their fix is the one operating systems found decades ago: pages.

## Build

In `m07/paged.py`:

1. `BlockAllocator(n_blocks, block_size)` with a free list and a block
   table per sequence: `allocate(seq, tokens)`, `append(seq)` (a new block
   only when the last is full), `release(seq)`, `can_allocate`, `slot(seq,
   position)` and `wasted_slots()`. Running out raises `MemoryError` and
   takes nothing.
2. `contiguous_waste(lengths, max_len)` for comparison.
3. `gather(pool, allocator, seq)`: a sequence's vectors in order from a
   pool shaped (n_blocks, block_size, d), following its block table, which
   is what a paged attention kernel does.

```python
lengths = np.random.default_rng(0).integers(20, 2048, size=64)
a = BlockAllocator(10_000, 16)
for i, n in enumerate(lengths):
    a.allocate(i, int(n))
a.wasted_slots(), contiguous_waste(lengths, 2048)
```

## Explain

A sequence only ever has one partly filled block, its last, so the waste
is **at most 15 slots** per sequence, against ~1000 on average for
contiguous reservation. vLLM fits several times more sequences into the
same memory, which is where most of its throughput gain comes from.

Block tables also make sharing cheap: sequences with a common prefix (a
system prompt, several samples of one prompt) can point at the same
physical blocks, copied only when one of them writes (copy-on-write).
Prefix caching across requests builds on exactly this.

## Stretch

Add reference counts and `fork(seq, new_seq)`, which shares all of seq's
blocks, and make `append` copy a shared last block before writing to it.
How much memory do 8 samples from one 1000-token prompt take?
