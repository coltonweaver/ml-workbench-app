"""Paged KV memory: fixed-size blocks and a block table per sequence, like
virtual memory pages (Kwon et al., "PagedAttention", 2023)."""

import math

import torch


class BlockAllocator:
    """``n_blocks`` blocks of ``block_size`` token slots each."""

    def __init__(self, n_blocks, block_size):
        self.block_size = block_size
        self.free = list(range(n_blocks))
        self.tables = {}     # sequence id -> list of block ids
        self.lengths = {}    # sequence id -> tokens stored

    @property
    def n_free(self):
        return len(self.free)

    def blocks_needed(self, tokens):
        return math.ceil(tokens / self.block_size)

    def can_allocate(self, tokens):
        return self.blocks_needed(tokens) <= self.n_free

    def allocate(self, seq, tokens):
        """Reserve blocks for a new sequence's first ``tokens`` tokens.
        Raises MemoryError (taking nothing) if there aren't enough."""
        if seq in self.tables:
            raise ValueError(f"sequence {seq} already has blocks")
        need = self.blocks_needed(tokens)
        if need > self.n_free:
            raise MemoryError(f"need {need} blocks, {self.n_free} free")
        self.tables[seq] = [self.free.pop(0) for _ in range(need)]
        self.lengths[seq] = tokens
        return list(self.tables[seq])

    def append(self, seq):
        """Make room for one more token, taking a new block only when the
        last one is full. Raises MemoryError if none is free."""
        if self.lengths[seq] == len(self.tables[seq]) * self.block_size:
            if not self.free:
                raise MemoryError("no free blocks")
            self.tables[seq].append(self.free.pop(0))
        self.lengths[seq] += 1

    def release(self, seq):
        """Return a sequence's blocks to the free list."""
        self.free.extend(self.tables.pop(seq))
        del self.lengths[seq]

    def slot(self, seq, position):
        """(block id, offset) where token ``position`` of ``seq`` lives."""
        return self.tables[seq][position // self.block_size], position % self.block_size

    def wasted_slots(self):
        """Reserved but unused token slots (only ever in a last block)."""
        return sum(len(t) * self.block_size - self.lengths[s] for s, t in self.tables.items())


def contiguous_waste(lengths, max_len):
    """Slots wasted when every sequence reserves ``max_len`` contiguous slots
    up front, as a server must if it doesn't know output lengths."""
    return sum(max_len - n for n in lengths)


def gather(pool, allocator, seq):
    """A sequence's stored vectors in order, from a pool of shape
    (n_blocks, block_size, d): (length, d)."""
    table = torch.as_tensor(allocator.tables[seq])
    return pool[table].reshape(-1, pool.shape[-1])[:allocator.lengths[seq]]
