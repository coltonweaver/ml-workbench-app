"""Paged KV memory: fixed-size blocks and a block table per sequence, like
virtual memory pages (Kwon et al., "PagedAttention", 2023).

Lesson 7.5."""
import math
import torch


class BlockAllocator:
    """``n_blocks`` blocks of ``block_size`` token slots each."""

    def __init__(self, n_blocks, block_size):
        raise NotImplementedError('lesson 7.5')

    @property
    def n_free(self):
        raise NotImplementedError('lesson 7.5')

    def blocks_needed(self, tokens):
        raise NotImplementedError('lesson 7.5')

    def can_allocate(self, tokens):
        raise NotImplementedError('lesson 7.5')

    def allocate(self, seq, tokens):
        """Reserve blocks for a new sequence's first ``tokens`` tokens.
        Raises MemoryError (taking nothing) if there aren't enough."""
        raise NotImplementedError('lesson 7.5')

    def append(self, seq):
        """Make room for one more token, taking a new block only when the
        last one is full. Raises MemoryError if none is free."""
        raise NotImplementedError('lesson 7.5')

    def release(self, seq):
        """Return a sequence's blocks to the free list."""
        raise NotImplementedError('lesson 7.5')

    def slot(self, seq, position):
        """(block id, offset) where token ``position`` of ``seq`` lives."""
        raise NotImplementedError('lesson 7.5')

    def wasted_slots(self):
        """Reserved but unused token slots (only ever in a last block)."""
        raise NotImplementedError('lesson 7.5')


def contiguous_waste(lengths, max_len):
    """Slots wasted when every sequence reserves ``max_len`` contiguous slots
    up front, as a server must if it doesn't know output lengths."""
    raise NotImplementedError('lesson 7.5')


def gather(pool, allocator, seq):
    """A sequence's stored vectors in order, from a pool of shape
    (n_blocks, block_size, d): (length, d)."""
    raise NotImplementedError('lesson 7.5')
